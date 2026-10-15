"""Stage D: multi-fidelity BO of the spot, after stage A.

Two fidelities:

- **Low: the model.** The error-free twin at the stage A optics, with the nominal
  beam. Free, so it is sampled densely (`n_model` Sobol points over the knob box)
  and a GP per readout is fitted to it: `f_L(x)`.
- **High: the machine.** `env.step()`, one step each. The machine is modelled as
  `f_H(x) = f_L(x) + delta(x)`, with an independent GP per readout on the
  discrepancy `delta` (Kennedy-O'Hagan with rho = 1). With the first reading
  `delta` is already a constant offset, so the first proposal is the model's own
  answer shifted by what the day's beam does differently.

Knobs (`x`, scaled to [-1, 1]): the five quadrupoles closest to the IP and the two
FEA sextupoles, about their stage A values. Each quadrupole's half-range moves
ln(sigma) at the readout by about `QUAD_REACH` in the model; the sextupoles span
their whole range. The five correctors per plane closest to the IP are not
searched: position is linear in them, so every proposal also sets the two FEH
correctors (after the last BPM, so they move the spot and nothing else) to cancel
the predicted centroid error. Their known, linear effect is taken out of the
measured centroid before it reaches the GP.

The objective is the benchmark mismatch, computed from posterior samples of the
four readouts and the transmission (a composite objective), and the acquisition is
expected improvement on it, maximised over a dense candidate set. The loop stops
when the expected improvement is below what one more step costs in the score, and
it ends on the best reading taken.
"""

from __future__ import annotations

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc

from .batched import probe_batch
from .harness import Session
from .responses import QUADRUPOLES, SEXTUPOLES
from .steering import FINAL_CORRECTORS, _clip, _corrector_zero

KNOBS = QUADRUPOLES + SEXTUPOLES
QUAD_REACH = 0.5  # ln(sigma) a quadrupole's half-range moves, at most
TRANSMISSION_WEIGHT = 2.0
SAMPLES = 128
CANDIDATES = 4096
# GPs in double precision, set per tensor: the tracking in the same process is
# float32, so the global default dtype stays as it is.
DTYPE = torch.float64


def step_cost(steps: int) -> float:
    """What one more step adds to 0.5 log2(1 + steps / 50)."""
    return 0.5 * (np.log2(1 + (steps + 1) / 50) - np.log2(1 + steps / 50))


def _fit(x: np.ndarray, y: np.ndarray, noise: float | None = None) -> SingleTaskGP:
    X = torch.as_tensor(x, dtype=DTYPE)
    Y = torch.as_tensor(y, dtype=DTYPE).reshape(-1, 1)
    kwargs = {}
    if noise is not None:
        kwargs["train_Yvar"] = torch.full_like(Y, noise**2)
    gp = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1), **kwargs)
    if len(x) > 1:
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return gp


class Discrepancy:
    """delta(x) for one readout: a constant offset from the first readings, and a
    GP on what is left once there are enough of them."""

    def __init__(self, x: np.ndarray, d: np.ndarray, noise: float):
        self.offset = float(d.mean())
        self.noise = noise
        self.gp = _fit(x, d - self.offset, noise) if len(d) >= 3 else None

    def __call__(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gp is None:
            n = X.shape[0]
            return (torch.full((n,), self.offset, dtype=DTYPE),
                    torch.full((n,), self.noise**2, dtype=DTYPE))
        p = self.gp.posterior(X)
        return p.mean.squeeze(-1) + self.offset, p.variance.squeeze(-1)


class SpotBO:
    """One episode's stage D, started after stage A on `session`."""

    def __init__(self, session: Session, k1: dict[str, float], model: dict,
                 n_model: int = 256, seed: int = 0):
        self.session = session
        self.env = env = session.env
        self.target = session.target
        self.model = model  # stage A responses: J_spot for the FEH correctors
        self.centre = np.array([session.settings()[n] for n in KNOBS])
        self.half = self._half_ranges(k1)
        self.base = {**_corrector_zero(env), **k1,
                     **{n: env.twin._knobs[n].value for n in SEXTUPOLES}}
        self.rng = np.random.default_rng(seed)
        # the low fidelity: Sobol points over the box, through the model
        x = qmc.Sobol(len(KNOBS), seed=seed).random(n_model) * 2 - 1
        x[0] = 0
        y = self._model_readouts(x)
        ok = np.all(np.isfinite(y), axis=1)
        self.low = [_fit(x[ok], y[ok, i]) for i in range(5)]
        self.x_high: list[np.ndarray] = []
        self.y_high: list[np.ndarray] = []
        self.readings: list = []
        self.high: list[Discrepancy] = []

    # ---------- the two fidelities ----------

    def _half_ranges(self, k1: dict[str, float]) -> np.ndarray:
        """Per knob, the half-width of the box, within the magnet limits."""
        from .responses import ResponseTable  # the table's size response
        at = ResponseTable().at(self.target[2], self.target[3])
        dof = ResponseTable().dof
        J = at["J"]
        half = []
        for name, centre in zip(KNOBS, self.centre):
            knob = self.env._knobs[name]
            if name in SEXTUPOLES:
                h = knob.high
            else:
                c = dof.index(name)
                pull = max(abs(J[2, c]) / self.target[2], abs(J[3, c]) / self.target[3])
                h = QUAD_REACH / max(pull, 1e-9)
            h = min(h, knob.high - centre, centre - knob.low) if name not in SEXTUPOLES else h
            half.append(max(h, 1e-3))
        return np.array(half)

    def _settings(self, x: np.ndarray) -> dict[str, float]:
        return dict(zip(KNOBS, (self.centre + self.half * np.asarray(x)).tolist()))

    def _model_readouts(self, x: np.ndarray) -> np.ndarray:
        """The twin's readout (mu_x, mu_y, sigma_x, sigma_y, T) at every row of x,
        in one vectorised Cheetah track."""
        values = self.centre + self.half * np.atleast_2d(x)
        stats, transmission, names = probe_batch(
            self.env, dict(zip(KNOBS, values.T)), self.base)
        readout = stats[:, names.index(self.env.config.target_screen)]
        return np.column_stack([readout, transmission])

    def _posterior(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and sd of the machine's five readouts at the points x."""
        X = torch.as_tensor(np.atleast_2d(x), dtype=DTYPE)
        mean, var = [], []
        with torch.no_grad():
            for i, low in enumerate(self.low):
                p = low.posterior(X)
                m, v = p.mean.squeeze(-1), p.variance.squeeze(-1)
                if self.high:
                    dm, dv = self.high[i](X)
                    m, v = m + dm, v + dv
                mean.append(m.numpy())
                var.append(v.numpy())
        return np.array(mean).T, np.sqrt(np.maximum(np.array(var).T, 0))

    # ---------- the objective ----------

    def mismatch(self, y: np.ndarray) -> np.ndarray:
        """The benchmark mismatch of readouts y (..., 5), as `SimEnv` scores it."""
        t = self.target
        scale = t[2:].mean()
        err = np.abs(y[..., :4] - t) / scale
        T = np.clip(y[..., 4], 0, 1)
        return np.minimum((err[..., 0] + err[..., 1]) / 2 + (err[..., 2] + err[..., 3]) / 2
                          + TRANSMISSION_WEIGHT * (1 - T), 10.0)

    def _position_fix(self, mu: np.ndarray) -> np.ndarray:
        """FEH corrector changes that move a predicted centroid onto the target."""
        return -np.linalg.pinv(self.model["J_spot"]) @ (mu - self.target[:2])

    # ---------- the loop ----------

    def observe(self, x: np.ndarray, reading, kicks: np.ndarray) -> None:
        """Book a machine reading taken at x with these FEH corrector changes."""
        info = reading.info
        if not info["on_screen"]:
            y = np.array([np.nan] * 4 + [info["transmission"]])
        else:
            y = np.array([info["mu_x"], info["mu_y"], info["sigma_x"],
                          info["sigma_y"], info["transmission"]])
            y[:2] -= self.model["J_spot"] @ kicks  # back to "no correction"
        self.x_high.append(np.asarray(x, dtype=float))
        self.y_high.append(y)
        self.readings.append(reading)
        self._refit()

    def _refit(self) -> None:
        x = np.array(self.x_high)
        y = np.array(self.y_high)
        ok = np.all(np.isfinite(y), axis=1)
        if not ok.any():
            return
        low_mean = np.array([[low.posterior(torch.as_tensor(x[ok], dtype=DTYPE)).mean.squeeze(-1).detach().numpy()]
                             for low in self.low]).squeeze(1).T
        delta = y[ok] - low_mean
        # shot jitter at the readout, roughly: 2 % of the size, 2 % of the spot in position
        noise = np.array([0.02 * self.target[2], 0.02 * self.target[3],
                          0.02 * self.target[2], 0.02 * self.target[3], 0.005])
        self.high = [Discrepancy(x[ok], delta[:, i], noise[i]) for i in range(5)]

    def best(self) -> tuple[int, float]:
        """The reading with the lowest scored mismatch so far."""
        scores = [r.info["scored_mismatch"] for r in self.readings]
        i = int(np.argmin(scores))
        return i, scores[i]

    def propose(self) -> tuple[np.ndarray, float]:
        """The candidate with the largest expected improvement, and that EI."""
        d = len(KNOBS)
        cands = [qmc.Sobol(d, seed=int(self.rng.integers(1 << 30))).random(CANDIDATES) * 2 - 1]
        if self.x_high:
            ib, _ = self.best()
            xb = self.x_high[ib]
            for width in (0.05, 0.15, 0.3):
                cands.append(np.clip(xb + width * self.rng.standard_normal((CANDIDATES // 4, d)), -1, 1))
        X = np.vstack(cands)
        mean, sd = self._posterior(X)
        mean[:, :2] = self.target[:2]  # the FEH correctors put the spot where we predict
        z = self.rng.standard_normal((SAMPLES, 1, 5))
        samples = mean[None] + sd[None] * z
        m = self.mismatch(samples)  # (SAMPLES, n)
        incumbent = self.best()[1] if self.readings else self.mismatch(mean[:1])[0]
        ei = np.maximum(incumbent - m, 0).mean(axis=0)
        i = int(np.argmax(ei))
        return X[i], float(ei[i])

    def shot(self, x: np.ndarray):
        """Set x, with the FEH correctors moving the predicted spot onto the
        target, and read."""
        mean, _ = self._posterior(x)
        kicks = self._position_fix(mean[0, :2])
        base_kicks = {n: self.base_feh[n] for n in FINAL_CORRECTORS}
        action = {**self._settings(x)}
        for name, dk in zip(FINAL_CORRECTORS, kicks):
            action[name] = _clip(self.env, name, base_kicks[name] + dk)
        applied = np.array([action[n] - base_kicks[n] for n in FINAL_CORRECTORS])
        reading = self.session.step({"magnets": action})
        self.observe(x, reading, applied)
        return reading

    def run(self, max_steps: int | None = None) -> None:
        session = self.session
        self.base_feh = {n: session.settings()[n] for n in FINAL_CORRECTORS}
        # the stage A reading is the first high-fidelity point, at x = 0
        self.observe(np.zeros(len(KNOBS)), session.last, np.zeros(4))
        budget = session.steps_left if max_steps is None else min(max_steps, session.steps_left)
        while budget > 1:
            x, ei = self.propose()
            if ei < step_cost(session.steps):
                break
            self.shot(x)
            budget -= 1
        i, _ = self.best()
        if self.readings[i] is not session.last and session.steps_left > 0:
            session.step({"magnets": dict(self.readings[i].info["magnets"])})


def mfbo(session: Session, max_steps: int | None = None) -> SpotBO:
    """Stage A, then stage D."""
    from .steering import optics_and_steering, table_responses
    optics_and_steering(session)
    t = session.target
    optics = session.env.optics_for_target(t[2], t[3], modes=False)
    bo = SpotBO(session, optics.k1, table_responses(session.env, t[2], t[3]))
    bo.run(max_steps)
    return bo
