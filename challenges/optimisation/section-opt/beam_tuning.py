"""Shared S04 experiment: normalized controls, simulated measurements and strict budgets.

No network or live-machine access. Particle-beam moments reproduce the kickoff target.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import cheetah
import numpy as np
import torch
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[1]
TARGET = np.array([0., 0., 100e-6, 100e-6])
QUADS = [f"cla_s04_mag_quad_0{i}" for i in (1, 2, 3, 4)]
CORRECTORS = ["cla_s04_mag_hvcor_01", "cla_s04_mag_hvcor_02"]
LOWER = np.array([-3.] * 4 + [-5e-4] * 4)
UPPER = -LOWER
KNOB_NAMES = QUADS + [f"{c}_{p}" for c in CORRECTORS for p in ("h", "v")]
ENERGY = 193.156e6
EPS = 1e-6 / (ENERGY / .51099895e6)
DTYPE = torch.float32


def normalize(x):
    return (np.asarray(x) - LOWER) / (UPPER - LOWER)


def physical(u):
    return LOWER + np.asarray(u) * (UPPER - LOWER)


def checked_cube(u, size=8):
    u = np.asarray(u, dtype=float)
    if u.shape != (size,) or not np.isfinite(u).all():
        raise ValueError(f"Expected a finite vector of length {size}")
    if np.any(u < -1e-9) or np.any(u > 1 + 1e-9):
        raise ValueError("Knob outside normalized [0, 1] bounds")
    return u.clip(0, 1)


def sample_beam(seed, num_particles=10_000):
    """Kickoff distribution; seed both Twiss sampling and macroparticles."""
    rng = np.random.default_rng(seed)
    bx, ax, by, ay = [rng.uniform(a, b) for a, b in [(8, 16), (-1, 1), (8, 16), (-1, 1)]]
    mx, mpx, my, mpy = [rng.uniform(-a, a) for a in [3e-4, 1e-4, 3e-4, 1e-4]]
    kwargs = dict(energy=ENERGY, mu_x=mx, mu_px=mpx, mu_y=my, mu_py=mpy,
                  sigma_x=np.sqrt(EPS * bx), sigma_px=np.sqrt(EPS * (1 + ax**2) / bx),
                  cov_xpx=-EPS * ax, sigma_y=np.sqrt(EPS * by),
                  sigma_py=np.sqrt(EPS * (1 + ay**2) / by), cov_ypy=-EPS * ay,
                  sigma_tau=1e-4, sigma_p=1e-3)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return cheetah.ParticleBeam.from_parameters(
            num_particles=num_particles, dtype=DTYPE,
            **{k: torch.tensor(v, dtype=DTYPE) for k, v in kwargs.items()})


def moments(beam):
    return np.array([getattr(beam, k).item() for k in ("mu_x", "mu_y", "sigma_x", "sigma_y")])


def beam_state(beam):
    """Synthetic reconstructed 6D state, NOT obtainable from one screen image."""
    p = beam.particles[..., :6].detach().double()
    return p.mean(0).numpy(), torch.cov(p.T).numpy()


def context(beam, mode="screen"):
    if mode == "none":
        return np.empty(0)
    if mode == "screen":
        return encode(moments(beam))
    if mode != "state":
        raise ValueError(mode)
    mu, cov = beam_state(beam)
    # Transverse state is sufficient in this uncoupled, fixed-energy straight section.
    return np.r_[mu[:4] / 1e-4, cov[0, 0] / 1e-8, cov[0, 1] / 1e-8,
                 cov[1, 1] / 1e-8, cov[2, 2] / 1e-8, cov[2, 3] / 1e-8, cov[3, 3] / 1e-8]


def encode(y):
    """Dimensionless centroids and log sizes; decoded sizes are always positive."""
    y = np.asarray(y)
    return np.concatenate([y[..., :2] / 1e-4, np.log(np.maximum(y[..., 2:], 1e-10) / 1e-4)], axis=-1)


def decode(y):
    y = np.asarray(y)
    return np.concatenate([y[..., :2] * 1e-4, np.exp(np.clip(y[..., 2:], -15, 15)) * 1e-4], axis=-1)


def loss(y, part="all"):
    y = np.asarray(y)
    if not np.isfinite(y).all():
        return 5.0
    residual = np.abs(y - TARGET) * 1e3
    return float(residual.mean() if part == "all" else residual[:2].mean() if part == "orbit" else residual[2:].mean())


class S04Machine:
    """One final-screen measurement per call; optional fixed hidden calibration errors.

    All imported Screen elements (including BPM/marker placeholders) are inactive.
    We use the propagated final moments as an idealized reading on SCR-03, as kickoff
    does. This never claims simultaneous upstream/downstream destructive-screen data.
    """
    def __init__(self, error_seed=None):
        lattice = cheetah.Segment.from_lattice_json(str(ROOT / "data/clara/lattice_data/CLARA_cheetah.json"))
        self.section = lattice.partition_at("cla_s04_sim_aper_01", mode="before")[-1].partition_at(
            "cla_s04_dia_scr_03", mode="after")[0]
        for element in self.section.elements:
            if isinstance(element, cheetah.Screen):
                element.is_active = False
        self.design = np.array([float(getattr(self.section, q).k1) for q in QUADS] + [0.] * 4)
        self.gains, self.offsets = np.ones(8), np.zeros(8)
        if error_seed is not None:
            rng = np.random.default_rng(error_seed)
            self.gains += rng.normal(0, .03, 8)
            self.offsets = rng.normal(size=8) * np.array([.03] * 4 + [8e-6] * 4)
        self.u0 = normalize(self.design)

    def apply(self, u):
        x = physical(checked_cube(u)) * self.gains + self.offsets
        for name, k1 in zip(QUADS, x[:4]):
            getattr(self.section, name).k1 = torch.tensor(float(k1), dtype=DTYPE)
        for i, name in enumerate(CORRECTORS):
            c = getattr(self.section, name)
            c.horizontal_angle = torch.tensor(float(x[4 + 2*i]), dtype=DTYPE)
            c.vertical_angle = torch.tensor(float(x[5 + 2*i]), dtype=DTYPE)

    def truth(self, u, beam):
        self.apply(u)
        with torch.no_grad():
            return moments(self.section.track(beam))

    def transport(self, u, beam):
        """First-order affine map in Cheetah coordinates (x, px, y, py, tau, p)."""
        self.apply(u)
        with torch.no_grad():
            t = self.section.first_order_transfer_map(beam.energy, beam.species).detach().double().numpy()
        return t[:6, :6], t[:6, 6]

    def predict_state(self, u, beam):
        mu, cov = beam_state(beam)
        r, d = self.transport(u, beam)
        out_mu, out_cov = r @ mu + d, r @ cov @ r.T
        return np.r_[out_mu[[0, 2]], np.sqrt(np.maximum(np.diag(out_cov)[[0, 2]], 0))]


class BudgetExhausted(RuntimeError):
    pass


class MeasurementSession:
    """Ledger of ALL screen shots, including response scans and repeats.

    Noise is a toy iid moment-readout model, not a validated camera/noise model.
    Reference truth is retained for simulation diagnostics, never passed to a tuner.
    """
    def __init__(self, machine, beam, budget=80, noise_fraction=0., seed=0):
        self.machine, self.beam, self.budget = machine, beam, budget
        self.noise_fraction = noise_fraction
        self.rng = np.random.default_rng(seed)
        self.records = []
        self.started = time.perf_counter()

    @property
    def remaining(self):
        return self.budget - len(self.records)

    def measure(self, u, part="all"):
        if self.remaining <= 0:
            raise BudgetExhausted("Screen-shot budget exhausted")
        u = checked_cube(u)
        clean = self.machine.truth(u, self.beam)
        sigma = self.noise_fraction * float(np.mean(clean[2:])) if np.isfinite(clean).all() else 0.
        observed = clean + self.rng.normal(0, sigma, 4)
        observed[2:] = np.maximum(observed[2:], 1e-10)
        n = 4 if part == "all" else 2
        # Conservative propagated SE away from the abs cusp; noise floor is numerical only.
        error = max(sigma * 1e3 / np.sqrt(n), 1e-8)
        self.records.append(dict(u=u.copy(), measured=observed.copy(), truth=clean.copy(),
                                 value=loss(observed, part), error=error, part=part,
                                 seconds=time.perf_counter() - self.started))
        return observed, error

    def objective(self, u, part="all"):
        y, e = self.measure(u, part)
        return loss(y, part), e

    def best(self):
        # Select using measured full objective, never using simulator truth.
        return min(self.records, key=lambda r: loss(r["measured"]))

    def curve(self):
        return np.minimum.accumulate([loss(r["measured"]) for r in self.records])


def rcds_search(objective, x0, budget, method="parabola", seed=0, directions=None, **options):
    """Use the user's RCDS on ANY unit-cube scalar (value, SE) objective.

    A wrapper enforces the budget even midway through a bracket or GP line search.
    Return the best actually evaluated point, not an unevaluated fit minimum.
    """
    rcds_path = str(ROOT / "work/rcds")
    if rcds_path not in sys.path:
        sys.path.insert(0, rcds_path)
    from rcds.wrapper import Wrapper
    from rcds.minimizer import Minimizer
    x0 = np.asarray(x0, dtype=float)
    if directions is None:
        directions = np.eye(len(x0))
    directions = np.asarray(directions, dtype=float)
    if directions.shape != (len(x0), len(x0)) or not np.isfinite(directions).all():
        raise ValueError(f"Expected a finite {len(x0)}x{len(x0)} direction matrix")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms < 1e-12) or np.linalg.matrix_rank(directions) != len(x0):
        raise ValueError("Initial RCDS directions must be nonzero and full rank")
    # The bundled RCDS consumes one direction per ROW. Unit rows make its step fraction
    # comparable across identity, physics, and learned bases.
    directions = directions / norms[:, None]
    history = []

    @Wrapper(nk=len(x0), dtype=torch.float64, cache=False)
    def wrapped(x):
        if len(history) >= budget:
            raise BudgetExhausted()
        u = checked_cube(x.detach().cpu().numpy(), len(x0))
        v, e = objective(u)
        history.append((u.copy(), float(v)))
        return torch.tensor(v, dtype=torch.float64), torch.tensor(max(e, 1e-8), dtype=torch.float64)

    wrapped()
    minimizer = Minimizer(wrapped)
    kwargs = dict(ni=1000, sf=.03, termination=False, warning=False)
    if method == "gp":
        kwargs.update(minimize=minimizer.minimize_gp, no_ucb=3, no_ei=2, nr=4,
                      rs=32, beta=2., use_parabola=False)
    elif method == "parabola":
        kwargs.update(minimize=minimizer.minimize_parabola, np=2, ns=3)
    else:
        raise ValueError(method)
    kwargs.update(options)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        try:
            minimizer.rcds(
                torch.tensor(x0, dtype=torch.float64),
                torch.tensor(directions, dtype=torch.float64),
                **kwargs,
            )
        except BudgetExhausted:
            pass
    if not history:
        raise ValueError("RCDS needs at least one evaluation")
    return min(history, key=lambda row: row[1])[0], history


def response_directions(predict, center, output_scale=None, step=0.02):
    """Return RCDS row directions from a local normalized output response.

    ``predict(u)`` returns observable outputs. Central finite differences form the
    response J with respect to normalized knobs. Rows of Vh from the full SVD of J are
    eigenvectors of the Gauss-Newton matrix J.T @ J and include a full orthonormal basis
    for the response null space when there are fewer outputs than controls.
    """
    center = np.asarray(center, dtype=float)
    if center.ndim != 1 or center.size == 0:
        raise ValueError("center must be a nonempty one-dimensional vector")
    center = checked_cube(center, center.size)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")
    baseline = np.asarray(predict(center), dtype=float)
    if baseline.ndim != 1 or baseline.size == 0 or not np.isfinite(baseline).all():
        raise ValueError("predict(center) must be a finite one-dimensional vector")
    scale = np.ones_like(baseline) if output_scale is None else np.asarray(output_scale, dtype=float)
    if scale.shape != baseline.shape or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("output_scale must be finite, positive, and match predict(center)")
    response = np.empty((baseline.size, center.size), dtype=float)
    for index in range(center.size):
        low, high = center.copy(), center.copy()
        low[index] = max(0.0, center[index] - step)
        high[index] = min(1.0, center[index] + step)
        distance = high[index] - low[index]
        if distance <= 0:
            raise ValueError("Finite-difference direction has zero length")
        response[:, index] = (
            np.asarray(predict(high), dtype=float) - np.asarray(predict(low), dtype=float)
        ) / distance / scale
    _, singular_values, directions = np.linalg.svd(response, full_matrices=True)
    # np.linalg.svd returns Vh: each row is a right singular vector, exactly the
    # orientation consumed by Minimizer.rcds(matrix[i]).
    return directions, singular_values, response


def tune(session, method, seed=0, **options):
    x0 = session.machine.u0
    if method in ("rcds", "rcds_bo"):
        rcds_search(session.objective, x0, session.remaining,
                    "gp" if method == "rcds_bo" else "parabola", seed, **options)
    elif method == "random":
        rng = np.random.default_rng(seed)
        session.objective(x0)
        while session.remaining:
            session.objective(rng.uniform(size=8))
    elif method == "bo":
        from skopt import gp_minimize
        from skopt.space import Real
        n = session.remaining
        gp_minimize(lambda u: session.objective(u)[0], [Real(0., 1.) for _ in range(8)],
                    x0=x0.tolist(), n_calls=n, n_initial_points=min(19, n - 1),
                    acq_func="EI", noise=1e-10 if session.noise_fraction == 0 else "gaussian",
                    random_state=seed)
    elif method == "sequential":
        # Optimize only sizes first, then calibrate steering at the chosen optics.
        optics_budget = max(1, session.remaining - 10)
        fixed = x0.copy()
        def optics(q):
            u = fixed.copy(); u[:4] = q
            return session.objective(u, "shape")
        q, _ = rcds_search(optics, fixed[:4], optics_budget, seed=seed)
        fixed[:4] = q
        if session.remaining >= 6:
            center, _ = session.measure(fixed)
            response = np.zeros((2, 4))
            for j in range(4):
                probe = fixed.copy(); probe[4+j] += .04
                y, _ = session.measure(probe)
                response[:, j] = (y[:2] - center[:2]) / .04
            # Regularization chooses small corrector changes in the 2D null space.
            a = np.vstack([response / 1e-4, .01 * np.eye(4)])
            b = np.r_[-center[:2] / 1e-4, np.zeros(4)]
            delta = lsq_linear(a, b, bounds=(-fixed[4:], 1-fixed[4:])).x
            fixed[4:] += delta
            session.measure(fixed)
            # Remaining shots assess repeatability; they are not hidden/free.
            while session.remaining:
                session.measure(fixed)
    else:
        raise ValueError(method)
    return session.best()["u"].copy()


def local_response_candidate(us, ys, center, radius=.1):
    """Ridge-regularized local response of four outputs to eight normalized knobs."""
    us, ys, center = np.asarray(us), np.asarray(ys), np.asarray(center)
    design = np.c_[np.ones(len(us)), us - center]
    coefficients = np.linalg.solve(design.T @ design + np.diag([1e-10] + [1e-5]*8),
                                   design.T @ (ys / 1e-4))
    low, high = np.maximum(-radius, -center), np.minimum(radius, 1-center)
    delta = lsq_linear(np.vstack([coefficients[1:].T, .01*np.eye(8)]),
                       np.r_[TARGET / 1e-4 - coefficients[0], np.zeros(8)],
                       bounds=(low, high)).x
    return center + delta
