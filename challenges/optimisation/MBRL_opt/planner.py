"""Choosing the next set of magnet settings, by asking the twin what they would do.

Everything here is anchored on the last measurement rather than on the twin's own
absolute prediction::

    predicted(delta) = measured + [twin(knobs + delta) - twin(knobs)]

The twin is asked only what *changes*, never what the reading is. That matters because
the twin is not a perfect model of the machine it is on — `system_id` fits the beam and
the survey error from a handful of shots and gets most of the way, not all of it — and
a model that is wrong by a constant still predicts the effect of moving a quadrupole
perfectly well. Anchoring cancels the constant exactly and leaves only the error in the
*derivative*, which is far smaller. It is also what an operator does: you do not trust
the model to tell you where the beam is, you trust it to tell you which way to turn.

Two stages, because they are different problems:

`steer` flattens the orbit on the BPMs with the correctors. The orbit is linear in the
corrector kicks, so the whole problem is one response matrix and a truncated
pseudo-inverse — the classical method, with the matrix coming from the twin's own
autograd rather than from shots on the machine. Same recipe as
`inference/steering/svd_correction.py`, which does it against the xsuite design model.

`focus` shapes the beam on the readout screen with the quadrupoles, sextupoles, RF
phases and the chicane. That is not linear and not even monotonic — a quadrupole swept
far enough takes the waist through the screen and out the other side — so it gets CEM
over a population of candidates, which a batched twin evaluates all at once, followed
by gradient descent to polish the best one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from knobs import KnobSet
from system_id import SOFT_APERTURE
from twin import DTYPE, CheetahTwin, LatentState

# Step used to measure a response, in units of each knob's natural step
# (`knobs.SCALE`). Big enough that the answer is not rounding, small enough that the
# response to it is still a derivative: a fifth of a step is 2e-5 rad on a corrector,
# which is what `MFopt.optics_env` probes with.
PROBE_FRACTION = 0.2
# Singular values below this fraction of the largest are dropped from the orbit
# pseudo-inverse. The small ones are corrector combinations the BPMs cannot see, and
# inverting them asks for enormous kicks that achieve nothing.
SVD_CUTOFF = 0.05
# Fraction of the measured orbit to remove per shot. Below 1 because the response is
# measured on a model and the orbit is measured through noise; converging over three
# shots beats overshooting in one.
STEERING_GAIN = 0.7

# How far the planner's cost is allowed to see, in requested spot sizes. The
# benchmark stops counting at ten, and a planner must not: an untuned machine sits
# at seventy or more — a 1.5 mm spot against an 85 micron request is eighteen spot
# sizes in size alone, before the position — so a cost clipped at the benchmark's own
# scale is flat across the entire region the tuning happens in, and CEM and gradient
# descent both see a plateau with no downhill on it. This cap exists only to keep a
# wild candidate from dominating the elite selection. `SimEnv._compute_reward` makes
# the same point about its own `MAX_MISMATCH`.
MAX_MISMATCH = 200.0
# What losing the beam costs, on top of whatever the mismatch already was. Larger
# than any mismatch worth delivering, so a tune that cannot transport beam is never
# preferred to a poor one that can.
LOST_BEAM_COST = 50.0
# What beam that never arrives costs, in requested spot sizes. `sim_env` and
# `score.py` both charge this, so it is part of the objective and not a preference
# of the planner's: delivering half the bunch is worth a whole spot size.
TRANSMISSION_WEIGHT = 2.0
# What scraping costs, per metre of beam outside the pipe. Transmission already
# charges for the particles that have gone; this charges for sitting close enough
# to the wall that the next shot's jitter will take some more.
SCRAPE_WEIGHT = 20.0
# Largest gradient step the plan polish may take, in units of the trust region.
MAX_GRAD_NORM = 1.0


@dataclass
class Plan:
    """What to set next, and what the twin thinks it will do."""

    values: np.ndarray  # native units, in the knob set's order
    predicted: np.ndarray  # (mu_x, mu_y, sigma_x, sigma_y) in m, or the orbit rms
    cost: float  # what the planner was minimising, at this candidate
    note: str = ""


def benchmark_mismatch(
    moments: torch.Tensor,
    target: torch.Tensor,
    cap: float = MAX_MISMATCH,
    position_weight: float = 1.0,
) -> torch.Tensor:
    """`SimEnv._mismatch`: the error in units of the spot size that was asked for.

    Position and size weighted equally, each component against the mean requested
    spot, which is what makes a tight request no harder to score on than a loose one.
    Differentiable, and batched over any leading dimension.

    `position_weight` of 0 scores the size alone. The focus uses that: where the spot
    lands is a corrector's job and a linear one, and leaving it in the objective only
    tempts the quadrupoles into fixing a 2 mm offset by distorting the optics, which
    they can do and which ruins the size. The position is put right afterwards, in
    one shot, by the magnets that actually move it.
    """
    scale = target[2:].mean()
    error = (moments - target).abs() / scale
    total = (error[..., 2] + error[..., 3]) / 2
    if position_weight:
        total = total + position_weight * (error[..., 0] + error[..., 1]) / 2
    return total.clamp(max=cap)


class Planner:
    """Plans a shot against the twin, anchored on the last reading of the machine."""

    def __init__(self, twin: CheetahTwin, config):
        self.twin = twin
        self.config = config

    # ---------- what the twin says a knob does ----------

    def _overrides(self, knobs: KnobSet, values: np.ndarray) -> dict:
        """A knob vector as the twin's override mapping."""
        return {
            (element, prop): torch.tensor(float(value), dtype=DTYPE)
            for element, prop, value in zip(knobs.elements, knobs.props, values)
        }

    def _batched_overrides(self, knobs: KnobSet, population: np.ndarray) -> dict:
        """A whole population of knob vectors, one batch row each."""
        tensor = torch.tensor(np.asarray(population), dtype=DTYPE)
        return {
            (element, prop): tensor[:, index]
            for index, (element, prop) in enumerate(zip(knobs.elements, knobs.props))
        }

    def response(
        self, knobs: KnobSet, values: np.ndarray, state: LatentState, outputs
    ) -> tuple[np.ndarray, np.ndarray]:
        """How each knob moves each output, and the outputs where they are now.

        One batched track does the lot: column 0 is the machine as it stands and
        column i + 1 is knob i nudged, so a sixteen-corrector response matrix costs
        one track rather than seventeen. `outputs` pulls the quantities of interest
        out of a `Readings`.
        """
        steps = knobs.scales * PROBE_FRACTION
        population = np.repeat(values[None, :], len(knobs) + 1, axis=0)
        for index in range(len(knobs)):
            population[index + 1, index] += steps[index]
        with torch.no_grad():
            readings = self.twin.readings(
                state=state,
                overrides=self._batched_overrides(knobs, population),
                batch=len(knobs) + 1,
            )
        sampled = outputs(readings).numpy()
        return (sampled[1:] - sampled[0]).T / steps, sampled[0]

    # ---------- stage one: steer on the BPMs ----------

    def steer(
        self,
        knobs: KnobSet,
        values: np.ndarray,
        state: LatentState,
        measured_x: np.ndarray,
        measured_y: np.ndarray,
        valid: np.ndarray,
    ) -> Plan:
        """Corrector settings that flatten the measured orbit.

        Only BPMs that are actually reading constrain anything — past the point the
        beam is lost there is no orbit to flatten — so the rows of the response matrix
        are masked down to those, and the correction is whatever flattens the stretch
        that can be seen. The next shot sees further, and the one after that further
        still, which is how the beam is walked down a line it does not currently reach.
        """
        reading = np.concatenate([measured_x, measured_y])
        mask = np.concatenate([valid, valid]).astype(bool)
        if not mask.any():
            return Plan(values, np.array([np.inf]), np.inf, "no BPM is reading")

        jacobian, _ = self.response(
            knobs,
            values,
            state,
            lambda r: torch.cat([r.bpm_x, r.bpm_y], dim=-1),
        )
        matrix = jacobian[mask]
        correction = -STEERING_GAIN * (_pseudo_inverse(matrix, SVD_CUTOFF) @ reading[mask])

        low, high = knobs.trust_region(values, self.config.trust_region)
        proposed = np.clip(values + correction, low, high)
        predicted = reading.copy()
        predicted[mask] = reading[mask] + matrix @ (proposed - values)
        rms = float(np.sqrt((predicted[mask] ** 2).mean()))
        return Plan(
            proposed,
            predicted,
            rms,
            f"{int(valid.sum())}/{len(valid)} BPMs reading, orbit -> {rms * 1e6:.0f} um",
        )

    # ---------- stage two: shape the beam on the readout ----------

    def focus(
        self,
        knobs: KnobSet,
        values: np.ndarray,
        state: LatentState,
        target: np.ndarray,
        measured: np.ndarray | None,
        rng: np.random.Generator,
        trust: float | None = None,
        explore: bool = True,
        measured_at: np.ndarray | None = None,
        position_weight: float = 1.0,
    ) -> Plan:
        """Knob settings that bring the readout closer to what was asked for.

        CEM first, over a population the twin evaluates in one batch, because the map
        from a quadrupole to a spot size is not monotonic and gradient descent will
        happily settle into whichever side of the waist it started on. Then gradient
        descent on the winner, because CEM gets the basin and is slow at the last
        factor of two.

        `explore` turns the CEM half off, and the caller should turn it off most of
        the time. Batching is what makes CEM cheap, and it stops being cheap for
        exactly these knobs: an RF phase or a chicane angle changes the energy the
        beam carries, so a batch of candidate phases is a batch of different beam
        energies and every element downstream has to be evaluated once per candidate
        instead of once. Measured on the twin at 128 candidates, six quadrupoles cost
        287 ms and the full seventeen knobs 1675 ms. Searching globally is worth that
        when the tune is stuck; it is not worth it on a shot that is already
        converging, where the gradient alone knows which way to go.
        """
        trust = self.config.trust_region if trust is None else trust
        # `values` is where the search starts, which a policy may have moved away
        # from the machine; `measured_at` is where the magnets actually are, and is
        # what the anchor and the trust region have to be about. The reading belongs
        # to the real setting, and a shot may only move so far from it however
        # ambitious the suggestion.
        at = values if measured_at is None else np.asarray(measured_at, dtype=float)
        anchor = self._anchor(knobs, at, state, measured)
        low, high = knobs.trust_region(at, trust)
        values = knobs.clip(np.clip(values, low, high))
        target_t = torch.tensor(target, dtype=DTYPE)

        best, cost = values.copy(), float("inf")
        if explore:
            best, cost = self._cem(
                knobs, values, state, target_t, anchor, low, high, rng, position_weight
            )
        best, cost = self._descend(
            knobs, best, state, target_t, anchor, low, high, cost, trust, position_weight
        )

        with torch.no_grad():
            readings = self.twin.readings(
                state=state, overrides=self._overrides(knobs, best)
            )
        predicted = (readings.planning_moments() + anchor).numpy()
        return Plan(best, predicted, float(cost), f"predicted mismatch {cost:.3f}")

    def _anchor(self, knobs, values, state, measured) -> torch.Tensor:
        """What to add to the twin's prediction so it agrees with the last reading.

        Zero when there was no reading to agree with — a beam that missed the screen
        says nothing about where it would have landed — in which case the twin's own
        absolute prediction is all there is, which is exactly the situation where its
        raw moments are the thing being steered by.
        """
        if measured is None:
            return torch.zeros(4, dtype=DTYPE)
        with torch.no_grad():
            readings = self.twin.readings(
                state=state, overrides=self._overrides(knobs, values)
            )
        return torch.tensor(measured, dtype=DTYPE) - readings.planning_moments()

    def _cost(self, readings, target, anchor, position_weight: float = 1.0) -> torch.Tensor:
        """What a candidate is worth: how far off the request, plus what it risks.

        The first two terms are the benchmark's own score — the mismatch against the
        request, and `TRANSMISSION_WEIGHT` for the fraction of the bunch that never
        arrived. Transmission matters here as much as the spot does: the nominal tune
        already loses about half the beam scraping through the FEA arc, which is a
        whole requested spot size of score before the focus has done anything.

        The last two are the planner's own caution. A candidate that puts the beam
        off the screen is charged more than the worst delivered beam can cost, and one
        merely sitting close to the pipe wall is charged in proportion — both because
        a tune on the edge will not survive the next shot's jitter, and because the
        twin's prediction is least trustworthy exactly there.
        """
        mismatch = benchmark_mismatch(
            readings.planning_moments() + anchor, target, position_weight=position_weight
        )
        missing = TRANSMISSION_WEIGHT * (1.0 - readings.transmission)
        lost = (1.0 - readings.on_screen) * LOST_BEAM_COST
        return mismatch + missing + lost + SCRAPE_WEIGHT * readings.scrape

    def _cem(self, knobs, values, state, target, anchor, low, high, rng, position_weight=1.0) -> tuple:
        """Cross-entropy method: sample, keep the best, resample around them."""
        config = self.config
        mean = values.copy()
        deviation = (high - low) / 4
        best, best_cost = values.copy(), np.inf
        for _ in range(config.cem_iterations):
            population = rng.normal(
                mean, np.maximum(deviation, 1e-12), (config.cem_population, len(knobs))
            )
            population = np.clip(population, low, high)
            population[0] = mean  # never lose the incumbent to bad luck
            with torch.no_grad():
                readings = self.twin.readings(
                    state=state,
                    overrides=self._batched_overrides(knobs, population),
                    batch=len(population),
                )
                costs = self._cost(readings, target, anchor, position_weight).numpy()
            # A candidate wild enough to make the twin produce nothing at all is
            # simply the worst candidate, not a reason to stop. Ranking it as
            # infinite keeps it out of the elite set and out of the next mean.
            costs = np.nan_to_num(costs, nan=np.inf, posinf=np.inf)
            elite = population[np.argsort(costs)[: config.cem_elites]]
            mean, deviation = elite.mean(axis=0), elite.std(axis=0)
            if costs.min() < best_cost:
                best_cost, best = float(costs.min()), population[int(np.argmin(costs))]
        return best, best_cost

    def _descend(self, knobs, values, state, target, anchor, low, high, cost, trust, position_weight=1.0) -> tuple:
        """Adam on the knobs, straight through Cheetah, to polish CEM's answer.

        Knobs are stepped in units of their own natural step so one learning rate
        suits a corrector angle and a quadrupole strength at once, and the apertures
        carry a surrogate gradient so a candidate that is scraping still knows which
        way to come back.
        """
        config = self.config
        # The step is measured in units of how far this shot is allowed to move,
        # not of the knob's full range. Adam moves each parameter by about the
        # learning rate per iteration whatever it is scaled in, so against the full
        # span a rate of 0.05 would be over half the trust region in one step: the
        # descent would hit the boundary immediately and spend every remaining
        # iteration clamped against it. Against the reach, the rate means what it
        # looks like — a twentieth of the allowed move.
        reach = torch.tensor(knobs.scales * trust, dtype=DTYPE)
        floor = torch.tensor(low, dtype=DTYPE)
        ceiling = torch.tensor(high, dtype=DTYPE)
        unit = (torch.tensor(values, dtype=DTYPE) / reach).requires_grad_(True)
        optimiser = torch.optim.Adam([unit], lr=config.plan_learning_rate)

        best, best_cost = values.copy(), cost
        for _ in range(config.plan_iterations):
            optimiser.zero_grad(set_to_none=True)
            native = torch.clamp(unit * reach, floor, ceiling)
            readings = self.twin.readings(
                state=state,
                overrides={
                    (element, prop): native[index]
                    for index, (element, prop) in enumerate(
                        zip(knobs.elements, knobs.props)
                    )
                },
                soft=SOFT_APERTURE,
            )
            loss = self._cost(readings, target, anchor, position_weight)
            # A candidate the twin could not evaluate is the worst candidate, not a
            # reason to reject every later one: comparing against NaN is always
            # False, so without this a single bad first step leaves `best_cost` at
            # infinity and the whole shot is thrown away having moved nothing.
            value = float(loss.detach())
            if not np.isfinite(value):
                value = np.inf
            if value < best_cost or not np.isfinite(best_cost):
                best_cost, best = value, native.detach().numpy().copy()

            loss.backward()
            # A candidate that has thrown the whole beam away measures its spot size
            # as the rms of nothing, whose derivative is enormous and occasionally
            # not a number at all. Left alone that reaches the knobs, and an RF phase
            # of NaN makes Cheetah refuse to track at all — the beam energy stops
            # being a positive number and the episode dies with it.
            #
            # One such iteration is not a reason to abandon the polish, though:
            # dropping the bad direction and carrying on recovers, where stopping
            # leaves the plan at whatever CEM happened to find. Zero the offending
            # components, cap what is left, and keep going.
            if unit.grad is not None:
                unit.grad = torch.nan_to_num(unit.grad, nan=0.0, posinf=0.0, neginf=0.0)
                torch.nn.utils.clip_grad_norm_([unit], MAX_GRAD_NORM)
            optimiser.step()
            if not torch.isfinite(unit).all():
                # The knobs themselves have gone; nothing downstream can use them.
                with torch.no_grad():
                    unit.copy_(torch.tensor(best, dtype=DTYPE) / reach)
                break
        return np.clip(best, low, high), best_cost


def _pseudo_inverse(matrix: np.ndarray, cutoff: float) -> np.ndarray:
    """Truncated pseudo-inverse, keeping singular values above `cutoff` of the largest.

    The dropped directions are corrector combinations the BPMs barely respond to;
    inverting them would ask for huge kicks to achieve nothing, and on a real machine
    would simply saturate the supplies. `svd_correction.svd_pseudo_inverse` does the
    same for the same reason.
    """
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    keep = int(np.sum(singular > cutoff * singular[0])) if singular.size else 0
    if not keep:
        return np.zeros((matrix.shape[1], matrix.shape[0]))
    return vt[:keep].T @ np.diag(1 / singular[:keep]) @ u[:, :keep].T
