"""Local section metric, survival-aware ideal readout, and joint-knob BO.

This is intentionally distinct from the official whole-lattice FEH camera score.
The ideal readout uses surviving particle moments, without a finite camera field of view.
"""
from __future__ import annotations

import numpy as np

from .beam_tuning import BudgetExhausted
from .beamline_tuning import SegmentSession, apply_knobs, beam_moments, track_interval

FAILURE_VALUE = 1e6
MIN_SURVIVAL = 0.10


def section_value(observed, target, scale, part="all"):
    observed = np.asarray(observed, dtype=float)
    if not np.isfinite(observed).all():
        return FAILURE_VALUE
    residual = np.abs((observed - target) / scale)
    if part == "orbit":
        residual = residual[:2]
    elif part == "shape":
        residual = residual[2:]
    elif part != "all":
        raise ValueError("part must be all, orbit, or shape")
    return float(min(residual.mean(), FAILURE_VALUE - 1))


def section_readout(beam):
    """Return a missing image below 10% survival; do not reward nearly lost beams."""
    survival = float(beam.survival_probabilities.detach().mean())
    if not np.isfinite(survival) or survival < MIN_SURVIVAL:
        return np.full(4, np.nan), survival
    return beam_moments(beam), survival


class SectionSession(SegmentSession):
    """One counted ideal screen image per call, with finite failed-reading handling."""

    def __init__(self, *args, scale=None, tracker=None, progress=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = tracker
        self.progress = progress
        if scale is not None:
            self.scale = np.asarray(scale, dtype=float).copy()
        if self.scale.shape != (4,) or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("scale must be four positive finite values")
        if not np.isfinite(self.target).all() or self.noise_fraction < 0:
            raise ValueError("target must be finite and readout noise nonnegative")

    def value(self, observed):
        return section_value(observed, self.target, self.scale)

    def measure(self, u):
        if self.remaining <= 0:
            raise BudgetExhausted("Section screen-shot budget exhausted")
        apply_knobs(self.lattice, self.plan.knobs, u)
        output = (self.tracker(self.input_beam) if self.tracker is not None else
                  track_interval(self.lattice, self.plan, self.input_beam))
        clean, survival = section_readout(output)
        valid = bool(np.isfinite(clean).all())
        sigma = self.noise_fraction * float(clean[2:].mean()) if valid else 0.0
        observed = clean + self.rng.normal(0.0, sigma, 4)
        if valid:
            observed[2:] = np.maximum(observed[2:], 1e-10)
        # Propagated standard error for mean absolute normalized residual, away from cusps.
        error = max(sigma * float(np.linalg.norm(1 / self.scale)) / 4, 1e-8)
        value = self.value(observed)
        self.records.append(dict(u=np.asarray(u).copy(), observed=observed,
                                 clean=clean, value=value, error=error,
                                 survival=survival, valid=valid))
        if self.progress is not None:
            self.progress(self)
        return value, error


def full_bo_search(session, start, *, seed=0, acquisition="ei_then_ucb",
                   switch_at=None, beta=1.0):
    """Joint GP search: EI before switch_at, confidence bound afterwards.

    switch_at is a zero-based measurement index including the start and random
    initialization; None uses budget // 2. Initialization still takes precedence.
    The minimization equivalent of BoTorch UCB(maximize=False) is skopt LCB with
    kappa=sqrt(beta). Pure "ei" and "ucb" are available for comparison.
    """
    from skopt import Optimizer
    from skopt.space import Real

    budget = session.remaining
    if budget < 1:
        raise ValueError("Full BO requires at least one available measurement")
    if acquisition not in {"ei_then_ucb", "ei", "ucb"}:
        raise ValueError("acquisition must be ei_then_ucb, ei, or ucb")
    switch_at = budget // 2 if switch_at is None else switch_at
    if not isinstance(switch_at, (int, np.integer)) or not 0 <= switch_at <= budget:
        raise ValueError("switch_at must be an integer between zero and the budget")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    initial_points = min(max(2, len(start) + 1), max(1, budget // 3))

    def phase(index):
        if acquisition == "ei":
            return "EI"
        if acquisition == "ucb":
            return "LCB"
        return "EI" if index < switch_at else "LCB"

    def make_optimizer(name, random_state=seed):
        return Optimizer(
            [Real(0.0, 1.0) for _ in start], base_estimator="GP",
            n_initial_points=initial_points, acq_func=name, random_state=random_state,
            acq_func_kwargs={"xi": 0.0, "kappa": float(np.sqrt(beta))},
            acq_optimizer="sampling", acq_optimizer_kwargs={"n_points": 1024},
        )

    current_phase = phase(0)
    optimizer = make_optimizer(current_phase)
    for index in range(budget):
        next_phase = phase(index)
        if next_phase != current_phase:
            # tell() precomputes the next ask(). Rebuild through the public API and
            # replay all observations so the boundary candidate uses the NEW AF.
            previous_x, previous_y = list(optimizer.Xi), list(optimizer.yi)
            # Continue the RNG stream; restarting at seed repeats candidate pools.
            optimizer = make_optimizer(next_phase, random_state=optimizer.rng)
            if previous_x:
                optimizer.tell(previous_x, previous_y)
            current_phase = next_phase
        candidate = list(start) if index == 0 else optimizer.ask()
        value, _ = session.measure(candidate)
        session.records[-1]["acquisition"] = (
            "start" if index == 0 else "random" if index < initial_points else current_phase
        )
        optimizer.tell(candidate, value)
    return session.best()["u"].copy()
