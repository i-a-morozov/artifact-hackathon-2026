"""Sequential screen-to-screen tuning helpers for the CLARA simulation.

The optimizer sees only the scalar objective from the selected downstream screen.
The full ParticleBeam is carried between segments by the simulator; it is not exposed
as a hardware measurement. A single screen image does not determine angle or the full
phase-space covariance.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import cheetah
import numpy as np
import torch

from .beam_tuning import BudgetExhausted, rcds_search

ROOT = Path(__file__).resolve().parents[1]
RANGES_FILE = ROOT / "data" / "clara" / "lattice_data" / "magnet_ranges.json"


def beam_moments(beam):
    """Return screen observables (mu_x, mu_y, sigma_x, sigma_y) in metres."""
    return np.array(
        [float(getattr(beam, name)) for name in ("mu_x", "mu_y", "sigma_x", "sigma_y")]
    )


def is_physical_screen(element):
    """Select interceptive diagnostic screens, excluding BPM and simulation markers."""
    return isinstance(element, cheetah.Screen) and (
        "_dia_scr_" in element.name or "_dcp_scr_" in element.name
    )


@dataclass(frozen=True)
class Knob:
    element_name: str
    property: str
    lower: float
    upper: float
    initial: float
    kind: str

    @property
    def label(self):
        return f"{self.element_name}.{self.property}"

    def physical(self, value):
        return self.lower + float(value) * (self.upper - self.lower)

    def normalized_initial(self):
        if self.upper <= self.lower:
            return 0.5
        return float(np.clip((self.initial - self.lower) / (self.upper - self.lower), 0, 1))


@dataclass
class SegmentPlan:
    input_screen: str
    output_screen: str
    start_index: int
    stop_index: int
    knobs: list[Knob]


def load_ranges(path=RANGES_FILE):
    return json.loads(Path(path).read_text())["elements"]


def knobs_for_elements(elements, ranges=None, include_rf_phase=True, rf_phase_span=10.0):
    """Build normalized controls with archiver bounds and an explicit RF phase span."""
    ranges = load_ranges() if ranges is None else ranges
    knobs = []
    for element in elements:
        specifications = []
        if isinstance(element, cheetah.Quadrupole):
            specifications = [("k1", 3.0, "quadrupole")]
        elif isinstance(element, cheetah.CombinedCorrector):
            specifications = [
                ("horizontal_angle", 5e-4, "corrector"),
                ("vertical_angle", 5e-4, "corrector"),
            ]
        elif include_rf_phase and isinstance(element, cheetah.Cavity):
            specifications = [("phase", rf_phase_span, "rf phase")]
        for prop, fallback, kind in specifications:
            current = float(getattr(element, prop))
            limits = ranges.get(element.name, {}).get(prop)
            if limits is None:
                lower, upper = current - fallback, current + fallback
            else:
                lower, upper = map(float, limits)
                # Reference set points can lie just outside archived extrema.
                lower, upper = min(lower, current), max(upper, current)
            if upper > lower:
                knobs.append(Knob(element.name, prop, lower, upper, current, kind))
    return knobs


def input_boundary_name(lattice):
    """Distinguish the updated S02-marker entrance from the older screen entrance."""
    names = {element.name for element in lattice.elements}
    return "cla_s02_sim_aper_01" if "cla_s02_dia_scr_01" in names else "cla_s02_dia_scr_01"


def make_plan(lattice, input_screen=None, output_screen=None,
              include_rf_phase=True):
    """Plan consecutive physical-screen intervals and controls strictly between them.

    The scorer lattice begins immediately after input_screen, so the first plan entry
    uses index -1 as a virtual upstream boundary.
    """
    input_screen = input_boundary_name(lattice) if input_screen is None else input_screen
    screen_indices = [i for i, element in enumerate(lattice.elements) if is_physical_screen(element)]
    if output_screen is not None:
        stop = next(i for i, e in enumerate(lattice.elements) if e.name == output_screen)
        screen_indices = [i for i in screen_indices if i <= stop]
    boundaries = [(-1, input_screen)] + [(i, lattice.elements[i].name) for i in screen_indices]
    plans = []
    for (start_index, start_name), (stop_index, stop_name) in zip(boundaries, boundaries[1:]):
        elements = lattice.elements[start_index + 1:stop_index]
        plans.append(
            SegmentPlan(
                start_name,
                stop_name,
                start_index,
                stop_index,
                knobs_for_elements(elements, include_rf_phase=include_rf_phase),
            )
        )
    return plans


def apply_knobs(lattice, knobs, u):
    u = np.asarray(u, dtype=float)
    if u.shape != (len(knobs),):
        raise ValueError(f"Expected {len(knobs)} normalized knob values, received {u.shape}")
    if np.any(~np.isfinite(u)) or np.any(u < -1e-9) or np.any(u > 1 + 1e-9):
        raise ValueError("Knobs must be finite and inside [0, 1]")
    for knob, value in zip(knobs, np.clip(u, 0, 1)):
        element = getattr(lattice, knob.element_name)
        setattr(element, knob.property, torch.tensor(knob.physical(value), dtype=torch.float32))


def track_interval(lattice, plan, beam):
    """Track from just after the input screen through the output screen."""
    interval = cheetah.Segment(lattice.elements[plan.start_index + 1:plan.stop_index + 1])
    with torch.no_grad():
        return interval.track(beam)


def collect_screen_moments(lattice, beam, input_screen=None, output_screen=None):
    """Track once and collect target or diagnostic moments at physical screens."""
    input_screen = input_boundary_name(lattice) if input_screen is None else input_screen
    result = {input_screen: beam_moments(beam)}
    current = beam.clone()
    with torch.no_grad():
        for element in lattice.elements:
            current = element.track(current)
            if is_physical_screen(element):
                result[element.name] = beam_moments(current)
                if output_screen is not None and element.name == output_screen:
                    break
    return result


class SegmentSession:
    """A strict downstream-screen measurement ledger for one interval."""

    def __init__(self, lattice, plan, input_beam, target, budget=24,
                 noise_fraction=0.0, seed=0):
        self.lattice = lattice
        self.plan = plan
        self.input_beam = input_beam
        self.target = np.asarray(target, dtype=float)
        self.budget = int(budget)
        self.noise_fraction = float(noise_fraction)
        self.rng = np.random.default_rng(seed)
        self.records = []
        # Orbit is judged on 100 um; shape is relative to the design spot.
        self.scale = np.r_[np.full(2, 100e-6), np.maximum(self.target[2:], 100e-6)]

    @property
    def remaining(self):
        return self.budget - len(self.records)

    @property
    def u0(self):
        return np.array([knob.normalized_initial() for knob in self.plan.knobs])

    def value(self, observed):
        if not np.isfinite(observed).all():
            return 10.0
        return float(np.mean(np.abs(observed - self.target) / self.scale))

    def measure(self, u):
        if self.remaining <= 0:
            raise BudgetExhausted("Segment screen-shot budget exhausted")
        apply_knobs(self.lattice, self.plan.knobs, u)
        output = track_interval(self.lattice, self.plan, self.input_beam)
        clean = beam_moments(output)
        sigma = self.noise_fraction * float(np.mean(clean[2:]))
        observed = clean + self.rng.normal(0.0, sigma, 4)
        observed[2:] = np.maximum(observed[2:], 1e-10)
        value = self.value(observed)
        error = max(sigma / float(np.mean(self.scale)), 1e-8)
        self.records.append(dict(u=np.asarray(u).copy(), observed=observed, clean=clean, value=value))
        return value, error

    def best(self):
        return min(self.records, key=lambda row: row["value"])


def tune_segment(session, method="rcds", seed=0, **rcds_options):
    """Tune one plan interval and return its propagated output beam."""
    if not session.plan.knobs:
        return track_interval(session.lattice, session.plan, session.input_beam), None, []
    _, history = rcds_search(
        session.measure,
        session.u0,
        session.remaining,
        method="gp" if method == "rcds_bo" else "parabola",
        seed=seed,
        **rcds_options,
    )
    best = session.best()["u"]
    apply_knobs(session.lattice, session.plan.knobs, best)
    return track_interval(session.lattice, session.plan, session.input_beam), best, history


def perturb_controls(lattice, seed=0, fraction=0.08, include_rf_phase=True):
    """Create reproducible hidden set-point errors inside each knob's allowed range."""
    rng = np.random.default_rng(seed)
    knobs = knobs_for_elements(lattice.elements, include_rf_phase=include_rf_phase)
    for knob in knobs:
        span = knob.upper - knob.lower
        value = np.clip(knob.initial + rng.normal(0.0, fraction * span), knob.lower, knob.upper)
        setattr(
            getattr(lattice, knob.element_name),
            knob.property,
            torch.tensor(float(value), dtype=torch.float32),
        )
    return knobs


def sequential_tune(lattice, beam, targets, plans, method="rcds", budget_per_segment=24,
                    noise_fraction=0.0, seed=0, **rcds_options):
    """Greedily correct each screen and carry the simulated output to the next interval."""
    current = beam.clone()
    rows = []
    sessions = []
    for index, plan in enumerate(plans):
        before = track_interval(lattice, plan, current)
        initial = beam_moments(before)
        if plan.knobs:
            session = SegmentSession(
                lattice, plan, current, targets[plan.output_screen],
                budget=budget_per_segment, noise_fraction=noise_fraction, seed=seed + index,
            )
            current, _, _ = tune_segment(
                session, method=method, seed=seed + index, **rcds_options
            )
            final = beam_moments(current)
            rows.append(dict(
                segment=index + 1,
                input_screen=plan.input_screen,
                output_screen=plan.output_screen,
                knobs=len(plan.knobs),
                evaluations=len(session.records),
                initial_truth=session.value(initial),
                best_observed=session.best()["value"],
                final_truth=session.value(final),
            ))
            sessions.append(session)
        else:
            current = before
            scale = np.r_[np.full(2, 100e-6), np.maximum(targets[plan.output_screen][2:], 100e-6)]
            value = float(np.mean(np.abs(initial - targets[plan.output_screen]) / scale))
            rows.append(dict(
                segment=index + 1, input_screen=plan.input_screen,
                output_screen=plan.output_screen, knobs=0, evaluations=0,
                initial_truth=value, best_observed=value, final_truth=value,
            ))
    return lattice, current, rows, sessions
