"""Ideal-model response correction for configurable screen-to-screen sections."""
from __future__ import annotations

from dataclasses import dataclass

import cheetah
import numpy as np
import torch
from scipy.optimize import lsq_linear

from .section_measurement import section_readout

from .beamline_tuning import (
    SegmentPlan,
    apply_knobs,
    beam_moments,
    knobs_for_elements,
    track_interval,
)


SCORE_INPUT_SCREEN = "cla_s02_dia_scr_01"
ELECTRON_REST_ENERGY = 0.51099895e6
KICKOFF_ENERGY = 193.156e6
KICKOFF_NORMALIZED_EMITTANCE = 1e-6


def make_section_plan(
    lattice,
    input_screen,
    output_screen,
    *,
    include_rf_phase=False,
):
    """Create one interval with all selected controls strictly between two boundaries."""
    names = [element.name for element in lattice.elements]
    if input_screen in {SCORE_INPUT_SCREEN, "cla_s02_sim_aper_01"} and input_screen not in names:
        start_index = -1
    else:
        try:
            start_index = names.index(input_screen)
        except ValueError as error:
            raise ValueError(f"Input boundary {input_screen!r} is not in the lattice") from error
    try:
        stop_index = names.index(output_screen)
    except ValueError as error:
        raise ValueError(f"Output boundary {output_screen!r} is not in the lattice") from error
    if stop_index <= start_index:
        raise ValueError("Output boundary must be downstream of the input boundary")
    knobs = knobs_for_elements(
        lattice.elements[start_index + 1:stop_index],
        include_rf_phase=include_rf_phase,
    )
    # This investigation intentionally compares only correctors and quadrupoles.
    knobs = [knob for knob in knobs if knob.kind in {"corrector", "quadrupole"}]
    if not knobs:
        raise ValueError("Selected interval has no corrector or quadrupole controls")
    return SegmentPlan(input_screen, output_screen, start_index, stop_index, knobs)


def track_to_boundary(lattice, beam, boundary):
    """Propagate an S02 input beam through the named boundary and return its output beam."""
    if boundary in {SCORE_INPUT_SCREEN, "cla_s02_sim_aper_01"} and boundary not in [e.name for e in lattice.elements]:
        return beam.clone()
    names = [element.name for element in lattice.elements]
    try:
        stop_index = names.index(boundary)
    except ValueError as error:
        raise ValueError(f"Boundary {boundary!r} is not in the lattice") from error
    upstream = cheetah.Segment(lattice.elements[:stop_index + 1])
    with torch.no_grad():
        return upstream.track(beam)


def ideal_kickoff_beam(num_particles=10_000):
    """Central kickoff S04 beam: beta=12 m, alpha=0, zero transverse centroid."""
    epsilon = KICKOFF_NORMALIZED_EMITTANCE / (
        KICKOFF_ENERGY / ELECTRON_REST_ENERGY
    )
    sigma = np.sqrt(epsilon * 12.0)
    sigma_p = np.sqrt(epsilon / 12.0)
    return cheetah.ParticleBeam.from_parameters(
        num_particles=int(num_particles),
        energy=torch.tensor(KICKOFF_ENERGY),
        mu_x=torch.tensor(0.0),
        mu_px=torch.tensor(0.0),
        mu_y=torch.tensor(0.0),
        mu_py=torch.tensor(0.0),
        sigma_x=torch.tensor(sigma),
        sigma_px=torch.tensor(sigma_p),
        cov_xpx=torch.tensor(0.0),
        sigma_y=torch.tensor(sigma),
        sigma_py=torch.tensor(sigma_p),
        cov_ypy=torch.tensor(0.0),
        sigma_tau=torch.tensor(1e-4),
        sigma_p=torch.tensor(1e-3),
        dtype=torch.float32,
    )


def transverse_state(beam):
    """Return simulated transverse centre and full 4x4 covariance for diagnostics."""
    particles = beam.particles[..., :4].detach().double()
    weights = beam.survival_probabilities.detach().double()
    live = weights > 0
    particles, weights = particles[live], weights[live]
    if len(weights) < 2 or not torch.isfinite(particles).all():
        return np.full(4, np.nan), np.full((4, 4), np.nan)
    center = (particles * weights[:, None]).sum(0) / weights.sum()
    centered = particles - center
    denominator = weights.sum() - weights.square().sum() / weights.sum()
    covariance = (centered.T * weights) @ centered / denominator
    return center.numpy(), covariance.numpy()


@dataclass(frozen=True)
class ResponseModel:
    target: np.ndarray
    scale: np.ndarray
    center: np.ndarray
    response: np.ndarray
    directions: np.ndarray
    singular_values: np.ndarray


def build_response_model(lattice, plan, ideal_input_beam, *, step=0.02, scale=None):
    """Linearize ideal exit-screen moments with respect to normalized section controls."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Response step must be finite and positive")
    center = np.array([knob.normalized_initial() for knob in plan.knobs], dtype=float)

    def predict(u):
        apply_knobs(lattice, plan.knobs, u)
        measured, _ = section_readout(track_interval(lattice, plan, ideal_input_beam))
        if not np.isfinite(measured).all():
            raise ValueError("Ideal beam is lost during response scan; shorten the section or reduce the step")
        return measured

    try:
        target = predict(center)
        if scale is None:
            scale = np.r_[np.full(2, 100e-6), np.maximum(target[2:], 100e-6)]
        scale = np.asarray(scale, dtype=float)
        if scale.shape != target.shape or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("scale must be finite, positive, and match the four outputs")
        response = np.empty((target.size, center.size), dtype=float)
        for index in range(center.size):
            low, high = center.copy(), center.copy()
            low[index] = max(0.0, center[index] - step)
            high[index] = min(1.0, center[index] + step)
            distance = high[index] - low[index]
            if distance <= 0:
                raise ValueError("Finite-difference direction has zero length")
            response[:, index] = (predict(high) - predict(low)) / distance / scale
        _, singular_values, directions = np.linalg.svd(response, full_matrices=True)
    finally:
        apply_knobs(lattice, plan.knobs, center)
    return ResponseModel(
        target=np.asarray(target, dtype=float),
        scale=scale,
        center=center,
        response=response,
        directions=directions,
        singular_values=singular_values,
    )


def _bounded_step(matrix, right_hand_side, lower, upper, ridge):
    """Solve one bounded, ridge-regularized least-squares response step."""
    columns = matrix.shape[1]
    augmented = np.vstack([matrix, np.sqrt(ridge) * np.eye(columns)])
    target = np.r_[right_hand_side, np.zeros(columns)]
    return lsq_linear(augmented, target, bounds=(lower, upper)).x


def response_candidate(
    observed,
    target,
    scale,
    u_start,
    response,
    knobs,
    *,
    mode="block",
    ridge=1e-4,
):
    """Return a bounded one-shot correction from a fixed ideal response matrix.

    ``mode='coupled'`` solves all four observables and all controls together.
    ``mode='block'`` first uses quadrupoles for shape, then correctors for orbit.
    """
    observed = np.asarray(observed, dtype=float)
    target = np.asarray(target, dtype=float)
    scale = np.asarray(scale, dtype=float)
    u_start = np.asarray(u_start, dtype=float)
    response = np.asarray(response, dtype=float)
    if observed.shape != (4,) or target.shape != (4,) or scale.shape != (4,):
        raise ValueError("Observed, target, and scale must be four-vectors")
    if response.shape != (4, len(knobs)) or u_start.shape != (len(knobs),):
        raise ValueError("Response and start do not match the knob list")
    if not np.isfinite(observed).all():
        return u_start.copy()
    residual = (observed - target) / scale
    if mode == "coupled":
        delta = _bounded_step(
            response,
            -residual,
            -u_start,
            1.0 - u_start,
            ridge,
        )
        return np.clip(u_start + delta, 0.0, 1.0)
    if mode != "block":
        raise ValueError("mode must be 'block' or 'coupled'")

    quadrupoles = np.array([
        index for index, knob in enumerate(knobs) if knob.kind == "quadrupole"
    ], dtype=int)
    correctors = np.array([
        index for index, knob in enumerate(knobs) if knob.kind == "corrector"
    ], dtype=int)
    if not len(quadrupoles) or not len(correctors):
        return response_candidate(
            observed,
            target,
            scale,
            u_start,
            response,
            knobs,
            mode="coupled",
            ridge=ridge,
        )

    # Shape first because quadrupoles can also move an off-axis orbit. Correctors then
    # remove the model-predicted orbit residual without disturbing first-order sizes.
    delta = np.zeros_like(u_start)
    shape_rows = np.array([2, 3])
    shape_step = _bounded_step(
        response[np.ix_(shape_rows, quadrupoles)],
        -residual[shape_rows],
        -u_start[quadrupoles],
        1.0 - u_start[quadrupoles],
        ridge,
    )
    delta[quadrupoles] = shape_step
    predicted_residual = residual + response @ delta
    orbit_rows = np.array([0, 1])
    orbit_step = _bounded_step(
        response[np.ix_(orbit_rows, correctors)],
        -predicted_residual[orbit_rows],
        -u_start[correctors],
        1.0 - u_start[correctors],
        ridge,
    )
    delta[correctors] = orbit_step
    return np.clip(u_start + delta, 0.0, 1.0)


def perturb_normalized_start(center, seed=0, fraction=0.04):
    """Create a reproducible starting setting in normalized knob coordinates."""
    rng = np.random.default_rng(seed)
    return np.clip(np.asarray(center) + rng.normal(0.0, fraction, len(center)), 0.0, 1.0)


def iterative_response_search(session, model, start, *, steps=None, learning_rate=0.1,
                              mode="block", ridge=1e-4):
    """Correct repeatedly with a fixed ideal-input model and measured exit residuals.

    One initial image plus one image after each damped update is charged. The actual
    input state is never used to build/refit the matrix. Return the best MEASURED
    setting; callers may restore it after the final step without another image.
    """
    available = session.remaining
    steps = available - 1 if steps is None else steps
    if not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if steps + 1 > available:
        raise ValueError("Response correction needs one initial image plus one per step")
    if not np.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError("learning_rate must be in (0, 1]")
    if mode not in {"block", "coupled"} or not np.isfinite(ridge) or ridge < 0:
        raise ValueError("Use block/coupled mode and a nonnegative finite ridge")
    u = np.asarray(start, dtype=float).copy()
    first_record = len(session.records)
    session.measure(u)
    session.records[-1]["response_step"] = 0
    for step_index in range(1, steps + 1):
        observed = session.records[-1]["observed"]
        candidate = response_candidate(
            observed, model.target, model.scale, u, model.response, session.plan.knobs,
            mode=mode, ridge=ridge,
        )
        u = np.clip(u + learning_rate * (candidate - u), 0.0, 1.0)
        session.measure(u)
        session.records[-1]["response_step"] = step_index
    return min(session.records[first_record:], key=lambda record: record["value"])["u"].copy()


def direction_intervals(start, directions):
    """Exact feasible scalar intervals for row directions in the unit cube."""
    start = np.asarray(start, dtype=float)
    intervals = []
    for row in np.asarray(directions, dtype=float):
        moving = row != 0
        if not moving.any():
            intervals.append((0., 0.))
            continue
        a, b = -start[moving] / row[moving], (1 - start[moving]) / row[moving]
        intervals.append((np.minimum(a, b).max(), np.maximum(a, b).min()))
    return np.asarray(intervals)


def model_search_directions(model, start, *, mode="boundary_aware", tolerance=1e-6):
    """Return unit ROW directions in normalized knob coordinates.

    boundary_aware recomputes the SVD of J restricted to free controls. These rows
    diagonalize the free-control Gauss-Newton matrix exactly (up to roundoff).
    Saturated controls use separate coordinate axes; they are not generally
    conjugate to the free rows under the full Hessian. The complete basis is
    orthonormal. Search order: observable free modes, saturated axes, free null modes.
    No conjugacy is claimed for the actual noisy/nonlinear L1 objective.

    raw_svd reproduces the original complete model basis; clean_svd removes small
    vector components for comparison only. tolerance applies to clean_svd, not to
    the new reduced SVD, whose vectors must not be clipped or projected afterward.
    """
    if mode == "raw_svd":
        return model.directions.copy()
    if mode not in {"clean_svd", "boundary_aware"}:
        raise ValueError("direction mode must be raw_svd, clean_svd, or boundary_aware")
    if mode == "clean_svd":
        directions = model.directions.copy()
        directions[np.abs(directions) < tolerance] = 0.
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        return directions
    start = np.asarray(start, dtype=float)
    response = np.asarray(model.response, dtype=float)
    if (start.ndim != 1 or response.ndim != 2 or response.shape[1] != len(start)
            or not np.isfinite(start).all() or not np.isfinite(response).all()
            or np.any(start < 0) or np.any(start > 1)):
        raise ValueError("Expected finite response columns matching unit-cube start")
    n = len(start)
    saturated = (start <= 1e-8) | (start >= 1-1e-8)
    free = np.flatnonzero(~saturated)
    axes = np.eye(n)[saturated]
    if not len(free):
        return axes
    _, singular_values, vh = np.linalg.svd(response[:, free], full_matrices=True)
    rank = int(np.count_nonzero(singular_values > max(
        singular_values.max(initial=0.) * 1e-8, 1e-10)))
    embedded = np.zeros((len(free), n))
    embedded[:, free] = vh
    return np.vstack((embedded[:rank], axes, embedded[rank:]))
