"""Screen-by-screen correction, with overlapping intervals and a fresh upstream state."""
from copy import deepcopy
from time import perf_counter
import numpy as np
import torch

from .beamline_tuning import (SegmentPlan, apply_knobs, is_physical_screen,
                              knobs_for_elements, track_interval, input_boundary_name)
from .beam_tuning import rcds_search
from .section_measurement import SectionSession, full_bo_search, section_readout, section_value
from .section_response import (SCORE_INPUT_SCREEN, build_response_model,
                               iterative_response_search, model_search_directions,
                               track_to_boundary)

METHODS = ('response matrix', 'full BO', 'RCDS', 'RCDS+BO',
           'RCDS + response directions', 'RCDS+BO + response directions')


def plan_screen_sequence(lattice, *, min_quadrupoles=4, min_correctors=4,
                         final_screen=None, max_sections=None, input_screen=None):
    """For each physical screen choose the nearest earlier boundary with enough knobs.

    Corrector count means independently adjustable angles (H and V), not devices.
    If even the root interval has too few knobs, use what exists and report it.
    """
    screens = [(i, e.name) for i, e in enumerate(lattice.elements) if is_physical_screen(e)]
    if final_screen is not None:
        names = [name for _, name in screens]
        if final_screen not in names:
            raise ValueError(f'Unknown physical final screen: {final_screen}')
        screens = screens[:names.index(final_screen)+1]
    if max_sections is not None:
        if not isinstance(max_sections, int) or max_sections < 1:
            raise ValueError('max_sections must be positive or None')
        screens = screens[:max_sections]
    input_screen = input_boundary_name(lattice) if input_screen is None else input_screen
    previous = [(-1, input_screen)]
    plans = []
    for stop, name in screens:
        for start, input_name in reversed(previous):
            knobs = knobs_for_elements(lattice.elements[start+1:stop], include_rf_phase=False)
            nq = sum(k.kind == 'quadrupole' for k in knobs)
            nc = sum(k.kind == 'corrector' for k in knobs)
            if nq >= min_quadrupoles and nc >= min_correctors:
                break
        plans.append(SegmentPlan(input_name, name, start, stop, knobs))
        previous.append((stop, name))
    return plans


def optimize_section(session, model, start, method, *, seed=0, learning_rate=.1,
                     response_steps=None, response_mode='block', ridge=1e-4,
                     direction_mode='boundary_aware', bo_options=None, rcds_bo_options=None):
    """Shared dispatch for independent sections and the sequential machine correction."""
    if method not in METHODS:
        raise ValueError(f'Choose a method from {METHODS}')
    if not len(start):
        session.measure(start)
    elif method == 'response matrix':
        iterative_response_search(session, model, start, steps=response_steps,
                                  learning_rate=learning_rate, mode=response_mode, ridge=ridge)
    elif method == 'full BO':
        full_bo_search(session, start, seed=seed, **(bo_options or {}))
    else:
        directions = (model_search_directions(model, start, mode=direction_mode)
                      if 'response directions' in method else np.eye(len(start)))
        rcds_search(session.measure, start, session.remaining,
                    method='gp' if 'BO' in method else 'parabola', seed=seed,
                    directions=directions, **((rcds_bo_options or {}) if 'BO' in method else {}))
    selected = session.best()['u'].copy()
    apply_knobs(session.lattice, session.plan.knobs, selected)
    return selected


def transmission(output, incoming):
    """Surviving particle-weight ratio; NaN when no beam entered (not zero loss)."""
    denominator = float(incoming.survival_probabilities.sum())
    return float(output.survival_probabilities.sum()) / denominator if denominator > 0 else np.nan


def correct_screen_sequence(machine, reference, actual_root, ideal_root, plans, *,
                            method='response matrix', budget=128, noise_fraction=.05,
                            seed=0, response_step=.025, verbose=False, log_every=16, section_callback=None, **method_options):
    """Modify one machine in place; optimizer receives exit images only.

    Upstream particle states are simulator internals, recomputed from the root after
    every committed correction. No measured initial-state covariance is supplied.
    Clean audits before/after each interval and at the end are diagnostics, not
    optimization inputs or official scorer measurements.
    """
    if method not in METHODS:
        raise ValueError(f'Unsupported method: {method}')
    if not isinstance(log_every, int) or log_every < 1:
        raise ValueError("log_every must be a positive integer")
    plans = list(plans)
    started = perf_counter()
    def log(message):
        if verbose:
            print(message, flush=True)
    log(f"Full correction: {method}; {len(plans)} sections; up to {budget} images per section")
    before = deepcopy(machine)
    records, sessions, models = [], [], []
    offline_response_tracks = 0
    for index, plan in enumerate(plans):
        section_started = perf_counter()
        prefix = f"[{index+1}/{len(plans)}] {plan.input_screen} -> {plan.output_screen}"
        log(f"{prefix}: {len(plan.knobs)} knobs; building ideal response")
        # Recompute after upstream changes, including for skipped ideal targets.
        actual_input = track_to_boundary(machine, actual_root, plan.input_screen)
        initial_output = track_interval(machine, plan, actual_input)
        initial = section_readout(initial_output)[0]
        transmissions = dict(
            input_transmission=transmission(actual_input, actual_root),
            initial_transmission=transmission(initial_output, actual_root),
            initial_section_transmission=transmission(initial_output, actual_input),
        )
        ideal_input = track_to_boundary(reference, ideal_root, plan.input_screen)
        ideal_exit, ideal_survival = section_readout(track_interval(reference, plan, ideal_input))
        offline_response_tracks += 1
        model = None
        reason = "ideal beam below survival threshold"
        if np.isfinite(ideal_exit).all():
            # A scan may hit an aperture even when the unperturbed target is valid.
            for shrink in (1., .25, .0625):
                try:
                    model = build_response_model(reference, plan, ideal_input, step=response_step*shrink)
                    offline_response_tracks += 1+2*len(plan.knobs)
                    break
                except ValueError as error:
                    if "Ideal beam is lost" not in str(error):
                        raise
                    reason = str(error)
                    # A failed scan stops early; report its count separately as an upper bound.
                    offline_response_tracks += 1+2*len(plan.knobs)
        if model is None:
            records.append(dict(input_screen=plan.input_screen, output_screen=plan.output_screen,
                                method=method, knobs=len(plan.knobs), online_measurements=0,
                                initial_objective=np.nan, corrected_objective=np.nan,
                                selected_evaluation=np.nan, best_measured_objective=np.nan,
                                survival=float(initial_output.survival_probabilities.mean()),
                                status="skipped: " + reason, ideal_survival=ideal_survival,
                                **transmissions,
                                corrected_transmission=transmissions['initial_transmission'],
                                corrected_section_transmission=transmissions['initial_section_transmission'],
                                elapsed_seconds=perf_counter()-section_started))
            log(f"{prefix}: SKIPPED ({reason}); exit transmission "
                f"{transmissions['initial_transmission']:.1%}; 0 images")
            models.append(None)
            if section_callback is not None:
                section_callback(dict(records[-1]), None)
            continue
        # Never reuse the output of a previous interval: this plan can overlap it.
        start = np.array([(float(getattr(getattr(machine, k.element_name), k.property))-k.lower)
                          / (k.upper-k.lower) for k in plan.knobs])
        start = np.clip(start, 0, 1)
        def progress(session, label=prefix):
            count = len(session.records)
            if count == 1 or count % log_every == 0 or session.remaining == 0:
                last = session.records[-1]
                root_survival = float(actual_root.survival_probabilities.mean())
                cumulative = last['survival']/root_survival if root_survival > 0 else np.nan
                log(f"  {label}: image {count}/{budget}, objective {last['value']:.4g}, "
                    f"best measured {session.best()['value']:.4g}, exit transmission {cumulative:.1%}")
        session = SectionSession(machine, plan, actual_input, model.target, scale=model.scale,
                                 budget=budget, noise_fraction=noise_fraction, seed=seed+index,
                                 progress=progress if verbose else None)
        optimize_section(session, model, start, method, seed=seed+index, **method_options)
        final_output = track_interval(machine, plan, actual_input)
        final, survival = section_readout(final_output)
        records.append(dict(input_screen=plan.input_screen, output_screen=plan.output_screen,
                            method=method, knobs=len(plan.knobs), online_measurements=len(session.records),
                            initial_objective=section_value(initial, model.target, model.scale),
                            corrected_objective=section_value(final, model.target, model.scale),
                            survival=survival, status=("no usable exit image" if not np.isfinite(final).all() else
                                                     "corrected" if len(plan.knobs) else "observed: no controls"),
                            ideal_survival=ideal_survival, **transmissions,
                            selected_evaluation=next(i+1 for i,r in enumerate(session.records) if r is session.best()),
                            best_measured_objective=session.best()["value"],
                            corrected_transmission=transmission(final_output, actual_root),
                            corrected_section_transmission=transmission(final_output, actual_input),
                            elapsed_seconds=perf_counter()-section_started))
        row = records[-1]
        log(f"{prefix}: {row['status']}; clean objective {row['initial_objective']:.4g} -> "
            f"{row['corrected_objective']:.4g}; exit transmission {row['initial_transmission']:.1%} -> "
            f"{row['corrected_transmission']:.1%}; within-section {row['corrected_section_transmission']:.1%}; "
            f"{row['online_measurements']} images; {row['elapsed_seconds']:.1f} s")
        if section_callback is not None:
            section_callback(dict(row), session)
        sessions.append(session)
        models.append(model)
    # Later overlapping windows may spoil previously tuned screens: explicitly audit all.
    audit = []
    final_beams = {SCORE_INPUT_SCREEN: actual_root}
    if plans:
        final_beams[plans[0].input_screen] = actual_root
    log("Final audit: rechecking all exit screens after overlapping corrections")
    for plan, model in zip(plans, models):
        initial_output = track_to_boundary(before, actual_root, plan.output_screen)
        initial = section_readout(initial_output)[0]
        final_output = track_to_boundary(machine, actual_root, plan.output_screen)
        final, survival = section_readout(final_output)
        final_beams[plan.output_screen] = final_output
        if plan.input_screen not in final_beams:
            final_beams[plan.input_screen] = track_to_boundary(machine, actual_root, plan.input_screen)
        trans = dict(initial_transmission=transmission(initial_output, actual_root),
                     final_transmission=transmission(final_output, actual_root),
                     final_section_transmission=transmission(final_output, final_beams[plan.input_screen]))
        log(f"  {plan.output_screen}: final exit transmission {trans['final_transmission']:.1%}; "
            f"within-section {trans['final_section_transmission']:.1%}")
        if model is None:
            audit.append(dict(screen=plan.output_screen, initial_objective=np.nan,
                              final_objective=np.nan, survival=survival, **trans, status="no valid ideal response"))
            continue
        row = dict(screen=plan.output_screen, status="audited", initial_objective=section_value(initial, model.target, model.scale),
                   final_objective=section_value(final, model.target, model.scale), survival=survival, **trans)
        for j, label in enumerate(('mu_x', 'mu_y', 'sigma_x', 'sigma_y')):
            row[f'target_{label}_um'] = model.target[j]*1e6
            row[f'initial_{label}_um'] = initial[j]*1e6
            row[f'final_{label}_um'] = final[j]*1e6
        audit.append(row)
    # Carry through remaining elements even when the user limits the corrected screens.
    import cheetah
    with torch.no_grad():
        output = cheetah.Segment(machine.elements).track(actual_root)
    total_images = sum(len(s.records) for s in sessions)
    elapsed = perf_counter()-started
    end_transmission = transmission(output, actual_root)
    log(f"Finished: {total_images} images; {elapsed:.1f} s; end-of-lattice transmission {end_transmission:.1%}")
    return dict(lattice=machine, initial_lattice=before, output_beam=output, records=records, audit=audit,
                elapsed_seconds=elapsed, final_transmission=end_transmission,
                sessions=sessions, models=models,
                online_measurements=sum(len(s.records) for s in sessions),
                offline_response_tracks_upper_bound=offline_response_tracks)
