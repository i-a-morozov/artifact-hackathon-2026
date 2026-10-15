"""Build section.ipynb from reviewable cell sources."""
from pathlib import Path
import nbformat as nbf
ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
def md(source, cell_id):
    return nbf.v4.new_markdown_cell(source, id=cell_id)
def code(source, cell_id):
    return nbf.v4.new_code_cell(source, id=cell_id)
cells = [
    md(r'''# Section correction

1. Response-matrix correction
2. RCDS
3. RCDS+BO
4. RCDS (response-matrix directions)
5. RCDS+BO (response-matrix directions)
6. BO
7. RL (PPO)''', 'section-title'),

    md(r'''## Import''', 'section-guide'),

    code(r'''from copy import deepcopy
from pathlib import Path
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from threadpoolctl import threadpool_limits

get_ipython().run_line_magic("matplotlib", "inline")
HERE = Path.cwd().resolve()
ROOT = next((p for p in (HERE, *HERE.parents) if (p / "work" / "section_response.py").exists()), None)
if ROOT is None:
    raise RuntimeError("Run this notebook from the optimisation project or its work directory")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score import sample_beam as sample_score_beam
from work.nominal_machine import load_lattice as load_model_lattice, load_beam as load_nominal_beam
from work.beam_tuning import rcds_search, sample_beam as sample_kickoff_beam
from work.beamline_tuning import (
    apply_knobs, 
    beam_moments,
    track_interval,
)
from work.section_response import (
    build_response_model,
    model_search_directions, direction_intervals,
    ideal_kickoff_beam,
    make_section_plan,
    perturb_normalized_start,
    iterative_response_search,
    track_to_boundary,
    transverse_state,
)
from work.section_measurement import (
    SectionSession, section_readout, section_value as local_value, full_bo_search,
)
from work.rl_tuning import ActorCritic, PPOConfig, train_ppo
from work.section_rl import (
    SectionRLConfig,
    SectionRLEnv,
    run_section_policy,
)

SEED = 101
FAST_RUN = os.environ.get("OPTIMISATION_SMOKE") == "1"
N_BEAMS = 2 if FAST_RUN else 16
NUM_PARTICLES = 750 if FAST_RUN else 10_0000
MEASUREMENT_BUDGET = 6 if FAST_RUN else 128
READOUT_NOISE = 0.025
START_ERROR_FRACTION = 0.05
RESPONSE_STEP = 0.025
RESPONSE_RIDGE = 1e-4
RESPONSE_LEARNING_RATE = 0.1
RESPONSE_STEPS = MEASUREMENT_BUDGET - 1  # one initial image, then one per update
RESPONSE_SOLVE_MODE = "block"
SCORE_BEAM_SEVERITY = 0.5

INITIAL_SCREEN = "cla_s04_sim_aper_01"
FINAL_SCREEN = "cla_s04_dia_scr_03"
PRIMARY_BEAM_SOURCE = "auto"
SECTION_MODEL = "nominal"  # "gpsr": scan settings; "scorer": legacy benchmark optics
SECTION_DETAILED_APERTURES = True
SECTION_BEAM_SEVERITY = SCORE_BEAM_SEVERITY


def load_section_lattice():
    # Independent fresh reference/plant copies; plant setting errors are applied via start.
    return load_model_lattice(model=SECTION_MODEL, errors=False,
                              apertures=SECTION_DETAILED_APERTURES, end=None)


def sample_section_input(configuration, reference_lattice, seed, num_particles, severity):
    source = configuration["beam_source"]
    if source == "kickoff":
        if severity == 0:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                return ideal_kickoff_beam(num_particles)
        return sample_kickoff_beam(seed, num_particles)
    if source == "nominal":
        root = load_nominal_beam(num_particles, seed=seed, severity=severity)
    elif source == "score":
        root = sample_score_beam(np.random.default_rng(seed), num_particles=num_particles, severity=severity)
    else:
        raise ValueError(f"Unknown beam source: {source}")
    return track_to_boundary(reference_lattice, root, configuration["input_screen"])


ENABLE_EXTRA_SECTION = os.environ.get("OPTIMISATION_EXTRA_SECTION") == "1"
EXTRA_SECTION_CONFIGS = [
    dict(
        name="S06 to S07",
        input_screen="cla_s06_dia_scr_02",
        output_screen="cla_s07_dia_scr_01",
        beam_source="auto",
    ),
] if ENABLE_EXTRA_SECTION else []
SECTION_CONFIGS = [
    dict(
        name=f"{INITIAL_SCREEN} to {FINAL_SCREEN}",
        input_screen=INITIAL_SCREEN,
        output_screen=FINAL_SCREEN,
        beam_source=PRIMARY_BEAM_SOURCE,
    ),
    *EXTRA_SECTION_CONFIGS,
]

METHODS = (
    "response matrix",
    "full BO",
    "RCDS",
    "RCDS+BO",
    "RCDS + response directions",
    "RCDS+BO + response directions",
)
FULL_BO_ACQUISITION = "ei_then_ucb"  # alternatives: "ei", "ucb"
FULL_BO_SWITCH_AT = None  # zero-based image index; None -> MEASUREMENT_BUDGET // 2
BO_UCB_BETA = 1.0
RCDS_BO_OPTIONS = dict(no_ucb=16, no_ei=16, ei_first=True, nr=4, rs=32, beta=BO_UCB_BETA, use_parabola=False)


# RL is trained independently for every section because the number of knobs can differ.
RL_SIMULATION_BEAMS = 2 if FAST_RUN else 2
RL_PRETRAIN_ITERATIONS = 1 if FAST_RUN else 128
RL_PRETRAIN_EPISODES = 2 if FAST_RUN else 16
RL_ADAPT_ITERATIONS = 1 if FAST_RUN else 8
RL_ADAPT_EPISODES = 1
RL_HORIZON = 2 if FAST_RUN else 8
RL_MAX_KNOB_STEP = 0.12
RL_PHYSICS_WEIGHT = 0.7  # zero recovers ordinary PPO
RL_HIDDEN = 32 if FAST_RUN else 64
assert N_BEAMS >= 2, "Need an adaptation beam plus at least one held-out beam"
assert RL_HORIZON + 1 <= MEASUREMENT_BUDGET
assert 0 <= RESPONSE_STEPS < MEASUREMENT_BUDGET
assert len({c["name"] for c in SECTION_CONFIGS}) == len(SECTION_CONFIGS)
for configuration in SECTION_CONFIGS:
    if configuration["beam_source"] == "auto":
        configuration["beam_source"] = (
            "score" if SECTION_MODEL == "scorer" else "nominal"
        )
    if configuration["beam_source"] == "kickoff" and configuration["input_screen"] != "cla_s04_sim_aper_01":
        raise ValueError("Kickoff beam is defined only at the S04 entrance; use score or auto")

torch.set_num_threads(1)
thread_limits = threadpool_limits(limits=1)
np.random.seed(SEED)
torch.manual_seed(SEED)

print({
    "sections": [case["name"] for case in SECTION_CONFIGS],
    "section lattice": SECTION_MODEL,
    "detailed apertures": SECTION_DETAILED_APERTURES if SECTION_MODEL != "scorer" else "scorer apertures",
    "input beam sources": [case["beam_source"] for case in SECTION_CONFIGS],
    "beams per section": N_BEAMS,
    "online RCDS budget": MEASUREMENT_BUDGET,
    "readout noise": READOUT_NOISE,
    "response solve": RESPONSE_SOLVE_MODE,
    "response learning rate": RESPONSE_LEARNING_RATE,
    "response updates": RESPONSE_STEPS,
    "RL pretrain iterations": RL_PRETRAIN_ITERATIONS,
    "RL one-beam adaptation iterations": RL_ADAPT_ITERATIONS,
    "RL deployment measurements": RL_HORIZON + 1,
    "physics action weight": RL_PHYSICS_WEIGHT,
})

# raw_svd reproduces the old basis; clean_svd only removes tiny cross-components.
DIRECTION_MODE = "boundary_aware"
from work.section_sequence import plan_screen_sequence, optimize_section, correct_screen_sequence
from work.beamline_tuning import is_physical_screen, knobs_for_elements

# Simulation pretraining resolution; adaptation and evaluation keep NUM_PARTICLES.
RL_TRAIN_PARTICLES = min(NUM_PARTICLES, 4096)
RL_FAST_TRACKING = True  # False restores the original screen-copy tracking path.
RL_PROGRESS_EVERY = 8
''', 'section-setup'),

    md(r'''## 1. New nominal lattice, input beam, target, and section controls

`SECTION_MODEL="nominal"` loads the new nominal-model lattice-file strengths directly, with
`data/clara/lattice_data/apertures.json` enabled. `"gpsr"` selects the reconstruction scan tune;
`"scorer"` explicitly selects the older benchmark loader and its own apertures. This setting
is shared by classical tuning, response-model construction, diagnostics, PPO pretraining,
one-beam adaptation and held-out transfer. The final full-machine experiment has its own
`FULL_MODEL` setting.

The default `beam_source="auto"` now means the measured nominal 6D distribution for nominal/GPSR
optics, including at S04. The ideal root beam is sampled from `nominal_model/initial_distribution.npz`
at the S02 marker and propagated through the chosen reference lattice to the selected entrance.
Varied beams use the same distribution with explicit affine jitter controlled by
`SECTION_BEAM_SEVERITY`; full cross-coordinate correlations are retained. The target and response
matrix use the ideal beam and unperturbed optics. Actual correction starts at perturbed normalized
magnet settings. This isolated comparison does not add hidden magnet misalignments.

Explicit `beam_source="kickoff"` remains available only at the S04 entrance for legacy experiments;
`"score"` selects the scorer input distribution. Neither is used by default with the new lattice.
All quadrupole and corrector controls strictly between the named boundaries are normalized to [0,1].
Only exit-screen position and projected sizes enter the objective; simulated input covariance is
shown for diagnostics, not supplied as a measured state to the tuner.
''', 'section-ideal-explanation'),

    code(r'''def make_section_beams(configuration, reference_lattice):
    ideal = sample_section_input(configuration, reference_lattice, SEED, NUM_PARTICLES, 0.)
    beams = [sample_section_input(configuration, reference_lattice, SEED+100+index,
                                  NUM_PARTICLES, SECTION_BEAM_SEVERITY)
             for index in range(N_BEAMS)]
    return ideal, beams


section_models = {}
plan_rows, input_state_rows = [], []
for section_index, configuration in enumerate(SECTION_CONFIGS):
    reference_lattice = load_section_lattice()
    plan = make_section_plan(
        reference_lattice,
        configuration["input_screen"],
        configuration["output_screen"],
        include_rf_phase=False,
    )
    ideal_input, test_beams = make_section_beams(configuration, reference_lattice)
    model = build_response_model(
        reference_lattice,
        plan,
        ideal_input,
        step=RESPONSE_STEP,
    )
    start = perturb_normalized_start(
        model.center,
        seed=SEED + section_index,
        fraction=START_ERROR_FRACTION,
    )
    centre, covariance = transverse_state(ideal_input)
    counts = pd.Series([knob.kind for knob in plan.knobs]).value_counts()
    plan_rows.append(dict(
        section=configuration["name"],
        input_screen=plan.input_screen,
        output_screen=plan.output_screen,
        beam_source=configuration["beam_source"],
        knobs=len(plan.knobs),
        quadrupoles=int(counts.get("quadrupole", 0)),
        correctors=int(counts.get("corrector", 0)),
        offline_response_tracks=1 + 2 * len(plan.knobs),
    ))
    input_state_rows.append(dict(
        section=configuration["name"],
        mu_x_um=centre[0] * 1e6,
        mu_px_urad=centre[1] * 1e6,
        mu_y_um=centre[2] * 1e6,
        mu_py_urad=centre[3] * 1e6,
        sigma_x_um=np.sqrt(covariance[0, 0]) * 1e6,
        sigma_px_urad=np.sqrt(covariance[1, 1]) * 1e6,
        sigma_y_um=np.sqrt(covariance[2, 2]) * 1e6,
        sigma_py_urad=np.sqrt(covariance[3, 3]) * 1e6,
        cov_xpx=covariance[0, 1],
        cov_ypy=covariance[2, 3],
    ))
    section_models[configuration["name"]] = dict(
        configuration=configuration,
        plan=plan,
        model=model,
        start=start,
        beams=test_beams,
        ideal_input=ideal_input,
    )

display(pd.DataFrame(plan_rows))
print("Ideal input states reconstructed from particles:")
display(pd.DataFrame(input_state_rows))

for name, item in section_models.items():
    print(f"\n{name} target [mu_x, mu_y, sigma_x, sigma_y] in um:")
    print(np.round(item["model"].target * 1e6, 3))
    display(pd.DataFrame({
        "knob": [knob.label for knob in item["plan"].knobs],
        "kind": [knob.kind for knob in item["plan"].knobs],
        "reference_u": item["model"].center,
        "start_u": item["start"],
    }))''', 'section-build-models'),

    code(r'''screen_lattice = load_section_lattice()
screen_rows = [dict(index=-1, name="cla_s02_dia_scr_01", kind="virtual scorer input")]
for index, element in enumerate(screen_lattice.elements):
    if isinstance(element, __import__("cheetah").Screen):
        screen_rows.append(dict(index=index, name=element.name,
                                kind="physical screen" if is_physical_screen(element) else "simulation marker / other"))
display(pd.DataFrame(screen_rows))''', '8f7be79f-9075-4503-947c-7a21115ce958'),

    md(r'''## 2. Response matrices and physics directions

For normalized controls u and scaled output y/s, central finite differences form

J_ki = d(y_k/s_k) / d u_i.

Orbit is scaled by 100 micrometres. Shape is scaled by its ideal output size, with a
100 micrometre floor. The notebook displays the full matrix and the physically motivated blocks:
correctors to orbit and quadrupoles to shape. The block response correction solves shape first
with quadrupoles and then orbit with correctors. The coupled option solves all entries together.

The rows of V-transpose from a full SVD of J are used as initial RCDS directions. They are the
eigenvectors of the local Gauss-Newton matrix J-transpose J. This is a response-aligned,
orthonormal basis; it is not guaranteed to remain conjugate once nonlinearities or model error
become large. Also, the optimized L1 objective has no fixed quadratic Hessian, so these
are model-derived search directions, not guaranteed conjugate directions for this objective.
Full SVD retains null-space directions so RCDS still receives one direction per
control.

### Why the previous direction basis performed poorly

The orientation is correct: NumPy returns Vh with one direction per row, and RCDS uses rows.
In the saved S04 run, two knobs start at exactly 1. Tiny SVD cross-components make three
rows completely blocked; they also constrain the sign of other useful rows. The raw matrix
has rank 4 for 8 knobs and a nonzero-singular-value condition ratio of about 723. Four null
modes cannot change the ideal observables to first order. With 16 EI + 16 UCB proposals per
line, a 128-image budget explores only a few lines. A basis can therefore be valid algebraically
and still be a poor search schedule.

`DIRECTION_MODE="boundary_aware"` removes components smaller than 1e-6, separates saturated knobs into coordinate axes, keeps feasible
observable directions in the remaining controls first, and completes a full-rank basis with independent coordinate axes.
`clean_svd` keeps the cleaned SVD including its null modes; `raw_svd` reproduces the old result.
This is an initial search heuristic, not guaranteed conjugacy or guaranteed improvement.
Bounds can become active again later. The model is local to the ideal beam; quadrupole transport
and the absolute-error objective are not globally quadratic. Reduce `no_ei`/`no_ucb` if you want
more direction sweeps within the same image budget. The table below exposes feasible intervals.
''', 'section-response-explanation'),

    code(r'''OUTPUT_LABELS = ("mu_x", "mu_y", "sigma_x", "sigma_y")
response_diagnostics = []
OUTPUT_DIR = ROOT / "work" / "experiment_outputs" / "section_models" / SECTION_MODEL / ("apertures" if SECTION_DETAILED_APERTURES else "no_apertures")
if FAST_RUN:
    OUTPUT_DIR = OUTPUT_DIR / "section_smoke"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for name, item in section_models.items():
    plan, model = item["plan"], item["model"]
    labels = [knob.label for knob in plan.knobs]
    matrix = pd.DataFrame(model.response, index=OUTPUT_LABELS, columns=labels)
    print(f"\n{name}: complete scaled response")
    display(matrix.style.background_gradient(cmap="coolwarm", axis=None))

    correctors = [index for index, knob in enumerate(plan.knobs) if knob.kind == "corrector"]
    quadrupoles = [index for index, knob in enumerate(plan.knobs) if knob.kind == "quadrupole"]
    print("Corrector-to-orbit block:")
    display(matrix.iloc[:2, correctors])
    print("Quadrupole-to-shape block:")
    display(matrix.iloc[2:, quadrupoles])

    active = model.singular_values[model.singular_values > 1e-10]
    response_diagnostics.append(dict(
        section=name,
        rows=model.response.shape[0],
        controls=model.response.shape[1],
        numerical_rank=int(np.linalg.matrix_rank(model.response)),
        largest_singular=model.singular_values[0],
        smallest_active_singular=active[-1],
        active_condition=model.singular_values[0] / active[-1],
        direction_orthogonality_error=np.max(np.abs(
            model.directions @ model.directions.T - np.eye(len(plan.knobs))
        )),
    ))

    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    path = OUTPUT_DIR / f"{slug}_ideal_response.npz"
    np.savez(
        path,
        lattice_model=np.asarray(SECTION_MODEL),
        detailed_apertures=np.asarray(SECTION_DETAILED_APERTURES),
        beam_source=np.asarray(item["configuration"]["beam_source"]),
        input_screen=np.asarray(plan.input_screen),
        output_screen=np.asarray(plan.output_screen),
        knob_labels=np.asarray(labels, dtype="U96"),
        knob_kinds=np.asarray([knob.kind for knob in plan.knobs], dtype="U16"),
        center=model.center,
        target=model.target,
        scale=model.scale,
        response=model.response,
        directions=model.directions,
        singular_values=model.singular_values,
    )
    item["response_file"] = path
    print("Saved:", path)

display(pd.DataFrame(response_diagnostics))

# Inspect the actual feasible line, not just orthogonality of the saved SVD.
direction_rows = []
for name, item in section_models.items():
    model, start = item["model"], item["start"]
    for mode in ("raw_svd", "clean_svd", "boundary_aware"):
        basis = model_search_directions(model, start, mode=mode)
        intervals = direction_intervals(start, basis)
        for index, (row, (lower, upper)) in enumerate(zip(basis, intervals)):
            direction_rows.append(dict(section=name, basis=mode, direction=index,
                                       lower_alpha=lower, upper_alpha=upper,
                                       blocked=upper-lower < 1e-8,
                                       model_gain=np.linalg.norm(model.response @ row)))
    # Store the basis actually passed to RCDS, with knob order and normalization metadata.
    item["search_directions"] = model_search_directions(model, start, mode=DIRECTION_MODE)
    np.savez(OUTPUT_DIR / (re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + "_search_directions.npz"),
             directions=item["search_directions"], mode=DIRECTION_MODE, start=start,
             knob_labels=[k.label for k in item["plan"].knobs],
             lower=[k.lower for k in item["plan"].knobs], upper=[k.upper for k in item["plan"].knobs])
display(pd.DataFrame(direction_rows))
''', 'section-response-matrices'),

    md(r'''## 3. Correction methods and measurement accounting

Every test starts from the same perturbed normalized setting for a section.

- Response correction uses the fixed matrix calculated from the **ideal input beam** and nominal
  section. It takes one starting image, then RESPONSE_STEPS measured correction steps. Each step
  uses the current exit residual to solve for a bounded full correction delta_u and applies only
  RESPONSE_LEARNING_RATE=0.1 of it: u_next = u + 0.1 * delta_u. A fresh exit-screen image is taken
  after every update. The actual input covariance is never supplied to the response model, and
  the matrix is not refitted from these images. The best measured setting across all steps is
  retained. Total cost is RESPONSE_STEPS + 1 images (the full budget by default).
- Full BO fits one GP over all normalized knobs, starting at the same setting. It uses EI
  for image indices i < FULL_BO_SWITCH_AT, then a confidence bound with BO_UCB_BETA=1.
  None selects half of the measurement budget. Index zero is the starting image; initial random
  samples take precedence over acquisition optimization. The per-image ledger records the actual
  phase (start/random/EI/LCB).
- For minimization, BoTorch UCB(maximize=False) maximizes -mu + sqrt(beta)*sigma.
  Full BO uses the equivalent scikit-optimize LCB, minimizing mu - sqrt(beta)*sigma.
  EI uses xi=0, matching improvement over the best observed value.
- RCDS+BO uses the same EI-then-UCB order within each 1D line search: no_ei EI steps,
  followed by no_ucb confidence-bound steps. Its EI is computed as LogEI for numerical stability.
  Set ei_first=False to restore the original UCB-then-EI order.
- RCDS and RCDS+BO receive the configured online budget. Their baseline versions use the identity
  basis.
- The direction-informed versions use DIRECTION_MODE (boundary-aware cleaned model rows by
  default), without changing the start point or budget.

No method receives the actual input beam centre or covariance. Those quantities are used only by
the simulator to create an unknown plant. Final clean moments are evaluated after selection for
diagnosis and are not passed back to an optimizer.''', 'section-method-explanation'),

    code(r'''def section_value(moments, model, part="all"):
    return local_value(moments, model.target, model.scale, part)


def optics_diagnostics(output, item):
    center, covariance = transverse_state(output)
    ideal_center, ideal_covariance = item["ideal_exit_state"]
    scale = np.sqrt(np.maximum(np.diag(ideal_covariance), 1e-30))
    return dict(
        # Dimensionless RMS of all 16 covariance errors scaled by ideal rms coordinates.
        covariance_error=float(np.sqrt(np.mean(
            ((covariance - ideal_covariance) / np.outer(scale, scale))**2
        ))),
        angle_error_urad=float(np.mean(np.abs(center[[1, 3]] - ideal_center[[1, 3]])) * 1e6),
        survival=float(output.survival_probabilities.mean()),
    )


# Extra ideal propagation is a diagnostic simulation, not a response or machine shot.
for item in section_models.values():
    lattice = load_section_lattice()
    apply_knobs(lattice, item["plan"].knobs, item["model"].center)
    item["ideal_exit_state"] = transverse_state(track_interval(lattice, item["plan"], item["ideal_input"]))


def run_method(section_name, item, beam, beam_index, method):
    machine = load_section_lattice()
    plan = make_section_plan(
        machine,
        item["configuration"]["input_screen"],
        item["configuration"]["output_screen"],
        include_rf_phase=False,
    )
    model, start = item["model"], item["start"]
    session = SectionSession(
        machine,
        plan,
        beam,
        model.target,
        budget=MEASUREMENT_BUDGET,
        scale=model.scale,
        noise_fraction=READOUT_NOISE,
        seed=SEED + 50_000 + 100 * beam_index,
    )

    apply_knobs(machine, plan.knobs, start)
    initial_truth = section_readout(track_interval(machine, plan, beam))[0]

    optimize_section(
        session, model, start, method, seed=SEED + 100 * beam_index,
        response_steps=RESPONSE_STEPS, learning_rate=RESPONSE_LEARNING_RATE,
        response_mode=RESPONSE_SOLVE_MODE, ridge=RESPONSE_RIDGE,
        direction_mode=DIRECTION_MODE, rcds_bo_options=RCDS_BO_OPTIONS,
        bo_options=dict(acquisition=FULL_BO_ACQUISITION,
                        switch_at=FULL_BO_SWITCH_AT, beta=BO_UCB_BETA),
    )

    selected = session.best()["u"]
    apply_knobs(machine, plan.knobs, selected)
    output = track_interval(machine, plan, beam)
    final_truth = section_readout(output)[0]
    row = dict(
        section=section_name,
        beam=beam_index,
        method=method,
        **optics_diagnostics(output, item),
        online_measurements=len(session.records),
        initial_truth=section_value(initial_truth, model),
        best_observed=session.best()["value"],
        final_truth=section_value(final_truth, model),
        orbit_truth=section_value(final_truth, model, "orbit"),
        shape_truth=section_value(final_truth, model, "shape"),
        improved=section_value(final_truth, model) < section_value(initial_truth, model),
    )
    for j, label in enumerate(("mu_x", "mu_y", "sigma_x", "sigma_y")):
        row[f"target_{label}_um"] = model.target[j] * 1e6
        row[f"initial_{label}_um"] = initial_truth[j] * 1e6
        row[f"final_{label}_um"] = final_truth[j] * 1e6
    return row, session, selected


result_rows, sessions, selected_controls = [], {}, {}
for section_name, item in section_models.items():
    for beam_index, beam in enumerate(item["beams"]):
        for method in METHODS:
            row, session, selected = run_method(
                section_name, item, beam, beam_index, method
            )
            result_rows.append(row)
            sessions[section_name, beam_index, method] = session
            selected_controls[section_name, beam_index, method] = selected

results = pd.DataFrame(result_rows)
display(results[["section", "beam", "method", "online_measurements", "initial_truth", "best_observed", "final_truth"]].rename(columns={"initial_truth": "initial objective (clean)", "best_observed": "selected objective (noisy)", "final_truth": "final objective (clean)"}))
assert (results[results.method == "response matrix"].online_measurements == RESPONSE_STEPS + 1).all()
assert (
    results[results.method != "response matrix"].online_measurements
    == MEASUREMENT_BUDGET
).all()''', 'section-run-methods'),

    code(r'''# The objective is dimensionless mean absolute scaled error, not the official score.
# Positions use 100 um scales; sizes use max(ideal size, 100 um).
objective_columns = {"initial_truth": "initial objective (clean)",
                     "best_observed": "selected objective (noisy)",
                     "final_truth": "final objective (clean)",
                     "orbit_truth": "orbit objective (clean)",
                     "shape_truth": "size objective (clean)"}
display(results[["beam", "method", "online_measurements", *objective_columns]].rename(columns=objective_columns))
moment_comparison = results.melt(id_vars=["section", "beam", "method"],
    value_vars=[f"{phase}_{label}_um" for label in OUTPUT_LABELS for phase in ("target", "initial", "final")],
    var_name="moment and phase", value_name="value [um]")
display(moment_comparison)
''', 'fb349a60-55e1-4d02-a38f-8e8b1868a860'),

    code(r'''summary = results.groupby(["section", "method"], sort=False).agg(
    mean_initial=("initial_truth", "mean"),
    mean_final=("final_truth", "mean"),
    worst_final=("final_truth", "max"),
    mean_orbit=("orbit_truth", "mean"),
    mean_shape=("shape_truth", "mean"),
    improvement_rate=("improved", "mean"),
    mean_online_measurements=("online_measurements", "mean"),
).reset_index()
summary["mean_reduction_percent"] = 100 * (
    1 - summary.mean_final / summary.mean_initial
)
summary["rank"] = summary.groupby("section").mean_final.rank()
display(summary.sort_values(["section", "rank"]))

direction_comparison = summary.pivot(
    index="section", columns="method", values="mean_final"
)
for algorithm in ("RCDS", "RCDS+BO"):
    direction_comparison[f"{algorithm} direction change [%]"] = 100 * (
        direction_comparison[f"{algorithm} + response directions"]
        / direction_comparison[algorithm]
        - 1
    )
print("Negative direction change means the response basis helped:")
display(direction_comparison[[
    "RCDS direction change [%]",
    "RCDS+BO direction change [%]",
]])''', 'section-scoreboard'),

    code(r'''fig, axes = plt.subplots(
    len(section_models), 2,
    figsize=(13, 4.5 * len(section_models)),
    squeeze=False,
)
for row_index, section_name in enumerate(section_models):
    section_results = results[results.section == section_name]
    for method in METHODS:
        rows = section_results[section_results.method == method]
        axes[row_index, 0].plot(
            rows.beam,
            rows.final_truth,
            "o-",
            label=method,
        )
    axes[row_index, 0].set(
        title=f"{section_name}: clean final score by beam",
        xlabel="test beam",
        ylabel="target-relative score",
        yscale="log",
    )
    axes[row_index, 0].grid(alpha=.25)
    axes[row_index, 0].legend(fontsize=7)

    methods = summary[summary.section == section_name].sort_values("rank")
    axes[row_index, 1].barh(methods.method, methods.mean_final)
    axes[row_index, 1].invert_yaxis()
    axes[row_index, 1].set(
        title=f"{section_name}: mean over beams",
        xlabel="clean final target-relative score",
        xscale="log",
    )
    axes[row_index, 1].grid(alpha=.25, axis="x")
plt.tight_layout()
plt.show()''', 'section-result-plots'),

    code(r'''fig, axes = plt.subplots(
    len(section_models), 1,
    figsize=(10, 4 * len(section_models)),
    squeeze=False,
)
for axis, section_name in zip(axes[:, 0], section_models):
    for method in METHODS:
        method_sessions = [
            sessions[section_name, beam_index, method]
            for beam_index in range(N_BEAMS)
        ]
        curves = [
            np.minimum.accumulate([record["value"] for record in session.records])
            for session in method_sessions
        ]
        common = min(map(len, curves))
        mean_curve = np.stack([curve[:common] for curve in curves]).mean(axis=0)
        axis.plot(np.arange(1, common + 1), mean_curve, label=method)
    axis.set(
        title=f"{section_name}: mean best observed objective",
        xlabel="output-screen measurement",
        ylabel="best measured score",
        yscale="log",
    )
    axis.grid(alpha=.25)
    axis.legend(fontsize=8)
plt.tight_layout()
plt.show()''', 'section-convergence'),

    code(r'''for section_name in section_models:
    ranked = summary[summary.section == section_name].sort_values("mean_final")
    best = ranked.iloc[0]
    response_row = ranked[ranked.method == "response matrix"].iloc[0]
    print(
        f"{section_name}: best mean method is {best.method} "
        f"at {best.mean_final:.4g}; response correction uses "
        f"{response_row.mean_online_measurements:.0f} measurements and improves "
        f"{100 * response_row.improvement_rate:.0f}% of beams."
    )
    for algorithm in ("RCDS", "RCDS+BO"):
        baseline = ranked[ranked.method == algorithm].iloc[0].mean_final
        informed = ranked[
            ranked.method == f"{algorithm} + response directions"
        ].iloc[0].mean_final
        relation = "helps" if informed < baseline else "hurts"
        print(
            f"  Response directions {relation} {algorithm}: "
            f"{baseline:.4g} -> {informed:.4g}."
        )''', 'section-interpretation-code'),

    code(r'''PLOT_BEAM_INDEX = 0
for name, item in section_models.items():
    beam = item["beams"][PLOT_BEAM_INDEX]
    machine = load_section_lattice()
    plan, model = item["plan"], item["model"]
    images = []
    for title, controls in [("initial", item["start"])] + [(method, selected_controls[name, PLOT_BEAM_INDEX, method]) for method in METHODS]:
        apply_knobs(machine, plan.knobs, controls)
        output = track_interval(machine, plan, beam)
        live = output.survival_probabilities.detach().cpu().numpy() > 0
        xy = output.particles.detach().cpu().numpy()[live][:, [0, 2]] * 1e3
        images.append((title, xy))
    from matplotlib.patches import Ellipse
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)
    # Shared spatial extent includes all surviving particles and the ideal 3-sigma ellipse.
    cloud = np.concatenate([xy for _, xy in images] + [np.array([model.target[:2]*1e3-3*model.target[2:]*1e3, model.target[:2]*1e3+3*model.target[2:]*1e3])])
    extent = [(cloud[:,j].min(), cloud[:,j].max()) for j in range(2)]
    for ax, (title, xy) in zip(axes.flat, images):
        ax.hist2d(xy[:,0], xy[:,1], bins=60, range=extent, cmap="viridis")
        ax.add_patch(Ellipse(model.target[:2]*1e3, 6*model.target[2]*1e3,
                             6*model.target[3]*1e3, fill=False, edgecolor="red", label="ideal 3 sigma"))
        ax.set(title=title, xlabel="x [mm]", ylabel="y [mm]")
    for ax in list(axes.flat)[len(images):]: ax.axis("off")
    fig.suptitle(f"{name}, beam {PLOT_BEAM_INDEX}; simulation diagnostic, red = ideal projected sizes")
    plt.tight_layout()
''', '8173e45d-7720-463f-bfd1-d6546f32734c'),

    md(r'''## 4. How to interpret this comparison

The response method repeatedly corrects the measured exit error with a damped ideal-model step. Its
failure mode is model error: it assumes the ideal input centre and covariance, a locally linear
section, and correct magnet calibration. The varied input beams deliberately violate the first
assumption.

Response directions have a weaker requirement. Even if the linear prediction is inaccurate, the
dominant right-singular vectors can still order RCDS searches along combinations that move the
measured output efficiently. Compare each direction-informed row only with the corresponding unit
direction row at the same budget.

A four-moment screen response has rank at most four. Sections with more than four controls
therefore contain response-null directions. The full SVD completes the basis, but those null
directions have no first-order evidence from one screen and may consume budget. A later experiment
can search only active modes first, then add null directions if the objective stalls.

This notebook does not use actual input-screen data and does not reconstruct angles. It also
models setting errors as a displaced normalized start rather than hidden calibration or alignment
errors. Those are deliberate boundaries for this isolated test.''', 'section-interpretation'),

    md(r'''## 5. RL workflow: simulation, one-beam adaptation, and transfer

The PPO experiment has three distinct phases for every selected section:

1. **Simulator pretraining.** Independent beams are sampled from the selected section beam source
   (the new correlated nominal distribution by default). Independent one-beam and several-beam policies have the same rollout budget,
   network initialization, and learning settings. These interactions are offline model calls.
2. **Machine adaptation on one beam.** A pretrained policy is copied and updated using repeated
   episodes on test beam 0. Each reset and action requires a final-screen image, so these calls are
   reported as machine adaptation measurements. This is a separate axis from the number of
   simulation training beams above.
3. **Frozen transfer.** Both the pretrained and adapted policies are frozen, then applied to beams
   1 onward without gradient updates. Only this held-out set enters the RL transfer table.

This notebook has no hardware connection, so phase 2 is an **emulated machine experiment** backed
by the same screen-measurement interface. The accounting and separation are ready for hardware,
but the result is not evidence of a real sim-to-real transfer. A hardware adapter must implement
reset/set-magnets/read-final-screen while preserving the normalized controls and safety limits.

### Physics-guided residual PPO

For full phase-space state z=(x, p_x, y, p_y, tau, p), ideal first-order propagation gives

mu_2 = M(u) mu_1 + d(u),                 Sigma_2 = M(u) Sigma_1 M(u)^T.

Use the full map in dispersive sections; a closed 4D transverse map needs negligible coupling
to longitudinal coordinates. The screen map keeps mu_x, mu_y, sqrt(Sigma_xx), and
sqrt(Sigma_yy). Differentiating that map
around the ideal beam and reference magnets gives the saved response J. For the measured scaled
residual r, a bounded ridge solve supplies a local physics correction

delta_u_phys = argmin ||J delta_u + r||^2 + alpha ||delta_u||^2.

The environment applies

a_applied = clip(a_PPO + lambda a_phys, -1, 1).

Thus PPO learns the residual left by the ideal model. RL_PHYSICS_WEIGHT=lambda controls the prior:
zero is ordinary PPO, while a positive value incorporates ideal propagation. This uses no actual
input-screen measurement and does not claim that the model is exact.

Simulation pretraining uses `RL_TRAIN_PARTICLES` (at most 4096 by default); adaptation and
held-out evaluation still use `NUM_PARTICLES`. This changes particle sampling accuracy, not the
transport physics. For a resolution-matched comparison set `RL_TRAIN_PARTICLES=NUM_PARTICLES`.
Apertures and losses remain active. `RL_FAST_TRACKING` omits nonblocking zero-length screen
copies because this environment reads exit particle moments, not camera images; blocking
screens remain in the transport. Results can differ slightly from floating-point map regrouping.
Iteration timing and ETA are printed during each training phase. PPO iterations, episodes,
horizon, and the four policy variants are unchanged.


All RL phases now use `load_section_lattice()` and the same `sample_section_input()` beam
construction as the classical comparison. The reduced-particle pretraining pool comes from the
new correlated nominal beam and is propagated to each section's entrance through the new optics
and apertures. Adaptation/evaluation retain NUM_PARTICLES. Physics priors use the matching nominal
response matrix; old-lattice checkpoints must not be reused. Saved artifacts record the lattice,
aperture setting and beam source, and are stored separately under `experiment_outputs/section_models`.
''', 'section-rl-explanation'),

    code(r'''def sample_rl_beam(configuration, reference_lattice, seed):
    return sample_section_input(configuration, reference_lattice, seed,
                                RL_TRAIN_PARTICLES, SECTION_BEAM_SEVERITY)


def make_rl_environment(item, beam, physics_weight):
    lattice = load_section_lattice()
    plan = make_section_plan(
        lattice,
        item["configuration"]["input_screen"],
        item["configuration"]["output_screen"],
        include_rf_phase=False,
    )
    return SectionRLEnv(
        lattice,
        plan,
        beam,
        item["model"],
        item["start"],
        config=SectionRLConfig(
            horizon=RL_HORIZON,
            max_knob_step=RL_MAX_KNOB_STEP,
            physics_weight=physics_weight,
            noise_fraction=READOUT_NOISE,
            ridge=RESPONSE_RIDGE,
            fast_tracking=RL_FAST_TRACKING,
        ),
    )


def training_progress(label, iterations):
    def report(row):
        i = row["iteration"]
        if i == 1 or i % RL_PROGRESS_EVERY == 0 or i == iterations:
            elapsed = row["elapsed_seconds"]
            eta = elapsed / i * (iterations - i)
            print(f"{label}: {i}/{iterations}, elapsed {elapsed/60:.1f} min, "
                  f"ETA {eta/60:.1f} min, best {row['mean_best_loss']:.4g}", flush=True)
    return report

print({"pretraining particles": RL_TRAIN_PARTICLES, "adaptation/evaluation particles": NUM_PARTICLES,
       "pretraining images per variant": RL_PRETRAIN_ITERATIONS * RL_PRETRAIN_EPISODES * (RL_HORIZON+1),
       "fast moment tracking": RL_FAST_TRACKING})

pretrain_config = PPOConfig(
    iterations=RL_PRETRAIN_ITERATIONS,
    episodes_per_iteration=RL_PRETRAIN_EPISODES,
    horizon=RL_HORIZON,
    max_knob_step=RL_MAX_KNOB_STEP,
    learning_rate=3e-4,
    update_epochs=3 if FAST_RUN else 5,
    minibatch_size=32,
    reward_scale=0.25,
)
adapt_config = PPOConfig(
    iterations=RL_ADAPT_ITERATIONS,
    episodes_per_iteration=RL_ADAPT_EPISODES,
    horizon=RL_HORIZON,
    max_knob_step=RL_MAX_KNOB_STEP,
    learning_rate=1e-4,
    update_epochs=3,
    minibatch_size=32,
    entropy_weight=0.002,
    reward_scale=0.25,
)

RL_VARIANTS = {
    "PPO / one beam": (0.0, 1),
    "PPO / several beams": (0.0, RL_SIMULATION_BEAMS),
    "physics PPO / one beam": (RL_PHYSICS_WEIGHT, 1),
    "physics PPO / several beams": (RL_PHYSICS_WEIGHT, RL_SIMULATION_BEAMS),
}
rl_artifacts = {}
training_rows = []

for section_index, (section_name, item) in enumerate(section_models.items()):
    reference_lattice = load_section_lattice()
    simulator_beams = [
        sample_rl_beam(
            item["configuration"],
            reference_lattice,
            SEED + 10_000 + 100 * section_index + index,
        )
        for index in range(RL_SIMULATION_BEAMS)
    ]
    for variant, (physics_weight, training_beam_count) in RL_VARIANTS.items():
        training_environments = [
            make_rl_environment(item, beam, physics_weight)
            for beam in simulator_beams[:training_beam_count]
        ]
        initialization_seed = SEED + 20_000 + section_index
        torch.manual_seed(initialization_seed)
        agent = ActorCritic(
            training_environments[0].observation_dim,
            training_environments[0].action_dim,
            hidden=RL_HIDDEN,
        )
        pretrain_history = train_ppo(
            agent,
            training_environments,
            pretrain_config,
            seed=SEED + 30_000 + section_index,
            progress=training_progress(f"{section_name} / {variant} / pretrain", pretrain_config.iterations),
        )
        pretrained = deepcopy(agent).eval()
        pretrained.requires_grad_(False)

        # Beam 0 is the only beam used for machine adaptation.
        adapted = deepcopy(pretrained).requires_grad_(True)
        adaptation_environment = make_rl_environment(
            item,
            item["beams"][0],
            physics_weight,
        )
        adapt_history = train_ppo(
            adapted,
            adaptation_environment,
            adapt_config,
            seed=SEED + 40_000 + section_index,
            progress=training_progress(f"{section_name} / {variant} / adapt", adapt_config.iterations),
        )
        adapted.eval().requires_grad_(False)

        safe_name = re.sub(r"[^a-z0-9]+", "_", section_name.lower()).strip("_")
        safe_variant = re.sub(r"[^a-z0-9]+", "_", variant.lower()).strip("_")
        checkpoint = OUTPUT_DIR / f"{safe_name}_{safe_variant}_adapted.pt"
        torch.save({
            "state_dict": adapted.state_dict(),
            "lattice_model": SECTION_MODEL,
            "detailed_apertures": SECTION_DETAILED_APERTURES if SECTION_MODEL != "scorer" else False,
            "beam_source": item["configuration"]["beam_source"],
            "beam_severity": SECTION_BEAM_SEVERITY,
            "observation_dim": adapted.observation_dim,
            "action_dim": adapted.action_dim,
            "hidden": RL_HIDDEN,
            "input_screen": item["configuration"]["input_screen"],
            "output_screen": item["configuration"]["output_screen"],
            "physics_weight": physics_weight,
            "training_beam_count": training_beam_count,
            "pretrain_particles": RL_TRAIN_PARTICLES,
            "evaluation_particles": NUM_PARTICLES,
            "fast_moment_tracking": RL_FAST_TRACKING,
            "knob_labels": [k.label for k in item["plan"].knobs],
            "knob_lower": [k.lower for k in item["plan"].knobs],
            "knob_upper": [k.upper for k in item["plan"].knobs],
            "target": item["model"].target.tolist(),
            "scale": item["model"].scale.tolist(),
            "response": item["model"].response.tolist(),
            "start": item["start"].tolist(),
            "max_knob_step": RL_MAX_KNOB_STEP,
            "horizon": RL_HORIZON,
            "readout_noise": READOUT_NOISE,
            "ridge": RESPONSE_RIDGE,
        }, checkpoint)

        rl_artifacts[section_name, variant] = dict(
            pretrained=pretrained,
            adapted=adapted,
            physics_weight=physics_weight,
            pretrain_history=pd.DataFrame(pretrain_history),
            adapt_history=pd.DataFrame(adapt_history),
            pretrain_measurements=pretrain_history[-1]["offline_measurements"],
            adaptation_measurements=adapt_history[-1]["offline_measurements"],
            checkpoint=checkpoint,
        )
        training_rows.append(dict(
            section=section_name,
            variant=variant,
            simulator_beams=training_beam_count,
            simulator_pretraining_calls=pretrain_history[-1]["offline_measurements"],
            one_beam_adaptation_calls=adapt_history[-1]["offline_measurements"],
            deployment_calls_per_beam=RL_HORIZON + 1,
            physics_weight=physics_weight,
            checkpoint=checkpoint.name,
        ))

training_accounting = pd.DataFrame(training_rows)
assert training_accounting.simulator_pretraining_calls.nunique() == 1
assert training_accounting.one_beam_adaptation_calls.nunique() == 1
display(training_accounting)''', 'section-rl-training'),

    code(r'''fig, axes = plt.subplots(
    len(section_models), 2,
    figsize=(13, 4 * len(section_models)),
    squeeze=False,
)
for row_index, section_name in enumerate(section_models):
    for variant in RL_VARIANTS:
        artifact = rl_artifacts[section_name, variant]
        pretrain = artifact["pretrain_history"]
        adaptation = artifact["adapt_history"]
        axes[row_index, 0].plot(
            pretrain.iteration,
            pretrain.mean_best_loss,
            label=variant,
        )
        axes[row_index, 1].plot(
            adaptation.iteration,
            adaptation.mean_best_loss,
            "o-",
            label=variant,
        )
    axes[row_index, 0].set(
        title=f"{section_name}: simulator pretraining",
        xlabel="PPO iteration",
        ylabel="mean best measured score",
        yscale="log",
    )
    axes[row_index, 1].set(
        title=f"{section_name}: adaptation on beam 0",
        xlabel="adaptation iteration",
        ylabel="best measured score",
        yscale="log",
    )
    for axis in axes[row_index]:
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
plt.tight_layout()
plt.show()''', 'section-rl-training-plots'),

    md(r'''## 6. Freeze policies and correct different beams

The deployment beams were not used for simulator training or machine adaptation.
Test beam 0 is reserved for adaptation and excluded from every combined comparison. Each policy gets
one reset image and RL_HORIZON action images. Selection uses the best noisy measured setting;
the clean score below is calculated afterward only for simulation diagnosis.

The sim-only rows show the reusable pretrained policy. The adapted rows show whether measurements
from beam 0 improve or overfit transfer to beams 1 onward.''', 'section-rl-transfer-explanation'),

    code(r'''rl_rows, rl_sessions = [], {}
for section_name, item in section_models.items():
    for beam_index in range(1, N_BEAMS):
        beam = item["beams"][beam_index]
        for variant, (physics_weight, _) in RL_VARIANTS.items():
            artifact = rl_artifacts[section_name, variant]
            for phase, policy in (
                ("sim-only", artifact["pretrained"]),
                ("sim + one-beam adaptation", artifact["adapted"]),
            ):
                environment = make_rl_environment(item, beam, physics_weight)
                apply_knobs(
                    environment.lattice,
                    environment.plan.knobs,
                    item["start"],
                )
                initial_truth = section_readout(track_interval(
                    environment.lattice,
                    environment.plan,
                    beam,
                ))[0]
                selected, session = run_section_policy(
                    policy,
                    environment,
                    seed=SEED + 50_000 + 100 * beam_index,
                )
                apply_knobs(environment.lattice, environment.plan.knobs, selected)
                output = track_interval(environment.lattice, environment.plan, beam)
                final_truth = section_readout(output)[0]
                method = f"{variant} ({phase})"
                rl_rows.append(dict(
                    section=section_name,
                    beam=beam_index,
                    method=method,
                    **optics_diagnostics(output, item),
                    online_measurements=len(session.records),
                    initial_truth=section_value(initial_truth, item["model"]),
                    best_observed=session.best()["value"],
                    final_truth=section_value(final_truth, item["model"]),
                    orbit_truth=section_value(final_truth, item["model"], "orbit"),
                    shape_truth=section_value(final_truth, item["model"], "shape"),
                    improved=(
                        section_value(final_truth, item["model"])
                        < section_value(initial_truth, item["model"])
                    ),
                    simulator_pretraining_calls=artifact["pretrain_measurements"],
                    one_beam_adaptation_calls=(
                        artifact["adaptation_measurements"]
                        if phase == "sim + one-beam adaptation"
                        else 0
                    ),
                    physics_weight=physics_weight,
                ))
                rl_sessions[section_name, beam_index, method] = session

rl_results = pd.DataFrame(rl_rows)
assert (rl_results.online_measurements == RL_HORIZON + 1).all()
assert (rl_results.beam > 0).all()
display(rl_results)''', 'section-rl-transfer'),

    code(r'''rl_summary = rl_results.groupby(["section", "method"], sort=False).agg(
    mean_initial=("initial_truth", "mean"),
    mean_final=("final_truth", "mean"),
    worst_final=("final_truth", "max"),
    mean_orbit=("orbit_truth", "mean"),
    mean_shape=("shape_truth", "mean"),
    improvement_rate=("improved", "mean"),
    deployment_calls=("online_measurements", "mean"),
    simulator_pretraining_calls=("simulator_pretraining_calls", "first"),
    one_beam_adaptation_calls=("one_beam_adaptation_calls", "first"),
).reset_index()
rl_summary["mean_reduction_percent"] = 100 * (
    1 - rl_summary.mean_final / rl_summary.mean_initial
)
display(rl_summary.sort_values(["section", "mean_final"]))

# Compare against the earlier methods on exactly the same held-out beams.
classic_transfer = results[
    (results.beam > 0)
    & results.method.isin([
        "response matrix",
        "full BO",
        "RCDS",
        "RCDS+BO",
        "RCDS + response directions",
        "RCDS+BO + response directions",
    ])
].copy()
classic_transfer["simulator_pretraining_calls"] = 0
classic_transfer["one_beam_adaptation_calls"] = 0

combined_transfer = pd.concat(
    [classic_transfer, rl_results],
    ignore_index=True,
    sort=False,
)
combined_summary = combined_transfer.groupby(
    ["section", "method"], sort=False
).agg(
    mean_final=("final_truth", "mean"),
    worst_final=("final_truth", "max"),
    improvement_rate=("improved", "mean"),
    deployment_calls=("online_measurements", "mean"),
    simulator_pretraining_calls=("simulator_pretraining_calls", "first"),
    one_beam_adaptation_calls=("one_beam_adaptation_calls", "first"),
).reset_index()
combined_summary["rank"] = combined_summary.groupby("section").mean_final.rank()
display(combined_summary.sort_values(["section", "rank"]))''', 'section-rl-scoreboard'),

    code(r'''fig, axes = plt.subplots(
    len(section_models), 1,
    figsize=(12, 7 * len(section_models)),
    squeeze=False,
)
for axis, section_name in zip(axes[:, 0], section_models):
    rows = combined_summary[combined_summary.section == section_name].sort_values("mean_final")
    axis.barh(rows.method, rows.mean_final)
    axis.invert_yaxis()
    axis.set(
        title=f"{section_name}: held-out beam correction",
        xlabel="mean clean final target-relative score",
        xscale="log",
    )
    axis.grid(alpha=.25, axis="x")
plt.tight_layout()
plt.show()

for section_name in section_models:
    table = rl_summary[rl_summary.section == section_name].set_index("method")
    print(f"\n{section_name}")
    for variant in RL_VARIANTS:
        before = table.loc[f"{variant} (sim-only)", "mean_final"]
        after = table.loc[
            f"{variant} (sim + one-beam adaptation)",
            "mean_final",
        ]
        effect = 100 * (after / before - 1)
        word = "improves" if effect < 0 else "degrades"
        print(
            f"  One-beam adaptation {word} held-out {variant} by "
            f"{abs(effect):.1f}% ({before:.4g} -> {after:.4g})."
        )
    for training in ("one beam", "several beams"):
        vanilla = table.loc[f"PPO / {training} (sim + one-beam adaptation)", "mean_final"]
        informed = table.loc[f"physics PPO / {training} (sim + one-beam adaptation)", "mean_final"]
        print(f"  Physics prior, {training}: {vanilla:.4g} -> {informed:.4g} (lower is better).")''', 'section-rl-interpretation-code'),

    md(r'''## 7. What the RL result means

The policy is section-specific: changing the screen pair can change the number and identity of
knobs, so the notebook trains a separate actor for each section. It observes current normalized
knobs, the four scaled final-screen residuals, current objective, remaining episode fraction,
and a valid-reading flag. Physics weight zero skips the response solver entirely.
It never observes the true input covariance, divergences, or clean simulation state.

Physics can enter RL at several levels. This experiment uses an action prior because it remains
usable when the input beam is unknown and its weight has a clear zero limit. Other valid choices
are response features in the observation, imitation pretraining on response corrections, or an
auxiliary consistency loss comparing measured transitions with r_next=r+J delta_u. An auxiliary
loss should be down-weighted when model error is large; otherwise it can force the policy to
reproduce the same bias that adaptation is meant to remove.

Judge sim-to-machine adaptation by held-out performance and by its explicit measurement cost.
Improvement on beam 0 alone is insufficient. If adaptation degrades beams 1 onward, it has
specialized to one input beam; reduce the learning rate or iterations, regularize toward the
pretrained weights, or adapt on several representative beams. The emulated result should be
repeated with calibration errors, misalignments, time drift, camera noise, and hardware limits
before choosing a real-machine measurement budget.''', 'section-rl-interpretation'),

    md(r'''## 8. Equal deployment budget and full-optics diagnostics

The preceding table retains each method's configured budget: response uses RESPONSE_STEPS+1 images,
classical search uses MEASUREMENT_BUDGET, and PPO uses RL_HORIZON+1. Its rank alone is not a fair speed
comparison. This table uses the first RL_HORIZON+1 measured points from each classical search,
selects the best by the **noisy** objective, and reports the cached clean value at that point.
No additional measurement or retraining is performed. Response uses the same maximum budget
by default; reducing RESPONSE_STEPS can use fewer images.

The local objective is mean absolute scaled error in (mu_x, mu_y, sigma_x, sigma_y). Missing images
receive a finite failure value and are never preferred to valid images. Readout noise is controlled
by READOUT_NOISE; the idealized screen uses all surviving particles and has no
finite camera field of view. This is deliberately separate from score.py's FEH camera metric.

The final diagnostic table reports angle error and all 16 transverse covariance entries relative
to the propagated ideal beam, using simulator truth only. A good spot-size score can coexist with
a large covariance error. Full optics matching would require additional measurements, such as
quadrupole scans or several transport views; actual input angles/covariance are not supplied here.

All policies currently train and deploy from the same section start setting. This tests transfer
between input beams, not transfer between arbitrary starting magnet configurations. Machine
adaptation remains simulated. Results from four held-out beams and one training seed are exploratory.''', 'section-review-explanation'),

    code(r'''matched_rows = []
for section_name, item in section_models.items():
    for beam_index in range(1, N_BEAMS):
        for method in METHODS:
            records = sessions[section_name, beam_index, method].records[:RL_HORIZON + 1]
            selected = min(records, key=lambda record: record["value"])
            matched_rows.append(dict(
                section=section_name, beam=beam_index, method=method,
                final_truth=section_value(selected["clean"], item["model"]),
                online_measurements=len(records),
            ))
matched_results = pd.concat([pd.DataFrame(matched_rows), rl_results], ignore_index=True)
matched_summary = matched_results.groupby(["section", "method"]).agg(
    mean_final=("final_truth", "mean"), worst_final=("final_truth", "max"),
    deployment_calls=("online_measurements", "mean"),
).reset_index()
print("Matched maximum deployment budget; actual calls shown for every method:")
display(matched_summary.sort_values(["section", "mean_final"]))

print("Full configured budgets: simulation-only optics diagnostics (held-out beams):")
optics_summary = combined_transfer.groupby(["section", "method"]).agg(
    screen_score=("final_truth", "mean"),
    covariance_error=("covariance_error", "mean"),
    angle_error_urad=("angle_error_urad", "mean"),
    minimum_survival=("survival", "min"),
).reset_index()
display(optics_summary)
# These are diagnostics, never targets supplied to a tuner.
matched_results.to_csv(OUTPUT_DIR / "section_matched_budget_results.csv", index=False)
combined_transfer.to_csv(OUTPUT_DIR / "section_comparison_results.csv", index=False)
training_accounting.to_csv(OUTPUT_DIR / "section_training_accounting.csv", index=False)''', 'section-review-comparison'),

    md(r'''## 9. Correct one machine screen by screen with a selected method

This cell can run after setup without completing PPO training. By default it loads the new
correlated nominal S02 beam, uses its unperturbed version for the model, and adds explicit
injection jitter for the unknown beam. The old scorer distribution remains selectable.
A single perturbed lattice is retained throughout. For each physical screen,
choose the nearest earlier screen with at least four quadrupole strengths and four corrector
angles in between. If that is impossible, use all controls from the S02 entrance and flag the
shortage. Windows may overlap and have more than eight controls. No RF controls are changed.

The actual input at each window is recomputed from the same root beam through the currently
corrected upstream machine. It is simulator state, not information given to the tuner. Targets
and response matrices always come from the ideal reference. Each measurement reads only the
selected exit screen; internal particle transport and clean audit plots are simulation diagnostics.
The model uses ideal moment readouts, not finite camera images or the official FEH score.

After each correction, magnets remain set for the next screen. Later overlapping corrections
can spoil earlier screens, so the final audit reevaluates **every** screen using the final lattice.
This is a single forward pass, not a guarantee of simultaneous full-optics correction. It finally
propagates to the end of the lattice even if correction was limited to the first few screens.
Choose any classical method in `FULL_METHOD`; existing PPO policies have section-specific inputs
and cannot be applied to new screen windows without training a matching policy.


Some downstream screens may have no valid ideal target because the nominal scorer beam is lost
in the reference lattice. These intervals are explicitly marked skipped; no objective or successful
correction is fabricated. If only the finite-difference scan loses the beam, the response scan
retries with smaller steps. The final audit reports actual survival even for skipped targets.


`FULL_VERBOSE` prints section starts, measurement progress every `FULL_LOG_EVERY` images,
skipped sections, section summaries, and the final audit. Reports distinguish **exit transmission**
(surviving particle weight relative to the original input beam) from **within-section transmission**
(exit weight divided by that window's incoming weight). Zero incoming weight gives an undefined
ratio (NaN), not 100% transmission. These are simulator diagnostics; no extra online images are
charged for reports. The first report describes each correction when it happened; the final audit
reflects all later overlapping changes. Percent columns are saved with the reports as CSV files.


### New nominal model, beam and apertures

The final workflow now defaults to the **lattice-file strengths** recommended by
`nominal_model/README.md`, loaded directly rather than overwritten with the scorer's reference
tune. `FULL_MODEL="gpsr"` selects `nominal_model/CLARA_cheetah_S07_GPSR.json`: those are the scan
settings used when reconstructing the input beam, not the default correction optics.
`FULL_MODEL="scorer"` retains the old scorer loader and its case-state behavior.
The earlier isolated-section/RL comparisons are unchanged.

`nominal_model/initial_distribution.npz` contains 200,000 correlated 6D particles at
`cla_s02_sim_aper_01`, 35.8055 MeV, in Cheetah coordinates. We subsample those particles,
preserving correlations; there is no tau sign conversion. The stored beam is distinct from
`score.sample_beam`. The plant uses an explicitly documented affine injection-jitter model.
The NPZ has no charge, so the loader assumes 250 pC; charge does not affect this local objective.

`data/clara/lattice_data/apertures.json` supplies the new vacuum chambers: e.g. 17 mm pipe,
10 mm linacs, 4.1195 mm L4H and 5 mm TDC half apertures. The lattice JSON alone does not
insert these chambers. The work-local Cheetah loader mirrors the nominal-model aperture
placement without requiring xsuite, checks chamber changes and element exits, and overlays
those actual checks on the envelope plots. The scorer still inserts its own 50 mm pipes;
`score.py` is not modified. `FULL_MODEL_END` defaults to the nominal model's validated FEH readout.
This is Cheetah transport, not the more chromatic xsuite transport used to reconstruct the beam.

`load_full_lattice(errors=False)` builds the reference. `errors=True` adds seeded corrector/quad
setting errors and scorer-scale magnet/screen misalignments to a separate instance. It never
perturbs the response-model lattice. Screen offsets do not change the current moment-based
readout, which observes propagated beam moments rather than camera coordinates.

Every optimizer restores `session.best()["u"]` before propagation to the next window.
`selected_evaluation` and the red convergence marker show the chosen image, which can precede
the last one. Reported clean objectives and simulated transmission never select the setting.
Exit plots show the design ellipse in physical units and a design circle in normalized units;
a round overlay in millimetres would misrepresent an anisotropic target.
''', 'section-full-explanation'),

    code(r'''METHODS''', 'a495a9ac-babf-47b5-8857-aba11d10d80e'),

    code(r'''MEASUREMENT_BUDGET''', '36b55bc8-d9db-4495-bc55-a3a30e13b040'),

    code(r'''# Independent of PPO training: run setup, then this cell.
FULL_METHOD = "response matrix"  # any entry in METHODS (section-specific PPO is not reusable here)
FULL_METHOD = "RCDS+BO"  # any entry in METHODS (section-specific PPO is not reusable here)
FULL_METHOD = "response matrix"  # any entry in METHODS (section-specific PPO is not reusable here)

FULL_FINAL_SCREEN = None  # None: last physical screen; otherwise its exact name
FULL_MAX_SECTIONS = 5 if FAST_RUN else None
FULL_MIN_QUADRUPOLES = 4
FULL_MIN_CORRECTORS = 4  # H/V angles, i.e. normally two combined corrector devices
FULL_BUDGET_PER_SCREEN = MEASUREMENT_BUDGET
FULL_BUDGET_PER_SCREEN = 256

FULL_BEAM_SEVERITY = SCORE_BEAM_SEVERITY
FULL_START_ERROR = START_ERROR_FRACTION
FULL_VERBOSE = True
FULL_LOG_EVERY = 16

from work.nominal_machine import load_lattice as load_full_lattice, load_beam as load_nominal_beam
from work.section_diagnostics import plot_section_convergence, plot_pass_envelopes, plot_exit_beams

FULL_MODEL = "nominal"  # nominal: new recommended lattice-file tune; gpsr: scan tune; scorer: old benchmark
FULL_DETAILED_APERTURES = True
FULL_MODEL_END = "cla_feh_dia_scr_01"  # validated nominal-model extent; None extends to the lattice end
FULL_MISALIGNMENT_SEVERITY = 1.0
FULL_PLOT_EACH_SECTION = True
# Keep the user's production 512-image budget; smoke runs use six images.
if FAST_RUN:
    FULL_BUDGET_PER_SCREEN = min(FULL_BUDGET_PER_SCREEN, 6)

reference_machine = load_full_lattice(model=FULL_MODEL, errors=False,
    apertures=FULL_DETAILED_APERTURES, end=FULL_MODEL_END)
full_machine = load_full_lattice(model=FULL_MODEL, errors=True, seed=SEED+9001,
    setpoint_fraction=FULL_START_ERROR, misalignment_severity=FULL_MISALIGNMENT_SEVERITY,
    apertures=FULL_DETAILED_APERTURES, end=FULL_MODEL_END)
if FULL_MODEL == "scorer":
    ideal_root = sample_score_beam(np.random.default_rng(SEED), num_particles=NUM_PARTICLES, severity=0.)
    actual_root = sample_score_beam(np.random.default_rng(SEED+9000), num_particles=NUM_PARTICLES, severity=FULL_BEAM_SEVERITY)
    from work.beamline_tuning import input_boundary_name
    full_input_name = input_boundary_name(reference_machine)
else:
    # Use the same sampled base particles for model and plant, then add explicit injection jitter.
    ideal_root = load_nominal_beam(NUM_PARTICLES, seed=SEED, severity=0.)
    actual_root = load_nominal_beam(NUM_PARTICLES, seed=SEED, severity=FULL_BEAM_SEVERITY)
    full_input_name = "cla_s02_sim_aper_01"
full_plot_screen = FULL_FINAL_SCREEN or FULL_MODEL_END or "cla_feh_dia_scr_01"
print({"model": FULL_MODEL, "input beam": full_input_name,
       "detailed apertures": FULL_DETAILED_APERTURES, "misalignment severity": FULL_MISALIGNMENT_SEVERITY})
full_plans = plan_screen_sequence(reference_machine, input_screen=full_input_name,
    min_quadrupoles=FULL_MIN_QUADRUPOLES, min_correctors=FULL_MIN_CORRECTORS,
    final_screen=FULL_FINAL_SCREEN, max_sections=FULL_MAX_SECTIONS)
plan_table = pd.DataFrame([dict(input_screen=p.input_screen, output_screen=p.output_screen,
    quadrupoles=sum(k.kind=="quadrupole" for k in p.knobs),
    corrector_angles=sum(k.kind=="corrector" for k in p.knobs)) for p in full_plans])
plan_table["enough_controls"] = ((plan_table.quadrupoles >= FULL_MIN_QUADRUPOLES)
                                  & (plan_table.corrector_angles >= FULL_MIN_CORRECTORS))
display(plan_table)
full_run = correct_screen_sequence(full_machine, reference_machine, actual_root, ideal_root, full_plans,
    verbose=FULL_VERBOSE, log_every=FULL_LOG_EVERY,
    section_callback=plot_section_convergence if FULL_PLOT_EACH_SECTION else None,
    method=FULL_METHOD, budget=FULL_BUDGET_PER_SCREEN, noise_fraction=READOUT_NOISE, seed=SEED+9100,
    response_step=RESPONSE_STEP, learning_rate=RESPONSE_LEARNING_RATE,
    response_mode=RESPONSE_SOLVE_MODE, ridge=RESPONSE_RIDGE,
    direction_mode=DIRECTION_MODE, rcds_bo_options=RCDS_BO_OPTIONS,
    bo_options=dict(acquisition=FULL_BO_ACQUISITION, switch_at=FULL_BO_SWITCH_AT, beta=BO_UCB_BETA))
full_steps = pd.DataFrame(full_run["records"])
full_audit = pd.DataFrame(full_run["audit"])
# Reports in percent; raw fractional values remain in full_steps and full_audit.
def transmission_report(frame):
    report = frame.copy()
    for column in list(report.columns):
        if "transmission" in column:
            report[column + "_pct"] = 100 * report.pop(column)
    return report

full_section_report = transmission_report(full_steps)
full_final_report = transmission_report(full_audit)
print("Per-section correction report (at the time each window was corrected)")
display(full_section_report[["input_screen", "output_screen", "status", "knobs", "online_measurements",
    "initial_objective", "corrected_objective", "selected_evaluation", "best_measured_objective", "input_transmission_pct",
    "initial_transmission_pct", "corrected_transmission_pct",
    "initial_section_transmission_pct", "corrected_section_transmission_pct", "elapsed_seconds"]])
print("Final audit (after all overlapping corrections)")
display(full_final_report[["screen", "status", "initial_objective", "final_objective",
    "initial_transmission_pct", "final_transmission_pct", "final_section_transmission_pct"]])
print({"online images": full_run["online_measurements"],
       "offline model tracks (upper bound)": full_run["offline_response_tracks_upper_bound"],
       "elapsed seconds": full_run["elapsed_seconds"],
       "end-of-lattice transmission [%]": 100 * full_run["final_transmission"]})
full_report_dir = ROOT / "work" / "experiment_outputs"
if FAST_RUN:
    full_report_dir = full_report_dir / "section_smoke"
full_report_dir.mkdir(parents=True, exist_ok=True)
full_section_report.to_csv(full_report_dir / "full_correction_sections.csv", index=False)
full_final_report.to_csv(full_report_dir / "full_correction_final_audit.csv", index=False)
print("Saved reports to", full_report_dir)
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
full_audit.plot(x="screen", y=["initial_objective", "final_objective"], marker="o", ax=axes[0], logy=True)
axes[0].set_ylabel("Local objective (clean diagnostic)")
full_final_report.plot(x="screen", y=["initial_transmission_pct", "final_transmission_pct"], marker="o", ax=axes[1])
axes[1].set_ylabel("Transmission from input [%]")
axes[1].set_ylim(0, 105)
plt.xticks(rotation=75, ha="right")
plt.tight_layout()

full_passes = [full_run]
full_envelopes = plot_pass_envelopes(full_run, reference_machine, actual_root, ideal_root, title="First correction pass")
plot_exit_beams(full_run, reference_machine, actual_root, ideal_root, full_plot_screen)
''', '42ae23d6-7d87-4a3b-aefb-e28b91591982'),

    md(r'''The model/beam/aperture investigation and diagnostic plots are integrated above.
The envelope plots include the aperture limits loaded into the simulation. Per-section plots mark
the best measured setting restored before proceeding. The next cell continues from that machine.''', 'a4f4afbe-53f9-4f4b-8c19-f11bdcc7ee3f'),

    code(r'''# Continue from pass 1: no reload, no fresh setting errors, no new input beam.
# Rerunning this cell performs another pass from the current corrected machine.
pass_number = len(full_passes) + 1
second_pass = correct_screen_sequence(full_machine, reference_machine, actual_root, ideal_root, full_plans,
    verbose=FULL_VERBOSE, log_every=FULL_LOG_EVERY,
    section_callback=plot_section_convergence if FULL_PLOT_EACH_SECTION else None,
    method=FULL_METHOD, budget=FULL_BUDGET_PER_SCREEN, noise_fraction=READOUT_NOISE,
    seed=SEED+9100+1000*(pass_number-1), response_step=RESPONSE_STEP,
    learning_rate=RESPONSE_LEARNING_RATE, response_mode=RESPONSE_SOLVE_MODE, ridge=RESPONSE_RIDGE,
    direction_mode=DIRECTION_MODE, rcds_bo_options=RCDS_BO_OPTIONS,
    bo_options=dict(acquisition=FULL_BO_ACQUISITION, switch_at=FULL_BO_SWITCH_AT, beta=BO_UCB_BETA))
full_passes.append(second_pass)
display(transmission_report(pd.DataFrame(second_pass["records"])))
display(transmission_report(pd.DataFrame(second_pass["audit"])))
plot_pass_envelopes(second_pass, reference_machine, actual_root, ideal_root,
                    title=f"Pass {pass_number}: previous corrected machine -> next correction")
plot_exit_beams(second_pass, reference_machine, actual_root, ideal_root, full_plot_screen)
transmission_report(pd.DataFrame(second_pass["records"])).to_csv(
    full_report_dir / f"full_correction_pass_{pass_number}_sections.csv", index=False)
transmission_report(pd.DataFrame(second_pass["audit"])).to_csv(
    full_report_dir / f"full_correction_pass_{pass_number}_audit.csv", index=False)
print({"pass": pass_number, "pass images": second_pass["online_measurements"],
       "total images across passes": sum(r["online_measurements"] for r in full_passes),
       "end transmission [%]": 100*second_pass["final_transmission"]})
''', 'a80b7eff-9fda-40db-b37a-cc4a4d5c8eea'),

]
metadata = {'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'}, 'language_info': {'codemirror_mode': {'name': 'ipython', 'version': 3}, 'file_extension': '.py', 'mimetype': 'text/x-python', 'name': 'python', 'nbconvert_exporter': 'python', 'pygments_lexer': 'ipython3', 'version': '3.14.4'}}
nbf.write(nbf.v4.new_notebook(cells=cells, metadata=metadata), WORK / "section.ipynb")
