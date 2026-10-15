# Stochastic SVD tuning

This directory contains the staged optimization workflow, its CLI, tests, diagnostic plots and saved runs. It calls the actual `RL_framework/rl_framework/envs/sim_env.py` environment and the BO and RCDS implementations in `section-opt/`. RCDS means **robust conjugate-direction search**, as agreed; it is not conjugate gradient.

## Run

From the repository root, with Python 3.12:

```sh
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python model-svd_opt/requirements.lock
.venv/bin/python model-svd_opt/run_workflow.py \
  --output model-svd_opt/results/my_run --seed 42 --target-mm 1 1
```

The output directory must be new. Default tracking uses Cheetah, 10,000 particles and three independent shots per objective evaluation. Use `--backend xsuite` for Xsuite, `--particles 200000` for the full distribution, and `--optics-budget 100 --center-budget 60` for longer searches. Xsuite compiles local tracking kernels on first use. No wakefields are added. L4H is a **0.75 m drift**; the VBC remains powered at the matched family's 0.13169935 rad magnitude.

Several episodes, with independent machine seeds and log-uniform target sizes within the spline's 0.1–20 mm domain:

```sh
.venv/bin/python model-svd_opt/run_workflow.py \
  --output model-svd_opt/results/ensemble --episodes 10 --sample-targets
```

Without `--sample-targets`, all episodes use `--target-mm SX SY`; seeds still change. Target drawing uses a separate RNG and does not consume SimEnv's machine/noise stream. Each episode exports a nominal lattice with **all 46** interpolated quadrupole strengths, the VBC settings and L4H drift before creating SimEnv. The original fit was made with L4H powered; using its strengths with L4H off does not guarantee that its labels equal the measured sizes, which is why a measured matching stage is necessary.

The CLI module also exposes the reusable `Config` and `run` API without starting a run on import:

```python
import sys
sys.path.insert(0, "model-svd_opt")
from run_workflow import Config, run
run(Config(seed=43, target=(0.001, 0.0005)), "model-svd_opt/results/custom")
```

## Stages

1. **Nominal calibration and stochastic reset.** `optics_matching_vbc_on.MatchedOptics(l4h_off=True)` supplies strengths for the episode's requested sizes. Ideal horizontal and vertical orbit responses are evaluated directly with `orbit_matching.vbc_reference.ReferenceOrbitEngine` using the selected tracking backend. This also handles off-grid targets without using a stale response interpolation. `SimEnv.reset(seed=...)` supplies the per-episode beam uncertainty, quadrupole/sextupole misalignments and diagnostic offsets.
2. **Threading.** Keep the IP screen selected, so no upstream interceptive screen shadows downstream BPMs. Start with the contiguous valid BPM prefix plus the next missing BPM. Recompute up to three leading right-singular directions in **each** plane on that objective prefix. For a complete BPM prefix, use the full calibrated matrix including the IP row with squared-error weight five (row multiplier √5). Earlier prefixes use equally weighted BPM rows. Thus the allowed six steering combinations adapt to the loss location instead of relying on downstream correctors before beam gets there. With fewer resolved rows, use fewer nonzero modes.

   Within each fixed prefix, try the current setting, a restricted least-squares response correction from valid BPM readings, then RCDS. Missing BPMs incur a dominant penalty; their zero placeholders never count as position measurements. A prefix advances only after a strict majority of repeated shots confirms availability. Rebuild the objective and its modal basis after advancement. Reaching the IP also requires a majority-valid camera reading and a complete valid BPM prefix. Restore and remeasure the selected setting with fresh shots. Threading is budget limited and reports failure if the restricted steering subspace cannot recover beam. Optics and centering are then skipped.
3. **Measured size correction, two branches.** Calculate the fractional-beta response `d ln(beta_x,y) / d k1` about the episode's ideal optics at all physical screens including the IP. The IP's two plane rows together carry half the squared-error weight (¼ each); all other screen-plane rows share the other half equally. Centered finite differences at `h=1e-4 m^-2` and `h/2` are Richardson extrapolated. The saved step-convergence diagnostic makes this numerical choice inspectable. Beta transport is calculated separately from the bunch; this is **not** a beam-size derivative or a 200,000-particle response calculation. Cheetah uses double-precision symmetric reference probes; Xsuite uses `nominal_model.optics_along` and its tracked transfer map.

   Use the five dominant resolved right-singular quadrupole vectors as fixed affine controls. The **measured** objective is the mean shot-by-shot sum of squared logarithmic RMS-size errors, plus `2*(1-on_screen_fraction)`. The optimizer has no access to the episode's hidden coordinates, misalignments or jitter draws. BO calls `section-opt.section_measurement.full_bo_search`: joint GP search with EI followed by LCB. RCDS calls `section-opt.beam_tuning.rcds_search` with its parabola line search. Both receive the same objective-evaluation budget and repeated-shot count. RCDS receives the measured standard error; the existing BO interface uses scalar means rather than per-observation error bars.
4. **Final screen centering.** Re-evaluate the ideal orbit response at the quadrupole settings actually selected by the branch. Move only `cla_feh_mag_hvcor_01_h`, `_02_h`, `_01_v`, `_02_v`, leaving all quadrupoles fixed. Start with a minimum-norm response correction, then refine with RCDS and fresh measurements. These four correctors can use their complete native machine bounds. The centering objective is a smooth bounded function of squared normalized screen-centroid error plus the same on-screen loss penalty. A missing camera has a larger finite failure value, so killing the beam is never a cheap centering solution. Restore the best measured setting and verify it on fresh shots. The final readout may still miss tolerance; `completed` means the stages ran and the camera survived, not that an arbitrary requested spot was achieved exactly.

## Stochasticity, bounds and comparisons

Every measurement and restoration calls `env.step`. SimEnv draws a fresh beam each time with `beam_variation * beam_jitter`; power-supply ripple is also redrawn and restored by SimEnv. Defaults are beam variation 1, misalignment variation 1, beam jitter 0.1, fractional supply ripple 0.001, and BPM noise 0 m. All are CLI options. Repeated shots are averaged, with majority-valid BPM/camera masks and sample standard errors. No frozen bunch or hidden-state inverse is used.

Both optics branches reconstruct SimEnv from the same episode lattice and seed and replay the same complete threading action sequence. Reset shots and replay shots are logged and count toward the reported acquisition cost. The replay records maximum BPM differences against the discovery run and between the branch starts; it asserts identical availability masks and rejects a difference above 100 µm. Exact equality is reported only when observed. The per-step jitter draws remain live throughout each optimization; the methods may consume different numbers of shots after the shared start.

Controls use native `k1` in m⁻² and corrector angle in **radians**, never currents. The existing SimEnv computes limits using its ±100 A calibration, including its existing extension of ranges to include a lattice's nominal setting. This workflow does not enlarge those native limits. Affine modal boxes allocate each magnet's available headroom across modes, ensuring every optimizer proposal lies inside all bounds **without per-magnet clipping**. Degenerate dimensions are removed and recorded. Default threading amplitudes are capped at 2 mrad per mode and quadrupole amplitudes at 2 m⁻², further limited by response sensitivity and hardware headroom. These conservative boxes do not span the entire feasible native polytope.

`--thread-budget`, `--optics-budget` and `--center-budget` are counts of repeated-shot objective evaluations, not individual shots. Threading includes its restore/verification evaluations in its budget. Optics and centering each add one repeated-shot final verification outside their search budget. The model-response calculations use ideal reference tracking and are separate from measured shots. The environment's truncation limit includes headroom for all stages; every measured shot appears in `shots.jsonl` with its native environment evaluation counter. RCDS may spend many evaluations on one direction; a short smoke-test budget need not explore all five optics modes.

## Outputs and checks

Each episode saves:

- `config.json`, `episode_lattice.json`, `summary.json`: reproducible nominal inputs, target, fingerprint, threading progress, branch measurements, native settings and pairing checks.
- `shots.jsonl`: every reset, search, restoration and replay shot, its BPM values/mask, screen measurements, transmission, commands and readbacks.
- `nominal_orbit_{x,y}.npz`: response, weighted response, SVD and ordered diagnostic/corrector names. `optics_response.npz`: fractional-beta response, weights, SVD, rank and finite-difference check.
- `threading_*.json`, `{bo,rcds}_optics.json`, `{bo,rcds}_centering.json`: affine bases, safe boxes, actual evaluated coordinates, objective means/errors and measured results.
- `{bo,rcds}_final_optics_orbit_{x,y}.npz`: updated ideal response matrices and SVD at each branch's final quadrupole settings.
- `progress.png`: measured transmission, centroid radius, RMS sizes and contiguous valid BPM count. Invalid camera readings appear as gaps.

```sh
.venv/bin/python -m unittest discover -s model-svd_opt/tests -v
.venv/bin/python model-svd_opt/verify_run.py model-svd_opt/results/my_run
.venv/bin/python model-svd_opt/report.py model-svd_opt/results/my_run
```

`verify_run.py` checks modal membership of all measured quadrupole settings, the four-corrector restriction, changing readings on repeated stochastic shots, and every environment shot counter. Validation runs and numerical outcomes are recorded in `VALIDATION.md`. Neither apertures nor the camera field of view guarantee that all 144 target labels are feasible; loss, noise and residual size errors remain explicit results.

## Shared dependencies repaired in this checkout

The VBC-on provider module and L4H-drift lattice were absent, so their shared public interface was restored under `optics_matching_vbc_on/`. The drift lattice matches the previously recorded physical-input hash. The 144-node orbit responses for both backends were re-evaluated and compiled against the restored provider; calibration freshness checks remain enabled. The old `optics_matching.OrbitModes`/`OrbitResponse` import is retained lazily for compatibility. `section-opt/beam_tuning.py` now locates its bundled `rcds` directory relative to itself. All new workflow logic and outputs live here.

## 100-episode Cheetah export

`run_ensemble.py` runs independent episode seeds in worker processes and saves both optimizer branches. The current batch uses seeds 42–141 with a fixed (1, 1) mm target and the standard 10,000-particle / three-repeat budgets:

```sh
.venv/bin/python model-svd_opt/run_ensemble.py \
  --output model-svd_opt/results/cheetah_100_episodes --episodes 100 --workers 8
```

The output path must be new. `results.json` contains metadata, aggregate statistics, and the complete per-episode acquired-shot data. `metrics.json` is a smaller, periodically checkpointed export containing the per-episode/per-method scores, counts, timings, statuses and artifact paths. Each episode also has `episode.json`, its existing workflow artifacts and a log. Exceptions and beam-loss failures are retained as outcomes rather than replaced with different seeds.

`final_score` is the average **SimEnv benchmark score** over fresh final verification shots when verification completed; lower is better. `last_shot_score` preserves the original single-shot score. The benchmark includes the environment's evaluation-cost term. Size, position and transmission measurements are retained alongside scores. Search objectives in the stage JSON files are different quantities from this benchmark score.

Step counts include every `SimEnv.step`: repeated measurements, threading replay, restoration and verification. Reset shots are recorded separately. Episode totals include shared threading discovery and **both** optimizer branches; each branch also has its own counts and native environment step counter. A screen change means selecting a different physical screen, so repeatedly requesting the IP screen produces zero changes. This workflow deliberately leaves the IP screen selected throughout.

Times use `time.perf_counter()` in seconds. Individual shots, workflow setup/calibration, threading, each optimizer branch and worker episode elapsed times are saved. Batch elapsed wall time measures actual parallel runtime and is not the sum of worker times. Combined JSON export overhead is separately included in the smaller metrics file's final timing metadata. Particle screen images are not exported; BPM readings/masks, scalar screen measurements, commands, readbacks and scores are included for every acquired shot.

The acquired data can be re-exported without tracking again:

```sh
.venv/bin/python model-svd_opt/collect_ensemble.py model-svd_opt/results/cheetah_100_episodes
.venv/bin/python model-svd_opt/audit_ensemble.py model-svd_opt/results/cheetah_100_episodes
```

The first 100-episode batch required an exporter repair for NumPy metadata; this is recorded explicitly in the export metadata. No simulation seeds or measurements were replaced. Replay-check failures remain error outcomes. If BO finished before a later RCDS exception prevented a global workflow summary, its final verification score is reconstructed from its saved shot records and marked `recovered_from_saved_final_verification`. Overall branch wall time can be unavailable in that case; measured-shot wall time is still reported. Filter branch `status == "completed"` when comparing final scores; an error branch's last available reading is not a completed optimization result.
