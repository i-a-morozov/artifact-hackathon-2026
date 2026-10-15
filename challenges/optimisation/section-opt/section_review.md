# Section correction review — 2026-10-15

Reviewed `section.ipynb`, its builder, the response/RL helpers, and the shared measurement and optimizer interfaces. All changes are under `work/`.

## Findings and resolutions

| Finding | Resolution |
| --- | --- |
| Full BO was absent; the BO label previously meant only one-dimensional GP searches inside RCDS. | Added joint GP/EI BO over all normalized section knobs with a counted budget and a measured starting point. |
| RL was pretrained on multiple beams only. Adaptation on one beam did not provide an independent single-beam training comparison. | Added separate one-beam and several-beam policies, with and without a physics action prior. Each has equal pretraining and adaptation budgets. |
| Missing or nearly lost beams could propagate NaNs through PPO and the physics least-squares solve; finite surviving moments alone did not ensure adequate survival. | Added a local survival-aware measurement session, finite failure cost, valid-reading flag, bounded reward on failed readings, and zero physics action for a missing reading. |
| Physics weight zero still called the physics solver. | Ordinary PPO now skips that solver. Regression test verifies independence. |
| The noise standard error did not reflect the individual scales of the four residual components. | Use sigma * norm(1 / scale) / 4, the propagated estimate away from absolute-value cusps. |
| Full-budget rankings compared 32 classical measurements against 9 PPO measurements, and response used only 2. | Retained those explicit costs and added a common maximum deployment-budget table using measured selections from each trajectory prefix. |
| Changing the input screen could silently keep the fixed-energy kickoff beam at an inappropriate location. | Automatic beam-source selection; explicitly reject kickoff beams at any entrance other than the S04 kickoff marker. |
| Ideal kickoff samples depended on ambient Torch RNG state. | Generate them in a seeded isolated RNG context. |
| Response scans could leave a probe setting applied after an exception. | Restore nominal controls in a finally block. |
| Input covariance diagnostics included particles lost upstream. | Compute survival-weighted centers and covariance. |
| Saved policies lacked the normalization/model context needed for reuse. | Store knob identities/bounds, target/scale, response, start, horizon, step, noise and training pool size with the adapted policy. |
| The default executed two unrelated sections. | Run the named primary section only by default; retain the downstream example as an explicit option. |

## Interpretation limits

- The optimized observable is `(mu_x, mu_y, sigma_x, sigma_y)` at the exit screen. It matches orbit position and projected spot sizes. It does **not** identify full Twiss parameters, angles, emittances, coupling, or the entire sigma matrix from one image. The notebook now reports simulator-only angle and covariance errors separately.
- The SVD rows diagonalize the ideal local `J.T @ J`. They are not guaranteed conjugate directions for the actual L1 objective or for a changed input beam.
- The local readout uses surviving-particle moments and a 10% survival threshold, with optional moment noise. It does not implement finite camera clipping. It is distinct from the official whole-lattice FEH scorer.
- Every policy starts from the same normalized setting for its section. Transfer is between beams, not between arbitrary starting machine settings. Adaptation is emulated, with no hardware connection.
- Simulator truth is used only after measured selection for diagnostics. Extra diagnostic tracks and ideal-response simulations are not online screen shots. Pretraining, adaptation, and deployment costs remain separate.
- The matched-budget classical results are prefixes of the full-budget trajectories; they do not retune optimizer hyperparameters for a shorter horizon. Response now uses one initial image plus one image per configured correction step.
- The current user configuration sets `READOUT_NOISE=0.05`; the earlier benchmark below used zero noise. Four held-out beams and one policy initialization are an exploratory comparison, not a statistical performance claim.

## Validation

- 23 regression tests passed, including full-BO budgeting, lost-beam PPO behavior, zero-weight physics independence, response-scan restoration, covariance weighting, and existing shared optimizer tests.
- All methods completed a smoke execution on both S04 and S06–S07.

- Full default S04 execution completed: 24 cells, 14 executed code cells, no errors; six classical methods and four independently trained PPO variants (before/after adaptation).
- On four held-out S04 beams, the nine-image comparison gave one-beam PPO 0.26395, response correction 0.26794 (two images), response-direction RCDS 0.27871, and ordinary RCDS 0.28265. Ordinary RCDS reached 0.09065 at 32 images. These are local screen objectives, not the official FEH score.

## Acquisition setting update

Full BO now supports EI followed by the minimization confidence bound (skopt LCB), with an explicit switch index and beta=1.0. Pure EI and pure confidence-bound modes remain selectable. RCDS reuses its existing LogEI/UCB implementation with the optional `ei_first` flag; the package default remains UCB first, while section.ipynb enables EI first. Three new schedule regressions and eight relevant existing tests passed.

## Iterative response correction and bounded line search

Response correction now keeps the fixed ideal-input response matrix and repeatedly corrects the measured exit residual, applying `u_next = u + 0.1 * (candidate - u)` in normalized controls. `RESPONSE_STEPS` controls the update count; its default is the measurement budget minus the initial image. The returned setting is the best measured iterate. No actual input-beam measurements enter the response model.

The reported `WRAPPER: bounds mismatch` arose when a mixed-sign search direction at a box corner had no feasible line interval. GP and parabola line searches now skip such directions without a measurement. Full BO also preserves its random generator when switching acquisitions, avoiding a restart of the candidate sequence.

Validation: 32 regression tests passed. A noisy smoke notebook completed on S04 and S06–S07, including all methods. With the preserved user settings (100,000 particles, noise 0.05, 128 measurements), the first S04 beam's response correction reduced the diagnostic noiseless local objective from 1.57416 to 0.01024; all 127 updates were checked against the requested learning rate. The large configured PPO training run was not repeated. Earlier benchmark figures above describe the previous configuration.

## Direction review and full-machine workflow

### Validation against the saved five-beam run

Using the same seed 101, 100,000 particles, noise 0.05, 128 images, and unchanged line-search
settings, the mean clean local objective was:

| Method | Unit directions (saved run) | Raw SVD (saved run) | Boundary-aware directions (rerun) |
|---|---:|---:|---:|
| RCDS | 0.12561 | 0.66387 | 0.12187 |
| RCDS+BO | 0.11923 | 1.75191 | 0.11376 |

The direction failure is largely resolved; the small remaining advantage over coordinate axes
is not evidence of a robust performance gain from five beams alone. Response correction remains
best in this comparison (0.01738). It uses the vector residual and a model-based correction on
every image, whereas RCDS sees a scalar objective. Reduced line budgets or a response warm start
are separate experiments; the line-search settings above were preserved.

The executed notebook before this review is saved in
`work/experiment_outputs/section_before_direction_review.ipynb`. PPO training in that run was
interrupted; no completed production PPO comparison is inferred from it. The updated notebook
has cleared outputs so old results cannot be mistaken for results from the new direction mode.

All TODOs in section.ipynb were replaced: screen listing; explicit objective labels and target/initial/final moments; beam images; and a persistent, selected-method forward correction pass. Windows use the nearest upstream screen with at least four quadrupoles and four corrector angles, or all available controls at the entrance when insufficient. Overlapping windows recompute the true incoming simulator state from the root beam. Final audits revisit all screens to reveal any degradation from overlaps.

Validation: 37 regression tests passed, including the skipped-target test. The complete notebook passed a small noisy run, including five sequential windows. A full forward pass with 750 particles and six images per valid screen considered 27 screens: 24 had valid ideal models (144 online images), and three had no valid ideal target. The final survival in that deliberately short-budget run was 1.2%; execution success does not imply successful full-machine correction. Large-budget full-lattice correction and production PPO training were not rerun.

All six selectable classical methods also completed five persistent correction windows with exactly 30 online images each in the short-budget integration check.

## Reduced-SVD direction fix

The boundary-aware basis now computes a fresh SVD of the ideal response restricted to unsaturated knobs. It embeds those vectors back into the full normalized knob space and orders them as observable free modes, saturated coordinate axes, then free null-space modes. It does not project the old vectors or complete them with overlapping free coordinate axes. The full basis is orthonormal (S04 condition number 1, formerly 13.68), and the free modes diagonalize the reduced Gauss–Newton matrix. Saturated coordinate axes are not generally conjugate to the free modes under the full matrix; the actual L1 objective also has no guaranteed quadratic conjugacy.

The package Hessian helper now uses symmetric `eigh` and returns eigenvectors as rows, matching `rcds` consumption. Tests check an independent non-diagonal quadratic and the actual vectors consumed by the line-search callback. Raw and cleaned legacy SVD modes remain available.

41 regression tests passed, including free-subspace conjugacy, basis completeness, interior equivalence to the raw SVD, zero response, and the Hessian helper orientation. An audit confirmed the wrapper passes the selected rows unchanged. `section.ipynb` was not edited.

## RL runtime improvements

Profiling at 100,000 particles attributed most rollout time to transport, with redundant screen tracking/clones taking about 38% of the measured loop. RL now reuses a moment-only section tracker that omits nonblocking, zero-length diagnostic screens while retaining magnets, apertures and active blocking screens. Camera caches are not populated by this path; the local objective already reads particle moments. `RL_FAST_TRACKING=False` restores the original path.

The fixed response inverse and ridge matrix are cached. Feasible unconstrained physics-prior solutions use that inverse; constrained solutions use dense BVLS, tested against the original bounded least-squares solver. Measurement budgets and noisy observation generation are unchanged.

`RL_TRAIN_PARTICLES=min(NUM_PARTICLES,4096)` separates simulation pretraining resolution from adaptation and held-out evaluation, which retain NUM_PARTICLES. This is a deliberate finite-particle approximation; set it to NUM_PARTICLES for matching resolution. Iterations, episodes, horizons and policy variants are unchanged. Training reports progress and ETA, and histories include rollout, update and elapsed seconds. Checkpoints record pretraining/evaluation particle counts and the tracking flag.

Validation: 45 regression tests passed. A short complete PPO benchmark (3 iterations, 4 episodes each, horizon 8, 108 counted images) took 4.02 s for ordinary PPO with original tracking/100,000 particles and 0.64 s with fast tracking/4,096 particles; physics PPO took 3.47 s and 0.77 s respectively. Both sides of this latter benchmark used the new physics solver. These are short local timing results, not guarantees for full training or evidence of equal learned-policy quality. The original-versus-final profiled physics rollout was 1.59 s versus 0.33 s. Production-length training was not rerun.

## Full correction progress and transmission reports

`FULL_VERBOSE=True` and `FULL_LOG_EVERY=16` enable section-start messages, periodic measured-objective progress, completion summaries, and final-audit/end-of-lattice transmission. Per-window records now include incoming transmission, exit transmission before/after correction, within-window transmission before/after, elapsed seconds and image counts. Transmission is surviving particle weight relative to the original root beam or the current window entrance, respectively; a zero entrance weight yields NaN for the conditional ratio. This is simulation survival information, not a new hardware diagnostic.

The final audit recomputes cumulative and within-window transmission after all overlapping corrections, including screens without a valid ideal target. CSV exports are `full_correction_sections.csv` and `full_correction_final_audit.csv`; the notebook displays percent columns and plots final versus initial transmission. Extra reports do not consume optimizer image budgets.

Validation: 16 relevant tests passed, including denominator checks, progress output and unchanged measurement counts. The full reporting cell completed a 27-screen smoke-resolution run with 144 counted images and valid transmission fields for all screens, including the three skipped ideal targets.

## New nominal workflow and remaining final-section TODOs

The final section now loads the recommended nominal lattice-file tune or the explicitly selected GPSR scan tune with the detailed `data/clara/lattice_data/apertures.json` vacuum chambers. `work.nominal_machine` provides a Cheetah-only loader because xsuite/xobjects are not installed in the current environment. It mirrors the nominal-model chamber insertion logic without modifying nominal_model or score.py. The new correlated `nominal_model/initial_distribution.npz` is sampled directly in Cheetah coordinates; the jitter model and assumed 250 pC charge are documented. Reference and seeded error-bearing machines are independent.

Added per-section objective curves with the restored best-measured point, orbit/envelope and size plots with actual aperture checks, initial/corrected/design exit distributions (ellipse in physical units, circle in normalized units), and a second pass that retains the previously corrected lattice and input beam. Reports include the selected evaluation index and best measured objective.

The scorer now starts at the S02 marker rather than after S02-SCR-01. Updated boundary detection and the legacy assertions accordingly; this avoids a duplicate first-screen interval and correctly propagates to the first physical screen.

Validation: 50 tests passed, including nominal particle/correlation preservation, narrow apertures, seeded setting/misalignment errors, best-setting commitment and second-pass continuity. The final notebook cells completed five windows in each of two passes using RCDS+BO, 750 particles and six images per window. Production RCDS+BO and 512-image settings were preserved. The complete production-budget run was not executed.

## Section and RL migration to the new nominal model

The isolated-section comparisons and every PPO phase now share `load_section_lattice()` and `sample_section_input()`. Default `SECTION_MODEL=nominal` uses the lattice-file tune and the detailed aperture table; auto beam selection uses the new correlated nominal distribution propagated from S02 to the selected entrance. Ideal targets and response matrices use zero-jitter nominal input; test and training beams use explicitly configured injection jitter. The plant retains the existing normalized setting perturbation (hidden misalignments are confined to the separate full-machine experiment). Legacy scorer/GPSR optics and explicit kickoff/score beam sources remain selectable.

New response/checkpoint artifacts are separated under `experiment_outputs/section_models`, and checkpoints include model, aperture, and beam-source metadata. Historical direction-comparison numbers are labelled as old-model results. User settings (16 production beams, noise 0.025, measurement and PPO budgets) were preserved.

Validation: the complete notebook executed with smoke budgets on S04 and S06–S07, with all classical methods, eight independent policy pretraining runs, eight adaptations, held-out transfer, and the two-pass full-correction cells; no error outputs. Production-length training was not rerun.
