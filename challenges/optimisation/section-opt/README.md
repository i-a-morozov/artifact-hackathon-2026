# Section correction

[section.ipynb](section.ipynb) compares response-matrix correction, full BO, RCDS/RCDS+BO (with unit or model based initial directions), and RL-PPO. It also corrects a machine screen by screen.

## Model and errors

The current notebook selects `SECTION_MODEL="nominal"` and `FULL_MODEL="nominal"`:

- **Optics:** `../data/clara/lattice_data/CLARA_cheetah.json`, loaded directly with its stored strengths, as recommended by the new nominal-model README. The scorer's reference tune is not applied.
- **Apertures:** `../data/clara/lattice_data/apertures.json`, enabled in tracking and shown on envelope plots.
- **Input beam:** `../nominal_model/initial_distribution.npz`, containing correlated 6D particles at the S02 marker. These are propagated to the selected section entrance.
- Select `"gpsr"` explicitly to use `../nominal_model/CLARA_cheetah_S07_GPSR.json` (the reconstruction scan tune), or `"scorer"` for the legacy benchmark loader.

| Experiment | Errors applied to the machine |
|---|---|
| Isolated section and RL | Corrector/quad settings start with seeded Gaussian perturbations in normalized coordinates, clipped to [0,1]. `START_ERROR_FRACTION=0.05` is the standard deviation as a fraction of the allowed range. No hidden magnet misalignments are added. |
| Full correction | A separate `errors=True` machine gets setting perturbations plus quadrupole/sextupole misalignments (50 µm RMS per plane at severity 1) and a readout-screen offset (30 µm RMS). The reference uses `errors=False`. |

Varied input beams also receive injection jitter (`SECTION_BEAM_SEVERITY` / `FULL_BEAM_SEVERITY`). Targets and response matrices use the unperturbed reference and nominal input. `READOUT_NOISE` controls observation noise; it is currently **0.0**. The local objective reads beam moments, so screen-coordinate offsets do not affect it as they would a camera image.

The objective matches exit position and projected sizes, not full Twiss parameters or the official scorer metric. The best **measured** setting is restored before proceeding. Reports distinguish cumulative transmission from transmission through each section; later overlapping corrections can change earlier results.

## Files

- `nominal_machine.py`: model, aperture, beam and error loading.
- `section_response.py`, `section_measurement.py`: ideal responses, directions, objectives and measurement budgets.
- `section_sequence.py`, `section_diagnostics.py`: sequential correction, reports and plots.
- `section_rl.py`, `section_tracking.py`, `rl_tuning.py`: PPO environments, fast transport and training.
- `beam_tuning.py`, `beamline_tuning.py`, `rcds/`: shared optimizer and beamline dependencies.
- `test_*.py`: regression tests. `section_review.md`: investigation history.
- `experiment_outputs/`: generated response matrices, checkpoints and CSV reports.

## Running

Open the notebook from this directory or the optimisation project root. Run setup first, then the desired workflow. RL pretraining uses `RL_TRAIN_PARTICLES`; adaptation/evaluation use `NUM_PARTICLES`. Full-correction cells start fresh machines; the dedicated second-pass cell continues from the most recently corrected machine.

From the optimisation project root:

```bash
python -m unittest discover -s work -p 'test_*.py'
OPTIMISATION_SMOKE=1 jupyter nbconvert --to notebook --execute work/section.ipynb --output /tmp/section-smoke.ipynb
```

Additional dependencies are listed in `requirements-experiments.txt`. `build_section_notebook.py` is a generation script, but currently differs from the manually edited notebook: synchronize it before using it, because running it overwrites `section.ipynb`.
