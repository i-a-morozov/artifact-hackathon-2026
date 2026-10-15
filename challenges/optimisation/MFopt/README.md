# MFopt

Getting through a `SimEnv` episode (a new day on CLARA) to the requested spot at
`cla_feh_dia_scr_01` in at most 50 steps: steering first, then multi-fidelity
Bayesian optimisation between the model and the machine. The plan is in the
project notes (`mfopt/plan.md`).

## How an episode is solved

The workflow, from `reset` to the score, is drawn in
[`talk_figures/mfopt/mfopt_workflow.svg`](../talk_figures/mfopt/mfopt_workflow.svg).
Purple boxes run in the error-free twin and cost no step; every other box is a shot
on the machine, one step each.

1. **Model, free.** The request (sigma_x, sigma_y) picks the target optics from the
   VBC-on node table, re-matched in the twin (`optics_env.py`), and the orbit
   responses at that optics from the precomputed table (`responses.py`).
2. **Stage A: optics and linear steering** (`steering.py`, about 5 steps). Load the
   optics. If the beam is lost part way, thread it: SVD steering on the BPMs that
   read, with R cropped to them and the correctors upstream of them, until every
   BPM reads. Then two SVD shots of the orbit onto the twin's orbit, and up to three
   Newton shots of the last two correctors (FEH HVCOR 01 and 02) to put the centroid
   on the request.
3. **Stage D: MF-BO of the spot size, linear steering of its position**
   (`mfbo.py`). The BO searches only the knobs that shape the spot: the five
   quadrupoles closest to the IP (FEA Q14, Q15, Q16, FEH Q1, Q2) and the two FEA
   sextupoles, about their stage A values. Position is left out of the search
   because it is linear in the correctors: on every BO shot the FEH correctors are
   set, from their known linear response, to cancel the centroid error the GP
   predicts for that quadrupole setting, and that known kick is taken back out of
   the measured centroid before it reaches the GP. They come after the last BPM, so
   they move the spot and nothing upstream.
   - Low fidelity: 256 Sobol points of the twin over the 7 knobs, in one batched
     Cheetah track, with a GP per readout (mu_x, mu_y, sigma_x, sigma_y, T).
   - High fidelity: the machine, modelled as the low-fidelity GP plus a discrepancy
     (a constant offset after the first reading, a GP from the third).
   - Each shot maximises the expected improvement of the benchmark mismatch,
     computed from posterior samples. The loop stops when that improvement is worth
     less than the score cost of one more step, and the episode ends by returning
     to the best reading.
4. **Score.** The benchmark score of the last reading of FEH-SCR-01: the mismatch in
   requested spot sizes plus 2 (1 - T), capped at 10, plus 0.5 log2(1 + steps / 50).

## Model settings

The environment and the twin are `SimEnv` defaults on master: Cheetah, every
drift, quadrupole and dipole at second order with second-order dipole edges (the
lattice file), VBC on at 131.7 mrad with edges, apertures on, L4H off (tracked as a
drift of its own length, `RLConfig.l4h_off`, as in `orbit_matching/`), and every
magnet limited to what +-100 A reaches.

## Target optics: `optics_env.py`

```python
from MFopt.optics_env import OpticsEnv

env = OpticsEnv()                        # a SimEnv; episodes are unchanged
optics = env.optics_for_target(150e-6, 250e-6)
k1, quad_sv, corrector_sv = optics.vectors

optics.quadrupoles.ranked(4)             # quadrupoles by their weight in the modes
optics.correctors.ranked(6)              # corrector planes, `_h` and `_v`
env.step(env.optics_action(optics))      # load the optics (one step)
```

- **Optics.** Starts from the 144 nodes rematched with the VBC on at 131.7 mrad
  (`optics_matching_vbc_on/`), interpolated in log size. Those were fitted with L4H
  powered and no magnet limits, so the four final-focus quadrupoles (FEA Q15/Q16,
  FEH Q1/Q2) are re-matched to the request in an error-free copy of the environment
  (see Model settings), within the magnet limits. Below about 0.13 mm in x the
  nodes need FEH Q2 above its limit (12.5 m^-2 at 100 A; the nodes go to 17.6), and
  no re-match within limits gets lower, even with all 12 downstream quadrupoles or
  the sextupoles free.
  `optics.achieved`, `optics.transmission` and `optics.clipped` say how well that went.
- **Quadrupole modes.** SVD of d ln(rms size) / d k1 at every screen and the readout,
  for all 46 quadrupoles. Rows and weights follow `SizeModes` (the readout carries
  half of the squared weight), but it is the beam-size response in this environment's
  model, so dispersion and the VBC are included.
- **Corrector modes.** The same for the centroid against all 58 corrector planes.
- **Scores.** An SVD has one singular value per mode, so each magnet also gets
  `scores[name] = sqrt(sum_i s_i^2 V_ij^2)` over the modes holding 99 % of the squared
  response. `ranked()` sorts by it.

Everything runs on the error-free model with the nominal beam: no environment steps.
One target takes 1.5 to 2.5 minutes (re-match plus 208 finite-difference tracks), and
results are cached per target.

## Steering responses and MF-BO knobs per spot: `responses.py`

```python
from MFopt.responses import DOF, ResponseTable

table = ResponseTable()                  # MFopt/data/responses_vbc_on.npz
at = table.at(150e-6, 250e-6)            # bilinear in log size between nodes
at["R"]      # centroid response: BPM x, BPM y, screen x, y  x  all 58 corrector planes
at["orbit"]  # the model's centroids at the same rows: what steering aims for
at["J"]      # readout (mu_x, mu_y, sigma_x, sigma_y, T)  x  DOF
```

- **MF-BO knobs (`DOF`).** The five quadrupoles closest to the IP (FEA Q14, Q15,
  Q16, FEH Q1, Q2), the five correctors per plane closest to it (FEA HVCOR 04, 05,
  06, FEH HVCOR 01, 02) and the two FEA sextupoles.
- **Table.** Built at every node of the VBC-on grid after the final-focus re-match,
  in the error-free twin: `k1`, `readout`, `clipped`, `feasible`
  (T >= 95 % and both sizes within 5 %), `relative_error`.
- **Twin.** The error-free twin tracks 5,000 particles (`OpticsEnv(twin_particles=...)`),
  half of the environment's 10,000, which keeps the rms sizes to about 1 %.
- **Rebuild.** `python -m MFopt.responses --workers 16` from `challenges/optimisation`,
  about 4 minutes on 16 cores. 105 of the 144 nodes are feasible (T >= 95 %, both
  sizes within 5 %). Inside the benchmark's 80 to 400 um range the misses are the
  nodes with sigma_x <= 0.162 mm, which need FEH Q2 beyond its limit (see above).
- **Check against `orbit_matching/`.** At the 105 feasible nodes our orbit response
  R agrees with Eiad's Cheetah responses (same BPMs and correctors, L4H a drift) to
  3.6 % median and 11 % at the 90th percentile (relative Frobenius norm). The rest
  is our re-matched final focus.

Stage A steering (`steering.py`) reads the orbit and spot responses from this table
instead of tracking them per episode.

## Running episodes: `run_baselines.py`

```bash
python -m MFopt.run_baselines --seeds 30 --workers 16
python -m MFopt.run_baselines --misalignment 5 --seeds 12 --tag lost_beam --policies optics_and_steering mfbo
```

One process per episode. Policies: `do_nothing`, `optics_only`, `optics_and_steering`
(stage A) and `mfbo` (stage A, then MF-BO in `mfbo.py`). `--misalignment 5` gives
days on which the beam is lost part way along the line; stage A then threads it
(`steering.thread`). Results go to `MFopt/results/<policy>[_<tag>].json`. MF-BO needs
`botorch`.

## Results (15 Oct, second order, L4H off, VBC 131.7 mrad)

| Days | Policy | Mean score | Median | Within 10 % | Within 25 % | Steps |
|---|---|---|---|---|---|---|
| 30 default (seeds 1000-1029) | do nothing | 9.30 | 10.00 | 0 % | 0 % | 0 |
| | optics only | 6.99 | 7.05 | 0 % | 0 % | 1 |
| | stage A | 0.59 | 0.47 | 17 % | 67 % | 4.7 |
| | stage A + MF-BO | 0.41 | 0.36 | 53 % | 87 % | 7.9 |
| 12 lost-beam (misalignment x5, seeds 1000-1011) | stage A | 0.62 | 0.65 | 0 % | 42 % | 6.2 |
| | stage A + MF-BO | 0.49 | 0.49 | 33 % | 83 % | 13.2 |

Every episode ends within 50 steps (longest 20) with the beam on the screen
(transmission at least 85 %). MF-BO improves on stage A by 0.18 +- 0.04 per day
(paired over the 30 days).

Two things limit it on this model, both worse than on the linear lattice it was
first tested on (mean 0.32, 77 % within 10 %):

- **Shot-to-shot jitter.** At fixed settings the readout sigma_x moves by about
  0.15 in ln(sigma) rms (175 to 270 um on a 200 um spot, seed 1000), because the
  launch jitter now reaches the spot through second-order chromatic terms. The
  score is a single reading, so this sets a floor of roughly 0.2 to 0.3. Treating
  the sizes as ln(sigma) in the GPs and ranking settings by posterior mean instead of
  by the best reading made no significant difference (+0.02 +- 0.02 and +0.04 +-
  0.03), so `mfbo.py` keeps the simpler scheme.
- **Small horizontal spots.** Requests below about 0.13 mm in x need FEH Q2 beyond
  its limit, so those days end with sigma_x too large.

## Log: energy discrepancy at the VBC

The lattice file's cavities bring the reference beam to 181.6 MeV at the VBC, while
`data/clara/lattice_data/utils.py` converts S05, VBC and S06 magnet currents at
`beam_energy` 196.6 MeV, about 8 % higher (see `nominal_model/README.md`). It does not
matter for MFopt: every knob here is set in model units (k1, k2, kick angle) and the
responses come from the same model, so no current is converted anywhere. It would
matter only when the settings are turned into currents for the real machine, where
the S05 to S06 magnets would come out about 8 % strong. Logged, not corrected.

## Batched Cheetah tracks: `batched.py`

`probe_batch(env, {knob: values}, base)` tracks N settings of the error-free twin in
one vectorised Cheetah pass and returns the same readings as N calls to `_probe`
(they agree to about 1e-7 relative). It runs on a private deep copy of the twin's
segment, so the twin that `_probe` and the re-match use is never left batched. The
MF-BO low fidelity (256 points) and every finite difference in `responses.py` go
through it: on the linear lattice a table node took 3 s instead of 15 s, the low-fidelity sample 1.5 s
instead of 10 s.
