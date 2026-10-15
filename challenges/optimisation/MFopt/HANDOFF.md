# MFopt handoff to the Mac (15 Oct 2026, 14:50)

Branch: `claude/mfopt-harness` on the fork InnovateInfinity87/artifact-hackathon-2026 (pushed, includes fork master with VBC 131.7 mrad and the rematched optics). Code in `challenges/optimisation/MFopt/`; plan in `/mnt/project-files/mfopt/plan.md`. Commits as Francesco, no AI attribution.

## Done
- `optics_env.py`: `optics_for_target` starts from `optics_matching_vbc_on/results/rematched_grid_vbc_on.npz` (bilinear in log size) and re-matches FEA Q15/Q16, FEH Q1/Q2 in the first-order env twin (the second-order rematch is up to ~30 % off in the scored first-order env).
- `responses.py`: per node (12x12) -> `k1`, model `orbit` (BPM x, BPM y, screen x,y), `R` (those rows x 58 corrector planes), `J` (readout mu_x, mu_y, sigma_x, sigma_y, T x 17 DOF), `feasible`. DOF (Francesco): FEA Q14/15/16, FEH Q1/Q2; FEA hvcor 04/05/06 + FEH hvcor 01/02 both planes; FEA sext 01/02. `ResponseTable.at(sx, sy)`. Sextupoles barely act in the first-order env (~2 % in sigma_y at full k2).
- `steering.py`: `thread()` crops R to BPMs with a reading and correctors upstream of them, iterates, undoes a shot that loses ground and lowers rcond; stage A (`optics_and_steering`) reads the table by default (`table=False` tracks instead).
- `mfbo.py` (draft, untested): low fidelity = GP on 256 Sobol model points over 5 quads + 2 sextupoles; high = model GP + discrepancy (constant offset, GP from 3 points); FEH correctors cancel the predicted centroid each shot; EI on the benchmark mismatch from posterior samples; stops when EI < cost of one more step, ends on the best reading.

## Done on the Mac (15 Oct, 17:00)
- Response table rebuilt with a 5,000-particle twin on 16 cores (3 minutes): `MFopt/data/responses_vbc_on.npz`, 118/144 nodes feasible.
- MF-BO fixed (it set torch's default dtype to float64, which broke the float32 tracking in the same process) and run against stage A: see the results table in `README.md`. 30 default days: mean 0.32 vs 0.49, 77 % vs 30 % within 10 %. 12 lost-beam days: 0.37 vs 0.48, 75 % vs 33 %.
- `run_baselines.py` runs one process per episode (`--workers`), with `--misalignment` for lost-beam days (replaces `test_threading.py`).
