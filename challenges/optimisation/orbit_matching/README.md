# CLARA orbit controls: VBC on, L4H replaced by a drift

The current controls use all 144 quadrupole settings from
[`optics_matching_vbc_on/results/rematched_grid_vbc_on.npz`](../optics_matching_vbc_on/results/rematched_grid_vbc_on.npz),
the new vacuum-aperture table, and a **131.69935 mrad VBC bend magnitude**.
**L4H is entirely replaced by a 0.75 m drift for the orbit responses and SVDs.**
The four bend angles are `(-theta,+theta,+theta,-theta)`, including the moving
inner dipole-edge angles. No wakefields, errors, jitter, ripple or current
clipping are added.

The new grid varies 30 downstream quadrupoles. All 46 settings are applied;
the former fixed-42-prefix/four-variable response model is no longer used.
The source grid was rematched with L4H powered. **Its quadrupoles are not
rematched again when L4H is removed.** Target sizes label the requested settings;
they are not assertions that those RMS sizes are achieved with L4H off.

## Current status — 15 October 2026

The VBC-on update is complete. The active runtime calibration is **schema 2,
L4H drift**, with separate Cheetah and Xsuite controls. Its saved configuration
identity matches the current optics provider.

| Deliverable | Status | Output |
|---|---|---|
| Transmission with supplied, powered L4H | Complete: 144 nodes per backend, 200,000 particles per node | [Comparison plot](results/vbc_on/transmission_comparison.png) |
| Separate transmission with L4H a drift | Complete: 144 nodes per backend, same bunch and quadrupoles | [Comparison plot](results/vbc_on_l4h_off/transmission_comparison.png) |
| L4H-drift horizontal and vertical orbit matrices | Complete for all 144 nodes in both backends | [Matrices and CSV exports](results/vbc_on_l4h_off/orbit/) |
| Weighted right-singular vectors and singular values | Complete: IP weight 5, each BPM weight 1 | [SVD comparison](results/vbc_on_l4h_off/orbit/orbit_svd_comparison.png) |
| Numerical and interface checks | Passed: all 576 backend/plane/node combinations and six compatibility tests | [API checks](results/vbc_on_l4h_off/orbit/api_validation.json), [compatibility checks](results/vbc_on_l4h_off/orbit/compatibility_validation.json) |

The operational resolved mode counts are **26 horizontal and 27 vertical**.
Transmission medians are approximately 98.95%, but three nodes in each
backend/cavity case transmit less than 10%. Readout-validity flags accompany
the current controls.

The remaining scope limits are unchanged: removing L4H does not trigger a
new quadrupole rematch; orbit matrices describe the linear reference orbit;
and off-grid transmission and achieved target sizes have not been validated.
The controls therefore do not establish transmission-aware centroid correction
for strongly scraped bunches. Details, data layouts and regeneration commands
follow below.

## Full-bunch transmission: both lattices and both L4H cases

Each of the 144 nodes is tracked with the **same complete 200,000-particle
nominal bunch**, separately in Cheetah and Xsuite, for each cavity case:

| Cavity case | Backend | Minimum transmission | Median transmission | Nodes with T >= 95% |
|---|---|---:|---:|---:|
| Supplied lattice, L4H powered | Cheetah | 2.6855% | 98.9455% | 127 |
| Supplied lattice, L4H powered | Xsuite | 2.6985% | 98.94525% | 123 |
| L4H replaced by a drift | Cheetah | 2.6505% | 98.9515% | 125 |
| L4H replaced by a drift | Xsuite | 2.6735% | 98.95275% | 123 |

The largest backend difference at a node is **1.062 percentage points** with
L4H powered and **0.964 percentage points** with L4H a drift. Removing L4H
changes individual-node transmission by up to 1.9215 percentage points in
Cheetah or 1.976 percentage points in Xsuite. Each case has three nodes below
10% transmission in each backend.

- [Powered-L4H transmission plot](results/vbc_on/transmission_comparison.png)
  ([PDF](results/vbc_on/transmission_comparison.pdf),
  [all 144 rows as CSV](results/vbc_on/transmission_nodes.csv)).
- [Separate L4H-drift transmission plot](results/vbc_on_l4h_off/transmission_comparison.png)
  ([PDF](results/vbc_on_l4h_off/transmission_comparison.pdf),
  [all 144 rows as CSV](results/vbc_on_l4h_off/transmission_nodes.csv)).

Each plot shows Cheetah, Xsuite and their difference; every node is annotated.
Array axes are `[sigma_x index, sigma_y index]`, using the 12 logarithmic
requested sizes from 0.1 to 20 mm in each plane.

### What is counted

Transmission is the fraction of launched particles alive at
`cla_feh_dia_scr_01`, **before camera field-of-view selection**. The separate
`on_screen_fraction` counts the fraction on the chip. No screen intercepts
the beam upstream. Sizes in `screen_stats` are about the surviving on-chip
centroid; they are zero with `valid=False` when fewer than 10% of the launched
particles illuminate the chip, following SimEnv's readout convention.

Both transports use `nominal_model.with_apertures` and
[`data/clara/lattice_data/apertures.json`](../data/clara/lattice_data/apertures.json).
The chamber is checked at the supplied markers and exits of finite-length
elements; local collimators are checked at entrance and exit. These are hard
particle losses, not the older 3-sigma envelope constraint, and are not a
continuous aperture check inside every magnet or RF cell. The framework's
additional **50 mm circular radius at diagnostic stops** is also retained,
even where the local chamber is larger. This reproduces the transmission
convention used by the new rematch scripts.

Cheetah uses the supplied second-order lattice, second-order dipole edges and
the framework's float32 tracking. Xsuite uses the current converter's thick
chromatic elements and standing-wave RF integration. Its bend-edge treatment
is the converter's native model, rather than a promise of identical higher-order
physics. The geometry, hard apertures, nominal bunch, VBC setting and requested
quadrupole strengths are shared. The identical pre-VBC bunch transport is
cached and replayed in Xsuite; exact replay was checked against full-line
tracking at two settings (see
[the cache check](results/vbc_on/prefix_cache_validation.json)).

## Per-node orbit response matrices and weighted SVDs

The orbit update is **VBC on / L4H drift only**, independently evaluated in both
backends. Each plane has 31 BPMs and the additional IP position row:

| Plane | Response shape, per node | Resolved modes | Modes for 99% squared sensitivity across the grid |
|---|---|---:|---|
| Horizontal | 32 x 28 | 26 | 1–6 |
| Vertical | 32 x 29 | 27 | 1–5 |

The separate VBC vertical corrector is included only in the vertical matrix.
The matrices and their weighting are

```
R_x[i,j] = d x_i / d theta_x_j
R_y[i,j] = d y_i / d theta_y_j
A = sqrt(row_weights)[:,None] * R
A = U @ diag(singular_values) @ Vt
row_weights = [1, ..., 1, 5]
```

Each IP row has five times an individual BPM's **squared-error weight**:
its matrix-row multiplier is `sqrt(5)`. Weights are not normalized. No column
rescaling or current conversion is applied. Positions are metres, corrector
increments radians, and singular values m/rad. The rows of `Vt` are the requested
weighted right-singular-vector controls, sorted by descending sensitivity.

### Numerical calculation and scope

At each setting, all columns are obtained from independent local kick probes
in a single tracking pass: 229 probe particles comprise the reference and
`+h,-h,+h/2,-h/2` perturbations for all 57 controls, with `h=1e-7 rad`.
A particle receives its kick immediately after its corrector. Richardson
extrapolation combines the two central derivatives. There is no global
transfer-matrix inverse or fixed-prefix optics approximation.

For Cheetah derivatives, the float32-loaded lattice and magnet values are
retained, then tracked with float64 probes to avoid subtraction cancellation.
Transmission tracking remains float32 as in the framework. Independent actual
magnet-strength perturbations at three grid nodes, four selected columns per
plane and both backends agree with the simultaneous-probe method to a maximum
weighted relative error of **9.15e-15**. Derivative step changes are below
4.92e-10 relative. Cross-plane responses are zero in these ideal lattices.

The numerical resolved-mode cutoff is **1e-8 of the largest singular value**.
This excludes weak finite-difference residuals near the null space; all singular
values and vectors, including those below this cutoff, are still saved in the
full grid files and CSVs. Full SVD reconstruction errors are below 1.1e-14.
The resolved-only API reconstruction is below 1.1e-10 in the worst horizontal
Xsuite case and about 1e-14 otherwise. Orthonormality and modal round trips
are checked at every node; see
[the numerical validation](results/vbc_on_l4h_off/orbit/api_validation.json) and
[independent perturbation checks](results/vbc_on_l4h_off/orbit/independent_reference_validation.json).

**These are linear reference-orbit responses, not derivatives of the clipped
full-bunch centroid.** Full-bunch transmission is tracked separately. In a
lossy setting, a mathematical reference response need not describe the BPM's
surviving-bunch reading. Saved `bpm_valid` and `ip_readout_valid` flags apply the
framework's 10% signal threshold; invalid readouts are flagged, not silently
removed from the requested full response matrix or its weighting.

## Load complete matrices and vectors

The canonical per-backend, per-plane files are under
[`results/vbc_on_l4h_off/orbit/`](results/vbc_on_l4h_off/orbit/):

- `cheetah_x_grid_svd.npz`, `cheetah_y_grid_svd.npz`.
- `xsuite_x_grid_svd.npz`, `xsuite_y_grid_svd.npz`.
- `{backend}_{plane}_response_rows.csv`: complete **unweighted** matrix rows
  at all settings, with target sizes, readout name and row weight.
- `{backend}_{plane}_right_singular_vectors.csv`: every right-vector coefficient,
  its singular value and whether the mode is resolved, for all settings.

```python
import numpy as np

with np.load('orbit_matching/results/vbc_on_l4h_off/orbit/xsuite_x_grid_svd.npz') as d:
    i, j = 5, 7
    R = d['response'][i, j]              # (32,28), m/rad
    A = d['weighted_response'][i, j]     # sqrt(W) R
    U = d['U'][i, j]                    # complete thin SVD, including weak modes
    s = d['singular_values'][i, j]
    Vt = d['Vt'][i, j]
    rank = int(d['rank'][i, j])
    settings = dict(zip(d['quadrupoles'].tolist(), d['k1'][i, j]))
    correctors = d['correctors'].tolist()
    readouts = d['monitors'].tolist()    # IP last
```

Each NPZ includes all 144 matrices, the complete thin U/s/Vt, the coarse and
half-step responses, the cutoff and rank per node, readout positions, names,
weights, VBC/L4H state, all 46 quadrupole settings, and the target-size axis.
The runtime `data/orbit_response_model.npz` also contains the full-bunch
transmission and readout-validity masks for both backends.

## Compatible runtime controls

```python
from optics_matching_vbc_on import MatchedOptics, build_xsuite, build_cheetah
from orbit_matching import OrbitModes, OrbitResponse, apply_to_xsuite, apply_to_cheetah

optics = MatchedOptics(l4h_off=True)      # essential: orbit calibration is L4H drift
sx, sy = map(float, (optics.axis[5], optics.axis[7]))
x_modes = OrbitModes('x', optics=optics, backend='xsuite')
y_modes = OrbitModes('y', optics=optics, backend='cheetah')
result = x_modes.evaluate(sx, sy)
s, Vt, U = result['singular_values'], result['Vt'], result['U']
T = result['transmission']
valid_bpms = result['bpm_valid']
valid_ip = result['ip_readout_valid']
variables = x_modes.ranked_variables(sx, sy, n_modes=5)

# Modal amplitudes and resulting corrector increments are radians.
delta_theta = x_modes.to_kicks([1e-6, -2e-6], sx, sy)
coordinates = x_modes.from_kicks(delta_theta, sx, sy)
line = build_xsuite(sx, sy, model=optics)
segment = build_cheetah(sx, sy, model=optics)
apply_to_xsuite(line, sx, sy, [1e-6, -2e-6], plane='x', model=x_modes)
apply_to_cheetah(segment, sx, sy, [1e-6, -2e-6], plane='y', model=y_modes)

# Response without SVD; defaults also select VBC on and L4H drift.
Rx = OrbitResponse(backend='xsuite').evaluate(sx, sy, 'x')['response']
```

Python target sizes remain metres; `OrbitModes` still defaults to plane `x`
and backend `xsuite`. The original method signatures remain compatible, with
`backend=` added as an optional keyword. Existing
`from optics_matching import OrbitModes, OrbitResponse` imports still forward
to the current controls. An explicitly supplied old VBC-off optics provider is
rejected by its calibration fingerprint.

Adapters **add increments** and never change quadrupoles or RF. Repeated calls
accumulate. Xsuite uses `knl[0] -= delta_theta_x`, `ksl[0] += delta_theta_y`;
Cheetah tensor dtype/device are preserved. First build the corresponding
physical lattice. Finite kicks can cause extra loss or violate apertures;
no corrector hardware limits or nonlinear correction guarantee is supplied.

At an exact grid node, queries use the saved tracked matrix. Off-grid quad
strengths use the new provider's cubic interpolation; with default
`exact_response=True`, the response is **re-evaluated with reference probes**
in the selected backend, rather than interpolating the old four-magnet model.
No full bunch is tracked at query time. Off-grid matching accuracy and
transmission are unvalidated; transmission flags are available only at saved
nodes. `exact_response=False` exposes diagnostic response interpolation.

Vector-gauge splines set local signs. `reference_Vt`, `transport_path`,
`projector` and `n_modes_for_fraction` retain their interfaces. Individually
sorted vectors are not promised a globally smooth orientation near degeneracies.

## Calibration identity and regeneration

The public optics contract remains `evaluate(sx,sy)['settings']`, keyed by
magnet name, and `configuration_fingerprint()`. Array/dictionary order is not
part of that contract. The VBC-on provider's identity covers its grid, original
and L4H-drift lattices, converter, nominal bunch, aperture table and transport
adapters. Response and mode checksums bind each compiled bundle to that identity.
Changes raise `StaleCalibrationError`; restart Python and regenerate the
responses instead of editing identity files to bypass the check.

Use the existing nominal-model environment; Gym is not required. Transport
adapters are extracted from SimEnv without its optional RL environment imports.
A full reproduction is:

```bash
# Refresh transport adapters if SimEnv was changed.
nominal_model/.venv/bin/python -m orbit_matching.sync_backend_transport

# Full 200k transmission, supplied L4H and separate cavity-drift case.
XSUITE_ALLOW_KERNEL_COMPILATION=1 nominal_model/.venv/bin/python -m orbit_matching.evaluate_vbc_transmission --workers 4
XSUITE_ALLOW_KERNEL_COMPILATION=1 nominal_model/.venv/bin/python -m orbit_matching.evaluate_vbc_transmission --workers 4 --l4h-off

# Re-evaluate both backend orbit matrices at every node, always L4H drift.
XSUITE_ALLOW_KERNEL_COMPILATION=1 nominal_model/.venv/bin/python -m orbit_matching.build_vbc_orbits --workers 2 --rebuild
nominal_model/.venv/bin/python -m orbit_matching.validate_vbc_reference
nominal_model/.venv/bin/python -m orbit_matching.validate_vbc_controls
nominal_model/.venv/bin/python -m orbit_matching.plot_vbc_results
nominal_model/.venv/bin/python -m unittest orbit_matching.test_compatibility
```

Transmission rows and orbit nodes are checkpointed for restart. If transmission
inputs change, archive that case's `transmission_checkpoints/` and
`transmission_inputs.json` before rerunning. For orbit rebuilding, `--rebuild`
automatically archives old checkpoints before re-evaluation. The old orbit
build/plot/validation module names forward to these current commands.

A JSON control export is available with
`python -m orbit_matching --sigma-x-mm 1 --sigma-y-mm 3 --backend cheetah --top 5`.
These off-grid requested sizes evaluate interpolated strengths, with the scope
limitations above.

`data/legacy_vbc_off/` preserves the former VBC-off runtime bundles.
Files directly under `results/` from the previous run are historical; current
results are in `results/vbc_on/` and `results/vbc_on_l4h_off/`. Input hashes,
checkpoints and validation reports accompany the current outputs.

![Supplied-lattice transmission](results/vbc_on/transmission_comparison.png)

![Separate L4H-drift transmission](results/vbc_on_l4h_off/transmission_comparison.png)

![Updated weighted orbit SVD comparison](results/vbc_on_l4h_off/orbit/orbit_svd_comparison.png)
