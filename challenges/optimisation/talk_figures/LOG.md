# Talk figures log

Figures for the CLARA hackathon talk, one section per figure: path (relative to this folder), what it shows, settings, key numbers and a suggested slide message.

## The deck

`talk.html` is the talk: 23 slides, 16:9, reveal.js, built from `notes.txt` and the figures here. Serve this folder and open it:

```bash
python3 -m http.server -d . 8000     # then http://localhost:8000/talk.html
```

Opening `talk.html` as a `file://` URL mostly works too, but serving it is safer: some browsers refuse the animation iframe otherwise.

Keys: arrows or space to move, `f` fullscreen, `o` overview, `b` blank, `s` speaker view. Adding `?print-pdf` to the URL and printing gives a static PDF.

reveal.js 5.1.0 is vendored in `talk_assets/reveal/` (`reveal.js`, `reveal.css`, `theme/white.css`), so the deck needs no network. The one local change is the webfont `@import` removed from `white.css`; it falls back to the system sans.

**The "seven workstreams" slide is the live animation.** It is `workflow_animation.html` in a reveal background iframe, so it animates in the browser and replays every time the slide is entered (reveal reloads background iframes on each visit). Click it to replay by hand. Edit the animation and the slide follows; there is nothing to re-render.

Figures used: `geometry/line_geometry_apertures_nominal_beam.png`, `transmission_along_s/losses_by_section_vbc_on.png`, `optics_nodes/transmission_matrix_vbc_on.png`, `optics_nodes/rematch_vbc_on.png`, `mfopt/mfopt_workflow.svg` (an `<img>`, so its internal CSS is honoured and it stays sharp at any size) and `screens/`. `screens/` holds copies of `image_processing/Noisy_beam.png` and `Weak_beam.png`: the deck must not reference paths above its own folder, or serving this directory breaks the images.

Numbers on the slides come from `MFopt/README.md` (30 days, seeds 1000-1029), `MBRL_opt/README.md` and its `results*.json` (32 episodes, seeds 1234-1265), `section-opt/section_review.md` (the S04 method comparison) and `orbit_matching/README.md` (the Cheetah/xsuite transmission cross-check). Re-run those and the slides need re-reading.

### Stills of the animation

The deck does not need them, but `scripts/make_anim_frames.py` renders `workflow_animation.html` to a numbered PNG sequence in `anim_frames/`, for a poster, a paper or a GIF. It reuses the model and drawing code from the HTML verbatim and only swaps the real-time loop for a deterministic one. Headless chromium screenshots come back blank for this canvas, so the page hands every frame back as a data URL in one `--dump-dom` instead.

```bash
python3 -I scripts/make_anim_frames.py            # 170 frames, light, 1280x720, 6.7 MB
python3 -I scripts/make_anim_frames.py --theme dark --fps 12
```

## Common settings (geometry and transmission-along-s figures)

- Model: upstream `amylizzle/artifact-hackathon-2026` master at `edfcf46` (includes PR #3 per-element SimFrame apertures and PR #4 VBC on at 118.5 mrad with edge angles).
- Since these figures were made, master runs the VBC at 131.7 mrad (R56 = −57.6 mm, the measured maximum compression into FEBE). With that default the nominal optics transmit 72% (Cheetah and xsuite), all lost upstream of the VBC; the numbers below are for 118.5 mrad.
- Simulator: Cheetah 0.8.4 through `SimEnv`, apertures on, nominal 200k-particle measured beam (35.8 MeV), no misalignments, jitter or ripple.
- Tracking order: second order for every element that supports it (Drift, Quadrupole, Dipole, CombinedCorrector, Sextupole), per Francesco. For reference, the nominal VBC-on transmission is 58.57% linear and 58.71% second order.
- Line: `cla_s02_sim_aper_01` to `cla_feh_dia_scr_01`, 64.4 m, 655 elements, 282 apertures.
- Optics nodes: Eiad Hamwi's matched-optics grid, `optics_matching/results/validated_grid.npz`. It has 12×12 target spot sizes σx, σy at the FEH screen, log-spaced from 0.1 to 20 mm. Each node sets all 46 quadrupole k1 values.
- "VBC off" means the VBC chicane knob is set to 0 rad (angles and edges), with the same quads.
- Transmission at each s is the fraction of the launched bunch still alive after that element.
- Scripts: `scripts/track_along_s.py` (tracking, about 9 s per node) and `scripts/make_figures.py` (plots). Raw data: `data/along_s_vbc_{on,off}.npz`. Numbers: `data/summary_along_s.json`.

## 1. Line geometry with apertures

**Paths:** `geometry/line_geometry_apertures.png` (geometry only) and `geometry/line_geometry_apertures_nominal_beam.png` (adds the nominal-optics beam envelope).

**What it shows:**
- Top: the floor plan (Z, X) from integrating the dipole angles. It shows the VBC chicane (4 × 118.5 mrad, about 0.2 m offset) and the FEA dogleg (4 × 14°, 3.7 m lateral offset).
- Middle: a lattice strip, with quads drawn up for focusing in x, plus dipoles, RF cavities, the TDC, sextupoles, correctors and screens.
- Bottom: x and y half-apertures against s, drawn as walls. The display is clipped at 40 mm.
- The second version overlays the ±3σ envelope of surviving particles for the nominal optics with VBC on.

**Key numbers:**
- The default beam pipe is 17 mm half-aperture (rectangular).
- The linac irises are 10 mm in L02, L03 and L04.
- The L4H iris is 4.12 mm, the tightest aperture in the line.
- The TDC region in S07 is 5 mm.
- The S02 collimator is 10 mm (elliptical).
- In x, the pipe is unlimited from S07 to FEA and 34 mm at the FEA dipoles.
- FEH is 68 mm (elliptical).

**Slide message:** The line has two bottlenecks, the 4.1 mm L4H iris and the 5 mm TDC aperture. With the nominal optics, the vertical beam (±3σ ≈ 9 mm) is larger than both.

## 2. Transmission along s, all 144 nodes (VBC on)

**Path:** `transmission_along_s/transmission_along_s_vbc_on.png`

**What it shows:**
- One thin line per node, coloured by its final transmission.
- Bold black: the nominal optics with VBC on. Dashed: the nominal optics with VBC off (reference).
- The right panel zooms in on 57 m to the screen.

**Key numbers (VBC on):**
- Nominal optics: 58.7% at the FEH screen. The losses are:
  - L4H iris and cavity: 26.6%.
  - TDC aperture and S07 pipes: 13.8%.
- Nominal optics with VBC off: 13.2%. The extra losses are at the L04 entrance iris (17%) and the L04 cavity (9%), and the TDC loss rises to 29%. This matches the "Add CLARA apertures" thread.
- The 144 nodes reach a median of 92.7% (range 17.3% to 98.2%). 77 of 144 nodes are above 90%, and 120 of 144 are above 50%.
- The matched nodes carry almost the whole bunch through L4H and the TDC (median loss there is 0.6%). Their losses all happen in the last 4 m, in the 17 mm pipe at the FEA shutter, FEH wall and FMB1 mirror (s = 60.7 to 63.8 m). There, large target spots mean a large beam upstream of the screen.

**Slide message:** Re-matching the quads removes the L4H and TDC losses. Whatever is left is set by the target spot size: once the target σ at the screen reaches a few mm, the beam scrapes the last 17 mm pipe.

## 3. Transmission along s, all 144 nodes (VBC off)

**Path:** `transmission_along_s/transmission_along_s_vbc_off.png`

**Key numbers:**
- Nominal optics: 13.2%.
- The 144 nodes reach a median of 96.8% (range 59.6% to 96.9%). 135 of 144 are above 90%, and all 144 are above 50%.
- Every node loses about 1.5% in S07 (TDC region), then the large-spot nodes lose more in the last 17 mm pipe.

**Slide message:** The nodes behave better with the VBC off, which suggests they were matched without the chicane's edge focusing. With VBC on (master), the large-spot corner loses up to 80%.

## 4. 12×12 small multiples: transmission along s per node

**Paths:** `transmission_along_s/transmission_along_s_grid_vbc_on.png` and `..._grid_vbc_off.png`

**What it shows:** One panel per node, laid out like the matrix (x: target σx, y: target σy, both in mm). Each panel shows transmission from 0 to 100% against s from 0 to 64 m. The fill colour and the number give the final %. The grey line in each panel is the nominal optics.

**Key numbers (VBC on):**
- Every node with σx, σy ≤ 1.8 mm reaches at least 90%. Most of them reach 98%.
- Transmission falls monotonically toward large σ. The worst nodes are:
  - (20, 20) mm: 17%.
  - (20, 12.4) mm: 22%.
  - (12.4, 20) mm: 23%.
- Two isolated dips at small σ are visible: (0.69, 0.16) mm at 90% and (2.9, 0.1) mm at 92%.

**Slide message:** The usable optics range for transmission above 90% is σ up to about 2 mm in both planes.

## 5. Where the beam is lost, by section (VBC on)

**Path:** `transmission_along_s/losses_by_section_vbc_on.png`

**What it shows:** The percentage of the bunch lost in each section. Black bars are the nominal optics. Blue bars are the median over the 144 nodes, with a line from the 10th to the 90th percentile.

**Key numbers:**
- Nominal optics: L4H 19.9%, S05 6.6% (L4H cavity exit pipe), S07 13.8%.
- Nodes (median): FEH 4.5% (90th percentile 48%), FEA 0.7%, L4H 0.6%.

**Slide message:** The nominal optics lose beam in the middle of the line, while the matched nodes only lose it at the end.

## 6. Transmission matrix of the optics nodes

**Paths:**

- `optics_nodes/transmission_matrix_vbc_on.png`
- `optics_nodes/transmission_matrix_vbc_off.png`
- `optics_nodes/transmission_matrix_design.png`

**What it shows:** Final transmission (%) for each of the 144 nodes. x is the target σx and y is the target σy. The cell number is the transmission.

**Settings:**

- Model: as above, with 50k particles. Second-order tracking also uses second-order dipole edges.
- Tools: `optics_matching_vbc_on/evaluate_nodes.py` and `plot_matrix.py`.
- Cases:
  - VBC on: 118.5 mrad, master.
  - VBC off: knob set to 0.
  - Design: Eiad's own anchor lattice, with the VBC off and L4H off.

**Key numbers:**

- VBC on: median 92.7%, minimum 17% at 20×20 mm.
- VBC off: about 97% at almost every node, minimum 60%.
- Design: about 97%, minimum 63%.
- Nominal optics: 59% (VBC on) and 13% (VBC off).

**Slide message:** Eiad's nodes fix transmission for the VBC-off machine they were matched for. With the VBC on, the large-spot nodes lose most of the beam.

## 7. Rematched nodes with the VBC at the paper setting

**Path:** `optics_nodes/rematch_vbc_on.png`

**What it shows:** The VBC is at 131.7 mrad, which gives R56 = −57.6 mm. That is the measured maximum compression into FEBE (Jones et al., IPAC'26 THP2100). The left column is Eiad's nodes and the right column is the rematch. The top row is transmission and the bottom row is the worst-plane size error.

**Settings:**

- Eiad's 42-quad prefix is kept. The 30 quads after the VBC are refitted per node (L-BFGS, differentiable Cheetah, 3k particles).
- Each node is then checked in `SimEnv` with 50k particles.
- Script: `optics_matching_vbc_on/rematch.py`.
- k1 table: `optics_matching_vbc_on/results/rematched_grid_vbc_on.npz`.

**Key numbers:**

| | Eiad's nodes | Rematch |
|---|---|---|
| Median transmission | 89% | 99% |
| Median size error | 285% | 1.6% |

- 96 of 144 nodes are at T ≥ 95% and within 5% of the target size.
- The 12–20 mm corner still fails.
- In the model, the FEBE arc R56 is +7.75 mm (paper: +7.7 mm).

**Slide message:** A rematch downstream of the VBC recovers both the target sizes and about 99% transmission over most of the range.

## 8. Quadrupole strengths vs target size (rematch)

**Path:** `optics_nodes/quads_vs_target.png`

**What it shows:** k1 of the 30 refitted quads against target size. Each panel shows the σx = σy diagonal, a σx scan at σy = 1.1 mm, a σy scan at σx = 1.1 mm, and Eiad's diagonal (dashed). The ±100 A range is shaded.

**Key numbers:**

- FEA Q15/Q16 and FEH Q1/Q2 still set the size, following Eiad's curves.
- The VBC correction is mainly S06 Q01/Q02, with changes up to 2.6 m⁻², and FEH Q1.
- The other quads jitter by a few tenths of a m⁻². They are degenerate.
- FEA Q16 and FEH Q2 exceed ±100 A at large and small σ respectively.

**Slide message:** Only a handful of magnets do the work.

## Caveats

- These figures are Cheetah only. Francesco judged the difference from xsuite to be small, but it was not checked here.
- A single nominal beam is used, with no errors or jitter, so these are ideal transmissions.
- The transmission matrix (final transmission per node) belongs to the "Optics nodes transmission map" thread. The end points of these curves should agree with it when that thread also uses second-order tracking.
- The "Optics nodes transmission map" thread uses a slightly different second-order setup: dipole edges are also second order (`second_order_edge`), correctors stay linear, and it tracks 50k particles (`CLARA_cheetah_second_order.json`, in the project files of that study). The two setups agree within statistics: nominal VBC on is 0.5870 there and 0.5871 here, and node (0,0) is 0.98 in both.

## MFopt episode workflow

`mfopt/mfopt_workflow.svg` shows one `SimEnv` episode from `reset` to the benchmark score, as `MFopt/run_baselines.py --policies mfbo` runs it. Purple boxes cost no step: they run in the error-free Cheetah twin. Blue and teal boxes are machine shots, one step each. Stage A loads the target optics, threads the beam when it is lost, then steers the orbit and the centroid. Stage D (MF-BO) treats the twin as the low fidelity and the machine as the high fidelity, and stops when the expected improvement is worth less than one more step.

Key numbers (30 default days, seeds 1000-1029): stage A scores a mean of 0.49, with 30 % of days within 10 %, in about 5 steps. Stage A plus MF-BO scores 0.30 to 0.32, with 73 to 80 % within 10 %, in about 9 steps.

Slide message: the model does most of the work for free, and the machine is spent only on what the model gets wrong.
