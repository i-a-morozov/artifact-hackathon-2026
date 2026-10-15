# MBRL_opt — the model-based RL stage of the tuning pipeline

Getting a `SimEnv` episode — a new day on CLARA, with its own beam, its own survey
error and a request for a spot at `cla_feh_dia_scr_01` — onto that spot in around
twenty-five shots.

This is the last stage of the team pipeline and it sits on the other stages rather
than replacing them: it takes the reference optics and the SVD control rankings from
[`MFopt`](../MFopt) / [`optics_matching`](../optics_matching), and reads its beam with
the screen measure from [`image_processing`](../image_processing). What is added here
is the twin and the tweak.

```
request (mu_x, mu_y, sigma_x, sigma_y)
        │
        ▼
1. reference optics      MFopt.optics_for_target  ──  no shots, ~40 s of model time
     46 quadrupole k1s that give the requested size
        │                                                         1 shot to load
        ▼
2. steer                 SVD: correctors vs BPMs at that optics
     leading corrector planes, response from the twin, inverted    1–4 shots
        │
        ▼
3. control set           SVD: leading quadrupoles, plus the
     sextupoles, RF phases and chicane                             no shots
        │
        ▼
4. match the size        differentiable twin + CEM/Adam            ~20 shots
     fit what the machine is, plan against it, re-read the camera
     scores the spot size and transmission only
        │
        ▼
5. place the spot        SVD: final correctors vs readout centroid  1 shot
     two-row least squares, not a search
```

**Size and position are separated on purpose.** The focus works only on the
quadrupoles and scores only the size; where the spot lands is settled afterwards by
the two correctors nearest the screen, in a single linear solve. Put the two in one
optimiser and a quadrupole will happily fix a 2 mm position error by distorting the
optics — which it can do, and which costs the size the focus just bought.

## Why it is shaped like this

**The lookup does most of the work, for free.** `optics_for_target` re-matches the
final focus on an error-free copy of the model and hits the requested *size* to a
fraction of a percent at 99 % transmission, without touching the machine. Stage 4
therefore is not solving the tune — it is correcting the difference between the
machine the optics were matched for and the one the episode dealt.

**The lookup matches size, not position.** The request also names a spot anywhere
within 2 mm of the screen centre, which is about eight requested spot sizes and most
of the score. That is what the final correctors in the focus set are for, and it is
why the centroid SVD alone is not enough: it ranks `cla_feh_mag_hvcor_01/02` near the
bottom precisely because they move nothing but the last screen, which is exactly the
property wanted here.

**Steps are measured in effect, not in units.** Before the focus starts, one batched
track gives the response of all four measured moments to every control knob, and a
step is redefined as the move that changes the reading by half a requested spot
(`trust_effect`). No fixed physical step works: 0.1 mrad is a large kick from a
corrector near the injector and nothing from the one a metre before the screen — and
it is the second that has to carry the spot 2 mm. Before this was fixed the planner
plateaued at a mismatch of 8 because it could not reach the requested position.

**The planner is anchored on the measurement.** It never asks the twin where the beam
is, only what *changes* if a knob moves:

```
predicted(delta) = measured + [twin(knobs + delta) − twin(knobs)]
```

A model wrong by a constant still predicts the effect of moving a quadrupole, and
anchoring cancels the constant exactly.

**The beam is read off the camera.** `screen.measure` wraps the measure written for
the real CLARA captures, so the agent is driven by the same quantity in simulation as
it would be on the machine. The benchmark still scores the particle moments; the agent
never sees them.

## Running it

```bash
cd challenges/optimisation
uv pip install --python .venv/bin/python \
  -r RL_framework/requirements.txt -r nominal_model/requirements.txt

.venv/bin/python -I MBRL_opt/smoke_test.py                      # ~2.5 min, 23 checks
.venv/bin/python -I MBRL_opt/evaluate.py --episodes 8 --workers 8
.venv/bin/python -I MBRL_opt/evaluate.py --episodes 8 --fit-misalignments
```

Episodes are independent and each pins itself to one thread, so `--workers` is close
to linear. An episode is two to three minutes; most of it is the twin.

## The files

| File | What it is |
|---|---|
| `agent.py` | the four stages and the shot budget |
| `twin.py` | `CheetahTwin`: the differentiable model, and `readings()` reproducing `SimEnv._track`/`_expose` exactly |
| `system_id.py` | fits the beam (and optionally the survey error) to the episode's shots, through Cheetah |
| `planner.py` | truncated-SVD steering, CEM + Adam focus, anchored on the last reading |
| `knobs.py` | the control vector, its physical scales and its effect-calibrated steps |
| `screen.py` | the robust camera measure, wrapping `image_processing` |
| `config.py` | `MBRLConfig` |
| `baselines.py` | the pipeline with its later stages removed |
| `evaluate.py`, `smoke_test.py` | the comparison and the checks |

## Results

32 episodes, seeds 1234–1265, at SimEnv's defaults (`beam_variation` 1.0,
`misalignment_variation` 1.0, shot jitter and the VBC on). Every method sees the same
episodes. Score is the mean `info["benchmark_score"]`, which is
`score.Report.calc_score` for one episode — mismatch in requested spot sizes, plus
`TRANSMISSION_WEIGHT` for the bunch that never arrived, plus the shot cost. Lower is
better. The hit columns are `score.HIT_TOLERANCES`: the fraction of episodes whose
mean component error came inside 50 %, 25 % and 10 % of the requested spot.

| method | score | median | mismatch | shots | <50% | <25% | <10% |
|---|---|---|---|---|---|---|---|
| **full pipeline, survey error fitted** | **0.95** | **0.66** | 0.65 | 25.2 | 91 % | 84 % | 41 % |
| full pipeline | 1.44 | 0.78 | 1.14 | 25.6 | 81 % | 72 % | 25 % |
| optics + steering | 5.88 | 6.23 | 5.81 | 5.0 | 6 % | 0 % | 0 % |
| reference optics only | 6.12 | 7.25 | 6.10 | 1.0 | 0 % | 0 % | 0 % |
| do nothing | 9.51 | 10.00 | 9.51 | 0 | 0 % | 0 % | 0 % |

Read down the table: the reference optics takes a failed episode to 6.1 in one shot,
steering is worth a little more, and the model-based tweak does the rest — 5.9 down to
0.95, with a median of 0.66 and 41 % of episodes landing inside a tenth of the
requested spot. Every method delivers beam on every episode.

The reference optics scoring 6.1 is not the lookup being wrong: on its own model it
hits the requested size to a fraction of a percent. It is the cost of the request
naming a *position* the matched family never matched for, plus the episode's own beam
and survey error. That is precisely the gap stage 4 exists to close.

**Fitting the survey error is worth a third of the score**, 1.44 to 0.95, and takes
the hit rate inside a tenth of the requested spot from 25 % to 41 %. It is still off
by default, so the aligned twin is the baseline and this is an ablation you switch on
with `--fit-misalignments`; on this evidence it should probably become the default. A
misaligned quadrupole kicks and a misaligned sextupole focuses, and no amount of
adjusting where the beam was launched imitates either.

Reproduce with `evaluate.py --episodes 32 --seed 1234 --workers 32`, adding
`--fit-misalignments` for the first row. Numbers are in `results.json` and
`results_fitmis.json`. These were run on the 64-core apml1 container, which is about
half the per-core speed of a laptop and ran 32 episodes of all four methods in around
40 minutes. Give `--workers` at least as many slots as there are episodes, or the
leftovers queue for a whole second wave.

### Cost per episode

Measured on seed 1234: 25 shots on the machine, 290,000 particle-tracks. The optics
lookup spends 2.8 M on its own error-free twin, and the agent's twin spends 30.7 M
across 611 calls — most of them batched, since a CEM iteration is 128 candidates in
one call. The machine is about 1 % of the arithmetic, which is the whole point: a
shot is the only thing that costs beam time.

## Things worth knowing

**The twin has to be the environment's own lattice.** It is built through
`xsuite_model.cheetah_segment`, from `data/clara/lattice_data/CLARA_cheetah.json`,
with apertures on — not from the copy at `RL_framework/FEBE_full.json`. The two were
identical until the VBC was turned on and only one was updated; a twin on the stale
copy disagreed with the machine by 1.5 mm at the readout and the planner spent the
episode optimising a different accelerator. `smoke_test.py` holds the two to 0.0001 µm
on an error-free machine, and that check is the one to run first after any change
upstream.

**Two aperture subtleties, both load-bearing.** Whether a particle is inside the beam
pipe is a step function, and half the beam is scraped in the FEA arc, so the objective
is a staircase with no slope. Blurring the wall is badly biased — at the true latent
it moved the predicted loss from 0.00 to 114.7 and the fit walked away chasing it — so
`twin._aperture` takes the *value* from the hard edge and the *gradient* from a
sigmoid. And the 282 aperture elements break Cheetah's map fusion, which is why a
track costs 90 ms rather than 37.

**Batching is not free for these knobs.** A batch of candidate RF phases is a batch of
different beam energies, so nothing downstream can be fused: at 128 candidates the six
quadrupoles cost 287 ms and all seventeen knobs 1675 ms. A candidate is ~13 ms against
~160 ms for a gradient step, so CEM still wins, but only enough to be worth running
when the tune is stuck — hence `explore`.

**Do not clip the planner's cost at the benchmark's scale.** The benchmark stops
counting at ten requested spot sizes; an untuned machine sits at seventy. A cost
clipped at ten is flat across the whole region where tuning happens.

## Limits, and what to try next

**The size residual is the binding constraint.** Position comes out to a few tens of
microns, but when the episode's beam is vertically mismatched the readout `sigma_y`
can end several times the request and the focus set cannot pull it back inside the
budget. The most promising fix is to close the loop on stage 1: re-run
`optics_for_target` against the beam the system identification has *fitted*, rather
than the nominal beam it assumes, and reload. That is a second 40-second lookup and no
extra shots.

**The lookup sometimes misses.** On one of the two smoke-test seeds it returned optics
21.8 % off the requested size in its own model, which caps what the tweak can do.
That is `MFopt`'s to fix; `EpisodeResult.optics_error` reports it per episode.

**Steering is nearly free and nearly useless here.** The reference optics already
delivers beam with every BPM reading, so there is no beam to recover, and the orbit
bottoms out around 550 µm rms — a handful of correctors against a survey error on 46
quadrupoles. It would earn its place on a machine that had lost the beam.

**Turn the misalignment fit on.** It is off by default so the aligned twin is the
baseline, but a survey error does things no launch offset can imitate.
`--fit-misalignments`.

**Not built: the Dyna layer.** The twin is a free simulator, so the natural next step
is to randomise the latent, roll out inside it, and train a policy that proposes the
first move — turning a twenty-shot MPC tweak into a few shots plus a refinement.
