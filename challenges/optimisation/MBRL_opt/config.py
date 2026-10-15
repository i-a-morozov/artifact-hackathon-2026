"""What the agent is allowed to turn, how many shots it may spend, and how hard it
thinks between them.

The knob groups are the two stages of the tune an operator does: steer the beam down
the line on the BPMs, then shape it on the readout screen. They are named here rather
than discovered, because which corrector steers which stretch is a fact about CLARA and
not something worth re-deriving every run. Every group is a plain list, so widening the
set is a one-line change — `MBRLConfig(focus_quads=[...])` or appending to the defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The last two correctors before the hutch, both planes. The request names a
# position as well as a size — anywhere within 2 mm of the screen centre, which is
# some eight spot sizes and most of the score — and the matched optics only ever
# matched the size. These put the spot where it was asked for, in one shot, after
# the focus has finished with the quadrupoles. The centroid SVD ranks them near the
# bottom precisely because they move nothing except the last screen, which is the
# whole reason they are wanted for this and nothing else.

# Extra knobs the SVD ranking cannot offer, because the optics lookup does not vary
# them: the two FEA sextupoles, the four powered cavity phases and the VBC chicane.
# They are the only handles on the energy spread the FEA dogleg turns into horizontal
# size at the readout, so the focus keeps them available even though the matched
# family was built with the chicane and L4H off.
FINAL_CORRECTORS = [
    "cla_feh_mag_hvcor_01_h", "cla_feh_mag_hvcor_01_v",
    "cla_feh_mag_hvcor_02_h", "cla_feh_mag_hvcor_02_v",
]

ENERGY_KNOBS = [
    "cla_fea_mag_sext_01",
    "cla_fea_mag_sext_02",
    "cla_l02_lin_cav_01",
    "cla_l03_lin_cav_01",
    "cla_l4h_lin_cav_01",
    "cla_l04_lin_cav_01",
    "cla_vbc_chicane",
]


@dataclass
class MBRLConfig:
    """Knobs, budgets and solver settings for `MBRLAgent`."""

    # ---------------- what may be turned ---------------- #
    #
    # Not names but counts: which magnets matter depends on the optics the episode
    # asked for, and `OpticsEnv.optics_for_target` works that out by SVD before the
    # machine is touched. These say how many of its ranking to take.

    # Corrector planes used to centre the beam, from the centroid modes.
    steering_correctors: int = 10
    # Quadrupoles used for the focus, from the beam-size modes.
    focus_quads: int = 6
    # Corrector planes turned alongside the quadrupoles during the focus. Zero:
    # the focus matches the *size*, and where the spot lands is settled afterwards
    # by `placement_correctors` in a single linear solve. Putting the two in one
    # optimiser lets a quadrupole chase a 2 mm position error by distorting the
    # optics — which it can do, and which costs the size the focus just bought.
    focus_correctors: int = 0
    # Knobs outside the SVD's reach that the focus should have anyway.
    focus_extra: list[str] = field(default_factory=lambda: list(ENERGY_KNOBS))

    # The last stage: one round of SVD steering that puts the finished spot on the
    # requested position, on the correctors closest to the screen.
    placement_correctors: list[str] = field(
        default_factory=lambda: list(FINAL_CORRECTORS)
    )
    # Rounds of it. The readout centroid is linear in these kicks, so one is
    # normally enough and the rest of the budget is better spent on the focus.
    placement_shots: int = 1
    # Singular values below this fraction of the largest are dropped from the
    # placement solve.
    placement_cutoff: float = 0.05

    # ---------------- the shot budget ---------------- #

    # `score.Report.calc_score` charges EVAL_WEIGHT (0.5) for every doubling of
    # EVAL_BUDGET (50) shots, so the cost of a shot is real but gentle: 40 shots cost
    # 0.34 of a requested spot size, 80 cost 0.58. These defaults sit just under one
    # budget so the tune is paid for at roughly a third of a spot.
    steering_shots: int = 4
    focus_shots: int = 12

    # Stop steering early once the orbit is this flat (m, rms over valid BPMs) and
    # every BPM is reading. Shots saved here are spent on the focus instead.
    steering_orbit_tolerance: float = 2e-4
    # Or once this many shots in a row have failed to flatten it any further, which
    # is the usual way steering ends: a handful of correctors against a survey error
    # on forty-six quadrupoles reaches a floor and stays there.
    steering_patience: int = 2
    # Stop the focus once the spot is this close to the requested size, in requested
    # spot sizes. The focus scores size and transmission only, and the reference
    # optics usually hands it something already within a tenth of a spot, so this
    # sits just above the shot-to-shot jitter — about 8 % of an rms, or 0.1 in these
    # units. Set it below that and the stage spends its whole budget chasing noise.
    focus_mismatch_tolerance: float = 0.15
    # Or stops after this many shots in a row that failed to improve on the best.
    # The jitter floor moves with the request, so knowing when to stop is better
    # decided by watching than by a threshold.
    focus_patience: int = 3

    # ---------------- the twin ---------------- #

    # Particles the twin tracks. Cheetah fuses linear maps, so a track costs much
    # the same at 2,000 as at 10,000, and this is mostly about how much sampling
    # noise the twin invents for itself. `smoke_test.py` uses SimEnv's own count,
    # which is what makes the fidelity check exact to float32.
    twin_particles: int = 4_000
    # Grid the environment bins its screen images onto. The agent measures its beam
    # from the picture, so this is its resolution: 128 is too coarse to see an 80
    # micron request, 512 reproduces the true moments to a tenth of a micron.
    screen_shape: tuple[int, int] = (512, 512)
    # Fit quadrupole and sextupole misalignments and diagnostic centring offsets as
    # well as the beam. Off by default: the aligned twin is the baseline, and the
    # orbit error it cannot explain is absorbed into the beam's launch offset.
    #
    # Worth switching on. On one episode at the defaults the aligned twin books 5.5
    # and the fitted one 3.1, because a survey error does things to the beam that no
    # launch offset can imitate — a misaligned quadrupole kicks and a misaligned
    # sextupole focuses, and both show up in the spot the readout is scored on.
    # `evaluate.py --fit-misalignments` runs the comparison.
    fit_misalignments: bool = False

    # ---------------- system identification ---------------- #

    # Adam steps in the first fit of a stage; later fits are warm-started from it and
    # take a quarter as many. Everything in this block and the next trades wall clock
    # for quality. At these settings a shot costs a few seconds and an episode a
    # minute or two, which is what makes a twenty-episode comparison practical;
    # tripling them is worth a little score and costs an afternoon.
    id_iterations: int = 60
    id_learning_rate: float = 0.05
    # Shots kept in the episode's history to fit against. The machine holds still
    # within an episode, so every shot is evidence about the same latent state; only
    # the shot jitter differs, and averaging over a few shots beats trusting one.
    id_history: int = 8
    # Refit this often, in shots. The machine does not change within an episode, so
    # a fit that has already seen several shots learns little from the next one.
    id_interval: int = 2

    # ---------------- planning ---------------- #

    # Gradient descent through Cheetah, then CEM to escape the local minimum it finds.
    # Batching is what usually makes a CEM population nearly free, and for this knob
    # set it is not: an RF phase or a chicane angle changes the energy the beam
    # carries, so a batch of candidates is a batch of different beam energies and
    # Cheetah can no longer fuse the elements downstream into one map. Measured at
    # 128 candidates, the six quadrupoles alone cost 287 ms and all seventeen knobs
    # 1675 ms. A candidate is therefore about 13 ms against 160 ms for a gradient
    # step — still much cheaper, but not free, so the population stays modest and
    # the agent only searches globally when it has a reason to (`explore`).
    plan_iterations: int = 12
    plan_learning_rate: float = 0.05
    cem_population: int = 128
    cem_elites: int = 16
    cem_iterations: int = 4
    # How far one shot may move a knob, in units of its natural step
    # (`knobs.SCALE`): one means 0.1 mrad on a corrector, 0.5 m^-2 on a quadrupole,
    # 2 degrees on a cavity phase.
    #
    # This is the most important number here. The planner predicts the *change* a
    # move makes and adds it to the last reading, which is exact for a small move
    # and drifts for a large one, so a generous trust region lets it leap somewhere
    # its own prediction no longer holds, overshoot and thrash. It shrinks by half
    # after any step that made the reading worse and grows slowly back.
    trust_region: float = 1.0
    # What one step is worth, in requested spot sizes of change at the readout. The
    # focus calibrates every knob against this before it starts, so a step of one
    # means the same thing whichever knob takes it.
    trust_effect: float = 0.5
    # The smallest the trust region may shrink to after repeated bad steps. Below
    # this a shot cannot move the beam further than the shot-to-shot jitter does,
    # so there is nothing to be learned from taking it.
    min_trust_region: float = 0.05
