"""Tuning a CLARA episode the way the pipeline does it, one shot at a time.

The episode is a new day on the machine: a beam of its own size and launch angle, a
survey error on the quadrupoles, and a request for a spot at `cla_feh_dia_scr_01`. The
agent gets there in four stages, and only the last of them is the model-based RL.

1. **Load the reference optics.** `OpticsEnv.optics_for_target` looks the requested
   spot up in the matched family in `optics_matching/` and re-matches the four
   final-focus quadrupoles to it in an error-free copy of the environment. On the
   nominal machine that already *is* the answer — it hits the requested sizes to
   within a fraction of a percent at 99 % transmission — and it costs no shots,
   because it is all done on the model. One shot loads the 46 strengths.

2. **Steer.** The same call returns an SVD of the beam centroid against all 58
   corrector planes *at that optics*, so the planes that actually move this beam are
   known before the machine is touched. The agent takes the leading ones, builds their
   response at the BPMs on its own twin, and inverts it. The orbit is linear in the
   kicks, so this is usually one shot and rarely more than four.

3. **Pick the coupled control set.** The same call also ranks all 46 quadrupoles by
   their weight in the beam-size modes. The focus works in the leading quadrupoles and
   the leading corrector planes together, a dozen or so numbers rather than 111.

4. **Tweak the reference tune.** What is left is the difference between the machine
   the optics were matched for and the one the episode dealt. The agent fits that
   difference — the beam's real size and launch, and optionally the survey error — onto
   a differentiable Cheetah twin, and plans each shot against it. See `twin.py`.

Throughout, the beam is read off the camera frame with `screen.measure`, the measure
written for the real CLARA screen captures, rather than from the particle moments
`info` offers. The benchmark still scores the moments; the agent never sees them.

    env = OpticsEnv(config_for(MBRLConfig()))
    result = MBRLAgent(MBRLConfig()).run(env, seed=0)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from config import MBRLConfig
from knobs import KnobSet
from planner import TRANSMISSION_WEIGHT, Planner, _pseudo_inverse
from features import expert_action, observe, pad_action
from screen import measure
from system_id import Shot, SystemID
from twin import CheetahTwin

# A mismatch ceiling high enough to watch a tune through. The benchmark stops
# counting at 10 and an untuned machine is far past it.
UNCAPPED = 1e3
# How much flatter the orbit has to get for a steering shot to count as progress.
# Below the shot-to-shot jitter it is not progress, it is noise.
IMPROVEMENT = 5e-6  # m
# How the focus trust region adapts: a step that improved the reading earns a longer
# leash, one that did not loses half of it.
TRUST_GROWTH = 1.3
TRUST_DECAY = 0.5
# What moving the screen costs on the real machine. Only one screen may be in the
# beam, and swapping which one takes about half a minute of mechanics — far longer
# than the shot itself — so it is counted separately from the shot budget.
SCREEN_SWITCH_SECONDS = 30.0
# Roughly what one shot costs on CLARA at 10 Hz once setting and settling are
# included. Only used to put screen time and shot time on one scale.
SHOT_SECONDS = 1.0
# How far the placement solve may move a corrector, in calibrated steps. Large
# enough that only the magnet's own range binds: the readout centroid is linear in
# these kicks and the whole job is to cross a 2 mm gap in one shot. A trust region
# belongs on a search, and this is not one — capping it at 5 mrad left the spot
# 1.6 mm short of a request that needed a 4.95 mrad kick the magnet could easily
# have given. `cla_feh_mag_hvcor_01` alone reaches 12.7 mm at the screen.
PLACEMENT_REACH = 1e4


@dataclass
class EpisodeResult:
    """How one tune went, in the terms the benchmark books it in."""

    benchmark_score: float
    mismatch: float  # requested spot sizes, `score.per_case_scores`
    relative_error: float  # what `score.Report.hit_rate` is thresholded on
    shots: int
    on_screen: bool
    transmission: float
    target: np.ndarray
    achieved: np.ndarray
    steering_shots: int
    orbit_rms: float  # m, over the BPMs that were reading at the end
    valid_bpms: int
    # The figures of merit, alongside the match itself.
    screen_switches: int = 0  # times the beam-intercepting screen was moved
    seconds: float = 0.0  # wall clock for the episode, model time included
    machine_seconds: float = 0.0  # what it would have cost on CLARA
    # How well the reference optics alone did, before anything was corrected: the
    # score the pipeline would book if it stopped after stage 1.
    optics_score: float = float("nan")
    optics_error: float = float("nan")  # `TargetOptics.relative_error`, in the model
    trace: list[dict] = field(default_factory=list)


class MBRLAgent:
    """Reference optics, SVD steering, then a twin-planned tweak of the tune."""

    def __init__(
        self,
        config: MBRLConfig | None = None,
        twin: CheetahTwin | None = None,
        policy=None,
    ):
        self.config = config or MBRLConfig()
        # Built lazily from the environment unless one was handed in, so the twin
        # is always a model of the machine actually being tuned. See
        # `CheetahTwin.for_env`.
        self.twin = twin
        self.identifier = None if twin is None else SystemID(
            twin, self.config.fit_misalignments
        )
        self.planner = None if twin is None else Planner(twin, self.config)
        # The Dyna layer, if one has been trained: proposes the focus move in a
        # millisecond where the planner takes seconds. `policy.py`.
        self.policy = policy
        self._since_fit = 0
        # Set True to collect behaviour-cloning data while running.
        self.record = False
        # Filled in during a run when `record` is on: the (observation, expert
        # move) pairs behaviour cloning is trained from.
        self.recorded: list[tuple[np.ndarray, np.ndarray]] = []

    def _ensure_twin(self, env) -> None:
        """Make sure there is a twin, and that it models this environment."""
        if self.twin is None:
            self.twin = CheetahTwin.for_env(env, self.config.twin_particles)
            self.identifier = SystemID(self.twin, self.config.fit_misalignments)
            self.planner = Planner(self.twin, self.config)

    # ---------- one episode ----------

    def run(self, env, seed: int | None = None, verbose: bool = False) -> EpisodeResult:
        """Tune the machine `env.reset()` deals, and report how close it got."""
        started = time.monotonic()
        self._ensure_twin(env)
        observation, info = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        self.identifier.reset()
        self.twin.reset_setpoints()
        self._extent = env.screen_extent()

        target = np.asarray(observation["target"], dtype=float)
        history = [self._shot(env, observation, info)]
        trace: list[dict] = []

        optics = self._load_optics(env, target, history, trace, verbose)
        steered = self._steer(env, optics, history, trace, verbose)
        self._focus(env, optics, target, history, trace, rng, verbose)
        self._place(env, target, history, trace, verbose)
        return self._result(history, trace, steered, optics, started)

    # ---------- stage one: the reference optics ----------

    def _load_optics(self, env, target, history, trace, verbose):
        """Look the requested spot up and put it on the machine.

        The lookup is free — it runs on an error-free copy of the model, not on the
        machine — so the only cost here is the one shot it takes to set 46
        quadrupoles, and it buys most of the tune.
        """
        optics = env.optics_for_target(float(target[2]), float(target[3]))
        self.twin.set_knobs(dict(optics.k1))
        observation, _, _, _, info = env.step(env.optics_action(optics))
        history.append(self._shot(env, observation, info))
        entry = {
            "stage": "optics",
            "shot": len(history) - 1,
            "mismatch": self._mismatch(history[-1], target, cap=UNCAPPED),
            "model_error": optics.relative_error,
            "model_transmission": optics.transmission,
            "clipped": list(optics.clipped),
        }
        trace.append(entry)
        if verbose:
            print(
                f"  optics     mismatch {entry['mismatch']:7.3f}   "
                f"model hits request to {optics.relative_error:.1%} at "
                f"{optics.transmission:.0%} transmission"
            )
        return optics

    # ---------- stage two: steer on the BPMs ----------

    def _steer(self, env, optics, history, trace, verbose) -> int:
        """Centre the beam at the loaded optics, on the corrector planes that matter.

        Which planes matter comes from the SVD the optics lookup already did against
        this optics; how far to move them comes from their response at the BPMs,
        which the twin gives for nothing. The orbit is linear in the kicks, so one
        shot usually does it.
        """
        knobs = KnobSet.from_knob_names(
            optics.correctors.ranked(self.config.steering_correctors),
            self.twin.element_types,
            env.magnet_limits,
        )
        values = knobs.read(env.magnet_settings())
        best, stalled, used = self._orbit_rms(history[-1]), 0, 0
        for step in range(self.config.steering_shots):
            shot = history[-1]
            if self._orbit_is_flat(shot) or stalled >= self.config.steering_patience:
                break
            self._identify(history, first=step == 0)
            plan = self.planner.steer(
                knobs, values, self.identifier.state,
                shot.bpm_x, shot.bpm_y, shot.bpm_valid,
            )
            if not np.isfinite(plan.cost):
                break
            values = plan.values
            self._apply(env, knobs, values, history)
            used += 1
            orbit = self._orbit_rms(history[-1])
            stalled = 0 if orbit < best - IMPROVEMENT else stalled + 1
            best = min(best, orbit)
            trace.append({
                "stage": "steer",
                "shot": len(history) - 1,
                "orbit_rms": orbit,
                "valid_bpms": int(history[-1].bpm_valid.sum()),
            })
            if verbose:
                print(
                    f"  steer {step:2d}   orbit {orbit * 1e6:7.1f} um   "
                    f"BPMs {int(history[-1].bpm_valid.sum()):2d}/{len(shot.bpm_valid)}"
                )
        return used

    def _orbit_is_flat(self, shot: Shot) -> bool:
        return bool(
            shot.bpm_valid.all()
            and self._orbit_rms(shot) <= self.config.steering_orbit_tolerance
        )

    @staticmethod
    def _orbit_rms(shot: Shot) -> float:
        valid = shot.bpm_valid.astype(bool)
        if not valid.any():
            return float("inf")
        return float(np.sqrt((shot.bpm_x[valid] ** 2 + shot.bpm_y[valid] ** 2).mean() / 2))

    # ---------- stages three and four: the coupled tweak ----------

    def _control_set(self, env, optics) -> KnobSet:
        """The significant control variables for the focus, from the two SVDs.

        The leading quadrupoles in the beam-size modes and the leading corrector
        planes in the centroid modes, turned together: the spot and its position are
        not independent once the beam is off axis in a quadrupole, so the two sets
        are optimised coupled rather than one after the other.
        """
        names = list(optics.quadrupoles.ranked(self.config.focus_quads))
        names += list(optics.correctors.ranked(self.config.focus_correctors))
        names += [n for n in self.config.focus_extra if n in env.magnet_limits]
        return KnobSet.from_knob_names(names, self.twin.element_types, env.magnet_limits)

    def _focus(self, env, optics, target, history, trace, rng, verbose) -> None:
        """Tweak the reference tune until the camera agrees with the request."""
        knobs = self._control_set(env, optics)
        values = knobs.read(env.magnet_settings())
        self._calibrate(knobs, values, target)
        reference = values.copy()
        best_values = values.copy()
        # Judged on what this stage is actually for. Scoring the focus on the full
        # mismatch would have it reject a real improvement in the spot because the
        # position — which it is not allowed to touch — happened to drift.
        best_mismatch = _mismatch(history[-1], target, position_weight=0.0)
        best_moments = history[-1].moments if history[-1].on_screen else None
        anchor_moments = best_moments
        trust = self.config.trust_region
        explore = True

        for step in range(self.config.focus_shots):
            if (
                _mismatch(history[-1], target, position_weight=0.0)
                <= self.config.focus_mismatch_tolerance
            ):
                break
            self._identify(history, first=step == 0)
            observed = self._observe(
                history[-1], target, knobs, values, reference, trust
            )
            proposal = self._propose(knobs, values, observed)
            plan = self.planner.focus(
                knobs, proposal, self.identifier.state, target,
                anchor_moments, rng, trust=trust, explore=explore,
                measured_at=values, position_weight=0.0,
            )
            if self.record:
                self.recorded.append(
                    (observed, expert_action(values, plan.values, knobs.scales))
                )
            values = plan.values
            self._apply(env, knobs, values, history)
            achieved = _mismatch(
                history[-1], target, cap=UNCAPPED, position_weight=0.0
            )

            # Trust the twin as far as it keeps being right. A step that made the
            # reading worse — above all one that threw the beam off the screen —
            # means the twin was wrong about this stretch, so go back to the best
            # setting found and try again from there, more carefully. Without it a
            # lost beam is unrecoverable: there is no reading left to anchor on, and
            # the planner spends the rest of the episode steering by a model it has
            # just been shown to be wrong.
            size_now = _mismatch(history[-1], target, position_weight=0.0)
            if size_now < best_mismatch:
                stalled = 0
                best_mismatch = size_now
                best_values, best_moments = values.copy(), history[-1].moments
                anchor_moments = best_moments
                trust = min(trust * TRUST_GROWTH, self.config.trust_region)
                explore = False
            else:
                stalled += 1
                values, anchor_moments = best_values.copy(), best_moments
                self.twin.set_knob_vector(knobs.elements, knobs.props, values)
                trust = max(trust * TRUST_DECAY, self.config.min_trust_region)
                explore = True

            trace.append({
                "stage": "focus",
                "shot": len(history) - 1,
                "mismatch": self._mismatch(history[-1], target),
                "raw_mismatch": achieved,
                "predicted": plan.cost,
                "predicted_moments": np.asarray(plan.predicted).tolist(),
                "achieved_moments": (
                    None if not history[-1].on_screen else history[-1].moments.tolist()
                ),
                "on_screen": history[-1].on_screen,
            })
            if verbose:
                print(
                    f"  focus {step:2d}   mismatch {achieved:7.3f}  "
                    f"(predicted {plan.cost:6.3f})  "
                    f"{'on screen' if history[-1].on_screen else 'LOST'}"
                )

        # The benchmark scores the last reading, not the best one, so a tune that
        # found a good spot and wandered off books the wandering. One shot to put
        # the best setting back is cheap insurance on a machine that jitters.
        if _mismatch(history[-1], target, position_weight=0.0) > best_mismatch:
            self._apply(env, knobs, best_values, history)
            trace.append({
                "stage": "restore",
                "shot": len(history) - 1,
                "mismatch": self._mismatch(history[-1], target),
                "raw_mismatch": self._mismatch(history[-1], target, cap=UNCAPPED),
                "on_screen": history[-1].on_screen,
            })
            if verbose:
                print(
                    f"  restore    mismatch "
                    f"{self._mismatch(history[-1], target, cap=UNCAPPED):7.3f}  "
                    f"(best seen {best_mismatch:.3f})"
                )

    def _observe(self, shot, target, knobs, values, reference, trust) -> np.ndarray:
        """What a policy would see at this point in the tune."""
        return observe(
            shot.moments if shot.on_screen else None,
            target,
            shot.transmission,
            self._orbit_rms(shot),
            float(shot.bpm_valid.mean()),
            (values - reference) / np.maximum(knobs.scales, 1e-30),
            0.0,
            trust,
        )

    def _propose(self, knobs: KnobSet, values: np.ndarray, observed) -> np.ndarray:
        """Where the policy thinks the knobs should go, or where they already are.

        The policy is a warm start, not a replacement: the planner still refines
        whatever it suggests, so a bad suggestion costs a little search and not the
        shot. With no policy loaded this is the identity and the agent behaves
        exactly as it did before there was one.
        """
        if self.policy is None:
            return values
        step = pad_action(self.policy.propose(observed), len(knobs))
        return knobs.clip(values + step * knobs.scales)

    def _calibrate(self, knobs: KnobSet, values: np.ndarray, target: np.ndarray) -> None:
        """Set each knob's step to whatever moves the readout by a useful amount.

        One batched track gives the response of all four measured moments to all the
        control knobs at once, and a step is then defined as the move that changes
        the reading by `trust_effect` of the requested spot. Without this the step
        sizes are physical constants, and no single constant fits: 0.1 mrad is a
        large kick from a corrector near the injector and nothing at all from the
        one just before the screen, which is the one that has to carry the spot the
        two millimetres out to where it was asked for.
        """
        jacobian, _ = self.planner.response(
            knobs, values, self.identifier.state, lambda r: r.planning_moments()
        )
        spot = float(np.asarray(target)[2:].mean())
        # Each knob's reach over the score, in requested spot sizes per unit: the
        # same combination `_mismatch` books, so a step is a step of the objective.
        sensitivity = (
            np.abs(jacobian[:2]).sum(axis=0) / 2 + np.abs(jacobian[2:]).sum(axis=0) / 2
        ) / spot
        knobs.rescale(sensitivity, self.config.trust_effect)

    # ---------- stage five: put the spot where it was asked for ----------

    def _place(self, env, target, history, trace, verbose) -> None:
        """One round of SVD steering onto the requested position.

        The focus has matched the size and left the spot wherever the optics put it;
        the request wants it somewhere within 2 mm of the screen centre, which is
        most of the remaining score. The readout centroid is linear in the kicks of
        the correctors just upstream of it, so this is a two-row least-squares solve
        and not a search — the response comes from the twin, the error from the
        camera, and one shot normally closes it.

        It is last because these correctors move almost nothing else. Putting them
        in the focus instead gives a quadrupole the option of fixing a position
        error by distorting the optics, which costs the size.
        """
        knobs = KnobSet.from_knob_names(
            self.config.placement_correctors, self.twin.element_types, env.magnet_limits
        )
        values = knobs.read(env.magnet_settings())
        for step in range(self.config.placement_shots):
            shot = history[-1]
            if not shot.on_screen or shot.moments is None:
                break
            error = np.asarray(shot.moments[:2]) - np.asarray(target[:2])
            jacobian, _ = self.planner.response(
                knobs,
                values,
                self.identifier.state,
                lambda r: torch.stack([r.mu_x, r.mu_y], dim=-1),
            )
            correction = -_pseudo_inverse(jacobian, self.config.placement_cutoff) @ error
            low, high = knobs.trust_region(values, PLACEMENT_REACH)
            values = np.clip(values + correction, low, high)
            self._apply(env, knobs, values, history)
            trace.append({
                "stage": "place",
                "shot": len(history) - 1,
                "mismatch": _mismatch(history[-1], target),
                "raw_mismatch": _mismatch(history[-1], target, cap=UNCAPPED),
                "position_um": (
                    None if not history[-1].on_screen
                    else (np.asarray(history[-1].moments[:2]) * 1e6).tolist()
                ),
                "on_screen": history[-1].on_screen,
            })
            if verbose:
                moved = np.abs(np.asarray(history[-1].moments[:2]) - target[:2]) * 1e6 \
                    if history[-1].on_screen else None
                print(
                    f"  place {step:2d}   mismatch "
                    f"{_mismatch(history[-1], target, cap=UNCAPPED):7.3f}  "
                    + (f"position off by {moved[0]:.0f}, {moved[1]:.0f} um"
                       if moved is not None else "LOST")
                )

    # ---------- the machinery each shot shares ----------

    def _identify(self, history: list[Shot], first: bool) -> None:
        """Refit the twin to the shots taken so far.

        The first fit of a stage gets the full iteration budget; later ones are warm
        and need a fraction of it. Not every shot earns one: the machine holds still
        for the whole episode, so the tenth shot tells the fit far less than the
        second did while costing the same.
        """
        self._since_fit = 0 if first else self._since_fit + 1
        if not first and self._since_fit % max(1, self.config.id_interval):
            return
        self.identifier.fit(
            history[-self.config.id_history :],
            iterations=self.config.id_iterations if first
            else max(1, self.config.id_iterations // 4),
            learning_rate=self.config.id_learning_rate,
        )

    def _apply(self, env, knobs: KnobSet, values: np.ndarray, history: list[Shot]):
        """Set the magnets, take the shot, and record what came back."""
        observation, _, _, _, info = env.step(knobs.action(values))
        self.twin.set_knob_vector(knobs.elements, knobs.props, values)
        history.append(self._shot(env, observation, info))
        return observation, info

    def _shot(self, env, observation, info) -> Shot:
        """One reading, with the magnet settings it was taken at.

        The spot comes off the camera frame through `screen.measure`, not from the
        particle moments in `info`: the agent is driven by the same quantity it would
        have on the machine. `info` is kept only so the episode can be booked against
        the benchmark at the end.
        """
        knobs = {}
        for name, value in env.magnet_settings().items():
            element, prop = _split(name, self.twin)
            knobs[(element, prop)] = float(value)
        reading = measure(observation["screen_image"], self._extent)
        shot = Shot(
            knobs=knobs,
            bpm_x=np.asarray(observation["bpm_x"], dtype=float),
            bpm_y=np.asarray(observation["bpm_y"], dtype=float),
            bpm_valid=np.asarray(observation["bpm_valid"], dtype=float),
            moments=reading.moments if reading.ok else None,
            on_screen=reading.ok,
            transmission=float(info["transmission"]),
        )
        shot.target = np.asarray(observation["target"], dtype=float)
        shot.info = info
        shot.reading = reading
        return shot

    def _mismatch(self, shot: Shot, target: np.ndarray, cap: float = 10.0) -> float:
        return _mismatch(shot, target, cap)

    def _result(self, history, trace, steering_shots, optics, started) -> EpisodeResult:
        """The episode as the benchmark books it, and what it cost to get there."""
        info = history[-1].info
        last = history[-1]
        # Every shot names the screen that was in the beam; a switch is a change
        # between consecutive shots. This agent should never make one — all 31 BPMs
        # sit upstream of the readout, so the beam can be steered on them without
        # moving anything into it — and counting is how that stays true.
        viewed = [shot.info["screen"] for shot in history if shot.info]
        switches = sum(a != b for a, b in zip(viewed, viewed[1:]))
        shots = int(info["evaluations"])
        seconds = time.monotonic() - started
        optics_entry = next((e for e in trace if e["stage"] == "optics"), {})
        return EpisodeResult(
            benchmark_score=float(info["benchmark_score"]),
            mismatch=float(info["scored_mismatch"]),
            relative_error=float(info["scored_relative_error"]),
            shots=int(info["evaluations"]),
            on_screen=bool(info["on_screen"]),
            transmission=float(info["transmission"]),
            target=np.asarray(info["target"], dtype=float),
            achieved=np.array(
                [info["mu_x"], info["mu_y"], info["sigma_x"], info["sigma_y"]]
            ),
            steering_shots=steering_shots,
            orbit_rms=self._orbit_rms(last),
            valid_bpms=int(last.bpm_valid.sum()),
            optics_score=float(optics_entry.get("mismatch", float("nan"))),
            optics_error=float(optics.relative_error),
            screen_switches=switches,
            seconds=seconds,
            machine_seconds=shots * SHOT_SECONDS + switches * SCREEN_SWITCH_SECONDS,
            trace=trace,
        )


def _split(name: str, twin: CheetahTwin) -> tuple[str, str]:
    """A SimEnv knob name back into the element and property the twin knows it by."""
    if name.endswith("_h"):
        return name[:-2], "horizontal_angle"
    if name.endswith("_v"):
        return name[:-2], "vertical_angle"
    return name, twin.knob_property(name)


def _mismatch(
    shot: Shot, target: np.ndarray, cap: float = 10.0, position_weight: float = 1.0
) -> float:
    """`SimEnv._mismatch` for a shot: how far off the request, in requested spots.

    Position and size against the spot that was asked for, plus `TRANSMISSION_WEIGHT`
    for the fraction of the bunch that never reached the readout — which is easy to
    forget and worth up to two whole spot sizes.

    `cap` is the benchmark's 10 when booking a score, and much larger when the number
    is being watched during a tune: a capped trace reads as a flat line whatever the
    planner is actually doing.
    """
    if not shot.on_screen or shot.moments is None:
        return cap
    error = np.abs(shot.moments - target) / float(target[2:].mean())
    total = (error[2] + error[3]) / 2 + TRANSMISSION_WEIGHT * (1.0 - shot.transmission)
    if position_weight:
        total += position_weight * (error[0] + error[1]) / 2
    return float(min(total, cap))
