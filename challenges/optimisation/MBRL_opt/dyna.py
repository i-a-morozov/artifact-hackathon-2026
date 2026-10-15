"""A machine made of the twin, so the agent can be run — and trained — for free.

`TwinMachine` presents the handful of methods `MBRLAgent` uses on an environment —
`reset`, `step`, `magnet_settings`, `magnet_limits`, `screen_extent`,
`optics_for_target`, `optics_action` — backed by a `CheetahTwin` carrying a randomly
drawn `LatentState` instead of by `SimEnv`. The agent cannot tell the difference, and
that is the point: the training data comes out of the real deployment code path
rather than a reimplementation of it that can quietly drift.

What it is for is the Dyna layer. The twin is a complete simulator of the machine, so
rolling episodes through it costs nothing but arithmetic — no beam, no shot budget,
no forty seconds per shot of real-machine time. Randomise the latent from the same
spreads `SimEnv` deals from, run the planner, and record what it decided.

Two things it is deliberately *not*. It is not a second model of the physics: it is
the same `CheetahTwin` class the agent plans against, so a policy trained here has
never seen a disagreement between model and machine and will not have learned to
correct for one. And it is not a substitute for evaluation — the numbers that count
still come from `SimEnv`, which has its own beam, its own survey error and its own
shot noise. This exists to generate decisions cheaply, not to score them.

    machine = TwinMachine(optics_source=real_env, seed=0)
    result = MBRLAgent(config, twin).run(machine, seed=0)
"""

from __future__ import annotations

import numpy as np
import torch

from twin import (
    BETA_SPREAD,
    CENTROID_JITTER,
    DIAGNOSTIC_OFFSET_RMS,
    DTYPE,
    MISALIGNMENT_RMS,
    SIGMA_SPREAD,
    CheetahTwin,
    LatentState,
)

# What the benchmark charges, repeated here so a twin episode books the same score
# as a real one. `sim_env.EVAL_WEIGHT`, `EVAL_BUDGET`, `TRANSMISSION_WEIGHT`.
EVAL_WEIGHT = 0.5
EVAL_BUDGET = 50.0
TRANSMISSION_WEIGHT = 2.0
FAILURE_SCORE = 10.0
# What the beam is asked for, as `score.sample_target` asks for it.
REQUESTED_SPOT = (80e-6, 400e-6)
REQUESTED_OFFSET = 2e-3


def draw_latent(
    rng: np.random.Generator,
    variation: float = 1.0,
    misalignment: float = 1.0,
    n_magnets: int = 0,
    n_diagnostics: int = 0,
) -> LatentState:
    """A machine, drawn the way `SimEnv.reset` deals one.

    Everything is in units of its own prior spread, so this is a standard normal for
    the survey error and a uniform for the launch offset, exactly matching the
    distributions `sim_env` samples and `system_id` fits under.
    """
    tensor = lambda a: torch.tensor(np.asarray(a), dtype=DTYPE)  # noqa: E731
    # `_perturb_beam` draws the offset uniform on +-CENTROID_JITTER * variation,
    # which in prior units is uniform on +-variation.
    centroid = rng.uniform(-variation, variation, 6)
    # and the size factors log-uniform, widening as sqrt(variation).
    reach = np.sqrt(variation)
    return LatentState(
        u_centroid=tensor(centroid),
        u_log_size=tensor(rng.uniform(-reach, reach, 4)),
        u_log_mismatch=tensor(
            rng.uniform(-reach, reach, 2) * np.log(BETA_SPREAD) / np.log(SIGMA_SPREAD)
        ),
        u_misalignments=(
            tensor(rng.normal(0, misalignment, (n_magnets, 2))) if n_magnets else None
        ),
        u_diagnostic_offsets=(
            tensor(rng.normal(0, misalignment, (n_diagnostics, 2)))
            if n_diagnostics
            else None
        ),
    )


def draw_target(rng: np.random.Generator) -> np.ndarray:
    """A request, as `score.sample_target` and `SimEnv._draw_target` draw one."""
    low, high = REQUESTED_SPOT
    return np.array(
        [
            rng.uniform(-REQUESTED_OFFSET, REQUESTED_OFFSET),
            rng.uniform(-REQUESTED_OFFSET, REQUESTED_OFFSET),
            float(np.exp(rng.uniform(np.log(low), np.log(high)))),
            float(np.exp(rng.uniform(np.log(low), np.log(high)))),
        ]
    )


class TwinMachine:
    """A `CheetahTwin` dressed as the environment the agent expects."""

    def __init__(
        self,
        optics_source,
        twin: CheetahTwin | None = None,
        num_particles: int = 4_000,
        screen_shape: tuple[int, int] = (512, 512),
        beam_variation: float = 1.0,
        misalignment_variation: float = 1.0,
        beam_jitter: float = 0.1,
        fit_misalignments: bool = True,
        seed: int | None = None,
    ):
        # The optics lookup is model-only and expensive, so it is borrowed from a
        # real `OpticsEnv` rather than rebuilt; its cache then serves every episode
        # that asks for the same spot.
        self._optics = optics_source
        self.twin = twin or CheetahTwin(num_particles=num_particles)
        self.magnet_limits = dict(self._optics.magnet_limits)
        self._shape = tuple(int(n) for n in screen_shape)
        self._variation = beam_variation
        self._misalignment = misalignment_variation
        self._jitter = beam_jitter
        # Whether the drawn machine carries a survey error at all. Off leaves the
        # twin's own alignment, which is what the aligned-twin ablation assumes.
        self._survey = fit_misalignments
        self._rng = np.random.default_rng(seed)
        self.target = draw_target(self._rng)
        self._latent = LatentState.nominal()
        self._settings: dict[str, float] = {}
        self._steps = 0
        self._scored = (0.0, 0.0, 0.0, 0.0, False)
        self._scored_transmission = 0.0

    # ---------- the environment's surface ----------

    def screen_extent(self, screen: str | None = None):
        return self.twin.screen_extent(screen)

    def optics_for_target(self, sigma_x: float, sigma_y: float):
        return self._optics.optics_for_target(sigma_x, sigma_y)

    def optics_action(self, optics) -> dict:
        return {"magnets": dict(optics.k1)}

    def magnet_settings(self) -> dict[str, float]:
        """Every settable magnet, at the lattice value unless it has been moved."""
        out = {}
        for name in self.magnet_limits:
            if name in self._settings:
                out[name] = self._settings[name]
            else:
                element, prop = _split(name)
                out[name] = self.twin.nominal_setting(element, prop)
        return out

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.target = draw_target(self._rng)
        self._latent = draw_latent(
            self._rng,
            self._variation,
            self._misalignment if self._survey else 0.0,
            len(self.twin.misalignable) if self._survey else 0,
            len(self.twin.diagnostic_names) if self._survey else 0,
        )
        self._settings.clear()
        self.twin.reset_setpoints()
        self._steps = 0
        return self._read()

    def step(self, action):
        settings = action.get("magnets", action) if isinstance(action, dict) else {}
        for name, value in settings.items():
            low, high = self.magnet_limits[name]
            self._settings[name] = float(np.clip(float(value), low, high))
            element, prop = _split(name)
            self.twin.set_knob_vector([element], [prop], [self._settings[name]])
        self._steps += 1
        observation, info = self._read()
        return observation, 0.0, False, False, info

    # ---------- taking a shot ----------

    def _shot_latent(self) -> LatentState:
        """The episode's machine, jittered as `SimEnv` jitters each shot."""
        if not self._jitter:
            return self._latent
        reach = self._variation * self._jitter
        shot = self._latent.detached()
        shot.u_centroid = shot.u_centroid + torch.tensor(
            self._rng.uniform(-reach, reach, 6), dtype=DTYPE
        )
        spread = np.sqrt(reach)
        shot.u_log_size = shot.u_log_size + torch.tensor(
            self._rng.uniform(-spread, spread, 4), dtype=DTYPE
        )
        return shot

    def _read(self):
        with torch.no_grad():
            readings = self.twin.readings(
                state=self._shot_latent(), image_shape=self._shape
            )
        stats = (
            float(readings.mu_x),
            float(readings.mu_y),
            float(readings.sigma_x),
            float(readings.sigma_y),
            bool(float(readings.on_screen)),
        )
        self._scored = stats
        self._scored_transmission = float(readings.transmission)
        observation = {
            "bpm_x": readings.bpm_x.numpy().astype(np.float32),
            "bpm_y": readings.bpm_y.numpy().astype(np.float32),
            "bpm_valid": readings.bpm_valid.numpy().astype(np.int8),
            "screen_image": readings.image.numpy().astype(np.float32),
            "target": self.target.astype(np.float32),
        }
        return observation, self._info(stats)

    def _info(self, stats) -> dict:
        mu_x, mu_y, sigma_x, sigma_y, on_screen = stats
        mismatch = self._mismatch(stats, self._scored_transmission)
        cost = EVAL_WEIGHT * float(np.log2(1 + self._steps / EVAL_BUDGET))
        return {
            "screen": self.twin.readout_screen,
            "target": self.target,
            "evaluations": self._steps,
            "scored_mismatch": mismatch,
            "scored_relative_error": self._relative_error(stats),
            "benchmark_score": mismatch + cost,
            "mu_x": mu_x,
            "mu_y": mu_y,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "on_screen": on_screen,
            "transmission": self._scored_transmission,
        }

    def _mismatch(self, stats, transmission, cap: float = FAILURE_SCORE) -> float:
        """`SimEnv._mismatch`, so a twin episode is booked like a real one."""
        mu_x, mu_y, sigma_x, sigma_y, on_screen = stats
        if not on_screen:
            return cap
        error = np.abs(np.array([mu_x, mu_y, sigma_x, sigma_y]) - self.target) / float(
            self.target[2:].mean()
        )
        return float(
            min(
                (error[0] + error[1]) / 2
                + (error[2] + error[3]) / 2
                + TRANSMISSION_WEIGHT * (1.0 - transmission),
                cap,
            )
        )

    def _relative_error(self, stats) -> float:
        mu_x, mu_y, sigma_x, sigma_y, on_screen = stats
        if not on_screen:
            return FAILURE_SCORE
        achieved = np.array([mu_x, mu_y, sigma_x, sigma_y])
        return float(np.abs(achieved - self.target).mean() / self.target[2:].mean())


def _split(name: str) -> tuple[str, str]:
    """A knob name into the element and property the twin knows it by."""
    if name.endswith("_h"):
        return name[:-2], "horizontal_angle"
    if name.endswith("_v"):
        return name[:-2], "vertical_angle"
    if "mag_quad" in name:
        return name, "k1"
    if "mag_sext" in name:
        return name, "k2"
    if name == "cla_vbc_chicane":
        return name, "angle"
    return name, "phase"
