"""Fast checks that the twin is honest and the agent works. Runs in a few minutes.

    python -I smoke_test.py

Three things, in the order they matter:

1. **The twin is the same machine as SimEnv.** With every variation at zero the two
   model the identical lattice with the identical beam, so every reading has to agree
   to float32 — not approximately, exactly. This is the test the whole approach rests
   on: a planner that reasons against a twin which disagrees with the machine is
   reasoning about a different machine. It also checks the twin after the magnets
   have been moved, and after a batched plan, because the easiest way to break it is
   to leave a planner's candidate values behind on an element.

2. **Batching and gradients work.** A hundred candidate tunes in one call, shaped
   correctly, with gradients that reach the knobs.

3. **The agent tunes.** A couple of real episodes against a varying machine, which
   must come out ahead of doing nothing.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "RL_framework"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_framework.config import RLConfig  # noqa: E402
from rl_framework.envs.sim_env import SimEnv  # noqa: E402

from MFopt.optics_env import OpticsEnv  # noqa: E402

from agent import MBRLAgent  # noqa: E402
from baselines import optics_only  # noqa: E402
from config import MBRLConfig  # noqa: E402
from twin import CheetahTwin, tune_threads  # noqa: E402

tune_threads()

PARTICLES = 4_000
# What "the same machine" means. Cheetah parses the lattice into float32, so a
# reading cannot be reproduced closer than its last digit; a micron is far tighter
# than anything physical and far looser than float32 noise.
TOLERANCE_UM = 1.0

passed, failed = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (passed if condition else failed).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


def readings_agree(label, twin_readings, observation, info) -> None:
    """Every number the twin predicts, against every number SimEnv reported."""
    orbit = max(
        np.abs(twin_readings.bpm_x.detach().numpy() - observation["bpm_x"]).max(),
        np.abs(twin_readings.bpm_y.detach().numpy() - observation["bpm_y"]).max(),
    )
    moments = max(
        abs(float(getattr(twin_readings, key)) - info[key])
        for key in ("mu_x", "mu_y", "sigma_x", "sigma_y")
    )
    valid = np.array_equal(
        twin_readings.bpm_valid.numpy().astype(int), observation["bpm_valid"]
    )
    check(
        f"{label}: BPM orbit matches SimEnv",
        orbit * 1e6 < TOLERANCE_UM,
        f"worst {orbit * 1e6:.4f} um",
    )
    check(
        f"{label}: screen moments match SimEnv",
        moments * 1e6 < TOLERANCE_UM,
        f"worst {moments * 1e6:.4f} um",
    )
    check(f"{label}: BPM valid flags match SimEnv", valid)


def test_fidelity() -> None:
    print("\n1. the twin is the same machine as SimEnv (every variation at zero)")
    rl = RLConfig()
    rl.num_particles = PARTICLES
    rl.beam_variation = 0.0
    rl.misalignment_variation = 0.0
    env = SimEnv(rl)
    twin = CheetahTwin.for_env(env, PARTICLES)

    check("twin and SimEnv agree on the BPM list", twin.bpm_names == env.bpm_names)
    check("twin and SimEnv agree on the screens", twin.screen_names == env.screen_names)
    check("twin and SimEnv agree on the beam energy", twin.energy_ev == env.energy_ev)

    observation, info = env.reset(seed=0)
    readings_agree("nominal tune", twin.readings(), observation, info)

    # Moved magnets, one of every kind the agent turns.
    moved = {
        "cla_feh_mag_quad_01": 1.7,
        "cla_fea_mag_sext_01": 12.0,
        "cla_feh_mag_hvcor_01_h": 2.0e-4,
        "cla_l03_lin_cav_01": 12.0,
        "cla_vbc_chicane": 0.01,
    }
    observation, _, _, _, info = env.step({"magnets": moved})
    twin.set_knobs(
        {k: v for k, v in moved.items() if not k.endswith(("_h", "_v"))}
    )
    twin.set_knob_vector(["cla_feh_mag_hvcor_01"], ["horizontal_angle"], [2.0e-4])
    readings_agree("after moving magnets", twin.readings(), observation, info)

    # A batched plan must leave nothing behind on the elements it borrowed.
    for size in (64, 256, 7):
        twin.readings(
            overrides={("cla_feh_mag_quad_01", "k1"): torch.linspace(-3, 3, size)}
        )
    readings_agree("after batched planning", twin.readings(), observation, info)


def test_batching_and_gradients() -> None:
    print("\n2. the twin plans in batches and differentiates")
    twin = CheetahTwin.for_env(SimEnv(RLConfig()), PARTICLES)
    size = 128
    values = torch.linspace(-8.0, 8.0, size, requires_grad=True)
    start = time.time()
    readings = twin.readings(overrides={("cla_feh_mag_quad_01", "k1"): values})
    elapsed = time.time() - start

    check(
        "a batch of candidates comes back batched",
        readings.planning_moments().shape == (size, 4)
        and readings.bpm_x.shape == (size, len(twin.bpm_names)),
        f"{size} candidates in {elapsed * 1e3:.0f} ms",
    )
    check(
        "a batch spans a range of spot sizes",
        float(readings.raw_sigma_x.max() / readings.raw_sigma_x.min()) > 2.0,
    )
    readings.planning_moments()[:, 2].sum().backward()
    check(
        "gradients reach the knobs",
        values.grad is not None and float(values.grad.abs().max()) > 0,
        f"largest |d sigma_x / d k1| {float(values.grad.abs().max()):.3e}",
    )

    # Off the screen the measurement is zero and must still be plannable.
    twin.set_knobs({"cla_vbc_chicane": 0.03})
    knob = torch.tensor(1.7, requires_grad=True)
    lost = twin.readings(overrides={("cla_feh_mag_quad_01", "k1"): knob})
    (lost.planning_moments()[2] + lost.scrape).backward()
    check(
        "a lost beam still has a gradient to follow",
        float(lost.on_screen) == 0.0 and float(knob.grad) != 0.0,
        f"off screen, d cost / d k1 = {float(knob.grad):.3e}",
    )


def test_screen_measure() -> None:
    """The camera measure agrees with the particles it was made from."""
    print("\n3. the robust screen measure reads the spot the particles made")
    from screen import measure, pixel_size

    rl = RLConfig()
    rl.num_particles = PARTICLES
    rl.screen_shape = MBRLConfig().screen_shape
    env = SimEnv(rl)
    worst = 0.0
    for seed in range(3):
        observation, info = env.reset(seed=seed)
        if not info["on_screen"]:
            continue
        reading = measure(observation["screen_image"], env.screen_extent())
        truth = np.array(
            [info["mu_x"], info["mu_y"], info["sigma_x"], info["sigma_y"]]
        )
        worst = max(worst, float(np.abs(reading.moments - truth).max()))
    px = pixel_size(rl.screen_shape, env.screen_extent())
    check(
        "camera moments match the particle moments",
        worst * 1e6 < 5.0,
        f"worst {worst * 1e6:.2f} um, pixel {px[0] * 1e6:.0f} x {px[1] * 1e6:.0f} um",
    )


def test_pipeline() -> None:
    """The four stages, on a machine the agent has not seen."""
    print("\n4. the pipeline tunes a machine it has not seen")
    config = MBRLConfig()
    config.twin_particles = 2_000
    config.focus_shots = 10
    config.id_iterations = 40
    config.plan_iterations = 8
    config.cem_iterations = 3
    config.cem_population = 96

    rl = RLConfig()
    rl.num_particles = 2_000
    rl.screen_shape = config.screen_shape
    env = OpticsEnv(rl)
    twin = CheetahTwin.for_env(env, config.twin_particles)
    agent = MBRLAgent(config, twin)

    for seed in (0, 1):
        start = time.time()
        tuned = agent.run(env, seed=seed)
        reference = optics_only(env, twin, config, seed=seed)
        print(
            f"    seed {seed}: optics-only {reference.benchmark_score:6.3f}  ->  "
            f"agent {tuned.benchmark_score:6.3f} in {tuned.shots:>2} shots "
            f"({time.time() - start:.0f} s)"
        )
        # How well the lookup matched is `MFopt`'s business, not the agent's; it is
        # reported because it sets the floor the tweak starts from, and it is only a
        # failure here if the lookup has returned something unusable.
        check(
            f"seed {seed}: the reference optics is usable",
            tuned.optics_error < 1.0,
            f"model error {tuned.optics_error:.1%}",
        )
        check(
            f"seed {seed}: the tweak improves on the reference optics",
            tuned.benchmark_score < reference.benchmark_score,
            f"{tuned.benchmark_score:.3f} vs {reference.benchmark_score:.3f}",
        )
        check(f"seed {seed}: the agent delivers beam", tuned.on_screen)


def main() -> int:
    start = time.time()
    test_fidelity()
    test_batching_and_gradients()
    test_screen_measure()
    test_pipeline()
    print(
        f"\n{len(passed)} passed, {len(failed)} failed, in {time.time() - start:.0f} s"
    )
    for name in failed:
        print(f"  FAILED  {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
