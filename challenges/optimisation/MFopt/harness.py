"""Running episodes the way the machine would: count every step, keep every reading.

A policy is a function `policy(session)` that drives one episode through a
`Session`: it reads `session.target`, `session.last` (the newest observation and
info) and `session.history`, and calls `session.step(action)` for each shot. The
episode is scored as `SimEnv` scores it: on the last reading of the target screen,
plus the cost of the steps taken (`info["benchmark_score"]`).

What a policy may use is what the machine gives: the observation and the info
dictionary of each step. The hidden truth (`env.misalignments`, `env.beam_offset`,
the beam's coordinates) never reaches it. The error-free model (`env.twin`,
`env.optics_for_target`) is ours and free to use.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .optics_env import OpticsEnv

MAX_STEPS = 50


@dataclass
class Reading:
    """One shot: what was set, and what came back."""

    action: dict
    obs: dict
    info: dict


@dataclass
class Session:
    """One episode as a policy sees it."""

    env: OpticsEnv
    target: np.ndarray  # (mu_x, mu_y, sigma_x, sigma_y), metres
    history: list[Reading] = field(default_factory=list)
    max_steps: int = MAX_STEPS

    @property
    def last(self) -> Reading:
        return self.history[-1]

    @property
    def steps(self) -> int:
        return self.last.info["evaluations"]

    @property
    def steps_left(self) -> int:
        return self.max_steps - self.steps

    def step(self, action: dict) -> Reading:
        if self.steps_left <= 0:
            raise RuntimeError(f"the episode's {self.max_steps} steps are spent")
        obs, _, _, _, info = self.env.step(action)
        reading = Reading(action=action, obs=obs, info=info)
        self.history.append(reading)
        return reading

    def settings(self) -> dict[str, float]:
        """What every magnet is set to now (we set them, so we know)."""
        return dict(self.last.info["magnets"])


@dataclass
class Result:
    seed: int
    target: list[float]
    steps: int
    score: float  # the benchmark score: mismatch + step cost, lower is better
    mismatch: float
    relative_error: float
    transmission: float
    achieved: list[float]  # mu_x, mu_y, sigma_x, sigma_y at the last scored reading
    seconds: float
    trace: list[dict]  # per step: mismatch, transmission, sizes and centroid


def run_episode(env: OpticsEnv, policy: Callable[[Session], Any], seed: int) -> Result:
    start = time.time()
    obs, info = env.reset(seed=seed)
    session = Session(env=env, target=np.array(obs["target"], dtype=float))
    session.history.append(Reading(action={}, obs=obs, info=info))
    policy(session)
    final = session.last.info
    trace = [
        {
            "step": r.info["evaluations"],
            "screen": r.info["screen"],
            "mismatch": r.info["scored_mismatch"],
            "transmission": r.info["transmission"],
            "mu": [r.info["mu_x"], r.info["mu_y"]],
            "sigma": [r.info["sigma_x"], r.info["sigma_y"]],
        }
        for r in session.history
    ]
    return Result(
        seed=seed,
        target=[float(v) for v in session.target],
        steps=session.steps,
        score=float(final["benchmark_score"]),
        mismatch=float(final["scored_mismatch"]),
        relative_error=float(final["scored_relative_error"]),
        transmission=float(final["scored_transmission"]),
        achieved=[float(final[k]) for k in ("mu_x", "mu_y", "sigma_x", "sigma_y")],
        seconds=time.time() - start,
        trace=trace,
    )


def summarise(results: list[Result]) -> dict:
    """The benchmark's view of a set of episodes."""
    score = np.array([r.score for r in results])
    error = np.array([r.relative_error for r in results])
    return {
        "episodes": len(results),
        "score_mean": float(score.mean()),
        "score_median": float(np.median(score)),
        "hit_50": float((error <= 0.50).mean()),
        "hit_25": float((error <= 0.25).mean()),
        "hit_10": float((error <= 0.10).mean()),
        "transmission_median": float(np.median([r.transmission for r in results])),
        "steps_mean": float(np.mean([r.steps for r in results])),
    }


def save(results: list[Result], path: str) -> None:
    with open(path, "w") as f:
        json.dump({"summary": summarise(results),
                   "episodes": [r.__dict__ for r in results]}, f, indent=1)
