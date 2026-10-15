"""What the agent has to beat, on the same episodes.

The pipeline has stages, so the honest baselines are the pipeline with its later
stages removed. That says what each stage is worth rather than only whether the whole
thing works.

`do_nothing` hands the lattice-file tune back untouched. It is where the episode
starts and it is not a strawman: the nominal optics do carry beam to the hutch, they
just deliver whatever spot they deliver.

`optics_only` loads the reference optics for the requested spot and stops — one shot,
no feedback. On the error-free model this is already the right answer, so what it
scores on a real episode is exactly the cost of the machine not being the model: the
beam the day dealt, the survey error, and a requested *position* that the matched
family never matched for.

`optics_steered` adds the SVD steering and stops before the focus. The gap between
this and the agent is what the model-based tweak is worth.
"""

from __future__ import annotations

import numpy as np

from agent import MBRLAgent
from config import MBRLConfig
from twin import CheetahTwin


def _staged(env, twin, config, seed, stages: str):
    """Run the agent with its later stages switched off."""
    agent = MBRLAgent(config, twin)
    if stages == "optics":
        agent.config = _without(config, steering_shots=0, focus_shots=0)
    elif stages == "steered":
        agent.config = _without(config, focus_shots=0)
    agent.planner.config = agent.config
    return agent.run(env, seed=seed)


def _without(config: MBRLConfig, **overrides) -> MBRLConfig:
    """A copy of `config` with some budgets zeroed."""
    return MBRLConfig(**{**config.__dict__, **overrides})


def do_nothing(env, twin: CheetahTwin, config: MBRLConfig, seed=None):
    """Hand the lattice-file tune straight back, without even loading the optics."""
    from agent import EpisodeResult, _mismatch

    observation, info = env.reset(seed=seed)
    valid = np.asarray(observation["bpm_valid"], dtype=float).astype(bool)
    x = np.asarray(observation["bpm_x"], dtype=float)
    y = np.asarray(observation["bpm_y"], dtype=float)
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
        steering_shots=0,
        screen_switches=0,
        seconds=0.0,
        machine_seconds=float(info["evaluations"]),
        orbit_rms=(
            float(np.sqrt((x[valid] ** 2 + y[valid] ** 2).mean() / 2))
            if valid.any()
            else float("inf")
        ),
        valid_bpms=int(valid.sum()),
    )


def optics_only(env, twin: CheetahTwin, config: MBRLConfig, seed=None):
    """Load the reference optics for the request, and stop."""
    return _staged(env, twin, config, seed, "optics")


def optics_steered(env, twin: CheetahTwin, config: MBRLConfig, seed=None):
    """Reference optics plus the SVD steering, without the focus."""
    return _staged(env, twin, config, seed, "steered")


BASELINES = {
    "do-nothing": do_nothing,
    "optics-only": optics_only,
    "optics-steered": optics_steered,
}
