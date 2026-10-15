"""Train the Dyna policy: roll episodes inside the twin, clone the planner's moves.

    python -I train.py --episodes 200 --workers 16
    python -I train.py --episodes 400 --workers 32 --out models/focus.pt

No beam time is spent. The machine each episode is a `TwinMachine` — the same
`CheetahTwin` the agent plans against, carrying a latent drawn from the spreads
`SimEnv` deals from — so an episode costs arithmetic and nothing else. The agent runs
against it unmodified, recording what the planner decided at each focus shot, and the
policy is fitted to those decisions.

Cloning rather than policy gradient, because the planner is already a good expert and
what the policy is for is speed: it proposes in a millisecond what the planner takes
seconds to re-derive. Deployed, it is a warm start the planner still refines, so a bad
suggestion costs a little search rather than a shot.

The targets are drawn once and shared, because `optics_for_target` is the one
expensive thing here — forty seconds of model time per distinct request — and a
worker that sees the same request twice pays for it once.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "RL_framework"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_framework.config import RLConfig  # noqa: E402

from MFopt.optics_env import OpticsEnv  # noqa: E402

from agent import MBRLAgent  # noqa: E402
from config import MBRLConfig  # noqa: E402
from dyna import TwinMachine  # noqa: E402
from policy import PolicyConfig, clone  # noqa: E402
from twin import CheetahTwin, tune_threads  # noqa: E402

tune_threads()

_WORKER: dict = {}


def _init(config: MBRLConfig, targets: np.ndarray) -> None:
    """One twin machine and one agent per process, reused for every episode."""
    tune_threads()
    rl = RLConfig()
    rl.screen_shape = config.screen_shape
    optics = OpticsEnv(rl)
    machine = TwinMachine(
        optics_source=optics,
        screen_shape=config.screen_shape,
        num_particles=config.twin_particles,
        fit_misalignments=config.fit_misalignments,
    )
    agent = MBRLAgent(config, CheetahTwin.for_env(optics, config.twin_particles))
    agent.record = True
    _WORKER.update(machine=machine, agent=agent, targets=targets)


def _episode(seed: int):
    """One twin episode; returns the pairs the planner generated and how it did."""
    machine, agent = _WORKER["machine"], _WORKER["agent"]
    # Reuse one of the shared requests so the optics cache is worth having.
    targets = _WORKER["targets"]
    machine.target = targets[seed % len(targets)]
    agent.recorded = []
    started = time.time()
    result = agent.run(machine, seed=seed)
    # `reset` draws its own request; put the shared one back for the next episode.
    machine.target = targets[seed % len(targets)]
    pairs = [(o.astype(np.float32), a.astype(np.float32)) for o, a in agent.recorded]
    return pairs, result.benchmark_score, result.shots, time.time() - started


def collect(config: MBRLConfig, episodes: int, workers: int, targets: np.ndarray):
    """Roll `episodes` twin episodes and gather every planner decision."""
    observations, actions, scores = [], [], []
    started = time.time()
    if workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(
            workers, initializer=_init, initargs=(config, targets)
        ) as pool:
            for index, (pairs, score, shots, seconds) in enumerate(
                pool.imap_unordered(_episode, range(episodes))
            ):
                observations += [o for o, _ in pairs]
                actions += [a for _, a in pairs]
                scores.append(score)
                print(
                    f"  episode {index + 1:>4}/{episodes}  score {score:7.3f}  "
                    f"shots {shots:>3}  pairs {len(pairs):>3}  "
                    f"[{len(observations)} total, {time.time() - started:.0f} s]",
                    flush=True,
                )
    else:
        _init(config, targets)
        for seed in range(episodes):
            pairs, score, shots, seconds = _episode(seed)
            observations += [o for o, _ in pairs]
            actions += [a for _, a in pairs]
            scores.append(score)
            print(
                f"  episode {seed + 1:>4}/{episodes}  score {score:7.3f}  "
                f"shots {shots:>3}  pairs {len(pairs):>3}",
                flush=True,
            )
    return np.array(observations), np.array(actions), np.array(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--targets", type=int, default=24, help="distinct requests")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("MBRL_opt/models/focus.pt"))
    parser.add_argument(
        "--data", type=Path, default=None, help="also save the collected pairs here"
    )
    parser.add_argument("--fit-misalignments", action="store_true")
    args = parser.parse_args()

    config = MBRLConfig()
    config.fit_misalignments = args.fit_misalignments

    from dyna import draw_target

    rng = np.random.default_rng(args.seed)
    targets = np.array([draw_target(rng) for _ in range(args.targets)])

    print(
        f"collecting {args.episodes} twin episodes over {args.targets} requests "
        f"on {args.workers} workers (no beam time)"
    )
    started = time.time()
    observations, actions, scores = collect(
        config, args.episodes, args.workers, targets
    )
    print(
        f"\n{len(observations)} decisions from {args.episodes} episodes in "
        f"{time.time() - started:.0f} s; twin score {scores.mean():.3f} "
        f"(median {np.median(scores):.3f})\n"
    )
    if not len(observations):
        raise SystemExit("no decisions recorded — nothing to clone")

    if args.data:
        args.data.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.data, observations=observations, actions=actions)
        print(f"pairs written to {args.data}")

    print("cloning the planner:")
    policy_config = PolicyConfig(epochs=args.epochs, seed=args.seed)
    policy_config.notes = {
        "episodes": args.episodes,
        "targets": args.targets,
        "decisions": int(len(observations)),
        "twin_score_mean": float(scores.mean()),
        "fit_misalignments": bool(args.fit_misalignments),
    }
    policy, history = clone(observations, actions, policy_config)
    policy.save(args.out)
    print(
        f"\nsaved to {args.out}  "
        f"(train {history['train'][-1]:.4f}, valid {history['valid'][-1]:.4f})"
    )
    (args.out.with_suffix(".json")).write_text(
        json.dumps({"history": history, "notes": policy_config.notes}, indent=1)
    )


if __name__ == "__main__":
    main()
