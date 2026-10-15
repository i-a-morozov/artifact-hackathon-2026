"""Run the baselines, stage A and MF-BO on seeded episodes, one process per episode.

    python -m MFopt.run_baselines --seeds 30 --workers 16 --out MFopt/results
    python -m MFopt.run_baselines --misalignment 5 --seeds 12 --tag lost_beam

`--misalignment` scales `RLConfig.misalignment_variation`: 5 gives days on which
the beam is lost part way along the line before any magnet is touched.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .harness import run_episode, save, summarise
from .optics_env import OpticsEnv
from .steering import optics_and_steering, optics_only


def _mfbo(session) -> None:
    from .mfbo import mfbo

    mfbo(session)


POLICIES = {
    "do_nothing": lambda session: None,
    "optics_only": optics_only,
    "optics_and_steering": optics_and_steering,
    "mfbo": _mfbo,
}

_env: OpticsEnv | None = None


def _episode(job: tuple[str, int, float]):
    """One episode in a worker process, which keeps its own environment."""
    global _env
    name, seed, misalignment = job
    if _env is None:
        import torch

        torch.set_num_threads(1)
        from rl_framework.config import RLConfig

        _env = OpticsEnv(RLConfig(misalignment_variation=misalignment))
    return name, run_episode(_env, POLICIES[name], seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--policies", nargs="+", default=list(POLICIES))
    parser.add_argument("--misalignment", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", default="MFopt/results")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = range(args.first_seed, args.first_seed + args.seeds)
    jobs = [(name, seed, args.misalignment) for name in args.policies for seed in seeds]
    results: dict[str, list] = {name: [] for name in args.policies}
    with ProcessPoolExecutor(args.workers) as pool:
        for name, result in pool.map(_episode, jobs):
            results[name].append(result)
            print(
                f"{name} seed {result.seed}: score {result.score:.2f} "
                f"rel.err {result.relative_error:.2f} T {result.transmission:.2f} "
                f"steps {result.steps} ({result.seconds:.0f} s)",
                flush=True,
            )
    suffix = f"_{args.tag}" if args.tag else ""
    for name, episodes in results.items():
        save(episodes, out / f"{name}{suffix}.json")
        print(name, summarise(episodes), flush=True)


if __name__ == "__main__":
    main()
