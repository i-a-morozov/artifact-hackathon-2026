"""Steering responses and MF-BO knobs as a function of the requested spot.

For each of the 144 VBC-on optics nodes (12 x 12 requested rms sizes, 0.1 to 20 mm),
in the error-free twin at that node's optics (re-matched in this environment's
model, see `OpticsEnv.optics_for_target`):

- `orbit`: the model's centroid at every BPM and at the readout screen, which is
  what steering aims for (with dispersion the error-free beam is not on axis).
- `R`: the response of those centroids to every corrector plane, central
  differences. Rows: x at every BPM, then y at every BPM, then the screen's x
  and y. Columns: `corrector_names`.
- `J`: the response of the readout (mu_x, mu_y, sigma_x, sigma_y, transmission)
  to the MF-BO knobs, `DOF`.

`ResponseTable.at(sigma_x, sigma_y)` interpolates all of it bilinearly in log
size, and `ResponseTable.nearest` gives the closest node.

The MF-BO knobs (Francesco, 15 Oct): the 5 quadrupoles and the 5 correctors per
plane closest to the IP, plus the two FEA sextupoles.

    python -m MFopt.responses --workers 4 --out MFopt/data/responses_vbc_on.npz
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .batched import probe_batch
from .optics_env import CORRECTOR_STEP, QUAD_STEP, OpticsEnv

QUADRUPOLES = ("cla_fea_mag_quad_14", "cla_fea_mag_quad_15", "cla_fea_mag_quad_16",
               "cla_feh_mag_quad_01", "cla_feh_mag_quad_02")
CORRECTORS = tuple(f"{name}_{plane}" for plane in ("h", "v") for name in (
    "cla_fea_mag_hvcor_04", "cla_fea_mag_hvcor_05", "cla_fea_mag_hvcor_06",
    "cla_feh_mag_hvcor_01", "cla_feh_mag_hvcor_02"))
SEXTUPOLES = ("cla_fea_mag_sext_01", "cla_fea_mag_sext_02")
DOF = QUADRUPOLES + CORRECTORS + SEXTUPOLES
SEXTUPOLE_STEP = 5.0  # k2, m^-3 (limits are +-50)
READOUT = ("mu_x", "mu_y", "sigma_x", "sigma_y", "transmission")
DEFAULT = Path(__file__).resolve().parent / "data" / "responses_vbc_on.npz"

_env: OpticsEnv | None = None


def _get_env() -> OpticsEnv:
    global _env
    if _env is None:
        import torch
        torch.set_num_threads(1)  # one process per node
        _env = OpticsEnv()
    return _env


def _step(name: str) -> float:
    if name in QUADRUPOLES:
        return QUAD_STEP
    if name in SEXTUPOLES:
        return SEXTUPOLE_STEP
    return CORRECTOR_STEP


def _centroids(env: OpticsEnv, readings: dict) -> np.ndarray:
    """BPM x, BPM y, then the screen's x and y, against each diagnostic's centre."""
    twin = env.twin
    stats = {n: row[:2] - twin._offsets[n][:2]
             for n, row in zip(readings["names"], readings["stats"])}
    bpm = np.array([stats[b] for b in env.bpm_names])
    return np.concatenate([bpm[:, 0], bpm[:, 1], stats[env.config.target_screen]])


def _readout(env: OpticsEnv, readings: dict) -> np.ndarray:
    row = readings["stats"][readings["names"].index(env.config.target_screen)]
    return np.append(row, readings["transmission"])


def node(ij: tuple[int, int]) -> dict:
    """Everything the table holds for one node."""
    env = _get_env()
    i, j = ij
    sx, sy = env.nodes.axis[i], env.nodes.axis[j]
    start = time.time()
    optics = env.optics_for_target(sx, sy, modes=False)
    k1 = optics.k1
    settings = {**{n: env.twin._knobs[n].value for n in env.corrector_names}, **k1}
    base = env._probe(settings)
    # every finite difference in one vectorised track: +h then -h per knob
    knobs = list(env.corrector_names) + list(DOF)
    steps = [CORRECTOR_STEP] * len(env.corrector_names) + [_step(n) for n in DOF]
    centre = {n: settings.get(n, env.twin._knobs[n].value) for n in knobs}
    columns = {n: np.full(2 * len(knobs), centre[n]) for n in set(knobs)}
    for row, (name, h) in enumerate(zip(knobs, steps)):
        columns[name][2 * row] += h
        columns[name][2 * row + 1] -= h
    stats, transmission, names = probe_batch(env, columns, settings)
    shots = [{"names": names, "stats": s, "transmission": t}
             for s, t in zip(stats, transmission)]
    nc = len(env.corrector_names)
    R = np.array([(_centroids(env, shots[2 * c]) - _centroids(env, shots[2 * c + 1]))
                  / (2 * CORRECTOR_STEP) for c in range(nc)]).T
    J = np.array([(_readout(env, shots[2 * (nc + c)]) - _readout(env, shots[2 * (nc + c) + 1]))
                  / (2 * h) for c, h in enumerate(steps[nc:])]).T
    return {
        "ij": (i, j),
        "k1": np.array([k1[q] for q in env.nodes.quadrupoles]),
        "orbit": _centroids(env, base),
        "readout": _readout(env, base),
        "R": np.nan_to_num(R),
        "J": np.nan_to_num(J),
        "clipped": [q in optics.clipped for q in env.nodes.quadrupoles],
        "seconds": time.time() - start,
    }


def build(out: Path, workers: int = 4) -> None:
    env = _get_env()
    n = len(env.nodes.axis)
    shape = (n, n)
    table: dict[str, np.ndarray] = {}
    with ProcessPoolExecutor(workers) as pool:
        for count, r in enumerate(pool.map(node, [(i, j) for i in range(n) for j in range(n)])):
            for key in ("k1", "orbit", "readout", "R", "J", "clipped"):
                value = np.asarray(r[key])
                if key not in table:
                    table[key] = np.zeros(shape + value.shape, dtype=value.dtype)
                table[key][r["ij"]] = value
            i, j = r["ij"]
            ro = r["readout"]
            print(f"{count + 1}/{n * n} node {i},{j}: {ro[2] * 1e3:.3f} x {ro[3] * 1e3:.3f} mm "
                  f"for {env.nodes.axis[i] * 1e3:.3f} x {env.nodes.axis[j] * 1e3:.3f}, "
                  f"T {ro[4]:.3f} ({r['seconds']:.0f} s)", flush=True)
    target = np.stack(np.meshgrid(env.nodes.axis, env.nodes.axis, indexing="ij"), axis=-1)
    error = np.abs(table["readout"][..., 2:4] / target - 1).max(axis=-1)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, spot_sizes_m=env.nodes.axis, quadrupoles=np.array(env.nodes.quadrupoles),
        correctors=np.array(env.corrector_names), bpms=np.array(env.bpm_names),
        dof=np.array(DOF), readout_names=np.array(READOUT),
        feasible=(table["readout"][..., 4] >= 0.95) & (error <= 0.05),
        relative_error=error, **table)


class ResponseTable:
    """The built table, looked up by requested size (metres)."""

    def __init__(self, path: Path = DEFAULT):
        with np.load(path) as data:
            self.data = {key: data[key] for key in data.files}
        self.axis = self.data["spot_sizes_m"]
        self.correctors = self.data["correctors"].tolist()
        self.bpms = self.data["bpms"].tolist()
        self.dof = self.data["dof"].tolist()
        log = np.log(self.axis)
        self._interp = {key: RegularGridInterpolator((log, log), self.data[key])
                        for key in ("k1", "orbit", "readout", "R", "J")}

    def _clamp(self, sigma_x, sigma_y):
        return np.log(np.clip([sigma_x, sigma_y], self.axis[0], self.axis[-1]))

    def at(self, sigma_x: float, sigma_y: float) -> dict[str, np.ndarray]:
        """Bilinear in log size between the four surrounding nodes."""
        point = self._clamp(sigma_x, sigma_y)
        return {key: f(point)[0] for key, f in self._interp.items()}

    def nearest(self, sigma_x: float, sigma_y: float) -> dict:
        point = self._clamp(sigma_x, sigma_y)
        log = np.log(self.axis)
        i, j = (int(np.abs(log - p).argmin()) for p in point)
        return {key: value[i, j] for key, value in self.data.items()
                if isinstance(value, np.ndarray) and value.shape[:2] == (len(log), len(log))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=str(DEFAULT))
    args = parser.parse_args()
    build(Path(args.out), args.workers)


if __name__ == "__main__":
    main()
