"""Stage A: load the optics, steer the orbit with BPMs, put the spot on the request.

All responses come from the error-free twin at the optics being used, as finite
differences, so they carry the VBC, the cavities and the chromatic averaging over
the beam's own energy spread. Readings are compared with what the twin reads at
the same settings (the model's error-free orbit), not with zero: with dispersion
the error-free beam is not on axis.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .harness import Session
from .optics_env import OpticsEnv

KICK = 2e-5  # rad, finite-difference step for the responses
# The last correctors, after the last BPM: they move the spot at the readout and
# nothing that a BPM sees.
FINAL_CORRECTORS = ("cla_feh_mag_hvcor_01_h", "cla_feh_mag_hvcor_01_v",
                    "cla_feh_mag_hvcor_02_h", "cla_feh_mag_hvcor_02_v")

_cache: dict[str, dict] = {}


def _key(settings: dict[str, float]) -> str:
    text = json.dumps({k: round(v, 6) for k, v in sorted(settings.items())})
    return hashlib.sha1(text.encode()).hexdigest()


def _measured(env: OpticsEnv, readings: dict) -> dict[str, np.ndarray]:
    """The twin's readings as the machine reports them: centroids against each
    diagnostic's own centre."""
    twin = env.twin
    out = {}
    for name, row in zip(readings["names"], readings["stats"]):
        out[name] = row[:2] - twin._offsets[name][:2]
    return out


def responses(env: OpticsEnv, settings: dict[str, float]) -> dict:
    """At these settings, in the twin: the BPM orbit and the spot centroid, and
    their responses to every corrector plane (central differences)."""
    key = _key(settings)
    if key in _cache:
        return _cache[key]
    base = _measured(env, env._probe(settings))
    bpms, screen = env.bpm_names, env.config.target_screen
    correctors = env.corrector_names
    upstream = [c for c in correctors if c not in FINAL_CORRECTORS]
    R_bpm = np.zeros((2 * len(bpms), len(upstream)))
    J_spot = np.zeros((2, len(FINAL_CORRECTORS)))
    for j, name in enumerate(correctors):
        centre = settings.get(name, env.twin._knobs[name].value)
        plus = _measured(env, env._probe({**settings, name: centre + KICK}))
        minus = _measured(env, env._probe({**settings, name: centre - KICK}))
        if name in FINAL_CORRECTORS:
            J_spot[:, FINAL_CORRECTORS.index(name)] = (plus[screen] - minus[screen]) / (2 * KICK)
        else:
            column = np.concatenate([(plus[b] - minus[b]) / (2 * KICK) for b in bpms])
            R_bpm[:, upstream.index(name)] = column.reshape(len(bpms), 2).T.ravel()
    orbit = np.array([base[b] for b in bpms])  # (n_bpm, 2)
    result = {
        "bpms": bpms, "upstream": upstream,
        "orbit_x": orbit[:, 0], "orbit_y": orbit[:, 1],
        "R_bpm": np.nan_to_num(R_bpm),  # rows: all x, then all y
        "spot": base[screen], "J_spot": np.nan_to_num(J_spot),
    }
    _cache[key] = result
    return result


def table_responses(env: OpticsEnv, sigma_x: float, sigma_y: float) -> dict:
    """The same as `responses`, read from the precomputed node table
    (`MFopt.responses`), interpolated to the requested size: no tracking."""
    from .responses import ResponseTable
    global _table
    if _table is None:
        _table = ResponseTable()
    at = _table.at(sigma_x, sigma_y)
    nb = len(_table.bpms)
    upstream = [c for c in env.corrector_names if c not in FINAL_CORRECTORS]
    up = [_table.correctors.index(c) for c in upstream]
    final = [_table.correctors.index(c) for c in FINAL_CORRECTORS]
    return {
        "bpms": _table.bpms, "upstream": upstream,
        "orbit_x": at["orbit"][:nb], "orbit_y": at["orbit"][nb:2 * nb],
        "R_bpm": at["R"][:2 * nb][:, up],
        "spot": at["orbit"][2 * nb:], "J_spot": at["R"][2 * nb:][:, final],
    }


_table = None


def _clip(env: OpticsEnv, name: str, value: float) -> float:
    knob = env._knobs[name]
    return float(np.clip(value, knob.low, knob.high))


def _upstream_of(model: dict, bpms: np.ndarray) -> np.ndarray:
    """Which correctors move at least one of these BPMs in the model."""
    nb = len(model["bpms"])
    rows = np.concatenate([bpms, bpms + nb])
    return np.abs(model["R_bpm"][rows]).max(axis=0) > 0


def steer_orbit(session: Session, model: dict, rcond: float = 1e-3) -> dict:
    """One shot of SVD steering to the model's orbit, on the BPMs that read.

    When the beam is lost part way, R is cropped to the BPMs with a reading
    (rows) and to the correctors upstream of the last of them (columns)."""
    obs = session.last.obs
    live = np.flatnonzero(obs["bpm_valid"])
    if live.size == 0:
        return {}
    nb = len(model["bpms"])
    rows = np.concatenate([live, live + nb])
    columns = _upstream_of(model, live)
    error = np.concatenate([obs["bpm_x"] - model["orbit_x"],
                            obs["bpm_y"] - model["orbit_y"]])[rows]
    R = model["R_bpm"][np.ix_(rows, columns)]
    kicks = -np.linalg.pinv(R, rcond=rcond) @ error
    now = session.settings()
    names = [n for n, c in zip(model["upstream"], columns) if c]
    action = {name: _clip(session.env, name, now[name] + dk)
              for name, dk in zip(names, kicks)}
    session.step({"magnets": action})
    return action


def _reach(reading) -> tuple[int, float]:
    """How far the beam gets: BPMs with a reading, then transmission."""
    return int(reading.obs["bpm_valid"].sum()), float(reading.info["transmission"])


def thread(session: Session, model: dict, max_shots: int = 8,
           rcond: float = 1e-3) -> None:
    """Get the beam to the end of the line: steer on the BPMs that read and
    repeat while BPMs are missing. A shot that loses ground is undone, and the
    next one keeps fewer singular values (gentler, better-conditioned kicks)."""
    best = session.last
    for _ in range(max_shots):
        if best.obs["bpm_valid"].all() and best.info["on_screen"]:
            return
        if session.last is not best:
            session.step({"magnets": dict(best.info["magnets"])})
        steer_orbit(session, model, rcond)
        if _reach(session.last) > _reach(best):
            best = session.last
        else:
            rcond = min(rcond * 10, 0.1)
    if session.last is not best:
        session.step({"magnets": dict(best.info["magnets"])})


def steer_spot(session: Session, model: dict) -> dict:
    """One Newton shot of the final correctors to the requested centroid."""
    info = session.last.info
    if not info["on_screen"]:
        return {}
    error = np.array([info["mu_x"], info["mu_y"]]) - session.target[:2]
    kicks = -np.linalg.pinv(model["J_spot"]) @ error
    now = session.settings()
    action = {name: _clip(session.env, name, now[name] + dk)
              for name, dk in zip(FINAL_CORRECTORS, kicks)}
    session.step({"magnets": action})
    return action


def optics_only(session: Session) -> None:
    """Baseline: load the target optics in one step and stop."""
    t = session.target
    optics = session.env.optics_for_target(t[2], t[3], modes=False)
    session.step({"magnets": dict(optics.k1)})


def optics_and_steering(session: Session, orbit_shots: int = 2,
                        spot_shots: int = 3, table: bool = True) -> None:
    """Stage A: the target optics, SVD orbit steering, then the spot on target."""
    env = session.env
    t = session.target
    optics = env.optics_for_target(t[2], t[3], modes=False)
    session.step({"magnets": dict(optics.k1)})
    if table:
        model = table_responses(env, t[2], t[3])
    else:
        model = responses(env, {**_corrector_zero(env), **optics.k1})
    thread(session, model)
    for _ in range(orbit_shots):
        steer_orbit(session, model)
    for _ in range(spot_shots):
        info = session.last.info
        if info["on_screen"]:
            miss = np.abs(np.array([info["mu_x"], info["mu_y"]]) - t[:2])
            if miss.max() < 0.05 * t[2:].mean():
                break
        steer_spot(session, model)


def _corrector_zero(env: OpticsEnv) -> dict[str, float]:
    """The correctors as the lattice file has them, which is where the twin's
    responses are taken."""
    return {name: env.twin._knobs[name].value for name in env.corrector_names}
