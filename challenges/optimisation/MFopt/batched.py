"""Many settings of the error-free twin in one vectorised Cheetah track.

`OpticsEnv._probe` tracks one setting at a time. Cheetah broadcasts element
parameters over leading batch dimensions, so setting each knob to a tensor of N
values and tracking the nominal beam once gives all N readings in one pass: the
transfer maps become (N, 7, 7) and the particles (N, n, 7).

    stats, transmission, names = probe_batch(env, {"cla_feh_mag_quad_01": k1s, ...})

`stats` is (N, n_diagnostics, 4): centroid x, y and rms x, y at every BPM and
screen in tracking order, NaN where too little beam arrives, exactly as `_probe`
returns them; `transmission` is (N,). Knobs named in neither `columns` nor `base`
are at the lattice file's strengths.

The batched track runs on its own deep copy of the twin's Cheetah segment, and
every knob is written on every call, so nothing of a batch can stay behind in the
twin that `_probe` and the re-match use.
"""

from __future__ import annotations

import copy

import cheetah
import numpy as np
import torch

# optics_env puts RL_framework on the path, so the names come through it
from .optics_env import BPM_TAG, DTYPE, SCREEN_TAG, VBC_EDGE, OpticsEnv

# Settings per vectorised track: memory goes as chunk x particles x 7.
CHUNK = 512


class _BatchSegment:
    """A private copy of the twin's segment, cut at the same stops."""

    def __init__(self, twin):
        segment = copy.deepcopy(twin.transport._segment)
        self.elements = {element.name: element for element in segment.elements}
        self.chunks, previous = [], 0
        for stop in [s.name for s in twin._stops]:
            index = segment.element_index(stop)
            self.chunks.append((stop, cheetah.Segment(segment.elements[previous:index])))
            previous = index

    def write(self, element: str, prop: str, values: torch.Tensor) -> None:
        """`CheetahTransport.write` with a tensor of values."""
        target = self.elements[element]
        setattr(target, prop, values)
        if prop == "angle" and element in VBC_EDGE:
            face = "dipole_e2" if VBC_EDGE[element] == "exit" else "dipole_e1"
            setattr(target, face, values)


def _segment(env: OpticsEnv) -> _BatchSegment:
    if getattr(env, "_batch_segment", None) is None:
        if env.twin.config.backend != "cheetah":
            raise ValueError("probe_batch needs the Cheetah backend")
        env._batch_segment = _BatchSegment(env.twin)
    return env._batch_segment


def _track(env: OpticsEnv, columns: dict[str, np.ndarray], count: int):
    twin = env.twin
    segment = _segment(env)
    for name, knob in twin._knobs.items():
        values = columns.get(name)
        if values is None:
            tensor = torch.tensor(float(twin._nominal_strengths[name]), dtype=DTYPE)
        else:
            tensor = torch.as_tensor(np.asarray(values, dtype=float), dtype=DTYPE)
        segment.write(knob.element_name, knob.prop, tensor)
        for element, sign in knob.coupled:
            segment.write(element, knob.prop, sign * tensor)
    coordinates = twin._coordinates
    particles = torch.tensor(
        np.column_stack([coordinates, np.ones(len(coordinates))]), dtype=DTYPE
    )
    beam = cheetah.ParticleBeam(particles=particles, energy=twin.transport._energy)
    radius = twin.config.beam_pipe_radius
    minimum = twin.config.min_screen_fraction * len(coordinates)
    live = torch.ones((count, len(coordinates)), dtype=torch.bool)
    names, stats = [], []
    with torch.no_grad():
        for stop, chunk in segment.chunks:
            if len(chunk.elements):
                beam = chunk.track(beam)
            x = beam.x.expand(count, -1)
            y = beam.y.expand(count, -1)
            live &= (beam.survival_probabilities > 0.5).expand(count, -1)
            if radius is not None:
                live &= x**2 + y**2 <= radius**2
            if BPM_TAG not in stop and SCREEN_TAG not in stop:
                continue
            names.append(stop)
            n = live.sum(dim=1).to(DTYPE)
            w = live.to(DTYPE)
            mx = (x * w).sum(1) / n
            my = (y * w).sum(1) / n
            # population std over the live particles, as numpy's std
            sx = torch.sqrt(((x - mx[:, None]) ** 2 * w).sum(1) / n)
            sy = torch.sqrt(((y - my[:, None]) ** 2 * w).sum(1) / n)
            row = torch.stack([mx, my, sx, sy], dim=1).double().numpy()
            row[(n < minimum).numpy()] = np.nan
            stats.append(row)
    transmission = live.sum(dim=1).double().numpy() / len(coordinates)
    return np.stack(stats, axis=1), transmission, names


def probe_batch(
    env: OpticsEnv, columns: dict[str, np.ndarray], base: dict[str, float] | None = None
):
    """Track every row of `columns` (name -> N values) on top of `base`."""
    count = len(next(iter(columns.values())))
    base = {k: v for k, v in (base or {}).items() if k not in columns}
    stats, transmission, names = [], [], None
    for start in range(0, count, CHUNK):
        part = {k: np.asarray(v)[start : start + CHUNK] for k, v in columns.items()}
        n = len(next(iter(part.values())))
        full = {**{k: np.full(n, v) for k, v in base.items()}, **part}
        s, t, names = _track(env, full, n)
        stats.append(s)
        transmission.append(t)
    return np.concatenate(stats), np.concatenate(transmission), names
