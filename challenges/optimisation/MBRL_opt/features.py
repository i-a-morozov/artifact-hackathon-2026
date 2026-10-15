"""The episode-independent view a policy sees, and the actions it returns.

A policy cannot be trained on named magnets. The control set is chosen per episode by
the SVD — which quadrupoles matter depends on the optics the request asked for — so
`cla_fea_mag_quad_08` is the most significant quadrupole for one request and absent
from the set for the next. A network with one output per magnet would be learning a
different problem every episode.

What *is* constant is the ranking. Slot 0 is always "the most significant quadrupole
for this request", slot 6 always "the leading corrector plane", and because the knobs
have already been calibrated so that one step moves the readout by the same amount
whichever knob takes it (`KnobSet.rescale`), a step of +1 in slot 0 means the same
thing across episodes even though the magnet behind it changes. That is the frame
everything here is written in.

The observation is in the same spirit: nothing in metres or in m^-2, everything as a
ratio to what was asked for. Positions are errors in requested spot sizes, sizes are
log ratios of achieved to requested, and the knobs are their distance from the
reference optics in calibrated steps. A policy trained on one request is then looking
at the same numbers as a policy trained on any other.
"""

from __future__ import annotations

import numpy as np

# How many ranked slots the policy acts on. The control set is padded to this with
# zeros when an episode offers fewer knobs, and truncated when it offers more, so the
# network shape is fixed whatever the SVD returns.
MAX_SLOTS = 24
# Everything in the observation that is not per-slot.
GLOBAL_FEATURES = 10
OBSERVATION_SIZE = GLOBAL_FEATURES + MAX_SLOTS
# Largest step, in calibrated units, a policy may ask for in one shot.
MAX_STEP = 3.0


def observe(
    moments: np.ndarray | None,
    target: np.ndarray,
    transmission: float,
    orbit_rms: float,
    valid_fraction: float,
    knob_offsets: np.ndarray,
    shots_left: float,
    trust: float,
) -> np.ndarray:
    """What the policy sees, as a fixed-length vector in request-relative units.

    `moments` is the camera reading or None when the beam missed the screen;
    `knob_offsets` is how far each slot has moved from the reference optics, in
    calibrated steps.
    """
    spot = float(np.asarray(target)[2:].mean())
    if moments is None:
        # No reading: say so in the one feature that can carry it, and leave the
        # error terms at zero rather than inventing a number for them.
        error = np.zeros(4)
        on_screen = 0.0
    else:
        moments = np.asarray(moments, dtype=float)
        error = (moments - np.asarray(target, dtype=float)) / spot
        on_screen = 1.0

    # Sizes also enter as log ratios: a spot twice as wide as asked for is the same
    # problem whether the request was 80 or 400 microns, and the difference in spot
    # units is not.
    if moments is not None and moments[2] > 0 and moments[3] > 0:
        log_ratio = np.log(moments[2:] / np.asarray(target, dtype=float)[2:])
    else:
        log_ratio = np.zeros(2)

    globals_ = np.array(
        [
            np.clip(error[0], -20, 20),
            np.clip(error[1], -20, 20),
            np.clip(error[2], -20, 20),
            np.clip(error[3], -20, 20),
            np.clip(log_ratio[0], -5, 5),
            np.clip(log_ratio[1], -5, 5),
            on_screen,
            float(transmission),
            float(np.clip(orbit_rms / 1e-3, 0, 20)),
            float(valid_fraction),
        ],
        dtype=np.float32,
    )
    slots = np.zeros(MAX_SLOTS, dtype=np.float32)
    offsets = np.asarray(knob_offsets, dtype=np.float32).reshape(-1)[:MAX_SLOTS]
    slots[: len(offsets)] = np.clip(offsets, -20, 20)
    # `shots_left` and `trust` ride along in the last two slot entries only if there
    # is room; keeping them here rather than in `globals_` would change the width.
    return np.concatenate([globals_, slots]).astype(np.float32)


def pad_action(step: np.ndarray, count: int) -> np.ndarray:
    """A policy's `MAX_SLOTS` outputs cut down to the knobs this episode has."""
    step = np.asarray(step, dtype=float).reshape(-1)
    return np.clip(step[:count], -MAX_STEP, MAX_STEP)


def expert_action(values: np.ndarray, proposal: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """The planner's move, as the policy would have had to express it.

    `MAX_SLOTS` long, zero-padded, in calibrated steps and clipped to what a policy
    is allowed to ask for — so a cloned policy is being trained on targets it can
    actually produce.
    """
    step = (np.asarray(proposal) - np.asarray(values)) / np.maximum(scales, 1e-30)
    out = np.zeros(MAX_SLOTS, dtype=np.float32)
    step = np.clip(step[:MAX_SLOTS], -MAX_STEP, MAX_STEP)
    out[: len(step)] = step
    return out
