"""The agent's own model of CLARA: the same Cheetah lattice, tracked in torch.

This is the "model" of model-based RL, and it is physics rather than a learned network.
`RL_framework/FEBE_full.json` is the lattice SimEnv tracks — the two files are identical
in content — so with the same partition the twin carries the same 373 elements with the
same strengths. Nothing about the machine's *structure* has to be learned. What is
unknown is its *state*: which way the beam was launched, how big it is, and where the
magnets and diagnostics actually sit. That is `LatentState`, and `system_id.py` fits it
to the shots the agent has taken.

Two things make this worth doing. Cheetah is torch, so the whole line is differentiable:
the gradient of the spot size at the readout with respect to a quadrupole's k1 comes
back from `backward()`, and so does the gradient with respect to a misalignment nobody
measured. And Cheetah fuses the linear maps between elements, so a full track costs a
couple of milliseconds whether it carries a thousand particles or ten thousand, and
element parameters may carry a leading batch dimension — a hundred candidate tunes
track in barely more time than one. Planning is therefore cheap enough to redo from
scratch after every shot.

The readings have to mean the same thing they do in SimEnv, or a tune that looks good on
the twin will not be good on the machine. `readings()` reproduces `SimEnv._track` and
`SimEnv._expose`: the same stops, the same cumulative loss of particles to the beam pipe,
the same minimum fraction before a diagnostic counts, and the same clipping to the
camera chip. `smoke_test.py` holds the two to within a micron on a machine with no
errors, which is the check that this file is honest.

    twin = CheetahTwin()
    twin.set_knobs({"cla_feh_mag_quad_01": 2.0})
    reading = twin.readings()           # at the nominal beam, perfectly aligned
    reading = twin.readings(state)      # at the beam and survey error that were fitted
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cheetah
import numpy as np
import torch

CHALLENGE_DIR = Path(__file__).resolve().parents[1]
# The lattice SimEnv tracks. `RL_framework/FEBE_full.json` is a second copy of the
# same machine and was identical to this one until the VBC was turned on, at which
# point only this one was updated and a twin reading the other disagreed with the
# environment by a millimetre and a half at the readout. The twin reads whatever the
# environment reads; there is no version of this that is worth a stale copy.
LATTICE_JSON = CHALLENGE_DIR / "data" / "clara" / "lattice_data" / "CLARA_cheetah.json"
DISTRIBUTION_NPZ = CHALLENGE_DIR / "nominal_model" / "initial_distribution.npz"

LATTICE_START = "cla_s02_sim_aper_01"
READOUT_SCREEN = "cla_feh_dia_scr_01"

BPM_TAG = "dia_bpm"
SCREEN_TAG = "dia_scr"
APERTURE_TAG = "sim_aper"

DTYPE = torch.float32  # what the Cheetah lattice file is parsed into

# What one knob of each element type is called, mirroring `knobs.PROPERTIES`. Kept
# here too so the twin can snapshot the lattice file's strengths without importing the
# agent's own configuration.
KNOB_PROPERTIES = {
    "Quadrupole": (("k1", ""),),
    "Sextupole": (("k2", ""),),
    "CombinedCorrector": (("horizontal_angle", "_h"), ("vertical_angle", "_v")),
    "Cavity": (("phase", ""),),
}

# The VBC chicane: four dipoles on one setting, alternating in sign, as SimEnv drives
# it. `sim_env.VBC_DIPOLES` and `VBC_SIGNS`.
VBC_CHICANE = "cla_vbc_chicane"
VBC_DIPOLES = ("cla_vbc_mag_dip_01", "cla_vbc_mag_dip_02",
               "cla_vbc_mag_dip_03", "cla_vbc_mag_dip_04")
VBC_SIGNS = (1.0, -1.0, -1.0, 1.0)
# Which face of each VBC dipole turns with its angle, as `sim_env.VBC_EDGE` and
# `score.VBC_EDGE`: the chicane's outer faces stay square to the beamline and the
# inner ones follow the bend. Setting the angle without the face is a different
# magnet — the edge focusing is what the vertical plane sees.
VBC_EDGE = {
    "cla_vbc_mag_dip_01": "dipole_e2",
    "cla_vbc_mag_dip_02": "dipole_e1",
    "cla_vbc_mag_dip_03": "dipole_e2",
    "cla_vbc_mag_dip_04": "dipole_e1",
}

# The machine SimEnv models, repeated here because the twin has to agree with it about
# what a reading is. `sim_env` and `RLConfig` carry the same numbers.
BEAM_PIPE_RADIUS = 50e-3  # m
MIN_SCREEN_FRACTION = 0.10  # of the launched beam, for a diagnostic to read

# How far one episode's machine is from the next, at a variation of 1. These are the
# priors `system_id.py` fits under, and they are `sim_env`'s own spreads.
CENTROID_JITTER = np.array([50e-6, 50e-6, 50e-6, 50e-6, 100e-6, 1e-3])
SIGMA_SPREAD = 1.3
BETA_SPREAD = 1.2
MISALIGNMENT_RMS = 50e-6  # quadrupoles and sextupoles
DIAGNOSTIC_OFFSET_RMS = 30e-6  # the centring of a camera or a BPM


@dataclass
class LatentState:
    """What the twin does not know about this episode's machine, as tensors.

    The beam is parameterised exactly as `SimEnv._perturb_beam` draws it: each
    transverse plane gets an rms size factor and a mismatch factor — size against
    divergence at constant emittance — the longitudinal planes a size factor each, and
    the whole beam a six-vector offset. Sizes are held as logarithms because that is
    how they are drawn, log-uniform, so a beam half the size is as likely as one twice
    the size, and because it keeps them positive without a constraint.

    Everything is stored **in units of its own prior spread**, so a stored value of 1
    is a one-sigma machine whatever the quantity is and a value of 0 is the nominal
    model. That whitening is not cosmetic. A launch offset is tens of microns and a
    log size factor is a few tenths, and Adam steps every parameter by roughly the
    learning rate in whatever units it is held in — so on raw values one shared rate
    either moves the sizes imperceptibly or throws the beam centimetres off axis and
    straight into the pipe wall. In these units one rate suits all of them, and the
    prior in `system_id` is just the sum of squares.

    `misalignments` and `diagnostic_offsets` are (n, 2) over the twin's `misalignable`
    and `diagnostic_names`, and are None for the aligned twin, which is the default.
    """

    u_centroid: torch.Tensor  # (6,) in units of CENTROID_JITTER
    u_log_size: torch.Tensor  # (4,) in units of log(SIGMA_SPREAD), x, y, tau, delta
    u_log_mismatch: torch.Tensor  # (2,) in units of log(BETA_SPREAD), x and y
    u_misalignments: torch.Tensor | None = None  # (n_magnets, 2) in MISALIGNMENT_RMS
    u_diagnostic_offsets: torch.Tensor | None = None  # (n, 2) in DIAGNOSTIC_OFFSET_RMS

    @classmethod
    def nominal(
        cls, n_magnets: int = 0, n_diagnostics: int = 0, requires_grad: bool = False
    ) -> "LatentState":
        """The machine as the lattice file describes it: nominal beam, no survey error."""

        def zeros(*shape):
            return torch.zeros(*shape, dtype=DTYPE, requires_grad=requires_grad)

        return cls(
            u_centroid=zeros(6),
            u_log_size=zeros(4),
            u_log_mismatch=zeros(2),
            u_misalignments=zeros(n_magnets, 2) if n_magnets else None,
            u_diagnostic_offsets=zeros(n_diagnostics, 2) if n_diagnostics else None,
        )

    def parameters(self) -> list[torch.Tensor]:
        """The tensors an optimiser should fit, skipping the ones not in use."""
        found = [self.u_centroid, self.u_log_size, self.u_log_mismatch]
        found += [
            t for t in (self.u_misalignments, self.u_diagnostic_offsets) if t is not None
        ]
        return found

    def detached(self) -> "LatentState":
        """A copy off the graph, to carry between shots without keeping history."""
        detach = lambda t: None if t is None else t.detach().clone()  # noqa: E731
        return replace(
            self,
            u_centroid=detach(self.u_centroid),
            u_log_size=detach(self.u_log_size),
            u_log_mismatch=detach(self.u_log_mismatch),
            u_misalignments=detach(self.u_misalignments),
            u_diagnostic_offsets=detach(self.u_diagnostic_offsets),
        )

    def requires_grad_(self, flag: bool = True) -> "LatentState":
        for tensor in self.parameters():
            tensor.requires_grad_(flag)
        return self

    # ---- the same quantities in metres, radians and log factors ---- #

    @property
    def centroid(self) -> torch.Tensor:
        """(6,) m and rad, added to every particle."""
        return self.u_centroid * torch.tensor(CENTROID_JITTER, dtype=DTYPE)

    @property
    def log_size(self) -> torch.Tensor:
        return self.u_log_size * float(np.log(SIGMA_SPREAD))

    @property
    def log_mismatch(self) -> torch.Tensor:
        return self.u_log_mismatch * float(np.log(BETA_SPREAD))

    @property
    def misalignments(self) -> torch.Tensor | None:
        if self.u_misalignments is None:
            return None
        return self.u_misalignments * MISALIGNMENT_RMS

    @property
    def diagnostic_offsets(self) -> torch.Tensor | None:
        if self.u_diagnostic_offsets is None:
            return None
        return self.u_diagnostic_offsets * DIAGNOSTIC_OFFSET_RMS

    @property
    def scale(self) -> torch.Tensor:
        """(6,) factor on each coordinate about the beam's own centre.

        A plane's size factor multiplies the position and its mismatch factor trades
        position against divergence, exactly as `SimEnv._perturb_beam` does it.
        """
        size = torch.exp(self.log_size)
        mismatch = torch.exp(self.log_mismatch)
        return torch.stack(
            [
                size[0] * mismatch[0],
                size[0] / mismatch[0],
                size[1] * mismatch[1],
                size[1] / mismatch[1],
                size[2],
                size[3],
            ]
        )


@dataclass
class Readings:
    """What the twin predicts the diagnostics would say, with gradient to the knobs.

    The same quantities `SimEnv._get_obs` and `_get_info` return, so a cost written
    against one can be read against the other. Tensors carry the planner's batch
    dimension when one was given: `bpm_x` is (n_bpms,) or (batch, n_bpms).

    The measured moments are what a camera would report, which means they are zero
    when too little beam reached the chip — the same cliff the real diagnostic has.
    That is right for comparing against SimEnv and useless for planning, because a
    tune that has lost the beam gets no gradient telling it which way to go back. The
    `raw_` moments are the same quantities over every particle tracked, clipped to
    nothing and gated by nothing, so they are always finite and always differentiable;
    `scrape` is how far outside the beam pipe the beam is sitting on average. Together
    they give the planner something to descend when the measurement itself is blind.
    """

    bpm_x: torch.Tensor
    bpm_y: torch.Tensor
    bpm_valid: torch.Tensor  # 1 where enough beam reached the BPM to read
    mu_x: torch.Tensor
    mu_y: torch.Tensor
    sigma_x: torch.Tensor
    sigma_y: torch.Tensor
    on_screen: torch.Tensor  # 1 where enough beam landed on the chip to measure
    on_screen_fraction: torch.Tensor
    transmission: torch.Tensor
    raw_mu_x: torch.Tensor
    raw_mu_y: torch.Tensor
    raw_sigma_x: torch.Tensor
    raw_sigma_y: torch.Tensor
    scrape: torch.Tensor  # mean metres outside the beam pipe, over stops and particles
    # The camera frame, when one was asked for: rows are y, columns x, origin at the
    # bottom, exactly as `SimEnv` lays its screen images out. Only rendered for an
    # unbatched track — a population of candidate frames is of no use to anyone and
    # would cost more than the tracking.
    image: torch.Tensor | None = None

    @property
    def moments(self) -> torch.Tensor:
        """(..., 4) of (mu_x, mu_y, sigma_x, sigma_y), as a target is written."""
        return torch.stack([self.mu_x, self.mu_y, self.sigma_x, self.sigma_y], dim=-1)

    @property
    def raw_moments(self) -> torch.Tensor:
        """(..., 4) of the same, over every particle and never gated."""
        return torch.stack(
            [self.raw_mu_x, self.raw_mu_y, self.raw_sigma_x, self.raw_sigma_y], dim=-1
        )

    def planning_moments(self) -> torch.Tensor:
        """The measured moments where there is a measurement, the raw ones elsewhere.

        A planner wants the real quantity when the beam is on the chip, because that
        is what it is scored on, and something that still points downhill when it is
        not. `on_screen` is 0 or 1 and carries no gradient of its own, so this is a
        plain switch between two differentiable branches rather than a blend.
        """
        gate = self.on_screen.unsqueeze(-1)
        return gate * self.moments + (1.0 - gate) * self.raw_moments


@lru_cache(maxsize=1)
def _nominal_model():
    """`nominal_model/xsuite_model.py`, which owns how the lattice is assembled.

    A module in a folder rather than a package, so it is put on the path the same
    way `SimEnv._import_xsuite_model` puts it there.
    """
    directory = str(CHALLENGE_DIR / "nominal_model")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    import xsuite_model

    return xsuite_model


def tune_threads(threads: int = 1) -> None:
    """Pin torch to `threads`, which for this workload means one.

    Tracking a lattice is hundreds of tiny tensor operations in a row — a 7 by 7
    transfer map against a few thousand particles — and torch's thread pool costs
    more to synchronise than the arithmetic it is parallelising. Measured on the
    twin: 82 ms a track on one thread against 107 ms on four. Entry points call
    this; importing the module does not, so a caller that knows better can choose.
    """
    torch.set_num_threads(threads)


def _broadcast(tensors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """The same tensors at one common shape.

    Only the elements downstream of a batched knob carry the planner's batch
    dimension — the beam reaches the first BPM before the quadrupole being varied has
    had any say — so readings taken along the line come back at mixed shapes and have
    to be lined up before they can be stacked or compared.
    """
    shape = torch.broadcast_shapes(*(tensor.shape for tensor in tensors))
    return [tensor.expand(shape) for tensor in tensors]


def _aperture(margin: torch.Tensor, soft: float | None) -> torch.Tensor:
    """Whether a particle is inside a wall it is `margin` metres clear of.

    With `soft` at None this is the step function the machine actually applies, and
    it has no useful derivative: whether one particle of ten thousand is inside the
    pipe does not change smoothly, so a sum over all of them is a staircase and
    gradient descent only rattles between its steps.

    With `soft` set it is still the step function — the value returned is exactly 0 or
    1 — but it carries the derivative of a sigmoid of that width instead of none at
    all. Blurring the wall outright was the obvious thing and it is badly biased: a
    millimetre of blur on a beam that is half scraped already moves the predicted
    orbit further than the thing being fitted, and the fit walks away from the right
    answer to chase it. Taking the value from one branch and the gradient from the
    other costs nothing and keeps both honest.
    """
    hard = (margin >= 0).to(DTYPE)
    if soft is None:
        return hard
    smooth = torch.sigmoid(margin / soft)
    return hard + (smooth - smooth.detach())


def _masked_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Mean of `values` over the particles `weight` keeps, differentiably.

    A boolean index would break the batch dimension and drop the gradient of the
    particles it discards; a weighted sum keeps both. The clamp only guards the
    all-dead case, which is reported separately through `bpm_valid`.
    """
    total = weight.sum(dim=-1).clamp(min=1.0)
    return (values * weight).sum(dim=-1) / total


def _masked_std(values: torch.Tensor, weight: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """Population rms about `mean`, to match numpy's `std` as SimEnv measures it."""
    total = weight.sum(dim=-1).clamp(min=1.0)
    variance = (weight * (values - mean.unsqueeze(-1)) ** 2).sum(dim=-1) / total
    return torch.sqrt(variance.clamp(min=0.0) + 1e-24)


class CheetahTwin:
    """A differentiable Cheetah model of the stretch SimEnv tracks."""

    @classmethod
    def for_env(cls, env, num_particles: int | None = None) -> "CheetahTwin":
        """A twin of the machine `env` is actually simulating.

        Everything that decides what the lattice *is* — which file, where it starts
        and ends, whether the vacuum chamber is in it, whether the L4H linearizer is
        powered or standing in as a drift — is read off the environment's own
        config and run through the environment's own helpers, rather than being
        guessed at from defaults that match today.

        That is worth the indirection. Three times now the environment has changed
        underneath this twin — apertures added, the VBC turned on, L4H switched for
        a drift — and each time a twin built from its own assumptions went on
        planning confidently against a machine that no longer existed, out by a
        millimetre or more at the readout with nothing obviously wrong. Deriving
        from the config means the next such change arrives here too.
        """
        from rl_framework.envs import sim_env as _sim

        config = env.config
        lattice = config.nominal_lattice_json or env.model.LATTICE_JSON
        if getattr(config, "l4h_off", False):
            lattice = _sim._without_l4h(lattice)
        return cls(
            lattice_json=lattice,
            distribution=config.distribution_path or DISTRIBUTION_NPZ,
            start=config.lattice_start,
            end=config.lattice_end,
            num_particles=num_particles or config.num_particles or 10_000,
            beam_pipe_radius=config.beam_pipe_radius,
            min_screen_fraction=config.min_screen_fraction,
            apertures=getattr(config, "apertures", True),
        )

    def __init__(
        self,
        lattice_json: str | Path = LATTICE_JSON,
        distribution: str | Path = DISTRIBUTION_NPZ,
        start: str = LATTICE_START,
        end: str = READOUT_SCREEN,
        num_particles: int = 10_000,
        beam_pipe_radius: float | None = BEAM_PIPE_RADIUS,
        min_screen_fraction: float = MIN_SCREEN_FRACTION,
        apertures: bool = True,
    ):
        # Built through the nominal model's own `cheetah_segment` rather than by
        # partitioning the file here. The two are the same lattice, and the moment
        # they are assembled by two different pieces of code they can stop being so:
        # `apertures` adds CLARA's real vacuum chamber on top of the file, SimEnv
        # turns it on by default, and a twin that quietly left it off disagreed with
        # the machine by a millimetre and a half at every BPM. Deferring to the model
        # means the next thing it learns about the lattice arrives here too.
        self.segment = _nominal_model().cheetah_segment(
            str(lattice_json), start, end, apertures
        )
        self._elements = {element.name: element for element in self.segment.elements}
        self.readout_screen = end
        self._radius = beam_pipe_radius
        self._min_fraction = min_screen_fraction

        for element in self.segment.elements:
            if isinstance(element, cheetah.Screen):
                # The agent reads moments, not camera frames; an active screen would
                # render an image at every diagnostic in the line for nothing.
                element.is_active = False

        # Where the beam is read and where it is scraped, in order along the line.
        # `SimEnv._stops`, and the same tags.
        diagnostics = [
            (index, element)
            for index, element in enumerate(self.segment.elements)
            if isinstance(element, cheetah.Screen)
        ]
        tagged = [
            (index, element)
            for index, element in diagnostics
            if any(tag in element.name for tag in (BPM_TAG, SCREEN_TAG, APERTURE_TAG))
        ]
        self._stop_names = [element.name for _, element in tagged]
        self.bpm_names = [n for n in self._stop_names if BPM_TAG in n]
        self.screen_names = [n for n in self._stop_names if SCREEN_TAG in n]
        self.diagnostic_names = self.bpm_names + self.screen_names
        self._bpm_slot = {name: i for i, name in enumerate(self.bpm_names)}
        self._diagnostic_slot = {n: i for i, n in enumerate(self.diagnostic_names)}

        self._extent = {
            element.name: tuple(float(v) for v in element.extent)
            for _, element in diagnostics
        }
        self._nominal_offset = {
            element.name: np.array([float(v) for v in element.misalignment])
            for _, element in diagnostics
        }

        # One sub-segment per stop, over the same element objects, so each stretch is
        # one fused transfer map instead of an element-by-element walk. `SimEnv`'s
        # `CheetahTransport` builds exactly this.
        self._chunks, previous = {}, 0
        for index, element in tagged:
            self._chunks[element.name] = cheetah.Segment(
                self.segment.elements[previous:index]
            )
            previous = index

        # Elements a survey error can move, as `score.sample_misalignments` moves them.
        self.misalignable = [
            element.name
            for element in self.segment.elements
            if isinstance(element, (cheetah.Quadrupole, cheetah.Sextupole))
        ]
        self._misalignable_slot = {n: i for i, n in enumerate(self.misalignable)}
        self._nominal_misalignment = {
            name: np.array([float(v) for v in self._elements[name].misalignment])
            for name in self.misalignable
        }

        self.element_types = {
            name: type(element).__name__ for name, element in self._elements.items()
        }
        # SimEnv offers the chicane as one knob; the twin has to know the name too.
        if all(name in self._elements for name in VBC_DIPOLES):
            self.element_types[VBC_CHICANE] = "Chicane"

        # The strengths the lattice file carries, taken once, before anything has been
        # written. They cannot be read back off the elements later: by then they hold
        # whatever was last set, which during planning is a whole batch of candidates.
        self._nominal_knob: dict[tuple[str, str], float] = {}
        for name, element in self._elements.items():
            for prop, _ in KNOB_PROPERTIES.get(type(element).__name__, ()):
                self._nominal_knob[(name, prop)] = float(getattr(element, prop))
        if VBC_CHICANE in self.element_types:
            self._nominal_knob[(VBC_CHICANE, "angle")] = float(
                getattr(self._elements[VBC_DIPOLES[0]], "angle")
            )

        with np.load(distribution, allow_pickle=False) as data:
            coordinates = data["particles"][:num_particles]
            self.energy_ev = float(data["energy_ev"])
        # The nominal beam, and its own centre, which the size factors act about.
        self._reference = torch.tensor(np.ascontiguousarray(coordinates), dtype=DTYPE)
        self._reference_centre = self._reference.mean(dim=0)
        self._energy = torch.tensor(self.energy_ev, dtype=DTYPE)
        self.num_particles = len(coordinates)

        # What every knob is set to, in the lattice file's units. The agent mirrors
        # its own set points into this, so the twin always models the machine as the
        # agent last left it. `_written` is everything that has ever been assigned,
        # set point or planner candidate, so each track can start from a clean slate.
        self._setpoints: dict[tuple[str, str], float] = {}
        self._written: set[tuple[str, str]] = set()

    # ---------- magnets ----------

    def knob_property(self, element: str, prop: str | None = None) -> str:
        """The property a knob name refers to, defaulting by element type."""
        if prop is not None:
            return prop
        kind = self.element_types[element]
        return {
            "Quadrupole": "k1",
            "Sextupole": "k2",
            "Cavity": "phase",
            "Chicane": "angle",
        }[kind]

    def nominal_setting(self, element: str, prop: str) -> float:
        """What the lattice file has a knob at, before the agent touches it."""
        try:
            return self._nominal_knob[(element, prop)]
        except KeyError:
            raise KeyError(f"{element} has no {prop} to set") from None

    def _write(self, element: str, prop: str, value: torch.Tensor | float) -> None:
        """Set one knob on the twin, the chicane's four dipoles together.

        The value is assigned rather than written into: `requires_grad` only takes
        effect on a tensor that is assigned, so `element.k1.requires_grad_(True)`
        after the fact silently tracks nothing and the output comes back with no
        gradient at all. `CheetahTransport.write` assigns for the same reason.
        """
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(float(value), dtype=DTYPE)
        self._written.add((element, prop))
        if element == VBC_CHICANE:
            for dipole, sign in zip(VBC_DIPOLES, VBC_SIGNS):
                setattr(self._elements[dipole], prop, sign * value)
                setattr(self._elements[dipole], VBC_EDGE[dipole], sign * value)
            return
        setattr(self._elements[element], prop, value)

    def set_knobs(self, settings: Mapping[str, float], props: Mapping[str, str] | None = None) -> None:
        """Mirror the agent's set points onto the twin, keyed by element name."""
        for element, value in settings.items():
            prop = self.knob_property(element, (props or {}).get(element))
            self._setpoints[(element, prop)] = float(value)

    def set_knob_vector(
        self, elements: Sequence[str], props: Sequence[str], values: Iterable[float]
    ) -> None:
        """Mirror a `KnobSet`'s vector onto the twin."""
        for element, prop, value in zip(elements, props, values):
            self._setpoints[(element, prop)] = float(value)

    def setting(self, element: str, prop: str) -> float:
        """What the twin currently has a knob at, set point or lattice file."""
        return self._setpoints.get((element, prop), self.nominal_setting(element, prop))

    def reset_setpoints(self) -> None:
        """Back to the lattice file, as `SimEnv.reset` puts the machine."""
        self._setpoints.clear()

    def _apply_setpoints(self, overrides: Mapping[tuple[str, str], torch.Tensor] | None) -> None:
        """Put every knob where the agent left it, then the planner's candidates on top.

        Every knob ever written is rewritten, not just the ones with a set point. A
        planner leaves a whole population of candidate values on an element — a
        tensor with a batch dimension — and if that were left in place the next
        track would either silently use it or fail to broadcast against a batch of a
        different size. Rewriting from the set points, or from the lattice file where
        there is none, makes each call independent of the last.
        """
        for element, prop in self._written | set(self._setpoints):
            self._write(element, prop, self.setting(element, prop))
        for (element, prop), value in (overrides or {}).items():
            self._write(element, prop, value)

    def _apply_misalignments(self, state: LatentState) -> None:
        """Where the survey error has each magnet, or the lattice file where it has none."""
        if state.misalignments is None:
            for name in self.misalignable:
                self._elements[name].misalignment = torch.tensor(
                    self._nominal_misalignment[name], dtype=DTYPE
                )
            return
        for name in self.misalignable:
            nominal = torch.tensor(self._nominal_misalignment[name], dtype=DTYPE)
            self._elements[name].misalignment = (
                nominal + state.misalignments[self._misalignable_slot[name]]
            )

    def _offset(self, name: str, state: LatentState) -> torch.Tensor:
        """Where a diagnostic's own centre sits, its fitted centring error included."""
        nominal = torch.tensor(self._nominal_offset[name], dtype=DTYPE)
        if state.diagnostic_offsets is None or name not in self._diagnostic_slot:
            return nominal
        return nominal + state.diagnostic_offsets[self._diagnostic_slot[name]]

    # ---------- the beam ----------

    def beam_coordinates(self, state: LatentState) -> torch.Tensor:
        """The episode's beam, from the nominal one, differentiably in `state`.

        `SimEnv._perturb_beam` exactly: each plane scaled about the beam's own centre,
        then the whole beam offset.
        """
        centred = self._reference - self._reference_centre
        return centred * state.scale + self._reference_centre + state.centroid

    # ---------- tracking ----------

    def readings(
        self,
        state: LatentState | None = None,
        overrides: Mapping[tuple[str, str], torch.Tensor] | None = None,
        screen: str | None = None,
        batch: int | None = None,
        soft: float | None = None,
        image_shape: tuple[int, int] | None = None,
    ) -> Readings:
        """Track the beam to `screen` and read every diagnostic on the way.

        `overrides` are the knobs being planned, as tensors, so the readings come back
        with a gradient to them; everything else sits at its mirrored set point. A
        tensor with a leading batch dimension plans a whole population at once, and
        every reading comes back batched to match.

        `batch` forces that leading dimension even when nothing varies across it,
        which is what a fit over several shots at identical magnet settings needs: the
        readings have to line up with the measurements row for row whether the knobs
        moved between shots or not.

        `soft` blurs the beam pipe and the camera chip over that many metres. Leave it
        None to get what SimEnv would read; set it, typically to a millimetre, when
        the answer is going to be differentiated, because whether a particle is inside
        a wall is a step function and a sum of ten thousand step functions is a
        staircase, not a slope.
        """
        state = state or LatentState.nominal()
        screen = screen or self.readout_screen
        self._apply_setpoints(overrides)
        self._apply_misalignments(state)

        particles = self.beam_coordinates(state)
        beam = cheetah.ParticleBeam(
            particles=torch.cat(
                [particles, torch.ones(len(particles), 1, dtype=DTYPE)], dim=1
            ),
            energy=self._energy,
        )

        count = self.num_particles
        minimum = self._min_fraction * count
        screen_index = self._stop_names.index(screen)

        bpm_x: dict[str, torch.Tensor] = {}
        bpm_y: dict[str, torch.Tensor] = {}
        bpm_valid: dict[str, torch.Tensor] = {}
        alive: torch.Tensor | None = None  # what Cheetah's own apertures left
        margin: torch.Tensor | None = None  # m inside the pipe at its tightest so far
        outside: list[torch.Tensor] = []
        exposure = None

        # Everything past the screen is in its shadow: a screen in the beam stops it.
        for name in self._stop_names[: screen_index + 1]:
            chunk = self._chunks[name]
            if len(chunk.elements):
                beam = chunk.track(beam)
            x, y = beam.x, beam.y
            survived = (beam.survival_probabilities > 0.5).to(DTYPE).expand_as(x)
            alive = survived if alive is None else alive * survived
            if self._radius is not None:
                # How far inside the pipe each particle is, at the worst stop it has
                # passed. A particle is lost if it was ever outside, so the running
                # minimum is exactly the product of the per-stop tests — and unlike
                # that product it is a single number that can be softened, which is
                # what gives a fit or a planner a gradient to follow. Half the beam
                # is scraped at the nominal tune, so without that the objective is a
                # staircase and gradient descent only rattles between steps of it.
                here = self._radius - torch.sqrt(x**2 + y**2 + 1e-18)
                margin = here if margin is None else torch.minimum(margin, here)
                outside.append((-here).clamp(min=0.0).mean(dim=-1))
            live = self._live(alive, margin, soft)

            if name in self._bpm_slot:
                offset = self._offset(name, state)
                bpm_valid[name] = (live.sum(dim=-1) >= minimum).to(DTYPE)
                bpm_x[name] = _masked_mean(x, live) - offset[0]
                bpm_y[name] = _masked_mean(y, live) - offset[1]
            if name == screen:
                exposure = self._expose(name, x, y, live, state, soft, image_shape)

        assert exposure is not None, f"{screen} is not a stop of this line"
        measured, raw, on_screen, fraction, image = exposure
        transmission = live.sum(dim=-1) / count
        scrape = (
            torch.stack(_broadcast(outside), dim=-1).mean(dim=-1)
            if outside
            else torch.zeros_like(on_screen)
        )

        # One common batch shape across the whole line, then the BPMs in `bpm_names`
        # order. A BPM in the screen's shadow was never read and reads zero.
        shape = torch.broadcast_shapes(
            on_screen.shape, transmission.shape, scrape.shape,
            *(tensor.shape for tensor in bpm_x.values()),
            *(() if batch is None else ((batch,),)),
        )
        zero = torch.zeros(shape, dtype=DTYPE)
        batch = shape
        stack = lambda source: torch.stack(  # noqa: E731 — three identical one-liners
            [source.get(name, zero).expand(batch) for name in self.bpm_names], dim=-1
        )
        return Readings(
            bpm_x=stack(bpm_x),
            bpm_y=stack(bpm_y),
            bpm_valid=stack(bpm_valid),
            mu_x=measured[0].expand(batch),
            mu_y=measured[1].expand(batch),
            sigma_x=measured[2].expand(batch),
            sigma_y=measured[3].expand(batch),
            on_screen=on_screen.expand(batch),
            on_screen_fraction=fraction.expand(batch),
            transmission=transmission.expand(batch),
            raw_mu_x=raw[0].expand(batch),
            raw_mu_y=raw[1].expand(batch),
            raw_sigma_x=raw[2].expand(batch),
            raw_sigma_y=raw[3].expand(batch),
            scrape=scrape.expand(batch),
            image=image,
        )

    @staticmethod
    def _live(alive: torch.Tensor, margin: torch.Tensor | None, soft: float | None):
        """Which particles still count, from Cheetah's apertures and the beam pipe."""
        if margin is None:
            return alive
        return alive * _aperture(margin, soft)

    def _expose(self, name, x, y, live, state, soft=None, image_shape=None):
        """What the camera measures, and what the beam is actually doing.

        The measured set is `SimEnv._expose` exactly: off the screen's own centre,
        only the particles that landed on the chip, and zero unless enough of them
        did. The raw set is the same moments over every particle tracked, which no
        camera could report but a planner needs — it is what still has a gradient
        once the beam has walked off the chip.
        """
        extent = self._extent[name]
        offset = self._offset(name, state)
        x = x - offset[0]
        y = y - offset[1]
        # The chip edge gets the same treatment as the pipe wall, and for the same
        # reason: a beam half off the screen has to know which way to come back.
        inside = torch.minimum(extent[1] - x.abs(), extent[3] - y.abs())
        on_chip = live * _aperture(inside, soft)

        fraction = on_chip.sum(dim=-1) / self.num_particles
        on_screen = (fraction >= self._min_fraction).to(DTYPE)
        mu_x = _masked_mean(x, on_chip)
        mu_y = _masked_mean(y, on_chip)
        measured = (
            mu_x * on_screen,
            mu_y * on_screen,
            _masked_std(x, on_chip, mu_x) * on_screen,
            _masked_std(y, on_chip, mu_y) * on_screen,
        )

        everything = torch.ones_like(x)
        raw_mu_x, raw_mu_y = x.mean(dim=-1), y.mean(dim=-1)
        raw = (
            raw_mu_x,
            raw_mu_y,
            _masked_std(x, everything, raw_mu_x),
            _masked_std(y, everything, raw_mu_y),
        )
        return measured, raw, on_screen, fraction, self._render(
            name, x, y, on_chip, image_shape
        )

    def _render(self, name, x, y, on_chip, shape):
        """The camera frame, binned as `SimEnv` bins it: rows y, columns x.

        A histogram of the particles that landed on the chip, over the chip's own
        extent, so a twin standing in for the environment hands a policy the same
        picture the environment would.
        """
        if shape is None or x.dim() > 1:
            return None
        keep = on_chip.detach() > 0.5
        extent = self._extent[name]
        rows = torch.linspace(extent[2], extent[3], int(shape[0]) + 1)
        columns = torch.linspace(extent[0], extent[1], int(shape[1]) + 1)
        image, _, _ = np.histogram2d(
            y.detach().numpy()[keep.numpy()],
            x.detach().numpy()[keep.numpy()],
            bins=(rows.numpy(), columns.numpy()),
        )
        return torch.tensor(image, dtype=DTYPE)

    def screen_extent(self, screen: str | None = None) -> tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max) in metres that a screen spans."""
        return self._extent[screen or self.readout_screen]
