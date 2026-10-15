from dataclasses import dataclass, field
from typing import Any


@dataclass
class RLConfig:
    # Path to the cheetah lattice JSON
    lattice_path: str = "FEBE_full.json"

    # Beam factory parameters (Twiss + energy + charge)
    beam: dict[str, Any] = field(default_factory=lambda: {
        "num_particles": 10000,
        "alpha_x": -0.949, "beta_x": 14.236, "emittance_x": 3.29e-9,
        "alpha_y": 72.003, "beta_y": 925.297, "emittance_y": 4.20e-9,
        "sigma_p": 0.01, "energy": 250e6, "total_charge": 10e-12,
    })

    # The screen the request is for: where every episode starts with the screen in
    # the beam, and the only screen an episode is scored on. The agent may move the
    # screen elsewhere to look around, but that costs a shot and brings back no new
    # reading of the request — the score holds at the last reading of this screen,
    # as `score.py` measures only at its readout however many screens a tuner used.
    target_screen: str = "cla_feh_dia_scr_01"

    # Magnets the RL agent can control.
    # Each entry: [name, min_native_value, max_native_value]
    # Native units: k1 (1/m²) for quadrupoles, angle (mrad) for correctors.
    magnet_config: list[list] = field(default_factory=lambda: [
        ["cla_s04_mag_hvcor_01_h",  -5.0e-3, 5.0e-3],
        ["cla_s04_mag_hvcor_01_v",  -5.0e-3, 5.0e-3],
        ["cla_s04_mag_quad_01",  -15.0, 15.0],
        ["cla_s04_mag_quad_02",  -15.0, 15.0],
        ["cla_s04_mag_quad_03",  -15.0, 15.0],
    ])

    # Episode length before truncation
    max_episode_steps: int = 100

    # ---------------- nominal xsuite model (SimEnv only) ---------------- #
    # Everything below describes the model in `nominal_model/`, which SimEnv tracks
    # through. `lattice_path`, `beam` and `magnet_config` above are for RealEnv.

    # Which model tracks the beam. "cheetah" is first-order magnets and achromatic
    # cavities, the physics `score.py` is scored on, and starts in a second;
    # "xsuite" is thick elements and cavities that focus at each particle's own
    # energy, closer to the machine off momentum, and spends a minute compiling a
    # tracking kernel. Same lattice, same beam, same interface either way.
    backend: str = "cheetah"

    # The X-band linearizer L4H (`cla_l4h_lin_cav_01`) is off on the machine: True
    # tracks it as a drift of its own length (its iris aperture stays), as
    # `orbit_matching/` does. False powers it as the lattice file has it.
    l4h_off: bool = True

    # Where `xsuite_model.py` lives, and what it is given. None takes the defaults
    # of the nominal model: the challenge lattice file and its starting beam.
    nominal_model_dir: str | None = None
    nominal_lattice_json: str | None = None
    distribution_path: str | None = None

    # The stretch of machine that is modelled. The default is the benchmark's:
    # the start of S02 to the FEBE hutch screen scored by `score.py`.
    lattice_start: str = "cla_s02_sim_aper_01"
    lattice_end: str = "cla_feh_dia_scr_01"

    # Particles tracked per step, taken from the front of the 200,000 in the
    # distribution file. None uses all of them (slow).
    num_particles: int | None = 10_000

    # Screen images are binned onto this (rows, columns) grid, covering the whole
    # chip whatever its native resolution is. `env.screen_extent(name)` gives the
    # physical size the grid spans.
    screen_shape: tuple[int, int] = (128, 128)
    # Divide each image by its brightest pixel, so observations are in [0, 1].
    normalise_image: bool = True

    # Beam pipe the particles have to stay inside; a particle that leaves it stops
    # counting towards every reading further down the line. None removes the pipe.
    beam_pipe_radius: float | None = 50e-3
    # The machine's vacuum chamber (`data/clara/lattice_data/apertures.json`, from
    # STFC's SimFrame model of CLARA FEBE), checked along the whole line by both
    # backends: a 17 mm half-aperture pipe, the linac irises, the TDC and the FEBE
    # arc dipole chambers. False leaves only `beam_pipe_radius`.
    apertures: bool = True
    # Fraction of the launched beam a diagnostic needs for its reading to count.
    min_screen_fraction: float = 0.10
    # Gaussian noise added to each BPM reading, in metres.
    bpm_noise: float = 0.0

    # How much the machine varies from one episode to the next. Each is a multiplier
    # on the spreads at the top of `sim_env.py`: 0 is the nominal model exactly, 1 is
    # the machine as it realistically differs shot to shot and day to day, and larger
    # values stay a harder version of the same problem rather than a different one.
    # `reset()` draws a new beam and a new survey error from them; the magnets
    # always start at the strengths the lattice file asks for.
    beam_variation: float = 1.0
    misalignment_variation: float = 1.0

    # What the beam is asked for at the observed screen: (mu_x, mu_y, sigma_x,
    # sigma_y) in metres, off the screen's centre. None draws a new request every
    # episode from the two lines below, as `score.sample_target` does; a 4-tuple
    # fixes it. The reward is the error against it in units of the requested spot.
    target: tuple[float, float, float, float] | None = None
    requested_spot: tuple[float, float] = (80e-6, 400e-6)  # m, log-uniform
    requested_offset: float = 2e-3  # m, uniform about the screen centre

    # How that error becomes a reward. "log" keeps a gradient however far the beam
    # is from the request; "linear" is the benchmark's own score, which stops
    # counting at ten requested spot sizes and so is flat across most of the range
    # an untuned machine sits in. `info["benchmark_score"]` is the benchmark's
    # number either way.
    reward_shape: str = "log"

    # Shot-to-shot jitter, as a fraction of `beam_variation`: within an episode the
    # machine holds still, but every step draws its own beam around the episode's,
    # so two identical actions do not give identical readings. Scales with
    # `beam_variation`, so setting that to 0 leaves the environment deterministic.
    beam_jitter: float = 0.1

    # Power supply ripple: how far a magnet's actual field sits from what it was
    # set to, as a fraction of the setting, drawn fresh for every magnet on every
    # step and undone once the shot has been tracked. 0.001 is 0.1%. At 0, the
    # default, nothing is drawn, written or restored at all. Cavity phases are not
    # rippled — RF stability is a different mechanism and a fraction of a phase
    # means nothing.
    power_supply_ripple: float = 0.0

    # Where each magnet's operating range comes from. "current" is what
    # +-`magnet_current_limit` amps reaches through the calibration in
    # `data/clara/lattice_data/utils.py` — what the hardware can physically do.
    # "archiver" is where the magnets happened to run over six months, from
    # `magnet_ranges.json`; far narrower, and it has the FEA arc quadrupoles at
    # [0, 0]. Magnets with no calibration (the sextupoles) fall back to the
    # archiver either way.
    magnet_range_source: str = "current"
    magnet_current_limit: float = 100.0  # A

    # Offer the three powered linac cavities' RF phases as knobs, in degrees, each
    # within `cavity_phase_range` of the phase the lattice file runs it at. Phase
    # sets the chirp, and so the energy spread the dogleg turns into beam size at
    # the readout. `score.TUNABLE` allows cavity phases, so a tune that moves one
    # is scorable; voltage and frequency are not.
    include_cavity_phases: bool = True
    cavity_phase_range: float = 30.0  # degrees either side of nominal

    # Offer the VBC chicane as a knob: four dipoles on one setting, up to 135 mrad.
    # With the cavity phases, it is the handle on the energy spread at the readout.
    # `score.TUNABLE_BY_NAME` allows the four VBC angles, so a tune that moves them
    # is scorable; every other dipole in the line is geometry and stays fixed.
    include_vbc_chicane: bool = True

    # Magnets the agent may set, by knob name (`<quadrupole>`, `<sextupole>`,
    # `<corrector>_h` / `_v`, or `cla_vbc_chicane`). None gives it every magnet in
    # the modelled stretch.
    controllable_magnets: list[str] | None = None
    # {knob name: [low, high]} overriding the ranges worked out above.
    magnet_limits: dict[str, list[float]] | None = None
