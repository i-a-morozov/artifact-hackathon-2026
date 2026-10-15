"""Cheetah-only access to the new nominal optics, aperture table, and measured beam.

Aperture placement mirrors nominal_model/xsuite_model.py without requiring xsuite.
No scorer globals or files are modified. The GPSR tune is an explicit alternative,
not the default nominal tune recommended by nominal_model/README.md.
"""
from pathlib import Path
import json
import cheetah
import numpy as np
import torch
import score
from .beamline_tuning import knobs_for_elements, apply_knobs

ROOT = Path(__file__).resolve().parents[1]
INPUT_MARKER = 'cla_s02_sim_aper_01'
APERTURES = ROOT / 'data/clara/lattice_data/apertures.json'
DISTRIBUTION = ROOT / 'nominal_model/initial_distribution.npz'


def add_apertures(segment, path=APERTURES):
    table = json.loads(Path(path).read_text())
    pipe, collimators = ({e['element']: e for e in table[k]} for k in ('pipe','collimators'))
    elements, chamber = [], None
    def aperture(entry, name):
        return cheetah.Aperture(x_max=torch.tensor(entry['x_max']),
            y_max=torch.tensor(entry['y_max']), shape=entry['shape'], name=name)
    for element in segment.elements:
        name = element.name
        collimator = collimators.get(name)
        if collimator is not None:
            elements.append(aperture(collimator, name+'_col_in'))
        elements.append(element)
        if name in pipe:
            chamber = pipe[name]
            elements.append(aperture(chamber,name+'_pipe'))
        elif collimator is not None:
            elements.append(aperture(collimator,name+'_col_out'))
        elif chamber is not None and float(getattr(element,'length',0.)) > 0:
            elements.append(aperture(chamber,name+'_pipe'))
    return cheetah.Segment(elements,name=segment.name)


def load_lattice(*, model='nominal', errors=False, seed=0, setpoint_fraction=.05,
                 misalignment_severity=1., apertures=True, end='cla_feh_dia_scr_01'):
    """Fresh ideal model, or a separately perturbed machine with reproducible errors.

    Errors are explicit normalized corrector/quad setting perturbations plus scorer-
    scale magnet/screen misalignments. These are not scorer sampled cases. model=scorer
    delegates to score.load_lattice and thus retains any active scorer-case errors.
    """
    if model == 'scorer':
        lattice = score.load_lattice(count_evaluations=False)
    else:
        paths = {'nominal': ROOT/'data/clara/lattice_data/CLARA_cheetah.json',
                 'gpsr': ROOT/'nominal_model/CLARA_cheetah_S07_GPSR.json'}
        if model not in paths:
            raise ValueError('model must be nominal, gpsr, or scorer')
        lattice = cheetah.Segment.from_lattice_json(str(paths[model]))
        if apertures:
            lattice = add_apertures(lattice)
        lattice = lattice.partition_at(INPUT_MARKER, mode='after')[-1]
    if end is not None:
        if end not in [e.name for e in lattice.elements]:
            raise ValueError(f'Unknown model endpoint: {end}')
        lattice = lattice.partition_at(end,mode='after')[0]
    if errors:
        if setpoint_fraction < 0 or misalignment_severity < 0:
            raise ValueError('Error magnitudes must be nonnegative')
        rng = np.random.default_rng(seed)
        knobs = knobs_for_elements(lattice.elements,include_rf_phase=False)
        center = np.array([k.normalized_initial() for k in knobs])
        apply_knobs(lattice,knobs,np.clip(center+rng.normal(0,setpoint_fraction,len(knobs)),0,1))
        offsets = {}
        for e in lattice.elements:
            if isinstance(e,(cheetah.Quadrupole,cheetah.Sextupole)):
                offsets[e.name] = rng.normal(0,score.MISALIGNMENT_RMS*misalignment_severity,2)
            elif isinstance(e,cheetah.Screen) and e.name == score.READOUT_SCREEN:
                offsets[e.name] = rng.normal(0,score.SCREEN_MISALIGNMENT_RMS*misalignment_severity,2)
        score.apply_misalignments(lattice,offsets)
    return lattice


def load_beam(num_particles=10000, *, seed=0, severity=0., path=DISTRIBUTION):
    """Sample stored correlated 6D particles; optional affine injection jitter.

    Stored coordinates already use Cheetah conventions (no tau sign conversion).
    Severity is our explicit jitter model, not score.sample_beam's distribution.
    Total charge is assumed 250 pC because the NPZ does not specify a charge.
    """
    if num_particles < 2 or severity < 0:
        raise ValueError('Need at least two particles and nonnegative severity')
    rng = np.random.default_rng(seed)
    with np.load(path,allow_pickle=False) as data:
        if str(data['location']) != INPUT_MARKER:
            raise ValueError('Nominal distribution is not at the expected entrance')
        source = data['particles']
        coordinates = source[rng.choice(len(source),num_particles,replace=num_particles>len(source))].copy()
        energy = float(data['energy_ev'])
    if severity:
        scales = np.exp(rng.normal(0,.1*severity,6))
        jitter = np.array([1e-4,2e-5,1e-4,2e-5,0.,0.])*severity
        coordinates = coordinates*scales+rng.normal(0,jitter)
    particles = np.column_stack((coordinates,np.ones(num_particles)))
    return cheetah.ParticleBeam(particles=torch.tensor(particles,dtype=torch.float32),
        energy=torch.tensor(energy,dtype=torch.float32),
        particle_charges=torch.full((num_particles,),250e-12/num_particles,dtype=torch.float32))
