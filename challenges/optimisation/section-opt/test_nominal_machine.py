"""New nominal beam/apertures and independent error-bearing machine regressions."""
from copy import deepcopy
import unittest
import numpy as np
import torch
import cheetah
from work.nominal_machine import load_lattice,load_beam,APERTURES,DISTRIBUTION,INPUT_MARKER
from work.beamline_tuning import knobs_for_elements
from work.section_sequence import plan_screen_sequence,correct_screen_sequence
from work.section_response import track_to_boundary


class NominalMachine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_stored_correlated_coordinates_preserved(self):
        beam=load_beam(64,seed=19)
        with np.load(DISTRIBUTION) as data:
            idx=np.random.default_rng(19).choice(len(data['particles']),64,replace=False)
            np.testing.assert_allclose(beam.particles[:,:6].numpy(),data['particles'][idx],rtol=1e-6)
            self.assertAlmostEqual(float(beam.energy),float(data['energy_ev']),delta=4.)
        torch.testing.assert_close(beam.particles,load_beam(64,seed=19).particles)
        self.assertFalse(torch.equal(beam.particles,load_beam(64,seed=19,severity=.5).particles))

    def test_apertures_are_explicit_and_include_narrow_chambers(self):
        lattice=load_lattice()
        apertures=[e for e in lattice.elements if isinstance(e,cheetah.Aperture)]
        self.assertGreater(len(apertures),100)
        self.assertTrue(any(abs(float(e.x_max)-.0041195)<1e-8 for e in apertures))
        self.assertEqual(lattice.elements[0].name,INPUT_MARKER+'_pipe')
        self.assertEqual(lattice.elements[-1].name,'cla_feh_dia_scr_01')
        without=load_lattice(apertures=False)
        self.assertLess(sum(isinstance(e,cheetah.Aperture) for e in without.elements),len(apertures))

    def test_machine_errors_are_reproducible_and_reference_untouched(self):
        ideal=load_lattice();first=load_lattice(errors=True,seed=7);second=load_lattice(errors=True,seed=7)
        ideal_knobs=knobs_for_elements(ideal.elements,include_rf_phase=False)
        changed=0
        for knob in ideal_knobs:
            a=float(getattr(getattr(first,knob.element_name),knob.property))
            b=float(getattr(getattr(second,knob.element_name),knob.property))
            self.assertEqual(a,b)
            changed+=a!=knob.initial
        self.assertGreater(changed,0)
        quad=next(e for e in ideal.elements if isinstance(e,cheetah.Quadrupole))
        torch.testing.assert_close(getattr(first,quad.name).misalignment,getattr(second,quad.name).misalignment)
        self.assertFalse(torch.equal(quad.misalignment,getattr(first,quad.name).misalignment))
        # Loading fresh nominal optics must not depend on the preceding error-bearing load.
        fresh=load_lattice()
        self.assertEqual(float(getattr(fresh,quad.name).k1),float(quad.k1))

    def test_second_pass_starts_from_committed_settings(self):
        ideal=load_lattice();machine=load_lattice(errors=True,seed=8,setpoint_fraction=.01)
        beam=load_beam(128,seed=3)
        plans=plan_screen_sequence(ideal,max_sections=2,input_screen=INPUT_MARKER)
        torch.testing.assert_close(track_to_boundary(ideal,beam,INPUT_MARKER).particles,beam.particles)
        callbacks=[]
        def check(record,session):
            if session is None:return
            best=session.best()
            for knob,u in zip(session.plan.knobs,best['u']):
                self.assertAlmostEqual(float(getattr(getattr(machine,knob.element_name),knob.property)),knob.physical(u),delta=1e-5)
            self.assertEqual(record['best_measured_objective'],min(r['value'] for r in session.records))
            callbacks.append(record)
        first=correct_screen_sequence(machine,ideal,beam,beam,plans,budget=3,section_callback=check)
        committed=deepcopy(machine)
        second=correct_screen_sequence(machine,ideal,beam,beam,plans,budget=3,section_callback=check)
        for k in knobs_for_elements(committed.elements,include_rf_phase=False):
            self.assertEqual(float(getattr(getattr(second['initial_lattice'],k.element_name),k.property)),k.initial)
        self.assertEqual(first['online_measurements'],6)
        self.assertEqual(second['online_measurements'],6)
        self.assertEqual(len(callbacks),4)

if __name__=='__main__':unittest.main()
