"""Directions at active bounds and sequential state consistency regressions."""
import unittest
import io
from contextlib import redirect_stdout
import torch
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from score import load_lattice, sample_beam
from work.section_response import (direction_intervals, model_search_directions,
                                   track_to_boundary)
from work.section_sequence import plan_screen_sequence, correct_screen_sequence
from work.section_measurement import section_readout


class Directions(unittest.TestCase):
    def test_tiny_coupling_does_not_block_physical_axis(self):
        theta=1e-9
        basis=np.array([[1., theta],[-theta,1.]])
        model=SimpleNamespace(directions=basis, response=np.eye(2))
        start=np.ones(2)
        raw=direction_intervals(start,basis)
        self.assertEqual(raw[1,1]-raw[1,0],0.)
        clean=model_search_directions(model,start)
        np.testing.assert_allclose(clean,np.eye(2))
        self.assertTrue(np.all(np.diff(direction_intervals(start,clean),axis=1)>0))
        np.testing.assert_array_equal(model_search_directions(model,start,mode='raw_svd'),basis)

    def test_real_blocked_mode_replaced_by_full_rank_axes(self):
        basis=np.array([[1.,-1.],[1.,1.]])/np.sqrt(2)
        model=SimpleNamespace(directions=basis,response=np.eye(2))
        result=model_search_directions(model,np.zeros(2))
        self.assertEqual(np.linalg.matrix_rank(result),2)
        self.assertTrue(np.all(np.diff(direction_intervals(np.zeros(2),result),axis=1)>0))


class Sequence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference=load_lattice(count_evaluations=False)

    def test_planner_uses_nearest_sufficient_boundary(self):
        plans=plan_screen_sequence(self.reference)
        previous=[-1]
        from work.beamline_tuning import knobs_for_elements
        for plan in plans:
            count=lambda knobs:(sum(k.kind=='quadrupole' for k in knobs),sum(k.kind=='corrector' for k in knobs))
            q,c=count(plan.knobs)
            if q<4 or c<4: self.assertEqual(plan.start_index,-1)
            for start in previous:
                if start>plan.start_index:
                    nq,nc=count(knobs_for_elements(self.reference.elements[start+1:plan.stop_index],include_rf_phase=False))
                    self.assertTrue(nq<4 or nc<4)
            self.assertTrue(all(plan.start_index < next(i for i,e in enumerate(self.reference.elements) if e.name==k.element_name) < plan.stop_index for k in plan.knobs))
            previous.append(plan.stop_index)

    def test_sequence_recomputes_inputs_and_final_audit(self):
        import work.section_sequence as seq
        machine=load_lattice(count_evaluations=False)
        ideal=sample_beam(np.random.default_rng(4),num_particles=128,severity=0.)
        actual=sample_beam(np.random.default_rng(5),num_particles=128,severity=.1)
        plans=plan_screen_sequence(self.reference,max_sections=3)
        seen=[]
        original=seq.track_to_boundary
        def tracking(lattice,beam,boundary):
            if lattice is machine and beam is actual: seen.append(boundary)
            return original(lattice,beam,boundary)
        stream = io.StringIO()
        with redirect_stdout(stream), patch.object(seq,'track_to_boundary',side_effect=tracking):
            result=correct_screen_sequence(machine,self.reference,actual,ideal,plans,budget=2,verbose=True,log_every=1)
        self.assertIn('image 1/2', stream.getvalue())
        self.assertIn('Final audit:', stream.getvalue())
        self.assertIn('end-of-lattice transmission', stream.getvalue())
        self.assertGreater(result['elapsed_seconds'], 0.)
        self.assertEqual(result['online_measurements'],6)
        self.assertEqual(seen[:3],[p.input_screen for p in plans])
        self.assertEqual(seen[3:],[p.output_screen for p in plans])
        for row,plan in zip(result['audit'],plans):
            moments,_=section_readout(track_to_boundary(machine,actual,plan.output_screen))
            np.testing.assert_allclose([row['final_'+k+'_um'] for k in ('mu_x','mu_y','sigma_x','sigma_y')],moments*1e6)
        self.assertEqual(len(result['records']),3)
        for row in result['records']:
            self.assertAlmostEqual(row['corrected_transmission'],
                                   row['input_transmission']*row['corrected_section_transmission'])
        for row, plan in zip(result['audit'], plans):
            output = track_to_boundary(machine,actual,plan.output_screen)
            incoming = track_to_boundary(machine,actual,plan.input_screen)
            self.assertAlmostEqual(row['final_transmission'], seq.transmission(output,actual))
            self.assertAlmostEqual(row['final_section_transmission'], seq.transmission(output,incoming))

    def test_missing_ideal_target_is_explicitly_skipped(self):
        import work.section_sequence as seq
        machine=load_lattice(count_evaluations=False)
        beam=sample_beam(np.random.default_rng(2),num_particles=64,severity=0.)
        plans=plan_screen_sequence(self.reference,max_sections=1)
        with patch.object(seq, 'section_readout', return_value=(np.full(4,np.nan),0.0)):
            result=correct_screen_sequence(machine,self.reference,beam,beam,plans,budget=2)
        self.assertEqual(result['online_measurements'],0)
        self.assertTrue(result['records'][0]['status'].startswith('skipped:'))
        self.assertTrue(np.isnan(result['audit'][0]['final_objective']))

    def test_transmission_denominator_and_zero_input(self):
        from work.section_sequence import transmission
        def beam(weights):
            return SimpleNamespace(survival_probabilities=torch.tensor(weights))
        initial=beam([1.,1.,1.,1.])
        incoming=beam([1.,1.,0.,0.])
        outgoing=beam([1.,0.,0.,0.])
        self.assertEqual(transmission(outgoing,initial),.25)
        self.assertEqual(transmission(outgoing,incoming),.5)
        self.assertTrue(np.isnan(transmission(outgoing,beam([0.,0.,0.,0.]))))

if __name__=='__main__': unittest.main()
