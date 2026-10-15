"""Regressions for missing beams, counting, and independent section methods."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from score import load_lattice
from work.beam_tuning import BudgetExhausted, sample_beam
from work.beamline_tuning import apply_knobs, track_interval, beam_moments
from work.section_response import make_section_plan, build_response_model, ideal_kickoff_beam, transverse_state
from work.section_measurement import SectionSession, FAILURE_VALUE, full_bo_search, section_readout
from work.section_rl import SectionRLConfig, SectionRLEnv


class SectionMeasurements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.lattice = load_lattice(count_evaluations=False)
        cls.plan = make_section_plan(cls.lattice, 'cla_s04_sim_aper_01', 'cla_s04_dia_scr_03')
        cls.beam = ideal_kickoff_beam(384)
        cls.model = build_response_model(cls.lattice, cls.plan, cls.beam)

    def test_joint_bo_budget_and_measured_selection(self):
        session = SectionSession(self.lattice, self.plan, sample_beam(717, 384),
                                 self.model.target, budget=5, noise_fraction=.01)
        choice = full_bo_search(session, self.model.center, seed=4)
        self.assertEqual(len(session.records), 5)
        np.testing.assert_equal(session.records[0]['u'], self.model.center)
        np.testing.assert_equal(choice, session.best()['u'])
        self.assertTrue(all(np.isfinite(r['value']) for r in session.records))
        with self.assertRaises(BudgetExhausted): session.measure(choice)

    def test_zero_physics_never_calls_solver(self):
        env = SectionRLEnv(self.lattice, self.plan, self.beam, self.model, self.model.center,
                          config=SectionRLConfig(horizon=1, physics_weight=0))
        env.reset()
        with patch.object(env, 'physics_action', side_effect=AssertionError('physics called')):
            env.step(np.zeros(env.action_dim))
        with self.assertRaises(RuntimeError): env.step(np.zeros(env.action_dim))

    def test_lost_beam_has_finite_reward_and_observation(self):
        lost = self.beam.clone()
        lost.survival_probabilities.zero_()
        self.assertTrue(np.isnan(section_readout(lost)[0]).all())
        for weight in [0, .7]:
            env = SectionRLEnv(self.lattice, self.plan, lost, self.model, self.model.center,
                              config=SectionRLConfig(horizon=1, physics_weight=weight))
            obs = env.reset()
            self.assertTrue(np.isfinite(obs).all())
            self.assertEqual(obs[-1], 0)
            self.assertEqual(env.current_loss, FAILURE_VALUE)
            obs, reward, done, _ = env.step(np.zeros(env.action_dim))
            self.assertTrue(np.isfinite(obs).all())
            self.assertTrue(np.isfinite(reward))
            self.assertTrue(done)
            self.assertEqual(len(env.session.records), 2)

    def test_noise_error_matches_scaled_components(self):
        session = SectionSession(self.lattice, self.plan, self.beam, self.model.target,
                                 budget=1, noise_fraction=.02, scale=self.model.scale)
        _, error = session.measure(self.model.center)
        sigma = .02 * session.records[0]['clean'][2:].mean()
        self.assertAlmostEqual(error, sigma*np.linalg.norm(1/self.model.scale)/4)

    def test_response_scan_restores_nominal_after_failure(self):
        original = track_interval
        calls = 0
        def failing_track(*args):
            nonlocal calls
            calls += 1
            if calls == 3: raise RuntimeError('failed probe')
            return original(*args)
        with patch('work.section_response.track_interval', side_effect=failing_track):
            with self.assertRaises(RuntimeError):
                build_response_model(self.lattice, self.plan, self.beam)
        for knob in self.plan.knobs:
            actual = float(getattr(getattr(self.lattice, knob.element_name), knob.property))
            self.assertAlmostEqual(actual, knob.physical(knob.normalized_initial()), places=6)

    def test_covariance_excludes_lost_particles(self):
        beam = self.beam.clone()
        beam.survival_probabilities[:100] = 0
        center, covariance = transverse_state(beam)
        live = beam.particles[100:, :4].double()
        np.testing.assert_allclose(center, live.mean(0).numpy())
        np.testing.assert_allclose(covariance, torch.cov(live.T).numpy(), atol=1e-18)


if __name__ == '__main__': unittest.main()
