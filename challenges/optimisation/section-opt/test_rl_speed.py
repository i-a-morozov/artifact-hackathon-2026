"""Fast RL transport must retain controls, loss physics, and measurement accounting."""
import unittest
import numpy as np
import torch
from score import load_lattice, sample_beam as score_beam
from work.beam_tuning import sample_beam
from work.beamline_tuning import apply_knobs, track_interval
from work.section_response import (make_section_plan, track_to_boundary,
                                   build_response_model, ideal_kickoff_beam)
from work.section_tracking import MomentTracker
from work.section_measurement import section_readout
from work.section_rl import SectionRLConfig, SectionRLEnv
from work.rl_tuning import ActorCritic, PPOConfig, train_ppo


class RLSpeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_particle_moments_and_losses_match_for_changed_knobs(self):
        for input_name, output_name in [('cla_s04_sim_aper_01','cla_s04_dia_scr_03'),
                                        ('cla_s06_dia_scr_02','cla_s07_dia_scr_01')]:
            lattice=load_lattice(False)
            plan=make_section_plan(lattice,input_name,output_name)
            beam=(sample_beam(12,2048) if 's04' in input_name else
                  track_to_boundary(lattice,score_beam(np.random.default_rng(12),num_particles=2048,severity=.2),input_name))
            fast=MomentTracker(lattice,plan)
            center=np.array([k.normalized_initial() for k in plan.knobs])
            for scale in (0.,.05,.4):
                u=np.clip(center+np.random.default_rng(41).normal(0,scale,len(center)),0,1)
                apply_knobs(lattice,plan.knobs,u)
                expected=track_interval(lattice,plan,beam)
                actual=fast(beam)
                np.testing.assert_array_equal(actual.survival_probabilities.numpy(),expected.survival_probabilities.numpy())
                np.testing.assert_allclose(section_readout(actual)[0],section_readout(expected)[0],rtol=2e-4,atol=2e-8,equal_nan=True)

    def test_cached_physics_solve_matches_bounded_reference(self):
        from scipy.optimize import lsq_linear
        lattice=load_lattice(False)
        plan=make_section_plan(lattice,'cla_s04_sim_aper_01','cla_s04_dia_scr_03')
        model=build_response_model(lattice,plan,ideal_kickoff_beam(256))
        env=SectionRLEnv(lattice,plan,sample_beam(12,256),model,model.center,
                        config=SectionRLConfig(physics_weight=.7))
        rng=np.random.default_rng(31)
        for index in range(12):
            env.u=rng.uniform(0,1,env.action_dim)
            if index % 2 == 0: env.u[0]=1.
            residual=rng.normal(0,2,4)
            env.measured=model.target+model.scale*residual
            actual=env.physics_action()*env.max_knob_step
            lower=np.maximum(-env.max_knob_step,-env.u)
            upper=np.minimum(env.max_knob_step,1-env.u)
            rhs=np.r_[-residual,np.zeros(env.action_dim)]
            reference=lsq_linear(env._regularized_response,rhs,
                                 bounds=(lower,upper),tol=1e-12).x
            np.testing.assert_allclose(env._regularized_response@actual,
                                       env._regularized_response@reference,atol=2e-5,rtol=2e-5)
            self.assertTrue(np.all(actual>=lower-1e-12) and np.all(actual<=upper+1e-12))

    def test_active_blocking_screen_is_preserved(self):
        lattice=load_lattice(False)
        plan=make_section_plan(lattice,'cla_s04_sim_aper_01','cla_s04_dia_scr_03')
        screen=lattice.elements[plan.stop_index]
        screen.is_active=True;screen.is_blocking=True
        output=MomentTracker(lattice,plan)(sample_beam(12,128))
        self.assertEqual(float(output.survival_probabilities.sum()),0.)

    def test_fast_slow_episode_and_progress_accounting(self):
        ref=load_lattice(False)
        plan=make_section_plan(ref,'cla_s04_sim_aper_01','cla_s04_dia_scr_03')
        model=build_response_model(ref,plan,ideal_kickoff_beam(256))
        beam=sample_beam(19,256)
        environments=[]
        for fast in (False,True):
            lattice=load_lattice(False)
            p=make_section_plan(lattice,plan.input_screen,plan.output_screen)
            env=SectionRLEnv(lattice,p,beam,model,model.center,
                config=SectionRLConfig(horizon=2,fast_tracking=fast,noise_fraction=.05))
            env.reset(seed=11)
            for _ in range(2):env.step(np.full(len(p.knobs),.1))
            environments.append(env)
        for slow,fast in zip(*(e.session.records for e in environments)):
            np.testing.assert_allclose(slow['observed'],fast['observed'],rtol=2e-4,atol=2e-8)
        env=environments[-1]
        callbacks=[]
        history=train_ppo(ActorCritic(env.observation_dim,env.action_dim,hidden=8),env,
                          PPOConfig(iterations=2,episodes_per_iteration=1,horizon=2,update_epochs=1),
                          seed=3,progress=callbacks.append)
        self.assertEqual(len(callbacks),2)
        self.assertEqual(history[-1]['offline_measurements'],6)
        self.assertGreater(history[-1]['elapsed_seconds'],0.)
        self.assertTrue(all(r['rollout_seconds']>0 and r['update_seconds']>0 for r in history))

if __name__=='__main__': unittest.main()
