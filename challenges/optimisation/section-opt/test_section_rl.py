import unittest

import numpy as np

from score import load_lattice
from work.beam_tuning import sample_beam
from work.rl_tuning import ActorCritic, PPOConfig, train_ppo
from work.section_response import build_response_model, ideal_kickoff_beam, make_section_plan
from work.section_rl import SectionRLConfig, SectionRLEnv, run_section_policy


class SectionRL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lattice = load_lattice(count_evaluations=False)
        cls.plan = make_section_plan(
            cls.lattice,
            "cla_s04_sim_aper_01",
            "cla_s04_dia_scr_03",
        )
        cls.model = build_response_model(
            cls.lattice,
            cls.plan,
            ideal_kickoff_beam(384),
        )
        cls.beam = sample_beam(918, 384)

    def make_env(self, physics_weight):
        lattice = load_lattice(count_evaluations=False)
        plan = make_section_plan(
            lattice,
            "cla_s04_sim_aper_01",
            "cla_s04_dia_scr_03",
        )
        start = np.clip(self.model.center + 0.02, 0.0, 1.0)
        return SectionRLEnv(
            lattice,
            plan,
            self.beam,
            self.model,
            start,
            config=SectionRLConfig(
                horizon=2,
                max_knob_step=0.08,
                physics_weight=physics_weight,
                noise_fraction=0.0,
            ),
        )

    def test_physics_prior_and_environment_contract(self):
        env = self.make_env(0.7)
        observation = env.reset(seed=2)
        self.assertEqual(observation.shape, (len(self.plan.knobs) + 7,))
        prior = env.physics_action()
        self.assertEqual(prior.shape, (len(self.plan.knobs),))
        self.assertTrue(np.isfinite(prior).all())
        self.assertTrue(np.all(np.abs(prior) <= 1.0 + 1e-12))

        before = env._scaled_residual()
        predicted = before + self.model.response @ (env.max_knob_step * prior)
        self.assertLessEqual(np.linalg.norm(predicted), np.linalg.norm(before) + 1e-8)
        observation, reward, done, info = env.step(np.zeros(env.action_dim))
        self.assertFalse(done)
        self.assertTrue(np.isfinite(observation).all())
        self.assertTrue(np.isfinite(reward))
        self.assertTrue(np.all((info["u"] >= 0) & (info["u"] <= 1)))

    def test_ppo_train_and_frozen_transfer_smoke(self):
        env = self.make_env(0.0)
        agent = ActorCritic(env.observation_dim, env.action_dim, hidden=16)
        config = PPOConfig(
            iterations=1,
            episodes_per_iteration=1,
            horizon=2,
            max_knob_step=0.08,
            update_epochs=1,
            minibatch_size=2,
        )
        history = train_ppo(agent, env, config, seed=3)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["offline_measurements"], 3)

        transfer_env = self.make_env(0.0)
        controls, session = run_section_policy(agent, transfer_env, seed=4)
        self.assertEqual(len(session.records), 3)
        self.assertTrue(np.all((controls >= 0) & (controls <= 1)))


if __name__ == "__main__":
    unittest.main()
