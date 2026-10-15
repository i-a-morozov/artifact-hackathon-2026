"""PPO environment for named screen-to-screen correction.

The environment exposes normalized magnet settings and one downstream-screen reading.
An optional ideal-response controller supplies a physics action prior; PPO learns the
remaining correction from measurements.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from .beamline_tuning import apply_knobs, track_interval
from .section_measurement import SectionSession, section_readout
from .section_tracking import MomentTracker


@dataclass(frozen=True)
class SectionRLConfig:
    horizon: int = 8
    max_knob_step: float = 0.12
    physics_weight: float = 0.0
    noise_fraction: float = 0.0
    reward_scale: float = 0.25
    action_penalty: float = 0.002
    terminal_weight: float = 0.25
    ridge: float = 1e-4
    fast_tracking: bool = True


class SectionRLEnv:
    """Continuous-control environment using one final-screen image per step.

    ``physics_weight=0`` is ordinary PPO. With a positive weight, the applied action is
    the policy action plus a bounded response-matrix action. The latter follows the
    local ideal model ``r_next = r + J delta_u``. The policy therefore learns a residual
    for input-beam changes, nonlinear response, noise, and machine/model mismatch.
    """

    def __init__(self, lattice, plan, input_beam, response_model, start, *,
                 config=SectionRLConfig()):
        self.lattice = lattice
        self.plan = plan
        self.input_beam = input_beam
        self.model = response_model
        self.start = np.asarray(start, dtype=float).copy()
        self.config = config
        self.horizon = int(config.horizon)
        self.max_knob_step = float(config.max_knob_step)
        self.physics_weight = float(config.physics_weight)
        self.noise_fraction = float(config.noise_fraction)
        self.reward_scale = float(config.reward_scale)
        self.action_penalty = float(config.action_penalty)
        self.terminal_weight = float(config.terminal_weight)
        self.ridge = float(config.ridge)
        self.action_dim = len(plan.knobs)
        self.observation_dim = self.action_dim + 7
        self.session = None
        self.tracker = MomentTracker(lattice, plan) if config.fast_tracking else None
        if self.start.shape != (self.action_dim,):
            raise ValueError("start must contain one normalized value per section knob")
        if np.any(~np.isfinite(self.start)) or np.any((self.start < 0) | (self.start > 1)):
            raise ValueError("start must be finite and inside [0, 1]")
        if response_model.response.shape != (4, self.action_dim):
            raise ValueError("response model does not match the section controls")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if not 0 < self.max_knob_step <= 1:
            raise ValueError("max_knob_step must be in (0, 1]")
        if self.reward_scale <= 0 or self.ridge < 0:
            raise ValueError("reward_scale must be positive and ridge nonnegative")
        if not 0 <= self.physics_weight <= 1:
            raise ValueError("physics_weight must be in [0, 1]")

        self._regularized_response = np.vstack([
            np.asarray(response_model.response, dtype=float),
            np.sqrt(self.ridge) * np.eye(self.action_dim),
        ])

        self._response_inverse = np.linalg.pinv(self._regularized_response)[:, :4]


    def _scaled_residual(self):
        return (self.measured - self.model.target) / self.model.scale

    def physics_action(self):
        """Bounded action from the ideal local response, expressed in policy units."""
        residual = self._scaled_residual()
        if not np.isfinite(residual).all():
            return np.zeros(self.action_dim)
        matrix = np.asarray(self.model.response, dtype=float)
        n_controls = matrix.shape[1]
        regularized = self._regularized_response
        right_hand_side = np.r_[-residual, np.zeros(n_controls)]
        lower = np.maximum(-self.max_knob_step, -self.u)
        upper = np.minimum(self.max_knob_step, 1.0 - self.u)
        # The unconstrained ridge solution is reusable while J remains fixed.
        delta = self._response_inverse @ (-residual)
        if np.any(delta < lower) or np.any(delta > upper):
            # Dense, small control problems suit BVLS better than iterative TRF.
            delta = lsq_linear(
                regularized, right_hand_side, bounds=(lower, upper),
                method="bvls", tol=1e-10,
            ).x
        return np.clip(delta / self.max_knob_step, -1.0, 1.0)

    def _observation(self):
        visible = bool(np.isfinite(self.measured).all())
        residual = np.clip(np.nan_to_num(self._scaled_residual(), nan=10.0,
                                        posinf=10.0, neginf=-10.0), -10.0, 10.0) / 5.0
        return np.r_[
            2.0 * self.u - 1.0,
            residual,
            np.clip(self.current_loss, 0.0, 5.0) / 5.0,
            1.0 - self.step_index / self.horizon,
            float(visible),
        ].astype(np.float32)

    def reset(self, seed=0):
        self.session = SectionSession(
            self.lattice,
            self.plan,
            self.input_beam,
            self.model.target,
            budget=self.horizon + 1,
            scale=self.model.scale,
            noise_fraction=self.noise_fraction,
            seed=int(seed),
            tracker=self.tracker,
        )
        self.u = self.start.copy()
        self.current_loss, _ = self.session.measure(self.u)
        self.measured = self.session.records[-1]["observed"].copy()
        self.best_loss = float(self.current_loss)
        self.best_u = self.u.copy()
        self.step_index = 0
        return self._observation()

    def step(self, action):
        if self.session is None or self.step_index >= self.horizon:
            raise RuntimeError("Reset before stepping a new episode")
        policy_action = np.asarray(action, dtype=float)
        if policy_action.shape != (self.action_dim,) or not np.isfinite(policy_action).all():
            raise ValueError(f"Expected a finite action vector of length {self.action_dim}")
        policy_action = np.clip(policy_action, -1.0, 1.0)
        prior_action = (self.physics_action() if self.physics_weight > 0
                        else np.zeros(self.action_dim))
        applied_action = np.clip(
            policy_action + self.physics_weight * prior_action,
            -1.0,
            1.0,
        )
        previous = self.current_loss
        self.u = np.clip(
            self.u + self.max_knob_step * applied_action,
            0.0,
            1.0,
        )
        self.current_loss, _ = self.session.measure(self.u)
        self.measured = self.session.records[-1]["observed"].copy()
        self.step_index += 1
        done = self.step_index >= self.horizon
        # Bound reward magnitude for failed images while keeping finite observations.
        reward = (min(previous, 10.0) - min(self.current_loss, 10.0)) / self.reward_scale
        reward -= self.action_penalty * float(np.mean(applied_action**2))
        if done:
            reward -= self.terminal_weight * min(self.current_loss, 10.0) / self.reward_scale
        if self.current_loss < self.best_loss:
            self.best_loss = float(self.current_loss)
            self.best_u = self.u.copy()
        info = {
            "loss": float(self.current_loss),
            "best_loss": float(self.best_loss),
            "u": self.u.copy(),
            "measured": self.measured.copy(),
            "policy_action": policy_action.copy(),
            "physics_action": prior_action.copy(),
            "applied_action": applied_action.copy(),
        }
        return self._observation(), float(reward), done, info


def run_section_policy(model, environment, *, seed=0):
    """Run a frozen deterministic policy and return its best measured setting."""
    observation = environment.reset(seed=seed)
    done = False
    while not done:
        action = model.deterministic_action(observation)
        observation, _, done, _ = environment.step(action)
    return environment.best_u.copy(), environment.session


def clean_section_result(environment, controls):
    """Evaluate selected controls once as simulation truth, outside the online ledger."""
    apply_knobs(environment.lattice, environment.plan.knobs, controls)
    return section_readout(
        track_interval(
            environment.lattice,
            environment.plan,
            environment.input_beam,
        )
    )[0]
