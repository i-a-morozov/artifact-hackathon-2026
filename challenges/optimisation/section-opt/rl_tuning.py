"""PPO helpers for S04 correction.

The agent observes only normalized controls and noisy downstream screen moments. Training
can use one fixed incoming beam or a pool of beams. Evaluation can freeze the policy and
change the beam without retraining. This module uses plain PyTorch because TorchRL is not a
project dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .beam_tuning import MeasurementSession, TARGET, checked_cube, loss


@dataclass(frozen=True)
class PPOConfig:
    iterations: int = 80
    episodes_per_iteration: int = 4
    horizon: int = 8
    max_knob_step: float = 0.12
    learning_rate: float = 3e-4
    gamma: float = 0.97
    clip_ratio: float = 0.2
    update_epochs: int = 6
    minibatch_size: int = 64
    value_weight: float = 0.5
    entropy_weight: float = 0.01
    max_grad_norm: float = 0.7
    reward_scale: float = 0.10
    action_penalty: float = 0.002
    terminal_weight: float = 0.25


class S04RLEnv:
    """Small continuous-control environment backed by counted screen measurements."""

    def __init__(self, machine, beam, *, horizon=8, max_knob_step=0.12,
                 noise_fraction=0.05, reward_scale=0.10, action_penalty=0.002,
                 terminal_weight=0.25):
        self.machine = machine
        self.beam = beam
        self.horizon = int(horizon)
        self.max_knob_step = float(max_knob_step)
        self.noise_fraction = float(noise_fraction)
        self.reward_scale = float(reward_scale)
        self.action_penalty = float(action_penalty)
        self.terminal_weight = float(terminal_weight)
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if not 0 < self.max_knob_step <= 1:
            raise ValueError("max_knob_step must be in (0, 1]")
        self.action_dim = len(machine.u0)
        self.observation_dim = self.action_dim + 6
        self.session = None

    def _observation(self):
        residual = (self.measured - TARGET) / 100e-6
        residual = np.clip(residual, -10.0, 10.0) / 5.0
        return np.r_[
            2.0 * self.u - 1.0,
            residual,
            np.clip(self.current_loss / 1.0, 0.0, 5.0) / 5.0,
            1.0 - self.step_index / self.horizon,
        ].astype(np.float32)

    def reset(self, seed=0):
        # The reset measurement is an online shot during deployment and an offline
        # simulator call during policy training.
        self.session = MeasurementSession(
            self.machine, self.beam, budget=self.horizon + 1,
            noise_fraction=self.noise_fraction, seed=int(seed),
        )
        self.u = np.asarray(self.machine.u0, dtype=float).copy()
        self.measured, _ = self.session.measure(self.u)
        self.current_loss = loss(self.measured)
        self.best_loss = self.current_loss
        self.best_u = self.u.copy()
        self.step_index = 0
        return self._observation()

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError(f"Expected a finite action vector of length {self.action_dim}")
        action = np.clip(action, -1.0, 1.0)
        previous = self.current_loss
        self.u = checked_cube(
            np.clip(self.u + self.max_knob_step * action, 0.0, 1.0),
            self.action_dim,
        )
        self.measured, _ = self.session.measure(self.u)
        self.current_loss = loss(self.measured)
        self.step_index += 1
        done = self.step_index >= self.horizon
        reward = (previous - self.current_loss) / self.reward_scale
        reward -= self.action_penalty * float(np.mean(action**2))
        if done:
            reward -= self.terminal_weight * self.current_loss / self.reward_scale
        if self.current_loss < self.best_loss:
            self.best_loss = self.current_loss
            self.best_u = self.u.copy()
        info = {
            "loss": self.current_loss,
            "best_loss": self.best_loss,
            "u": self.u.copy(),
            "measured": self.measured.copy(),
        }
        return self._observation(), float(reward), done, info


class ActorCritic(nn.Module):
    """Squashed-Gaussian actor and scalar critic used by clipped PPO."""

    def __init__(self, observation_dim, action_dim, hidden=64):
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.actor = nn.Sequential(
            nn.Linear(self.observation_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, self.action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.observation_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -0.6))
        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def distribution_and_value(self, observations):
        mean = self.actor(observations)
        std = self.log_std.clamp(-3.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std), self.critic(observations).squeeze(-1)

    @staticmethod
    def _squashed_log_prob(distribution, raw_action):
        action = torch.tanh(raw_action)
        correction = torch.log(torch.clamp(1.0 - action.square(), min=1e-6))
        return (distribution.log_prob(raw_action) - correction).sum(-1)

    @torch.no_grad()
    def sample_action(self, observation):
        observation = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        distribution, value = self.distribution_and_value(observation)
        raw_action = distribution.sample()
        action = torch.tanh(raw_action)
        log_prob = self._squashed_log_prob(distribution, raw_action)
        return (
            action.squeeze(0).cpu().numpy(),
            raw_action.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    def evaluate_actions(self, observations, raw_actions):
        distribution, values = self.distribution_and_value(observations)
        log_prob = self._squashed_log_prob(distribution, raw_actions)
        entropy = distribution.entropy().sum(-1)
        return log_prob, entropy, values

    @torch.no_grad()
    def deterministic_action(self, observation):
        observation = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        distribution, _ = self.distribution_and_value(observation)
        return torch.tanh(distribution.mean).squeeze(0).cpu().numpy()


def _discounted_returns(rewards, gamma):
    result = np.empty(len(rewards), dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        result[index] = running
    return result


def _training_environment_pool(env_or_envs, config):
    """Validate and normalize one environment or a finite environment collection."""
    required = ("observation_dim", "action_dim", "horizon", "reset", "step")

    def is_environment(candidate):
        return all(hasattr(candidate, attribute) for attribute in required)

    if is_environment(env_or_envs):
        environments = (env_or_envs,)
    else:
        try:
            environments = tuple(env_or_envs)
        except TypeError as error:
            raise TypeError("env_or_envs must be an S04RLEnv or an iterable of them") from error
    if not environments:
        raise ValueError("At least one training environment is required")
    if not all(is_environment(environment) for environment in environments):
        raise TypeError("Every training environment must implement the PPO environment contract")
    dimensions = {
        (environment.observation_dim, environment.action_dim) for environment in environments
    }
    if len(dimensions) != 1:
        raise ValueError("All training environments must have the same dimensions")
    if any(environment.horizon != config.horizon for environment in environments):
        raise ValueError("Every training environment horizon must match PPOConfig.horizon")
    return environments


def _balanced_environment_indices(count, episodes, rng):
    """Return a shuffled, approximately balanced environment schedule."""
    schedule = []
    while len(schedule) < episodes:
        schedule.extend(rng.permutation(count).tolist())
    return schedule[:episodes]


def train_ppo(model, env_or_envs, config=PPOConfig(), seed=0, *, progress=None):
    """Train PPO on one beam or a balanced pool of beams and return diagnostics."""
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(seed)
    environments = _training_environment_pool(env_or_envs, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    total_measurements = 0
    training_started = perf_counter()

    for iteration in range(config.iterations):
        rollout_started = perf_counter()
        observations, raw_actions, old_log_probs = [], [], []
        values, returns = [], []
        episode_best, episode_final, episode_rewards = [], [], []

        environment_indices = _balanced_environment_indices(
            len(environments), config.episodes_per_iteration, rng
        )
        for episode, environment_index in enumerate(environment_indices):
            env = environments[environment_index]
            observation = env.reset(seed=int(rng.integers(2**31)))
            rewards = []
            episode_observations, episode_raw, episode_logp, episode_values = [], [], [], []
            done = False
            while not done:
                action, raw_action, log_prob, value = model.sample_action(observation)
                next_observation, reward, done, info = env.step(action)
                episode_observations.append(observation)
                episode_raw.append(raw_action)
                episode_logp.append(log_prob)
                episode_values.append(value)
                rewards.append(reward)
                observation = next_observation

            observations.extend(episode_observations)
            raw_actions.extend(episode_raw)
            old_log_probs.extend(episode_logp)
            values.extend(episode_values)
            returns.extend(_discounted_returns(rewards, config.gamma))
            episode_best.append(env.best_loss)
            episode_final.append(env.current_loss)
            episode_rewards.append(sum(rewards))
            total_measurements += len(env.session.records)

        rollout_seconds = perf_counter() - rollout_started
        update_started = perf_counter()
        obs = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
        raw = torch.as_tensor(np.asarray(raw_actions), dtype=torch.float32)
        old_logp = torch.as_tensor(old_log_probs, dtype=torch.float32)
        target_return = torch.as_tensor(returns, dtype=torch.float32)
        advantage = target_return - torch.as_tensor(values, dtype=torch.float32)
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)

        indices = np.arange(len(obs))
        policy_losses, value_losses, entropies = [], [], []
        for _ in range(config.update_epochs):
            rng.shuffle(indices)
            for start in range(0, len(indices), config.minibatch_size):
                batch = torch.as_tensor(indices[start:start + config.minibatch_size])
                logp, entropy, prediction = model.evaluate_actions(obs[batch], raw[batch])
                ratio = torch.exp(logp - old_logp[batch])
                unclipped = ratio * advantage[batch]
                clipped = torch.clamp(
                    ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio
                ) * advantage[batch]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = (prediction - target_return[batch]).square().mean()
                entropy_mean = entropy.mean()
                objective = (
                    policy_loss
                    + config.value_weight * value_loss
                    - config.entropy_weight * entropy_mean
                )
                optimizer.zero_grad()
                objective.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                policy_losses.append(float(policy_loss.detach()))
                value_losses.append(float(value_loss.detach()))
                entropies.append(float(entropy_mean.detach()))

        history.append({
            "iteration": iteration + 1,
            "mean_best_loss": float(np.mean(episode_best)),
            "mean_final_loss": float(np.mean(episode_final)),
            "mean_episode_reward": float(np.mean(episode_rewards)),
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "offline_measurements": total_measurements,
            "training_environments": len(environments),
            "rollout_seconds": rollout_seconds,
            "update_seconds": perf_counter() - update_started,
            "elapsed_seconds": perf_counter() - training_started,
        })
        if progress is not None:
            progress(dict(history[-1]))
    return history


def run_policy(model, machine, beam, *, horizon=8, max_knob_step=0.12,
               noise_fraction=0.05, seed=0):
    """Deploy a frozen deterministic policy and return its best measured setting."""
    env = S04RLEnv(
        machine, beam, horizon=horizon, max_knob_step=max_knob_step,
        noise_fraction=noise_fraction,
    )
    observation = env.reset(seed=seed)
    done = False
    while not done:
        action = model.deterministic_action(observation)
        observation, _, done, _ = env.step(action)
    return env.best_u.copy(), env.session
