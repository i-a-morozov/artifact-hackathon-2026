"""The learned half: a policy that proposes the focus move, trained in the twin.

This is the Dyna layer. The twin is a complete, differentiable, batched simulator of
the machine, so a policy can be trained against it for nothing — randomise the latent
from the same distributions `SimEnv` deals from, roll out, learn. No beam time is
spent on training at all; the environment is only ever used to measure the result.

The policy is cloned from the planner rather than grown by policy gradient. The
planner already solves these episodes — CEM plus gradient descent against a model
that matches the machine to a micron — so it is a free and very good expert, and
what the policy is for is *speed*: the planner spends four to seven seconds deciding
each shot, and most of that is re-deriving a move that looks much like the move it
made on the last similar episode. A network that proposes the same move in a
millisecond turns that into a warm start the planner only has to refine, or skips the
planner entirely on the shots where it is confident.

    from policy import FocusPolicy
    policy = FocusPolicy.load("models/focus.pt")
    agent = MBRLAgent(config, twin, policy=policy)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from features import MAX_SLOTS, MAX_STEP, OBSERVATION_SIZE


@dataclass
class PolicyConfig:
    """Shape and training settings for the focus policy."""

    hidden: tuple[int, ...] = (256, 256)
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 60
    weight_decay: float = 1e-5
    # Held back from training to report an honest validation loss.
    validation_fraction: float = 0.1
    seed: int = 0
    # Bookkeeping, filled in by `train.py` so a checkpoint says where it came from.
    notes: dict = field(default_factory=dict)


class FocusPolicy(nn.Module):
    """Observation -> a step for each ranked control slot, in calibrated units."""

    def __init__(self, config: PolicyConfig | None = None):
        super().__init__()
        self.config = config or PolicyConfig()
        widths = [OBSERVATION_SIZE, *self.config.hidden]
        layers: list[nn.Module] = []
        for a, b in zip(widths, widths[1:]):
            layers += [nn.Linear(a, b), nn.SiLU()]
        layers.append(nn.Linear(widths[-1], MAX_SLOTS))
        self.net = nn.Sequential(*layers)
        # Observations are standardised by statistics measured on the training set;
        # until `fit_normalisation` runs these are the identity.
        self.register_buffer("mean", torch.zeros(OBSERVATION_SIZE))
        self.register_buffer("scale", torch.ones(OBSERVATION_SIZE))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalised = (observation - self.mean) / self.scale
        # Bounded by construction: a policy cannot ask for a step the planner was
        # never allowed to take, however far out of distribution it is pushed.
        return MAX_STEP * torch.tanh(self.net(normalised))

    @torch.no_grad()
    def propose(self, observation: np.ndarray) -> np.ndarray:
        """One move, in calibrated steps, for a single observation."""
        self.eval()
        x = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        return self.forward(x).numpy().reshape(-1)

    def fit_normalisation(self, observations: np.ndarray) -> None:
        """Standardise inputs on the training set, so the net sees unit-ish numbers."""
        x = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
        self.mean.copy_(x.mean(dim=0))
        self.scale.copy_(x.std(dim=0).clamp(min=1e-3))

    # ---------- persistence ----------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.state_dict(), "config": self.config.__dict__}, path
        )

    @classmethod
    def load(cls, path: str | Path) -> "FocusPolicy":
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        policy = cls(PolicyConfig(**blob["config"]))
        policy.load_state_dict(blob["state_dict"])
        policy.eval()
        return policy


def clone(
    observations: np.ndarray,
    actions: np.ndarray,
    config: PolicyConfig | None = None,
    report: bool = True,
) -> tuple[FocusPolicy, dict]:
    """Behaviour cloning: fit the policy to the planner's moves.

    Returns the policy and the training history. The validation split is held out by
    episode-blind random sampling, which is fine here because the pairs within an
    episode are not near-duplicates — each follows a different machine state.
    """
    config = config or PolicyConfig()
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    x = np.asarray(observations, dtype=np.float32)
    y = np.asarray(actions, dtype=np.float32)
    order = rng.permutation(len(x))
    x, y = x[order], y[order]
    cut = max(1, int(len(x) * (1 - config.validation_fraction)))
    train_x, train_y = torch.from_numpy(x[:cut]), torch.from_numpy(y[:cut])
    valid_x, valid_y = torch.from_numpy(x[cut:]), torch.from_numpy(y[cut:])

    policy = FocusPolicy(config)
    policy.fit_normalisation(x[:cut])
    optimiser = torch.optim.AdamW(
        policy.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history = {"train": [], "valid": []}
    for epoch in range(config.epochs):
        policy.train()
        permutation = torch.randperm(len(train_x))
        total = 0.0
        for start in range(0, len(train_x), config.batch_size):
            batch = permutation[start : start + config.batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(policy(train_x[batch]), train_y[batch])
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(batch)
        history["train"].append(total / max(len(train_x), 1))
        policy.eval()
        with torch.no_grad():
            valid = (
                float(nn.functional.mse_loss(policy(valid_x), valid_y))
                if len(valid_x)
                else float("nan")
            )
        history["valid"].append(valid)
        if report and (epoch % 10 == 0 or epoch == config.epochs - 1):
            print(
                f"  epoch {epoch:>3}  train {history['train'][-1]:.4f}  "
                f"valid {valid:.4f}",
                flush=True,
            )
    return policy, history
