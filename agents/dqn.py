# agents/dqn.py
# A DQN agent module that works with your scaffold's plugin_loader.py:
#   - must expose train(cfg=..., seed=..., episodes=..., timesteps=...)
#   - must expose load(model_path=..., cfg=...)
#
# This version:
#   - reads hyperparams/env from YAML robustly (supports many common YAML layouts)
#   - wraps CarRacing-v2 continuous actions into 5 discrete actions (int -> [steer, gas, brake])
#   - avoids Box2D float32 issues by returning Python floats (not np.float32)
#
# If your YAML uses unusual keys, this script still tries many common paths automatically.

import os
import random
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
import torchvision.transforms.functional as F
import gymnasium as gym
from tqdm import trange

import warnings
warnings.filterwarnings("ignore")

from utils.seeding import set_global_seeds
from agents.api import LoadedAgent


# =============================================================================
# Core DQN implementation (yours, unchanged except tiny safety tweaks)
# =============================================================================

class DQNAgent:
    """Deep Q-Network with replay memory and target network."""

    def __init__(
        self,
        env: gym.Env,
        gamma: float,
        epsilon_init: float,
        epsilon_min: float,
        epsilon_decay: float,
        lr: float = 1e-4,
        C: int = 1000,
        batch_size: int = 64,
        memory_size: int = 100000,
        n_actions: int = 5,
        n_frames: int = 4,
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.epsilon_init = epsilon_init
        self.epsilon_min = epsilon_min
        self.C = C
        self.batch_size = batch_size
        self.epsilon_decay = epsilon_decay

        self.n_frames = n_frames
        self.n_actions = n_actions

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.memory = ReplayMemory(memory_size)

        self.model = RacingNet(self.n_frames, self.n_actions).to(self.device)
        self.target = RacingNet(self.n_frames, self.n_actions).to(self.device)
        self.target.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def act(self, obs: torch.Tensor, epsilon: float = 0.0) -> int:
        if torch.rand(1) < epsilon:
            action = torch.randint(high=self.n_actions, size=(1,))
        else:
            with torch.no_grad():
                logits = self.model(obs.unsqueeze(0).to(self.device))
                action = logits.view(-1).argmax(0)
        return int(action.item())

    def play(self, n_episodes: int) -> dict:
        print(f"### Device: {self.device} ###")
        results = {"episode": [], "score": []}
        frames = deque(maxlen=self.n_frames)

        pbar = trange(n_episodes)
        for episode in pbar:
            frame, _ = self.env.reset()
            frames.clear()
            for _ in range(self.n_frames):
                frames.append(self.transfrom_frame(frame))

            score = 0.0
            while True:
                obs = torch.vstack(list(frames)).squeeze(1)  # (4,84,84)
                action = self.act(obs, epsilon=0.0)

                frame, reward, terminated, truncated, _ = self.env.step(action)
                frames.append(self.transfrom_frame(frame))
                score += float(reward)

                if terminated or truncated:
                    break

            results["episode"].append(episode + 1)
            results["score"].append(score)
            pbar.set_description(f"Score={score:.0f}")

        return results

    def train(
        self,
        n_episodes: int,
        save_every: int = 100,
        save_dir: str = "results/models",
        run_name: str = "dqn",
    ) -> dict:
        print(f"### Device: {self.device} ###")
        os.makedirs(save_dir, exist_ok=True)

        results = {"episode": [], "score": []}
        frames = deque(maxlen=self.n_frames)

        epsilons = self.schedule_decay(
            n_episodes, self.epsilon_init, self.epsilon_min, self.epsilon_decay
        )
        steps = 0

        pbar = trange(n_episodes)
        for episode in pbar:
            frame, _ = self.env.reset()
            frames.clear()
            for _ in range(self.n_frames):
                frames.append(self.transfrom_frame(frame))

            score = 0.0

            while True:
                obs = torch.vstack(list(frames)).squeeze(1)  # (4,84,84)
                action = self.act(obs, float(epsilons[episode]))

                frame, reward, terminated, truncated, _ = self.env.step(action)

                frames.append(self.transfrom_frame(frame))
                obs_ = torch.vstack(list(frames)).squeeze(1)

                self.memory.remember((obs, action, float(reward), obs_, bool(terminated)))

                if len(self.memory) >= self.batch_size:
                    self.optimize()

                if steps % self.C == 0:
                    self.target.load_state_dict(self.model.state_dict())

                steps += 1
                score += float(reward)

                if terminated or truncated:
                    break

            results["episode"].append(episode + 1)
            results["score"].append(score)
            pbar.set_description(f"Score={score:.0f} eps={epsilons[episode]:.3f}")

            if save_every > 0 and (episode + 1) % save_every == 0:
                self.save_model(save_dir, f"{run_name}_ep{episode+1}")

        self.save_model(save_dir, f"{run_name}_final")
        return results

    def optimize(self) -> None:
        obs, action, reward, obs_, terminated = self.memory.sample(self.batch_size)

        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        obs_ = obs_.to(self.device)
        terminated = terminated.to(self.device)

        qs = self.model(obs).gather(1, action)

        with torch.no_grad():
            max_qs_ = self.target(obs_).max(1, keepdim=True)[0]
            td_target = reward + self.gamma * max_qs_ * (1 - terminated)

        loss = self.criterion(qs, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def transfrom_frame(self, frame: np.ndarray) -> torch.Tensor:
        # frame: expected (H,W,3) uint8
        image_tensor = F.to_tensor(frame)                # (3,H,W)
        image_tensor = F.rgb_to_grayscale(image_tensor)  # (1,H,W)
        image_tensor = F.resized_crop(image_tensor, 0, 0, 84, 96, [84, 84])  # (1,84,84)
        return image_tensor

    def schedule_decay(self, n_episodes: int, init_value: float, min_value: float, decay_ratio: float) -> np.ndarray:
        steps = int(n_episodes * decay_ratio)
        steps = max(1, min(steps, n_episodes))
        epsilons = np.concatenate([
            np.geomspace(start=init_value, stop=min_value, num=steps),
            np.full(n_episodes - steps, min_value)
        ])
        return epsilons

    def save_model(self, dir: str, name: str) -> None:
        file = os.path.join(os.getcwd(), dir, name + ".pt")
        torch.save(self.model.state_dict(), file)

    def load_model(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.model.state_dict())


class RacingNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=16, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.experiences = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.experiences)

    def remember(self, experience) -> None:
        self.experiences.append(experience)

    def sample(self, batch_size: int):
        samples = random.choices(list(self.experiences), k=batch_size)
        obs, action, reward, obs_, terminated = zip(*samples)

        return (
            torch.stack(obs),  # (B,4,84,84)
            torch.from_numpy(np.stack(action)).to(torch.int64).view(batch_size, -1),
            torch.tensor(reward, dtype=torch.float32).view(batch_size, -1),
            torch.stack(obs_),
            torch.tensor(terminated, dtype=torch.int64).view(batch_size, -1)
        )


# =============================================================================
# Action wrapper: Discrete (int) -> CarRacing continuous action [steer, gas, brake]
# =============================================================================

class DiscreteCarRacingAction(gym.ActionWrapper):
    """
    Maps discrete actions {0..4} to CarRacing continuous actions [steer, gas, brake].

    IMPORTANT: returns Python floats (lists), not np.float32, to avoid Box2D errors like:
      TypeError: ... argument 2 of type 'float32'
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(5)

    def action(self, act: int):
        # steer, gas, brake
        if act == 0:   # left
            return [-1.0, 0.0, 0.0]
        if act == 1:   # right
            return [1.0, 0.0, 0.0]
        if act == 2:   # gas
            return [0.0, 1.0, 0.0]
        if act == 3:   # brake
            return [0.0, 0.0, 0.8]
        # 4: do nothing
        return [0.0, 0.0, 0.0]


# =============================================================================
# Config helpers: "use the yaml file" even if keys are nested / OmegaConf / etc.
# =============================================================================

def _to_plain(obj: Any) -> Any:
    """Convert OmegaConf-like / Namespace-like configs into plain Python types when possible."""
    # OmegaConf has .items only when converted; but we can't import omegaconf safely.
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    # SimpleNamespace / argparse Namespace / dataclass-ish
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes)):
        try:
            d = vars(obj)
            return {k: _to_plain(v) for k, v in d.items()}
        except Exception:
            pass
    return obj

def _deep_get(cfg: Any, path: Union[str, List[str]], default: Any = None) -> Any:
    """Get cfg['a']['b'] or cfg.a.b with path 'a.b'."""
    if isinstance(path, str):
        keys = path.split(".")
    else:
        keys = path

    cur = cfg
    for k in keys:
        if cur is None:
            return default
        # dict-like
        if isinstance(cur, dict):
            if k in cur:
                cur = cur[k]
            else:
                return default
        else:
            # attribute-like
            if hasattr(cur, k):
                cur = getattr(cur, k)
            else:
                return default
    return cur

def _first_present(cfg: Any, paths: List[str], default: Any = None) -> Any:
    for p in paths:
        v = _deep_get(cfg, p, default=None)
        if v is not None:
            return v
    return default

def _as_int(x: Any, default: int) -> int:
    if x is None:
        return default
    try:
        return int(x)
    except Exception:
        return default

def _as_float(x: Any, default: float) -> float:
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default

def _as_str(x: Any, default: str) -> str:
    if x is None:
        return default
    try:
        return str(x)
    except Exception:
        return default


def _make_env(cfg: Any) -> gym.Env:
    """
    Create env from cfg using many common YAML layouts.

    Tries these for env id (first hit wins):
      env_id
      env.id / env.name
      environment.id / environment.name / environment
      task.env_id / task.env / task.name
      gym_id / gym_env / gym.env_id

    Also tries kwargs in:
      env_kwargs
      env.kwargs / env.args
      environment.kwargs / environment.args
    """
    env_id = _first_present(cfg, [
        "env_id",
        "env.id", "env.name",
        "environment.id", "environment.name", "environment",
        "task.env_id", "task.env", "task.name",
        "gym_id", "gym_env", "gym.env_id",
    ], default=None)

    if env_id is None:
        # fall back: sometimes cfg has a single key like cfg["env"] == "CarRacing-v2"
        env_id = _deep_get(cfg, "env", None)

    env_id = _as_str(env_id, "")

    env_kwargs = _first_present(cfg, [
        "env_kwargs",
        "env.kwargs", "env.args",
        "environment.kwargs", "environment.args",
    ], default={})

    env_kwargs = _to_plain(env_kwargs)
    if env_kwargs is None:
        env_kwargs = {}
    if not isinstance(env_kwargs, dict):
        env_kwargs = {}

    if not env_id:
        raise KeyError(
            "Could not determine environment id from config. "
            "Make sure your YAML contains something like env_id: ... or env: {id: ...}."
        )

    env = gym.make(env_id, **env_kwargs)

    # Convert continuous CarRacing controls to discrete actions for DQN
    if "CarRacing" in env_id:
        env = DiscreteCarRacingAction(env)

    return env


def _read_hparams(cfg: Any) -> Dict[str, Any]:
    """
    Read hyperparameters from cfg using many common YAML layouts.
    We try hard to use YOUR YAML; defaults only apply if key is missing.
    """
    # common blocks: agent:, dqn:, algo:, hyperparams:
    blocks = [
        "agent", "dqn", "algo", "hyperparams", "hparams", "params", "config",
        "training", "train", "model"
    ]

    def gp(key: str, default: Any):
        # try top-level first
        v = _deep_get(cfg, key, None)
        if v is not None:
            return v
        # try inside common blocks
        for b in blocks:
            v = _deep_get(cfg, f"{b}.{key}", None)
            if v is not None:
                return v
        return default

    # DQN hyperparams
    hp = {
        "gamma": _as_float(gp("gamma", 0.99), 0.99),
        "epsilon_init": _as_float(gp("epsilon_init", gp("eps_start", gp("epsilon_start", 1.0))), 1.0),
        "epsilon_min": _as_float(gp("epsilon_min", gp("eps_end", gp("epsilon_end", 0.05))), 0.05),
        "epsilon_decay": _as_float(gp("epsilon_decay", gp("eps_decay", 0.5)), 0.5),
        "lr": _as_float(gp("lr", gp("learning_rate", 1e-4)), 1e-4),
        "C": _as_int(gp("C", gp("target_update", gp("target_update_steps", 1000))), 1000),
        "batch_size": _as_int(gp("batch_size", 64), 64),
        "memory_size": _as_int(gp("memory_size", gp("replay_size", 100000)), 100000),
        "n_actions": _as_int(gp("n_actions", gp("num_actions", 5)), 5),
        "n_frames": _as_int(gp("n_frames", gp("frame_stack", 4)), 4),

        # saving/logging (optional)
        "save_every": _as_int(gp("save_every", 100), 100),
        "save_dir": _as_str(gp("save_dir", "results/models"), "results/models"),
        "run_name": _as_str(gp("run_name", gp("experiment_name", "dqn")), "dqn"),
    }
    return hp


# =============================================================================
# REQUIRED MODULE-LEVEL ENTRYPOINTS for agents/plugin_loader.py
# =============================================================================

def train(*, cfg, seed, episodes, timesteps):
    """
    Called by plugin_loader.train_agent(...)
    Must exist at module scope with this signature.

    NOTE: This implementation uses episodes; timesteps is accepted but not used,
    since your DQNAgent.train() is episode-based.
    """
    set_global_seeds(seed)

    env = _make_env(cfg)
    hp = _read_hparams(cfg)

    agent = DQNAgent(
        env=env,
        gamma=hp["gamma"],
        epsilon_init=hp["epsilon_init"],
        epsilon_min=hp["epsilon_min"],
        epsilon_decay=hp["epsilon_decay"],
        lr=hp["lr"],
        C=hp["C"],
        batch_size=hp["batch_size"],
        memory_size=hp["memory_size"],
        n_actions=hp["n_actions"],
        n_frames=hp["n_frames"],
    )

    results = agent.train(
        n_episodes=int(episodes),
        save_every=hp["save_every"],
        save_dir=hp["save_dir"],
        run_name=hp["run_name"],
    )

    env.close()
    return results


def load(*, model_path, cfg):
    """
    Called by plugin_loader.load_agent(...)
    Returns a LoadedAgent(env=..., act=...) for evaluation.
    """
    env = _make_env(cfg)
    hp = _read_hparams(cfg)

    agent = DQNAgent(
        env=env,
        gamma=hp["gamma"],
        epsilon_init=0.0,
        epsilon_min=0.0,
        epsilon_decay=0.0,
        lr=hp["lr"],
        C=hp["C"],
        batch_size=hp["batch_size"],
        memory_size=hp["memory_size"],
        n_actions=hp["n_actions"],
        n_frames=hp["n_frames"],
    )

    state = torch.load(model_path, map_location=agent.device)
    agent.model.load_state_dict(state)
    agent.model.eval()

    frames = deque(maxlen=agent.n_frames)

    def act_fn(obs: np.ndarray) -> int:
        # Maintain a frame stack during evaluation (same as training)
        frame_t = agent.transfrom_frame(obs)
        if len(frames) == 0:
            for _ in range(agent.n_frames):
                frames.append(frame_t)
        else:
            frames.append(frame_t)

        stacked = torch.vstack(list(frames)).squeeze(1)  # (4,84,84)
        with torch.no_grad():
            q = agent.model(stacked.unsqueeze(0).to(agent.device))
            return int(q.argmax(dim=1).item())

    return LoadedAgent(env=env, act=act_fn)
