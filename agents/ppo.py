import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from agents.api import LoadedAgent, AgentSpec
from runner.common_env import make_env


def _to_chw(obs: np.ndarray) -> np.ndarray:
    # expects obs HxWxC -> CxHxW
    if obs.ndim == 3 and obs.shape[-1] in (1, 3, 4):
        return np.transpose(obs, (2, 0, 1))
    return obs


class Policy:
    def __init__(self, model: PPO, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def predict(self, obs, env=None):
        # runner sends obs as HWC (84,84,3)
        obs_chw = _to_chw(obs)
        action, _ = self.model.predict(obs_chw, deterministic=self.deterministic)
        # discrete action -> int
        return int(action)


def train(cfg: dict, seed: int, episodes: int, timesteps: int) -> None:
    print("Training PPO agent")

    agent_cfg = cfg["agent"]

    # PPO hyperparameters
    lr = float(agent_cfg.get("learning_rate", 2.5e-4))
    gamma = float(agent_cfg.get("discount_factor", 0.99))
    n_steps = int(agent_cfg.get("n_steps", 2048))
    batch_size = int(agent_cfg.get("batch_size", 64))
    n_epochs = int(agent_cfg.get("n_epochs", 10))
    gae_lambda = float(agent_cfg.get("gae_lambda", 0.95))
    clip_range = float(agent_cfg.get("clip_range", 0.2))
    ent_coef = float(agent_cfg.get("ent_coef", 0.0))
    vf_coef = float(agent_cfg.get("vf_coef", 0.5))
    max_grad_norm = float(agent_cfg.get("max_grad_norm", 0.5))

    def _make_single_env():
        env = make_env(
            cfg,
            seed,
            render_mode=cfg["env"]["render_mode"],
            needs_pixels=agent_cfg["needs_pixels"],
            discrete_wrapper=agent_cfg["discrete_wrapper"],
            action_set_name=agent_cfg.get("action_set"),
            resize=int(agent_cfg.get("resize", 84)),
            frame_stack=int(agent_cfg.get("frame_stack", 1)),
        )
        return env

    venv = DummyVecEnv([_make_single_env])
    # for CNN: HWC -> CHW
    venv = VecTransposeImage(venv)

    model = PPO(
        policy="CnnPolicy",
        env=venv,
        learning_rate=lr,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        seed=seed,
        verbose=1,
    )

    model.learn(total_timesteps=int(timesteps))

    model_dir = "results/models/ppo_3"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "ppo.zip")
    model.save(model_path)
    print(f"PPO model saved to {model_path}")


def load(model_path: str, cfg: dict) -> LoadedAgent:
    spec = AgentSpec(
        needs_pixels=cfg["agent"]["needs_pixels"],
        discrete_wrapper=cfg["agent"]["discrete_wrapper"],
        action_set_name=cfg["agent"].get("action_set"),
        resize=int(cfg["agent"].get("resize", 84)),
        frame_stack=int(cfg["agent"].get("frame_stack", 1)),
    )

    model = PPO.load(model_path)

    policy = Policy(model, deterministic=True)

    return LoadedAgent(name="ppo", spec=spec, policy=policy)
