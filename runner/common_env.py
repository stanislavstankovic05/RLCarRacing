from __future__ import annotations
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation, FrameStack

from env.reward_shaping import RewardParams
from env.carracing_wrapper import CarRacing, CarRacingMetrics, ActionWrapper
from env.actions import action_set

def make_env(cfg, seed, render_mode, *, needs_pixels, discrete_wrapper, action_set_name, resize, frame_stack):
    env = gym.make(cfg["env_id"], render_mode=render_mode)
    env.reset(seed=seed)

    reward_cfg = dict(cfg.get("reward", {}))
    use_native = bool(reward_cfg.pop("use_native", False))

    if use_native:
        env = CarRacingMetrics(env)
    else:
        rp = RewardParams(**reward_cfg)
        term = cfg["termination"]
        env = CarRacing(
            env,
            rp,
            offtrack_frames=term["offtrack_frames"],
            no_progress_steps=term["no_progress_steps"],
        )

    if discrete_wrapper:
        assert action_set_name is not None, "action_set_name required for discrete env"
        env = ActionWrapper(env, action_set(action_set_name))

    if needs_pixels:
        env = ResizeObservation(env, (resize, resize))
        env = GrayScaleObservation(env, keep_dim=True)
        env = FrameStack(env, frame_stack)

    return env
