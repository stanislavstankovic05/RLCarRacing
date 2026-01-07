from __future__ import annotations
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, FrameStackObservation

from env.reward_shaping import RewardParams
from env.carracing_wrapper import CarRacing, CarRacingMetrics, ActionWrapper
from env.actions import action_set

def make_env(cfg, seed, render_mode, *, needs_pixels, discrete_wrapper, action_set_name, resize, frame_stack):
    env = gym.make(cfg["env"]["id"], render_mode=render_mode)
    env.reset(seed=seed)

    reward_cfg = dict(cfg.get("reward", {}))
    use_native = bool(reward_cfg.pop("use_native", False))

    if use_native:
        env = CarRacingMetrics(env)
    else:
        shaped_reward_cfg = reward_cfg.get("shaped", {})
        rp = RewardParams(**shaped_reward_cfg)
        term = cfg["termination"]
        env = CarRacing(
            env,
            reward_params=rp,
            offtrack_frames=term["off_course_for"],
            no_progress_steps=term["stable_for"],
        )

    if discrete_wrapper:
        assert action_set_name is not None, "action_set_name required for discrete env"
        env = ActionWrapper(env, action_set(action_set_name))

    if needs_pixels:
        env = ResizeObservation(env, (resize, resize))
        env = GrayscaleObservation(env, keep_dim=True)
        if frame_stack > 1:
            env = FrameStackObservation(env, frame_stack)

    return env
