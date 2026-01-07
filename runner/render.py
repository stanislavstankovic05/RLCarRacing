from __future__ import annotations
import os
import time
import numpy as np

from utils.video import save_video
from runner.common_env import make_env
from agents.plugin_loader import load_agent

def render_agent(cfg, agent_name, model_path, seed = 123, episodes = 3, render = "human", video_path = None,):
    render_mode = "human" if render == "human" else ("rgb_array" if render == "video" else None)
    loaded = load_agent(agent_name, model_path, cfg)

    if render == "video" and video_path is None:
        os.makedirs("results/videos", exist_ok=True)
        video_path = os.path.join("results", "videos", f"{agent_name}_render_{int(time.time())}.mp4")

    env = make_env(
        cfg=cfg,
        seed=seed,
        render_mode=render_mode,
        needs_pixels=loaded.spec.needs_pixels,
        discrete_wrapper=loaded.spec.discrete_wrapper,
        action_set_name=loaded.spec.action_set_name,
        resize=loaded.spec.resize,
        frame_stack=loaded.spec.frame_stack,
    )

    frames = []
    rewards = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 1000 + ep)
        done = False
        truncated = False

        while not (done or truncated):
            action = loaded.policy.predict(obs, env)

            obs, _, done, truncated, _ = env.step(action)

            if render == "video":
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

        # Find the CarRacing wrapper and get metrics
        cr_wrapper = env
        while not hasattr(cr_wrapper, "metrics") and hasattr(cr_wrapper, "env"):
            cr_wrapper = cr_wrapper.env
        
        m = getattr(cr_wrapper, "metrics", None)
        rewards.append(m.reward_sum)
        print(f"[{agent_name.upper()} RENDER] ep={ep+1} reward={m.reward_sum:.1f} steps={m.steps} finished={m.finished} offtrack={m.offtrack_ratio:.2f}")

    if render == "video" and video_path:
        save_video(frames, video_path, fps=30)
        print(f"Saved video: {video_path}")

    env.close()
    print(f"Mean reward: {float(np.mean(rewards)):.2f} ± {float(np.std(rewards)):.2f}")
