from __future__ import annotations

import os
import time

from utils.config import load_config
from utils.seeding import set_global_seeds
from runner.common_env import make_env


def main():
    cfg = load_config(os.path.join("config", "env_base.yaml"))
    set_global_seeds(0)
    env = make_env(
        cfg=cfg,
        seed=0,
        render_mode="human",
        needs_pixels=False,
        discrete_wrapper=True,
        action_set_name="dqn5_simple",
        resize=84,
        frame_stack=4,
    )

    obs, _ = env.reset(seed=0)
    done = False
    truncated = False
    while not (done or truncated):
        a = env.action_space.sample()
        obs, _, done, truncated, _ = env.step(a)
        time.sleep(0.03)

    env.close()


if __name__ == "__main__":
    main()
