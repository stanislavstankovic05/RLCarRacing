from __future__ import annotations

import os
import argparse

from utils.seeding import set_global_seeds
from utils.config import load_config
from agents.plugin_loader import train_agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--timesteps", type=int, default=500000)
    args = ap.parse_args()

    set_global_seeds(args.seed)

    if args.config is None:
        args.config = os.path.join("config", f"{args.agent}.yaml")
    cfg = load_config(args.config)

    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/videos", exist_ok=True)

    train_agent(args.agent, cfg, args.seed, episodes=args.episodes, timesteps=args.timesteps)


if __name__ == "__main__":
    main()
