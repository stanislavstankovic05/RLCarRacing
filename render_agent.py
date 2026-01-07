from __future__ import annotations
import os
import argparse

from utils.seeding import set_global_seeds
from utils.config import load_config
from runner.render import render_agent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="agent's name that is in agents/<agent>.py and config/<agent>.yaml",)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--render", choices=["none", "human", "video"], default="human")
    ap.add_argument("--video_path", default=None)
    args = ap.parse_args()

    set_global_seeds(args.seed)

    if args.config is None:
        args.config = os.path.join("config", f"{args.agent}.yaml")
    cfg = load_config(args.config)

    render_agent(
        cfg=cfg,
        agent_name=args.agent,
        model_path=args.model_path,
        seed=args.seed,
        episodes=args.episodes,
        render=args.render,
        video_path=args.video_path,
    )

if __name__ == "__main__":
    main()
