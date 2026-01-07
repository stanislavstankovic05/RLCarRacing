import importlib
from types import ModuleType

from agents.api import LoadedAgent


def import_agent_module(agent_name):
    return importlib.import_module(f"agents.{agent_name}")


def load_agent(agent_name, model_path, cfg):
    mod = import_agent_module(agent_name)
    return mod.load(model_path=model_path, cfg=cfg)


def train_agent(agent_name, cfg, seed, *, episodes, timesteps):
    mod = import_agent_module(agent_name)
    return mod.train(
        cfg=cfg,
        seed=seed,
        episodes=episodes,
        timesteps=timesteps,
    )
