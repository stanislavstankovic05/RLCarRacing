import os
import gymnasium as gym
import numpy as np
import pickle
from tqdm import tqdm

from agents.api import LoadedAgent, AgentSpec
from runner.common_env import make_env
from env.features import extract_features

class Policy:
    def __init__(self, q_table, epsilon=0.0):
        self.q_table = q_table
        self.epsilon = epsilon

    def predict(self, obs, env=None):
        state = extract_features(obs, "tile_v1", {"num_tiles": 8})
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        
        # Check if state exists in Q-table, if not, return random action
        if state not in self.q_table:
            return env.action_space.sample()
            
        return np.argmax(self.q_table[state])

def train(cfg: dict, seed: int, episodes: int, timesteps: int) -> None:
    print("Training Q-Learning agent")
    
    # Hyperparameters
    lr = cfg["agent"]["learning_rate"]
    gamma = cfg["agent"]["discount_factor"]
    epsilon_start = cfg["agent"]["epsilon_start"]
    epsilon_end = cfg["agent"]["epsilon_end"]
    epsilon_decay_episodes = cfg["agent"]["epsilon_decay_episodes"]

    # Environment setup
    agent_cfg = cfg["agent"]
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
    
    # Q-table initialization
    q_table = {}

    epsilon = epsilon_start
    
    for episode in tqdm(range(episodes)):
        obs, info = env.reset()
        state = extract_features(obs, cfg["agent"]["feature_extractor"], cfg["agent"][cfg["agent"]["feature_extractor"]])
        
        done = False
        truncated = False
        
        while not done and not truncated:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                if state not in q_table:
                    q_table[state] = np.zeros(env.action_space.n)
                action = np.argmax(q_table[state])

            # Ensure state is in q_table before taking step
            if state not in q_table:
                q_table[state] = np.zeros(env.action_space.n)

            next_obs, reward, done, truncated, info = env.step(action)
            next_state = extract_features(next_obs, cfg["agent"]["feature_extractor"], cfg["agent"][cfg["agent"]["feature_extractor"]])

            # Ensure next_state is in q_table
            if next_state not in q_table:
                q_table[next_state] = np.zeros(env.action_space.n)

            # Q-learning update rule
            old_value = q_table[state][action]
            next_max = np.max(q_table[next_state])
            
            new_value = (1 - lr) * old_value + lr * (reward + gamma * next_max)
            q_table[state][action] = new_value

            state = next_state

        # Epsilon decay
        if episode < epsilon_decay_episodes:
            epsilon = epsilon_start - (epsilon_start - epsilon_end) * (episode / epsilon_decay_episodes)
        else:
            epsilon = epsilon_end

    print("Training finished.")
    
    # Save Q-table
    model_dir = "results/models/q_learning_3"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "q_table.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(q_table, f)
    print(f"Q-table saved to {model_path}")


def load(model_path: str, cfg: dict) -> LoadedAgent:
    spec = AgentSpec(
        needs_pixels=cfg["agent"]["needs_pixels"],
        discrete_wrapper=cfg["agent"]["discrete_wrapper"],
        action_set_name=cfg["agent"].get("action_set"),
        resize=int(cfg["agent"].get("resize", 84)),
        frame_stack=int(cfg["agent"].get("frame_stack", 1)),
    )
    
    with open(model_path, "rb") as f:
        q_table = pickle.load(f)
        
    policy = Policy(q_table, epsilon=0.0) # Epsilon=0 for evaluation
    
    return LoadedAgent(name="q_learning", spec=spec, policy=policy)
