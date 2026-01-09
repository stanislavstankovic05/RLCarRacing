from __future__ import annotations
import gymnasium as gym
import numpy as np
from typing import Optional

from env.reward_shaping import RewardParams
from env.metrics import EpisodeMetrics
from env.features import is_offtrack

class CarRacing(gym.Wrapper):
    def __init__(self, env, reward_params, offtrack_frames = 50, no_progress_steps = 100):
        super().__init__(env)
        self.rp = reward_params
        self.max_offtrack_frames = offtrack_frames
        self.max_no_progress_steps = no_progress_steps

        self.metrics = EpisodeMetrics()
        self._prev_tiles = 0
        self._offtrack_consecutive = 0
        self._no_progress_counter = 0
        self._track_len: Optional[int] = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.metrics = EpisodeMetrics()
        self._prev_tiles = int(getattr(self.env.unwrapped, "tile_visited_count", 0))
        self._offtrack_consecutive = 0
        self._no_progress_counter = 0
        track = getattr(self.env.unwrapped, "track", None)
        self._track_len = len(track) if track is not None else None
        return obs, info

    def finished(self):
        tiles = int(getattr(self.env.unwrapped, "tile_visited_count", 0))
        track = getattr(self.env.unwrapped, "track", None)
        track_len = self._track_len if self._track_len is not None else (len(track) if track is not None else None)
        if track_len is None:
            return False
        return tiles >= max(1, track_len - 1)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        tiles = int(getattr(self.env.unwrapped, "tile_visited_count", 0))
        progress_delta = max(0, tiles - self._prev_tiles)

        off = is_offtrack(self.env, obs_rgb=(obs if isinstance(obs, np.ndarray) and obs.ndim == 3 else None))
        if off:
            self._offtrack_consecutive += 1
            self.metrics.offtrack_frames += 1
        else:
            self._offtrack_consecutive = 0

        if progress_delta > 0:
            self._no_progress_counter = 0
        else:
            self._no_progress_counter += 1

        finished = self.finished()

        reward = 0.0
        reward += self.rp.progress_scale * float(progress_delta)
        reward -= self.rp.step_penalty
        if off:
            reward -= self.rp.offtrack_penalty
        if finished:
            reward += self.rp.finish_bonus

        shaped_terminated = False
        if self._offtrack_consecutive >= self.max_offtrack_frames:
            # print("Termination: offtrack limit reached")
            shaped_terminated = True
        if self._no_progress_counter >= self.max_no_progress_steps:
            # print("Termination: no progress limit reached")
            shaped_terminated = True
        if finished:
            shaped_terminated = True

        terminated = bool(terminated or shaped_terminated)

        self.metrics.reward_sum += float(reward)
        self.metrics.steps += 1
        self.metrics.tiles_visited = tiles
        if finished:
            self.metrics.finished = True

        info = dict(info)
        info["tiles_visited"] = tiles
        info["progress_delta"] = progress_delta
        info["offtrack"] = off
        info["finished"] = finished

        self._prev_tiles = tiles
        return obs, float(reward), terminated, truncated, info

class ActionWrapper(gym.Wrapper):
    def __init__(self, env, actions):
        super().__init__(env)
        self.actions = actions
        self.action_space = gym.spaces.Discrete(len(actions))

    def step(self, action_idx):
        a = self.actions[int(action_idx)]
        # Pass plain Python floats (not numpy scalars) to avoid Box2D SWIG type issues
        action = (float(a[0]), float(a[1]), float(a[2]))
        # Convert the tuple action to a numpy array before passing it to the environment
        return self.env.step(np.array(action))


class CarRacingMetrics(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.metrics = EpisodeMetrics()
        self._track_len: Optional[int] = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.metrics = EpisodeMetrics()
        track = getattr(self.env.unwrapped, "track", None)
        self._track_len = len(track) if track is not None else None
        return obs, info

    def finished(self):
        tiles = int(getattr(self.env.unwrapped, "tile_visited_count", 0))
        track = getattr(self.env.unwrapped, "track", None)
        track_len = self._track_len if self._track_len is not None else (len(track) if track is not None else None)
        if track_len is None:
            return False
        return tiles >= max(1, track_len - 1)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        off = is_offtrack(self.env, obs_rgb=(obs if isinstance(obs, np.ndarray) and obs.ndim == 3 else None))
        tiles = int(getattr(self.env.unwrapped, "tile_visited_count", 0))
        finished = self.finished()

        self.metrics.reward_sum += float(reward)
        self.metrics.steps += 1
        self.metrics.tiles_visited = tiles
        if off:
            self.metrics.offtrack_frames += 1
        if finished:
            self.metrics.finished = True

        info = dict(info)
        info["tiles_visited"] = tiles
        info["offtrack"] = off
        info["finished"] = finished

        return obs, float(reward), bool(terminated), bool(truncated), info
