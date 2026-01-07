import os
from typing import List
import imageio

def save_video(frames, path, fps = 30):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, codec="libx264")
