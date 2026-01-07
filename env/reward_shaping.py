from dataclasses import dataclass

@dataclass
class RewardParams:
    step_penalty: float = 0.1
    offtrack_penalty: float = 5.0
    progress_scale: float = 10.0
    finish_bonus: float = 1000.0

