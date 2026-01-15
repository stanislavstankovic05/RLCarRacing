from dataclasses import dataclass

@dataclass
class RewardParams:
    step_penalty: float = 0.1
    offtrack_penalty: float = 5.0
    progress_scale: float = 10.0
    finish_bonus: float = 1000.0

    speed_p: float = 0.0
    speed_k: float = 0.1
    speed_b: float = 0.0
    
    on_grass_p: float = -1.0
    on_grass_k: float = 0.0
    on_grass_b: float = 0.0

    steering_p: float = -0.5
    steering_k: float = 0.0
    steering_b: float = 0.0

    gyro_p: float = -0.5
    gyro_k: float = 0.0
    gyro_b: float = 0.0

