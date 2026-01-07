from dataclasses import dataclass

@dataclass
class EpisodeMetrics:
    reward_sum = 0.0
    steps = 0
    finished = False
    offtrack_frames = 0
    tiles_visited = 0

    @property
    def offtrack_ratio(self):
        return 0.0 if self.steps == 0 else self.offtrack_frames / self.steps
