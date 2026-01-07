import os
import csv
from dataclasses import dataclass, asdict

@dataclass
class EpisodeLog:
    episode: int
    reward: float
    steps: int
    finished: int
    offtrack_ratio: float
    tiles_visited: int

class CSVLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = None

    def log(self, row):
        d = asdict(row)
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(d.keys()))
            self._writer.writeheader()
        self._writer.writerow(d)
        self._file.flush()

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass
