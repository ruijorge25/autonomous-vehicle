"""
logger.py — Episode metrics logger.

Writes one CSV row per episode with:
  episode, total_reward, steps, success, collision, out_of_lane,
  avg_lateral_deviation, min_lidar_min, run_name

Usage (inside CityCarEnv or a wrapper):
    from logger import EpisodeLogger
    logger = EpisodeLogger("ppo_dense_fixed", log_dir="../../logs")
    logger.log(episode=1, total_reward=45.2, steps=300, success=False, ...)
"""

import csv
import os
from dataclasses import dataclass, fields, astuple


@dataclass
class EpisodeRecord:
    episode:              int
    total_reward:         float
    steps:                int
    success:              bool
    collision:            bool
    out_of_lane:          bool
    avg_lateral_deviation: float
    min_lidar_min:        float
    run_name:             str


class EpisodeLogger:
    """Appends episode records to a CSV file."""

    def __init__(self, run_name: str, log_dir: str = "../../logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.run_name = run_name
        self.csv_path = os.path.join(log_dir, f"{run_name}_episodes.csv")
        self._write_header()

    def _write_header(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([field.name for field in fields(EpisodeRecord)])

    def log(self, episode: int, total_reward: float, steps: int,
            success: bool, collision: bool, out_of_lane: bool,
            avg_lateral_deviation: float, min_lidar_min: float):
        record = EpisodeRecord(
            episode=episode,
            total_reward=total_reward,
            steps=steps,
            success=success,
            collision=collision,
            out_of_lane=out_of_lane,
            avg_lateral_deviation=avg_lateral_deviation,
            min_lidar_min=min_lidar_min,
            run_name=self.run_name,
        )
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(astuple(record))
