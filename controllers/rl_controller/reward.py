"""
Reward functions for the CityCarEnv.

Two variants are provided:
  - dense_reward : multi-component shaped reward (Core Experiment 1, variant A)
  - sparse_reward: terminal-only reward             (Core Experiment 1, variant B)

Both functions receive the same RewardInfo dataclass so the env does not
need to change depending on which reward is active.
"""

import math
from dataclasses import dataclass


@dataclass
class RewardInfo:
    """All quantities needed to compute a reward at one timestep."""
    # Progress
    progress_m: float          # metres advanced along the track this step
    # Lane
    lateral_deviation: float   # metres from lane centre (signed)
    heading_error: float       # radians, signed angle between vehicle heading and road heading
    # Obstacle
    min_lidar: float           # metres to nearest object from frontal LiDAR
    # Smoothness
    steering_delta: float      # |steering_now - steering_prev|
    # Terminal flags
    collision: bool
    out_of_lane: bool
    success: bool


# ── Weights (tune these during experiments) ──────────────────────────────────
W_PROGRESS    =  1.0
W_HEADING     =  1.5   # increased: agent must align with road on curves
W_LATERAL     =  0.5   # increased: discourage drifting off centre
W_OBSTACLE    =  0.5   # penalty when closer than OBSTACLE_THRESHOLD_M
W_SMOOTH      =  0.1
OBSTACLE_THRESHOLD_M = 3.0
PENALTY_COLLISION    = -100.0
PENALTY_OUT_OF_LANE  = -100.0
BONUS_SUCCESS        = +200.0


def dense_reward(info: RewardInfo) -> float:
    """
    Multi-component dense reward used as the main training signal.

    Returns a single scalar reward for one environment step.
    """
    # ── Terminal conditions (override everything) ─────────────────────────
    if info.collision:
        return PENALTY_COLLISION
    if info.out_of_lane:
        return PENALTY_OUT_OF_LANE
    if info.success:
        return BONUS_SUCCESS

    # ── Per-step components ───────────────────────────────────────────────
    r_progress = W_PROGRESS * max(info.progress_m, 0.0)

    # cos(heading_error) → 1 when perfectly aligned, 0 at 90°, -1 reversed
    r_heading = W_HEADING * math.cos(info.heading_error)

    r_lateral = -W_LATERAL * abs(info.lateral_deviation)

    # Penalty that grows linearly as the vehicle approaches an obstacle
    # within OBSTACLE_THRESHOLD_M.  Zero beyond the threshold.
    if info.min_lidar < OBSTACLE_THRESHOLD_M:
        r_obstacle = -W_OBSTACLE * (1.0 - info.min_lidar / OBSTACLE_THRESHOLD_M)
    else:
        r_obstacle = 0.0

    r_smooth = -W_SMOOTH * info.steering_delta

    return r_progress + r_heading + r_lateral + r_obstacle + r_smooth


def sparse_reward(info: RewardInfo) -> float:
    """
    Sparse reward: only terminal feedback, zero during the episode.

    Used in Core Experiment 1 to compare against the dense variant.
    """
    if info.collision:
        return PENALTY_COLLISION
    if info.out_of_lane:
        return PENALTY_OUT_OF_LANE
    if info.success:
        return BONUS_SUCCESS
    return 0.0
