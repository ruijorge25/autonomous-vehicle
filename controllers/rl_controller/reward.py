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
W_PROGRESS    =  1.0   # metres advanced per step (primary driving signal)
W_HEADING     =  1.5   # alignment with road direction (cos curve, critical on bends)
W_LATERAL     =  1.2   # normalized deviation from yellow centre line
W_OBSTACLE    =  2.0   # proximity penalty — large weight, early threshold
W_SMOOTH      =  0.05  # small: don't over-constrain detour manoeuvres

OBSTACLE_THRESHOLD_M = 8.0   
LANE_LIMIT_M         = 9.0   

# -- UPDATED TERMINAL REWARDS --
PENALTY_COLLISION    = -500.0  # Increased from -100 to strongly discourage crashes
PENALTY_OUT_OF_LANE  = -100.0  # Kept lower than collision; grass is better than a wall
BONUS_SUCCESS        = +500.0  # Big carrot for completing the lap


def dense_reward(info: RewardInfo) -> float:
    """
    Multi-component dense reward.

    Design rationale
    ----------------
    r_progress   : primary signal — every metre forward is rewarded.
    r_heading    : cos(error) = 1 aligned, -1 reversed — forces correct orientation
                   on curves before out_of_lane fires.
    r_lateral    : penalises deviation from yellow centre line, normalised to
                   LANE_LIMIT_M so weight is scale-independent.  When an obstacle
                   sits on the centre line the obstacle penalty naturally grows
                   faster than the lateral penalty, creating a detour gradient.
    r_obstacle   : linear penalty from OBSTACLE_THRESHOLD_M=8 m down to 0 m.
                   Wide threshold gives the agent time to steer around obstacles.
    r_smooth     : tiny penalty for steering jerks; does not block sharp avoidance.
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

    # cos(heading_error) → 1 perfectly aligned, 0 at 90°, -1 reversed
    r_heading  = W_HEADING * math.cos(info.heading_error)

    # Normalized lateral deviation: 0 on centre line, -W_LATERAL at edge
    r_lateral  = -W_LATERAL * (abs(info.lateral_deviation) / LANE_LIMIT_M)

    # Obstacle proximity — linear, starts from 8 m
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
