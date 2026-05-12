"""
CityCarEnv — Gymnasium-compatible environment for the Webots City world.

Architecture
------------
This class runs *inside* the Webots controller process.  It uses the
Webots Supervisor API to reset the simulation state (vehicle pose + barrel
positions) at the start of every episode, then drives the simulation
forward one timestep at a time via supervisor.step().

Observation space  (Box, shape=(18,), dtype=float32)
    [0]     lateral_deviation   / 5.0          (clipped to [-1, 1])
    [1]     heading_error       / π             (clipped to [-1, 1])
    [2]     speed               / 20.0
    [3]     prev_steering                       (already in [-1, 1])
    [4-15]  lidar[0:12]         / 30.0         (frontal subset)
    [16]    min_lidar           / 30.0
    [17]    progress_norm       = total_progress / 500.0  (clipped to [0, 1])

Action space  (Box, shape=(2,), dtype=float32)
    [0]  steering   ∈ [-1, 1]  → mapped to [-0.5, 0.5] rad
    [1]  throttle   ∈ [-1, 1]  → mapped to [-5, 30] rad/s wheel velocity

Episode termination
    • collision  : min frontal LiDAR < COLLISION_DIST_M
    • out_of_lane: |lateral_deviation| > LANE_LIMIT_M
    • timeout    : step_count > MAX_STEPS
    • success    : wp_index completes one full lap (wraps around all waypoints)
"""

import math
import random
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from reward import RewardInfo, dense_reward, sparse_reward
from waypoints import (
    WAYPOINTS, BARREL_FIXED_POSITIONS, BARREL_SPAWN_CANDIDATES,
    TRAFFIC_WAYPOINTS, TRAFFIC_SPEED_MS, NUM_TRAFFIC_CARS,
)

# Logger lives in utils/ — add it to path
_UTILS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)
from logger import EpisodeLogger

# ── Tunable constants ─────────────────────────────────────────────────────────
COLLISION_DIST_M   = 0.3    # metres — episode ends if LiDAR reads below this
                             # NOTE: BmwX5 hood reflects LiDAR at ~0.78m; 0.3m avoids false collisions
LANE_LIMIT_M       = 9.0    # metres — triggers ~1.75 m before the physical barrier (road half-width=10.75m)
FRAME_SKIP         = 5      # simulate N physics steps per RL step (160 ms per step @ 32 ms)
MAX_STEPS          = 5000   # RL steps per episode (800 s); generous enough for a full lap
STUCK_SPEED_MS     = 0.1    # m/s — below this the car is considered stuck
STUCK_STEPS        = 40     # consecutive steps below STUCK_SPEED_MS → terminate (2.4 s)
LIDAR_MAX_M        = 30.0   # normalisation range — SICK LMS 291 in Webots has 30 m max range
SPEED_MAX          = 20.0   # normalisation range for speed
STEERING_RANGE     = 0.5    # rad — maps action [-1,1] to [-0.5, 0.5] rad
THROTTLE_MIN       = 0.0    # rad/s wheel velocity at action=-1 (0 = no reverse; avoids lazy-reverse exploit)
THROTTLE_MAX       = 30.0   # rad/s wheel velocity at action=+1
N_LIDAR_RAYS       = 12     # number of frontal rays used in observation
WAYPOINT_REACH_M   = 12.0   # metres — > half road width (10.75m); triggers anywhere on the asphalt
NUM_BARRELS        = 14

# Spawn positions and headings for the vehicle at episode start.
# Each tuple is (x, y, z, heading_rad).  heading is the Webots rotation
# around Z axis.  A random one is picked at reset().
# Spawn poses on the YELLOW CENTRE LINE of each road segment.
# Coordinates derived from road segment translations in city.wbt.
# z=0.332 (road surface height, confirmed from Webots).
BARREL_COLLISION_M = 1.5   # geometric barrel collision radius (barrel r=0.4 + car half-width ~1.0 + margin)

SPAWN_POSES = [
    (-105.0, -50.0, 0.332,  math.pi / 2),   # west straight, heading north
    (-105.0, -20.0, 0.332,  math.pi / 2),   # west straight, heading north
    ( -50.0,  45.0, 0.332,  0.0),            # north straight, heading east (shifted from -35 to avoid barrel)
    (  45.0, -35.0, 0.332, -math.pi / 2),   # east straight, heading south
    ( -30.0,-105.0, 0.332,  math.pi),        # south straight, heading west
]


class CityCarEnv(gym.Env):
    """Gymnasium environment wrapping the Webots City simulation."""

    metadata = {"render_modes": []}

    def __init__(self, supervisor, reward_fn="dense", procedural_obstacles=False, run_name="default"):
        """
        Parameters
        ----------
        supervisor : webots_api.Supervisor
            The Webots Supervisor instance from the controller entry point.
        reward_fn : str
            "dense" or "sparse"
        procedural_obstacles : bool
            If True, barrel positions are randomised each episode.
            If False, default fixed positions from BARREL_FIXED_POSITIONS are used.
        """
        super().__init__()
        self.supervisor = supervisor
        self.timestep = int(supervisor.getBasicTimeStep())
        self.reward_fn = dense_reward if reward_fn == "dense" else sparse_reward
        self.procedural_obstacles = procedural_obstacles
        self.run_name = run_name

        # ── Webots devices ────────────────────────────────────────────────
        self._init_devices()

        # ── Gymnasium spaces ──────────────────────────────────────────────
        obs_low  = np.full(18, -1.0, dtype=np.float32)
        obs_high = np.full(18,  1.0, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0], dtype=np.float32),
        )

        # ── Episode state ─────────────────────────────────────────────────
        self.step_count      = 0
        self.prev_steering   = 0.0
        self.prev_pos        = None
        self.wp_index        = 0       # index of next target waypoint
        self._wp_laps        = 0       # incremented each time wp_index wraps around
        self.total_progress  = 0.0    # cumulative metres along track
        self.episode_num     = 0

        # ── Per-episode accumulators (for logger) ─────────────────────────
        self._ep_reward      = 0.0
        self._ep_lateral_sum = 0.0
        self._ep_lidar_min   = float("inf")
        self._ep_collision   = False
        self._ep_out_lane    = False
        self._ep_success     = False
        self._stuck_counter  = 0     # consecutive steps with speed < STUCK_SPEED_MS
        self._barrel_positions = []  # [(bx, by)] updated each reset by _reset_barrels

        # ── Logger ───────────────────────────────────────────────────────
        log_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs"
        )
        run_name = getattr(self, "run_name", "default")
        self.logger = EpisodeLogger(run_name, log_dir=log_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # Device initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _init_devices(self):
        sv = self.supervisor

        # Actuators
        self.left_steer   = sv.getDevice("left_steer")
        self.right_steer  = sv.getDevice("right_steer")
        self.left_wheel   = sv.getDevice("left_front_wheel")
        self.right_wheel  = sv.getDevice("right_front_wheel")
        self.left_wheel.setPosition(float("inf"))
        self.right_wheel.setPosition(float("inf"))

        # LiDAR (SICK LMS 291 — 180°, 361 rays)
        self.lidar = sv.getDevice("Sick LMS 291")
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        # GPS
        self.gps = sv.getDevice("gps")
        self.gps.enable(self.timestep)

        # Gyro
        self.gyro = sv.getDevice("gyro")
        self.gyro.enable(self.timestep)

        # Supervisor node references
        self.vehicle_node = sv.getFromDef("VEHICLE")
        self.barrel_nodes = [sv.getFromDef(f"BARREL_{i}") for i in range(NUM_BARRELS)]

        # Traffic car nodes (kinematically controlled by supervisor)
        self.traffic_nodes = [sv.getFromDef(f"TRAFFIC_CAR_{i}") for i in range(NUM_TRAFFIC_CARS)]
        self._traffic_progress = [0.0] * NUM_TRAFFIC_CARS
        self._traffic_circuit_length = self._compute_traffic_circuit_length()

        # GPS history for speed estimation
        self._prev_gps = None

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Pick a random spawn pose
        pose = random.choice(SPAWN_POSES)
        x0, y0, z0, heading0 = pose

        # Optional small perturbation for robustness experiments
        x0      += random.uniform(-1.0, 1.0)
        y0      += random.uniform(-1.0, 1.0)
        heading0 += random.uniform(-0.1, 0.1)

        # Teleport vehicle
        trans_field = self.vehicle_node.getField("translation")
        rot_field   = self.vehicle_node.getField("rotation")
        trans_field.setSFVec3f([x0, y0, z0])
        rot_field.setSFRotation([0.0, 0.0, 1.0, heading0])
        self.vehicle_node.resetPhysics()

        # Position barrels
        self._reset_barrels()

        # Space traffic cars evenly around secondary circuit
        self._reset_traffic()

        # Stop wheels
        self._apply_action(0.0, 0.0)

        # Advance a few steps to let physics settle
        for _ in range(5):
            self.supervisor.step(self.timestep)

        # Reset episode state
        self.step_count      = 0
        self.prev_steering   = 0.0
        self._prev_gps       = self.gps.getValues()
        self.wp_index        = self._nearest_waypoint_index(x0, y0)
        self._wp_laps        = 0
        # Skip the nearest waypoint if the spawn is already within reach distance
        # (prevents immediate advance to a far waypoint on step 1)
        _wx, _wy, _ = WAYPOINTS[self.wp_index]
        if math.hypot(x0 - _wx, y0 - _wy) < WAYPOINT_REACH_M:
            self.wp_index = (self.wp_index + 1) % len(WAYPOINTS)
        self.total_progress  = 0.0
        self._ep_reward      = 0.0
        self._ep_lateral_sum = 0.0
        self._ep_lidar_min   = float("inf")
        self._ep_collision   = False
        self._ep_out_lane    = False
        self._ep_success     = False
        self._stuck_counter  = 0
        self._wp_visited     = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], -1.0, 1.0))

        self._apply_action(steering, throttle)
        for _ in range(FRAME_SKIP):
            self.supervisor.step(self.timestep)
        self._update_traffic()
        self.step_count += 1

        obs  = self._get_obs()
        info = self._get_reward_info(steering)

        # ── Stuck detection ───────────────────────────────────────────────
        speed = self._get_speed()
        if speed < STUCK_SPEED_MS:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0
        stuck = self._stuck_counter >= STUCK_STEPS

        reward      = self.reward_fn(info)
        terminated  = info.collision or info.out_of_lane or info.success or stuck
        truncated   = self.step_count >= MAX_STEPS

        # ── Accumulate per-episode stats ──────────────────────────────────
        self._ep_reward      += reward
        self._ep_lateral_sum += abs(info.lateral_deviation)
        self._ep_lidar_min    = min(self._ep_lidar_min, info.min_lidar)
        if info.collision:   self._ep_collision = True
        if info.out_of_lane: self._ep_out_lane  = True
        if info.success:     self._ep_success   = True
        if stuck:            self._ep_out_lane  = True   # log stuck as out_of_lane

        self.prev_steering = steering

        # Update waypoint progress
        gps_vals = self.gps.getValues()
        cx, cy   = gps_vals[0], gps_vals[1]
        self._advance_waypoint(cx, cy)
        self._prev_gps = gps_vals

        # ── Log episode when it ends ──────────────────────────────────────
        if terminated or truncated:
            self.episode_num += 1
            avg_lat = self._ep_lateral_sum / max(self.step_count, 1)
            self.logger.log(
                episode               = self.episode_num,
                total_reward          = self._ep_reward,
                steps                 = self.step_count,
                success               = self._ep_success,
                collision             = self._ep_collision,
                out_of_lane           = self._ep_out_lane,
                avg_lateral_deviation = avg_lat,
                min_lidar_min         = self._ep_lidar_min,
            )

        return obs, reward, terminated, truncated, {}

    def close(self):
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_action(self, steering_norm, throttle_norm):
        steer   = steering_norm * STEERING_RANGE
        vel     = THROTTLE_MIN + (throttle_norm + 1.0) / 2.0 * (THROTTLE_MAX - THROTTLE_MIN)
        self.left_steer.setPosition(steer)
        self.right_steer.setPosition(steer)
        self.left_wheel.setVelocity(vel)
        self.right_wheel.setVelocity(vel)

    def _get_lidar_frontal(self):
        """Return N_LIDAR_RAYS values from the central frontal sector using min-pooling."""
        values = self.lidar.getRangeImage()
        if not values:
            return [LIDAR_MAX_M] * N_LIDAR_RAYS
        
        n = len(values)
        
        # Focus on the frontal region (e.g., middle half of the 180-degree field of view)
        front_rays = n // 2 
        start_idx = (n - front_rays) // 2
        end_idx = start_idx + front_rays
        
        frontal_values = values[start_idx:end_idx]
        
        # Clean up NaNs and infinities
        cleaned_values = [
            min(v, LIDAR_MAX_M) if v != float("inf") and not math.isnan(v) else LIDAR_MAX_M 
            for v in frontal_values
        ]
        
        # Divide into sectors and take the minimum (closest object) per sector
        sector_size = max(1, len(cleaned_values) // N_LIDAR_RAYS)
        rays = []
        
        for i in range(N_LIDAR_RAYS):
            sector_start = i * sector_size
            # Ensure the last sector catches any remainder
            sector_end = (i + 1) * sector_size if i < N_LIDAR_RAYS - 1 else len(cleaned_values)
            
            sector_min = min(cleaned_values[sector_start:sector_end])
            rays.append(sector_min)
            
        return rays

    def _nearest_waypoint_index(self, x, y):
        """Return index of the waypoint closest to (x, y)."""
        best_i, best_d = 0, float("inf")
        for i, (wx, wy, _) in enumerate(WAYPOINTS):
            d = math.hypot(x - wx, y - wy)
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    # ── Traffic car helpers ───────────────────────────────────────────────────

    def _compute_traffic_circuit_length(self):
        """Total length in metres of the TRAFFIC_WAYPOINTS circuit."""
        total = 0.0
        n = len(TRAFFIC_WAYPOINTS)
        for i in range(n):
            p1 = TRAFFIC_WAYPOINTS[i]
            p2 = TRAFFIC_WAYPOINTS[(i + 1) % n]
            total += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        return total

    def _circuit_position(self, progress_m):
        """Return (x, y, heading_rad) at *progress_m* along TRAFFIC_WAYPOINTS."""
        remaining = progress_m % self._traffic_circuit_length
        n = len(TRAFFIC_WAYPOINTS)
        for i in range(n):
            p1 = TRAFFIC_WAYPOINTS[i]
            p2 = TRAFFIC_WAYPOINTS[(i + 1) % n]
            seg_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if remaining <= seg_len or i == n - 1:
                t = min(remaining / max(seg_len, 1e-6), 1.0)
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                heading = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                return x, y, heading
            remaining -= seg_len
        p = TRAFFIC_WAYPOINTS[0]
        return p[0], p[1], 0.0

    def _reset_traffic(self):
        """Evenly space all traffic cars around the secondary circuit."""
        n = NUM_TRAFFIC_CARS
        circuit_len = self._traffic_circuit_length
        for i in range(n):
            self._traffic_progress[i] = i * circuit_len / n
            if self.traffic_nodes[i] is not None:
                x, y, heading = self._circuit_position(self._traffic_progress[i])
                self.traffic_nodes[i].getField("translation").setSFVec3f([x, y, 0.4])
                self.traffic_nodes[i].getField("rotation").setSFRotation([0.0, 0.0, 1.0, heading])

    def _update_traffic(self):
        """Advance each traffic car kinematically along the secondary circuit."""
        dt = self.timestep * FRAME_SKIP / 1000.0   # seconds per RL step
        for i, node in enumerate(self.traffic_nodes):
            if node is None:
                continue
            self._traffic_progress[i] = (
                self._traffic_progress[i] + TRAFFIC_SPEED_MS * dt
            ) % self._traffic_circuit_length
            x, y, heading = self._circuit_position(self._traffic_progress[i])
            node.getField("translation").setSFVec3f([x, y, 0.4])
            node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, heading])

    def _advance_waypoint(self, x, y):
        """Move wp_index forward if the vehicle has reached the current target."""
        wx, wy, _ = WAYPOINTS[self.wp_index]
        if math.hypot(x - wx, y - wy) < WAYPOINT_REACH_M:
            next_idx = (self.wp_index + 1) % len(WAYPOINTS)
            
            # Conta mais um waypoint visitado nesta vida
            self._wp_visited += 1
            
            # Só regista uma volta completa se passar efetivamente pelos 8 pontos
            if self._wp_visited >= len(WAYPOINTS):
                self._wp_laps += 1
                self._wp_visited = 0  # Faz reset para a próxima volta
                
            self.wp_index = next_idx

    def _compute_lateral_and_heading(self, x, y, vehicle_heading):
        """
        Compute signed lateral deviation and heading error using exact circuit
        geometry (4 straights + 4 curves), NOT waypoint midpoints.

        Road layout (yellow centre line, clockwise in XY plane):
          West  straight: x = -105,  y ∈ [-64.5,  4.5],  heading =  π/2
          North straight: y =   45,  x ∈ [-64.5,  4.5],  heading =  0
          East  straight: x =   45,  y ∈ [-64.5,  4.5],  heading = -π/2
          South straight: y = -105,  x ∈ [-64.5,  4.5],  heading =  π
          NW curve: centre (-64.5,  4.5) R=40.5  (active when x≤-64.5 & y≥ 4.5)
          NE curve: centre ( 4.5,   4.5) R=40.5  (active when x≥ 4.5  & y≥ 4.5)
          SE curve: centre ( 4.5, -64.5) R=40.5  (active when x≥ 4.5  & y≤-64.5)
          SW curve: centre (-64.5,-64.5) R=40.5  (active when x≤-64.5 & y≤-64.5)
        """
        _R  =  40.5   # curve radius
        _LO = -64.5   # junction low  coordinate
        _HI =   4.5   # junction high coordinate

        # Each entry: (abs_lat, signed_lat, road_heading)
        candidates = []

        # ── Straight segments ─────────────────────────────────────────────
        if _LO <= y <= _HI:                          # west straight
            lat = x - (-105.0)
            candidates.append((abs(lat), lat,  math.pi / 2))
        if _LO <= x <= _HI:                          # north straight
            lat = y - 45.0
            candidates.append((abs(lat), lat,  0.0))
        if _LO <= y <= _HI:                          # east straight
            lat = x - 45.0
            candidates.append((abs(lat), lat, -math.pi / 2))
        if _LO <= x <= _HI:                          # south straight
            lat = y - (-105.0)
            candidates.append((abs(lat), lat,  math.pi))

        # ── Curve segments ────────────────────────────────────────────────
        for cx, cy, in_quad in (
            (-64.5,   4.5, x <= -64.5 and y >=   4.5),   # NW
            (  4.5,   4.5, x >=   4.5 and y >=   4.5),   # NE
            (  4.5, -64.5, x >=   4.5 and y <= -64.5),   # SE
            (-64.5, -64.5, x <= -64.5 and y <= -64.5),   # SW
        ):
            if in_quad:
                d   = math.hypot(x - cx, y - cy)
                lat = d - _R
                # Clockwise tangent heading: theta - π/2
                rh  = math.atan2(y - cy, x - cx) - math.pi / 2
                rh  = (rh + math.pi) % (2 * math.pi) - math.pi
                candidates.append((abs(lat), lat, rh))

        if candidates:
            _, lateral, road_heading = min(candidates, key=lambda c: c[0])
        else:
            # Fallback (car deep inside circuit, should not happen in training)
            near_i = self._nearest_waypoint_index(x, y)
            wp_x, wp_y, road_heading = WAYPOINTS[near_i]
            dx = x - wp_x;  dy = y - wp_y
            lateral = dx * (-math.sin(road_heading)) + dy * math.cos(road_heading)

        err = vehicle_heading - road_heading
        err = (err + math.pi) % (2 * math.pi) - math.pi
        return lateral, err

    def _get_vehicle_heading(self):
        """Estimate vehicle heading from Gyro or GPS differential."""
        gyro_vals = self.gyro.getValues()
        # gyro gives angular velocity; we need cumulative heading.
        # Instead, derive heading from GPS displacement if possible.
        gps = self.gps.getValues()
        if self._prev_gps is not None:
            dx = gps[0] - self._prev_gps[0]
            dy = gps[1] - self._prev_gps[1]
            if abs(dx) > 1e-4 or abs(dy) > 1e-4:
                return math.atan2(dy, dx)
        # Fallback: use the vehicle node's rotation field
        rot = self.vehicle_node.getField("rotation").getSFRotation()
        # rot = [ax, ay, az, angle]; vehicle forward is +X, so heading = angle if az>0
        angle = rot[3]
        if rot[2] < 0:
            angle = -angle
        return angle

    def _get_speed(self):
        gps = self.gps.getValues()
        if self._prev_gps is None:
            return 0.0
        dx = gps[0] - self._prev_gps[0]
        dy = gps[1] - self._prev_gps[1]
        dist = math.hypot(dx, dy)
        dt = (self.timestep * FRAME_SKIP) / 1000.0  # total elapsed time per RL step
        return dist / dt if dt > 0 else 0.0

    def _get_obs(self):
        gps    = self.gps.getValues()
        x, y   = gps[0], gps[1]
        speed  = self._get_speed()
        heading = self._get_vehicle_heading()

        lateral, heading_err = self._compute_lateral_and_heading(x, y, heading)
        lidar_rays = self._get_lidar_frontal()
        min_lidar  = min(lidar_rays)

        progress_norm = np.clip(self.total_progress / 500.0, 0.0, 1.0)

        obs = np.array([
            np.clip(lateral       / 5.0,        -1.0, 1.0),
            np.clip(heading_err   / math.pi,    -1.0, 1.0),
            np.clip(speed         / SPEED_MAX,   0.0, 1.0),
            self.prev_steering,
            *[np.clip(r / LIDAR_MAX_M, 0.0, 1.0) for r in lidar_rays],
            np.clip(min_lidar     / LIDAR_MAX_M, 0.0, 1.0),
            float(progress_norm),
        ], dtype=np.float32)
        return obs

    def _get_reward_info(self, steering):
        gps  = self.gps.getValues()
        x, y = gps[0], gps[1]
        heading = self._get_vehicle_heading()

        lateral, heading_err = self._compute_lateral_and_heading(x, y, heading)
        lidar_rays = self._get_lidar_frontal()
        min_lidar  = min(lidar_rays)

        # Progress: distance advanced toward the next waypoint
        if self._prev_gps is not None:
            dx = x - self._prev_gps[0]
            dy = y - self._prev_gps[1]
            wp_x, wp_y, wh = WAYPOINTS[self.wp_index]
            road_dx = math.cos(wh)
            road_dy = math.sin(wh)
            progress = dx * road_dx + dy * road_dy
        else:
            progress = 0.0

        # LiDAR-based collision (walls, static objects)
        lidar_collision = min_lidar < COLLISION_DIST_M
        # Geometric barrel collision — reliable even if barrel moved after impact
        barrel_collision = any(
            math.hypot(x - bx, y - by) < BARREL_COLLISION_M
            for bx, by in self._barrel_positions
        )
        collision   = lidar_collision or barrel_collision
        out_of_lane = abs(lateral) > LANE_LIMIT_M
        # Success: completed a full loop (wp_index wrapped around at least once)
        # Simple proxy: total_progress > circuit perimeter (~500 m)
        self.total_progress += max(progress, 0.0)
        success = self._wp_laps >= 1  # true only after visiting every waypoint once

        return RewardInfo(
            progress_m        = progress,
            lateral_deviation = lateral,
            heading_error     = heading_err,
            min_lidar         = min_lidar,
            steering_delta    = abs(steering - self.prev_steering),
            collision         = collision,
            out_of_lane       = out_of_lane,
            success           = success,
        )

    def _reset_barrels(self):
        """Reposition all barrels — fixed or procedurally generated.
        Always resets rotation to upright so fallen barrels don't persist.
        Stores barrel centres in self._barrel_positions for geometric collision.
        """
        if self.procedural_obstacles:
            candidates = list(BARREL_SPAWN_CANDIDATES)
            random.shuffle(candidates)
            positions = candidates[:NUM_BARRELS]
            while len(positions) < NUM_BARRELS:
                positions.append((-200.0, -200.0))
        else:
            positions = [(p[0], p[1]) for p in BARREL_FIXED_POSITIONS]

        self._barrel_positions = []
        for i, node in enumerate(self.barrel_nodes):
            if node is None:
                continue
            bx, by = positions[i]
            if self.procedural_obstacles:
                bx += random.uniform(-1.5, 1.5)
                by += random.uniform(-1.5, 1.5)
            node.getField("translation").setSFVec3f([bx, by, 0.6])
            node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, 0.0])  # reset upright
            node.resetPhysics()
            self._barrel_positions.append((bx, by))
