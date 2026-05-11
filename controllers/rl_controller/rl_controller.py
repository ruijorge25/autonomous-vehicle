"""
rl_controller.py — Webots controller entry point.

This file is launched by Webots when the simulation starts.
It instantiates the CityCarEnv, trains a PPO (or SAC) agent via
Stable-Baselines3, and saves the resulting model + logs.

Configuration
-------------
Edit the CONFIG dict below to switch between experiments:
  reward_fn            : "dense" | "sparse"
  algorithm            : "ppo"   | "sac"
  procedural_obstacles : True    | False
  total_timesteps      : how long to train
"""

import os
import sys

# ── Make sure the controller folder is on the path ───────────────────────────
CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
if CONTROLLER_DIR not in sys.path:
    sys.path.insert(0, CONTROLLER_DIR)

# ── Webots Supervisor ─────────────────────────────────────────────────────────
from controller import Supervisor

# ── Stable-Baselines3 ─────────────────────────────────────────────────────────
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback

# ── Project modules ───────────────────────────────────────────────────────────
from city_car_env import CityCarEnv

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT CONFIGURATION — change these to switch experiments
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "reward_fn":             "dense",   # "dense" or "sparse"
    "algorithm":             "ppo",     # "ppo"   or "sac"
    "procedural_obstacles":  False,     # False = fixed, True = procedural
    "total_timesteps":       200_000,
    "save_freq":             10_000,    # checkpoint every N steps
}

# Derive a run name from the config for organised logging
RUN_NAME = (
    f"{CONFIG['algorithm']}_"
    f"{CONFIG['reward_fn']}_"
    f"{'proc' if CONFIG['procedural_obstacles'] else 'fixed'}"
)
LOG_DIR   = os.path.join(CONTROLLER_DIR, "..", "..", "logs", RUN_NAME)
MODEL_DIR = os.path.join(CONTROLLER_DIR, "..", "..", "logs", "models")
os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    supervisor = Supervisor()

    env = CityCarEnv(
        supervisor            = supervisor,
        reward_fn             = CONFIG["reward_fn"],
        procedural_obstacles  = CONFIG["procedural_obstacles"],
        run_name              = RUN_NAME,
    )

    # ── Choose algorithm ──────────────────────────────────────────────────
    AlgoClass = PPO if CONFIG["algorithm"] == "ppo" else SAC

    policy_kwargs = dict(net_arch=[256, 256])

    model = AlgoClass(
        policy        = "MlpPolicy",
        env           = env,
        verbose       = 1,
        tensorboard_log = LOG_DIR,
        policy_kwargs = policy_kwargs,
        # PPO-specific defaults that work well for continuous control
        **(dict(
            n_steps          = 2048,
            batch_size       = 64,
            n_epochs         = 10,
            learning_rate    = 3e-4,
            clip_range       = 0.2,
            ent_coef         = 0.01,
        ) if CONFIG["algorithm"] == "ppo" else dict(
            learning_rate    = 3e-4,
            batch_size       = 256,
            buffer_size      = 100_000,
            learning_starts  = 1000,
            tau              = 0.005,
            ent_coef         = "auto",
        )),
    )

    # ── Checkpoint callback ───────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = CONFIG["save_freq"],
        save_path   = MODEL_DIR,
        name_prefix = RUN_NAME,
    )

    print(f"\n[rl_controller] Starting training: {RUN_NAME}")
    print(f"  timesteps : {CONFIG['total_timesteps']}")
    print(f"  logs      : {LOG_DIR}\n")

    model.learn(
        total_timesteps = CONFIG["total_timesteps"],
        callback        = checkpoint_cb,
        progress_bar    = False,
    )

    # ── Save final model ──────────────────────────────────────────────────
    final_path = os.path.join(MODEL_DIR, f"{RUN_NAME}_final")
    model.save(final_path)
    print(f"\n[rl_controller] Training complete. Model saved to {final_path}.zip")


if __name__ == "__main__":
    main()
