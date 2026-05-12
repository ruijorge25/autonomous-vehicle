"""
evaluate.py — Load a trained model and run N evaluation episodes.

Run this OUTSIDE of Webots (it connects via the same controller mechanism).
Actually it must be set as the active controller in Webots, just like
rl_controller.py — but instead of training, it only runs the policy.

Usage: set as the Webots controller, edit CONFIG below, press Play.

Results are printed to stdout and saved to logs/<run_name>_eval.csv.
"""

import os
import sys
import csv

CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
if CONTROLLER_DIR not in sys.path:
    sys.path.insert(0, CONTROLLER_DIR)

from controller import Supervisor
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from city_car_env import CityCarEnv

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "model_path":            "../../logs/models/ppo_dense_fixed_final",
    "algorithm":             "ppo",        # "ppo" or "sac"
    "reward_fn":             "dense",
    "procedural_obstacles":  False,
    "n_episodes":            100,
    "run_name":              "ppo_dense_fixed_eval",
}

LOG_DIR = os.path.join(CONTROLLER_DIR, "..", "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def main():
    supervisor = Supervisor()

    base_env = CityCarEnv(
        supervisor           = supervisor,
        reward_fn            = CONFIG["reward_fn"],
        procedural_obstacles = CONFIG["procedural_obstacles"],
        run_name             = CONFIG["run_name"],
    )

    # MUST match the training wrapper!
    vec_env = DummyVecEnv([lambda: base_env])
    env = VecFrameStack(vec_env, n_stack=4)

    AlgoClass = PPO if CONFIG["algorithm"] == "ppo" else SAC
    model = AlgoClass.load(CONFIG["model_path"], env=env)

    results = []
    for ep in range(CONFIG["n_episodes"]):
        obs = env.reset() # Note: VecEnv reset() returns just the obs, not (obs, info)
        
        ep_reward   = 0.0
        ep_steps    = 0
        terminated  = False
        
        # DummyVecEnv handles the done flags differently, it returns arrays
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            ep_reward += rewards[0]
            ep_steps  += 1

            if dones[0]:
                break

        # Read terminal flags from the underlying base env
        success   = base_env._ep_success
        collision = base_env._ep_collision

        results.append({
            "episode":      ep + 1,
            "total_reward": ep_reward,
            "steps":        ep_steps,
            "success":      success,
            "collision":    collision,
        })

        print(
            f"Ep {ep+1:>3}/{CONFIG['n_episodes']} | "
            f"reward={ep_reward:>8.1f} | steps={ep_steps:>4} | "
            f"success={success} | collision={collision}"
        )

    # ── Summary statistics ────────────────────────────────────────────────
    n = len(results)
    success_rate   = sum(r["success"]   for r in results) / n * 100
    collision_rate = sum(r["collision"] for r in results) / n * 100
    avg_reward     = sum(r["total_reward"] for r in results) / n
    avg_steps      = sum(r["steps"] for r in results) / n

    print("\n── Evaluation Summary ──────────────────────────────────")
    print(f"  Episodes        : {n}")
    print(f"  Success rate    : {success_rate:.1f}%")
    print(f"  Collision rate  : {collision_rate:.1f}%")
    print(f"  Avg reward      : {avg_reward:.2f}")
    print(f"  Avg steps       : {avg_steps:.1f}")
    print("────────────────────────────────────────────────────────\n")

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(LOG_DIR, f"{CONFIG['run_name']}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
