"""
plots.py — Generate training curve plots from episode CSV logs.

Usage:
    python plots.py --log_dir ../../logs --runs ppo_dense_fixed ppo_sparse_fixed

Produces:
  - reward_curve.png     : smoothed mean episode reward over time
  - success_rate.png     : rolling success rate (%)
  - collision_rate.png   : rolling collision rate (%)
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # headless rendering


WINDOW = 50   # rolling average window size


def load_run(log_dir: str, run_name: str) -> pd.DataFrame:
    path = os.path.join(log_dir, f"{run_name}_episodes.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No log file found: {path}")
    return pd.read_csv(path)


def smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def plot_metric(dfs: dict, col: str, ylabel: str, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    for run_name, df in dfs.items():
        ax.plot(df["episode"], smooth(df[col], WINDOW), label=run_name)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="../../logs")
    parser.add_argument("--runs",    nargs="+",
                        default=["ppo_dense_fixed", "ppo_sparse_fixed"])
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or args.log_dir
    os.makedirs(out_dir, exist_ok=True)

    dfs = {}
    for run in args.runs:
        try:
            dfs[run] = load_run(args.log_dir, run)
            print(f"Loaded {run}: {len(dfs[run])} episodes")
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    if not dfs:
        print("No data to plot.")
        return

    plot_metric(dfs, "total_reward",
                "Mean Episode Reward",
                f"Training Curve (window={WINDOW})",
                os.path.join(out_dir, "reward_curve.png"))

    # Convert bool columns to float for rolling mean
    for df in dfs.values():
        df["success_pct"]   = df["success"].astype(float)   * 100.0
        df["collision_pct"] = df["collision"].astype(float) * 100.0

    plot_metric(dfs, "success_pct",
                "Success Rate (%)",
                f"Rolling Success Rate (window={WINDOW})",
                os.path.join(out_dir, "success_rate.png"))

    plot_metric(dfs, "collision_pct",
                "Collision Rate (%)",
                f"Rolling Collision Rate (window={WINDOW})",
                os.path.join(out_dir, "collision_rate.png"))

    plot_metric(dfs, "avg_lateral_deviation",
                "Avg Lateral Deviation (m)",
                f"Lane Deviation (window={WINDOW})",
                os.path.join(out_dir, "lateral_deviation.png"))


if __name__ == "__main__":
    main()
