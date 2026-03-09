#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_action_rate_stats(path: Path):
    steps = []
    mean_abs = []
    max_abs = []

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip header lines like: policy_step\tmean_abs\tmax_abs
            if line[0].isalpha() or line.startswith("#"):
                continue

            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                continue

            try:
                steps.append(float(parts[0]))
                mean_abs.append(float(parts[1]))
                max_abs.append(float(parts[2]))
            except ValueError:
                continue

    if not steps:
        raise ValueError(f"No valid rows found in {path}")

    steps = np.asarray(steps, dtype=float)
    mean_abs = np.asarray(mean_abs, dtype=float)
    max_abs = np.asarray(max_abs, dtype=float)

    mean_abs_time_avg = np.cumsum(mean_abs) / np.arange(1, len(mean_abs) + 1)
    max_abs_time_avg = np.cumsum(max_abs) / np.arange(1, len(max_abs) + 1)
    mean_abs_overall = float(np.mean(mean_abs))
    max_abs_overall = float(np.mean(max_abs))

    return steps, mean_abs, max_abs, mean_abs_time_avg, max_abs_time_avg, mean_abs_overall, max_abs_overall


def main():
    parser = argparse.ArgumentParser(
        description="Plot action-rate stats (mean/max and their time averages)."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="action_rate_stats_sim.csv",
        help="Path to action rate stats csv/txt file (default: action_rate_stats_sim.csv)",
    )
    parser.add_argument(
        "--save",
        default="",
        help="Optional output image path. If omitted, shows interactive window.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path(__file__).resolve().parent / input_path

    (
        steps,
        mean_abs,
        max_abs,
        mean_abs_time_avg,
        max_abs_time_avg,
        mean_abs_overall,
        max_abs_overall,
    ) = load_action_rate_stats(input_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    # Expand to a contiguous integer step window based on the data itself.
    # This supports arbitrary logging ranges (e.g., 200..700, 200..2200).
    use_fixed_window = steps.size > 0
    if use_fixed_window:
        step_start = int(np.floor(np.min(steps)))
        step_end = int(np.ceil(np.max(steps)))
        full_steps = np.arange(step_start, step_end + 1, dtype=float)
        mean_plot = np.full(full_steps.shape, np.nan, dtype=float)
        max_plot = np.full(full_steps.shape, np.nan, dtype=float)
        mean_time_avg_plot = np.full(full_steps.shape, np.nan, dtype=float)
        max_time_avg_plot = np.full(full_steps.shape, np.nan, dtype=float)

        step_to_idx = {int(s): i for i, s in enumerate(full_steps.astype(int))}
        for i, s in enumerate(steps.astype(int)):
            if s in step_to_idx:
                j = step_to_idx[s]
                mean_plot[j] = mean_abs[i]
                max_plot[j] = max_abs[i]
                mean_time_avg_plot[j] = mean_abs_time_avg[i]
                max_time_avg_plot[j] = max_abs_time_avg[i]

        # For readability, hold the final running-average value to step 700.
        last_valid_mean_avg = np.where(~np.isnan(mean_time_avg_plot))[0]
        if last_valid_mean_avg.size > 0:
            last_idx = last_valid_mean_avg[-1]
            mean_time_avg_plot[last_idx + 1 :] = mean_time_avg_plot[last_idx]
        last_valid_max_avg = np.where(~np.isnan(max_time_avg_plot))[0]
        if last_valid_max_avg.size > 0:
            last_idx = last_valid_max_avg[-1]
            max_time_avg_plot[last_idx + 1 :] = max_time_avg_plot[last_idx]

        x_plot = full_steps
    else:
        x_plot = steps
        mean_plot = mean_abs
        max_plot = max_abs
        mean_time_avg_plot = mean_abs_time_avg
        max_time_avg_plot = max_abs_time_avg

    # Left: mean action rate + time-average(mean action rate)
    axes[0].plot(x_plot, mean_plot, label="Mean action rate", linewidth=1.5)
    axes[0].plot(x_plot, mean_time_avg_plot, label="Time-avg mean", linewidth=2.0)
    axes[0].axhline(mean_abs_overall, linestyle="--", linewidth=1.5, label=f"Overall mean={mean_abs_overall:.4f}")
    axes[0].set_title("Mean Action Rate")
    axes[0].set_xlabel("Policy step")
    axes[0].set_ylabel("Abs action delta")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Right: max action rate + time-average(max action rate)
    axes[1].plot(x_plot, max_plot, label="Max action rate", linewidth=1.5)
    axes[1].plot(x_plot, max_time_avg_plot, label="Time-avg max", linewidth=2.0)
    axes[1].axhline(max_abs_overall, linestyle="--", linewidth=1.5, label=f"Overall mean max={max_abs_overall:.4f}")
    axes[1].set_title("Max Action Rate")
    axes[1].set_xlabel("Policy step")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Keep the contiguous step window visible and mark the last available real sample.
    if use_fixed_window:
        axes[0].set_xlim(full_steps[0], full_steps[-1])
        axes[1].set_xlim(full_steps[0], full_steps[-1])
        last_step = float(np.max(steps))
        axes[0].axvline(last_step, color="gray", linestyle=":", linewidth=1.0)
        axes[1].axvline(last_step, color="gray", linestyle=":", linewidth=1.0)
        axes[0].text(last_step, axes[0].get_ylim()[1], f" last={int(last_step)}",
                     va="top", ha="left", fontsize=9, color="gray")
        axes[1].text(last_step, axes[1].get_ylim()[1], f" last={int(last_step)}",
                     va="top", ha="left", fontsize=9, color="gray")

    fig.suptitle(f"Action Rate Stats: {input_path.name} (N={len(steps)})")
    fig.tight_layout()

    if args.save:
        save_path = Path(args.save)
        if not save_path.is_absolute():
            save_path = Path.cwd() / save_path
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()


