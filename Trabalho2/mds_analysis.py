import os
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# Helpers
def load_results(path="results_unified/full_experiment_suite.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")
    df = pd.read_csv(path)
    return df


def format_time(seconds: float) -> str:
    if seconds is None or pd.isna(seconds):
        return "N/A"
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    if seconds < 3600:
        return f"{seconds / 60:.2f} min"
    if seconds < 86400:
        return f"{seconds / 3600:.2f} h"
    if seconds < 31536000:
        return f"{seconds / 86400:.2f} days"
    return f"{seconds / 31536000:.2f} years"


def exponential_model(n, a, b):
    return a * np.power(b, n)


def polynomial_model(n, a, b):
    return a * np.power(n, b)


# Plots for random graphs (main experiments)
def plot_execution_times_random(df, out_folder="results_unified"):
    os.makedirs(out_folder, exist_ok=True)
    random_df = df[df["source"] == "random"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Execution time vs number of vertices (random graphs)", fontsize=16)

    densities = sorted(random_df["density"].unique())
    algo_colors = {
        "exact_time": "tab:blue",
        "greedy_time": "tab:orange",
        "random_time": "tab:green",
        "randgreedy_time": "tab:red",
    }

    for idx, d in enumerate(densities):
        ax = axes[idx // 2, idx % 2]
        sub = random_df[random_df["density"] == d].sort_values("n")

        if sub["exact_time"].notna().any():
            ax.plot(sub["n"], sub["exact_time"], "o-", label="Exact", color=algo_colors["exact_time"])
        ax.plot(sub["n"], sub["greedy_time"], "s-", label="Greedy", color=algo_colors["greedy_time"])
        ax.plot(sub["n"], sub["random_time"], "^-", label="Random search", color=algo_colors["random_time"])
        ax.plot(sub["n"], sub["randgreedy_time"], "v-", label="Randomized greedy", color=algo_colors["randgreedy_time"])

        ax.set_yscale("log")
        ax.set_xlabel("Number of vertices n")
        ax.set_ylabel("Execution time (s)")
        ax.set_title(f"Density = {d:.3f}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(out_folder, "random_execution_times.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_operations_random(df, out_folder="results_unified"):
    os.makedirs(out_folder, exist_ok=True)
    random_df = df[df["source"] == "random"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Number of basic operations (random graphs)", fontsize=16)

    densities = sorted(random_df["density"].unique())

    for idx, d in enumerate(densities):
        ax = axes[idx // 2, idx % 2]
        sub = random_df[random_df["density"] == d].sort_values("n")

        if sub["exact_ops"].notna().any():
            ax.plot(sub["n"], sub["exact_ops"], "o-", label="Exact ops")
        ax.plot(sub["n"], sub["greedy_ops"], "s-", label="Greedy ops")
        ax.plot(sub["n"], sub["random_ops"], "^-", label="Random search ops")
        ax.plot(sub["n"], sub["randgreedy_ops"], "v-", label="Randomized greedy ops")

        ax.set_yscale("log")
        ax.set_xlabel("Number of vertices n")
        ax.set_ylabel("Operations count")
        ax.set_title(f"Density = {d:.3f}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(out_folder, "random_operations.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_configurations_random(df, out_folder="results_unified"):
    os.makedirs(out_folder, exist_ok=True)
    random_df = df[df["source"] == "random"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Configurations tested (random graphs)", fontsize=16)

    densities = sorted(random_df["density"].unique())

    for idx, d in enumerate(densities):
        ax = axes[idx // 2, idx % 2]
        sub = random_df[random_df["density"] == d].sort_values("n")

        if sub["exact_configs"].notna().any():
            ax.plot(sub["n"], sub["exact_configs"], "o-", label="Exact configs")
        ax.plot(sub["n"], sub["random_configs"], "^-", label="Random search configs")
        ax.plot(sub["n"], sub["randgreedy_configs"], "v-", label="Randomized greedy configs")

        ax.set_yscale("log")
        ax.set_xlabel("Number of vertices n")
        ax.set_ylabel("Configurations tested")
        ax.set_title(f"Density = {d:.3f}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(out_folder, "random_configs.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_solution_sizes_random(df, out_folder="results_unified"):
    os.makedirs(out_folder, exist_ok=True)
    random_df = df[df["source"] == "random"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Solution sizes (random graphs)", fontsize=16)

    densities = sorted(random_df["density"].unique())

    for idx, d in enumerate(densities):
        ax = axes[idx // 2, idx % 2]
        sub = random_df[random_df["density"] == d].sort_values("n")

        if sub["exact_size"].notna().any():
            ax.plot(sub["n"], sub["exact_size"], "o-", label="Exact size")
        ax.plot(sub["n"], sub["greedy_size"], "s-", label="Greedy size")
        ax.plot(sub["n"], sub["random_size"], "^-", label="Random search size")
        ax.plot(sub["n"], sub["randgreedy_size"], "v-", label="Randomized greedy size")

        ax.set_xlabel("Number of vertices n")
        ax.set_ylabel("Dominating set size")
        ax.set_title(f"Density = {d:.3f}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(out_folder, "random_solution_sizes.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# Accuracy vs exact solutions
def accuracy_plots_random(df, out_folder="results_unified"):
    os.makedirs(out_folder, exist_ok=True)
    random_df = df[(df["source"] == "random") & df["exact_size"].notna()].copy()

    if random_df.empty:
        print("No exact solutions recorded for random graphs. Skipping accuracy plots.")
        return

    random_df["greedy_ratio"] = random_df["greedy_size"] / random_df["exact_size"]
    random_df["random_ratio"] = random_df["random_size"] / random_df["exact_size"]
    random_df["randgreedy_ratio"] = random_df["randgreedy_size"] / random_df["exact_size"]

    # ratio vs n
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]

    for d in sorted(random_df["density"].unique()):
        sub = random_df[random_df["density"] == d].sort_values("n")
        ax.plot(sub["n"], sub["greedy_ratio"], "s-", label=f"Greedy, d={d}")
        ax.plot(sub["n"], sub["random_ratio"], "^-", label=f"Random, d={d}")
        ax.plot(sub["n"], sub["randgreedy_ratio"], "v-", label=f"RandGreedy, d={d}")

    ax.axhline(1.0, color="black", linestyle="--", alpha=0.6)
    ax.set_xlabel("Number of vertices n")
    ax.set_ylabel("Approximation ratio (size / optimal)")
    ax.set_title("Approximation ratio vs n (random graphs)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # histogram of ratios
    ax = axes[1]
    ax.hist(random_df["greedy_ratio"], bins=20, alpha=0.5, label="Greedy")
    ax.hist(random_df["random_ratio"], bins=20, alpha=0.5, label="Random search")
    ax.hist(random_df["randgreedy_ratio"], bins=20, alpha=0.5, label="Randomized greedy")
    ax.set_xlabel("Approximation ratio (size / optimal)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of approximation ratios (random graphs)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    path = os.path.join(out_folder, "random_accuracy.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # simple text summary
    print("\n=== Accuracy summary (random graphs with exact) ===")
    for algo in ["greedy_ratio", "random_ratio", "randgreedy_ratio"]:
        mean = random_df[algo].mean()
        std = random_df[algo].std()
        worst = random_df[algo].max()
        best = random_df[algo].min()
        print(f"{algo}: mean={mean:.3f}, std={std:.3f}, min={best:.3f}, max={worst:.3f}")
    print("Times with optimal solution (ratio == 1.0):")
    for algo in ["greedy_ratio", "random_ratio", "randgreedy_ratio"]:
        count_opt = (random_df[algo] == 1.0).sum()
        print(f"  {algo}: {count_opt} / {len(random_df)}")


# SW and benchmark graphs summary
def summary_sw_and_bench(df):
    sw_df = df[df["source"] == "SW"].copy()
    bench_df = df[df["source"].str.startswith("benchmark_")].copy()

    print("\n=== SW_ALGUNS_GRAFOS summary ===")
    if sw_df.empty:
        print("No SW graphs found.")
    else:
        cols = ["filename", "n", "edges",
                "greedy_size", "random_size", "randgreedy_size",
                "greedy_time", "random_time", "randgreedy_time"]
        print(sw_df[cols].to_string(index=False))

    print("\n=== Benchmark graphs summary ===")
    if bench_df.empty:
        print("No benchmark graphs found.")
    else:
        cols = ["source", "name", "n", "edges",
                "exact_size", "greedy_size", "random_size", "randgreedy_size",
                "exact_time", "greedy_time", "random_time", "randgreedy_time"]
        print(bench_df[cols].to_string(index=False))


# Complexity model fitting and estimation
def fit_and_estimate(df, out_folder="results_unified", time_limit=60.0):
    os.makedirs(out_folder, exist_ok=True)
    random_df = df[df["source"] == "random"].copy()

    if random_df.empty:
        print("No random graphs data for fitting.")
        return

    max_n = random_df["n"].max()
    print(f"\nMax n observed in random graphs: {max_n}")

    # choose 50 percent density as representative for avg case
    d_mid = 0.5
    mid_df = random_df[(random_df["density"] == d_mid)].copy()
    mid_df = mid_df.sort_values("n")

    # filter positive times
    mid_df = mid_df[(mid_df["random_time"] > 0) & (mid_df["randgreedy_time"] > 0)]

    if mid_df.empty:
        print("Not enough data at density 0.5 for fitting randomized algorithms.")
        return

    n_vals = mid_df["n"].values

    # exact exponential fit (worst case) using low density 0.125
    print("\n=== Exponential fit for exact algorithm (worst case) ===")
    low_d = 0.125
    exact_df = random_df[(random_df["density"] == low_d) &
                         random_df["exact_time"].notna() &
                         (random_df["exact_time"] > 0)].copy()
    if exact_df.empty:
        print("No exact timing data at density 0.125. Skipping exact model.")
        exact_model = None
    else:
        x = exact_df["n"].values
        y = exact_df["exact_time"].values
        log_y = np.log(y)
        coeffs = np.polyfit(x, log_y, 1)
        ln_b = coeffs[0]
        ln_a = coeffs[1]
        a_exact = math.exp(ln_a)
        b_exact = math.exp(ln_b)
        print(f"Exact model at d={low_d}: T(n) = {a_exact:.3e} * {b_exact:.3f}^n")
        exact_model = (a_exact, b_exact)

        # estimate n where time ~ time_limit
        est_n = x.max()
        est_time = y[-1]
        while est_time < time_limit and est_n < 200:
            est_n += 1
            est_time = exponential_model(est_n, a_exact, b_exact)
        max_feasible = est_n - 1
        print(f"Approx max n for exact (under {time_limit}s): {max_feasible}")
        print(f"Estimated time at n={max_feasible}: {format_time(exponential_model(max_feasible, a_exact, b_exact))}")

    # polynomial fit for randomized search
    print("\n=== Polynomial fit for Random search ===")
    y_rand = mid_df["random_time"].values
    log_x = np.log(n_vals)
    log_y = np.log(y_rand)
    coeffs = np.polyfit(log_x, log_y, 1)
    b_rand = coeffs[0]
    ln_a_rand = coeffs[1]
    a_rand = math.exp(ln_a_rand)
    print(f"Random search model (density {d_mid}): T(n) = {a_rand:.3e} * n^{b_rand:.2f}")

    # polynomial fit for randomized greedy
    print("\n=== Polynomial fit for Randomized greedy ===")
    y_rg = mid_df["randgreedy_time"].values
    log_y2 = np.log(y_rg)
    coeffs2 = np.polyfit(log_x, log_y2, 1)
    b_rg = coeffs2[0]
    ln_a_rg = coeffs2[1]
    a_rg = math.exp(ln_a_rg)
    print(f"Randomized greedy model (density {d_mid}): T(n) = {a_rg:.3e} * n^{b_rg:.2f}")

    # Plot fitted curves for documentation
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(n_vals, y_rand, label="Random search (data)", marker="^")
    ax.scatter(n_vals, y_rg, label="Randomized greedy (data)", marker="v")

    n_fit = np.linspace(n_vals.min(), n_vals.max(), 100)
    ax.plot(n_fit, polynomial_model(n_fit, a_rand, b_rand), label="Random search fit", linestyle="--")
    ax.plot(n_fit, polynomial_model(n_fit, a_rg, b_rg), label="Randomized greedy fit", linestyle="--")

    ax.set_yscale("log")
    ax.set_xlabel("Number of vertices n")
    ax.set_ylabel("Execution time (s)")
    ax.set_title(f"Fitted complexity models at density = {d_mid}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = os.path.join(out_folder, "complexity_fits_randomized.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # estimate time for much larger instances
    print("\n=== Estimated times for large n (randomized algorithms) ===")
    targets = [100, 500, 1000, 5000, 10000, 100000]
    print(f"{'n':>8} | {'Random search':>20} | {'Rand greedy':>20}")
    print("-" * 60)
    for n in targets:
        t_rand = polynomial_model(n, a_rand, b_rand)
        t_rg = polynomial_model(n, a_rg, b_rg)
        print(f"{n:8d} | {format_time(t_rand):>20} | {format_time(t_rg):>20}")

    # largest n from the data where randomized is still very fast
    max_n_random = random_df["n"].max()
    max_time_rand = random_df["random_time"].max()
    max_time_rg = random_df["randgreedy_time"].max()
    print("\nMax n tested for randomized algorithms:", max_n_random)
    print("Worst observed time for random search:", format_time(max_time_rand))
    print("Worst observed time for randomized greedy:", format_time(max_time_rg))


# Summary statistics
def summary_statistics(df):
    print("\n=== Basic summary statistics (random graphs) ===")
    random_df = df[df["source"] == "random"].copy()
    if random_df.empty:
        print("No random graphs.")
        return

    # average time by n
    time_stats = random_df.groupby("n")[["exact_time", "greedy_time",
                                         "random_time", "randgreedy_time"]].mean()
    print("\nAverage execution time by n:")
    print(time_stats.to_string())

    # accuracy by density where exact exists
    exact_df = random_df[random_df["exact_size"].notna()].copy()
    if not exact_df.empty:
        exact_df["greedy_ratio"] = exact_df["greedy_size"] / exact_df["exact_size"]
        exact_df["random_ratio"] = exact_df["random_size"] / exact_df["exact_size"]
        exact_df["randgreedy_ratio"] = exact_df["randgreedy_size"] / exact_df["exact_size"]
        acc_stats = exact_df.groupby("density")[["greedy_ratio", "random_ratio", "randgreedy_ratio"]].agg(
            ["mean", "std", "min", "max"]
        )
        print("\nApproximation ratio stats by density (only where exact known):")
        print(acc_stats)


# Main
def main():
    print("=" * 80)
    print("MINIMUM DOMINATING SET - RANDOMIZED ALGORITHMS ANALYSIS")
    print("=" * 80)

    df = load_results()

    print(f"\nLoaded {len(df)} rows")
    print("Sources:", df["source"].unique())
    if "n" in df:
        print("n range:", df["n"].min(), "to", df["n"].max())

    # main random graph analysis
    print("\nGenerating plots for random graphs...")
    plot_execution_times_random(df)
    plot_operations_random(df)
    plot_configurations_random(df)
    plot_solution_sizes_random(df)
    accuracy_plots_random(df)

    # SW and benchmark summary
    summary_sw_and_bench(df)

    # complexity fits and estimations
    fit_and_estimate(df)

    # text summary for report
    summary_statistics(df)

    print("\nAnalysis complete. Figures saved in 'results_unified' folder.")


if __name__ == "__main__":
    main()
