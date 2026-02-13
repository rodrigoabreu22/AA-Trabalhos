#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# Output folders
def ensure_dirs(out_root: str) -> Dict[str, str]:
    paths = {
        "root": out_root,
        "fig": os.path.join(out_root, "figures"),
        "tab": os.path.join(out_root, "tables"),
        "snip": os.path.join(out_root, "latex_snippets"),
        "raw": os.path.join(out_root, "raw"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


# Data loading
def load_release_year_stream(csv_path: str) -> List[int]:
    df = pd.read_csv(csv_path)
    if "release_year" not in df.columns:
        raise ValueError("Missing column release_year in CSV.")
    years = pd.to_numeric(df["release_year"], errors="coerce").dropna().astype(int).tolist()
    if not years:
        raise ValueError("No valid release_year values after cleaning.")
    return years


# Memory estimate
def deep_getsizeof(obj: Any, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(deep_getsizeof(k, seen) + deep_getsizeof(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(deep_getsizeof(i, seen) for i in obj)

    return size


# Exact counting
def exact_counts(stream: Iterable[int]) -> Dict[int, int]:
    return dict(Counter(stream))


# Approx counter: Fixed probability p=1/2
def fixed_probability_counter(stream: Iterable[int], p: float, seed: int) -> Dict[int, float]:
    rng = random.Random(seed)
    sampled = Counter()
    for x in stream:
        if rng.random() < p:
            sampled[x] += 1
    return {k: v / p for k, v in sampled.items()}


# Frequent-Count (Misra-Gries)
def misra_gries(stream: Iterable[int], k: int) -> Dict[int, int]:
    if k < 2:
        raise ValueError("k must be >= 2")
    counters: Dict[int, int] = {}
    for x in stream:
        if x in counters:
            counters[x] += 1
        elif len(counters) < k - 1:
            counters[x] = 1
        else:
            to_del = []
            for key in list(counters.keys()):
                counters[key] -= 1
                if counters[key] == 0:
                    to_del.append(key)
            for key in to_del:
                del counters[key]
    return counters


def second_pass_counts(stream: Iterable[int], candidates: Iterable[int]) -> Dict[int, int]:
    cand = set(candidates)
    out = {c: 0 for c in cand}
    for x in stream:
        if x in out:
            out[x] += 1
    return out


# Metrics
@dataclass
class TrialMetrics:
    mae: float
    mre_pct: float
    spearman: float


def ranks_with_ties(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)

    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    if _HAS_SCIPY:
        return float(spearmanr(a, b).correlation)
    ra = ranks_with_ties(a)
    rb = ranks_with_ties(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (np.sqrt((ra * ra).sum()) * np.sqrt((rb * rb).sum()))
    if denom == 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def align_to_exact(exact: Dict[int, int], est: Dict[int, float], fill: float = 0.0) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    items = sorted(exact.keys())
    exact_arr = np.array([exact[i] for i in items], dtype=float)
    est_arr = np.array([est.get(i, fill) for i in items], dtype=float)
    return exact_arr, est_arr, items


def trial_metrics(exact_arr: np.ndarray, est_arr: np.ndarray) -> TrialMetrics:
    abs_err = np.abs(est_arr - exact_arr)
    mae = float(abs_err.mean())
    rel = abs_err / np.maximum(exact_arr, 1.0)
    mre_pct = float(rel.mean() * 100.0)
    rho = spearman_corr(exact_arr, est_arr)
    return TrialMetrics(mae=mae, mre_pct=mre_pct, spearman=rho)


def top_n_from_counts(counts: Dict[int, float], n: int) -> List[Tuple[int, float]]:
    return sorted(((k, float(v)) for k, v in counts.items()), key=lambda kv: kv[1], reverse=True)[:n]


def to_percent(count: float, total: float) -> float:
    return 100.0 * (count / total) if total > 0 else 0.0


# LaTeX table helpers
def df_to_latex_booktabs(
    df: pd.DataFrame,
    caption: str,
    label: str,
    column_format: str | None = None
) -> str:
    latex = df.to_latex(
        index=False,
        escape=True,
        column_format=column_format,
        longtable=False
    )
    out = []
    out.append("\\begin{table}[!ht]\n\\centering")
    out.append(latex.strip())
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def save_text(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# Plot helpers (PNG)
def savefig_png(path: str, dpi: int = 250):
    plt.tight_layout()
    plt.savefig(path, format="png", dpi=dpi)
    plt.close()


def plot_cumulative_distribution(exact: Dict[int, int], out_png: str):
    counts = np.array(sorted(exact.values(), reverse=True), dtype=float)
    cum = np.cumsum(counts)
    total = cum[-1]
    y = 100.0 * cum / total
    x = np.arange(1, len(counts) + 1)

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Number of items (ranked by frequency)")
    plt.ylabel("Cumulative percentage of occurrences (%)")
    plt.title("Cumulative distribution of release_year frequencies")
    plt.xscale("log")
    plt.ylim(0, 100)
    savefig_png(out_png)


def plot_trials_line(y: List[float], ylabel: str, title: str, out_png: str):
    x = list(range(1, len(y) + 1))
    mean = float(np.mean(y)) if y else float("nan")

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.axhline(mean, linestyle="--")
    plt.xlabel("Trial number")
    plt.ylabel(ylabel)
    plt.title(title)
    savefig_png(out_png)


def plot_fc_errors(df_fc: pd.DataFrame, out_png: str):
    plt.figure()
    plt.plot(df_fc["n"], df_fc["mae_all"], marker="o", label="MAE (all items)")
    plt.plot(df_fc["n"], df_fc["mre_pct_all"], marker="o", label="MRE% (all items)")
    plt.xlabel("n (top-n target)")
    plt.ylabel("Error")
    plt.title("Frequent-Count errors vs n")
    plt.legend()
    savefig_png(out_png)


def plot_memory_vs_k(df_fc: pd.DataFrame, out_png: str):
    plt.figure()
    plt.plot(df_fc["k"], df_fc["mem_kb"], marker="o")
    plt.xlabel("k parameter (max counters + 1)")
    plt.ylabel("Memory (KB)")
    plt.title("Frequent-Count memory consumption vs k")
    savefig_png(out_png)


def plot_tracked_vs_max(df_fc: pd.DataFrame, out_png: str):
    plt.figure()
    plt.plot(df_fc["k"], df_fc["tracked"], marker="o", label="Tracked items")
    plt.plot(df_fc["k"], (df_fc["k"] - 1), linestyle="--", label="Theoretical max (k-1)")
    plt.xlabel("k parameter")
    plt.ylabel("Number of items")
    plt.title("Actual items tracked vs theoretical maximum")
    plt.legend()
    savefig_png(out_png)


def plot_time_comparison(metrics_df: pd.DataFrame, out_png: str):
    plt.figure()
    plt.bar(metrics_df["algorithm"], metrics_df["time_ms"])
    plt.xlabel("Algorithm")
    plt.ylabel("Execution time (ms)")
    plt.title("Execution time comparison")
    savefig_png(out_png)


def plot_pareto(df_points: pd.DataFrame, out_png: str):
    plt.figure()
    plt.scatter(df_points["mem_kb"], df_points["spearman"], s=60)
    for _, r in df_points.iterrows():
        plt.text(r["mem_kb"], r["spearman"], str(r["label"]), fontsize=8)
    plt.xlabel("Memory usage (KB)")
    plt.ylabel("Spearman rank correlation")
    plt.title("Memory-accuracy trade-off")
    savefig_png(out_png)


# Main runner
def run(csv_path: str, out_root: str, p: float, trials: int, topn_list: List[int], k_multiplier: int, seed: int):
    paths = ensure_dirs(out_root)

    stream = load_release_year_stream(csv_path)
    N = len(stream)

    # Exact
    t0 = time.perf_counter()
    exact = exact_counts(stream)
    t_exact_ms = (time.perf_counter() - t0) * 1000.0
    mem_exact_kb = deep_getsizeof(exact) / 1024.0

    # Table: Top-10 exact
    top10 = top_n_from_counts({k: float(v) for k, v in exact.items()}, 10)
    df_top10 = pd.DataFrame(
        [{
            "Rank": rank,
            "release_year": int(item),
            "Count": int(cnt),
            "Perc.": f"{to_percent(cnt, N):.2f}\\%"
        } for rank, (item, cnt) in enumerate(top10, start=1)]
    )
    save_text(
        os.path.join(paths["tab"], "table_top10_exact.tex"),
        df_to_latex_booktabs(df_top10, "Top-10 Most Frequent release\\_year (Exact)", "tab:top10_exact")
    )

    # Figure: cumulative distribution
    plot_cumulative_distribution(exact, os.path.join(paths["fig"], "fig_cumulative_distribution.png"))

    # Fixed probability trials
    trial_rows = []
    maes, mres, rhos = [], [], []

    t0 = time.perf_counter()
    for i in range(trials):
        est = fixed_probability_counter(stream, p=p, seed=seed + i)
        exact_arr, est_arr, _ = align_to_exact(exact, est, fill=0.0)
        m = trial_metrics(exact_arr, est_arr)
        maes.append(m.mae)
        mres.append(m.mre_pct)
        rhos.append(m.spearman)
        trial_rows.append({"trial": i + 1, "mae": m.mae, "mre_pct": m.mre_pct, "spearman": m.spearman})
    t_fixed_total_ms = (time.perf_counter() - t0) * 1000.0

    est_rep = fixed_probability_counter(stream, p=p, seed=seed)
    mem_fixed_kb = deep_getsizeof(est_rep) / 1024.0

    # --- NEW: Table Top-10 Fixed Probability (using representative run) ---
    top10_fp = top_n_from_counts(est_rep, 10)
    df_top10_fp = pd.DataFrame(
        [{
            "Rank": rank,
            "release_year": int(item),
            "Est. Count": f"{cnt:.2f}",
            "Perc.": f"{to_percent(cnt, N):.2f}\\%"
        } for rank, (item, cnt) in enumerate(top10_fp, start=1)]
    )
    save_text(
        os.path.join(paths["tab"], "table_top10_fixedprob.tex"),
        df_to_latex_booktabs(df_top10_fp, f"Top-10 Est. (Fixed Prob p={p})", "tab:top10_fixedprob")
    )
    # ---------------------------------------------------------------------

    df_trials = pd.DataFrame(trial_rows)
    df_trials.to_csv(os.path.join(paths["raw"], "fixedprob_trials.csv"), index=False)

    plot_trials_line(maes, "Mean absolute error", f"FixedProb p={p}: MAE across trials", os.path.join(paths["fig"], "fig_fixedprob_mae_trials.png"))
    plot_trials_line(mres, "Mean relative error (%)", f"FixedProb p={p}: MRE across trials", os.path.join(paths["fig"], "fig_fixedprob_mre_trials.png"))
    plot_trials_line(rhos, "Spearman correlation", f"FixedProb p={p}: Spearman across trials", os.path.join(paths["fig"], "fig_fixedprob_spearman_trials.png"))

    fixed_summary = {
        "algorithm": f"FixedProb(p={p})",
        "time_ms": t_fixed_total_ms / max(trials, 1),  # avg per run
        "mem_kb": mem_fixed_kb,
        "mae_all": float(np.mean(maes)),
        "mre_pct_all": float(np.mean(mres)),
        "spearman": float(np.mean(rhos)),
    }

    # Frequent-Count sweep
    fc_rows = []
    for n in topn_list:
        k = max(2, k_multiplier * n)

        t0 = time.perf_counter()
        counters = misra_gries(stream, k=k)
        t_stream_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        cand_exact = second_pass_counts(stream, counters.keys())
        t_second_ms = (time.perf_counter() - t1) * 1000.0

        est_all = {item: float(cand_exact.get(item, 0)) for item in exact.keys()}
        exact_arr, est_arr, _ = align_to_exact(exact, est_all, fill=0.0)
        m = trial_metrics(exact_arr, est_arr)

        mem_kb = deep_getsizeof(counters) / 1024.0
        tracked = len(counters)

        exact_topn = [x for x, _ in top_n_from_counts({k: float(v) for k, v in exact.items()}, n)]
        fc_topn = [x for x, _ in top_n_from_counts({k: float(v) for k, v in cand_exact.items()}, n)]
        overlap = len(set(exact_topn) & set(fc_topn))
        jaccard = overlap / len(set(exact_topn) | set(fc_topn)) if n > 0 else 1.0

        fc_rows.append({
            "n": n,
            "k": k,
            "tracked": tracked,
            "mem_kb": mem_kb,
            "time_ms_stream": t_stream_ms,
            "time_ms_secondpass": t_second_ms,
            "time_ms_total": t_stream_ms + t_second_ms,
            "mae_all": m.mae,
            "mre_pct_all": m.mre_pct,
            "spearman": m.spearman,
            "topn_overlap": overlap,
            "topn_jaccard": jaccard,
        })

    df_fc = pd.DataFrame(fc_rows)
    df_fc.to_csv(os.path.join(paths["raw"], "frequentcount_params.csv"), index=False)

    plot_fc_errors(df_fc, os.path.join(paths["fig"], "fig_fc_errors_vs_n.png"))
    plot_memory_vs_k(df_fc, os.path.join(paths["fig"], "fig_fc_memory_vs_k.png"))
    plot_tracked_vs_max(df_fc, os.path.join(paths["fig"], "fig_fc_tracked_vs_max.png"))

    # Choose FC config for summary (largest n)
    best_fc = df_fc.sort_values(["n"], ascending=False).iloc[0].to_dict()
    
    # --- NEW: Table Top-10 Frequent Count (using best/largest configuration) ---
    best_k = int(best_fc['k'])
    counters_best = misra_gries(stream, k=best_k)
    top10_fc = top_n_from_counts({k: float(v) for k, v in counters_best.items()}, 10)
    
    df_top10_fc = pd.DataFrame(
        [{
            "Rank": rank,
            "release_year": int(item),
            "Count": int(cnt),
            "Perc.": f"{to_percent(cnt, N):.2f}\\%"
        } for rank, (item, cnt) in enumerate(top10_fc, start=1)]
    )
    save_text(
        os.path.join(paths["tab"], "table_top10_frequentcount.tex"),
        df_to_latex_booktabs(df_top10_fc, f"Top-10 (Frequent-Count k={best_k})", "tab:top10_fc")
    )
    # -------------------------------------------------------------------------

    fc_summary = {
        "algorithm": f"FrequentCount(k={int(best_fc['k'])})",
        "time_ms": float(best_fc["time_ms_total"]),
        "mem_kb": float(best_fc["mem_kb"]),
        "mae_all": float(best_fc["mae_all"]),
        "mre_pct_all": float(best_fc["mre_pct_all"]),
        "spearman": float(best_fc["spearman"]),
    }

    exact_summary = {
        "algorithm": "Exact",
        "time_ms": t_exact_ms,
        "mem_kb": mem_exact_kb,
        "mae_all": 0.0,
        "mre_pct_all": 0.0,
        "spearman": 1.0,
    }

    df_summary = pd.DataFrame([exact_summary, fixed_summary, fc_summary])
    df_summary.to_csv(os.path.join(paths["raw"], "summary_metrics.csv"), index=False)

    df_summary_fmt = df_summary.copy()
    df_summary_fmt["time_ms"] = df_summary_fmt["time_ms"].map(lambda x: f"{x:.2f}")
    df_summary_fmt["mem_kb"] = df_summary_fmt["mem_kb"].map(lambda x: f"{x:.2f}")
    df_summary_fmt["mae_all"] = df_summary_fmt["mae_all"].map(lambda x: f"{x:.4f}")
    df_summary_fmt["mre_pct_all"] = df_summary_fmt["mre_pct_all"].map(lambda x: f"{x:.2f}\\%")
    df_summary_fmt["spearman"] = df_summary_fmt["spearman"].map(lambda x: f"{x:.4f}")

    save_text(
        os.path.join(paths["tab"], "table_summary_metrics.tex"),
        df_to_latex_booktabs(df_summary_fmt, "Algorithm comparison summary (Netflix release\\_year)", "tab:summary_metrics")
    )

    plot_time_comparison(df_summary, os.path.join(paths["fig"], "fig_time_comparison.png"))

    df_pareto = pd.DataFrame([
        {"mem_kb": mem_exact_kb, "spearman": 1.0, "label": "Exact"},
        {"mem_kb": mem_fixed_kb, "spearman": float(np.mean(rhos)), "label": f"FixedProb p={p}"},
        {"mem_kb": float(best_fc["mem_kb"]), "spearman": float(best_fc["spearman"]), "label": f"FC k={int(best_fc['k'])}"},
    ])
    plot_pareto(df_pareto, os.path.join(paths["fig"], "fig_pareto_memory_accuracy.png"))

    # LaTeX snippets for figures (PNG)
    fig_files = [
        ("fig_cumulative_distribution.png", "Cumulative distribution of release\\_year frequencies.", "fig:cumdist"),
        ("fig_fixedprob_mae_trials.png", f"FixedProb p={p}: mean absolute error across trials.", "fig:fp_mae"),
        ("fig_fixedprob_mre_trials.png", f"FixedProb p={p}: mean relative error across trials.", "fig:fp_mre"),
        ("fig_fixedprob_spearman_trials.png", f"FixedProb p={p}: Spearman rank correlation across trials.", "fig:fp_spear"),
        ("fig_fc_errors_vs_n.png", "Frequent-Count errors vs n.", "fig:fc_errors"),
        ("fig_fc_memory_vs_k.png", "Frequent-Count memory usage vs k.", "fig:fc_mem"),
        ("fig_fc_tracked_vs_max.png", "Items tracked vs theoretical maximum (k-1).", "fig:fc_tracked"),
        ("fig_time_comparison.png", "Execution time comparison across algorithms.", "fig:time"),
        ("fig_pareto_memory_accuracy.png", "Memory-accuracy trade-off.", "fig:pareto"),
    ]

    snippets = []
    for fn, cap, lab in fig_files:
        snippets.append(
            "\n".join([
                "\\begin{figure}[!ht]",
                "\\centering",
                f"\\includegraphics[width=0.95\\linewidth]{{figures/{fn}}}",
                f"\\caption{{{cap}}}",
                f"\\label{{{lab}}}",
                "\\end{figure}",
                ""
            ])
        )
    save_text(os.path.join(paths["snip"], "figures.tex"), "\n".join(snippets))

    print(f"Done. Outputs written to: {out_root}")
    print(f"Figures (PNG): {paths['fig']}")
    print(f"Tables (TEX):  {paths['tab']}")
    print(f"Snippets:      {paths['snip']}")
    print(f"Raw CSV:       {paths['raw']}")


def parse_int_list(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="netflix_titles.csv", help="Path to netflix_titles.csv")
    ap.add_argument("--out", default="out", help="Output folder root")

    ap.add_argument("--p", type=float, default=0.5, help="Fixed probability (assigned p=0.5)")
    ap.add_argument("--trials", type=int, default=20, help="Number of trials for FixedProb")

    ap.add_argument("--topn", default="5,10,15,20,25,30,40,50", help="n values for Frequent-Count sweep")
    ap.add_argument("--k-mult", type=int, default=2, help="k = k_mult * n")
    ap.add_argument("--seed", type=int, default=113626, help="Base RNG seed")
    args = ap.parse_args()

    topn_list = parse_int_list(args.topn)

    run(
        csv_path=args.csv,
        out_root=args.out,
        p=args.p,
        trials=args.trials,
        topn_list=topn_list,
        k_multiplier=args.k_mult,
        seed=args.seed
    )


if __name__ == "__main__":
    main()