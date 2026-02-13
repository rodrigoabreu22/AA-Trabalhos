import time
import random
import json
import os
import logging
from itertools import combinations
from typing import Set

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# LOGGING SETUP
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 1. EXACT + GREEDY SOLVER
class MinimumDominatingSet:

    def __init__(self, G: nx.Graph):
        self.G = G
        self.n = G.number_of_nodes()
        self.exhaustive_ops = 0
        self.exhaustive_configs = 0
        self.greedy_ops = 0

    def is_dominating_set(self, S: Set[int]) -> bool:
        self.exhaustive_ops += 1
        for v in self.G.nodes():
            if v not in S:
                if not set(self.G.neighbors(v)).intersection(S):
                    return False
        return True

    def exhaustive_search(self, time_limit=30, max_configs=None):
        logger.info(f"    [Exact] Starting exhaustive search on N={self.n}...")
        start = time.time()
        nodes = list(self.G.nodes())
        self.exhaustive_ops = 0
        self.exhaustive_configs = 0

        for size in range(1, self.n + 1):
            for comb in combinations(nodes, size):

                if max_configs and self.exhaustive_configs >= max_configs:
                    logger.warning("    [Exact] Max configs reached. Aborting.")
                    return None, {
                        "size": None,
                        "ops": self.exhaustive_ops,
                        "configs_tested": self.exhaustive_configs,
                        "execution_time": time.time() - start,
                    }

                if time.time() - start > time_limit:
                    logger.warning(
                        f"    [Exact] Time limit ({time_limit}s) exceeded. Aborting."
                    )
                    return None, {
                        "size": None,
                        "ops": self.exhaustive_ops,
                        "configs_tested": self.exhaustive_configs,
                        "execution_time": time.time() - start,
                    }

                S = set(comb)
                self.exhaustive_configs += 1
                if self.is_dominating_set(S):
                    elapsed = time.time() - start
                    logger.info(
                        f"    [Exact] Found solution size {len(S)} in {elapsed:.4f}s"
                    )
                    return S, {
                        "size": len(S),
                        "ops": self.exhaustive_ops,
                        "configs_tested": self.exhaustive_configs,
                        "execution_time": elapsed,
                    }

        return None, {
            "size": None,
            "ops": self.exhaustive_ops,
            "configs_tested": self.exhaustive_configs,
            "execution_time": time.time() - start,
        }

    def greedy_heuristic(self):
        start = time.time()
        dominating = set()
        dominated = set()
        all_nodes = set(self.G.nodes())
        self.greedy_ops = 0

        while dominated != all_nodes:

            best_v = None
            best_gain = -1

            for v in all_nodes:
                self.greedy_ops += 1
                if v in dominating:
                    continue

                gain = len((set(self.G.neighbors(v)) | {v}) - dominated)

                if gain > best_gain:
                    best_gain = gain
                    best_v = v

            dominating.add(best_v)
            dominated |= (set(self.G.neighbors(best_v)) | {best_v})

        elapsed = time.time() - start
        return dominating, {
            "size": len(dominating),
            "ops": self.greedy_ops,
            "execution_time": elapsed,
        }

# 2. RANDOMIZED ALGORITHMS
class RandomizedMDS:

    def __init__(self, G: nx.Graph, seed=None):
        self.G = G
        self.n = G.number_of_nodes()
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.operations = 0
        self.configs_tested = 0
        self.tested = set()

    def is_dominating(self, S: Set[int]):
        self.operations += 1
        for v in self.G.nodes():
            if v not in S:
                if not set(self.G.neighbors(v)).intersection(S):
                    return False
        return True

    def mark(self, S: Set[int]):
        self.tested.add(tuple(sorted(S)))

    def seen(self, S: Set[int]):
        return tuple(sorted(S)) in self.tested

    def randomized_mds(self, iterations=200, time_limit=None):
        """
        Randomised search for a small dominating set.
    
        Strategy:
          - For k = 1..n, try up to `iterations` random subsets of size k.
          - Stop as soon as a dominating set is found.
          - Because k increases, the first dominating set found has
            minimum size among all sampled configurations.
        """
        self.operations = 0
        self.configs_tested = 0
        self.tested.clear()
    
        start = time.time()
        nodes = list(self.G.nodes())
        best = None
        best_size = self.n
    
        for k in range(1, self.n + 1):
            attempts = 0
    
            while attempts < iterations:
                
                if time_limit is not None and (time.time() - start) > time_limit:
                    elapsed = time.time() - start
                    return best if best is not None else set(nodes), {
                        "size": (best_size if best is not None else self.n),
                        "ops": self.operations,
                        "configs_tested": self.configs_tested,
                        "iterations": iterations,
                        "execution_time": elapsed,
                        "timeout": True,
                    }
    
                S = set(random.sample(nodes, k))
                attempts += 1  
    
                if self.seen(S):
                    continue
                
                self.mark(S)
                self.configs_tested += 1
    
                if self.is_dominating(S):
                    best = S
                    best_size = k
                    elapsed = time.time() - start
                    return best, {
                        "size": best_size,
                        "ops": self.operations,
                        "configs_tested": self.configs_tested,
                        "iterations": iterations,
                        "execution_time": elapsed,
                        "timeout": False,
                    }
    
        # If no dominating set was sampled, fall back to full V
        elapsed = time.time() - start
        return set(nodes), {
            "size": self.n,
            "ops": self.operations,
            "configs_tested": self.configs_tested,
            "iterations": iterations,
            "execution_time": elapsed,
            "timeout": False,
        }

    def randomized_greedy(self, randomness=0.3):
        self.operations = 0
        self.configs_tested = 0

        start = time.time()

        dominating = set()
        dominated = set()
        nodes = list(self.G.nodes())

        while len(dominated) < self.n:

            gains = []
            for v in nodes:
                if v in dominating:
                    continue
                self.operations += 1
                gains.append(
                    (len((set(self.G.neighbors(v)) | {v}) - dominated), v)
                )

            gains.sort(reverse=True)

            if random.random() < randomness:
                K = max(1, len(gains) // 3)
                _, chosen = random.choice(gains[:K])
            else:
                _, chosen = gains[0]

            dominating.add(chosen)
            dominated |= (set(self.G.neighbors(chosen)) | {chosen})
            self.configs_tested += 1

        elapsed = time.time() - start
        return dominating, {
            "size": len(dominating),
            "ops": self.operations,
            "configs_tested": self.configs_tested,
            "randomness": randomness,
            "execution_time": elapsed,
        }

# GRAPH GENERATION
class GraphGenerator:

    def __init__(self, seed):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def generate_graph(self, n, density):
        G = nx.Graph()
        G.add_nodes_from(range(n))

        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < density:
                    G.add_edge(u, v)
        return G

    def save_graph(self, G, path):
        data = {
            "nodes": list(G.nodes()),
            "edges": [list(e) for e in G.edges()],
        }
        with open(path, "w") as f:
            json.dump(data, f)

# VISUALIZATION
def visualize_solution(G, S, title, path):
    if G.number_of_nodes() > 200:
        return
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6, 6))
    nx.draw(
        G,
        pos,
        node_color=["red" if v in S else "lightblue" for v in G],
        with_labels=True,
    )
    plt.title(title)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# SW_ALGUNS_GRAFOS LOADER
class GraphUtils:

    @staticmethod
    def load_graph(path):
        with open(path) as f:
            lines = [l.strip() for l in f.readlines()]

        if lines[0] == "0" and lines[1] == "0":
            n = int(lines[2])
            m = int(lines[3])

            G = nx.Graph()
            G.add_nodes_from(range(n))

            for ln in lines[4:4 + m]:
                u, v = map(int, ln.split())
                G.add_edge(u, v)

            return G

        raise ValueError("Unknown format")

    @staticmethod
    def benchmark_graphs():
        return {
            "petersen": nx.petersen_graph(),
            "karate": nx.karate_club_graph(),
            "cycle20": nx.cycle_graph(20),
            "complete10": nx.complete_graph(10),
        }

# MAIN EXPERIMENT SUITE
def run_suite(
    student_number,
    max_random_n=40,
    sw_folder="SW_ALGUNS_GRAFOS",
    save_folder="results_unified",
):

    logger.info("========================================")
    logger.info("   STARTING UNIFIED EXPERIMENT SUITE    ")
    logger.info("========================================")

    os.makedirs(save_folder, exist_ok=True)
    os.makedirs("graphs", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)

    all_results = []
    generator = GraphGenerator(student_number)

    # RANDOM GRAPHS
    logger.info(">>> PHASE 1: Processing Random Graphs")

    densities = [0.125, 0.25, 0.50, 0.75]

    for n in range(4, max_random_n + 1):
        for d in densities:
            logger.info(f"--- Random Graph: n={n}, density={d} ---")

            # generate and save graph structure
            G = generator.generate_graph(n, d)
            graph_filename = f"random_n{n}_d{int(d*100)}.json"
            generator.save_graph(G, os.path.join("graphs", graph_filename))

            solver = MinimumDominatingSet(G)

            # Exact
            exact_sol, exact_stats = solver.exhaustive_search()

            # Greedy
            greedy_sol, greedy_stats = solver.greedy_heuristic()

            # Random search
            rsolver = RandomizedMDS(G)
            rand_sol, rand_stats = rsolver.randomized_mds(iterations=200)

            # Randomized greedy
            rsolver2 = RandomizedMDS(G)
            randg_sol, randg_stats = rsolver2.randomized_greedy(randomness=0.3)

            # visualizations for this random graph
            base = f"random_n{n}_d{int(d*100)}"
            if exact_sol is not None:
                visualize_solution(
                    G,
                    exact_sol,
                    f"Exact MDS (n={n}, d={d}, size={exact_stats['size']})",
                    os.path.join("visualizations", base + "_exact.png"),
                )
            visualize_solution(
                G,
                greedy_sol,
                f"Greedy MDS (n={n}, d={d}, size={greedy_stats['size']})",
                os.path.join("visualizations", base + "_greedy.png"),
            )
            visualize_solution(
                G,
                rand_sol,
                f"Random search MDS (n={n}, d={d}, size={rand_stats['size']})",
                os.path.join("visualizations", base + "_random.png"),
            )
            visualize_solution(
                G,
                randg_sol,
                f"Randomized greedy MDS (n={n}, d={d}, size={randg_stats['size']})",
                os.path.join("visualizations", base + "_randgreedy.png"),
            )

            all_results.append({
                "source": "random",
                "n": n,
                "density": d,
                "edges": G.number_of_edges(),

                "exact_size": exact_stats["size"],
                "exact_time": exact_stats["execution_time"],
                "exact_ops": exact_stats["ops"],
                "exact_configs": exact_stats["configs_tested"],

                "greedy_size": greedy_stats["size"],
                "greedy_time": greedy_stats["execution_time"],
                "greedy_ops": greedy_stats["ops"],

                "random_size": rand_stats["size"],
                "random_time": rand_stats["execution_time"],
                "random_ops": rand_stats["ops"],
                "random_configs": rand_stats["configs_tested"],
                "random_iterations": rand_stats.get("iterations"),

                "randgreedy_size": randg_stats["size"],
                "randgreedy_time": randg_stats["execution_time"],
                "randgreedy_ops": randg_stats["ops"],
                "randgreedy_configs": randg_stats["configs_tested"],
                "randgreedy_randomness": randg_stats.get("randomness"),
            })

    # SW ALGUNS GRAFOS
    logger.info(">>> PHASE 2: Processing SW_ALGUNS_GRAFOS")

    if not os.path.exists(sw_folder):
        logger.error(f"Folder {sw_folder} not found! Skipping Phase 2.")
    else:
        for file in os.listdir(sw_folder):
            if not file.endswith(".txt") or "SWlargeG" in file:
                continue

            logger.info(f"--- Processing File: {file} ---")

            try:
                G = GraphUtils.load_graph(os.path.join(sw_folder, file))
                solver = MinimumDominatingSet(G)

                greedy_sol, greedy_stats = solver.greedy_heuristic()

                rsolver = RandomizedMDS(G)
                rand_sol, rand_stats = rsolver.randomized_mds(iterations=2000)

                rsolver2 = RandomizedMDS(G)
                randg_sol, randg_stats = rsolver2.randomized_greedy(randomness=0.3)

                # visualizations (if graph not huge)
                name_no_ext = os.path.splitext(file)[0]
                base = f"SW_{name_no_ext}"
                visualize_solution(
                    G,
                    greedy_sol,
                    f"SW {file} - greedy (size={greedy_stats['size']})",
                    os.path.join("visualizations", base + "_greedy.png"),
                )
                visualize_solution(
                    G,
                    rand_sol,
                    f"SW {file} - random (size={rand_stats['size']})",
                    os.path.join("visualizations", base + "_random.png"),
                )
                visualize_solution(
                    G,
                    randg_sol,
                    f"SW {file} - randgreedy (size={randg_stats['size']})",
                    os.path.join("visualizations", base + "_randgreedy.png"),
                )

                all_results.append({
                    "source": "SW",
                    "filename": file,
                    "n": G.number_of_nodes(),
                    "edges": G.number_of_edges(),

                    "exact_size": None,
                    "exact_time": None,
                    "exact_ops": None,
                    "exact_configs": None,

                    "greedy_size": greedy_stats["size"],
                    "greedy_time": greedy_stats["execution_time"],
                    "greedy_ops": greedy_stats["ops"],

                    "random_size": rand_stats["size"],
                    "random_time": rand_stats["execution_time"],
                    "random_ops": rand_stats["ops"],
                    "random_configs": rand_stats["configs_tested"],
                    "random_iterations": rand_stats.get("iterations"),

                    "randgreedy_size": randg_stats["size"],
                    "randgreedy_time": randg_stats["execution_time"],
                    "randgreedy_ops": randg_stats["ops"],
                    "randgreedy_configs": randg_stats["configs_tested"],
                    "randgreedy_randomness": randg_stats.get("randomness"),
                })
                logger.info(f"    Finished {file}. Nodes: {G.number_of_nodes()}")
            except Exception as e:
                logger.error(f"    Error processing {file}: {e}")

    # BENCHMARK GRAPHS
    logger.info(">>> PHASE 3: Processing Benchmark Graphs")

    for name, G in GraphUtils.benchmark_graphs().items():
        logger.info(f"--- Benchmark: {name} (N={G.number_of_nodes()}) ---")

        solver = MinimumDominatingSet(G)

        if G.number_of_nodes() <= 18:
            exact_sol, exact_stats = solver.exhaustive_search()
        else:
            logger.info("    [Exact] Skipping (N > 18)")
            exact_sol = None
            exact_stats = {
                "size": None,
                "execution_time": None,
                "ops": None,
                "configs_tested": None,
            }

        greedy_sol, greedy_stats = solver.greedy_heuristic()

        rsolver = RandomizedMDS(G)
        rand_sol, rand_stats = rsolver.randomized_mds(iterations=300)

        rsolver2 = RandomizedMDS(G)
        randg_sol, randg_stats = rsolver2.randomized_greedy(randomness=0.3)

        # visualizations
        base = f"benchmark_{name}"
        if exact_sol is not None:
            visualize_solution(
                G,
                exact_sol,
                f"{name} - exact (size={exact_stats['size']})",
                os.path.join("visualizations", base + "_exact.png"),
            )
        visualize_solution(
            G,
            greedy_sol,
            f"{name} - greedy (size={greedy_stats['size']})",
            os.path.join("visualizations", base + "_greedy.png"),
        )
        visualize_solution(
            G,
            rand_sol,
            f"{name} - random (size={rand_stats['size']})",
            os.path.join("visualizations", base + "_random.png"),
        )
        visualize_solution(
            G,
            randg_sol,
            f"{name} - randgreedy (size={randg_stats['size']})",
            os.path.join("visualizations", base + "_randgreedy.png"),
        )

        all_results.append({
            "source": "benchmark_" + name,
            "name": name,
            "n": G.number_of_nodes(),
            "edges": G.number_of_edges(),

            "exact_size": exact_stats["size"],
            "exact_time": exact_stats["execution_time"],
            "exact_ops": exact_stats["ops"],
            "exact_configs": exact_stats["configs_tested"],

            "greedy_size": greedy_stats["size"],
            "greedy_time": greedy_stats["execution_time"],
            "greedy_ops": greedy_stats["ops"],

            "random_size": rand_stats["size"],
            "random_time": rand_stats["execution_time"],
            "random_ops": rand_stats["ops"],
            "random_configs": rand_stats["configs_tested"],
            "random_iterations": rand_stats.get("iterations"),

            "randgreedy_size": randg_stats["size"],
            "randgreedy_time": randg_stats["execution_time"],
            "randgreedy_ops": randg_stats["ops"],
            "randgreedy_configs": randg_stats["configs_tested"],
            "randgreedy_randomness": randg_stats.get("randomness"),
        })

    # FINAL SAVE
    logger.info(">>> Experiment Finished. Saving data...")

    df = pd.DataFrame(all_results)
    output_path = f"{save_folder}/full_experiment_suite.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"Saved to: {output_path}")

    return df


# MAIN
if __name__ == "__main__":
    df = run_suite(
        student_number=113626,
        max_random_n=40,
        sw_folder="SW_ALGUNS_GRAFOS",
        save_folder="results_unified",
    )

    print("\nDONE.\n")
    print(df.head())
