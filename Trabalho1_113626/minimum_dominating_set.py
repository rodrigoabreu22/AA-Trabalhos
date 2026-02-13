import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import json
import os
from itertools import combinations
from typing import Set, List, Tuple, Dict
import pandas as pd

class MinimumDominatingSet:
    """
    Class to solve the Minimum Dominating Set problem using:
    1. Exhaustive Search (Exact Algorithm)
    2. Greedy Heuristic
    """
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.n = graph.number_of_nodes()
        self.m = graph.number_of_edges()
        
        # Statistics for analysis
        self.exhaustive_operations = 0
        self.greedy_operations = 0
        self.exhaustive_configs_tested = 0
        
    def is_dominating_set(self, candidate_set: Set[int]) -> bool:
        """
        Check if a given set is a dominating set.
        A set D is dominating if every vertex not in D has at least one neighbor in D.
        
        Operation count: Increments once per complete domination check.
        """
        
        for v in self.graph.nodes():
            self.exhaustive_operations += 1
            
            if v not in candidate_set:
                neighbors = set(self.graph.neighbors(v))
                if not neighbors.intersection(candidate_set):
                    return False
        return True
    
    def exhaustive_search(self) -> Tuple[Set[int], Dict]:
        """
        Exhaustive search algorithm to find the minimum dominating set.
        Tests all possible subsets of vertices in increasing size order.
        
        Time Complexity: O(2^n * n * d_avg) where d_avg is average degree
        Space Complexity: O(n)
        
        Operation count: Number of candidate sets verified (calls to is_dominating_set).
        """
        self.exhaustive_operations = 0
        self.exhaustive_configs_tested = 0
        
        start_time = time.time()
        
        nodes = list(self.graph.nodes())
        
        # Try subsets of increasing size
        for size in range(1, self.n + 1):
            # Generate all combinations of given size
            for candidate in combinations(nodes, size):
                candidate_set = set(candidate)
                self.exhaustive_configs_tested += 1
                
                if self.is_dominating_set(candidate_set):
                    end_time = time.time()
                    
                    stats = {
                        'solution': candidate_set,
                        'size': len(candidate_set),
                        'operations': self.exhaustive_operations,
                        'configs_tested': self.exhaustive_configs_tested,
                        'execution_time': end_time - start_time
                    }
                    return candidate_set, stats
        
        # Should never reach here for connected graphs
        end_time = time.time()
        stats = {
            'solution': set(nodes),
            'size': self.n,
            'operations': self.exhaustive_operations,
            'configs_tested': self.exhaustive_configs_tested,
            'execution_time': end_time - start_time
        }
        return set(nodes), stats
    
    def greedy_heuristic(self) -> Tuple[Set[int], Dict]:
        """
        Greedy heuristic for minimum dominating set.
        Strategy: Repeatedly select the vertex that dominates the most undominated vertices.
        
        Time Complexity: O(n^2 * d_avg) where d_avg is average degree
        Space Complexity: O(n)
        
        Operation count: Number of vertex evaluations (inner loop iterations).
        This better reflects the O(n^2) computational complexity.
        """
        self.greedy_operations = 0
        start_time = time.time()
        
        dominating_set = set()
        dominated = set()
        all_nodes = set(self.graph.nodes())
        
        while dominated != all_nodes:
            # Find the vertex that dominates the most undominated vertices
            best_vertex = None
            max_new_dominated = -1
            
            for v in all_nodes:
                self.greedy_operations += 1  # Count each vertex evaluation
                
                if v in dominating_set:
                    continue
                
                # Count how many new vertices would be dominated by adding v
                neighbors = set(self.graph.neighbors(v))
                new_dominated = neighbors.union({v}) - dominated
                
                if len(new_dominated) > max_new_dominated:
                    max_new_dominated = len(new_dominated)
                    best_vertex = v
            
            if best_vertex is not None:
                dominating_set.add(best_vertex)
                neighbors = set(self.graph.neighbors(best_vertex))
                dominated.update(neighbors.union({best_vertex}))
        
        end_time = time.time()
        
        stats = {
            'solution': dominating_set,
            'size': len(dominating_set),
            'operations': self.greedy_operations,
            'execution_time': end_time - start_time
        }
        return dominating_set, stats


class GraphGenerator:
    """Generate random graphs for experiments according to specifications."""
    
    def __init__(self, student_number: int):
        self.seed = student_number
        random.seed(student_number)
        np.random.seed(student_number)
    
    def generate_graph(self, n_vertices: int, edge_percentage: float) -> nx.Graph:
        """
        Generate a random graph with specified number of vertices and edge density.
        
        Args:
            n_vertices: Number of vertices
            edge_percentage: Percentage of maximum possible edges (0.125, 0.25, 0.5, 0.75)
        """
        G = nx.Graph()
        
        # Generate 2D coordinates for vertices
        positions = {}
        min_distance = 10  # Minimum distance between vertices
        
        for i in range(n_vertices):
            while True:
                x = random.randint(1, 500)
                y = random.randint(1, 500)
                
                # Check if position is not too close to existing vertices
                too_close = False
                for pos in positions.values():
                    dist = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
                    if dist < min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    positions[i] = (x, y)
                    G.add_node(i, pos=(x, y))
                    break
        
        # Calculate number of edges to add
        max_edges = n_vertices * (n_vertices - 1) // 2
        n_edges = int(max_edges * edge_percentage)
        
        # Generate random edges
        possible_edges = [(i, j) for i in range(n_vertices) 
                         for j in range(i + 1, n_vertices)]
        random.shuffle(possible_edges)
        
        for i in range(min(n_edges, len(possible_edges))):
            G.add_edge(possible_edges[i][0], possible_edges[i][1])
        
        # Ensure graph is connected (for meaningful dominating sets)
        if not nx.is_connected(G):
            # Connect components
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                node1 = random.choice(list(components[i]))
                node2 = random.choice(list(components[i + 1]))
                G.add_edge(node1, node2)
        
        return G
    
    def save_graph(self, G: nx.Graph, filename: str):
        """Save graph to file."""
        data = {
            'nodes': list(G.nodes()),
            'edges': list(G.edges()),
            'positions': nx.get_node_attributes(G, 'pos')
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
    
    def load_graph(self, filename: str) -> nx.Graph:
        """Load graph from file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        G = nx.Graph()
        G.add_nodes_from(data['nodes'])
        G.add_edges_from(data['edges'])
        
        for node, pos in data['positions'].items():
            G.nodes[int(node)]['pos'] = tuple(pos)
        
        return G


def visualize_solution(G: nx.Graph, dominating_set: Set[int], 
                       title: str, filename: str = None):
    """Visualize the graph with the dominating set highlighted."""
    pos = nx.get_node_attributes(G, 'pos')
    
    plt.figure(figsize=(12, 10))
    
    # Draw all edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1)
    
    # Draw non-dominating vertices
    non_dominating = set(G.nodes()) - dominating_set
    nx.draw_networkx_nodes(G, pos, nodelist=list(non_dominating),
                          node_color='lightblue', node_size=300, alpha=0.7)
    
    # Draw dominating set vertices
    nx.draw_networkx_nodes(G, pos, nodelist=list(dominating_set),
                          node_color='red', node_size=500, alpha=0.9)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def run_experiments(student_number: int, max_vertices: int = 10):
    """
    Run comprehensive experiments for the assignment.
    
    Args:
        student_number: My student number (used as random seed)
        max_vertices: Maximum number of vertices to test
    """
    generator = GraphGenerator(student_number)
    edge_percentages = [0.125, 0.25, 0.5, 0.75]
    
    # Create directories for results
    os.makedirs('graphs', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    results = []
    
    print("=" * 80)
    print("MINIMUM DOMINATING SET - COMPUTATIONAL EXPERIMENTS")
    print("=" * 80)
    print(f"Student Number (Seed): {student_number}")
    print(f"Testing graphs with {4} to {max_vertices} vertices")
    print(f"Edge densities: {[f'{int(p*100)}%' for p in edge_percentages]}")
    print("=" * 80)
    
    for n in range(4, max_vertices + 1):
        print(f"\n{'='*80}")
        print(f"TESTING GRAPHS WITH {n} VERTICES")
        print(f"{'='*80}")
        
        for edge_pct in edge_percentages:
            print(f"\n--- Edge Density: {edge_pct*100}% ---")
            
            # Generate graph
            G = generator.generate_graph(n, edge_pct)
            graph_filename = f"graphs/graph_n{n}_e{int(edge_pct*100)}.json"
            generator.save_graph(G, graph_filename)
            
            print(f"Graph: {G.number_of_nodes()} vertices, {G.number_of_edges()} edges")
            
            # Initialize solver
            solver = MinimumDominatingSet(G)
            
            # Run exhaustive search
            print("Running exhaustive search...")
            exact_solution, exact_stats = solver.exhaustive_search()
            print(f"  Exact solution size: {exact_stats['size']}")
            print(f"  Operations: {exact_stats['operations']}")
            print(f"  Configurations tested: {exact_stats['configs_tested']}")
            print(f"  Time: {exact_stats['execution_time']:.6f}s")
            
            # Run greedy heuristic
            print("Running greedy heuristic...")
            greedy_solution, greedy_stats = solver.greedy_heuristic()
            print(f"  Greedy solution size: {greedy_stats['size']}")
            print(f"  Operations: {greedy_stats['operations']}")
            print(f"  Time: {greedy_stats['execution_time']:.6f}s")
            
            # Calculate precision
            precision = exact_stats['size'] / greedy_stats['size'] if greedy_stats['size'] > 0 else 1.0
            print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
            
            # Store results
            results.append({
                'n_vertices': n,
                'n_edges': G.number_of_edges(),
                'edge_percentage': edge_pct,
                'exact_size': exact_stats['size'],
                'exact_operations': exact_stats['operations'],
                'exact_configs': exact_stats['configs_tested'],
                'exact_time': exact_stats['execution_time'],
                'greedy_size': greedy_stats['size'],
                'greedy_operations': greedy_stats['operations'],
                'greedy_time': greedy_stats['execution_time'],
                'precision': precision
            })
            
            # Visualize solutions
            vis_file_exact = f"visualizations/exact_n{n}_e{int(edge_pct*100)}.png"
            vis_file_greedy = f"visualizations/greedy_n{n}_e{int(edge_pct*100)}.png"
            
            visualize_solution(G, exact_solution, 
                             f"Exact MDS (n={n}, edges={edge_pct*100}%, size={exact_stats['size']})",
                             vis_file_exact)
            visualize_solution(G, greedy_solution,
                             f"Greedy MDS (n={n}, edges={edge_pct*100}%, size={greedy_stats['size']})",
                             vis_file_greedy)
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('results/experimental_results.csv', index=False)
    
    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETED")
    print("="*80)
    print(f"Results saved to: results/experimental_results.csv")
    print(f"Graphs saved to: graphs/")
    print(f"Visualizations saved to: visualizations/")
    
    return df


if __name__ == "__main__":
    STUDENT_NUMBER = 113626
    
    results_df = run_experiments(STUDENT_NUMBER, max_vertices=40)
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(results_df.groupby('n_vertices')[['exact_time', 'greedy_time', 'precision']].mean())