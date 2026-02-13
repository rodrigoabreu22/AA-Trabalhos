import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import seaborn as sns
import json
import networkx as nx
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_results(filename='results/experimental_results.csv'):
    """Load experimental results."""
    return pd.read_csv(filename)

def plot_execution_time_comparison(df):
    """Plot execution time for exhaustive vs greedy."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Execution Time Analysis', fontsize=16, fontweight='bold')
    
    edge_percentages = df['edge_percentage'].unique()
    
    for idx, edge_pct in enumerate(edge_percentages):
        ax = axes[idx // 2, idx % 2]
        subset = df[df['edge_percentage'] == edge_pct]
        
        ax.plot(subset['n_vertices'], subset['exact_time'], 
                'o-', label='Exhaustive Search', linewidth=2, markersize=8)
        ax.plot(subset['n_vertices'], subset['greedy_time'], 
                's-', label='Greedy Heuristic', linewidth=2, markersize=8)
        
        ax.set_xlabel('Number of Vertices', fontsize=12)
        ax.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax.set_title(f'Edge Density: {edge_pct*100}%', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('results/execution_time_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: results/execution_time_comparison.png")
    plt.close()

def plot_operations_count(df):
    """Plot number of operations for both algorithms."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Number of Operations Analysis', fontsize=16, fontweight='bold')
    
    edge_percentages = df['edge_percentage'].unique()
    
    for idx, edge_pct in enumerate(edge_percentages):
        ax = axes[idx // 2, idx % 2]
        subset = df[df['edge_percentage'] == edge_pct]
        
        ax.plot(subset['n_vertices'], subset['exact_operations'], 
                'o-', label='Exhaustive Search', linewidth=2, markersize=8)
        ax.plot(subset['n_vertices'], subset['greedy_operations'], 
                's-', label='Greedy Heuristic', linewidth=2, markersize=8)
        
        ax.set_xlabel('Number of Vertices', fontsize=12)
        ax.set_ylabel('Number of Operations', fontsize=12)
        ax.set_title(f'Edge Density: {edge_pct*100}%', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('results/operations_count.png', dpi=300, bbox_inches='tight')
    print("Saved: results/operations_count.png")
    plt.close()

def plot_configurations_tested(df):
    """Plot number of configurations tested by exhaustive search."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for edge_pct in df['edge_percentage'].unique():
        subset = df[df['edge_percentage'] == edge_pct]
        ax.plot(subset['n_vertices'], subset['exact_configs'], 
                'o-', label=f'{edge_pct*100}% edges', linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Vertices', fontsize=14)
    ax.set_ylabel('Configurations Tested', fontsize=14)
    ax.set_title('Exhaustive Search: Configurations Tested', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('results/configurations_tested.png', dpi=300, bbox_inches='tight')
    print("Saved: results/configurations_tested.png")
    plt.close()

def plot_precision_analysis(df):
    """Plot precision of greedy heuristic."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Precision vs number of vertices
    ax = axes[0]
    for edge_pct in df['edge_percentage'].unique():
        subset = df[df['edge_percentage'] == edge_pct]
        ax.plot(subset['n_vertices'], subset['precision'], 
                'o-', label=f'{edge_pct*100}% edges', linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Vertices', fontsize=12)
    ax.set_ylabel('Precision (Optimal/Greedy)', fontsize=12)
    ax.set_title('Greedy Heuristic Precision', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect precision')
    
    # Precision distribution
    ax = axes[1]
    ax.hist(df['precision'], bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Precision', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Precision Distribution', fontsize=13, fontweight='bold')
    ax.axvline(x=df['precision'].mean(), color='r', linestyle='--', 
               linewidth=2, label=f'Mean: {df["precision"].mean():.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/precision_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: results/precision_analysis.png")
    plt.close()

def plot_solution_size_comparison(df):
    """Compare solution sizes between exact and greedy algorithms."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Solution Size Comparison', fontsize=16, fontweight='bold')
    
    edge_percentages = df['edge_percentage'].unique()
    
    for idx, edge_pct in enumerate(edge_percentages):
        ax = axes[idx // 2, idx % 2]
        subset = df[df['edge_percentage'] == edge_pct]
        
        ax.plot(subset['n_vertices'], subset['exact_size'], 
                'o-', label='Exact (Optimal)', linewidth=2, markersize=8)
        ax.plot(subset['n_vertices'], subset['greedy_size'], 
                's-', label='Greedy Heuristic', linewidth=2, markersize=8)
        
        ax.set_xlabel('Number of Vertices', fontsize=12)
        ax.set_ylabel('Dominating Set Size', fontsize=12)
        ax.set_title(f'Edge Density: {edge_pct*100}%', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/solution_size_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: results/solution_size_comparison.png")
    plt.close()

def exponential_model(x, a, b):
    """Exponential growth model: y = a * b^x"""
    return a * np.power(b, x)

def polynomial_model(x, a, b):
    """Polynomial model: y = a * x^b"""
    return a * np.power(x, b)

def fit_complexity_models(df):
    """Fit theoretical complexity models to experimental data."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Complexity Model Fitting', fontsize=16, fontweight='bold')
    
    edge_pct = 0.5  # Use 50% edge density for fitting
    subset = df[df['edge_percentage'] == edge_pct].copy()
    
    # Remove any rows with zero or negative values for fitting
    subset = subset[(subset['exact_time'] > 0) & (subset['greedy_time'] > 0)]
    
    x = subset['n_vertices'].values
    
    # ============================================================
    # Exhaustive search time (exponential)
    # ============================================================
    ax = axes[0, 0]
    y_exact_time = subset['exact_time'].values
    
    try:
        # Use log transform for better exponential fitting
        # ln(y) = ln(a) + x*ln(b) becomes linear
        log_y = np.log(y_exact_time)
        
        # Fit linear model to log-transformed data
        coeffs = np.polyfit(x, log_y, 1)  # y = mx + c
        ln_b = coeffs[0]  # slope = ln(b)
        ln_a = coeffs[1]  # intercept = ln(a)
        
        # Convert back to exponential parameters
        a_exp = np.exp(ln_a)
        b_exp = np.exp(ln_b)
        
        # Generate fitted curve
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = exponential_model(x_fit, a_exp, b_exp)
        
        ax.plot(x, y_exact_time, 'o', markersize=10, label='Experimental', color='#1f77b4')
        ax.plot(x_fit, y_fit, '-', linewidth=2, color='#ff7f0e',
                label=f'Fit: {a_exp:.2e} × {b_exp:.2f}$^n$')
        ax.set_ylabel('Execution Time (s)', fontsize=12)
        ax.set_xlabel('Number of Vertices', fontsize=12)
        ax.set_title('Exhaustive Search - Time', fontsize=13)
        ax.legend(fontsize=11)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        print(f"\nExhaustive Time Model: T(n) = {a_exp:.6e} * {b_exp:.4f}^n")
        
    except Exception as e:
        print(f"Exhaustive time fitting failed: {e}")
        ax.plot(x, y_exact_time, 'o-', markersize=10)
        ax.set_ylabel('Execution Time (s)', fontsize=12)
        ax.set_xlabel('Number of Vertices', fontsize=12)
        ax.set_title('Exhaustive Search - Time (fit failed)', fontsize=13)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # ============================================================
    # Greedy time (polynomial)
    # ============================================================
    ax = axes[0, 1]
    y_greedy_time = subset['greedy_time'].values
    
    try:
        # Use log-log transform for power law fitting
        # ln(y) = ln(a) + b*ln(x) becomes linear
        log_x = np.log(x)
        log_y = np.log(y_greedy_time)
        
        # Fit linear model to log-log transformed data
        coeffs = np.polyfit(log_x, log_y, 1)  # y = mx + c
        b_poly = coeffs[0]  # slope = exponent
        ln_a = coeffs[1]   # intercept = ln(a)
        a_poly = np.exp(ln_a)
        
        # Generate fitted curve
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = polynomial_model(x_fit, a_poly, b_poly)
        
        ax.plot(x, y_greedy_time, 's', markersize=10, label='Experimental', color='#1f77b4')
        ax.plot(x_fit, y_fit, '-', linewidth=2, color='#ff7f0e',
                label=f'Fit: {a_poly:.2e} × n$^{{{b_poly:.2f}}}$')
        ax.set_ylabel('Execution Time (s)', fontsize=12)
        ax.set_xlabel('Number of Vertices', fontsize=12)
        ax.set_title('Greedy Heuristic - Time', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        print(f"Greedy Time Model: T(n) = {a_poly:.6e} * n^{b_poly:.4f}")
        
    except Exception as e:
        print(f"Greedy time fitting failed: {e}")
        ax.plot(x, y_greedy_time, 's-', markersize=10)
        ax.set_ylabel('Execution Time (s)', fontsize=12)
        ax.set_xlabel('Number of Vertices', fontsize=12)
        ax.set_title('Greedy Heuristic - Time (fit failed)', fontsize=13)
        ax.grid(True, alpha=0.3)
    
    # ============================================================
    # Configurations tested (exponential)
    # ============================================================
    ax = axes[1, 0]
    y_configs = subset['exact_configs'].values
    
    try:
        # Use log transform for exponential fitting
        log_y = np.log(y_configs)
        
        # Fit linear model to log-transformed data
        coeffs = np.polyfit(x, log_y, 1)
        ln_b = coeffs[0]
        ln_a = coeffs[1]
        
        a_configs = np.exp(ln_a)
        b_configs = np.exp(ln_b)
        
        # Generate fitted curve
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = exponential_model(x_fit, a_configs, b_configs)
        
        ax.plot(x, y_configs, 'o', markersize=10, label='Experimental', color='#1f77b4')
        ax.plot(x_fit, y_fit, '-', linewidth=2, color='#ff7f0e',
                label=f'Fit: {a_configs:.2e} × {b_configs:.2f}$^n$')
        ax.set_ylabel('Configurations Tested', fontsize=12)
        ax.set_xlabel('Number of Vertices', fontsize=12)
        ax.set_title('Exhaustive Search - Configurations', fontsize=13)
        ax.legend(fontsize=11)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        print(f"Configurations Model: C(n) = {a_configs:.6e} * {b_configs:.4f}^n")
        
    except Exception as e:
        print(f"Configurations fitting failed: {e}")
        ax.plot(x, y_configs, 'o-', markersize=10)
        ax.set_ylabel('Configurations Tested', fontsize=12)
        ax.set_xlabel('Number of Vertices', fontsize=12)
        ax.set_title('Exhaustive Search - Configs (fit failed)', fontsize=13)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # ============================================================
    # Operations comparison
    # ============================================================
    ax = axes[1, 1]
    y_exact_ops = subset['exact_operations'].values
    y_greedy_ops = subset['greedy_operations'].values
    
    ax.plot(x, y_exact_ops, 'o-', markersize=8, label='Exhaustive', 
            linewidth=2, color='#1f77b4')
    ax.plot(x, y_greedy_ops, 's-', markersize=8, label='Greedy', 
            linewidth=2, color='#ff7f0e')
    ax.set_xlabel('Number of Vertices', fontsize=12)
    ax.set_ylabel('Operations Count', fontsize=12)
    ax.set_title('Operations Comparison', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/complexity_model_fitting.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/complexity_model_fitting.png")
    plt.close()

def format_time(time_seconds):
    """Helper function to format seconds into readable strings."""
    if time_seconds < 1:
        return f"{time_seconds * 1000:.2f} ms"
    if time_seconds < 60:
        return f"{time_seconds:.2f} seconds"
    elif time_seconds < 3600:
        return f"{time_seconds / 60:.2f} minutes"
    elif time_seconds < 86400:
        return f"{time_seconds / 3600:.2f} hours"
    elif time_seconds < 31536000:
        return f"{time_seconds / 86400:.2f} days"
    else:
        return f"{time_seconds / 31536000:.2f} years"

# --- REWRITTEN FUNCTION ---

def estimate_larger_instances(df):
    """
    Estimate execution time for larger instances, showing the
    impact of edge density on exhaustive search and also
    estimating for the greedy heuristic.
    """
    max_n_data = df['n_vertices'].max()
    
    # === 1. EXHAUSTIVE SEARCH ===
    print("\n" + "="*80)
    print("ESTIMATION FOR LARGER INSTANCES (EXHAUSTIVE SEARCH)")
    print("="*80)
    print("Fitting models to show impact of edge density...")

    # We will fit models for the best, average, and worst cases observed
    densities_to_fit = {
        'Worst-Case (12.5%)': 0.125,
        'Average-Case (50.0%)': 0.50,
        'Best-Case (75.0%)': 0.75
    }
    
    fitted_models = {}
    
    print("\n--- Fitted Exponential Models (T(n) = a * b^n) ---")
    for label, density in densities_to_fit.items():
        subset = df[(df['edge_percentage'] == density) & (df['exact_time'] > 0)].copy()
        x = subset['n_vertices'].values
        y = subset['exact_time'].values
        
        try:
            # Provide initial guess p0 to help fit convergence
            params, _ = curve_fit(exponential_model, x, y, p0=[1e-9, 1.1], maxfev=10000)
            a, b = params
            fitted_models[density] = (a, b)
            print(f"{label:<22} T(n) = {a:.2e} * {b:.3f}^n")
        except Exception as e:
            print(f"Could not fit model for {label}: {e}")
            fitted_models[density] = None

    # --- 1a. Extrapolation Table (Exhaustive) ---
    print("\n--- Extrapolated Times (Exhaustive Search) ---")
    target_vertices_exp = [max_n_data + 5, max_n_data + 10, max_n_data + 20]
    header = f"{'Vertices (n)':<15}"
    for label in densities_to_fit.keys():
        header += f" {label:<22}"
    print(header)
    print("-" * len(header))

    for n in target_vertices_exp:
        row = f"{n:<15}"
        for density in densities_to_fit.values():
            model = fitted_models.get(density)
            if model:
                a, b = model
                time_est = exponential_model(n, a, b)
                row += f" {format_time(time_est):<22}"
            else:
                row += f" {'N/A':<22}"
        print(row)

    # --- 1b. Determine Practical Limit (Exhaustive) ---
    print("\n" + "="*80)
    print("DETERMINING PRACTICAL LIMIT (Exhaustive Search)")
    print("="*80)
    
    worst_case_model = fitted_models.get(0.125)
    time_limit_seconds = 60  # Define "too much time" as 60 seconds
    
    if worst_case_model:
        a_worst, b_worst = worst_case_model
        print(f"Using worst-case model (12.5%): T(n) = {a_worst:.2e} * {b_worst:.3f}^n")
        
        # Check if our largest data point already exceeds the limit
        last_time = df[df['edge_percentage'] == 0.125]['exact_time'].max()
        if last_time >= time_limit_seconds:
            max_feasible_n = df[(df['edge_percentage'] == 0.125) & 
                                (df['exact_time'] < time_limit_seconds)]['n_vertices'].max()
            print(f"Experimental data shows limit already reached.")
            print(f"Practical limit (time < {time_limit_seconds}s): n = {max_feasible_n} vertices")
        else:
            # Loop forward from last data point until limit is breached
            n = int(max_n_data)
            time_est = last_time
            while time_est <= time_limit_seconds:
                n += 1
                time_est = exponential_model(n, a_worst, b_worst)
                if n > 200: # Safety break
                    print("Limit is > 200 vertices (model may be inaccurate)")
                    break
            
            max_feasible_n = n - 1
            time_at_max = exponential_model(max_feasible_n, a_worst, b_worst)
            print(f"Practical limit (time < {time_limit_seconds}s): n = {max_feasible_n} vertices")
            print(f"Estimated time at n={max_feasible_n}: {format_time(time_at_max)}")
            
    else:
        print("Could not determine practical limit (12.5% model fit failed).")

    # === 2. GREEDY HEURISTIC ===
    print("\n" + "="*80)
    print("ESTIMATION FOR LARGER INSTANCES (GREEDY HEURISTIC)")
    print("="*80)
    
    # Density has little impact, so 50% is a good representative model
    subset_g = df[(df['edge_percentage'] == 0.5) & (df['greedy_time'] > 0)].copy()
    x_g = subset_g['n_vertices'].values
    y_g = subset_g['greedy_time'].values

    try:
        # Fit polynomial model T(n) = a * n^c
        log_x = np.log(x_g)
        log_y = np.log(y_g)
        coeffs = np.polyfit(log_x, log_y, 1)
        c_poly = coeffs[0]  # exponent
        a_poly = np.exp(coeffs[1])  # coefficient
        
        model_str_g = f"T(n) = {a_poly:.2e} * n^{c_poly:.2f}"
        print(f"Using 50% density (average-case) model: {model_str_g}")
        
        target_vertices_g = [1000, 10000, 100000, 1000000]
        print("\n--- Extrapolated Times (Greedy Heuristic) ---")
        print(f"{'Vertices (n)':<15} {'Estimated Time':<20} {'Human Readable'}")
        print("-" * 60)
        
        for n in target_vertices_g:
            time_seconds = polynomial_model(n, a_poly, c_poly)
            print(f"{n:<15} {time_seconds:<20.6e} {format_time(time_seconds)}")
        
        print("-" * 60)
        print("\nNOTE: This is an average-case estimation based on the fitted O(n^c) model.")
        print(f"The theoretical O(n^3) worst-case would be significantly slower.")

    except Exception as e:
        print(f"Could not fit polynomial model for greedy heuristic: {e}")

def generate_summary_statistics(df):
    """Generate summary statistics table."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print("\n1. Average Execution Time by Number of Vertices:")
    print("-" * 60)
    summary_time = df.groupby('n_vertices')[['exact_time', 'greedy_time']].mean()
    summary_time['speedup'] = summary_time['exact_time'] / summary_time['greedy_time']
    print(summary_time.to_string())
    
    print("\n\n2. Average Precision by Edge Density:")
    print("-" * 60)
    summary_precision = df.groupby('edge_percentage')['precision'].agg(['mean', 'std', 'min', 'max'])
    print(summary_precision.to_string())
    
    print("\n\n3. Complexity Growth Analysis:")
    print("-" * 60)
    for edge_pct in df['edge_percentage'].unique():
        subset = df[df['edge_percentage'] == edge_pct]
        if len(subset) > 1:
            # Calculate growth rate between consecutive n values
            growth_rates = []
            for i in range(1, len(subset)):
                if subset.iloc[i-1]['exact_time'] > 0:
                    rate = subset.iloc[i]['exact_time'] / subset.iloc[i-1]['exact_time']
                    growth_rates.append(rate)
            
            if growth_rates:
                avg_growth = np.mean(growth_rates)
                print(f"Edge density {edge_pct*100}%: Avg growth rate = {avg_growth:.2f}x per vertex")
    
    print("\n\n4. Solution Quality Analysis:")
    print("-" * 60)
    print(f"Overall precision: {df['precision'].mean():.4f} ± {df['precision'].std():.4f}")
    print(f"Times greedy found optimal: {(df['precision'] == 1.0).sum()} / {len(df)}")
    print(f"Worst case precision: {df['precision'].min():.4f}")
    print(f"Best case precision: {df['precision'].max():.4f}")

def create_latex_table(df):
    """Generate LaTeX table for the report."""
    print("\n" + "="*80)
    print("LATEX TABLE FOR REPORT")
    print("="*80)
    
    latex_str = """
\\begin{table}[h]
\\centering
\\caption{Experimental Results for Minimum Dominating Set}
\\label{tab:results}
\\begin{tabular}{|c|c|c|c|c|c|c|}
\\hline
\\textbf{n} & \\textbf{Edges\\%} & \\textbf{Exact Size} & \\textbf{Greedy Size} & \\textbf{Precision} & \\textbf{Exact Time (s)} & \\textbf{Configs} \\\\
\\hline
"""
    
    for _, row in df.iterrows():
        latex_str += f"{row['n_vertices']} & {int(row['edge_percentage']*100)}\\% & "
        latex_str += f"{row['exact_size']} & {row['greedy_size']} & "
        latex_str += f"{row['precision']:.3f} & {row['exact_time']:.6f} & {row['exact_configs']} \\\\\n"
    
    latex_str += """\\hline
\\end{tabular}
\\end{table}
"""
    
    print(latex_str)
    
    with open('results/latex_table.tex', 'w') as f:
        f.write(latex_str)
    print("\nSaved to: results/latex_table.tex")

def plot_graph_from_file(filename):
    """Load and plot a graph from the 'graphs/' folder in the XOY plane."""
    filepath = os.path.join('graphs', filename)
    
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return
    
    # Load graph data
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    G = nx.Graph()
    G.add_nodes_from(data['nodes'])
    G.add_edges_from(data['edges'])
    
    # Convert positions (ensure tuples)
    positions = {int(k): tuple(v) for k, v in data['positions'].items()}
    
    # Plot
    plt.figure(figsize=(8, 8))
    nx.draw(
        G,
        pos=positions,
        with_labels=True,
        node_color='skyblue',
        node_size=600,
        font_size=10,
        edge_color='gray',
        linewidths=1.5
    )
    plt.title(f"Graph Visualization: {filename}", fontsize=14, fontweight='bold')
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    output_path = filename.replace('.json', '.png')
    output_path = os.path.join('images', output_path)
    plt.savefig(output_path, dpi=300)
    print(f"Graph saved as {output_path}")



def main():
    """Main analysis function."""
    print("="*80)
    print("MINIMUM DOMINATING SET - ANALYSIS SCRIPT")
    print("="*80)
    
    # Load results
    df = load_results()
    
    print(f"\nLoaded {len(df)} experimental results")
    print(f"Vertices range: {df['n_vertices'].min()} to {df['n_vertices'].max()}")
    print(f"Edge densities: {sorted(df['edge_percentage'].unique())}")
    
    # Generate all visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_execution_time_comparison(df)
    plot_operations_count(df)
    plot_configurations_tested(df)
    plot_precision_analysis(df)
    plot_solution_size_comparison(df)
    fit_complexity_models(df)
    
    # Generate statistics
    generate_summary_statistics(df)
    
    # Estimate larger instances
    estimate_larger_instances(df)
    
    # Generate LaTeX table
    create_latex_table(df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nAll results saved to 'results/' directory")
    print("All visualizations saved as PNG files")
    print("\nYou can now use these results for your report!")

    plot_graph_from_file("graph_n8_e50.json")

if __name__ == "__main__":
    main()