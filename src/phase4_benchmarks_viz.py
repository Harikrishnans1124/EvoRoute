"""
Phase 4: Baselines, Ablation/Sensitivity Suite, Visualization & Final Deliverables
Implements:
1. 6-Way Comparative Benchmark (Random, Cost-Greedy, Betweenness-Greedy, AHP-TOPSIS, Fuzzy-Greedy, Fuzzy+GA).
2. Ablation Study: GA with Fuzzy Guidance vs. Vanilla GA (Random Seeding & Random Repair).
3. Fuzzy Sensitivity Analysis: Breakpoint shift (+/-10%) and Kendall's tau correlation.
4. Publication Figures (Fig 1: FIS Surface, Fig 2: Convergence & Ablation, Fig 3: Benchmark Comparisons).
5. Interactive Folium Flood Resilience Map (results/maps/flood_resilience_map.html).
6. Comprehensive ASSUMPTIONS.md and benchmark_results.csv.
"""
import math
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
import folium
from scipy.stats import kendalltau
from typing import Dict, List, Tuple, Any, Set

from src.config import (
    BUDGET_TIERS,
    RANDOM_SEED,
    RESULTS_DIR,
    FIGURES_DIR,
    MAPS_DIR,
    TABLES_DIR,
    GA_CONFIG,
    AHP_WEIGHTS,
)
from src.phase2_accessibility_criteria import compute_network_accessibility_state
from src.phase3_soft_computing import (
    FuzzyConstrainedGA,
    build_mamdani_fis,
    evaluate_fuzzy_priorities,
)
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Set matplotlib style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10

# ==============================================================================
# 1. SIX-METHOD COMPARATIVE BENCHMARK
# ==============================================================================

def evaluate_portfolio(
    G: nx.Graph,
    villages_df: pd.DataFrame,
    facilities_dict: Dict[str, List[int]],
    baseline_fai: float,
    protected_cids: Set[int],
    cost_usd: float,
    budget_usd: float
) -> Dict[str, Any]:
    """
    Evaluates a specific portfolio of protected road corridors across all 3 flood scenarios.
    """
    fai_retained_list = []
    iso_pop_list = []
    iso_villages_list = []
    oai_list = []
    
    for s_key in ["s1", "s2", "s3"]:
        st = compute_network_accessibility_state(
            G, villages_df, facilities_dict,
            scenario=s_key, restored_corridors=protected_cids
        )
        fai_retained_pct = (st["fai_net"] / baseline_fai) * 100.0
        fai_retained_list.append(fai_retained_pct)
        iso_pop_list.append(st["isolated_pop_share"] * 100.0)
        iso_villages_list.append(st["isolated_villages_count"])
        oai_list.append(st["oai_share"] * 100.0)
        
    return {
        "fai_retained_mean": float(np.mean(fai_retained_list)),
        "fai_retained_std": float(np.std(fai_retained_list)),
        "hosp_isolated_pop_pct": float(np.mean(iso_pop_list)),
        "isolated_villages_count": float(np.mean(iso_villages_list)),
        "oai_pct": float(np.mean(oai_list)),
        "cost_usd": cost_usd,
        "budget_usd": budget_usd,
        "budget_utilized_pct": float(round((cost_usd / budget_usd) * 100.0, 2)),
        "num_roads_protected": len(protected_cids),
    }

def run_crisp_ahp_topsis(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes TOPSIS Closeness Coefficient (CC_j) using AHP criteria weights:
    Criteria: [Exposure (0.20), Accessibility Gain (0.35), Isolation Criticality (0.45)]
    """
    df = candidates_df.copy()
    weights = np.array([0.20, 0.35, 0.45], dtype=np.float64)
    matrix = df[["exp_norm", "acc_norm", "iso_norm"]].values.astype(np.float64)
    
    # Vector normalization: r_ij = x_ij / sqrt(sum_i x_ij^2)
    col_norms = np.sqrt(np.sum(matrix ** 2, axis=0))
    col_norms[col_norms == 0] = 1.0
    norm_matrix = matrix / col_norms
    
    # Weighted normalized matrix
    weighted_matrix = norm_matrix * weights
    
    # Ideal best (A+) and ideal worst (A-)
    # All 3 are benefit criteria (higher exposure = higher fortification need)
    a_plus = np.max(weighted_matrix, axis=0)
    a_minus = np.min(weighted_matrix, axis=0)
    
    # Euclidean distances
    d_plus = np.sqrt(np.sum((weighted_matrix - a_plus) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((weighted_matrix - a_minus) ** 2, axis=1))
    
    # Closeness coefficient CC_j = D- / (D+ + D-)
    denom = d_plus + d_minus
    denom[denom == 0] = 1.0
    cc = d_minus / denom
    
    df["topsis_score"] = cc
    df["topsis_cost_ratio"] = cc / df["cost_usd"]
    return df

def run_six_benchmarks(
    G: nx.Graph,
    villages_df: pd.DataFrame,
    facilities_dict: Dict[str, List[int]],
    candidates_df: pd.DataFrame,
    ga_results: Dict[str, Any],
    baseline_fai: float
) -> pd.DataFrame:
    """
    Evaluates 6 methods across all 4 budget tiers:
    1. Random Selection (Floor)
    2. Cheapest-First (Cost-Greedy)
    3. Betweenness-Only Greedy (Topology Baseline)
    4. Crisp AHP-TOPSIS Greedy (MCDM Baseline)
    5. Mamdani Fuzzy Greedy (Soft Computing Layer 1 Alone)
    6. Proposed Hybrid Fuzzy + GA (Soft Computing Layer 1 + 2)
    """
    total_cost = float(candidates_df["cost_usd"].sum())
    topsis_df = run_crisp_ahp_topsis(candidates_df)
    
    benchmark_records = []
    
    for b_frac in BUDGET_TIERS:
        budget_usd = b_frac * total_cost
        b_label = f"{int(round(b_frac * 100))}%"
        b_pct = int(round(b_frac * 100))
        
        # 1. Random Selection Baseline (average over 20 random repaired portfolios)
        rng = np.random.RandomState(RANDOM_SEED + b_pct)
        rand_runs = []
        for _ in range(20):
            perm = rng.permutation(len(candidates_df))
            spent = 0.0
            sel_cids = set()
            for idx in perm:
                c = candidates_df.iloc[idx]["cost_usd"]
                if spent + c <= budget_usd:
                    spent += c
                    sel_cids.add(int(candidates_df.iloc[idx]["corridor_id"]))
            ev = evaluate_portfolio(G, villages_df, facilities_dict, baseline_fai, sel_cids, spent, budget_usd)
            rand_runs.append(ev)
            
        benchmark_records.append({
            "budget_pct": b_pct,
            "budget_usd": budget_usd,
            "method": "1. Random Floor",
            "fai_retained_mean": float(np.mean([r["fai_retained_mean"] for r in rand_runs])),
            "fai_retained_std": float(np.std([r["fai_retained_mean"] for r in rand_runs])),
            "hosp_isolated_pop_pct": float(np.mean([r["hosp_isolated_pop_pct"] for r in rand_runs])),
            "isolated_villages_count": float(np.mean([r["isolated_villages_count"] for r in rand_runs])),
            "oai_pct": float(np.mean([r["oai_pct"] for r in rand_runs])),
            "budget_utilized_pct": float(np.mean([r["budget_utilized_pct"] for r in rand_runs])),
            "roads_protected": float(np.mean([r["num_roads_protected"] for r in rand_runs])),
        })
        
        # 2. Cheapest-First (Cost-Greedy)
        sorted_cost = candidates_df.sort_values("cost_usd", ascending=True)
        spent = 0.0
        sel_cids = set()
        for _, row in sorted_cost.iterrows():
            if spent + row["cost_usd"] <= budget_usd:
                spent += row["cost_usd"]
                sel_cids.add(int(row["corridor_id"]))
        ev_cheap = evaluate_portfolio(G, villages_df, facilities_dict, baseline_fai, sel_cids, spent, budget_usd)
        benchmark_records.append({
            "budget_pct": b_pct,
            "budget_usd": budget_usd,
            "method": "2. Cheapest-First",
            "fai_retained_mean": ev_cheap["fai_retained_mean"],
            "fai_retained_std": ev_cheap["fai_retained_std"],
            "hosp_isolated_pop_pct": ev_cheap["hosp_isolated_pop_pct"],
            "isolated_villages_count": ev_cheap["isolated_villages_count"],
            "oai_pct": ev_cheap["oai_pct"],
            "budget_utilized_pct": ev_cheap["budget_utilized_pct"],
            "roads_protected": ev_cheap["num_roads_protected"],
        })
        
        # 3. Betweenness-Only Greedy (Topology Baseline)
        df_bc = candidates_df.copy()
        df_bc["bc_ratio"] = df_bc["betweenness"] / df_bc["cost_usd"]
        sorted_bc = df_bc.sort_values("bc_ratio", ascending=False)
        spent = 0.0
        sel_cids = set()
        for _, row in sorted_bc.iterrows():
            if spent + row["cost_usd"] <= budget_usd:
                spent += row["cost_usd"]
                sel_cids.add(int(row["corridor_id"]))
        ev_bc = evaluate_portfolio(G, villages_df, facilities_dict, baseline_fai, sel_cids, spent, budget_usd)
        benchmark_records.append({
            "budget_pct": b_pct,
            "budget_usd": budget_usd,
            "method": "3. Betweenness-Only",
            "fai_retained_mean": ev_bc["fai_retained_mean"],
            "fai_retained_std": ev_bc["fai_retained_std"],
            "hosp_isolated_pop_pct": ev_bc["hosp_isolated_pop_pct"],
            "isolated_villages_count": ev_bc["isolated_villages_count"],
            "oai_pct": ev_bc["oai_pct"],
            "budget_utilized_pct": ev_bc["budget_utilized_pct"],
            "roads_protected": ev_bc["num_roads_protected"],
        })
        
        # 4. Crisp AHP-TOPSIS Greedy (MCDM Baseline)
        sorted_topsis = topsis_df.sort_values("topsis_cost_ratio", ascending=False)
        spent = 0.0
        sel_cids = set()
        for _, row in sorted_topsis.iterrows():
            if spent + row["cost_usd"] <= budget_usd:
                spent += row["cost_usd"]
                sel_cids.add(int(row["corridor_id"]))
        ev_topsis = evaluate_portfolio(G, villages_df, facilities_dict, baseline_fai, sel_cids, spent, budget_usd)
        benchmark_records.append({
            "budget_pct": b_pct,
            "budget_usd": budget_usd,
            "method": "4. Crisp AHP-TOPSIS",
            "fai_retained_mean": ev_topsis["fai_retained_mean"],
            "fai_retained_std": ev_topsis["fai_retained_std"],
            "hosp_isolated_pop_pct": ev_topsis["hosp_isolated_pop_pct"],
            "isolated_villages_count": ev_topsis["isolated_villages_count"],
            "oai_pct": ev_topsis["oai_pct"],
            "budget_utilized_pct": ev_topsis["budget_utilized_pct"],
            "roads_protected": ev_topsis["num_roads_protected"],
        })
        
        # 5. Mamdani Fuzzy Greedy (Soft Computing Layer 1 Alone)
        sorted_fuzzy = candidates_df.sort_values("fuzzy_cost_ratio", ascending=False)
        spent = 0.0
        sel_cids = set()
        for _, row in sorted_fuzzy.iterrows():
            if spent + row["cost_usd"] <= budget_usd:
                spent += row["cost_usd"]
                sel_cids.add(int(row["corridor_id"]))
        ev_fuzzy = evaluate_portfolio(G, villages_df, facilities_dict, baseline_fai, sel_cids, spent, budget_usd)
        benchmark_records.append({
            "budget_pct": b_pct,
            "budget_usd": budget_usd,
            "method": "5. Mamdani Fuzzy Alone",
            "fai_retained_mean": ev_fuzzy["fai_retained_mean"],
            "fai_retained_std": ev_fuzzy["fai_retained_std"],
            "hosp_isolated_pop_pct": ev_fuzzy["hosp_isolated_pop_pct"],
            "isolated_villages_count": ev_fuzzy["isolated_villages_count"],
            "oai_pct": ev_fuzzy["oai_pct"],
            "budget_utilized_pct": ev_fuzzy["budget_utilized_pct"],
            "roads_protected": ev_fuzzy["num_roads_protected"],
        })
        
        # 6. Proposed Hybrid Fuzzy + GA (Layer 1 + Layer 2)
        ga_b = ga_results[b_label]
        best_sol = ga_b["best_solution"]
        sel_cids = set(best_sol["best_corridor_ids"])
        ev_ga = evaluate_portfolio(G, villages_df, facilities_dict, baseline_fai, sel_cids, best_sol["cost_usd"], budget_usd)
        benchmark_records.append({
            "budget_pct": b_pct,
            "budget_usd": budget_usd,
            "method": "6. Proposed Fuzzy + GA",
            "fai_retained_mean": ga_b["fai_retained_mean"],
            "fai_retained_std": ga_b["fai_retained_std"],
            "hosp_isolated_pop_pct": ga_b["isolated_pop_mean"],
            "isolated_villages_count": ev_ga["isolated_villages_count"],
            "oai_pct": ev_ga["oai_pct"],
            "budget_utilized_pct": ga_b["budget_used_mean"],
            "roads_protected": best_sol["num_roads_protected"],
        })
        
    benchmark_df = pd.DataFrame(benchmark_records)
    return benchmark_df

# ==============================================================================
# 2. ABLATION STUDY: FUZZY GA VS. VANILLA GA
# ==============================================================================

class VanillaGA(FuzzyConstrainedGA):
    """
    Ablated GA baseline:
    - 100% Random Population Initialization (no fuzzy seeding)
    - Random Drop Repair Operator (drops random active genes instead of lowest rho_j)
    """
    def repair_chromosome(self, chrom: np.ndarray) -> np.ndarray:
        chrom = chrom.copy()
        current_cost = np.sum(chrom * self.costs)
        # Purely random drop without fuzzy guidance
        while current_cost > self.budget:
            active_indices = np.where(chrom == 1)[0]
            if len(active_indices) == 0:
                break
            drop_idx = self.rng.choice(active_indices)
            chrom[drop_idx] = 0
            current_cost -= self.costs[drop_idx]
            
        # Greedy fill purely by cost
        inactive_indices = np.where(chrom == 0)[0]
        self.rng.shuffle(inactive_indices)
        for gene_idx in inactive_indices:
            if current_cost + self.costs[gene_idx] <= self.budget:
                chrom[gene_idx] = 1
                current_cost += self.costs[gene_idx]
        return chrom

    def initialize_population(self) -> np.ndarray:
        # 100% random exploration
        population = []
        for _ in range(self.config["pop_size"]):
            chrom = self.rng.randint(0, 2, size=self.N, dtype=int)
            population.append(self.repair_chromosome(chrom))
        return np.array(population)

def run_ablation_study(
    G: nx.Graph,
    villages_df: pd.DataFrame,
    facilities_dict: Dict[str, List[int]],
    candidates_df: pd.DataFrame,
    baseline_fai: float,
    budget_pct: float = 0.10
) -> Dict[str, Any]:
    """
    Compares Proposed Fuzzy-Guided GA against Vanilla GA at 10% budget.
    """
    total_cost = float(candidates_df["cost_usd"].sum())
    budget = budget_pct * total_cost
    
    # 1. Proposed Fuzzy-GA
    fuzzy_ga = FuzzyConstrainedGA(
        G, villages_df, facilities_dict, candidates_df,
        baseline_fai=baseline_fai, budget=budget, seed=RANDOM_SEED
    )
    res_fuzzy = fuzzy_ga.run()
    
    # 2. Vanilla GA
    vanilla_ga = VanillaGA(
        G, villages_df, facilities_dict, candidates_df,
        baseline_fai=baseline_fai, budget=budget, seed=RANDOM_SEED
    )
    res_vanilla = vanilla_ga.run()
    
    return {
        "budget_pct": int(round(budget_pct * 100)),
        "fuzzy_ga_history": res_fuzzy["fitness_history"],
        "vanilla_ga_history": res_vanilla["fitness_history"],
        "fuzzy_best_fitness": res_fuzzy["best_fitness"],
        "vanilla_best_fitness": res_vanilla["best_fitness"],
        "fuzzy_fai": res_fuzzy["mean_fai_retained_pct"],
        "vanilla_fai": res_vanilla["mean_fai_retained_pct"],
    }

# ==============================================================================
# 3. FUZZY SENSITIVITY ANALYSIS
# ==============================================================================

def run_fuzzy_sensitivity_analysis(
    candidates_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Shifts membership function breakpoints by -10% and +10% to test stability.
    Computes Kendall's tau correlation against the baseline fuzzy ranking.
    """
    base_scores = candidates_df["fuzzy_priority"].values
    
    def evaluate_shifted_fis(delta: float) -> np.ndarray:
        U_in = np.arange(0.0, 1.01, 0.01)
        U_out = np.arange(0.0, 100.5, 0.5)
        
        exp_ant = ctrl.Antecedent(U_in, "exposure")
        acc_ant = ctrl.Antecedent(U_in, "accessibility")
        iso_ant = ctrl.Antecedent(U_in, "isolation")
        prio_con = ctrl.Consequent(U_out, "priority")
        
        # Shift interior breakpoints by factor (1 + delta)
        factor = 1.0 + delta
        for ant in [exp_ant, acc_ant, iso_ant]:
            ant["L"] = fuzz.trapmf(ant.universe, [0.0, 0.0, np.clip(0.20 * factor, 0.05, 0.35), np.clip(0.45 * factor, 0.30, 0.60)])
            ant["M"] = fuzz.trimf(ant.universe, [np.clip(0.25 * factor, 0.15, 0.40), np.clip(0.50 * factor, 0.40, 0.65), np.clip(0.75 * factor, 0.60, 0.85)])
            ant["H"] = fuzz.trapmf(ant.universe, [np.clip(0.55 * factor, 0.45, 0.70), np.clip(0.80 * factor, 0.70, 0.90), 1.0, 1.0])
            
        prio_con["VL"] = fuzz.trapmf(prio_con.universe, [0, 0, 10, 25])
        prio_con["L"]  = fuzz.trimf(prio_con.universe, [15, 30, 45])
        prio_con["M"]  = fuzz.trimf(prio_con.universe, [35, 50, 65])
        prio_con["H"]  = fuzz.trimf(prio_con.universe, [55, 70, 85])
        prio_con["VH"] = fuzz.trapmf(prio_con.universe, [75, 90, 100, 100])
        
        rules = []
        level_map = {"L": 0, "M": 1, "H": 2}
        for l_e in ["L", "M", "H"]:
            for l_a in ["L", "M", "H"]:
                for l_i in ["L", "M", "H"]:
                    i_e, i_a, i_i = level_map[l_e], level_map[l_a], level_map[l_i]
                    u = i_e + i_a + 2 * i_i
                    if i_i == 2 and (i_e >= 1 or i_a >= 1):
                        con = "VH"
                    elif u <= 1:
                        con = "VL"
                    elif u <= 3:
                        con = "L"
                    elif u <= 5:
                        con = "M"
                    elif u == 6:
                        con = "H"
                    else:
                        con = "VH"
                    rules.append(ctrl.Rule(exp_ant[l_e] & acc_ant[l_a] & iso_ant[l_i], prio_con[con]))
                    
        sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
        scores = []
        for _, row in candidates_df.iterrows():
            sim.input["exposure"] = float(np.clip(row["exp_norm"], 0.0, 1.0))
            sim.input["accessibility"] = float(np.clip(row["acc_norm"], 0.0, 1.0))
            sim.input["isolation"] = float(np.clip(row["iso_norm"], 0.0, 1.0))
            sim.compute()
            scores.append(float(np.clip(sim.output["priority"], 0.0, 100.0)))
        return np.array(scores)
        
    scores_minus10 = evaluate_shifted_fis(-0.10)
    scores_plus10 = evaluate_shifted_fis(+0.10)
    
    tau_minus, _ = kendalltau(base_scores, scores_minus10)
    tau_plus, _ = kendalltau(base_scores, scores_plus10)
    
    return {
        "tau_minus_10pct": float(round(tau_minus, 4)),
        "tau_plus_10pct": float(round(tau_plus, 4)),
        "scores_minus10": scores_minus10,
        "scores_plus10": scores_plus10,
    }

# ==============================================================================
# 4. PUBLICATION-QUALITY FIGURES
# ==============================================================================

def plot_fig1_fuzzy_surfaces(candidates_df: pd.DataFrame):
    """
    Fig 1: Membership Functions and 3D Control Surface.
    """
    fig = plt.figure(figsize=(15, 6))
    
    # Left subplot: Membership functions
    ax1 = fig.add_subplot(1, 2, 1)
    U_in = np.linspace(0, 1, 200)
    mf_L = fuzz.trapmf(U_in, [0.0, 0.0, 0.20, 0.45])
    mf_M = fuzz.trimf(U_in, [0.25, 0.50, 0.75])
    mf_H = fuzz.trapmf(U_in, [0.55, 0.80, 1.0, 1.0])
    
    ax1.plot(U_in, mf_L, "b-", linewidth=2.5, label="Low (Trapezoidal)")
    ax1.plot(U_in, mf_M, "g--", linewidth=2.5, label="Medium (Triangular)")
    ax1.plot(U_in, mf_H, "r-.", linewidth=2.5, label="High (Trapezoidal)")
    ax1.set_title("(a) Input Membership Functions (E, ΔA, B_I)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Normalized Criterion Value", fontsize=11)
    ax1.set_ylabel("Degree of Membership (μ)", fontsize=11)
    ax1.legend(loc="center right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Right subplot: 3D Control Surface of Priority(E, ΔA | Isolation = 0.8)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    fis_sim = build_mamdani_fis()
    
    e_vals = np.linspace(0.05, 0.95, 20)
    a_vals = np.linspace(0.05, 0.95, 20)
    E_grid, A_grid = np.meshgrid(e_vals, a_vals)
    Z_grid = np.zeros_like(E_grid)
    
    iso_fixed = 0.80
    for i in range(E_grid.shape[0]):
        for j in range(E_grid.shape[1]):
            fis_sim.input["exposure"] = float(E_grid[i, j])
            fis_sim.input["accessibility"] = float(A_grid[i, j])
            fis_sim.input["isolation"] = float(iso_fixed)
            fis_sim.compute()
            Z_grid[i, j] = fis_sim.output["priority"]
            
    surf = ax2.plot_surface(E_grid, A_grid, Z_grid, cmap=cm.viridis, edgecolor="none", alpha=0.85)
    ax2.set_title("(b) 3D Control Surface: Priority(E, ΔA | Isolation=0.8)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Exposure (E)", fontsize=10)
    ax2.set_ylabel("Accessibility Gain (ΔA)", fontsize=10)
    ax2.set_zlabel("Fuzzy Priority (0-100)", fontsize=10)
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10, label="Priority Score")
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig1_fuzzy_membership_and_surface.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"-> Saved Fig 1 to {out_path}")

def plot_fig2_ga_convergence_and_ablation(
    ga_results: Dict[str, Any],
    ablation_res: Dict[str, Any]
):
    """
    Fig 2: GA Convergence across budgets & Fuzzy vs. Vanilla GA ablation curve.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Convergence across 4 budget tiers
    colors = {"5%": "tab:blue", "10%": "tab:green", "15%": "tab:orange", "20%": "tab:purple"}
    for b_label, res in ga_results.items():
        hist = res["best_solution"]["fitness_history"]
        gens = np.arange(1, len(hist) + 1)
        ax1.plot(gens, hist, linewidth=2.2, color=colors[b_label], label=f"Budget {b_label} (${res['budget_usd']:,.0f})")
        
    ax1.set_title("(a) GA Multi-Tier Convergence Trajectories", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Generation", fontsize=11)
    ax1.set_ylabel("Best Network Fitness F(x)", fontsize=11)
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Subplot 2: Ablation Study at 10% Budget
    h_fuzzy = ablation_res["fuzzy_ga_history"]
    h_vanilla = ablation_res["vanilla_ga_history"]
    g_fuzzy = np.arange(1, len(h_fuzzy) + 1)
    g_vanilla = np.arange(1, len(h_vanilla) + 1)
    
    ax2.plot(g_fuzzy, h_fuzzy, "g-", linewidth=2.5, label=f"Proposed Fuzzy-GA (Best={ablation_res['fuzzy_best_fitness']:.3f})")
    ax2.plot(g_vanilla, h_vanilla, "r--", linewidth=2.2, label=f"Vanilla GA (Random Repair) (Best={ablation_res['vanilla_best_fitness']:.3f})")
    ax2.set_title(f"(b) Ablation Study (10% Budget Tier): Fuzzy Guidance vs. Random", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Generation", fontsize=11)
    ax2.set_ylabel("Best Fitness F(x)", fontsize=11)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig2_ga_convergence_and_ablation.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"-> Saved Fig 2 to {out_path}")

def plot_fig3_benchmark_comparisons(benchmark_df: pd.DataFrame):
    """
    Fig 3: 6-Way Benchmark comparison charts across budget tiers.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    
    methods = benchmark_df["method"].unique()
    budget_tiers = sorted(benchmark_df["budget_pct"].unique())
    
    markers = ["o", "s", "^", "v", "D", "*"]
    colors = ["#7f7f7f", "#8c564b", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    # Subplot 1: FAI Retained % vs Budget Tier
    for idx, m in enumerate(methods):
        sub = benchmark_df[benchmark_df["method"] == m].sort_values("budget_pct")
        lw = 3.0 if "Proposed" in m else 1.8
        ms = 10 if "Proposed" in m else 6
        ax1.plot(
            sub["budget_pct"], sub["fai_retained_mean"],
            marker=markers[idx % len(markers)], color=colors[idx % len(colors)],
            linewidth=lw, markersize=ms, label=m
        )
        
    ax1.set_title("(a) Network Accessibility Retention vs. Budget Tier", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Rehabilitation Budget (% of Flooded Cost)", fontsize=11)
    ax1.set_ylabel("Vulnerability-Weighted FAI Retained (%)", fontsize=11)
    ax1.set_xticks(budget_tiers)
    ax1.set_xticklabels([f"{b}%" for b in budget_tiers])
    ax1.legend(loc="lower right", frameon=True, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Subplot 2: Hospital Isolated Population % vs Budget Tier
    for idx, m in enumerate(methods):
        sub = benchmark_df[benchmark_df["method"] == m].sort_values("budget_pct")
        lw = 3.0 if "Proposed" in m else 1.8
        ms = 10 if "Proposed" in m else 6
        ax2.plot(
            sub["budget_pct"], sub["hosp_isolated_pop_pct"],
            marker=markers[idx % len(markers)], color=colors[idx % len(colors)],
            linewidth=lw, markersize=ms, label=m
        )
        
    ax2.set_title("(b) Hospital Isolated Vulnerable Population vs. Budget Tier", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Rehabilitation Budget (% of Flooded Cost)", fontsize=11)
    ax2.set_ylabel("Isolated Vulnerable Population (%)", fontsize=11)
    ax2.set_xticks(budget_tiers)
    ax2.set_xticklabels([f"{b}%" for b in budget_tiers])
    ax2.legend(loc="upper right", frameon=True, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig3_benchmark_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"-> Saved Fig 3 to {out_path}")

# ==============================================================================
# 5. INTERACTIVE FOLIUM RESILIENCE MAP
# ==============================================================================

def generate_interactive_resilience_map(
    G: nx.Graph,
    villages_df: pd.DataFrame,
    facilities_dict: Dict[str, List[int]],
    candidates_df: pd.DataFrame,
    best_ga_corridors: Set[int]
):
    """
    Generates an interactive Folium map centered on the geographic region.
    Colors:
    - Gray: Baseline Intact Roads
    - Red (Dashed): Flooded Impassable Roads (Unprotected)
    - Emerald Green (Thick): GA-Fortified Resilient Corridors
    - Markers: Hospitals (Red Cross), Schools (Blue), Markets (Orange), Villages (Circle, colored by isolation)
    """
    # Geographic transformation: Map synthetic km (25x20) to real coordinates
    # Anchored in Sri Lanka / South Asia rural valley (7.91 N, 80.28 E)
    origin_lat, origin_lon = 7.915, 80.280
    scale = 0.009  # ~1 km per 0.009 deg
    
    m = folium.Map(location=[origin_lat + 10.0 * scale, origin_lon + 12.0 * scale], zoom_start=12, tiles="OpenStreetMap")
    
    # 1. Plot Roads
    for u, v, d in G.edges(data=True):
        cid = d["corridor_id"]
        ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
        vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
        
        lat_u, lon_u = origin_lat + uy * scale, origin_lon + ux * scale
        lat_v, lon_v = origin_lat + vy * scale, origin_lon + vx * scale
        
        is_flooded = d["closed_s2"] or d["closed_s3"]
        is_ga_protected = cid in best_ga_corridors
        
        if is_ga_protected:
            color = "#10b981"  # Emerald Green
            weight = 5.5
            dash_array = None
            tooltip = f"GA PROTECTED: Corridor {cid} ({d['highway']}) | Cost: ${d['cost_usd']:,.0f}"
        elif is_flooded:
            color = "#ef4444"  # Red
            weight = 3.0
            dash_array = "6, 6"
            tooltip = f"FLOODED IMPASSABLE: Corridor {cid} ({d['highway']}) | Water: {d['w_s3']:.0f}mm"
        else:
            color = "#9ca3af"  # Gray
            weight = 2.0
            dash_array = None
            tooltip = f"Intact Road: Corridor {cid} ({d['highway']}) | {d['length_km']:.1f}km"
            
        folium.PolyLine(
            locations=[[lat_u, lon_u], [lat_v, lon_v]],
            color=color,
            weight=weight,
            dash_array=dash_array,
            tooltip=tooltip
        ).add_to(m)
        
    # 2. Plot Facilities
    fac_styles = {
        "hospital": {"icon": "plus", "color": "red", "prefix": "fa"},
        "school": {"icon": "graduation-cap", "color": "blue", "prefix": "fa"},
        "market": {"icon": "shopping-cart", "color": "orange", "prefix": "fa"}
    }
    
    for cat, fn_list in facilities_dict.items():
        style = fac_styles[cat]
        for fn in fn_list:
            nd = G.nodes[fn]
            lat = origin_lat + nd["y"] * scale
            lon = origin_lon + nd["x"] * scale
            folium.Marker(
                location=[lat, lon],
                popup=f"<b>{cat.capitalize()} Hub</b> (Node {fn})",
                tooltip=f"{cat.capitalize()} Center",
                icon=folium.Icon(icon=style["icon"], color=style["color"], prefix=style["prefix"])
            ).add_to(m)
            
    # 3. Plot Villages
    for _, v_row in villages_df.iterrows():
        lat = origin_lat + v_row["y"] * scale
        lon = origin_lon + v_row["x"] * scale
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color="#1e3a8a",
            fill=True,
            fill_color="#60a5fa",
            fill_opacity=0.85,
            popup=f"<b>{v_row['name']}</b><br>Pop: {v_row['population']}<br>SVI: {v_row['svi']:.2f}<br>Eff Pop: {v_row['effective_pop']:.0f}",
            tooltip=v_row["name"]
        ).add_to(m)
        
    map_out = MAPS_DIR / "flood_resilience_map.html"
    m.save(str(map_out))
    print(f"-> Saved Interactive Resilience Map to {map_out}")

# ==============================================================================
# 6. ASSUMPTIONS.MD GENERATION
# ==============================================================================

def generate_assumptions_doc():
    """
    Generates the comprehensive ASSUMPTIONS.md document.
    """
    doc_content = """# EvoRoute System Assumptions & Technical Specifications

This document records the mathematical, physical, and algorithmic assumptions underpinning the **Flood-Resilient Road Priority Ranking System**.

---

## 1. Hydrological & Topographic Assumptions
1. **Height Above Nearest Drainage (HAND) Proxy**:
   * Ground surface elevation $z(e)$ and nearest river drainage elevation $z_{\\text{river}}(e)$ define edge relative height $h_{\\text{rel}}(e) = \\max(0, z(e) - z_{\\text{river}}(e))$.
   * Flood stages $H_1 = 1.0\\text{ m}$, $H_2 = 2.0\\text{ m}$, and $H_3 = 3.0\\text{ m}$ represent progressive hydrostatic water-stage rises (Moderate, Severe, Extreme climate scenarios), consistent with Sangeeta et al. (EGU 2025).
2. **Speed Degradation & Closure**:
   * Calculated via the empirical Mitropoulos (2025) / Pregnolato et al. (2017) model:
     $$v(w) = 0.0009w^2 - 0.5529w + 86.9448$$
   * Roads with water depth $w \\ge 200\\text{ mm}$ ($0.2\\text{ m}$) are completely impassable ($v = 0\\text{ km/h}, t = \\infty$).
   * Speeds under partial flooding are capped at $85\\%$ of free-flow velocity.

---

## 2. Accessibility & Demographic Weighting
1. **Social Vulnerability Index (SVI)**:
   * Rural settlements are weighted by demographic vulnerability $\\text{SVI}_i \\in [0.80, 1.40]$, reflecting higher proportions of children, elderly, disabled, and low-income residents near river basins (Sangeeta et al. 2025).
   * Effective vulnerable population: $P_i^{\\text{eff}} = P_i \\times \\text{SVI}_i$.
2. **Hansen Gravity Accessibility**:
   * Inverse travel-time impedance: $\\beta = 1.0$.
   * Facility hierarchy determined via Saaty's Analytic Hierarchy Process ($CR = 0.0032 \\le 0.10$):
     * Hospital ($W_{\\text{hosp}} = 0.648$)
     * School ($W_{\\text{school}} = 0.230$)
     * Market ($W_{\\text{market}} = 0.122$)
3. **Emergency Capacity Tracking**:
   * Destination capacities ($H_j$) from Akhlaghi et al. (arXiv:2302.07983) are tracked diagnostically to verify secondary facility offloading.

---

## 3. Soft Computing & Optimization Architecture
1. **Mamdani Fuzzy Inference System (Layer 1)**:
   * Universe of Discourse: Inputs $[0.0, 1.0]$ (step $0.01$), Output $[0.0, 100.0]$ (step $0.5$).
   * $27$ linguistic rules generated programmatically with safety-critical overrides for lifeline isolation.
   * Centroid defuzzification used for crisp priority scoring.
2. **Constrained Genetic Algorithm (Layer 2)**:
   * Binary encoding $\\mathbf{x} \\in \\{0, 1\\}^{19}$.
   * Domain-specific fuzzy repair operator guarantees $100\\%$ budget feasibility by prioritizing high $\\rho_j = \\frac{P_{\\text{fuzzy}, j} + 1.0}{\\text{Cost}_j}$ corridors.
   * Fitness balancing accessibility retention ($60\\%$), hospital reconnection ($40\\%$), and fuzzy preference ($15\\%$ regularizer).
   * Evaluated across 4 graduated budget tiers ($5\\%, 10\\%, 15\\%, 20\\%$) over 5 independent random seeds.
"""
    assumptions_path = RESULTS_DIR.parent / "ASSUMPTIONS.md"
    with open(assumptions_path, "w", encoding="utf-8") as f:
        f.write(doc_content)
    print(f"-> Generated {assumptions_path}")

# ==============================================================================
# PIPELINE EXECUTION ENTRY POINT
# ==============================================================================

def run_phase4_pipeline():
    """
    Executes Phase 4 end-to-end:
    - 6-way benchmarks
    - Ablation study
    - Fuzzy sensitivity analysis
    - Publication figures
    - Interactive folium map
    - Assumptions documentation
    """
    print("=" * 65)
    print("EVOROUTE: RUNNING PHASE 4 BENCHMARKS, ABLATION & VISUALIZATION")
    print("=" * 65)
    
    # Load Phase 2 & Phase 3 data contracts
    with open(RESULTS_DIR / "phase2_data.pkl", "rb") as f:
        p2 = pickle.load(f)
    with open(RESULTS_DIR / "phase3_data.pkl", "rb") as f:
        p3 = pickle.load(f)
        
    G = p2["G"]
    villages_df = p2["villages_df"]
    facilities_dict = p2["facilities_dict"]
    candidates_df = p3["candidates_df"]
    ga_results = p3["ga_results"]
    baseline_fai = p2["base_state"]["fai_net"]
    
    # Step 4.1: Six-Method Comparative Benchmark
    print("\n--- Step 4.1: Running 6-Way Comparative Benchmark across 4 Budgets ---")
    benchmark_df = run_six_benchmarks(
        G, villages_df, facilities_dict, candidates_df, ga_results, baseline_fai
    )
    benchmark_out = TABLES_DIR / "benchmark_results.csv"
    benchmark_df.to_csv(benchmark_out, index=False)
    print(f"-> Exported 6-Way Benchmark Results to {benchmark_out}")
    print("\nBenchmark Summary (10% Budget Tier):")
    b10_sub = benchmark_df[benchmark_df["budget_pct"] == 10]
    print(b10_sub[["method", "fai_retained_mean", "hosp_isolated_pop_pct", "budget_utilized_pct", "roads_protected"]].to_string(index=False))
    
    # Step 4.2: Ablation Study (Fuzzy GA vs. Vanilla GA)
    print("\n--- Step 4.2a: Running GA Ablation Study (10% Budget) ---")
    ablation_res = run_ablation_study(G, villages_df, facilities_dict, candidates_df, baseline_fai, budget_pct=0.10)
    print(f"-> Proposed Fuzzy-GA Fitness: {ablation_res['fuzzy_best_fitness']:.4f} (FAI Retained: {ablation_res['fuzzy_fai']:.1f}%)")
    print(f"-> Vanilla GA Fitness:        {ablation_res['vanilla_best_fitness']:.4f} (FAI Retained: {ablation_res['vanilla_fai']:.1f}%)")
    
    # Step 4.2b: Fuzzy Sensitivity Analysis
    print("\n--- Step 4.2b: Running Fuzzy Membership Sensitivity Analysis (+/-10%) ---")
    sens_res = run_fuzzy_sensitivity_analysis(candidates_df)
    print(f"-> Kendall's Tau (-10% Shift): tau = {sens_res['tau_minus_10pct']:.4f}")
    print(f"-> Kendall's Tau (+10% Shift): tau = {sens_res['tau_plus_10pct']:.4f}")
    
    # Step 4.3: Figures
    print("\n--- Step 4.3: Generating Publication Figures ---")
    plot_fig1_fuzzy_surfaces(candidates_df)
    plot_fig2_ga_convergence_and_ablation(ga_results, ablation_res)
    plot_fig3_benchmark_comparisons(benchmark_df)
    
    # Step 4.4: Interactive Folium Resilience Map
    print("\n--- Step 4.4: Generating Interactive Folium Resilience Map ---")
    best_20_cids = set(ga_results["20%"]["best_solution"]["best_corridor_ids"])
    generate_interactive_resilience_map(G, villages_df, facilities_dict, candidates_df, best_20_cids)
    
    # Step 4.5: ASSUMPTIONS.md
    print("\n--- Step 4.5: Writing ASSUMPTIONS.md Specification Document ---")
    generate_assumptions_doc()
    
    # Save final results pickle
    with open(RESULTS_DIR / "phase4_final_data.pkl", "wb") as f:
        pickle.dump({
            "benchmark_df": benchmark_df,
            "ablation_res": ablation_res,
            "sens_res": sens_res,
        }, f)
        
    print("\n" + "=" * 65)
    print("PHASE 4 COMPLETE: ALL EXPERIMENTS, BENCHMARKS, PLOTS & MAPS PRODUCED")
    print("=" * 65 + "\n")
    return benchmark_df

if __name__ == "__main__":
    run_phase4_pipeline()
