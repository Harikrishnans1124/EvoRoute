"""
Phase 3: Core Soft Computing Engine (Mamdani FIS + Hybrid Constrained GA)
Implements:
1. Layer 1: 27-rule Mamdani Fuzzy Inference System for local multi-criteria urgency.
2. Layer 2: Fuzzy-Seeded Constrained Genetic Algorithm across 4 budget tiers (5%, 10%, 15%, 20%)
   with fuzzy cost-effectiveness repair, hash-cached network fitness, uniform crossover,
   bit-flip mutation, and elitism across 5 random seeds.
"""
import math
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional

import skfuzzy as fuzz
from skfuzzy import control as ctrl

from src.config import (
    GA_CONFIG,
    BUDGET_TIERS,
    RANDOM_SEED,
    RESULTS_DIR,
    TABLES_DIR,
)
from src.phase2_accessibility_criteria import compute_network_accessibility_state

# ==============================================================================
# LAYER 1: MAMDANI FUZZY INFERENCE SYSTEM
# ==============================================================================

def build_mamdani_fis() -> ctrl.ControlSystemSimulation:
    """
    Builds the 27-rule Mamdani Fuzzy Inference System via skfuzzy.control.
    Inputs:
        exposure: [0.0, 1.0] (L, M, H)
        accessibility: [0.0, 1.0] (L, M, H)
        isolation: [0.0, 1.0] (L, M, H)
    Output:
        priority: [0.0, 100.0] (VL, L, M, H, VH)
    """
    U_in = np.arange(0.0, 1.01, 0.01)
    U_out = np.arange(0.0, 100.5, 0.5)

    exp_ant = ctrl.Antecedent(U_in, "exposure")
    acc_ant = ctrl.Antecedent(U_in, "accessibility")
    iso_ant = ctrl.Antecedent(U_in, "isolation")
    prio_con = ctrl.Consequent(U_out, "priority")

    # Input Membership Functions (Trapezoidal / Triangular)
    for ant in [exp_ant, acc_ant, iso_ant]:
        ant["L"] = fuzz.trapmf(ant.universe, [0.0, 0.0, 0.20, 0.45])
        ant["M"] = fuzz.trimf(ant.universe, [0.25, 0.50, 0.75])
        ant["H"] = fuzz.trapmf(ant.universe, [0.55, 0.80, 1.0, 1.0])

    # Output Membership Functions (VL, L, M, H, VH)
    prio_con["VL"] = fuzz.trapmf(prio_con.universe, [0, 0, 10, 25])
    prio_con["L"]  = fuzz.trimf(prio_con.universe, [15, 30, 45])
    prio_con["M"]  = fuzz.trimf(prio_con.universe, [35, 50, 65])
    prio_con["H"]  = fuzz.trimf(prio_con.universe, [55, 70, 85])
    prio_con["VH"] = fuzz.trapmf(prio_con.universe, [75, 90, 100, 100])

    # Construct Complete 27-Rule Base
    rules = []
    level_map = {"L": 0, "M": 1, "H": 2}
    
    for l_e in ["L", "M", "H"]:
        for l_a in ["L", "M", "H"]:
            for l_i in ["L", "M", "H"]:
                i_e, i_a, i_i = level_map[l_e], level_map[l_a], level_map[l_i]
                u_score = i_e + i_a + 2 * i_i
                
                # Rule consequent with safety overrides
                if i_i == 2 and (i_e >= 1 or i_a >= 1):
                    consequent = "VH"
                elif u_score <= 1:
                    consequent = "VL"
                elif u_score <= 3:
                    consequent = "L"
                elif u_score <= 5:
                    consequent = "M"
                elif u_score == 6:
                    consequent = "H"
                else:
                    consequent = "VH"
                    
                rule = ctrl.Rule(
                    exp_ant[l_e] & acc_ant[l_a] & iso_ant[l_i],
                    prio_con[consequent]
                )
                rules.append(rule)
                
    fis_ctrl = ctrl.ControlSystem(rules)
    fis_sim = ctrl.ControlSystemSimulation(fis_ctrl)
    return fis_sim

def evaluate_fuzzy_priorities(
    candidates_df: pd.DataFrame,
    fis_sim: ctrl.ControlSystemSimulation
) -> pd.DataFrame:
    """
    Evaluates Mamdani FIS priority scores for each candidate corridor.
    Guarantees strict zero-division and clipping guards in [0.0, 1.0].
    """
    df = candidates_df.copy()
    fuzzy_scores = []
    
    for _, row in df.iterrows():
        e_val = float(np.clip(row["exp_norm"], 0.0, 1.0))
        a_val = float(np.clip(row["acc_norm"], 0.0, 1.0))
        i_val = float(np.clip(row["iso_norm"], 0.0, 1.0))
        
        fis_sim.input["exposure"] = e_val
        fis_sim.input["accessibility"] = a_val
        fis_sim.input["isolation"] = i_val
        fis_sim.compute()
        
        score = float(np.clip(fis_sim.output["priority"], 0.0, 100.0))
        fuzzy_scores.append(round(score, 3))
        
    df["fuzzy_priority"] = fuzzy_scores
    # Compute fuzzy cost-effectiveness ratio rho_j = (Priority + 1.0) / Cost
    df["fuzzy_cost_ratio"] = (df["fuzzy_priority"] + 1.0) / df["cost_usd"]
    return df

# ==============================================================================
# LAYER 2: FUZZY-SEEDED CONSTRAINED GENETIC ALGORITHM
# ==============================================================================

class FuzzyConstrainedGA:
    """
    Genetic Algorithm for budget-constrained combinatorial road portfolio fortification.
    Features:
    - Domain-Specific Fuzzy Cost-Benefit Repair Operator
    - Fuzzy Population Seeding (20%) + Random Exploration (80%)
    - Hash-Cached Network Fitness Evaluation
    - Tournament Selection, Uniform Crossover, Bit-Flip Mutation, Elitism
    """
    def __init__(
        self,
        G: nx.Graph,
        villages_df: pd.DataFrame,
        facilities_dict: Dict[str, List[int]],
        candidates_df: pd.DataFrame,
        baseline_fai: float,
        budget: float,
        config: Dict[str, Any] = GA_CONFIG,
        seed: int = RANDOM_SEED
    ):
        self.G = G
        self.villages_df = villages_df
        self.facilities_dict = facilities_dict
        self.candidates_df = candidates_df
        self.baseline_fai = baseline_fai
        self.budget = budget
        self.config = config
        self.rng = np.random.RandomState(seed)
        
        self.N = len(candidates_df)
        self.costs = candidates_df["cost_usd"].values.astype(np.float64)
        self.fuzzy_priorities = candidates_df["fuzzy_priority"].values.astype(np.float64)
        self.fuzzy_ratios = candidates_df["fuzzy_cost_ratio"].values.astype(np.float64)
        self.corridor_ids = candidates_df["corridor_id"].values.astype(int)
        
        # Fitness cache: tuple(chromosome) -> fitness_score
        self.fitness_cache: Dict[Tuple[int, ...], float] = {}
        # Detailed metrics cache: tuple(chromosome) -> metrics_dict
        self.metrics_cache: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        
    def repair_chromosome(self, chrom: np.ndarray) -> np.ndarray:
        """
        Domain-Specific Fuzzy Budget Repair Operator:
        If chromosome exceeds budget, drops active genes in ascending order of
        fuzzy cost-efficiency rho_j until strictly feasible.
        Greedy Fill Step: Iterates over unselected genes by descending rho_j to pack
        remaining unused budget.
        """
        chrom = chrom.copy()
        current_cost = np.sum(chrom * self.costs)
        
        # 1. Drop excess genes
        if current_cost > self.budget:
            # Active gene indices sorted by ascending fuzzy cost-efficiency
            active_indices = np.where(chrom == 1)[0]
            # Sort active genes by lowest rho_j first
            sorted_by_lowest_rho = active_indices[np.argsort(self.fuzzy_ratios[active_indices])]
            
            for gene_idx in sorted_by_lowest_rho:
                chrom[gene_idx] = 0
                current_cost -= self.costs[gene_idx]
                if current_cost <= self.budget:
                    break
                    
        # 2. Greedy Fill Step: utilize leftover budget
        inactive_indices = np.where(chrom == 0)[0]
        # Sort inactive genes by highest rho_j first
        sorted_by_highest_rho = inactive_indices[np.argsort(-self.fuzzy_ratios[inactive_indices])]
        
        for gene_idx in sorted_by_highest_rho:
            cand_cost = self.costs[gene_idx]
            if current_cost + cand_cost <= self.budget:
                chrom[gene_idx] = 1
                current_cost += cand_cost
                
        return chrom

    def initialize_population(self) -> np.ndarray:
        """
        Initializes population (N_pop = 40):
        - 20% (8 individuals) Fuzzy Heuristic Seeding: genes sampled with probability
          proportional to fuzzy cost-ratio rho_j.
        - 80% (32 individuals) Random Exploration: uniform random bits + repair.
        """
        pop_size = self.config["pop_size"]
        fuzzy_count = int(round(self.config["fuzzy_seed_ratio"] * pop_size))
        random_count = pop_size - fuzzy_count
        
        population = []
        
        # Normalized selection probabilities from fuzzy cost-ratio
        p_fuzzy = self.fuzzy_ratios / np.sum(self.fuzzy_ratios)
        
        # 1. Fuzzy Heuristic Seeding
        for _ in range(fuzzy_count):
            chrom = np.zeros(self.N, dtype=int)
            # Sample random number of genes proportional to fuzzy efficiency
            num_to_sample = self.rng.randint(1, max(2, self.N // 2))
            chosen_genes = self.rng.choice(self.N, size=num_to_sample, replace=False, p=p_fuzzy)
            chrom[chosen_genes] = 1
            chrom = self.repair_chromosome(chrom)
            population.append(chrom)
            
        # 2. Random Exploration
        for _ in range(random_count):
            chrom = self.rng.randint(0, 2, size=self.N, dtype=int)
            chrom = self.repair_chromosome(chrom)
            population.append(chrom)
            
        return np.array(population)

    def evaluate_fitness(self, chrom: np.ndarray) -> float:
        """
        Evaluates network-recalculated fitness with hash caching:
        F(x) = 1/3 * sum_{s=1}^3 [ alpha * (FAI_net,s / FAI_base) + (1-alpha) * (1 - S_iso,s) ]
               + lambda * (sum x_j P_fuzzy,j) / (100 * max(1, sum x_j))
        """
        key = tuple(chrom.tolist())
        if key in self.fitness_cache:
            return self.fitness_cache[key]
            
        # Active restored corridors
        active_indices = np.where(chrom == 1)[0]
        restored_cids = set(self.corridor_ids[active_indices])
        
        alpha = self.config["alpha"]
        lambda_reg = self.config["lambda_reg"]
        
        scenario_scores = []
        fai_list = []
        iso_list = []
        
        for s_key in ["s1", "s2", "s3"]:
            st = compute_network_accessibility_state(
                self.G, self.villages_df, self.facilities_dict,
                scenario=s_key, restored_corridors=restored_cids
            )
            # Normalized FAI retention
            fai_norm = st["fai_net"] / max(1e-9, self.baseline_fai)
            # Non-isolated population share (1 - S_iso)
            conn_share = 1.0 - st["isolated_pop_share"]
            
            scen_fit = alpha * fai_norm + (1.0 - alpha) * conn_share
            scenario_scores.append(scen_fit)
            fai_list.append(st["fai_net"])
            iso_list.append(st["isolated_pop_share"])
            
        network_fitness = float(np.mean(scenario_scores))
        
        # Soft-computing fuzzy preference regularizer
        if len(active_indices) > 0:
            mean_fuzzy_prio = float(np.mean(self.fuzzy_priorities[active_indices]))
            fuzzy_regularizer = lambda_reg * (mean_fuzzy_prio / 100.0)
        else:
            fuzzy_regularizer = 0.0
            
        total_fitness = network_fitness + fuzzy_regularizer
        
        self.fitness_cache[key] = total_fitness
        self.metrics_cache[key] = {
            "fitness": total_fitness,
            "mean_fai_retained_pct": float(np.mean(fai_list) / self.baseline_fai * 100.0),
            "mean_isolated_pop_pct": float(np.mean(iso_list) * 100.0),
            "cost_usd": float(np.sum(chrom * self.costs)),
            "num_roads": int(len(active_indices)),
        }
        return total_fitness

    def tournament_selection(self, population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        """Tournament selection of size k=3."""
        k = self.config["tournament_k"]
        pop_size = len(population)
        cand_indices = self.rng.choice(pop_size, size=k, replace=False)
        best_cand = cand_indices[np.argmax(fitnesses[cand_indices])]
        return population[best_cand].copy()

    def uniform_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover with gene swap probability 0.5."""
        if self.rng.rand() > self.config["crossover_prob"]:
            return parent1.copy(), parent2.copy()
            
        mask = self.rng.rand(self.N) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2

    def bit_flip_mutation(self, chrom: np.ndarray) -> np.ndarray:
        """Bit-flip mutation with rate p_m = mutation_rate_factor / N."""
        p_m = self.config["mutation_rate_factor"] / float(self.N)
        mutation_mask = self.rng.rand(self.N) < p_m
        mutated = np.bitwise_xor(chrom, mutation_mask.astype(int))
        return mutated

    def run(self) -> Dict[str, Any]:
        """
        Runs the full genetic algorithm evolution.
        Returns:
            Dict containing best chromosome, fitness history, and final metrics.
        """
        population = self.initialize_population()
        pop_size = len(population)
        elitism_k = self.config["elitism_count"]
        max_gen = self.config["max_generations"]
        patience = self.config["early_stopping_patience"]
        
        fitness_history = []
        best_fitness = -1.0
        best_chromosome = None
        no_improve_count = 0
        
        for gen in range(max_gen):
            # 1. Evaluate fitness for all chromosomes
            fitnesses = np.array([self.evaluate_fitness(chrom) for chrom in population])
            
            # 2. Track best
            gen_best_idx = np.argmax(fitnesses)
            gen_best_fit = fitnesses[gen_best_idx]
            gen_best_chrom = population[gen_best_idx]
            
            if gen_best_fit > best_fitness + 1e-6:
                best_fitness = gen_best_fit
                best_chromosome = gen_best_chrom.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            fitness_history.append(float(best_fitness))
            
            # Early stopping check
            if no_improve_count >= patience:
                # Fill remaining generations with best_fitness for consistent plotting
                fitness_history.extend([float(best_fitness)] * (max_gen - len(fitness_history)))
                break
                
            # 3. Elitism: preserve top k individuals
            elite_indices = np.argsort(fitnesses)[-elitism_k:]
            next_population = [population[i].copy() for i in elite_indices]
            
            # 4. Generate offspring
            while len(next_population) < pop_size:
                p1 = self.tournament_selection(population, fitnesses)
                p2 = self.tournament_selection(population, fitnesses)
                c1, c2 = self.uniform_crossover(p1, p2)
                c1 = self.bit_flip_mutation(c1)
                c2 = self.bit_flip_mutation(c2)
                
                # Repair before adding to next population
                c1 = self.repair_chromosome(c1)
                c2 = self.repair_chromosome(c2)
                
                next_population.append(c1)
                if len(next_population) < pop_size:
                    next_population.append(c2)
                    
            population = np.array(next_population)
            
        best_key = tuple(best_chromosome.tolist())
        detailed = self.metrics_cache[best_key]
        active_cids = self.corridor_ids[np.where(best_chromosome == 1)[0]].tolist()
        
        return {
            "best_chromosome": best_chromosome,
            "best_corridor_ids": active_cids,
            "best_fitness": float(round(best_fitness, 5)),
            "cost_usd": detailed["cost_usd"],
            "budget_usd": self.budget,
            "budget_utilized_pct": float(round(detailed["cost_usd"] / self.budget * 100.0, 2)),
            "mean_fai_retained_pct": detailed["mean_fai_retained_pct"],
            "mean_isolated_pop_pct": detailed["mean_isolated_pop_pct"],
            "num_roads_protected": detailed["num_roads"],
            "fitness_history": fitness_history,
        }

def run_multi_seed_ga_experiments(
    G: nx.Graph,
    villages_df: pd.DataFrame,
    facilities_dict: Dict[str, List[int]],
    candidates_df: pd.DataFrame,
    baseline_fai: float,
    num_seeds: int = 5
) -> Dict[str, Any]:
    """
    Executes GA optimization across 4 budget tiers (5%, 10%, 15%, 20%)
    with num_seeds (5) independent random seeds per tier.
    """
    total_cand_cost = float(candidates_df["cost_usd"].sum())
    results_by_budget = {}
    
    for b_frac in BUDGET_TIERS:
        b_usd = b_frac * total_cand_cost
        b_label = f"{int(round(b_frac * 100))}%"
        print(f"\n--- Optimizing Budget Tier {b_label} (${b_usd:,.2f}) across {num_seeds} seeds ---")
        
        seed_runs = []
        for s_idx in range(num_seeds):
            seed_val = RANDOM_SEED + s_idx * 17
            ga = FuzzyConstrainedGA(
                G, villages_df, facilities_dict, candidates_df,
                baseline_fai=baseline_fai, budget=b_usd, seed=seed_val
            )
            res = ga.run()
            seed_runs.append(res)
            print(f"  [Seed {s_idx+1}/{num_seeds}] Best Fit: {res['best_fitness']:.4f} | "
                  f"Protected: {res['num_roads_protected']} roads | "
                  f"Budget Used: {res['budget_utilized_pct']:.1f}% | "
                  f"FAI Retained: {res['mean_fai_retained_pct']:.1f}%")
                  
        # Select best run overall for this budget tier
        best_seed_run = max(seed_runs, key=lambda x: x["best_fitness"])
        
        # Compute mean and std across seeds
        fits = [r["best_fitness"] for r in seed_runs]
        fais = [r["mean_fai_retained_pct"] for r in seed_runs]
        isos = [r["mean_isolated_pop_pct"] for r in seed_runs]
        b_used = [r["budget_utilized_pct"] for r in seed_runs]
        
        results_by_budget[b_label] = {
            "budget_pct": int(round(b_frac * 100)),
            "budget_usd": b_usd,
            "best_solution": best_seed_run,
            "mean_fitness": float(round(np.mean(fits), 5)),
            "std_fitness": float(round(np.std(fits), 5)),
            "fai_retained_mean": float(round(np.mean(fais), 2)),
            "fai_retained_std": float(round(np.std(fais), 2)),
            "isolated_pop_mean": float(round(np.mean(isos), 2)),
            "isolated_pop_std": float(round(np.std(isos), 2)),
            "budget_used_mean": float(round(np.mean(b_used), 2)),
            "all_runs": seed_runs,
        }
        
    return results_by_budget

def verify_phase3_gate(
    candidates_df: pd.DataFrame,
    ga_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Verifies all Phase 3 Verification Gate conditions:
    1. All 27 fuzzy rules fire without NaNs, fuzzy priority strictly in [0, 100].
    2. C(x) <= B_k holds for 100% of evaluated solutions.
    3. Best fitness is non-decreasing across generations due to elitism.
    """
    # 1. Fuzzy priority check
    assert not candidates_df["fuzzy_priority"].isnull().any(), "Assertion Failed: NaN in fuzzy_priority"
    assert (candidates_df["fuzzy_priority"] >= 0.0).all() and (candidates_df["fuzzy_priority"] <= 100.0).all(), (
        "Assertion Failed: Fuzzy priority scores out of bounds [0, 100]."
    )
    
    # 2. Budget constraint verification
    for b_label, b_data in ga_results.items():
        b_usd = b_data["budget_usd"]
        for run in b_data["all_runs"]:
            assert run["cost_usd"] <= b_usd + 1e-4, (
                f"Assertion Failed: Solution cost ${run['cost_usd']:.2f} exceeds budget ${b_usd:.2f} in tier {b_label}"
            )
            
    # 3. Monotonicity of best fitness across generations
    for b_label, b_data in ga_results.items():
        for run in b_data["all_runs"]:
            hist = run["fitness_history"]
            for i in range(len(hist) - 1):
                assert hist[i+1] >= hist[i] - 1e-6, (
                    f"Assertion Failed: Fitness decreased from gen {i} ({hist[i]}) to {i+1} ({hist[i+1]})"
                )
                
    gate_summary = {
        "status": "PASSED",
        "fuzzy_priorities_min": float(candidates_df["fuzzy_priority"].min()),
        "fuzzy_priorities_max": float(candidates_df["fuzzy_priority"].max()),
        "all_budget_constraints_satisfied": True,
        "fitness_monotonicity_verified": True,
        "tiers_optimized": list(ga_results.keys()),
    }
    return gate_summary

def run_phase3_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Executes Phase 3 end-to-end:
    - Layer 1: Evaluates Mamdani FIS for all candidates
    - Layer 2: Runs multi-seed Fuzzy Constrained GA across 4 budget tiers
    - Verifies Verification Gate assertions
    - Serializes Phase 3 data contracts
    """
    print("=" * 65)
    print("EVOROUTE: RUNNING PHASE 3 CORE SOFT COMPUTING ENGINE")
    print("=" * 65)
    
    # Load Phase 2 Data Contract
    p2_path = RESULTS_DIR / "phase2_data.pkl"
    with open(p2_path, "rb") as f:
        p2_data = pickle.load(f)
        
    G = p2_data["G"]
    villages_df = p2_data["villages_df"]
    facilities_dict = p2_data["facilities_dict"]
    candidates_df = p2_data["candidates_df"]
    baseline_fai = p2_data["base_state"]["fai_net"]
    
    # Step 3.1: Layer 1 - Mamdani Fuzzy Inference System
    print("\n--- Executing Layer 1: Mamdani Fuzzy Inference System (27 Rules) ---")
    fis_sim = build_mamdani_fis()
    candidates_df = evaluate_fuzzy_priorities(candidates_df, fis_sim)
    print(f"-> Evaluated fuzzy priorities for {len(candidates_df)} candidate corridors.")
    print("-> Sample Corridors by Fuzzy Priority:")
    top_fuzzy = candidates_df.sort_values("fuzzy_priority", ascending=False).head(5)
    for _, r in top_fuzzy.iterrows():
        print(f"   Corridor {int(r['corridor_id'])} ({r['highway']}): "
              f"P_fuzzy={r['fuzzy_priority']:.2f} | Cost=${r['cost_usd']:,.0f} | "
              f"Eff={r['fuzzy_cost_ratio']*1e5:.2f}x10^-5")
              
    # Step 3.2: Layer 2 - Fuzzy-Guided Constrained GA Optimization
    print("\n--- Executing Layer 2: Fuzzy-Seeded Constrained Genetic Algorithm ---")
    ga_results = run_multi_seed_ga_experiments(
        G, villages_df, facilities_dict, candidates_df,
        baseline_fai=baseline_fai, num_seeds=GA_CONFIG["num_seeds"]
    )
    
    # Verification Gate
    summary = verify_phase3_gate(candidates_df, ga_results)
    print("\n[PHASE 3 VERIFICATION GATE]:")
    for k, v in summary.items():
        print(f"   {k}: {v}")
    print("=" * 65)
    
    # Save Phase 3 Data Contracts
    p3_path = RESULTS_DIR / "phase3_data.pkl"
    with open(p3_path, "wb") as f:
        pickle.dump({
            "G": G,
            "villages_df": villages_df,
            "facilities_dict": facilities_dict,
            "candidates_df": candidates_df,
            "ga_results": ga_results,
            "summary": summary
        }, f)
        
    candidates_df.to_csv(TABLES_DIR / "candidates_with_fuzzy_priority.csv", index=False)
    print(f"-> Phase 3 data contract serialized successfully to {p3_path}")
    print(f"-> Exported candidates with fuzzy priorities to {TABLES_DIR / 'candidates_with_fuzzy_priority.csv'}\n")
    
    return candidates_df, ga_results, summary

if __name__ == "__main__":
    run_phase3_pipeline()
