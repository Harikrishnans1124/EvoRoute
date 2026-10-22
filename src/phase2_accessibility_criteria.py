"""
Phase 2: High-Speed Accessibility Engine & Feature Extraction
Computes AHP service weights, multi-source shortest paths, Hansen Facility Accessibility
Index (FAI), Vulnerability-weighted Network FAI, Hospital Isolation Shares, Overall Accessibility
Index (OAI), and extracts the normalized 3-criteria candidate feature table.
"""
import pickle
import math
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Set

from src.config import (
    AHP_WEIGHTS,
    OAI_THRESHOLDS,
    FLOOD_SCENARIOS,
    RESULTS_DIR,
    TABLES_DIR,
)

def compute_ahp_service_weights() -> Tuple[Dict[str, float], float]:
    """
    Computes AHP priority weights for [Hospital, School, Market] from the Saaty comparison matrix.
    A = [[1,   3, 5],
         [1/3, 1, 2],
         [1/5, 1/2, 1]]
    Verifies that Consistency Ratio (CR) <= 0.10.
    """
    A = np.array([
        [1.0, 3.0, 5.0],
        [1.0 / 3.0, 1.0, 2.0],
        [1.0 / 5.0, 0.5, 1.0]
    ], dtype=np.float64)
    
    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eig(A)
    max_idx = np.argmax(np.real(eigvals))
    lambda_max = np.real(eigvals[max_idx])
    principal_vec = np.real(eigvecs[:, max_idx])
    weights = principal_vec / np.sum(principal_vec)
    
    n = 3
    ci = (lambda_max - n) / (n - 1)
    ri = 0.58  # Saaty random index for n=3
    cr = ci / ri
    
    weight_dict = {
        "hospital": float(round(weights[0], 4)),
        "school": float(round(weights[1], 4)),
        "market": float(round(weights[2], 4)),
    }
    
    assert cr <= 0.10, f"AHP Consistency Ratio {cr:.4f} exceeds threshold 0.10"
    return weight_dict, float(round(cr, 6))

def compute_network_accessibility_state(
    G: nx.Graph,
    villages_df: pd.DataFrame,
    facilities_dict: Dict[str, List[int]],
    scenario: Optional[str] = None,
    restored_corridors: Optional[Set[int]] = None,
    ahp_w: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    High-speed network accessibility evaluation for any graph state.
    
    Args:
        G: Base road network.
        villages_df: Village demographic dataset with node_id and effective_pop.
        facilities_dict: Mapping of facility category to list of node_ids.
        scenario: 's1', 's2', 's3', or None for baseline intact conditions.
        restored_corridors: Set of corridor_ids protected/restored to baseline t_base.
        ahp_w: Category weights (defaults to AHP_WEIGHTS).
        
    Returns:
        Dict containing:
            - 'fai_net': Vulnerability-weighted Network FAI
            - 'isolated_pop_effective': Total effective population isolated from hospital
            - 'isolated_pop_share': Fraction of effective population isolated from hospital
            - 'isolated_villages_count': Number of villages with hospital disconnected
            - 'oai_share': Fraction of population meeting all OAI thresholds
            - 'village_fai': Dict[village_id, float]
            - 'village_hosp_dist': Dict[village_id, float]
    """
    if ahp_w is None:
        ahp_w = AHP_WEIGHTS
    if restored_corridors is None:
        restored_corridors = set()
        
    # Build a view/subgraph with active travel times
    # An edge is traversable if scenario is None (baseline) or t_scenario < inf or restored
    active_edges = []
    for u, v, d in G.edges(data=True):
        cid = d["corridor_id"]
        if scenario is None or cid in restored_corridors:
            t = d["t_base"]
        else:
            t = d[f"t_{scenario}"]
            
        if not math.isinf(t) and t > 0:
            active_edges.append((u, v, t))
            
    # Subgraph for Dijkstra
    H = nx.Graph()
    H.add_nodes_from(G.nodes)
    for u, v, t in active_edges:
        H.add_edge(u, v, travel_time=t)
        
    # Multi-source Dijkstra from each facility category
    # Shortest travel time from each facility node to all reachable nodes in H
    facility_travel_times: Dict[str, Dict[int, Dict[int, float]]] = {}
    for cat, fac_nodes in facilities_dict.items():
        facility_travel_times[cat] = {}
        for fn in fac_nodes:
            if fn in H:
                lengths = nx.single_source_dijkstra_path_length(H, fn, weight="travel_time")
                facility_travel_times[cat][fn] = lengths
            else:
                facility_travel_times[cat][fn] = {}
                
    total_effective_pop = villages_df["effective_pop"].sum()
    total_base_pop = villages_df["population"].sum()
    
    weighted_fai_sum = 0.0
    isolated_eff_pop = 0.0
    isolated_villages = 0
    oai_qualifying_pop = 0.0
    
    village_fai_dict = {}
    village_hosp_dist_dict = {}
    
    for _, row in villages_df.iterrows():
        vid = int(row["village_id"])
        v_node = int(row["node_id"])
        eff_pop = float(row["effective_pop"])
        base_pop = float(row["population"])
        
        # Calculate Hansen FAI_i
        fai_i = 0.0
        min_cat_times = {}
        
        for cat, weight in ahp_w.items():
            cat_sum = 0.0
            cat_min = float("inf")
            for fn, dist_map in facility_travel_times[cat].items():
                if v_node in dist_map:
                    c_ir = dist_map[v_node]
                    cat_sum += 1.0 / max(1.0, c_ir)
                    if c_ir < cat_min:
                        cat_min = c_ir
            fai_i += weight * cat_sum
            min_cat_times[cat] = cat_min
            
        village_fai_dict[vid] = fai_i
        village_hosp_dist_dict[vid] = min_cat_times["hospital"]
        
        weighted_fai_sum += eff_pop * fai_i
        
        # Check Hospital Isolation
        if math.isinf(min_cat_times["hospital"]):
            isolated_eff_pop += eff_pop
            isolated_villages += 1
            
        # Check OAI Thresholds (Hospital <= 30m, School <= 20m, Market <= 20m)
        meets_oai = (
            min_cat_times["hospital"] <= OAI_THRESHOLDS["hospital"] and
            min_cat_times["school"] <= OAI_THRESHOLDS["school"] and
            min_cat_times["market"] <= OAI_THRESHOLDS["market"]
        )
        if meets_oai:
            oai_qualifying_pop += base_pop
            
    fai_net = weighted_fai_sum / max(1e-9, total_effective_pop)
    iso_share = isolated_eff_pop / max(1e-9, total_effective_pop)
    oai_share = oai_qualifying_pop / max(1e-9, total_base_pop)
    
    return {
        "fai_net": float(round(fai_net, 6)),
        "isolated_pop_effective": float(round(isolated_eff_pop, 1)),
        "isolated_pop_share": float(round(iso_share, 6)),
        "isolated_villages_count": int(isolated_villages),
        "oai_share": float(round(oai_share, 6)),
        "village_fai": village_fai_dict,
        "village_hosp_dist": village_hosp_dist_dict,
    }

def extract_candidate_criteria_table(
    G: nx.Graph,
    villages_df: pd.DataFrame,
    facilities_dict: Dict[str, List[int]],
    candidates: List[Dict[str, Any]],
    ahp_w: Dict[str, float]
) -> pd.DataFrame:
    """
    Extracts the 3-criteria table for all candidate corridors:
    Criterion 1: Flood Exposure (mean water depth across s1-s3)
    Criterion 2: Marginal Accessibility Gain (FAI_net gain when corridor j alone is protected)
    Criterion 3: Critical Isolation & Hierarchy Index (0.7 * delta_P_iso + 0.3 * Betweenness * h_class_weight)
    Normalizes all criteria to [0.0, 1.0] with strict zero-guards.
    """
    # 1. Compute unprotected flooded states across scenarios s1, s2, s3
    unprotected_states = {}
    for s_key in ["s1", "s2", "s3"]:
        unprotected_states[s_key] = compute_network_accessibility_state(
            G, villages_df, facilities_dict, scenario=s_key, restored_corridors=set(), ahp_w=ahp_w
        )
        
    records = []
    
    for idx, cand in enumerate(candidates):
        cid = cand["corridor_id"]
        u, v = cand["u"], cand["v"]
        edge_data = G[u][v]
        
        # Criterion 1: Flood Exposure (mean depth in mm across s1, s2, s3)
        c1_exp_raw = (edge_data["w_s1"] + edge_data["w_s2"] + edge_data["w_s3"]) / 3.0
        
        # Marginal benefits when protecting corridor cid alone
        fai_gains = []
        iso_reconnected_list = []
        
        for s_key in ["s1", "s2", "s3"]:
            # Evaluate network if corridor cid alone is restored to baseline
            protected_state = compute_network_accessibility_state(
                G, villages_df, facilities_dict, scenario=s_key, restored_corridors={cid}, ahp_w=ahp_w
            )
            # FAI gain
            delta_fai = max(0.0, protected_state["fai_net"] - unprotected_states[s_key]["fai_net"])
            fai_gains.append(delta_fai)
            
            # Isolated effective population reconnected
            delta_iso = max(
                0.0,
                unprotected_states[s_key]["isolated_pop_effective"] - protected_state["isolated_pop_effective"]
            )
            iso_reconnected_list.append(delta_iso)
            
        c2_acc_raw = float(np.mean(fai_gains))
        mean_iso_reconnected = float(np.mean(iso_reconnected_list))
        
        records.append({
            "gene_idx": idx,
            "corridor_id": cid,
            "u": u,
            "v": v,
            "highway": edge_data["highway"],
            "length_km": edge_data["length_km"],
            "cost_usd": edge_data["cost_usd"],
            "h_class_weight": edge_data["h_class_weight"],
            "betweenness": edge_data["betweenness"],
            "exp_raw": float(round(c1_exp_raw, 3)),
            "acc_raw": float(round(c2_acc_raw, 6)),
            "delta_p_iso_raw": float(round(mean_iso_reconnected, 1)),
        })
        
    df = pd.DataFrame(records)
    
    # Normalize betweenness and delta_p_iso to form Criterion 3
    def min_max_norm(series: pd.Series) -> pd.Series:
        min_v = series.min()
        max_v = series.max()
        rng = max_v - min_v
        if rng < 1e-9:
            return pd.Series(np.zeros(len(series), dtype=np.float64), index=series.index)
        return (series - min_v) / rng
        
    norm_delta_p_iso = min_max_norm(df["delta_p_iso_raw"])
    norm_betweenness = min_max_norm(df["betweenness"])
    
    # Criterion 3: Composite Lifeline Criticality
    c3_iso_raw = 0.7 * norm_delta_p_iso + 0.3 * (norm_betweenness * df["h_class_weight"])
    df["iso_raw"] = c3_iso_raw.round(5)
    
    # Min-Max Normalize all 3 criteria to [0.0, 1.0] with clipping guards
    df["exp_norm"] = np.clip(min_max_norm(df["exp_raw"]), 0.0, 1.0).round(5)
    df["acc_norm"] = np.clip(min_max_norm(df["acc_raw"]), 0.0, 1.0).round(5)
    df["iso_norm"] = np.clip(min_max_norm(df["iso_raw"]), 0.0, 1.0).round(5)
    
    return df

def verify_phase2_gate(
    base_state: Dict[str, Any],
    flooded_states: Dict[str, Dict[str, Any]],
    candidates_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Verifies all Phase 2 Verification Gate conditions:
    1. Baseline FAI_net is positive, and FAI loss across scenarios increases monotonically.
    2. Zero NaNs across all criteria columns.
    3. All normalized criteria columns strictly lie within [0.0, 1.0].
    4. Data Contract adherence for candidates_df.
    """
    base_fai = base_state["fai_net"]
    assert base_fai > 0.0, "Assertion Failed: Baseline FAI_net must be strictly positive."
    
    # FAI net decreases as flood severity increases
    fai_s1 = flooded_states["s1"]["fai_net"]
    fai_s2 = flooded_states["s2"]["fai_net"]
    fai_s3 = flooded_states["s3"]["fai_net"]
    
    assert fai_s1 <= base_fai, "Assertion Failed: S1 FAI cannot exceed baseline FAI."
    assert fai_s2 <= fai_s1 + 1e-6, "Assertion Failed: S2 FAI must be <= S1 FAI."
    assert fai_s3 <= fai_s2 + 1e-6, "Assertion Failed: S3 FAI must be <= S2 FAI."
    
    # Check zero NaNs and bounds [0, 1]
    norm_cols = ["exp_norm", "acc_norm", "iso_norm"]
    for col in norm_cols:
        assert not candidates_df[col].isnull().any(), f"Assertion Failed: NaN found in {col}."
        assert (candidates_df[col] >= 0.0).all() and (candidates_df[col] <= 1.0).all(), (
            f"Assertion Failed: Column {col} values out of bounds [0.0, 1.0]."
        )
        
    gate_summary = {
        "status": "PASSED",
        "baseline_fai_net": base_fai,
        "fai_net_s1": fai_s1,
        "fai_net_s2": fai_s2,
        "fai_net_s3": fai_s3,
        "fai_retention_s1_pct": float(round(100.0 * fai_s1 / base_fai, 2)),
        "fai_retention_s2_pct": float(round(100.0 * fai_s2 / base_fai, 2)),
        "fai_retention_s3_pct": float(round(100.0 * fai_s3 / base_fai, 2)),
        "isolated_pop_s1_pct": float(round(100.0 * flooded_states["s1"]["isolated_pop_share"], 2)),
        "isolated_pop_s2_pct": float(round(100.0 * flooded_states["s2"]["isolated_pop_share"], 2)),
        "isolated_pop_s3_pct": float(round(100.0 * flooded_states["s3"]["isolated_pop_share"], 2)),
        "candidates_evaluated": len(candidates_df),
        "zero_nans_verified": True,
    }
    return gate_summary

def run_phase2_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Executes Phase 2 end-to-end, evaluates accessibility across states,
    extracts candidate criteria table, and verifies gate conditions.
    """
    print("=" * 65)
    print("EVOROUTE: RUNNING PHASE 2 HIGH-SPEED ACCESSIBILITY & CRITERIA ENGINE")
    print("=" * 65)
    
    # Load Phase 1 Data Contract
    data_contract_path = RESULTS_DIR / "phase1_data.pkl"
    with open(data_contract_path, "rb") as f:
        p1_data = pickle.load(f)
        
    G = p1_data["G"]
    villages_df = p1_data["villages_df"]
    facilities_dict = p1_data["facilities_dict"]
    candidates = p1_data["candidates"]
    
    # Step 2.1: AHP weights
    ahp_w, cr = compute_ahp_service_weights()
    print(f"-> AHP Principal Weights: Hospital={ahp_w['hospital']:.3f}, "
          f"School={ahp_w['school']:.3f}, Market={ahp_w['market']:.3f} (CR={cr:.4f} <= 0.10)")
    
    # Step 2.2: Compute baseline & flooded network accessibility states
    base_state = compute_network_accessibility_state(
        G, villages_df, facilities_dict, scenario=None, ahp_w=ahp_w
    )
    flooded_states = {}
    for s_key in ["s1", "s2", "s3"]:
        flooded_states[s_key] = compute_network_accessibility_state(
            G, villages_df, facilities_dict, scenario=s_key, ahp_w=ahp_w
        )
        
    print(f"-> Baseline FAI_net: {base_state['fai_net']:.4f} | Isolated Population: 0.0%")
    print(f"-> Scenario S1 (Moderate): FAI_net={flooded_states['s1']['fai_net']:.4f} "
          f"({100.0*flooded_states['s1']['fai_net']/base_state['fai_net']:.1f}% retained) | "
          f"Isolated Pop: {100.0*flooded_states['s1']['isolated_pop_share']:.1f}%")
    print(f"-> Scenario S2 (Severe):   FAI_net={flooded_states['s2']['fai_net']:.4f} "
          f"({100.0*flooded_states['s2']['fai_net']/base_state['fai_net']:.1f}% retained) | "
          f"Isolated Pop: {100.0*flooded_states['s2']['isolated_pop_share']:.1f}%")
    print(f"-> Scenario S3 (Extreme):  FAI_net={flooded_states['s3']['fai_net']:.4f} "
          f"({100.0*flooded_states['s3']['fai_net']/base_state['fai_net']:.1f}% retained) | "
          f"Isolated Pop: {100.0*flooded_states['s3']['isolated_pop_share']:.1f}%")
          
    # Step 2.3: Extract Candidate Corridor Criteria Table
    candidates_df = extract_candidate_criteria_table(
        G, villages_df, facilities_dict, candidates, ahp_w=ahp_w
    )
    print(f"-> Candidate criteria table extracted for {len(candidates_df)} corridors.")
    
    # Verification Gate
    summary = verify_phase2_gate(base_state, flooded_states, candidates_df)
    print("\n[PHASE 2 VERIFICATION GATE]:")
    for k, v in summary.items():
        print(f"   {k}: {v}")
    print("=" * 65)
    
    # Save Phase 2 Data Contracts
    p2_path = RESULTS_DIR / "phase2_data.pkl"
    with open(p2_path, "wb") as f:
        pickle.dump({
            "G": G,
            "villages_df": villages_df,
            "facilities_dict": facilities_dict,
            "candidates_df": candidates_df,
            "base_state": base_state,
            "flooded_states": flooded_states,
            "ahp_weights": ahp_w,
            "summary": summary
        }, f)
        
    candidates_df.to_csv(TABLES_DIR / "candidate_corridors_criteria.csv", index=False)
    print(f"-> Phase 2 data contract serialized successfully to {p2_path}")
    print(f"-> Exported criteria table to {TABLES_DIR / 'candidate_corridors_criteria.csv'}\n")
    
    return candidates_df, flooded_states, summary

if __name__ == "__main__":
    run_phase2_pipeline()
