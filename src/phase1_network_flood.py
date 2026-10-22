"""
Phase 1: Geospatial Graph Engineering & Hydro-Simulation
Builds an undirected road network graph, integrates HAND elevation proxy,
applies the Mitropoulos (2025) speed-closure model across 3 flood scenarios,
snaps villages and facilities with demographic vulnerability (SVI),
and prunes candidate corridors to satisfy the Phase 1 Verification Gate.
"""
import math
import numpy as np
import pandas as pd
import networkx as nx
from typing import Tuple, Dict, List, Any
import pickle

from src.config import (
    ROAD_CLASSES,
    FLOOD_SCENARIOS,
    FLOOD_CLOSURE_DEPTH_MM,
    SPEED_CAP_RATIO,
    MAX_CANDIDATE_CORRIDORS,
    MIN_CANDIDATE_CORRIDORS,
    RANDOM_SEED,
    RESULTS_DIR,
    mitropoulos_speed_degradation,
)

def generate_synthetic_flood_prone_network(seed: int = RANDOM_SEED) -> Tuple[nx.Graph, pd.DataFrame, Dict[str, List[int]]]:
    """
    Generates a realistic synthetic rural road network spanning a 25km x 20km region.
    Topography includes a meandering river basin at low elevation (z ~ 5m to 8m)
    and surrounding rolling uplands (z ~ 12m to 25m).
    
    Returns:
        G (nx.Graph): Connected undirected graph with physical and topological attributes.
        villages_df (pd.DataFrame): Origin communities with population and SVI.
        facilities_dict (dict): Essential destinations (hospitals, schools, markets).
    """
    rng = np.random.RandomState(seed)
    G = nx.Graph()
    
    # 1. Define Hubs and Intersection Nodes
    # 6x5 semi-regular grid with random jitter for realistic road layout
    nx_grid, ny_grid = 7, 5
    x_coords = np.linspace(2.0, 24.0, nx_grid)
    y_coords = np.linspace(2.0, 18.0, ny_grid)
    
    node_id_counter = 0
    grid_nodes = {}
    
    # River centerline modeled as y_river(x) = 10.0 + 3.0 * sin(0.3 * x)
    def river_y(x):
        return 10.0 + 3.0 * math.sin(0.28 * x)
    
    def river_elevation(x):
        # River descends gradually from west (8m) to east (5m)
        return 8.0 - 0.12 * x
    
    for iy in range(ny_grid):
        for ix in range(nx_grid):
            base_x = x_coords[ix] + rng.uniform(-0.6, 0.6)
            base_y = y_coords[iy] + rng.uniform(-0.6, 0.6)
            
            # Ground elevation: river bed elevation + quadratic rise away from river channel
            dist_to_river = abs(base_y - river_y(base_x))
            riv_z = river_elevation(base_x)
            ground_z = riv_z + 0.35 * (dist_to_river ** 1.8) + rng.uniform(0.1, 0.5)
            
            node_id = node_id_counter
            node_id_counter += 1
            grid_nodes[(ix, iy)] = node_id
            
            G.add_node(
                node_id,
                x=float(round(base_x, 3)),
                y=float(round(base_y, 3)),
                elevation=float(round(ground_z, 2)),
                node_type="intersection"
            )
            
    # 2. Add Road Edges
    corridor_counter = 0
    edges_added = set()
    
    def add_road_edge(u: int, v: int, h_class: str):
        nonlocal corridor_counter
        edge_key = tuple(sorted((u, v)))
        if edge_key in edges_added or u == v:
            return
        edges_added.add(edge_key)
        
        ux, uy = G.nodes[u]['x'], G.nodes[u]['y']
        vx, vy = G.nodes[v]['x'], G.nodes[v]['y']
        uz, vz = G.nodes[u]['elevation'], G.nodes[v]['elevation']
        
        # Euclidean length in km
        dist_km = math.sqrt((ux - vx) ** 2 + (uy - vy) ** 2)
        mid_x = 0.5 * (ux + vx)
        mid_y = 0.5 * (uy + vy)
        mid_z = 0.5 * (uz + vz)
        
        # Hydrological HAND proxy calculation
        nearest_riv_z = river_elevation(mid_x)
        # Relative height above river water line
        dist_riv = abs(mid_y - river_y(mid_x))
        # Roads directly crossing or abutting the river have low relative elevation
        h_rel = max(0.05, mid_z - nearest_riv_z)
        # If road crosses river line, lower the bridge approach/culvert relative elevation
        crosses_river = (uy - river_y(ux)) * (vy - river_y(vx)) < 0
        if crosses_river:
            h_rel = min(h_rel, rng.uniform(0.6, 1.8))  # River crossing bridge/causeway
            
        c_info = ROAD_CLASSES[h_class]
        v_lim = c_info["speed_limit"]
        t_base = (dist_km / v_lim) * 60.0  # minutes
        cost_usd = dist_km * c_info["cost_per_km"]
        h_weight = c_info["hierarchy_weight"]
        
        G.add_edge(
            u, v,
            corridor_id=corridor_counter,
            u=u,
            v=v,
            highway=h_class,
            length_km=float(round(dist_km, 3)),
            v_limit=float(v_lim),
            t_base=float(round(t_base, 3)),
            elevation_mid=float(round(mid_z, 2)),
            river_elevation=float(round(nearest_riv_z, 2)),
            h_rel=float(round(h_rel, 3)),
            cost_usd=float(round(cost_usd, 2)),
            h_class_weight=float(h_weight)
        )
        corridor_counter += 1

    # Connect grid: horizontal and vertical links
    for iy in range(ny_grid):
        for ix in range(nx_grid):
            u = grid_nodes[(ix, iy)]
            # Horizontal (East-West)
            if ix < nx_grid - 1:
                v = grid_nodes[(ix + 1, iy)]
                # Main east-west spine along iy=2 is primary/trunk, others secondary/tertiary
                if iy == 2:
                    h_class = "primary"
                elif iy in [1, 3]:
                    h_class = "secondary"
                else:
                    h_class = "tertiary"
                add_road_edge(u, v, h_class)
            # Vertical (North-South)
            if iy < ny_grid - 1:
                v = grid_nodes[(ix, iy + 1)]
                # North-South crossings
                if ix in [1, 3, 5]:
                    h_class = "secondary"
                else:
                    h_class = "residential"
                add_road_edge(u, v, h_class)
                
    # Add diagonal lifeline connectors for network redundancy
    for (ix, iy) in [(1, 1), (3, 1), (4, 2), (2, 2), (0, 3), (4, 0)]:
        if ix + 1 < nx_grid and iy + 1 < ny_grid:
            u = grid_nodes[(ix, iy)]
            v = grid_nodes[(ix + 1, iy + 1)]
            add_road_edge(u, v, "tertiary")

    # 3. Designate Essential Facilities (Destinations)
    # Facilities located in urban/semi-urban clusters across the region
    facilities_dict: Dict[str, List[int]] = {
        "hospital": [
            grid_nodes[(1, 3)],  # Regional District Hospital West (Upland)
            grid_nodes[(5, 2)],  # Sub-divisional Base Hospital East
        ],
        "school": [
            grid_nodes[(2, 3)],  # Central Senior Secondary High School
            grid_nodes[(4, 1)],  # Community College / Vocational School South
            grid_nodes[(0, 1)],  # Primary Education Center West
        ],
        "market": [
            grid_nodes[(3, 2)],  # Central Agricultural Wholesale Market
            grid_nodes[(5, 3)],  # Eastern Commercial Trading Hub
            grid_nodes[(1, 1)],  # Southern Farmers Market
        ]
    }
    
    # Label facility nodes
    for fac_type, f_nodes in facilities_dict.items():
        for fn in f_nodes:
            G.nodes[fn]["node_type"] = fac_type

    # 4. Generate Village Settlements (Origins) with SVI
    # 25 rural hamlets/villages snapped directly to network nodes
    villages_data = []
    # Candidate village node candidates are intersection nodes that are not major hospital hubs
    candidate_village_nodes = [
        nid for nid in G.nodes
        if G.nodes[nid]["node_type"] == "intersection"
    ]
    # Select 25 distinct nodes as village community centers
    selected_village_nodes = rng.choice(candidate_village_nodes, size=25, replace=False)
    
    for idx, vn in enumerate(selected_village_nodes):
        node_attr = G.nodes[vn]
        # Base population between 400 and 2,800 residents
        base_pop = int(rng.choice([500, 800, 1200, 1500, 2200, 2800]))
        # Social Vulnerability Index from Sangeeta et al. (2025):
        # Villages closer to river floodplain or isolated peripheries have higher demographic vulnerability
        dist_to_riv = abs(node_attr['y'] - river_y(node_attr['x']))
        svi_score = float(round(np.clip(1.35 - 0.05 * dist_to_riv + rng.uniform(-0.1, 0.15), 0.80, 1.40), 2))
        effective_pop = float(round(base_pop * svi_score, 1))
        
        villages_data.append({
            "village_id": idx + 1,
            "name": f"Hamlet_{idx+1}_{['North', 'South', 'Riverine', 'Hillside', 'Valley'][idx % 5]}",
            "node_id": int(vn),
            "population": int(base_pop),
            "svi": float(svi_score),
            "effective_pop": float(effective_pop),
            "x": float(node_attr['x']),
            "y": float(node_attr['y']),
        })
        G.nodes[vn]["node_type"] = "village"
        
    villages_df = pd.DataFrame(villages_data)
    
    return G, villages_df, facilities_dict

def apply_hydro_simulation(G: nx.Graph) -> nx.Graph:
    """
    Applies the HAND hydrostatic proxy across the 3 flood scenarios:
    H1 = 1.0m (Moderate), H2 = 2.0m (Severe), H3 = 3.0m (Extreme).
    Computes flood depth (mm), Mitropoulos speed degradation, and travel time.
    """
    for u, v, data in G.edges(data=True):
        h_rel = data["h_rel"]  # meters
        v_lim = data["v_limit"]
        dist_km = data["length_km"]
        
        for s_key, s_info in FLOOD_SCENARIOS.items():
            stage_m = s_info["stage_m"]
            
            # Water depth in millimeters
            w_mm = 1000.0 * max(0.0, stage_m - h_rel)
            data[f"w_{s_key}"] = float(round(w_mm, 2))
            
            # Closure and speed degradation
            if w_mm >= FLOOD_CLOSURE_DEPTH_MM:
                data[f"v_{s_key}"] = 0.0
                data[f"t_{s_key}"] = float("inf")
                data[f"closed_{s_key}"] = True
            elif w_mm > 0.0:
                deg_speed = mitropoulos_speed_degradation(w_mm)
                # Capped at SPEED_CAP_RATIO * v_limit
                eff_speed = max(5.0, min(deg_speed, SPEED_CAP_RATIO * v_lim))
                t_flood = (dist_km / eff_speed) * 60.0
                data[f"v_{s_key}"] = float(round(eff_speed, 2))
                data[f"t_{s_key}"] = float(round(t_flood, 3))
                data[f"closed_{s_key}"] = False
            else:
                data[f"v_{s_key}"] = float(v_lim)
                data[f"t_{s_key}"] = float(data["t_base"])
                data[f"closed_{s_key}"] = False
                
    return G

def identify_and_prune_candidates(G: nx.Graph) -> Tuple[nx.Graph, List[Dict[str, Any]]]:
    """
    Identifies all road corridors closed in at least one scenario.
    If the number of closed corridors exceeds MAX_CANDIDATE_CORRIDORS (50),
    ranks by (Edge Betweenness Centrality * Mean Flood Depth) and keeps top 50.
    Ensures 10 <= |E_cand| <= 50 to satisfy the Phase 1 Verification Gate.
    """
    # Compute baseline edge betweenness centrality on the intact graph
    ebc = nx.edge_betweenness_centrality(G, weight="t_base", normalized=True)
    for (u, v), val in ebc.items():
        G[u][v]["betweenness"] = float(round(val, 6))
        
    closed_edges = []
    for u, v, data in G.edges(data=True):
        is_closed_any = (
            data["closed_s1"] or data["closed_s2"] or data["closed_s3"]
        )
        if is_closed_any:
            mean_depth = (data["w_s1"] + data["w_s2"] + data["w_s3"]) / 3.0
            score = G[u][v]["betweenness"] * (mean_depth + 1.0)
            closed_edges.append({
                "corridor_id": data["corridor_id"],
                "u": u,
                "v": v,
                "highway": data["highway"],
                "length_km": data["length_km"],
                "cost_usd": data["cost_usd"],
                "h_class_weight": data["h_class_weight"],
                "mean_depth_mm": mean_depth,
                "betweenness": G[u][v]["betweenness"],
                "score": score,
                "closed_s1": data["closed_s1"],
                "closed_s2": data["closed_s2"],
                "closed_s3": data["closed_s3"],
            })
            
    # Sort candidates by score descending
    closed_edges.sort(key=lambda x: x["score"], reverse=True)
    
    # Prune if > MAX_CANDIDATE_CORRIDORS
    if len(closed_edges) > MAX_CANDIDATE_CORRIDORS:
        pruned_candidates = closed_edges[:MAX_CANDIDATE_CORRIDORS]
    else:
        pruned_candidates = closed_edges
        
    # Mark in G whether an edge is in the candidate portfolio
    candidate_cids = {c["corridor_id"] for c in pruned_candidates}
    for u, v, data in G.edges(data=True):
        data["is_candidate"] = data["corridor_id"] in candidate_cids
        
    return G, pruned_candidates

def verify_phase1_gate(
    G: nx.Graph,
    villages_df: pd.DataFrame,
    facilities_dict: Dict[str, List[int]],
    candidates: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Verifies all Phase 1 Verification Gate conditions:
    1. G is undirected and fully connected under baseline (t_base < inf for all villages to all facilities).
    2. Candidate corridors 10 <= |E_cand| <= 50.
    3. Flood closures increase monotonically from s1 to s3.
    4. Zero NaNs across all edge and node attributes.
    """
    # Check 1: Graph is connected undirected
    assert not G.is_directed(), "Assertion Failed: G must be an undirected graph."
    assert nx.is_connected(G), "Assertion Failed: G must be a connected graph under baseline conditions."
    
    # Check 1b: Baseline travel times finite for all villages
    for v_node in villages_df["node_id"]:
        for f_category, f_nodes in facilities_dict.items():
            reachable = any(nx.has_path(G, v_node, fn) for fn in f_nodes)
            assert reachable, f"Assertion Failed: Village node {v_node} has no path to {f_category} facilities."
            
    # Check 2: Candidate corridor size bound
    num_cand = len(candidates)
    assert MIN_CANDIDATE_CORRIDORS <= num_cand <= MAX_CANDIDATE_CORRIDORS, (
        f"Assertion Failed: Candidate corridor count {num_cand} out of bounds "
        f"[{MIN_CANDIDATE_CORRIDORS}, {MAX_CANDIDATE_CORRIDORS}]."
    )
    
    # Check 3: Monotonicity of flood closures
    n_closed_s1 = sum(1 for _, _, d in G.edges(data=True) if d["closed_s1"])
    n_closed_s2 = sum(1 for _, _, d in G.edges(data=True) if d["closed_s2"])
    n_closed_s3 = sum(1 for _, _, d in G.edges(data=True) if d["closed_s3"])
    assert n_closed_s1 <= n_closed_s2 <= n_closed_s3, (
        f"Assertion Failed: Closures must be non-decreasing across flood stages. "
        f"Got S1={n_closed_s1}, S2={n_closed_s2}, S3={n_closed_s3}."
    )
    
    # Check 4: Zero NaNs
    for _, _, d in G.edges(data=True):
        for k, v in d.items():
            if isinstance(v, float) and math.isnan(v):
                raise AssertionError(f"NaN found in edge attribute {k}")
    for _, d in G.nodes(data=True):
        for k, v in d.items():
            if isinstance(v, float) and math.isnan(v):
                raise AssertionError(f"NaN found in node attribute {k}")
                
    gate_summary = {
        "status": "PASSED",
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_villages": len(villages_df),
        "total_population": int(villages_df["population"].sum()),
        "total_effective_pop": float(round(villages_df["effective_pop"].sum(), 1)),
        "num_candidates": num_cand,
        "closures_s1": n_closed_s1,
        "closures_s2": n_closed_s2,
        "closures_s3": n_closed_s3,
        "total_cand_cost_usd": float(round(sum(c["cost_usd"] for c in candidates), 2)),
    }
    return gate_summary

def run_phase1_pipeline() -> Tuple[nx.Graph, pd.DataFrame, Dict[str, List[int]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes Phase 1 end-to-end, saves data contracts to disk, and prints verification summary.
    """
    print("=" * 65)
    print("EVOROUTE: RUNNING PHASE 1 GEOSPATIAL GRAPH & HYDRO-SIMULATION")
    print("=" * 65)
    
    # Step 1.1 & 1.2: Build network, snap villages with SVI, snap facilities
    G, villages_df, facilities_dict = generate_synthetic_flood_prone_network()
    print(f"-> Generated network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"-> Snapped {len(villages_df)} villages with Social Vulnerability Index (SVI)")
    print(f"-> Snapped facilities: {len(facilities_dict['hospital'])} hospitals, "
          f"{len(facilities_dict['school'])} schools, {len(facilities_dict['market'])} markets")
    
    # Step 1.3: Hydro-simulation & speed-closure modeling
    G = apply_hydro_simulation(G)
    print("-> Hydro-simulation complete: applied HAND depth & Mitropoulos speed-degradation")
    
    # Step 1.4: Identify and prune candidate corridors
    G, candidates = identify_and_prune_candidates(G)
    print(f"-> Candidate corridors identified and pruned: {len(candidates)} corridors")
    
    # Verification Gate
    summary = verify_phase1_gate(G, villages_df, facilities_dict, candidates)
    print("\n[PHASE 1 VERIFICATION GATE]:")
    for k, v in summary.items():
        print(f"   {k}: {v}")
    print("=" * 65)
    
    # Save Phase 1 Data Contracts for downstream phases
    data_contract_path = RESULTS_DIR / "phase1_data.pkl"
    with open(data_contract_path, "wb") as f:
        pickle.dump({
            "G": G,
            "villages_df": villages_df,
            "facilities_dict": facilities_dict,
            "candidates": candidates,
            "summary": summary
        }, f)
    print(f"-> Phase 1 data contract serialized successfully to {data_contract_path}\n")
    
    # Also save villages_df as CSV for human inspection
    villages_df.to_csv(RESULTS_DIR / "tables" / "villages_svi.csv", index=False)
    
    return G, villages_df, facilities_dict, candidates, summary

if __name__ == "__main__":
    run_phase1_pipeline()
