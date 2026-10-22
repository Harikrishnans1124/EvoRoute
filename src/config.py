"""
EvoRoute Configuration Module
Defines physical parameters, cost models, flood scenarios, and paths.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MAPS_DIR = RESULTS_DIR / "maps"
TABLES_DIR = RESULTS_DIR / "tables"

# Ensure output directories exist
for d in [RESULTS_DIR, FIGURES_DIR, MAPS_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Road Classification Parameters
# Speed limit (km/h), Elevation/Armoring Cost per km (USD), Hierarchy Weight h_c
ROAD_CLASSES = {
    "primary": {
        "speed_limit": 60.0,
        "cost_per_km": 150_000.0,
        "hierarchy_weight": 1.0,
    },
    "trunk": {
        "speed_limit": 60.0,
        "cost_per_km": 150_000.0,
        "hierarchy_weight": 1.0,
    },
    "secondary": {
        "speed_limit": 50.0,
        "cost_per_km": 100_000.0,
        "hierarchy_weight": 0.8,
    },
    "tertiary": {
        "speed_limit": 40.0,
        "cost_per_km": 70_000.0,
        "hierarchy_weight": 0.6,
    },
    "residential": {
        "speed_limit": 30.0,
        "cost_per_km": 40_000.0,
        "hierarchy_weight": 0.4,
    },
    "unclassified": {
        "speed_limit": 30.0,
        "cost_per_km": 40_000.0,
        "hierarchy_weight": 0.4,
    },
    "track": {
        "speed_limit": 30.0,
        "cost_per_km": 40_000.0,
        "hierarchy_weight": 0.4,
    },
}

# Flood Hazard Parameters (HAND Hydrostatic Proxy)
# Water-stage rise levels in meters for progressive climate failure scenarios
FLOOD_SCENARIOS = {
    "s1": {"name": "Moderate", "stage_m": 1.0},
    "s2": {"name": "Severe", "stage_m": 2.0},
    "s3": {"name": "Extreme", "stage_m": 3.0},
}

# Closure & Speed Degradation Parameters (Mitropoulos 2025 / Pregnolato 2017)
FLOOD_CLOSURE_DEPTH_MM = 200.0  # Roads impassable at >= 200 mm water depth
SPEED_CAP_RATIO = 0.85          # Maximum speed fraction under partial water (<= 85% free flow)

def mitropoulos_speed_degradation(w_mm: float) -> float:
    """
    Computes maximum safe speed (km/h) as a function of water depth (mm).
    Formula: v(w) = 0.0009 * w^2 - 0.5529 * w + 86.9448
    """
    if w_mm <= 0.0:
        return 86.9448
    return 0.0009 * (w_mm ** 2) - 0.5529 * w_mm + 86.9448

# Facility Hierarchy & AHP Weights
# Consistent with Saaty Pairwise Matrix: [Hospital, School, Market] with CR <= 0.10
AHP_WEIGHTS = {
    "hospital": 0.648,
    "school": 0.230,
    "market": 0.122,
}

# Travel Time Thresholds for Overall Accessibility Index (OAI in minutes)
OAI_THRESHOLDS = {
    "hospital": 30.0,
    "school": 20.0,
    "market": 20.0,
}

# Candidate Corridor Pruning Limit
MAX_CANDIDATE_CORRIDORS = 50
MIN_CANDIDATE_CORRIDORS = 10

# Budget Tiers (Fractions of Total Flooded Corridor Rehabilitation Cost)
BUDGET_TIERS = [0.05, 0.10, 0.15, 0.20]

# GA Hyperparameters
GA_CONFIG = {
    "pop_size": 40,
    "max_generations": 50,
    "early_stopping_patience": 12,
    "tournament_k": 3,
    "crossover_prob": 0.80,
    "mutation_rate_factor": 2.0,  # pm = 2.0 / N
    "elitism_count": 2,
    "fuzzy_seed_ratio": 0.20,     # 20% seeded by fuzzy priority
    "alpha": 0.60,                # Tradeoff between continuous FAI and connected population
    "lambda_reg": 0.15,           # Regularizer weight for fuzzy expert score
    "num_seeds": 5,               # For statistical confidence reporting
}
