"""
EvoRoute: Full End-to-End Execution Pipeline
Runs all 4 Phases sequentially:
Phase 1: Geospatial Graph Engineering & Hydro-Simulation
Phase 2: High-Speed Accessibility Engine & Criteria Extraction
Phase 3: Core Soft Computing Engine (Mamdani FIS + Hybrid GA)
Phase 4: Baselines, Ablation/Sensitivity Suite, Visualization & Final Deliverables
"""
import sys
import time

from src.phase1_network_flood import run_phase1_pipeline
from src.phase2_accessibility_criteria import run_phase2_pipeline
from src.phase3_soft_computing import run_phase3_pipeline
from src.phase4_benchmarks_viz import run_phase4_pipeline

def main():
    start_total = time.time()
    print("\n" + "#" * 70)
    print("EVOROUTE: FLOOD-RESILIENT ROAD PRIORITY RANKING SYSTEM")
    print("SOFT COMPUTING PIPELINE (MAMDANI FIS + HYBRID GENETIC ALGORITHM)")
    print("#" * 70 + "\n")
    
    # Phase 1
    t0 = time.time()
    run_phase1_pipeline()
    print(f">> Phase 1 execution time: {time.time() - t0:.2f}s\n")
    
    # Phase 2
    t0 = time.time()
    run_phase2_pipeline()
    print(f">> Phase 2 execution time: {time.time() - t0:.2f}s\n")
    
    # Phase 3
    t0 = time.time()
    run_phase3_pipeline()
    print(f">> Phase 3 execution time: {time.time() - t0:.2f}s\n")
    
    # Phase 4
    t0 = time.time()
    run_phase4_pipeline()
    print(f">> Phase 4 execution time: {time.time() - t0:.2f}s\n")
    
    elapsed = time.time() - start_total
    print("#" * 70)
    print(f"SUCCESS: FULL EVOROUTE PIPELINE COMPLETED IN {elapsed:.2f} SECONDS")
    print("#" * 70 + "\n")

if __name__ == "__main__":
    main()
