# Fail-Proof 4-Phase Implementation Plan: Flood-Resilient Road Priority Ranking System (Soft Computing Architecture)

## 1. Executive Summary & Architectural Philosophy

### 1.1 Problem Statement
During extreme rainfall and riverine flooding, critical rural road segments become submerged or impassable, isolating villages from emergency healthcare, schools, and food markets. Because authorities operate under strict fiscal constraints, they cannot elevate or armor every flood-prone road simultaneously. 

### 1.2 Soft Computing Value Proposition
Classical road prioritization relies on either purely topological metrics (e.g., betweenness centrality) or static Multi-Criteria Decision-Making (e.g., crisp AHP-TOPSIS). Both fail in flood-resilience planning for two reasons:
1. **Epistemic Uncertainty & Non-Linear Trade-offs:** Evaluating whether a moderately flooded road that reconnects an isolated hamlet is more urgent than a severely flooded highway requires human-like approximate reasoning. We solve this using a **Mamdani Fuzzy Inference System (FIS)** with 27 linguistic rules.
2. **Combinatorial Network Synergy:** Selecting a portfolio of roads under a strict budget is an NP-hard knapsack-network problem where protecting one road changes the value of protecting adjacent roads. We solve this using a **Fuzzy-Guided Genetic Algorithm (GA)** that evaluates full network accessibility dynamically.

```
+-----------------------------------------------------------------------------+
| PHASE 1: GEOSPATIAL GRAPH ENGINEERING & HYDRO-SIMULATION                    |
|  - Undirected Simple Graph Consolidation (solves asymmetric lane bugs)      |
|  - Contiguous Corridor Grouping & Candidate Pruning (N <= 50 corridors)     |
|  - HAND Proxy Flood Depth & Mitropoulos (2025) Speed/Closure Model          |
+--------------------------------------+--------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| PHASE 2: HIGH-SPEED ACCESSIBILITY ENGINE & FEATURE EXTRACTION               |
|  - AHP Facility Weighting (Hospital, School, Market; CR <= 0.10)            |
|  - Multi-Source Dijkstra from Facilities (10x speedup over village-origin)  |
|  - Hansen FAI, Threshold OAI, Isolation Flags & Link Criteria Extraction    |
+--------------------------------------+--------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| PHASE 3: CORE SOFT COMPUTING ENGINE (MAMDANI FIS + HYBRID GA)               |
|  - Layer 1 (Mamdani FIS): 3 Inputs -> 27 Linguistic Rules -> Priority [0-100]|
|  - Layer 2 (Hybrid GA): Binary Chromosome, Fuzzy Population Seeding (20%),  |
|    Fuzzy Cost-Benefit Repair Operator, Hash-Cached Network Fitness          |
+--------------------------------------+--------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| PHASE 4: BASELINES, ABLATION/SENSITIVITY SUITE & VISUALIZATION              |
|  - 6-Way Benchmark (Random, Cost, Betweenness, AHP-TOPSIS, Fuzzy, Fuzzy+GA) |
|  - Ablation Study (GA with vs. without Fuzzy Seeding) & Sensitivity (tau)   |
|  - Interactive Folium Map, Convergence Plots & ASSUMPTIONS.md               |
+-----------------------------------------------------------------------------+
```

---

## 2. Literature Alignment & Methodological Traceability

Every equation, parameter, and experimental design choice in this plan is grounded in the project reference set:

| Reference Source | Core Methodological Contribution | Exact Implementation in This System |
| :--- | :--- | :--- |
| **Mitropoulos et al. (2025) / Pregnolato et al. (2017)** | Empirical floodwater depth vs. vehicle speed degradation curve; $200\text{ mm}$ road impassability threshold; Hansen Facility Accessibility Index ($FAI$) and Overall Accessibility Index ($OAI$). | Phase 1 implements $v(w) = 0.0009w^2 - 0.5529w + 86.9448$ and closes roads at $w \ge 200\text{ mm}$. Phase 2 computes Hansen $FAI_i$ and $OAI$ across baseline and 3 flood severities. |
| **Sangeeta et al. (EGU 2025)** | Resilience framework evaluating healthcare/emergency facility accessibility loss across 3 progressive climate flood scenarios (Present, Mid-Range, High-End); Betweenness Centrality for road network criticality; Social Vulnerability Index (SVI). | Phase 1 adopts 3 progressive flood severity scenarios ($H_1, H_2, H_3$) and edge betweenness pruning. Phase 1 & 2 weight populations by Social Vulnerability ($P_i^{\text{eff}} = P_i \times \text{SVI}_i$) prioritizing high-risk demographic clusters. |
| **Akhlaghi, Campbell & Demir (arXiv:2302.07983)** | Mathematical formulation of the Road Network Flood Mitigation Problem (RNFMP); network design under graduated budget tiers; healthcare facility access under extreme flood events; network reduction bounds. | Phase 3 and Phase 4 optimize road portfolios under graduated budget tiers ($5\%, 10\%, 15\%, 20\%$ of total mitigation cost) and origin-to-facility shortest path objectives. Facility capacity bounds ($H_j$) are retained for diagnostic load tracking. |
| **Rathnayaka, Pasindu & Kulathunga (2025)** | Multi-criteria risk assessment framework (Exposure, Likelihood, Severity) for prioritizing road infrastructure mitigation and completion under severe fiscal constraints in developing road networks. | Phase 2 and Phase 3 translate the multi-criteria risk philosophy into the 3-input Mamdani FIS (Exposure, Accessibility Benefit, Isolation & Criticality) to rank interventions under strict resource constraints. |
| **AHP-TOPSIS & Soft Computing Literature** | Analytic Hierarchy Process (Saaty pairwise matrices) for criteria weighting; Mamdani Fuzzy Inference for approximate reasoning; Binary Genetic Algorithms (GA) for combinatorial knapsack network synergy; TOPSIS distance-to-ideal ranking. | Phase 2 uses AHP for facility weights ($W_{\text{hosp}}, W_{\text{school}}, W_{\text{market}}$). Phase 3 uses Mamdani FIS for heuristic expert guidance and a binary GA for global combinatorial portfolio optimization. Phase 4 benchmarks against crisp AHP-TOPSIS. |

---

## 3. Built-In Engineering Safeguards (Resolving All 5 Blind Spots)

To prevent runtime crashes, infinite loops, or methodological critiques during evaluation, the following five architectural safeguards are hard-coded into the 4-phase design:

1. **Safeguard 1: Undirected Corridor Consolidation (Fixes Fragmented & Asymmetric OSM Edges)**
   * *Problem:* Raw OSM graphs are directed (`MultiDiGraph`) and split single roads into tiny segments. Protecting one direction or a $50\text{ m}$ middle slice leaves the road unusable.
   * *Solution:* Convert the graph to a simplified undirected `nx.Graph` where each edge represents a complete, bidirectional topological link between intersections, keeping the minimum travel-time attribute among parallel edges. Group contiguous flooded edges sharing the same road name/highway class into unified **Candidate Corridors**.
2. **Safeguard 2: Multi-Source Facility Dijkstra & Hash Caching (Fixes GA Runtime Bottleneck)**
   * *Problem:* Running village-to-facility Dijkstra $12,000+$ times during GA evolution takes hours.
   * *Solution:* Run `nx.multi_source_dijkstra_path_length(G, sources=facility_nodes, weight='travel_time')` once per facility category ($3$ calls per scenario instead of $\vert V_{\text{villages}} \vert \times \vert F \vert$ calls). Furthermore, cache evaluated chromosome bit-tuples in a Python dictionary `fitness_cache[tuple(x)]` so identical chromosomes are never evaluated twice, and prune candidate corridors to $N \le 50$.
3. **Safeguard 3: Complementary Division of Labor Between FIS and GA (Fixes Redundancy)**
   * *Problem:* If the GA fitness function simply duplicates the Fuzzy inputs, the Fuzzy system becomes redundant.
   * *Solution:* 
     * **Mamdani FIS** acts as a **Local Multi-Criteria Expert** evaluating individual corridor urgency based on *Physical Flood Exposure ($E$)*, *Marginal Accessibility Gain ($\Delta A$)*, and *Critical Isolation & Road Hierarchy Index ($B_I$)*.
     * **Genetic Algorithm** acts as a **Global Combinatorial Portfolio Optimizer** capturing multi-corridor topological synergies under hard budget limits. The FIS directly drives the GA in three ways: (a) **Fuzzy Heuristic Seeding** of $20\%$ of the initial population, (b) **Fuzzy Cost-Efficiency Repair Operator** ($\rho_j = \text{Priority}_j / \text{Cost}_j$) to fix over-budget chromosomes, and (c) a **Preference Regularizer** ($\lambda = 0.15$) in the fitness function to break ties in favor of structurally critical corridors.
4. **Safeguard 4: Zero-Division & Boundary Guards in Fuzzy Inference (Fixes `scikit-fuzzy` NaNs)**
   * *Problem:* If an input value falls slightly outside $[0, 1]$ due to floating-point precision, or if a rule combination is missing, `scikit-fuzzy` fails with a zero-area defuzzification error.
   * *Solution:* Clip all normalized inputs strictly to $[0.0, 1.0]$ via `np.clip()`, define overlapping membership functions covering the exact universe $[0.0, 1.0]$, and programmatically generate all $3^3 = 27$ rules so the rule base is $100\%$ complete.
5. **Safeguard 5: Explicit Hydrostatic HAND Proxy Framing (Fixes Hydrological Critique)**
   * *Problem:* Subtracting nearest river elevation from road elevation is not a 2D hydrodynamic simulation.
   * *Solution:* Explicitly frame the flood model as a **Height Above Nearest Drainage (HAND) hydrostatic proxy** under synthetic water-stage rises ($H \in \{1.0\text{ m}, 2.0\text{ m}, 3.0\text{ m}\}$), never mislabeling them as meteorological return periods.

---

## 4. The 4-Phase Step-by-Step Implementation Blueprint

### Phase 1: Geospatial Graph Engineering & Hydro-Simulation
**Files:** `src/config.py`, `src/phase1_network_flood.py`

#### Step 1.1: Graph Extraction or Synthetic Fallback
1. Attempt OSM extraction for a compact flood-prone study area (or use the deterministic Synthetic Flood-Prone Rural Network generator built into `phase1_network_flood.py` if offline or if OSM amenity tags are sparse, following established network benchmark protocols).
2. Convert the raw network to a connected undirected graph $G = (V, E)$:
   * Retain only the largest connected component.
   * Assign free-flow speed limits $v_{\text{limit}}(e)$ by highway class:
     * `primary` / `trunk`: $60\text{ km/h}$, Cost: $\$150\text{k/km}$, Hierarchy weight $h_c = 1.0$
     * `secondary`: $50\text{ km/h}$, Cost: $\$100\text{k/km}$, Hierarchy weight $h_c = 0.8$
     * `tertiary`: $40\text{ km/h}$, Cost: $\$70\text{k/km}$, Hierarchy weight $h_c = 0.6$
     * `residential` / `unclassified` / `track`: $30\text{ km/h}$, Cost: $\$40\text{k/km}$, Hierarchy weight $h_c = 0.4$
   * Compute baseline travel time in minutes:
     $$t_{\text{base}}(e) = \frac{\text{length}_{\text{km}}(e)}{v_{\text{limit}}(e)} \times 60$$

#### Step 1.2: Origin-Destination Snapping, Vulnerability & Elevation Sampling
1. **Villages ($V_{\text{vill}}$) & Social Vulnerability:** Snap $15\text{--}40$ village centroids to nearest graph nodes $u \in V$. Attach base population $P_i$ and a **Social Vulnerability Index** ($\text{SVI}_i \in [0.8, 1.4]$, following Sangeeta et al. 2025) reflecting vulnerable demographics (elderly, children, low-income, healthcare-dependent). Compute effective vulnerable population:
   $$P_i^{\text{eff}} = P_i \times \text{SVI}_i$$
   (with uniform baseline $\text{SVI}_i = 1.0$ and $P_i = 1000$ documented in `ASSUMPTIONS.md`).
2. **Facilities ($F_k$):** Snap Hospital/Clinic ($k=1$), School ($k=2$), and Market ($k=3$) locations to nearest graph nodes.
3. **HAND Elevation Proxy:** For each edge midpoint, sample elevation $z(e)$ and nearest river channel elevation $z_{\text{river}}(e)$. Compute relative elevation:
   $$h_{\text{rel}}(e) = \max\left(0.0, z(e) - z_{\text{river}}(e)\right)$$

#### Step 1.3: Flood Scenarios & Mitropoulos (2025) Speed-Closure Model
Evaluate three synthetic water-stage rise scenarios $s \in \{1, 2, 3\}$ representing progressive failure states (Moderate, Severe, Extreme, analogous to Sangeeta et al.'s climate progression): $H_1 = 1.0\text{ m}$, $H_2 = 2.0\text{ m}$, and $H_3 = 3.0\text{ m}$:
1. Flood depth in millimeters on edge $e$ in scenario $s$:
   $$w_{e,s} = 1000 \times \max\left(0.0, H_s - h_{\text{rel}}(e)\right)$$
2. Mitropoulos (2025) / Pregnolato et al. (2017) speed-depth equation (for $0 < w_{e,s} < 200\text{ mm}$):
   $$v(w_{e,s}) = 0.0009 w_{e,s}^2 - 0.5529 w_{e,s} + 86.9448$$
3. Effective edge speed $v_{e,s}$ and closure rule:
   $$v_{e,s} = \begin{cases} 0 \text{ (Closed)}, & \text{if } w_{e,s} \ge 200\text{ mm} \\ \min\left(v(w_{e,s}), 0.85 \cdot v_{\text{limit}}(e)\right), & \text{if } 0 < w_{e,s} < 200\text{ mm} \\ v_{\text{limit}}(e), & \text{if } w_{e,s} = 0\text{ mm} \end{cases}$$
4. Scenario travel time $t_{e,s}$:
   $$t_{e,s} = \begin{cases} \infty, & \text{if } w_{e,s} \ge 200\text{ mm} \\ \frac{\text{length}_{\text{km}}(e)}{v_{e,s}} \times 60, & \text{otherwise} \end{cases}$$

#### Step 1.4: Candidate Corridor Pruning
Identify all edges closed ($w_{e,s} \ge 200\text{ mm}$) in at least one scenario. If the number of closed edges exceeds $50$, rank them by edge betweenness centrality $\times$ mean flood depth and keep the top $N = 50$ corridors as the candidate upgrade set $E_{\text{cand}}$.

> **Phase 1 Verification Gate:** Assert $G$ is undirected and connected under baseline conditions ($t_{\text{base}} < \infty$ for all villages), $10 \le \vert E_{\text{cand}} \vert \le 50$, and flood closures monotonically increase from Scenario 1 to Scenario 3.

---

### Phase 2: High-Speed Accessibility Engine & Feature Extraction
**Files:** `src/phase2_accessibility_criteria.py`

#### Step 2.1: AHP Service Category Weighting
Define the Saaty pairwise comparison matrix $\mathbf{A}$ for $[ \text{Hospital}, \text{School}, \text{Market} ]$:
$$\mathbf{A} = \begin{bmatrix} 1 & 3 & 5 \\ 1/3 & 1 & 2 \\ 1/5 & 1/2 & 1 \end{bmatrix}$$
1. Compute the principal eigenvector to obtain normalized weights:
   $$W_{\text{hosp}} = 0.648, \quad W_{\text{school}} = 0.230, \quad W_{\text{market}} = 0.122$$
2. Verify Consistency Ratio $CR = \frac{CI}{RI} \approx 0.0036 \le 0.10$ ($RI = 0.58$ for $n=3$).

#### Step 2.2: Multi-Source Dijkstra & Hansen Accessibility ($FAI$, $OAI$, Isolation)
For any graph state (baseline, flooded scenario $s$, or GA-restored state):
1. For each facility category $k \in \{\text{hosp}, \text{school}, \text{market}\}$ and each facility $r \in F_k$, compute shortest travel time $c_{ir}$ (in minutes) to every village $i \in V_{\text{vill}}$ using multi-source/single-source Dijkstra from facility nodes $r$.
2. **Facility Accessibility Index ($FAI_i$, Hansen $\beta = 1$):**
   $$\text{FAI}_i = \sum_{k} W_k \sum_{r \in F_k \text{ s.t. } c_{ir} < \infty} \frac{1}{\max\left(1.0, c_{ir}\right)}$$
3. **Vulnerability-Weighted Network FAI (integrating SVI from Sangeeta et al. 2025):**
   $$\text{FAI}_{\text{net}} = \frac{\sum_{i \in V_{\text{vill}}} P_i^{\text{eff}} \cdot \text{FAI}_i}{\sum_{i \in V_{\text{vill}}} P_i^{\text{eff}}}$$
4. **Village Isolation Indicator ($I_{i,k}$ and Hospital Isolation $I_i^{\text{hosp}}$):**
   * A village $i$ is isolated from category $k$ ($I_{i,k} = 1$) if $\min_{r \in F_k} c_{ir} = \infty$.
   * **Overall Vulnerable Isolation Share ($S_{\text{iso}}$):** Fraction of total effective vulnerable population isolated from healthcare facilities:
     $$S_{\text{iso}} = \frac{\sum_{i \in V_{\text{vill}}} P_i^{\text{eff}} \cdot I_i^{\text{hosp}}}{\sum_{i \in V_{\text{vill}}} P_i^{\text{eff}}}$$
5. **Overall Accessibility Index ($OAI$):** Share of total population reaching at least one hospital within $30\text{ min}$, school within $20\text{ min}$, and market within $20\text{ min}$.
6. **Facility Capacity Diagnostic (following Akhlaghi et al. 2023):** While unconstrained gravity decay models continuous potential accessibility, track destination loads $\sum_{i \to r} P_i$ against nominal capacities $H_j$ in post-run diagnostic reporting to identify potential healthcare facility bottlenecks.

#### Step 2.3: Candidate Corridor Criteria Table Extraction
For each candidate corridor $j \in \{1, \dots, N\}$ in $E_{\text{cand}}$, compute its marginal protection value by temporarily keeping corridor $j$ open ($w_{j,s} = 0$) while all other flooded edges remain degraded/closed:
1. **Criterion 1 — Flood Exposure ($C_{\text{exp}, j}$):** Mean flood depth across the 3 scenarios:
   $$C_{\text{exp}, j} = \frac{1}{3} \sum_{s=1}^3 w_{j,s}$$
2. **Criterion 2 — Marginal Accessibility Benefit ($C_{\text{acc}, j}$):** Mean vulnerability-weighted $\text{FAI}_{\text{net}}$ gained across scenarios when corridor $j$ alone is protected:
   $$C_{\text{acc}, j} = \frac{1}{3} \sum_{s=1}^3 \left( \text{FAI}_{\text{net}, s}(\text{protect } j) - \text{FAI}_{\text{net}, s}(\text{unprotected}) \right)$$
3. **Criterion 3 — Isolation & Criticality Index ($C_{\text{iso}, j}$):** Composite of effective vulnerable population reconnected from isolation ($\Delta P_{\text{iso}, j}^{\text{eff}}$) and topological/hierarchy criticality ($0.7 \cdot \widehat{\Delta P}_{\text{iso}, j}^{\text{eff}} + 0.3 \cdot \widehat{\text{Betweenness}}_j \cdot h_{c,j}$), ensuring that even in networks where multiple bridge failures require pairs of roads to reconnect a village, critical lifeline corridors still receive non-zero isolation criticality signal.
4. **Min-Max Normalization to $[0, 1]$ with Safe Zero-Guard:**
   $$\hat{C}_{m, j} = \text{clip}\left( \frac{C_{m, j} - \min_j(C_m)}{\max\left(10^{-9}, \max_j(C_m) - \min_j(C_m)\right)}, \, 0.0, \, 1.0 \right)$$

> **Phase 2 Verification Gate:** Confirm baseline $\text{FAI}_{\text{net}}$ loss is $0\%$, post-flood $\text{FAI}_{\text{net}}$ drops across scenarios, and all normalized criteria columns $\hat{E}_j, \widehat{\Delta A}_j, \hat{B}_{I,j}$ contain zero `NaN`s and lie strictly within $[0.0, 1.0]$.

---

### Phase 3: Core Soft Computing Engine (Mamdani FIS + Hybrid GA)
**Files:** `src/phase3_soft_computing.py`

#### Step 3.1: Soft Computing Layer 1 — Mamdani Fuzzy Inference System
Implemented via `skfuzzy.control` with universe of discourse $U_{\text{in}} = [0.0, 1.0]$ (step $0.01$) for inputs and $U_{\text{out}} = [0.0, 100.0]$ (step $0.5$) for output Priority $P_{\text{fuzzy}}$.

1. **Input Membership Functions (for $\hat{E}$, $\widehat{\Delta A}$, $\hat{B}_I$):**
   * `Low (L)` (Trapezoidal): $[0.0, 0.0, 0.20, 0.45]$
   * `Medium (M)` (Triangular): $[0.25, 0.50, 0.75]$
   * `High (H)` (Trapezoidal): $[0.55, 0.80, 1.0, 1.0]$
2. **Output Membership Functions (for $\text{Priority} \in [0, 100]$):**
   * `Very Low (VL)` (Trapezoidal): $[0, 0, 10, 25]$
   * `Low (L)` (Triangular): $[15, 30, 45]$
   * `Medium (M)` (Triangular): $[35, 50, 65]$
   * `High (H)` (Triangular): $[55, 70, 85]$
   * `Very High (VH)` (Trapezoidal): $[75, 90, 100, 100]$
3. **Complete 27-Rule Base ($3 \times 3 \times 3$):**
   Map linguistic levels $\{\text{L}, \text{M}, \text{H}\} \mapsto \{0, 1, 2\}$. For every triplet $(l_E, l_A, l_I) \in \{0, 1, 2\}^3$, compute the expert urgency index:
   $$U = l_E + l_A + 2 \cdot l_I \in \{0, 1, \dots, 8\}$$
   Map $U$ to consequent levels with safety-critical overrides:
   * If $l_I = 2$ and $(l_E \ge 1 \text{ or } l_A \ge 1) \implies \text{Very High (VH)}$
   * Else if $U \in \{0, 1\} \implies \text{Very Low (VL)}$
   * Else if $U \in \{2, 3\} \implies \text{Low (L)}$
   * Else if $U \in \{4, 5\} \implies \text{Medium (M)}$
   * Else if $U = 6 \implies \text{High (H)}$
   * Else ($U \ge 7$) $\implies \text{Very High (VH)}$
4. **Defuzzification:** Compute crisp priority score $P_{\text{fuzzy}, j} \in [0, 100]$ for each candidate road using the **Centroid** method:
   $$P_{\text{fuzzy}, j} = \frac{\int_{0}^{100} z \cdot \mu_{\text{agg}}(z) \, dz}{\int_{0}^{100} \mu_{\text{agg}}(z) \, dz}$$

#### Step 3.2: Soft Computing Layer 2 — Fuzzy-Seeded Constrained Genetic Algorithm
For each budget tier $B_k \in \{0.05, 0.10, 0.15, 0.20\} \times \sum_{j=1}^N \text{Cost}_j$:

1. **Chromosome Representation:** Binary vector $\mathbf{x} = [x_1, x_2, \dots, x_N] \in \{0, 1\}^N$, where $x_j = 1$ means candidate corridor $j$ is elevated/protected against all flood scenarios ($w_{j,s} = 0$).
2. **Domain-Specific Fuzzy Budget Repair Operator (`repair(x, B_k)`):**
   Executed immediately on every initial, crossed-over, or mutated chromosome before fitness evaluation:
   * Compute cost $C(\mathbf{x}) = \sum_{j=1}^N x_j \cdot \text{Cost}_j$.
   * While $C(\mathbf{x}) > B_k$: identify all active genes $\{j \mid x_j = 1\}$ and flip $x_{j^*} \leftarrow 0$ for the gene with the lowest **Fuzzy Cost-Effectiveness Ratio**:
     $$\rho_j = \frac{P_{\text{fuzzy}, j} + 1.0}{\text{Cost}_j}$$
   * *Optional Greedy Fill Step:* After dropping excess genes, iterate over unselected genes in descending order of $\rho_j$; if adding gene $j$ keeps $C(\mathbf{x}) \le B_k$, set $x_j \leftarrow 1$ so budget capacity is not wasted.
3. **Fuzzy-Guided Population Initialization ($N_{\text{pop}} = 40$):**
   * **Fuzzy Heuristic Seeding ($20\%$ of population = 8 individuals):** Select genes with probability proportional to $\rho_j$ and repair to budget $B_k$.
   * **Random Exploration ($80\%$ of population = 32 individuals):** Uniform random bit generation followed by `repair(x, B_k)`.
4. **Network-Evaluated Fitness Function (with Hash Caching):**
   Given feasible chromosome $\mathbf{x}$, check `cache[tuple(x)]`. If not cached, restore all edges with $x_j = 1$ to their unflooded travel times across scenarios $s \in \{1, 2, 3\}$, recompute $\text{FAI}_{\text{net}, s}(\mathbf{x})$ and non-isolated population share $R_{\text{conn}, s}(\mathbf{x}) = 1 - S_{\text{iso}, s}(\mathbf{x})$, and evaluate:
   $$F(\mathbf{x}) = \frac{1}{3} \sum_{s=1}^3 \left[ \alpha \cdot \frac{\text{FAI}_{\text{net}, s}(\mathbf{x})}{\text{FAI}_{\text{net, base}}} + (1 - \alpha) \cdot R_{\text{conn}, s}(\mathbf{x}) \right] + \lambda \cdot \frac{\sum_{j=1}^N x_j P_{\text{fuzzy}, j}}{100 \cdot \max\left(1, \sum_{j=1}^N x_j\right)}$$
   where $\alpha = 0.60$ balances continuous travel-time accessibility against binary village isolation, and $\lambda = 0.15$ embeds the fuzzy expert preference.
5. **Evolutionary Operators ($G_{\text{max}} = 50$ generations, early stop patience $= 12$):**
   * **Selection:** Tournament selection ($k = 3$).
   * **Crossover:** Uniform crossover ($p_c = 0.80$, gene swap probability $0.5$).
   * **Mutation:** Bit-flip mutation ($p_m = 2.0 / N$ per gene).
   * **Elitism:** Top $2$ individuals preserved unchanged into the next generation.
   * **Multi-Seed Statistical Rigor:** Run $5$ random seeds per budget level and record `mean ± std` of retained FAI, isolated population, and convergence history.

> **Phase 3 Verification Gate:** Verify that all $27$ fuzzy rules fire without `NaN`s, `C(x) <= B_k` holds for $100\%$ of evaluated GA solutions, and best fitness $F(\mathbf{x})$ is non-decreasing across generations due to elitism.

---

### Phase 4: Baselines, Ablation/Sensitivity Suite, Visualization & Report
**Files:** `src/phase4_benchmarks_viz.py`, `run_pipeline.py`, `ASSUMPTIONS.md`

#### Step 4.1: Six-Method Comparative Benchmark
Evaluate all methods on the exact same network fitness metrics ($\text{FAI Retained } \%$, $\text{Hospital Isolated Pop } \%$, $\text{Any-Service Isolated Villages}$, $\text{OAI } \%$, $\text{Budget Used } \%$) across all 4 budgets ($5\%, 10\%, 15\%, 20\%$):
1. **Random Selection (Floor Baseline):** Average of $20$ random budget-repaired portfolios.
2. **Cheapest-First (Cost-Greedy):** Sort candidate corridors by ascending $\text{Cost}_j$ and pack greedily until budget $B_k$.
3. **Betweenness-Only Greedy (Topology Baseline):** Sort by descending $\text{Betweenness}_j / \text{Cost}_j$ and pack greedily.
4. **Crisp AHP-TOPSIS Greedy (MCDM Baseline):** Compute weighted normalized decision matrix on $[\hat{E}, \widehat{\Delta A}, \hat{B}_I]$ with AHP weights $[0.20, 0.35, 0.45]$, measure Euclidean distances $D_j^+, D_j^-$ to ideal/anti-ideal solutions, compute closeness $CC_j = \frac{D_j^-}{D_j^+ + D_j^-}$, and pack greedily by $CC_j / \text{Cost}_j$.
5. **Mamdani Fuzzy Greedy (Soft Computing Layer 1 Alone):** Pack greedily by fuzzy cost-efficiency $\rho_j = P_{\text{fuzzy}, j} / \text{Cost}_j$ without GA network synergy.
6. **Proposed Hybrid Fuzzy + GA (Soft Computing Layer 1 + Layer 2):** Full evolutionary optimization with network re-evaluation.

#### Step 4.2: Soft Computing Validation Experiments (Ablation & Sensitivity)
1. **Ablation Study:** Compare **GA with Fuzzy Seeding & Fuzzy Repair** vs. **Vanilla GA (Random Seeding & Random Drop Repair)** at the $10\%$ and $15\%$ budgets to prove that the Mamdani FIS accelerates convergence and reaches higher fitness.
2. **Fuzzy Sensitivity Analysis:** Shift all triangular/trapezoidal membership function interior breakpoints by $-10\%$ and $+10\%$, recompute fuzzy scores, and measure **Kendall's rank correlation ($\tau$)** and top-$20$ overlap against the baseline fuzzy ranking.

#### Step 4.3: Automated Figures, Interactive Map & Tables
Generate and save into `results/`:
1. `results/figures/fig1_fuzzy_membership_and_surface.png`: Membership functions and 3D control surface of $\text{Priority}(\hat{E}, \widehat{\Delta A} \mid \hat{B}_I = 0.8)$.
2. `results/figures/fig2_ga_convergence_and_ablation.png`: GA convergence curves across budgets + Fuzzy-GA vs. Vanilla-GA ablation curve.
3. `results/figures/fig3_benchmark_comparison.png`: Grouped plot of $\text{FAI Retained } (\%)$ and $\text{Isolated Population } (\%)$ vs. Budget Tier for all 6 methods.
4. `results/maps/flood_resilience_map.html`: Interactive `folium` map (or high-res spatial plot) showing intact roads (gray), flooded unprotected roads (red dashed), GA-protected resilient corridors (thick green), facilities (blue/purple markers), and villages colored by isolation status.
5. `results/tables/benchmark_results.csv` and `ASSUMPTIONS.md`: Complete quantitative tables ready to paste into the report and slides.

---

## 5. Data Contracts Between Phases (Zero-Error Interface Specification)

To guarantee that no phase ever crashes due to missing columns or mismatched types, every module must adhere to the following exact data structures:

| Interface Object | Produced By | Consumed By | Required Keys / Columns & Types |
| :--- | :--- | :--- | :--- |
| `G` (`nx.Graph`) | Phase 1 | Phases 2, 3, 4 | Node attrs: `x` (`float`), `y` (`float`), `elevation` (`float`). Edge attrs: `corridor_id` (`int`), `length_km` (`float`), `v_limit` (`float`), `t_base` (`float`), `h_rel` (`float`), `cost_usd` (`float`), `h_class_weight` (`float`), `w_s1, w_s2, w_s3` (`float`), `t_s1, t_s2, t_s3` (`float`). |
| `villages_df` (`DataFrame`) | Phase 1 | Phases 2, 3, 4 | `village_id` (`int`), `name` (`str`), `node_id` (`int`), `population` (`int`), `svi` (`float`), `effective_pop` (`float`), `x` (`float`), `y` (`float`). |
| `facilities_dict` (`dict`) | Phase 1 | Phases 2, 3, 4 | Keys: `'hospital'`, `'school'`, `'market'`. Values: `list[int]` of snapped graph `node_id`s. |
| `candidates_df` (`DataFrame`) | Phase 2 | Phases 3, 4 | `gene_idx` (`0..N-1`), `corridor_id` (`int`), `u` (`int`), `v` (`int`), `length_km` (`float`), `cost_usd` (`float`), `exp_raw` (`float`), `acc_raw` (`float`), `iso_raw` (`float`), `exp_norm` (`float`), `acc_norm` (`float`), `iso_norm` (`float`), `betweenness` (`float`). |
| `fuzzy_scores` (`np.ndarray`) | Phase 3.1 | Phase 3.2, 4 | `shape=(N,)`, `dtype=float64`, strictly bounded in $[0.0, 100.0]$. Added as column `fuzzy_priority` to `candidates_df`. |
| `benchmark_df` (`DataFrame`) | Phase 4 | Report / Plots | `budget_pct` (`int`), `method` (`str`), `fai_retained_mean` (`float`), `fai_retained_std` (`float`), `hosp_isolated_pop_pct` (`float`), `isolated_villages_count` (`float`), `oai_pct` (`float`), `budget_utilized_pct` (`float`), `runtime_sec` (`float`). |

---

## 6. Recommended Execution Workflow

1. **Step 1:** Implement and run **Phase 1** (`src/config.py` and `src/phase1_network_flood.py`) to build the graph, run the flood depth/speed model, and pass the Phase 1 Verification Gate.
2. **Step 2:** Implement and run **Phase 2** (`src/phase2_accessibility_criteria.py`) to verify the AHP weights, multi-source Dijkstra $FAI$/$OAI$ metrics, and normalized candidate criteria table.
3. **Step 3:** Implement and run **Phase 3** (`src/phase3_soft_computing.py`) to execute the 27-rule Mamdani Fuzzy Inference System and the Fuzzy-Seeded Constrained Genetic Algorithm.
4. **Step 4:** Implement and run **Phase 4** (`src/phase4_benchmarks_viz.py` and `run_pipeline.py`) to generate all baseline comparisons, ablation/sensitivity metrics, plots, interactive maps, and `ASSUMPTIONS.md`.