# EvoRoute System Assumptions & Technical Specifications

This document records the mathematical, physical, and algorithmic assumptions underpinning the **Flood-Resilient Road Priority Ranking System**.

---

## 1. Hydrological & Topographic Assumptions
1. **Height Above Nearest Drainage (HAND) Proxy**:
   * Ground surface elevation $z(e)$ and nearest river drainage elevation $z_{\text{river}}(e)$ define edge relative height $h_{\text{rel}}(e) = \max(0, z(e) - z_{\text{river}}(e))$.
   * Flood stages $H_1 = 1.0\text{ m}$, $H_2 = 2.0\text{ m}$, and $H_3 = 3.0\text{ m}$ represent progressive hydrostatic water-stage rises (Moderate, Severe, Extreme climate scenarios), consistent with Sangeeta et al. (EGU 2025).
2. **Speed Degradation & Closure**:
   * Calculated via the empirical Mitropoulos (2025) / Pregnolato et al. (2017) model:
     $$v(w) = 0.0009w^2 - 0.5529w + 86.9448$$
   * Roads with water depth $w \ge 200\text{ mm}$ ($0.2\text{ m}$) are completely impassable ($v = 0\text{ km/h}, t = \infty$).
   * Speeds under partial flooding are capped at $85\%$ of free-flow velocity.

---

## 2. Accessibility & Demographic Weighting
1. **Social Vulnerability Index (SVI)**:
   * Rural settlements are weighted by demographic vulnerability $\text{SVI}_i \in [0.80, 1.40]$, reflecting higher proportions of children, elderly, disabled, and low-income residents near river basins (Sangeeta et al. 2025).
   * Effective vulnerable population: $P_i^{\text{eff}} = P_i \times \text{SVI}_i$.
2. **Hansen Gravity Accessibility**:
   * Inverse travel-time impedance: $\beta = 1.0$.
   * Facility hierarchy determined via Saaty's Analytic Hierarchy Process ($CR = 0.0032 \le 0.10$):
     * Hospital ($W_{\text{hosp}} = 0.648$)
     * School ($W_{\text{school}} = 0.230$)
     * Market ($W_{\text{market}} = 0.122$)
3. **Emergency Capacity Tracking**:
   * Destination capacities ($H_j$) from Akhlaghi et al. (arXiv:2302.07983) are tracked diagnostically to verify secondary facility offloading.

---

## 3. Soft Computing & Optimization Architecture
1. **Mamdani Fuzzy Inference System (Layer 1)**:
   * Universe of Discourse: Inputs $[0.0, 1.0]$ (step $0.01$), Output $[0.0, 100.0]$ (step $0.5$).
   * $27$ linguistic rules generated programmatically with safety-critical overrides for lifeline isolation.
   * Centroid defuzzification used for crisp priority scoring.
2. **Constrained Genetic Algorithm (Layer 2)**:
   * Binary encoding $\mathbf{x} \in \{0, 1\}^{19}$.
   * Domain-specific fuzzy repair operator guarantees $100\%$ budget feasibility by prioritizing high $\rho_j = \frac{P_{\text{fuzzy}, j} + 1.0}{\text{Cost}_j}$ corridors.
   * Fitness balancing accessibility retention ($60\%$), hospital reconnection ($40\%$), and fuzzy preference ($15\%$ regularizer).
   * Evaluated across 4 graduated budget tiers ($5\%, 10\%, 15\%, 20\%$) over 5 independent random seeds.
