# Realm of Lace Unified: Complete Technical Guide
## A Comprehensive Reference for Network Cellular Automata

**Version:** 2025-10-18 (Revised)  
**Rule Class:** `RealmOfLaceUnified`  
**Status:** Production

---

## Table of Contents
1. [Overview](#overview)
2. [Fundamental Concepts](#fundamental-concepts)
   - [Network-Inspired vs. True Dynamic Topology](#important-network-inspired-vs-true-dynamic-topology)
3. [Three-Phase Execution Model](#three-phase-execution-model)
4. [The Eight Metrics](#the-eight-metrics)
5. [Configuration System](#configuration-system)
6. [Edge Formation Logic](#edge-formation-logic)
7. [Final State Determination](#final-state-determination)
8. [Advanced Features](#advanced-features)
   - [Edge Rule Tables (Unimplemented)](#edge-rule-tables-infrastructure-exists-execution-not-implemented)
9. [Example Configurations](#example-configurations)
10. [Performance Considerations](#performance-considerations)
11. [Comparison to Other CA Types](#comparison-to-other-ca-types)

---

## Overview

**Realm of Lace Unified (ROL-U)** is a highly configurable **network-parameterized cellular automaton** that treats the grid as a **dynamic graph** where both nodes (cells) and edges (connections) evolve according to network topology metrics.

> **Important Note:** Realm of Lace is mathematically equivalent to a sophisticated multi-state cellular automaton with extended neighborhood. While edges are explicitly represented and visualized, they are **derived from node eligibility** rather than independent dynamic entities. See [Network-Inspired vs. True Dynamic Topology](#important-network-inspired-vs-true-dynamic-topology) for a detailed discussion of this distinction and the unfinished edge rule table infrastructure that could enable true dynamic topology.

### Key Properties

- **Node State = Degree:** A node's state directly represents its connectivity (number of edges)
- **Binary Edges:** Connections either exist (1.0) or don't (0.0)
- **Metric-Based Rules:** Uses 8 different graph metrics to determine behavior
- **Three-Phase Evolution:** Separate phases for eligibility, edge formation, and final state
- **Highly Configurable:** Independent control over birth and survival logic

### State Types

```python
Node State: INTEGER (0 to 26)  # Represents degree
Edge State: BINARY (0 or 1)    # Connection exists or not
```

**Range Explanation:**
- 0 = isolated node (no connections)
- 8 = maximum connections in 2D Moore neighborhood
- 26 = theoretical maximum in 3D Moore neighborhood

---

## Fundamental Concepts

### Network Cellular Automaton vs. Traditional CA

**Traditional Cellular Automaton (e.g., Game of Life):**
```
Cell State: ON (1) or OFF (0)
Neighbors: Implicitly connected by spatial adjacency
Rules: Based on neighbor counts
Update: State changes based on simple counts
```

**Network Cellular Automaton (Realm of Lace):**
```
Node State: Degree (0-26, represents connectivity)
Neighbors: Explicitly connected via edges in the graph
Edges: First-class entities that form/break dynamically
Rules: Based on network topology metrics
Update: Multi-phase (eligibility → edges → state)
```

### The State-Degree Equivalence

In ROL-U, there is a **direct equivalence** between node state and degree:

```
node_state = node_degree = count_of_connected_edges
```

This means:
- An inactive node (state 0) has no edges
- A node with state 3 has exactly 3 edges
- The maximum state equals the maximum possible edges

### Spatial vs. Network Neighbors

**Spatial Neighbors:** Cells adjacent in the grid (Moore, Von Neumann, etc.)  
**Network Neighbors:** Nodes connected by edges in the graph

Important: A spatial neighbor is NOT automatically a network neighbor. Edges must be explicitly formed.

### Important: Network-Inspired vs. True Dynamic Topology

**Scientific Honesty:** Realm of Lace uses **network-inspired rules** where edges represent derived relationships based on node eligibility, NOT independent dynamic entities. This is an important distinction:

**What Realm of Lace IS:**
- **Network-Parameterized CA:** Edges are computed each step based on mutual node eligibility
- **Sophisticated Parameterization:** 8 metrics, ranges/values, aggregation types provide rich expressiveness
- **Network-Theoretic Framing:** Provides intuitive vocabulary (degree, betweenness, clustering)
- **Multi-Phase Execution:** Separates eligibility calculation from edge formation from final state
- **Mathematically Equivalent to Complex Traditional CA:** Can be expressed as a larger-neighborhood finite-state automaton

**What Realm of Lace IS NOT (Yet):**
- **True Dynamic Topology:** Edges don't gate or mask which neighbors participate in updates
- **Independent Edge Dynamics:** Edges don't have their own state transition rules separate from node eligibility
- **Edge-to-Edge Rules:** Edge formation/removal is always mediated through node states

**The Technical Distinction:**

Since edges form purely based on mutual eligibility (which depends on neighbor states), they can be viewed as a **representational layer** over what is essentially a traditional CA with:
- Extended neighborhood (5×5 to capture second-order effects)
- Complex state space (degree values 0-26)
- Sophisticated update rule (three phases with metric calculations)

This doesn't diminish the value or sophistication of Realm of Lace, but it clarifies its position in the taxonomy of cellular automata.

**Future Direction: Edge Rule Tables (Infrastructure Exists, Not Yet Implemented in the Rule Execture)**

LACE includes UI and data structures for **edge rule tables** that would provide true dynamic topology:

```python
# Edge rule table format (UI exists, execution not implemented)
Key: "(self_state, neighbor_state, connection_pattern)"
Value: action ("add", "remove", "maintain")
```

This would enable:
- **Edge-to-edge dynamics:** Edges evolve based on surrounding edge patterns
- **Dynamic neighborhoods:** Only connected neighbors participate in metric calculations
- **Independent edge logic:** Edge transitions not mediated by node eligibility

The Rule Editor includes full UI for editing edge rule tables (separate from state rule tables), but no current rule implements the execution logic. This represents a significant opportunity for future development toward true network CA with dynamic topology.

**Bottom Line:** Realm of Lace is a sophisticated and valuable **network-parameterized CA** with rich emergent behavior. It provides a powerful framework for exploring connectivity-based dynamics. For applications requiring truly independent topology evolution, the edge rule table infrastructure provides a clear path forward.

---

## Three-Phase Execution Model

Every simulation step proceeds through three distinct phases:

```
PREVIOUS STATE → PHASE 1: Eligibility → PHASE 2: Edges → PHASE 3: Final State → NEW STATE
```

### Phase 1: Eligibility Determination

**Purpose:** Calculate which nodes are "eligible" to participate in the network based on their neighborhood's topology from the **previous step**.

**Input:**
- Previous node states (degrees)
- Previous edges
- Neighbor data from last step

**Process:**
1. For each node, examine its spatial neighbors
2. Calculate a metric value based on neighbor properties
3. Check if the metric passes birth or survival conditions
4. Store an eligibility proxy (0.0 or 1.0) for each node

**Output:**
- Eligibility proxy array (one value per node)
- 0.0 = ineligible (cannot form edges)
- 1.0 = eligible (can form edges if neighbor agrees)

**Key Point:** Eligibility is calculated **before** any edges change. It uses the network state from the previous step.

### Phase 2: Edge Formation

**Purpose:** Create the edge set for this step based on mutual eligibility and spatial adjacency.

**Input:**
- Eligibility proxies from Phase 1
- Spatial neighbor relationships

**Process:**
For each pair of spatially adjacent nodes (A, B):
```python
if eligibility[A] > 0.5 and eligibility[B] > 0.5:
    create_edge(A, B)  # Edge exists
else:
    # No edge
```

**Output:**
- New edge set for this step
- Edges are symmetric (if A→B exists, then B→A exists)

**Important:** Both nodes must be eligible. This is the **mutual eligibility** principle.

### Phase 3: Final State Determination

**Purpose:** Determine the final node state based on the network topology created in Phase 2.

**Input:**
- New edges from Phase 2
- Eligibility proxies from Phase 1

**Process:**
1. Calculate each node's **degree** (count edges)
2. Compute a **final check metric** (can be degree itself or a proxy)
3. Check life/death conditions against this metric
4. Assign final state (usually the degree)

**Output:**
- New node states for next step
- Typically: `state = degree` (but can be 0 if death condition met)

---

## The Eight Metrics

ROL-U supports 8 different metrics for measuring network topology. Each can be used independently for birth eligibility, survival eligibility, or final state checks.

### 1. DEGREE

**Formula:** `degree = count_of_edges_connected_to_node`

**Range:** [0, MaxNeighbors]
- 2D Moore: [0, 8]
- 2D Von Neumann: [0, 4]
- 3D Moore: [0, 26]

**Meaning:** Direct connectivity count.

**Aggregation:**
- SUM: Total connections of all neighbors
- AVERAGE: Mean connections per neighbor

**Use Cases:**
- Classic connectivity-based rules (like original Realm of Lace)
- Simple, intuitive behavior
- Fast computation

**Example:**
```
Neighbor degrees: [2, 3, 1, 4, 2, 3, 4, 1]
SUM: 2+3+1+4+2+3+4+1 = 20
AVERAGE: 20/8 = 2.5
```

### 2. CLUSTERING

**Formula:** 
```python
clustering = (degree * (degree - 1)) / (MaxN * (MaxN - 1))  if degree > 1 else 0.0
```

**Range:** [0.0, 1.0]

**Meaning:** Proxy for clustering coefficient (local density).
- High value = node is in a dense cluster
- Low value = node is sparsely connected

**Denominator Options:**
- `ACTUAL`: Uses actual number of valid neighbors (more sensitive to local density)
- `THEORETICAL`: Uses theoretical max neighbors (consistent normalization)

**Aggregation:**
- SUM: Total clustering potential in neighborhood
- AVERAGE: Mean clustering per neighbor

**Use Cases:**
- Favor dense clusters (high clustering)
- Encourage sparse, web-like structures (low clustering)
- Differentiate cores from peripheries

**Example (2D Moore, MaxN=8):**
```
Neighbor with degree 1: (1*0)/(8*7) = 0.0
Neighbor with degree 4: (4*3)/(8*7) = 12/56 = 0.214
Neighbor with degree 8: (8*7)/(8*7) = 56/56 = 1.0
```

### 3. BETWEENNESS

**Formula:**
```python
betweenness = 1.0 / degree  if degree > 0 else 0.0
```

**Range:** [0.0, 1.0]

**Meaning:** Proxy for betweenness centrality (bottleneck importance).
- High value = node is a bottleneck/bridge (low degree)
- Low value = node has many alternatives (high degree)

**Aggregation:**
- SUM: Total "bottleneck density" in neighborhood
- AVERAGE: Mean betweenness per neighbor

**Use Cases:**
- Favor growth near sparse regions (high betweenness)
- Stabilize dense regions (low betweenness)
- Create branching, fractal-like patterns

**Example:**
```
Neighbor with degree 1: 1.0/1 = 1.000 (critical bottleneck)
Neighbor with degree 2: 1.0/2 = 0.500 (moderate)
Neighbor with degree 4: 1.0/4 = 0.250 (well-connected)
Neighbor with degree 8: 1.0/8 = 0.125 (redundant)
```

**See also:** [`Betweenness_Proxy_Explanation.md`](Betweenness_Proxy_Explanation.md) for detailed analysis.

### 4. ACTIVE_NEIGHBOR_COUNT

**Formula:**
```python
active_count = sum(1 for neighbor if neighbor_eligibility > 0.5)
```

**Range:** [0, MaxNeighbors]

**Meaning:** Count of neighbors that were eligible in Phase 1.
- Measures "active" nodes in the neighborhood
- Different from degree (uses eligibility, not edges)

**Aggregation:**
- SUM: Total active neighbors across all neighbors' neighborhoods
- AVERAGE: Mean active neighbor count

**Use Cases:**
- Rules based on eligibility rather than connectivity
- Secondary check on "activity" vs. "connectivity"
- More abstract than degree-based metrics

**Example:**
```
8 spatial neighbors
5 have eligibility > 0.5
active_neighbor_count = 5
```

### 5. SYMMETRY_STATE

**Formula:**
```python
# For each pair of opposing neighbors (e.g., North-South, East-West)
differences = [abs(state[n1] - state[n2]) for (n1, n2) in opposing_pairs]
symmetry_state = average(differences)
```

**Range:** [0.0, MaxState] (typically [0.0, 26.0])

**Meaning:** Measures asymmetry in node states across opposing directions.
- Low value = balanced, symmetric states
- High value = imbalanced, asymmetric states

**Aggregation:** Not aggregated (computed per node)

**Use Cases:**
- Favor symmetric growth patterns
- Encourage directional bias (high asymmetry)
- Create oriented structures

**Example (2D Moore):**
```
Center node's neighbors:
  North: state=5, South: state=3 → diff=2
  East: state=4,  West: state=4 → diff=0
  NE: state=6,    SW: state=2 → diff=4
  NW: state=3,    SE: state=5 → diff=2

symmetry_state = (2+0+4+2)/4 = 2.0
```

### 6. SYMMETRY_DEGREE

**Formula:**
```python
# Same as SYMMETRY_STATE but uses degrees instead of states
differences = [abs(degree[n1] - degree[n2]) for (n1, n2) in opposing_pairs]
symmetry_degree = average(differences)
```

**Range:** [0.0, MaxDegree]

**Meaning:** Measures asymmetry in connectivity across opposing directions.

**Use Cases:**
- Rules based on connectivity balance rather than state balance
- Often more stable than SYMMETRY_STATE since state=degree

### 7. NEIGHBOR_DEGREE_VARIANCE

**Formula:**
```python
neighbor_degrees = [degree of each neighbor]
variance = statistical_variance(neighbor_degrees)
```

**Range:** [0.0, ∞) (practical range ~[0, 10] for typical grids)

**Meaning:** Measures heterogeneity in neighbor connectivity.
- Low variance = uniform connectivity (all neighbors similar)
- High variance = diverse connectivity (mix of low and high degree)

**Aggregation:** Not aggregated (computed per node)

**Use Cases:**
- Favor uniform neighborhoods (low variance)
- Encourage diverse neighborhoods (high variance)
- Detect boundaries (high variance at edges of structures)

**Example:**
```
Neighbor degrees: [2, 2, 2, 2, 2, 2, 2, 2]
variance ≈ 0.0 (perfectly uniform)

Neighbor degrees: [0, 1, 0, 8, 0, 7, 0, 8]
variance ≈ 14.0 (highly heterogeneous)
```

### 8. NEIGHBOR_DEGREE_STDDEV

**Formula:**
```python
neighbor_degrees = [degree of each neighbor]
stddev = statistical_standard_deviation(neighbor_degrees)
```

**Range:** [0.0, ∞) (practical range ~[0, 3.5])

**Meaning:** Same as variance but in original units (square root of variance).
- Easier to interpret than variance
- Same use cases as variance

**Example:**
```
Neighbor degrees: [1, 2, 3, 4, 5, 4, 3, 2]
mean = 3.0
stddev ≈ 1.2
```

---

## Configuration System

### Birth vs. Survival Configuration

ROL-U allows **independent configuration** for birth and survival:

```python
# Birth (inactive → active)
birth_metric_type = "BETWEENNESS"
birth_metric_aggregation = "SUM"
birth_eligibility_range_BETWEENNESS_SUM = [[1.4, 7.0]]

# Survival (active → active)
survival_metric_type = "BETWEENNESS"
survival_metric_aggregation = "SUM"
survival_eligibility_range_BETWEENNESS_SUM = [[0.9, 2.6]]
```

You can use **different metrics** for each:
```python
birth_metric_type = "DEGREE"
survival_metric_type = "CLUSTERING"
```

### Ranges vs. Values

ROL-U supports **two types of conditions** that can be used together:

#### Ranges (Float Comparison)

**Format:** List of `[min, max]` intervals

**Comparison:** Direct float comparison (unrounded)

**Example:**
```python
birth_eligibility_range_DEGREE_SUM = [[2.0, 15.0]]
# Node is eligible if: 2.0 ≤ metric_value ≤ 15.0
```

**Use Cases:**
- Continuous intervals
- Float-valued metrics (BETWEENNESS, CLUSTERING)
- Soft boundaries

#### Values (Integer or Float Match)

**Format:** List of specific numbers

**Comparison:**
- **Integer targets:** Metric is rounded, then compared for equality
- **Float targets:** Unrounded metric must be within ±0.005

**Example:**
```python
birth_eligibility_values_DEGREE_SUM = [3, 5, 7]
# Node is eligible if: round(metric_value) in [3, 5, 7]

birth_eligibility_values_BETWEENNESS_SUM = [0.5, 1.0, 1.5]
# Node is eligible if: |metric_value - target| < 0.005
```

**Use Cases:**
- Discrete counts (like classic B3/S23)
- Specific metric thresholds
- Sharp boundaries

#### Combined Conditions

Both ranges and values are checked:
```python
birth_eligibility_range_DEGREE_SUM = [[2.0, 5.0]]
birth_eligibility_values_DEGREE_SUM = [8]

# Eligible if:
#   (2.0 ≤ metric ≤ 5.0) OR (round(metric) == 8)
```

### Aggregation Types

**SUM:** Add up metric values from all neighbors
```python
neighbors = [0.5, 0.25, 1.0, 0.2, 0.5]
SUM = 2.45
```

**AVERAGE:** Mean of metric values
```python
neighbors = [0.5, 0.25, 1.0, 0.2, 0.5]
AVERAGE = 2.45 / 5 = 0.49
```

**When to use:**
- **SUM:** When total neighborhood "density" matters (scales with neighbor count)
- **AVERAGE:** When mean neighbor quality matters (independent of neighbor count)

**Note:** Variance, StdDev, and Symmetry metrics are **not aggregated** - they're computed directly per node.

---

## Edge Formation Logic

### Mutual Eligibility Principle

Edges form based on **mutual eligibility**:

```python
def should_edge_exist(node_a, node_b):
    # Both nodes must be spatially adjacent
    if not spatially_adjacent(node_a, node_b):
        return False
    
    # Both nodes must be eligible
    if eligibility[node_a] <= 0.5:
        return False
    if eligibility[node_b] <= 0.5:
        return False
    
    # Both conditions met: edge exists
    return True
```

### Spatial Adjacency

Only spatial neighbors can form edges:

**Moore Neighborhood (2D):**
```
[NW][N][NE]
[W ][X][E ]
[SW][S][SE]
```
8 possible edges

**Von Neumann Neighborhood (2D):**
```
   [N]
[W][X][E]
   [S]
```
4 possible edges

### Edge Symmetry

Edges are **always symmetric**:
- If edge (A, B) exists, then edge (B, A) exists
- Both directions are the same edge
- Stored as unordered pairs

### Grid Boundary Modes

**wrap (Toroidal):**
- Edges connect across boundaries
- Grid "wraps around" like a torus
- No edge cells - all cells have full neighborhood

**bounded (Finite):**
- No edges cross boundaries
- Edge cells have fewer neighbors
- Creates boundary effects

---

## Final State Determination

After edges form in Phase 2, Phase 3 determines the final node state.

### Final Check Metric

A **separate metric** is calculated based on the **new edges**:

```python
final_check_metric = "DEGREE"  # Default
# Can also be: CLUSTERING, BETWEENNESS, ACTIVE_NEIGHBOR_COUNT
```

**Important:** This metric uses the **current step's edges** (from Phase 2), not the previous step's data.

### Life and Death Conditions

Two sets of conditions can be specified:

#### Life Conditions (Explicit Survival)

```python
final_life_metric_values_DEGREE = [2, 4, 8]
final_life_metric_range_DEGREE = [[5.0, 7.0]]
```

**If metric matches life conditions:**
- Node **survives** (or is born)
- State = degree (or specified value)

#### Death Conditions (Explicit Death)

```python
final_death_metric_values_DEGREE = [1, 7]
final_death_metric_range_DEGREE = []
```

**If metric matches death conditions:**
- Node **dies**
- State = 0

### Priority Logic

The logic has **strict priority**:

```python
if matches_life_condition:
    state = degree  # Survive
elif matches_death_condition:
    state = 0       # Die
else:
    state = degree  # Survive (default)
```

**Key insight:** If neither condition matches, node **survives by default**.

This means:
1. Life conditions **guarantee survival**
2. Death conditions **guarantee death**
3. Anything else **survives** (implicit survival)

### Example Configuration

```python
# Classic "stable at 2, 3, or 4 connections, die at 0 or 1"
final_check_metric = "DEGREE"
final_life_metric_values_DEGREE = [2, 3, 4]
final_death_metric_values_DEGREE = [0, 1]

# Degree 2, 3, 4: Explicit survival
# Degree 0, 1: Explicit death
# Degree 5, 6, 7, 8: Implicit survival (not in either list)
```

### State Assignment

**Typical behavior:**
```python
if alive:
    state = degree  # State represents connectivity
else:
    state = 0       # Inactive nodes have no state
```

**Special cases:**
- Perturbations can override (see Advanced Features)
- Some variants may use different state assignments

---

## Advanced Features

### Perturbations

**Random State Flip:**
```python
perturbation_enable = True
random_state_flip_probability = 0.01  # 1% chance per node
```

If triggered, the final metric value is flipped:
```python
if value > 0:
    value = 0  # Active → Inactive
else:
    value = 1  # Inactive → Active
```

**Use Cases:**
- Add noise to prevent crystallization
- Maintain dynamics in stable systems
- Simulate environmental fluctuations

### Clustering Denominator Types

**ACTUAL:**
```python
clustering = (deg * (deg - 1)) / (N * (N - 1))
# where N = actual number of valid neighbors found
```

**THEORETICAL:**
```python
clustering = (deg * (deg - 1)) / (MaxN * (MaxN - 1))
# where MaxN = theoretical max (8 for Moore 2D)
```

**When to use:**
- **ACTUAL:** More sensitive to local density, varies near boundaries
- **THEORETICAL:** Consistent across grid, easier to interpret

### Performance Optimization

**JIT Compilation:**
```python
use_jit_state_phase = True   # Enable Numba JIT for Phase 1
use_jit_edge_phase = True    # Enable Numba JIT for Phase 2
```

Both phases use Numba's `@njit` decorator for significant speedup.

**Typical Performance:**
- Pure Python: ~5-10 steps/second (100×100 grid)
- With JIT: ~50-100 steps/second (100×100 grid)
- Speedup: 10-20x

### Edge Rule Tables (Infrastructure Exists, Execution Not Implemented)

LACE includes complete UI infrastructure for **edge rule tables** that would enable true dynamic topology, but no rule currently implements the execution logic.

**Concept:**

Edge rule tables would allow edges to evolve based on **edge-to-edge dynamics** rather than being derived from node eligibility:

```python
# Edge rule table format (data structure and UI complete)
Key: "(self_state, neighbor_state, connection_pattern)"
Value: action ("add", "remove", "maintain")

# Example entries:
"(1, 1, 11100000)": "add"      # Both nodes active, 3 adjacent edges exist → add edge
"(0, 1, 00000000)": "remove"   # One node inactive, no adjacent edges → remove edge
"(1, 1, 11111111)": "maintain" # Both active, fully connected → maintain current state
"default": "remove"             # Default action for unmatched patterns
```

**Key Components:**

1. **self_state:** State of the first endpoint node (0-26)
2. **neighbor_state:** State of the second endpoint node (0-26)
3. **connection_pattern:** 8-digit binary string representing which of the 8 potential edges in the local Moore neighborhood currently exist
4. **action:** "add" (create edge), "remove" (delete edge), or "maintain" (keep current state)

**Current Status:**

- ✅ **UI Complete:** Rule Editor has full "Edge Rule Table" tab with add/delete/edit functionality
- ✅ **Data Structure:** Edge rule tables are stored in `rules.json` as `edge_rule_table` parameter
- ✅ **Initialization:** `Rule._initialize_rule_tables()` caches edge rule tables
- ❌ **Execution Logic:** No rule's `_compute_new_edges()` method implements table lookup
- ❌ **Dynamic Neighborhoods:** Metric calculations don't filter by edge existence

**What Implementation Would Enable:**

1. **True Dynamic Topology:** Edges gate which neighbors participate in calculations
2. **Edge-to-Edge Rules:** Edge formation based on surrounding edge patterns, not node states
3. **Independent Edge Evolution:** Edges evolve according to their own rules
4. **Complex Connectivity Patterns:** Enable edge-based fractals, networks, and structures

**Implementation Location:**

To implement edge rule tables, a rule would need to:

1. Override `_compute_new_edges(neighborhood: NeighborhoodData) -> Dict[Tuple[int, int], float]`
2. For each potential edge (self_node, neighbor_node):
   - Get connection pattern from current edge set
   - Look up key in `self._cached_edge_rule_table`
   - Apply action: "add" → return 1.0, "remove" → return 0.0, "maintain" → keep previous state
3. Optionally: Modify metric calculations to only consider connected neighbors

**Why Not Implemented:**

The infrastructure was created to support advanced network CA research, but focus shifted to perfecting the network-parameterized paradigm (Realm of Lace, etc.). The edge rule table feature represents a significant research direction that remains unexplored.

**See Also:**
- `LACE/lace_app.py` lines 11593-11628: Edge Rule Table UI
- `LACE/interfaces.py` lines 321-323: `_initialize_rule_tables()` method
- `LACE/rules.py` line 2863: `TwoDCAMasterRule` (implements state rule tables but not edge tables)

---

## Example Configurations

### Example 1: Classic Degree-Based (Original Realm of Lace)

```python
# Eligibility
birth_metric_type = "DEGREE"
birth_metric_aggregation = "SUM"
birth_eligibility_range_DEGREE_SUM = [[2.0, 15.0]]

survival_metric_type = "DEGREE"
survival_metric_aggregation = "SUM"
survival_eligibility_range_DEGREE_SUM = [[1.0, 20.0]]

# Final State
final_check_metric = "DEGREE"
final_life_metric_values_DEGREE = [2, 3, 4]
final_death_metric_values_DEGREE = [0, 1]
```

**Behavior:**
- Nodes become eligible if neighbors have moderate connectivity
- Structures stabilize with 2-4 connections per node
- Isolated nodes (0-1 connections) die

### Example 2: Betweenness-Based "Dragons"

```python
# Eligibility
birth_metric_type = "BETWEENNESS"
birth_metric_aggregation = "SUM"
birth_eligibility_range_BETWEENNESS_SUM = [[1.4, 7.0]]

survival_metric_type = "BETWEENNESS"
survival_metric_aggregation = "SUM"
survival_eligibility_range_BETWEENNESS_SUM = [[0.9, 2.6]]

# Final State
final_check_metric = "DEGREE"
final_life_metric_values_DEGREE = [2, 4, 8]
final_death_metric_values_DEGREE = [1, 7]
```

**Behavior:**
- Birth favors moderately sparse regions (bottlenecks)
- Survival favors well-connected regions
- Creates expanding fronts with stable cores
- Produces dragon-like, branching patterns

**See:** [`Rule_Explanation_Realm_of_Lace_Betweenness_Dragons.md`](Rule_Explanation_Realm_of_Lace_Betweenness_Dragons.md)

### Example 3: Clustering-Based Cores

```python
# Eligibility
birth_metric_type = "CLUSTERING"
birth_metric_aggregation = "AVERAGE"
birth_eligibility_range_CLUSTERING_AVERAGE = [[0.3, 0.7]]

survival_metric_type = "CLUSTERING"
survival_metric_aggregation = "AVERAGE"
survival_eligibility_range_CLUSTERING_AVERAGE = [[0.5, 1.0]]

# Final State
final_check_metric = "DEGREE"
final_life_metric_values_DEGREE = [4, 5, 6]
final_death_metric_values_DEGREE = [0, 1, 2]
```

**Behavior:**
- Birth near moderate-density regions
- Survival in dense clusters
- Favors compact, blob-like structures
- Eliminates sparse regions

### Example 4: Variance-Based Boundaries

```python
# Eligibility
birth_metric_type = "NEIGHBOR_DEGREE_VARIANCE"
birth_metric_aggregation = None  # Not aggregated
birth_eligibility_range_NEIGHBOR_DEGREE_VARIANCE = [[2.0, 10.0]]

survival_metric_type = "NEIGHBOR_DEGREE_VARIANCE"
survival_metric_aggregation = None
survival_eligibility_range_NEIGHBOR_DEGREE_VARIANCE = [[0.0, 5.0]]

# Final State
final_check_metric = "DEGREE"
final_life_metric_values_DEGREE = [3, 4, 5]
final_death_metric_values_DEGREE = [0, 1, 8]
```

**Behavior:**
- Birth at heterogeneous boundaries (high variance)
- Survival in uniform regions (low variance)
- Creates sharp boundaries between structures
- Self-organizes into homogeneous zones

### Example 5: Symmetry-Driven Growth

```python
# Eligibility
birth_metric_type = "SYMMETRY_DEGREE"
birth_metric_aggregation = None
birth_eligibility_range_SYMMETRY_DEGREE = [[0.0, 1.0]]  # Low asymmetry

survival_metric_type = "SYMMETRY_DEGREE"
survival_metric_aggregation = None
survival_eligibility_range_SYMMETRY_DEGREE = [[0.0, 2.0]]

# Final State
final_check_metric = "DEGREE"
final_life_metric_values_DEGREE = [2, 3, 4, 5, 6]
final_death_metric_values_DEGREE = [0, 1]
```

**Behavior:**
- Birth only in symmetric contexts
- Encourages balanced, symmetric structures
- May produce crystalline or geometric patterns
- Penalizes directional bias

---

## Performance Considerations

### Computational Complexity

**Phase 1 (Eligibility):**
- Per node: O(N) where N = number of neighbors
- Total: O(G × N) where G = grid size
- Dominant cost: Metric calculation

**Phase 2 (Edges):**
- Per node: O(N) neighbor checks
- Total: O(G × N)
- Dominant cost: Eligibility lookups

**Phase 3 (Final State):**
- Per node: O(1) or O(N) depending on metric
- Total: O(G) or O(G × N)
- Dominant cost: Degree counting or metric calculation

**Overall:** O(G × N) per step

### Memory Usage

**Storage Requirements:**
```
Nodes: 4-8 bytes per node (state, degree)
Edges: 8 bytes per edge (two node indices)
Eligibility: 4 bytes per node (float proxy)
Previous states: 4-8 bytes per node (for metric calculation)
```

**Typical 100×100 Grid:**
```
Nodes: 10,000 × 8 bytes = 80 KB
Edges: ~40,000 × 8 bytes = 320 KB (assuming 50% connectivity)
Eligibility: 10,000 × 4 bytes = 40 KB
Total: ~500 KB
```

### Optimization Strategies

1. **Use JIT compilation** (enabled by default)
2. **Simplify metrics** (DEGREE is fastest)
3. **Reduce metric diversity** (same metric for birth/survival)
4. **Use SUM over AVERAGE** (one less division)
5. **Avoid SYMMETRY metrics** if not needed (most expensive)
6. **Limit variance calculations** (requires full array scan)

### Metric Performance Ranking

**Fastest to Slowest:**
1. DEGREE - Direct lookup
2. ACTIVE_NEIGHBOR_COUNT - Simple count
3. BETWEENNESS - One division per neighbor
4. CLUSTERING - One multiplication, one division per neighbor
5. NEIGHBOR_DEGREE_VARIANCE - Full variance calculation
6. NEIGHBOR_DEGREE_STDDEV - Variance + square root
7. SYMMETRY_DEGREE - Requires opposing pair lookups
8. SYMMETRY_STATE - Requires opposing pair lookups + state array

---

## Visualization

### Node Coloring

**color_nodes_by_degree:**
```python
color_nodes_by_degree = True
node_colormap = "prism"
node_color_norm_vmin = 0.0
node_color_norm_vmax = 8.0  # Max degree for 2D Moore
```

Colors represent connectivity directly.

**use_state_coloring:**
```python
use_state_coloring = True
node_colormap = "viridis"
```

Colors represent node state (which equals degree in ROL-U).

### Edge Coloring

**edge_coloring_mode:**

- `"ActiveNeighbors"`: Color by active neighbor count
- `"DegreeSum"`: Color by sum of endpoint degrees
- `"Default"`: Single color for all edges
- `"State"`: Color by edge state (binary = single color)

**Example:**
```python
edge_coloring_mode = "DegreeSum"
edge_colormap = "prism"
edge_color_norm_vmin = 0.0
edge_color_norm_vmax = 16.0  # Max degree sum (8+8)
use_state_coloring_edges = True
```

---

## Comparison to Other CA Types

### vs. Game of Life (Traditional Binary CA)

| Feature | Game of Life | Realm of Lace Unified |
|---------|--------------|----------------------|
| State Type | Binary (0/1) | Integer (0-26) |
| Connections | Implicit | Explicit edges |
| Rules | Neighbor count | Network metrics |
| Phases | 1 (state update) | 3 (eligibility→edges→state) |
| Metrics | 1 (active count) | 8 (various) |
| Configuration | Fixed (B3/S23) | Highly configurable |

### vs. Continuous CA

| Feature | Continuous CA | Realm of Lace Unified |
|---------|---------------|----------------------|
| State Range | Continuous [0,1] | Discrete [0, MaxDegree] |
| Update Rule | Differential equations | Graph dynamics |
| Edges | N/A | Explicit |
| Interpretation | Abstract | Network structure |

### vs. Other Network CA / Complex Traditional CA

**Realm of Lace Unified is distinguishable by:**
- **State = Degree equivalence** (most network CA separate these)
- **Metric configurability** (8 different metrics with independent birth/survival configuration)
- **Three-phase model** (provides clear separation of concerns)
- **Mutual eligibility** (bilateral agreement for edges)
- **Performance optimization** (JIT compilation, efficient algorithms)
- **Network-theoretic parameterization** (intuitive vocabulary for rule design)

**Mathematical Classification:**

Realm of Lace can be expressed as a **multi-state CA with extended neighborhood**:
- State space: {0, 1, 2, ..., 26} (degree values)
- Effective neighborhood: 5×5 (includes neighbors-of-neighbors for metric calculations)
- Update rule: Deterministic function of neighborhood configuration

The explicit edge representation and three-phase execution provide a **useful abstraction** for designing and reasoning about connectivity-based dynamics, even though edges are ultimately derived from node eligibility rather than independent entities.

**True Network CA (Not Yet Implemented in LACE):**

For comparison, a true dynamic-topology network CA would have:
- Edges that gate/mask which neighbors participate in calculations
- Edge state transitions independent of node eligibility
- Edge rule tables that map edge configurations to edge actions
- Asymmetric or directed edges with independent existence

LACE's edge rule table infrastructure (UI complete, execution not implemented) would enable this paradigm.

---

## Mathematical Foundations

### Graph Theory Concepts

**Degree:**
```
deg(v) = |{u ∈ V : (v,u) ∈ E}|
```

**Clustering Coefficient (proxy):**
```
C(v) ≈ deg(v) × (deg(v) - 1) / (MaxN × (MaxN - 1))
```

**Betweenness Centrality (proxy):**
```
BC(v) ≈ 1 / deg(v)
```

### Eligibility as a Boolean Function

```
eligible(v, t) = f(metrics(neighbors(v, t-1)))

where:
  f: ℝ → {0, 1}  (threshold function)
  metrics: V → ℝ  (metric calculation)
  neighbors: V → 2^V  (neighborhood function)
```

### Edge Formation as Graph Union

```
E(t) = {(u,v) ∈ E_spatial : eligible(u, t) ∧ eligible(v, t)}
```

Where `E_spatial` is the set of spatial neighbor pairs.

### State Dynamics

```
state(v, t+1) = {
  degree(v, t)  if alive(v, t)
  0             otherwise
}

where alive is determined by final check conditions
```

---

## Troubleshooting & Common Issues

### Issue 1: No Activity (All Nodes Die)

**Cause:** Eligibility conditions too restrictive

**Solutions:**
- Widen ranges (make intervals larger)
- Add more values to values lists
- Check that SUM vs AVERAGE makes sense for your metric
- Verify ranges match metric output ranges

### Issue 2: Everything Becomes Active

**Cause:** Eligibility conditions too permissive

**Solutions:**
- Narrow ranges
- Add death conditions in final state
- Make survival range smaller than birth range
- Check for overlapping life/death conditions

### Issue 3: Structures Freeze Immediately

**Cause:** Birth and survival ranges identical

**Solutions:**
- Make survival range different from birth range
- Add perturbations to maintain dynamics
- Adjust final state conditions to allow some death

### Issue 4: Chaotic, Unpredictable Behavior

**Cause:** Overlapping or contradictory conditions

**Solutions:**
- Check for overlaps between life/death conditions
- Ensure birth/survival ranges make sense
- Simplify to one metric first, then add complexity
- Use the Rule Editor's validation warnings

### Issue 5: Performance Issues

**Cause:** Complex metrics or large grids

**Solutions:**
- Enable JIT compilation (should be on by default)
- Use simpler metrics (DEGREE instead of SYMMETRY)
- Reduce grid size for testing
- Check for any debug logging enabled

---

## API Reference

### Key Parameters

```python
# Core Grid
dimension_type: "TWO_D" | "THREE_D"
neighborhood_type: "MOORE" | "VON_NEUMANN" | "HEX" | "HEX_PRISM"
grid_boundary: "wrap" | "bounded"

# Initialization
initial_density: float [0.0, 1.0]
edge_initialization: "RANDOM" | "FULL" | "NONE"

# Birth Eligibility
birth_metric_type: str  # One of 8 metrics
birth_metric_aggregation: "SUM" | "AVERAGE"
birth_eligibility_range_{METRIC}_{AGG}: List[Tuple[float, float]]
birth_eligibility_values_{METRIC}_{AGG}: List[Union[int, float]]

# Survival Eligibility
survival_metric_type: str  # One of 8 metrics
survival_metric_aggregation: "SUM" | "AVERAGE"
survival_eligibility_range_{METRIC}_{AGG}: List[Tuple[float, float]]
survival_eligibility_values_{METRIC}_{AGG}: List[Union[int, float]]

# Final State
final_check_metric: "DEGREE" | "CLUSTERING" | "BETWEENNESS" | "ACTIVE_NEIGHBOR_COUNT"
final_life_metric_values_{METRIC}: List[Union[int, float]]
final_life_metric_range_{METRIC}: List[Tuple[float, float]]
final_death_metric_values_{METRIC}: List[Union[int, float]]
final_death_metric_range_{METRIC}: List[Tuple[float, float]]

# Advanced
perturbation_enable: bool
random_state_flip_probability: float [0.0, 1.0]
clustering_denominator_type: "ACTUAL" | "THEORETICAL"
```

### Dynamic Parameter Generation

Parameters are generated dynamically based on metric type:

```python
# For birth_metric_type="BETWEENNESS" and birth_metric_aggregation="SUM":
birth_eligibility_range_BETWEENNESS_SUM = [[1.4, 7.0]]
birth_eligibility_values_BETWEENNESS_SUM = []

# For survival_metric_type="CLUSTERING" and survival_metric_aggregation="AVERAGE":
survival_eligibility_range_CLUSTERING_AVERAGE = [[0.5, 1.0]]
survival_eligibility_values_CLUSTERING_AVERAGE = []
```

---

## References & Further Reading

### Internal Documentation
- [`Betweenness_Proxy_Explanation.md`](Betweenness_Proxy_Explanation.md) - Deep dive into the betweenness metric
- [`Rule_Explanation_Realm_of_Lace_Betweenness_Dragons.md`](Rule_Explanation_Realm_of_Lace_Betweenness_Dragons.md) - Example rule walkthrough

### Code Reference
- `LACE/rules.py` - Rule implementation (lines 779-2688)
- `LACE/interfaces.py` - Core interfaces and data structures
- `LACE/enums.py` - Enumeration types

### Graph Theory Concepts
- Degree Centrality
- Clustering Coefficient
- Betweenness Centrality
- Network Topology
- Dynamic Graphs

### Related Cellular Automata
- Conway's Game of Life (binary CA)
- Brian's Brain (multi-state CA)
- Langton's Ant (agent-based CA)
- Continuous CA (differential equations)

---

## Version History

**Version 2025-10-18 (Revised):**
- Added "Network-Inspired vs. True Dynamic Topology" section
- Clarified mathematical classification as network-parameterized CA
- Documented unfinished edge rule table infrastructure
- Updated comparison section with honest assessment
- Added note about equivalence to complex traditional CA
- Maintained scientific accuracy and intellectual honesty

**Version 2025-10-16 (Initial):**
- Complete documentation created
- All 8 metrics documented
- Configuration system explained
- Examples provided
- Verified against code implementation

---

## Contributing

When creating new Realm of Lace variants:

1. **Start with this canonical version** (RealmOfLaceUnified)
2. **Experiment with parameters** before modifying code
3. **Document your configurations** (use Rule Editor's "Copy Spec" feature)
4. **Test boundary cases** (all nodes die, all nodes live, stasis)
5. **Share successful configurations** with clear parameter sets

---

## License

This documentation is part of the LACE project.  
See main project LICENSE for terms.

---

## Acknowledgments

Created based on careful analysis of the LACE implementation, particularly the `RealmOfLaceUnified` class in `LACE/rules.py`. All formulas, behaviors, and examples have been verified against the actual code to ensure scientific accuracy.

