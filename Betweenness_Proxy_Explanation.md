# The Betweenness Proxy in LACE
## Detailed Technical Explanation

---

> **Note:** This document focuses on the **BETWEENNESS** metric, but Realm of Lace rules support multiple eligibility metrics. Each metric provides a different lens on network topology, allowing you to create diverse emergent behaviors.
>
> **Available Metrics:**
> - **DEGREE** - Connection count (number of edges)
> - **CLUSTERING** - Local density proxy (triangular connections)
> - **BETWEENNESS** - Centrality/bottleneck proxy (1/degree) ← *This document*
> - **ACTIVE_NEIGHBOR_COUNT** - Count of neighbors with eligibility > 0.5
> - **SYMMETRY_STATE** - State balance of opposing neighbors
> - **SYMMETRY_DEGREE** - Degree balance of opposing neighbors
> - **NEIGHBOR_DEGREE_VARIANCE** - Variance in neighbor connectivity
> - **NEIGHBOR_DEGREE_STDDEV** - Standard deviation in neighbor connectivity
>
> You can configure **different metrics** for birth vs. survival, and choose between **SUM** or **AVERAGE** aggregation. This flexibility enables fine-tuned control over network growth dynamics.

---

## What is Betweenness Centrality?

**True Betweenness Centrality** (in graph theory) measures how often a node lies on the shortest path between other pairs of nodes in the graph. It's computed as:

```
Betweenness(v) = Σ (σ_st(v) / σ_st)
```

Where:
- `σ_st` = total number of shortest paths from node s to node t
- `σ_st(v)` = number of those paths that pass through node v

**Why it matters:** Nodes with high betweenness are "bridges" or "gatekeepers" - removing them fragments the network.

**The problem:** Computing true betweenness requires:
1. Finding all shortest paths between all pairs of nodes (expensive!)
2. Counting which paths go through each node (very expensive!)
3. For a 100×100 grid, that's ~10,000 nodes × 10,000 nodes = 100 million path computations per step

This is computationally **prohibitive** for real-time cellular automata.

---

## The Betweenness Proxy

LACE uses a **fast approximation** that correlates with betweenness but can be computed instantly:

```python
betweenness_proxy = 1.0 / degree  if degree > 0 else 0.0
```

### The Formula

For a node with degree `d`:
- **Degree = 0:** `betweenness_proxy = 0.0`
- **Degree = 1:** `betweenness_proxy = 1.0` (maximum proxy value)
- **Degree = 2:** `betweenness_proxy = 0.5`
- **Degree = 4:** `betweenness_proxy = 0.25`
- **Degree = 8:** `betweenness_proxy = 0.125` (minimum for Moore 2D)

### Why This Works

The proxy is based on an **inverse relationship** between degree and betweenness:

**Low Degree (high proxy value):**
- A node with degree 1 is a **bottleneck** - if you want to reach its neighbor, you MUST go through it
- These nodes are like bridges connecting regions
- Example: A peninsula connecting to a mainland

**High Degree (low proxy value):**
- A node with degree 8 (in Moore 2D) is in a **dense cluster**
- There are many alternative paths around it
- It's rarely a necessary bridge
- Example: A node in the middle of a solid block

**The Correlation:**
```
True Betweenness ≈ 1/Degree (for local network structures)
```

This isn't perfect, but it captures the key insight: **sparse connections = high betweenness**.

---

## How It's Used in Your Rule

### During Eligibility Calculation

For each node, the rule computes the **SUM** of betweenness proxies from all neighbors:

```python
betweenness_sum = 0.0
for neighbor in neighbors:
    neighbor_degree = get_neighbor_degree(neighbor)  # from previous step
    if neighbor_degree > 0:
        betweenness_sum += 1.0 / neighbor_degree
```

### Concrete Example

Imagine a node with 8 neighbors in Moore 2D, with these degrees from the previous step:

```
Neighbor 1: degree = 2  →  proxy = 1.0/2 = 0.500
Neighbor 2: degree = 4  →  proxy = 1.0/4 = 0.250
Neighbor 3: degree = 1  →  proxy = 1.0/1 = 1.000
Neighbor 4: degree = 0  →  proxy = 0.000
Neighbor 5: degree = 5  →  proxy = 1.0/5 = 0.200
Neighbor 6: degree = 3  →  proxy = 1.0/3 = 0.333
Neighbor 7: degree = 2  →  proxy = 1.0/2 = 0.500
Neighbor 8: degree = 4  →  proxy = 1.0/4 = 0.250

Total SUM = 0.5 + 0.25 + 1.0 + 0.0 + 0.2 + 0.333 + 0.5 + 0.25
          = 3.033
```

### Interpreting the Sum

**For Birth (inactive → active):**
- Your rule requires: `betweenness_sum ∈ [(1.4, 7.0)]`
- Sum = 3.033 → **PASSES** (within range)
- **Meaning:** This node is surrounded by **moderately connected neighbors** (mix of sparse and dense)
- It can be born into this configuration

**For Survival (active → active):**
- Your rule requires: `betweenness_sum ∈ [(0.9, 2.6)]`
- Sum = 3.033 → **FAILS** (too high)
- **Meaning:** The neighbors are **too sparsely connected** for stability
- This node would die or become ineligible

---

## What The Ranges Mean

### Birth Range: [(1.4, 7.0)]

**Lower bound (1.4):**
- Equivalent to ~5 neighbors with degree 2 (5 × 0.5 = 2.5, but we need some variety)
- Or ~2 neighbors with degree 1 (2 × 1.0 = 2.0)
- **Interpretation:** Needs at least some sparse connectivity in the neighborhood

**Upper bound (7.0):**
- Equivalent to ~7 neighbors with degree 1 (7 × 1.0 = 7.0)
- Or ~28 neighbors with degree 4 (but you only have 8 neighbors max in Moore 2D)
- **Interpretation:** Prevents birth in extremely sparse, isolated regions

**Sweet spot:** Birth happens when you have a **mix of moderately connected nodes** - not too isolated, not too dense.

### Survival Range: [(0.9, 2.6)]

**Lower bound (0.9):**
- Equivalent to ~4 neighbors with degree 8 (4 × 0.125 = 0.5, need more)
- Or ~2 neighbors with degree 2 (2 × 0.5 = 1.0)
- **Interpretation:** Needs high density / well-connected neighborhood

**Upper bound (2.6):**
- Equivalent to ~5 neighbors with degree 2 (5 × 0.5 = 2.5)
- Or ~3 neighbors with degree 1 (3 × 1.0 = 3.0, but we cut off earlier)
- **Interpretation:** Can't survive in too sparse a neighborhood

**Sweet spot:** Survival happens when surrounded by **well-connected neighbors** (higher degrees = lower proxy values = lower sum).

---

## Network Topology Implications

### The Phase Transition

```
Birth:    betweenness_sum ∈ [1.4, 7.0]  →  Medium-high proxy values
                                          →  Medium-low degrees
                                          →  Moderately sparse neighborhoods

Survival: betweenness_sum ∈ [0.9, 2.6]  →  Low-medium proxy values
                                          →  Medium-high degrees
                                          →  Dense, well-connected neighborhoods
```

### The Dynamics

1. **Expansion Front:**
   - New nodes grow at the edges of structures (moderate connectivity)
   - Birth range is HIGHER (1.4-7.0), allowing growth into sparser regions

2. **Stable Core:**
   - Existing nodes survive in dense regions (high connectivity)
   - Survival range is LOWER (0.9-2.6), requiring denser neighborhoods

3. **Pruning:**
   - Nodes in intermediate density (sum 2.6-1.4) can't birth OR survive
   - This creates a **gap** that prunes unstable regions

4. **Result:**
   - Structures have **dense cores** (survival zone)
   - Structures have **expanding fronts** (birth zone)
   - There's a **forbidden zone** between them that creates boundaries

---

## Why "Proxy" Instead of True Betweenness?

### Computational Cost

**True Betweenness:**
- O(N³) or O(N²log N) with optimizations
- For 100×100 grid: ~1 billion operations per step
- Requires global graph analysis
- **Infeasible for real-time CA**

**Proxy:**
- O(1) per node (just 1/degree)
- For 100×100 grid: ~10,000 operations per step
- Uses only local information
- **Perfect for real-time CA**

### Approximation Quality

The proxy **captures the essential relationship:**
- Sparse connectivity → High importance (bottleneck/bridge)
- Dense connectivity → Low importance (redundant paths)

It **doesn't capture:**
- Global network position
- Long-range path structures
- Community boundaries

But for local CA rules, the **local approximation is sufficient and meaningful**.

---

## Visual Intuition

Think of betweenness proxy as "**flow capacity**":

```
Degree 1: |===[BOTTLENECK]===|     (proxy = 1.0)
          Everything must go through here!

Degree 4: |≡≡≡[NODE]≡≡≡|          (proxy = 0.25)
          Four paths available, less critical

Degree 8: |≡≡≡≡[NODE]≡≡≡≡|        (proxy = 0.125)
          Many alternative paths, rarely needed
```

The **sum** tells you about the neighborhood's overall "bottleneck density":
- **High sum** = many bottlenecks nearby (sparse, fragmented)
- **Low sum** = few bottlenecks (dense, well-connected)

---

## Connection to Real Graph Theory

The formula `1/degree` is related to several graph-theoretic concepts:

1. **Harmonic Centrality:** Inversely related to degree
2. **Resistance Distance:** Higher for sparse connections
3. **Flow Betweenness:** Nodes with low degree are natural bottlenecks

The proxy is a **legitimate graph metric**, just not the traditional betweenness centrality formula.

---

## Summary

**The Betweenness Proxy:**
- Formula: `1.0 / degree` (if degree > 0, else 0)
- Fast: O(1) per node
- Meaningful: Captures bottleneck behavior
- Local: Uses only immediate connectivity

**In the Amazing Dragons Rule:**
- Summed across neighbors to measure local network density
- Birth favors moderate connectivity (sum 1.4-7.0)
- Survival favors high connectivity (sum 0.9-2.6)
- Creates expanding structures with stable cores

**Why It Works:**
- Inversely correlates with true betweenness for local structures
- Computationally efficient for real-time simulation
- Captures essential network topology dynamics
- Creates rich, dragon-like emergent patterns 🐉

