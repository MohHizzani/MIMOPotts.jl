# MIMO Potts detection: model, equations, and pseudocode

This document specifies the full detection pipeline implemented in
`src/solver.jl`, including the two extensions added in July 2026: the
**whitened reliability ranking** (`reliability_mode = :whitened`) and the
**depth-first sphere-decoder branch enumeration** (`branch_strategy = :sphere`).

## 1. System model

Complex transmission is expressed in the real-valued augmented form

```
y = H x + n,      H ∈ R^{m×n},  m = 2 n_r,  n = 2 n_t,
```

where each entry of `x` is drawn from the per-dimension PAM alphabet
`L = {ℓ_1 < … < ℓ_k}` (the real/imaginary projection of the QAM
constellation, unit average symbol energy) and `n ~ N(0, σ² I)`.
Maximum-likelihood detection is the lattice search

```
x_ML = argmin_{x ∈ L^n}  ‖y − H x‖².                                (1)
```

The pipeline approximates (1) in three stages:

1. **Linear preprocessing** → soft estimate, initial radius, reliability ranking.
2. **Branch enumeration** over the `n − f` *fixed* dimensions
   (`f = free_dims`) → a list of branches with lower bounds.
3. **Potts machine** per branch over the `f` *free* dimensions, with
   radius-based branch pruning across trials.

## 2. Stage 1 — linear preprocessing and reliability

ZF and MMSE estimates:

```
x_zf   = H⁺ y,
x_mmse = (HᵀH + σ² I)⁻¹ Hᵀ y.
```

Both are quantized to the alphabet; whichever quantized point has the smaller
residual `‖y − H x̂‖²` becomes the *base* solution. Its distance initializes
the search radius

```
r₀² = ‖y − H x_base‖².                                              (2)
```

### Reliability ranking (which dimensions become free)

For each dimension `i`, let `d(1)_i ≤ d(2)_i` be the squared distances from the
soft estimate `x̂_i` to its nearest and second-nearest alphabet levels.

* `reliability_mode = :margin` (legacy):

  ```
  ρ_i = d(2)_i − d(1)_i.                                            (3)
  ```

* `reliability_mode = :whitened`: the margin is divided by the
  post-equalization error variance of the chosen linear detector,

  ```
  ε_i = σ² [(HᵀH)⁻¹]_{ii}            (ZF base)
  ε_i = σ² [(HᵀH + σ²I)⁻¹]_{ii}      (MMSE base)

  ρ_i = (d(2)_i − d(1)_i) / ε_i.                                    (4)
  ```

  Rationale: a raw margin (3) is measured in x-space and is blind to the
  channel — a noise-amplified dimension (ill-conditioned column of `H`) looks
  as trustworthy as a well-conditioned one. Normalizing by `ε_i` (the diagonal
  of the estimator's error covariance) restores the SQRD-style ordering, so
  the Potts search budget is spent on the dimensions that are actually
  uncertain.

The `f` dimensions with the **smallest** `ρ_i` become the free set `F`;
the remainder `X = {1..n} \ F` is fixed and enumerated in Stage 2.

## 3. Stage 2 — branch enumeration

Each *branch* is a full assignment of the fixed dimensions
`x_X ∈ L^{|X|}` together with the exact lower bound

```
LB(x_X) = min_{x_F ∈ R^f} ‖y − H_F x_F − H_X x_X‖²                  (5)
        = ‖P⊥_{H_F} (y − H_X x_X)‖²,
```

i.e. the residual after unconstrained (relaxed) elimination of the free
columns. Since the true free values are constrained to `L^f ⊂ R^f`, (5) is a
valid lower bound on any completion of the branch, which Stage 3 uses for
pruning.

### 3.1 `branch_strategy = :beam` (legacy)

Breadth-first beam over the fixed dimensions in index order, keeping only the
`fixed_candidates_per_dim` alphabet levels nearest to the soft estimate and at
most `4·max_branches` partial assignments per depth, ranked by the proxy
metric `Σ_i (ℓ_{s_i} − x̂_i)²` (channel-blind). Exact bounds (5) are computed
once per surviving leaf. With `fixed_candidates_per_dim = 1` this degenerates
to the single quantized ZF/MMSE point.

### 3.2 `branch_strategy = :sphere` (depth-first sphere decoder)

Order the columns as `perm = [F_sorted ; X_asc]`, where the fixed dimensions
are sorted by **ascending** reliability so the most reliable ones are assigned
first at the tree root. QR-factorize the permuted channel:

```
H[:, perm] = Q R,   q = Qᵀ y,   c₀ = ‖y‖² − ‖q_{1..n}‖²  (tail energy).
```

Because `R` is upper triangular, assigning levels from column `n` down to
column `f+1` accumulates the **exact partial distance**

```
D_i = D_{i+1} + ( q_i − Σ_{j>i} R_{ij} x_j − R_{ii} x_i )²,   D_{n+1} = 0,  (6)
```

and at the leaf level `i = f+1`,

```
LB = D_{f+1} + c₀                                                  (7)
```

equals the projection bound (5) exactly (rows `1..f` are zeroable by the free
columns). Pseudocode:

```
SPHERE-ENUMERATE(R, q, L, f, r0, max_branches, max_nodes):
  leaves ← ∅ ;  bound ← r0² ;  i ← n
  compute residual ri = q_i and Schnorr–Euchner order of L around ri/R_ii
  loop:
    s ← next untried level at depth i (SE order: increasing |ri − R_ii ℓ_s|)
    if none left: i ← i+1; if i > n stop; continue        # backtrack
    D ← D_parent + (ri − R_ii ℓ_s)²
    if D + c₀ ≥ min(bound, worst(leaves) if |leaves| = max_branches):
        i ← i+1; continue        # SE order ⇒ all remaining siblings prune too
    assign x_i ← ℓ_s
    if i = f+1:                                            # leaf
        push (D + c₀, x_{f+1..n}) into leaves (keep best max_branches)
        # Babai completion: back-substitute rows f..1, quantize each to L
        d_full ← D + Σ_{row=f..1} (q_row − Σ_{j>row} R_row,j x_j − R_row,row x̂_row)² + c₀
        bound ← min(bound, d_full)                         # radius shrinking
    else:
        i ← i−1; compute residual and SE order at new depth # descend
    abort when visited nodes exceed max_nodes (keep collected leaves)
```

Properties:

* **All `k` levels** are considered at every depth (no candidate truncation).
* Pruning uses the **exact metric** (6) against a radius that starts at (2)
  and **shrinks** via Babai completion of the free dimensions at every leaf.
* The bounded leaf set returns the **true best `max_branches` leaves** inside
  the radius — a beam with an exact metric *and* backtracking.
* With `f = 0` the leaves are complete lattice points, so the best leaf **is
  the exact ML solution (1)** whenever the node budget is not exhausted
  (verified against brute-force enumeration in the test suite).
* Requires `m ≥ n`; overloaded systems fall back to `:beam` with a warning.

## 4. Stage 3 — Potts subproblem per branch

For a branch with fixed part `x_X`, the free-dimension objective is the
quadratic Potts energy

```
E(x_F) = x_Fᵀ G x_F + hᵀ x_F + c,
G = H_FᵀH_F,   h = −2 H_Fᵀ(y − H_X x_X),   c = ‖y − H_X x_X‖².      (8)
```

Every outer trial and inner restart initializes each free dimension
**uniformly at random over the full alphabet** `L`, then runs the noisy
adjacent-transition dynamics (`:batch`, `:singleflip`, `:dau`, `:batchdau`)
with the annealed noise schedule. Across branches, a trial keeps a running
incumbent and skips any branch with `LB ≥` its current radius; the exact
sphere bounds (7) make this pruning tight.

## 5. Benchmarked behavior (July 2026 A/B, paired seeds, tuned HPs)

* `:whitened` vs `:margin` (beam branches): overall BER −4–5 %, growing to
  **−49 % BER / −62 % FER** at high SNR and larger `free_dims`; zero runtime
  cost. The gain concentrates exactly where errors are channel-induced.
* `:sphere` vs `:beam` (whitened ranking): overall **BER −70 %**
  (8×8 64-QAM) and **−60 %** (16×16 16-QAM); with `free_dims = 0` frame
  errors vanish entirely at mid/high SNR (exact ML). Enumeration cost is
  ≤ 18 ms/instance in the worst measured case (16×16 at low SNR) and
  ≤ 0.3 ms typically; the 2M node budget was hit on ~0.1 % of low-SNR
  instances.
* Under `:sphere` the `free_dims` trend inverts: smaller free sets are
  better, with `free_dims = 0` dominating at the measured operating points.
  The Potts stage matters when the enumeration is budget-limited (very low
  SNR, very large trees) or when `m < n` forces the beam fallback.

## 6. Defaults and compatibility

Defaults (`reliability_mode = :margin`, `branch_strategy = :beam`) reproduce
the pre-July-2026 behavior bit-for-bit for fixed seeds. Both options are
plumbed through `solve_mimo_potts`, the batched GPU path, and
`curunanmimoinstance` (`reliabilityMode` / `branchStrategy` /
`sphereMaxNodes`); branch enumeration always runs on the CPU, so the CUDA
backend is unaffected. Result files record both options in their metadata.
