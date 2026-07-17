# MIMOPotts.jl

Standalone Julia package for discrete Potts-machine MIMO detection with
sphere-decoder branch enumeration.

The detector solves the maximum-likelihood problem
`argmin_x ||y - Hx||^2` over the PAM lattice by splitting the real-valued
dimensions into a *fixed* set (enumerated as branch-and-bound tree leaves) and
a *free* set (optimized per branch by a stochastic Potts machine). See
[docs/detection_algorithm.md](docs/detection_algorithm.md) for the model,
equations, and pseudocode of every stage.

## Usage

```julia
using MIMOPotts

result = solve_mimo_potts("path/to/mimo_instance.npz";
    snr_index = 1,
    instance_index = 1,
    trials = 16,
    free_dims = 4,
    optimizer = :batch,
    preprocess = :qr,
    cacheCouplings = true,
    reliability_mode = :whitened,   # :margin (default) or :whitened
    branch_strategy = :sphere,      # :beam (default) or :sphere
)
```

Key options:

| Option | Values (default first) | Effect |
| --- | --- | --- |
| `free_dims` | `4` | Number of dimensions handed to the Potts machine; the rest are enumerated. With `branch_strategy = :sphere`, `free_dims = 0` is exact ML detection (subject to `sphere_max_nodes`). |
| `reliability_mode` | `:margin`, `:whitened` | How dimensions are ranked when choosing the free set. `:whitened` divides the decision margin by the per-dimension post-equalization noise variance (SQRD-style) and is strictly better in benchmarks. |
| `branch_strategy` | `:beam`, `:sphere` | Fixed-dimension enumeration. `:beam` is the legacy candidate beam; `:sphere` is a depth-first Schnorr–Euchner sphere decoder with exact partial distances and radius pruning. |
| `sphere_max_nodes` | `2_000_000` | Node budget for the sphere search (guards low-SNR worst cases). |
| `max_branches` | `256` | Maximum leaves kept as Potts branches. |
| `backend` | `:cpu`, `:cuda` | Potts subproblems run batched on GPU with `:cuda` (load CUDA.jl first). |

For sweeps, use `curunanmimoinstance` with compact output (camel-case kwargs
`reliabilityMode`, `branchStrategy`, `sphereMaxNodes`):

```julia
out = curunanmimoinstance("path/to/mimo_instance.npz";
    resultFormat = :compact,
    jld2file = "results/run1.jld2",
    reliabilityMode = :whitened,
    branchStrategy = :sphere,
)

out.runs
out.trials
```

Saved files can be loaded with:

```julia
out = load_mimo_potts_results("results/run1.jld2")
```
