# MIMOPotts.jl

Standalone Julia package for discrete Potts-machine MIMO detection.

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
)
```

For sweeps, use `curunanmimoinstance` with compact output:

```julia
out = curunanmimoinstance("path/to/mimo_instance.npz";
    resultFormat = :compact,
    jld2file = "results/run1.jld2",
)

out.runs
out.trials
```

Saved files can be loaded with:

```julia
out = load_mimo_potts_results("results/run1.jld2")
```
