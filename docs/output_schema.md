# MIMO Potts Output Schema

`curunanmimoinstance` supports two output formats:

```julia
curunanmimoinstance(path; resultFormat = :flat)
curunanmimoinstance(path; resultFormat = :compact)
```

The default `:flat` format returns one `DataFrame` with one row per outer trial. The `:compact` format returns a named tuple:

```julia
out = curunanmimoinstance(path; resultFormat = :compact)
out.runs
out.trials
```

`out.runs` stores fields shared by all trials in one run configuration. `out.trials` stores per-trial metrics and links back to `out.runs` through `runId`.

## `out.runs`

| Column | Meaning |
| --- | --- |
| `runId` | Shared run configuration id. Join key into `out.trials`. |
| `source` | Path to the MIMO `.npz` instance file. |
| `snr_index` | SNR index selected from the `.npz`. |
| `instance_index` | Instance/sample index selected from the `.npz`. |
| `ebnodb` | Eb/N0 value for this `snr_index`. |
| `modulation` | Detected or provided modulation, for example `QPSK` or `16QAM`. |
| `optimizer` | Potts update rule, for example `batch`, `singleflip`, `dau`, or `batchdau`. |
| `numCycles` | Base number of Potts cycles. |
| `noiseRatio` | Multiplier used to set annealing noise scale from the gradient bound. |
| `cyclesScaler` | Multiplier applied to `numCycles`; actual steps are approximately `ceil(cyclesScaler * numCycles)` per visited subproblem. |
| `freeDims` | Number of real-valued dimensions left free in each sub-Potts machine. |
| `initialRadius` | Starting sphere radius, currently the better distance between quantized ZF and MMSE. |
| `zfDistance` | Distance of the quantized ZF solution: `||y - Hx_zf||^2`. |
| `mmseDistance` | Distance of the quantized MMSE solution: `||y - Hx_mmse||^2`. |
| `zfBer` | BER of quantized ZF against `x_true`. |
| `mmseBer` | BER of quantized MMSE against `x_true`. |
| `branchesGenerated` | Number of sub-Potts branches built for this configuration. |
| `preprocess` | Geometry preprocessing used, currently `normal` or `qr`. |
| `cacheCouplings` | Whether shared coupling terms were cached across branches. |

## `out.trials`

| Column | Meaning |
| --- | --- |
| `runId` | Join key back to `out.runs`. |
| `outerTrial` | Outer independent trial number. |
| `trials` | Number of internal trials inside this solve. Currently `1` because each outer trial is a separate solve. |
| `finalRadius` | Final sphere radius after this trial. Same as the best distance found in that trial. |
| `bestDistance` | Lowest original MIMO distance found in this trial: `||y - Hx||^2`. |
| `stepFoundBest` | Global step count within this trial where the best candidate was first found. |
| `trialFoundBest` | Internal trial index where best was found. Currently usually `1`. |
| `branchFoundBest` | Branch rank where the best candidate was found. |
| `totalSteps` | Total Potts update steps executed in this outer trial. |
| `totalGradientEvals` | Total gradient evaluations. Currently tracks `totalSteps`. |
| `branchesVisited` | Branches actually solved by Potts. |
| `branchesPruned` | Branches skipped because their lower bound exceeded the current radius. |
| `radiusUpdates` | Number of times the sphere radius was improved. |
| `ber` | Bit error rate of the best found solution. |
| `ser` | Real-dimension symbol/state error rate of the best found solution. |
| `fer` | Frame error flag/rate for this instance: `0.0` if no bit errors, `1.0` otherwise. |
| `bestStates` | Best discrete Potts state indices as a compact string. |
| `bestValues` | Best detected real-valued vector reconstructed from `bestStates`. |
| `stepTime` | Wall-clock solve time for this outer trial, in seconds. |

## Flat Format

The default flat output combines the shared and trial fields into one `DataFrame`. It is convenient for quick analysis, but repeats shared fields once per outer trial. Use `resultFormat = :compact` for larger sweeps to avoid that repetition.

## Saving and Loading

Use JLD2 through the MIMO Potts helpers:

```julia
out = curunanmimoinstance(
    path;
    resultFormat = :compact,
    jld2file = "results/mimo/run1.jld2",
    saveMetadata = (; note = "first sweep"),
)
```

This writes a compact file with:

| JLD2 key | Meaning |
| --- | --- |
| `schemaVersion` | Integer schema version for these result files. |
| `resultFormat` | Either `"compact"` or `"flat"`. |
| `metadata` | Metadata dictionary, including `createdAtUnix`, `creator`, and anything passed through `saveMetadata`. |
| `runDF` | Shared run table, present for compact files. |
| `trialDF` | Per-trial table, present for compact files. |
| `anIncDF` | Flat result table, present for flat files. |

You can also save an already-computed result:

```julia
out = curunanmimoinstance(path; resultFormat = :compact)
save_mimo_potts_results("results/mimo/run1.jld2", out)
```

Load compact results with:

```julia
out = load_mimo_potts_results("results/mimo/run1.jld2")
out.runs
out.trials
```

Load with metadata when needed:

```julia
loaded = load_mimo_potts_results("results/mimo/run1.jld2"; includeMetadata = true)
loaded.format
loaded.result
loaded.metadata
loaded.schemaVersion
```

For flat files, `load_mimo_potts_results` returns the saved `DataFrame` by default.
