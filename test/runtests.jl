using MIMOPotts
using DataFrames
using LinearAlgebra
using NPZ
using Random
using Test

@testset "PAM levels" begin
    @test MIMOPotts.pam_levels("QPSK") == [-inv(sqrt(2.0)), inv(sqrt(2.0))]
    @test length(MIMOPotts.pam_levels("16QAM")) == 4
end

@testset "JLD2 result save/load" begin
    result = (
        runs = DataFrame(runId = [1], source = ["synthetic"], noiseRatio = [0.5]),
        trials = DataFrame(runId = [1], outerTrial = [1], bestDistance = [1.25]),
    )
    path = tempname() * ".jld2"
    saved = save_mimo_potts_results(path, result; metadata = (; note = "unit"))
    loaded = load_mimo_potts_results(saved; includeMetadata = true)
    @test loaded.format == :compact
    @test loaded.schemaVersion == 1
    @test loaded.metadata["note"] == "unit"
    @test loaded.result.runs == result.runs
    @test loaded.result.trials == result.trials
end

@testset "backend plumbing" begin
    @test MIMOPotts._mimo_backend_available(Val(:cpu))
    @test MIMOPotts._check_mimo_backend(:cpu) isa Val{:cpu}
end

function _write_unit_mimo_instance()
    dir = mktempdir()
    path = joinpath(dir, "unit_qpsk.npz")
    levels = MIMOPotts.pam_levels("QPSK")
    H = Matrix{Float64}(I, 2, 2)
    x = [levels[1], levels[2]]
    y = copy(x)
    NPZ.npzwrite(path, Dict(
        "nt" => 1,
        "nr" => 1,
        "ebnodb" => [10.0],
        "no" => [0.01],
        "Haug" => reshape(H, 1, 1, 2, 2),
        "yaug" => reshape(y, 1, 1, 2),
        "xaug" => reshape(x, 1, 1, 2),
    ))
    return path
end

@testset "CPU optimized MIMO solve" begin
    path = _write_unit_mimo_instance()
    kwargs = (;
        snr_index = 1,
        instance_index = 1,
        free_dims = 2,
        fixed_candidates_per_dim = 1,
        max_branches = 1,
        trials = 8,
        num_cycles = 8,
        cycles_scaler = 1.0,
        noise_ratio = 0.5,
        optimizer = :batch,
        cacheCouplings = true,
        preprocess = :normal,
        seed = 42,
        backend = :cpu,
    )

    serial_a = solve_mimo_potts(path; kwargs..., cpuThreads = false)
    serial_b = solve_mimo_potts(path; kwargs..., cpuThreads = false)
    threaded = solve_mimo_potts(path; kwargs..., cpuThreads = true)

    @test isfinite(serial_a.best_distance)
    @test 0.0 <= serial_a.ber <= 1.0
    @test serial_a.best_distance == serial_b.best_distance
    @test serial_a.best_states == serial_b.best_states
    @test serial_a.best_distance == threaded.best_distance
    @test serial_a.best_states == threaded.best_states
    @test serial_a.total_steps == threaded.total_steps
end

@testset "CPU subproblem energy recomputes" begin
    path = _write_unit_mimo_instance()
    inst = MIMOPotts.load_mimo_potts_instance(path)
    prep = MIMOPotts.zf_mmse_preprocess(inst)
    geometry = MIMOPotts._preprocessed_geometry(inst, :normal)
    branches = MIMOPotts.build_mimo_potts_branches(inst, prep;
        free_dims = 2,
        fixed_candidates_per_dim = 1,
        max_branches = 1,
        H = geometry.H,
        y = geometry.y,
        const_offset = geometry.const_offset,
    )
    branch = only(branches)
    cache = MIMOPotts._make_coupling_cache(geometry.H, geometry.y, branch.free_indices, branch.fixed_indices, geometry.const_offset, true)
    sub = MIMOPotts._run_potts_subproblem(inst, branch, prep;
        H = geometry.H,
        y = geometry.y,
        const_offset = geometry.const_offset,
        coupling_cache = cache,
        optimizer = :batch,
        num_cycles = 8,
        cycles_scaler = 1.0,
        noise_ratio = 0.5,
        noise_stepper = :linear,
        batch_rate = 0.5,
        eoffset = 0.0,
        rng = MersenneTwister(7),
    )
    G, h, const_term = MIMOPotts._subproblem_terms(geometry.H, geometry.y, branch, geometry.const_offset, cache)
    values = MIMOPotts.states_to_values(sub.best_free_states, inst.levels)
    exact = dot(values, G * values) + dot(h, values) + const_term
    @test isapprox(sub.best_distance, exact; rtol = 1e-10, atol = 1e-10)
end
