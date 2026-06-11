
export MIMOPottsInstance, MIMOPottsBranch, MIMOPottsSolveResult
export pam_levels, load_mimo_potts_instance, solve_mimo_potts, curunanmimoinstance
export save_mimo_potts_results, load_mimo_potts_results

using LinearAlgebra
using Random
using DataFrames
using JLD2
const _MIMO_POTTS_NPZ_PKGID = Base.PkgId(Base.UUID("15e1cf62-19b3-5cfa-8e77-841668bca605"), "NPZ")
const MIMO_POTTS_RESULTS_SCHEMA_VERSION = 1

function _npzread(args...)
    mod = try
        Base.require(_MIMO_POTTS_NPZ_PKGID)
    catch err
        error("NPZ.jl is required for MIMO .npz loading. Run `import Pkg; Pkg.add(\"NPZ\")` in the active Julia environment, then `Pkg.resolve()`.")
    end
    return Base.invokelatest(getfield(mod, :npzread), args...)
end

Base.@kwdef struct MIMOPottsInstance
    source::String
    snr_index::Int
    instance_index::Int
    ebnodb::Float64
    noise_variance::Float64
    modulation::String
    nt::Int
    nr::Int
    H::Matrix{Float64}
    y::Vector{Float64}
    x_true::Vector{Float64}
    levels::Vector{Float64}
end

Base.@kwdef struct MIMOPottsBranch
    rank::Int
    free_indices::Vector{Int}
    fixed_indices::Vector{Int}
    fixed_states::Vector{Int}
    fixed_values::Vector{Float64}
    lower_bound::Float64
    proxy_distance::Float64
end

Base.@kwdef struct MIMOPottsSolveResult
    source::String
    snr_index::Int
    instance_index::Int
    ebnodb::Float64
    modulation::String
    optimizer::Symbol
    trials::Int
    num_cycles::Int
    noise_ratio::Float64
    cycles_scaler::Float64
    free_dims::Int
    initial_radius::Float64
    final_radius::Float64
    best_distance::Float64
    zf_distance::Float64
    mmse_distance::Float64
    step_found_best::Int
    trial_found_best::Int
    branch_found_best::Int
    total_steps::Int
    total_gradient_evals::Int
    branches_generated::Int
    branches_visited::Int
    branches_pruned::Int
    radius_updates::Int
    ber::Float64
    ser::Float64
    fer::Float64
    zf_ber::Float64
    mmse_ber::Float64
    best_states::Vector{Int}
    best_values::Vector{Float64}
    step_time::Float64
    preprocess::Symbol
    cache_couplings::Bool
end

function _as_scalar_string(x)
    if x isa AbstractArray
        return string(only(x))
    end
    return string(x)
end

function infer_mimo_modulation(path::AbstractString)
    name = lowercase(basename(path))
    dir = lowercase(basename(dirname(path)))
    text = name * "_" * dir
    m = match(r"(qpsk|[0-9]+qam)", text)
    if m !== nothing
        s = uppercase(m.captures[1])
        return replace(s, "QAM" => "QAM")
    end
    m_old = match(r"m([0-9]+)", text)
    if m_old !== nothing
        val = parse(Int, m_old.captures[1])
        return val == 4 ? "QPSK" : "$(val)QAM"
    end
    error("Could not infer MIMO modulation from path $(path); pass modulation=\"QPSK\" or \"16QAM\" etc.")
end

function pam_levels(modulation::AbstractString)
    mod = uppercase(String(modulation))
    if mod == "QPSK"
        alpha = inv(sqrt(2.0))
        return [-alpha, alpha]
    elseif endswith(mod, "QAM")
        M = parse(Int, replace(mod, "QAM" => ""))
        bits_per_symbol = Int(round(log2(M)))
        iseven(bits_per_symbol) || error("Expected square QAM, got $(mod)")
        bits_per_real_dim = bits_per_symbol >>> 1
        nlevels = 1 << bits_per_real_dim
        gamma = sqrt(3.0 / (2.0 * (M - 1.0)))
        return [gamma * (2.0 * u - (nlevels - 1.0)) for u in 0:(nlevels - 1)]
    else
        error("Unsupported modulation $(modulation)")
    end
end

bits_per_real_dim(levels::AbstractVector) = Int(round(log2(length(levels))))

function nearest_level_index(value::Real, levels::AbstractVector{<:Real})
    best_i = 1
    best_d = abs2(float(value) - float(levels[1]))
    @inbounds for i in 2:length(levels)
        d = abs2(float(value) - float(levels[i]))
        if d < best_d
            best_i = i
            best_d = d
        end
    end
    return best_i
end

function quantize_to_states(values::AbstractVector{<:Real}, levels::AbstractVector{<:Real})
    return [nearest_level_index(v, levels) for v in values]
end

states_to_values(states::AbstractVector{<:Integer}, levels::AbstractVector{<:Real}) = [Float64(levels[s]) for s in states]

function mimo_distance(H::AbstractMatrix, y::AbstractVector, x::AbstractVector)
    r = y - H * x
    return dot(r, r)
end

function _state_bit_errors(a::Int, b::Int, nbits::Int)
    ua = a - 1
    ub = b - 1
    errs = 0
    for k in 0:(nbits - 1)
        errs += ((ua >>> k) & 1) != ((ub >>> k) & 1)
    end
    return errs
end

function detection_error_rates(states::AbstractVector{<:Integer}, true_x::AbstractVector, levels::AbstractVector)
    true_states = quantize_to_states(true_x, levels)
    nbits = bits_per_real_dim(levels)
    bit_errors = 0
    sym_errors = 0
    @inbounds for i in eachindex(states)
        bit_errors += _state_bit_errors(states[i], true_states[i], nbits)
        sym_errors += states[i] != true_states[i]
    end
    total_bits = length(states) * nbits
    ber = total_bits == 0 ? 0.0 : bit_errors / total_bits
    ser = isempty(states) ? 0.0 : sym_errors / length(states)
    fer = bit_errors == 0 ? 0.0 : 1.0
    return ber, ser, fer
end

function load_mimo_potts_instance(path::AbstractString; snr_index::Int=1, instance_index::Int=1, modulation=nothing)
    data = _npzread(path, ["nt", "nr", "ebnodb", "no", "Haug", "yaug", "xaug"])
    mod = isnothing(modulation) ? infer_mimo_modulation(path) : String(modulation)
    levels = pam_levels(mod)
    H = Matrix{Float64}(data["Haug"][snr_index, instance_index, :, :])
    y = Vector{Float64}(data["yaug"][snr_index, instance_index, :])
    x_true = Vector{Float64}(data["xaug"][snr_index, instance_index, :])
    return MIMOPottsInstance(
        source = String(path),
        snr_index = snr_index,
        instance_index = instance_index,
        ebnodb = Float64(data["ebnodb"][snr_index]),
        noise_variance = Float64(data["no"][snr_index]),
        modulation = uppercase(mod),
        nt = Int(data["nt"]),
        nr = Int(data["nr"]),
        H = H,
        y = y,
        x_true = x_true,
        levels = levels,
    )
end

function zf_mmse_preprocess(inst::MIMOPottsInstance)
    H, y, levels = inst.H, inst.y, inst.levels
    zf_est = H \ y
    mmse_est = (transpose(H) * H + inst.noise_variance * I(size(H, 2))) \ (transpose(H) * y)

    zf_states = quantize_to_states(zf_est, levels)
    mmse_states = quantize_to_states(mmse_est, levels)
    zf_values = states_to_values(zf_states, levels)
    mmse_values = states_to_values(mmse_states, levels)
    zf_distance = mimo_distance(H, y, zf_values)
    mmse_distance = mimo_distance(H, y, mmse_values)

    if zf_distance <= mmse_distance
        base_est = Vector{Float64}(zf_est)
        base_states = zf_states
        base_values = zf_values
        base_distance = zf_distance
        base_decoder = :zf
    else
        base_est = Vector{Float64}(mmse_est)
        base_states = mmse_states
        base_values = mmse_values
        base_distance = mmse_distance
        base_decoder = :mmse
    end

    reliability = similar(base_est, Float64)
    @inbounds for i in eachindex(base_est)
        ds = sort(abs2.(levels .- base_est[i]))
        reliability[i] = length(ds) == 1 ? Inf : ds[2] - ds[1]
    end

    zf_ber = detection_error_rates(zf_states, inst.x_true, levels)[1]
    mmse_ber = detection_error_rates(mmse_states, inst.x_true, levels)[1]

    return (; zf_est, mmse_est, zf_states, mmse_states, zf_values, mmse_values,
        zf_distance, mmse_distance, zf_ber, mmse_ber, base_decoder, base_est,
        base_states, base_values, base_distance, reliability)
end

function _preprocessed_geometry(inst::MIMOPottsInstance, preprocess)
    prep = Symbol(preprocess)
    if prep === :normal
        return (; H = inst.H, y = inst.y, const_offset = 0.0, preprocess = prep)
    elseif prep === :qr
        fact = qr(inst.H)
        Hq = Matrix{Float64}(fact.R)
        qty = Vector{Float64}(fact.Q' * inst.y)
        active_rows = size(Hq, 1)
        if length(qty) > active_rows
            tail = @view qty[(active_rows + 1):end]
            const_offset = dot(tail, tail)
            yq = qty[1:active_rows]
        else
            const_offset = 0.0
            yq = qty
        end
        return (; H = Hq, y = yq, const_offset = const_offset, preprocess = prep)
    else
        error("Unsupported MIMO Potts preprocess $(preprocess). Use :normal or :qr.")
    end
end

function _make_coupling_cache(H::AbstractMatrix, y::AbstractVector, free_indices::Vector{Int}, fixed_indices::Vector{Int}, const_offset::Real, cacheCouplings::Bool)
    if !cacheCouplings || isempty(free_indices)
        return nothing
    end
    Hfree = H[:, free_indices]
    Hfixed = H[:, fixed_indices]
    return (;
        Hfree = Hfree,
        Hfixed = Hfixed,
        G = transpose(Hfree) * Hfree,
        Gfc = transpose(Hfree) * Hfixed,
        h0 = -2.0 .* (transpose(Hfree) * y),
        y = y,
        const_offset = Float64(const_offset),
    )
end

function _branch_lower_bound(H::AbstractMatrix, y::AbstractVector, free_indices::Vector{Int}, fixed_indices::Vector{Int}, fixed_values::Vector{Float64}; const_offset::Real=0.0)
    residual = isempty(fixed_indices) ? copy(y) : y - H[:, fixed_indices] * fixed_values
    if isempty(free_indices)
        return dot(residual, residual) + Float64(const_offset)
    end
    Hfree = H[:, free_indices]
    projected = Hfree * (Hfree \ residual)
    r = residual - projected
    return dot(r, r) + Float64(const_offset)
end

function build_mimo_potts_branches(inst::MIMOPottsInstance, prep; free_dims::Int, fixed_candidates_per_dim::Int=2, max_branches::Int=256, H=inst.H, y=inst.y, const_offset::Real=0.0)
    n = length(prep.base_states)
    free_count = clamp(free_dims, 0, n)
    order = sortperm(prep.reliability; rev=false)
    free_indices = sort(order[1:free_count])
    fixed_indices = sort(setdiff(collect(1:n), free_indices))
    levels = inst.levels

    items = [(states = Int[], proxy = 0.0)]
    beam_width = max(max_branches * 4, max_branches)
    for dim in fixed_indices
        candidates = sortperm(abs2.(levels .- prep.base_est[dim]))[1:min(fixed_candidates_per_dim, length(levels))]
        new_items = typeof(items)()
        for item in items, state in candidates
            proxy = item.proxy + abs2(levels[state] - prep.base_est[dim])
            push!(new_items, (states = [item.states; state], proxy = proxy))
        end
        sort!(new_items, by = x -> x.proxy)
        if length(new_items) > beam_width
            resize!(new_items, beam_width)
        end
        items = new_items
    end

    branches = MIMOPottsBranch[]
    for item in items
        fixed_values = states_to_values(item.states, levels)
        lb = _branch_lower_bound(H, y, free_indices, fixed_indices, fixed_values; const_offset = const_offset)
        push!(branches, MIMOPottsBranch(
            rank = 0,
            free_indices = copy(free_indices),
            fixed_indices = copy(fixed_indices),
            fixed_states = copy(item.states),
            fixed_values = fixed_values,
            lower_bound = lb,
            proxy_distance = item.proxy,
        ))
    end
    sort!(branches, by = b -> (b.lower_bound, b.proxy_distance))
    if length(branches) > max_branches
        resize!(branches, max_branches)
    end
    return [MIMOPottsBranch(
        rank = i,
        free_indices = branches[i].free_indices,
        fixed_indices = branches[i].fixed_indices,
        fixed_states = branches[i].fixed_states,
        fixed_values = branches[i].fixed_values,
        lower_bound = branches[i].lower_bound,
        proxy_distance = branches[i].proxy_distance,
    ) for i in eachindex(branches)]
end

function _noise_scale(noise_stepper::Symbol, noise_start::Real, num_steps::Real, step::Real)
    if noise_stepper === :linear
        return Float64(noise_start) * (1.0 - Float64(step) / Float64(num_steps))
    elseif noise_stepper === :exponential
        return Float64(noise_start) * exp(-10.0 * Float64(step) / Float64(num_steps))
    elseif noise_stepper === :exponential1
        return Float64(noise_start) * exp(-Float64(step) / Float64(num_steps))
    else
        error("Unsupported MIMO Potts noise_stepper $(noise_stepper)")
    end
end

function _gradient_bound(G::AbstractMatrix, h::AbstractVector, levels::AbstractVector)
    max_level = maximum(abs.(levels))
    bound = 0.0
    for i in axes(G, 1)
        row_bound = sum(abs.(2.0 .* G[i, :])) * max_level + abs(h[i])
        bound = max(bound, row_bound)
    end
    return bound
end

function _energy_batch(G::AbstractMatrix, h::AbstractVector, const_term::Real, values::AbstractMatrix)
    return vec(sum(values .* (G * values); dims=1)) .+ vec(sum(h .* values; dims=1)) .+ Float64(const_term)
end

function _subproblem_terms(H::AbstractMatrix, y::AbstractVector, branch::MIMOPottsBranch, const_offset::Real, coupling_cache)
    if coupling_cache === nothing
        residual = y - H[:, branch.fixed_indices] * branch.fixed_values
        Hfree = H[:, branch.free_indices]
        G = transpose(Hfree) * Hfree
        h = -2.0 .* (transpose(Hfree) * residual)
        const_term = dot(residual, residual) + Float64(const_offset)
        return G, h, const_term
    end
    residual = coupling_cache.y - coupling_cache.Hfixed * branch.fixed_values
    G = coupling_cache.G
    h = coupling_cache.h0 + 2.0 .* (coupling_cache.Gfc * branch.fixed_values)
    const_term = dot(residual, residual) + coupling_cache.const_offset
    return G, h, const_term
end

function _run_potts_subproblem(inst::MIMOPottsInstance, branch::MIMOPottsBranch, prep;
    H::AbstractMatrix,
    y::AbstractVector,
    const_offset::Real,
    coupling_cache,
    optimizer::Symbol,
    num_cycles::Int,
    cycles_scaler::Real,
    noise_ratio::Real,
    noise_stepper::Symbol,
    batch_rate::Real,
    eoffset::Real,
    rng::AbstractRNG,
)
    free_indices = branch.free_indices
    f = length(free_indices)
    levels = inst.levels
    k = length(levels)
    steps = max(1, ceil(Int, cycles_scaler * num_cycles))

    if f == 0
        full_states = copy(prep.base_states)
        full_states[branch.fixed_indices] .= branch.fixed_states
        full_values = states_to_values(full_states, levels)
        dist = mimo_distance(H, y, full_values) + Float64(const_offset)
        return (; best_distance = dist, best_free_states = Int[], step_found = 0,
            total_steps = 0, total_gradient_evals = 0)
    end

    G, h, const_term = _subproblem_terms(H, y, branch, const_offset, coupling_cache)
    noise_start = Float64(noise_ratio) * _gradient_bound(G, h, levels)

    states = rand(rng, 1:k, f)
    values = levels[states]
    best_distance = values' * G * values + dot(h, values) + const_term
    best_states = copy(states)
    step_found = 0
    dau_offset = 0.0

    for step in 1:steps
        scale = _noise_scale(noise_stepper, noise_start, steps, step) + dau_offset
        grad = 2.0 .* (G * values) .+ h
        noise = scale .* (2.0 .* rand(rng, f) .- 1.0)
        proposal_direction = ifelse.(grad .< noise, 1, -1)
        proposed_states = clamp.(states .+ proposal_direction, 1, k)
        changed = proposed_states .!= states

        if optimizer === :batch || optimizer === :batchdau
            mask = rand(rng, f) .< Float64(batch_rate)
            apply = changed .& mask
            if optimizer === :batchdau
                dau_offset = any(apply) ? 0.0 : dau_offset + Float64(eoffset)
            end
            states[apply] .= proposed_states[apply]
        elseif optimizer === :singleflip || optimizer === :dau
            candidates = findall(changed)
            if isempty(candidates)
                if optimizer === :dau
                    dau_offset += Float64(eoffset)
                end
            else
                dau_offset = 0.0
                chosen = rand(rng, candidates)
                states[chosen] = proposed_states[chosen]
            end
        else
            error("Unsupported MIMO Potts optimizer $(optimizer). Use :batch, :singleflip, :dau, or :batchdau.")
        end

        values = levels[states]
        distance = values' * G * values + dot(h, values) + const_term
        if distance < best_distance
            best_distance = distance
            best_states .= states
            step_found = step
        end
    end

    return (; best_distance = Float64(best_distance), best_free_states = Vector{Int}(best_states),
        step_found = step_found, total_steps = steps, total_gradient_evals = steps)
end

function _full_states(prep, branch::MIMOPottsBranch, free_states::Vector{Int})
    states = copy(prep.base_states)
    states[branch.fixed_indices] .= branch.fixed_states
    states[branch.free_indices] .= free_states
    return states
end

function solve_mimo_potts(path::AbstractString;
    snr_index::Int=1,
    instance_index::Int=1,
    modulation=nothing,
    free_dims::Int=4,
    fixed_candidates_per_dim::Int=2,
    max_branches::Int=256,
    trials::Int=128,
    num_cycles::Int=100,
    cycles_scaler::Real=1.0,
    noise_ratio::Real=1.0,
    noise_stepper::Symbol=:linear,
    optimizer::Symbol=:batch,
    batch_rate::Real=0.5,
    eoffset::Real=0.0,
    cacheCouplings::Bool=true,
    preprocess=:normal,
    seed=nothing,
)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    inst = load_mimo_potts_instance(path; snr_index, instance_index, modulation)
    prep = zf_mmse_preprocess(inst)
    geometry = _preprocessed_geometry(inst, preprocess)
    initial_radius = Float64(prep.base_distance)
    best_states = copy(prep.base_states)
    best_values = copy(prep.base_values)
    best_distance = initial_radius
    step_found_best = 0
    trial_found_best = 0
    branch_found_best = 0
    total_steps = 0
    total_gradient_evals = 0
    branches_visited = 0
    branches_pruned = 0
    radius_updates = 0

    branches = build_mimo_potts_branches(inst, prep;
        free_dims = free_dims,
        fixed_candidates_per_dim = fixed_candidates_per_dim,
        max_branches = max_branches,
        H = geometry.H,
        y = geometry.y,
        const_offset = geometry.const_offset,
    )
    coupling_cache = isempty(branches) ? nothing : _make_coupling_cache(geometry.H, geometry.y, branches[1].free_indices, branches[1].fixed_indices, geometry.const_offset, cacheCouplings)

    elapsed = @elapsed begin
        for trial in 1:trials
            current_radius = initial_radius
            trial_best_distance = initial_radius
            trial_best_states = copy(prep.base_states)
            trial_best_values = copy(prep.base_values)

            for branch in branches
                if branch.lower_bound >= current_radius
                    branches_pruned += 1
                    continue
                end
                branches_visited += 1
                step_offset = total_steps
                sub = _run_potts_subproblem(inst, branch, prep;
                    H = geometry.H,
                    y = geometry.y,
                    const_offset = geometry.const_offset,
                    coupling_cache = coupling_cache,
                    optimizer = optimizer,
                    num_cycles = num_cycles,
                    cycles_scaler = cycles_scaler,
                    noise_ratio = noise_ratio,
                    noise_stepper = noise_stepper,
                    batch_rate = batch_rate,
                    eoffset = eoffset,
                    rng = rng,
                )
                total_steps += sub.total_steps
                total_gradient_evals += sub.total_gradient_evals

                if sub.best_distance < trial_best_distance
                    candidate_states = _full_states(prep, branch, sub.best_free_states)
                    candidate_values = states_to_values(candidate_states, inst.levels)
                    candidate_distance = mimo_distance(inst.H, inst.y, candidate_values)
                    if candidate_distance < trial_best_distance
                        trial_best_states = candidate_states
                        trial_best_values = candidate_values
                        trial_best_distance = candidate_distance
                        current_radius = candidate_distance
                        radius_updates += 1

                        if candidate_distance < best_distance
                            best_states = candidate_states
                            best_values = candidate_values
                            best_distance = candidate_distance
                            step_found_best = step_offset + sub.step_found
                            trial_found_best = trial
                            branch_found_best = branch.rank
                        end
                    end
                end
            end

            if trial_best_distance < best_distance
                best_states = trial_best_states
                best_values = trial_best_values
                best_distance = trial_best_distance
                trial_found_best = trial
            end
        end
    end

    ber, ser, fer = detection_error_rates(best_states, inst.x_true, inst.levels)
    return MIMOPottsSolveResult(
        source = inst.source,
        snr_index = inst.snr_index,
        instance_index = inst.instance_index,
        ebnodb = inst.ebnodb,
        modulation = inst.modulation,
        optimizer = optimizer,
        trials = trials,
        num_cycles = num_cycles,
        noise_ratio = Float64(noise_ratio),
        cycles_scaler = Float64(cycles_scaler),
        free_dims = free_dims,
        initial_radius = Float64(prep.base_distance),
        final_radius = best_distance,
        best_distance = best_distance,
        zf_distance = Float64(prep.zf_distance),
        mmse_distance = Float64(prep.mmse_distance),
        step_found_best = step_found_best,
        trial_found_best = trial_found_best,
        branch_found_best = branch_found_best,
        total_steps = total_steps,
        total_gradient_evals = total_gradient_evals,
        branches_generated = length(branches),
        branches_visited = branches_visited,
        branches_pruned = branches_pruned,
        radius_updates = radius_updates,
        ber = ber,
        ser = ser,
        fer = fer,
        zf_ber = Float64(prep.zf_ber),
        mmse_ber = Float64(prep.mmse_ber),
        best_states = best_states,
        best_values = best_values,
        step_time = elapsed,
        preprocess = geometry.preprocess,
        cache_couplings = cacheCouplings,
    )
end

function _drop_keys(row, keys_to_drop::Tuple)
    names = Tuple(k for k in keys(row) if !(k in keys_to_drop))
    return NamedTuple{names}(row)
end

function _result_row(r::MIMOPottsSolveResult)
    shared = _drop_keys(_shared_result_row(r, 0), (:runId,))
    trial = _drop_keys(_trial_result_row(r, 0, 0), (:runId, :outerTrial))
    return merge(shared, trial)
end

function _shared_result_row(r::MIMOPottsSolveResult, run_id::Int)
    return (
        runId = run_id,
        source = r.source,
        snr_index = r.snr_index,
        instance_index = r.instance_index,
        ebnodb = r.ebnodb,
        modulation = r.modulation,
        optimizer = String(r.optimizer),
        numCycles = r.num_cycles,
        noiseRatio = r.noise_ratio,
        cyclesScaler = r.cycles_scaler,
        freeDims = r.free_dims,
        initialRadius = r.initial_radius,
        zfDistance = r.zf_distance,
        mmseDistance = r.mmse_distance,
        zfBer = r.zf_ber,
        mmseBer = r.mmse_ber,
        branchesGenerated = r.branches_generated,
        preprocess = String(r.preprocess),
        cacheCouplings = r.cache_couplings,
    )
end

function _trial_result_row(r::MIMOPottsSolveResult, run_id::Int, outer_trial::Int)
    return (
        runId = run_id,
        outerTrial = outer_trial,
        trials = r.trials,
        finalRadius = r.final_radius,
        bestDistance = r.best_distance,
        stepFoundBest = r.step_found_best,
        trialFoundBest = r.trial_found_best,
        branchFoundBest = r.branch_found_best,
        totalSteps = r.total_steps,
        totalGradientEvals = r.total_gradient_evals,
        branchesVisited = r.branches_visited,
        branchesPruned = r.branches_pruned,
        radiusUpdates = r.radius_updates,
        ber = r.ber,
        ser = r.ser,
        fer = r.fer,
        bestStates = join(r.best_states, ""),
        bestValues = copy(r.best_values),
        stepTime = r.step_time,
    )
end

function _hp_values(hps, key::Symbol, default)
    if hps isa AbstractDict && haskey(hps, key)
        v = hps[key]
        return v isa AbstractVector ? collect(v) : [v]
    end
    return [default]
end

function _ensure_parent_dir(path::AbstractString)
    parent = dirname(path)
    if !isempty(parent) && parent != "."
        mkpath(parent)
    end
    return path
end

function _metadata_dict(metadata)
    out = Dict{String, Any}(
        "schemaVersion" => MIMO_POTTS_RESULTS_SCHEMA_VERSION,
        "createdAtUnix" => time(),
        "creator" => "SATField.mimo_potts",
    )
    if metadata !== nothing
        for (k, v) in pairs(metadata)
            out[string(k)] = v
        end
    end
    return out
end

function _compact_mimo_result(result)
    return hasproperty(result, :runs) && hasproperty(result, :trials)
end

"""
    save_mimo_potts_results(path, result; metadata=nothing)

Save MIMO Potts benchmark results to a JLD2 file.

`result` can be either the flat `DataFrame` returned by `curunanmimoinstance`
with `resultFormat=:flat`, or the compact named tuple returned with
`resultFormat=:compact`.
"""
function save_mimo_potts_results(path::AbstractString, result; metadata=nothing)
    filepath = _ensure_parent_dir(String(path))
    meta = _metadata_dict(metadata)

    if result isa DataFrame
        jldsave(filepath;
            schemaVersion = MIMO_POTTS_RESULTS_SCHEMA_VERSION,
            resultFormat = "flat",
            metadata = meta,
            anIncDF = result,
        )
        return filepath
    elseif _compact_mimo_result(result)
        jldsave(filepath;
            schemaVersion = MIMO_POTTS_RESULTS_SCHEMA_VERSION,
            resultFormat = "compact",
            metadata = meta,
            runDF = result.runs,
            trialDF = result.trials,
        )
        return filepath
    else
        error("Unsupported MIMO Potts result object. Save either a flat DataFrame or a compact result with `.runs` and `.trials`.")
    end
end

"""
    load_mimo_potts_results(path; includeMetadata=false)

Load MIMO Potts benchmark results saved with `save_mimo_potts_results` or the
older `curunanmimoinstance(...; jld2file=...)` path.

By default this returns the original analysis shape: a flat `DataFrame` for flat
files, or `(runs=runDF, trials=trialDF)` for compact files. Set
`includeMetadata=true` to get `(format=..., result=..., metadata=..., schemaVersion=...)`.
"""
function load_mimo_potts_results(path::AbstractString; includeMetadata::Bool=false)
    data = load(String(path))
    format = if haskey(data, "resultFormat")
        Symbol(data["resultFormat"])
    elseif haskey(data, "runDF") && haskey(data, "trialDF")
        :compact
    elseif haskey(data, "anIncDF")
        :flat
    else
        error("Could not identify MIMO Potts result format in $(path).")
    end

    result = if format === :compact
        (runs = data["runDF"], trials = data["trialDF"])
    elseif format === :flat
        data["anIncDF"]
    else
        error("Unsupported MIMO Potts result format $(format) in $(path).")
    end

    if includeMetadata
        return (
            format = format,
            result = result,
            metadata = get(data, "metadata", Dict{String, Any}()),
            schemaVersion = get(data, "schemaVersion", nothing),
        )
    end
    return result
end

function curunanmimoinstance(instance_path::AbstractString;
    hps = Dict(:noiseRatio => [1.0], :cyclesScaler => [1.0], :freeDims => [4]),
    trials::Int = 128,
    numCycles::Int = 100,
    optimizer::Symbol = :batch,
    snr_indices = 1,
    instance_indices = 1,
    modulation = nothing,
    fixed_candidates_per_dim::Int = 2,
    max_branches::Int = 256,
    noise_stepper::Symbol = :linear,
    batch_rate::Real = 0.5,
    eoffset::Real = 0.0,
    cacheCouplings::Bool = true,
    preprocess = :normal,
    seed = nothing,
    jld2file = nothing,
    saveMetadata = nothing,
    showProgress::Bool = true,
    resultFormat = :flat,
)
    format = Symbol(resultFormat)
    if !(format in (:flat, :compact))
        error("Unsupported MIMO Potts resultFormat $(resultFormat). Use :flat or :compact.")
    end

    snrs = snr_indices isa Colon ? eachindex(_npzread(instance_path, ["ebnodb"])["ebnodb"]) : collect(snr_indices)
    instances = collect(instance_indices)
    noise_ratios = _hp_values(hps, :noiseRatio, 1.0)
    cycles_scalers = _hp_values(hps, :cyclesScaler, 1.0)
    free_dims_values = _hp_values(hps, :freeDims, _hp_values(hps, :free_dims, 4)[1])

    rows = NamedTuple[]
    run_rows = NamedTuple[]
    trial_rows = NamedTuple[]
    run_id = 0
    run_counter = 0

    for snr_index in snrs, instance_index in instances, noise_ratio in noise_ratios, cycles_scaler in cycles_scalers, free_dims in free_dims_values
        run_id += 1
        shared_row_added = false
        for outer_trial in 1:trials
            run_counter += 1
            if showProgress
                println("MIMO Potts run $(run_counter): runId=$(run_id), snr=$(snr_index), instance=$(instance_index), outerTrial=$(outer_trial), noiseRatio=$(noise_ratio), cyclesScaler=$(cycles_scaler), freeDims=$(free_dims), optimizer=$(optimizer)")
            end
            run_seed = isnothing(seed) ? nothing : hash((seed, snr_index, instance_index, noise_ratio, cycles_scaler, free_dims, outer_trial))
            r = solve_mimo_potts(instance_path;
                snr_index = Int(snr_index),
                instance_index = Int(instance_index),
                modulation = modulation,
                free_dims = Int(free_dims),
                fixed_candidates_per_dim = fixed_candidates_per_dim,
                max_branches = max_branches,
                trials = 1,
                num_cycles = numCycles,
                cycles_scaler = cycles_scaler,
                noise_ratio = noise_ratio,
                noise_stepper = noise_stepper,
                optimizer = optimizer,
                batch_rate = batch_rate,
                eoffset = eoffset,
                cacheCouplings = cacheCouplings,
                preprocess = preprocess,
                seed = run_seed,
            )

            if format === :flat
                push!(rows, merge(_result_row(r), (outerTrial = outer_trial,)))
            else
                if !shared_row_added
                    push!(run_rows, _shared_result_row(r, run_id))
                    shared_row_added = true
                end
                push!(trial_rows, _trial_result_row(r, run_id, outer_trial))
            end
        end
    end

    if format === :flat
        df = DataFrame(rows)
        if !isnothing(jld2file)
            save_mimo_potts_results(String(jld2file), df; metadata = saveMetadata)
        end
        return df
    end

    result = (runs = DataFrame(run_rows), trials = DataFrame(trial_rows))
    if !isnothing(jld2file)
        save_mimo_potts_results(String(jld2file), result; metadata = saveMetadata)
    end
    return result
end
