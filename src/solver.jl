
export MIMOPottsInstance, MIMOPottsBranch, MIMOPottsSolveResult
export pam_levels, load_mimo_potts_instance, solve_mimo_potts, curunanmimoinstance
export save_mimo_potts_results, load_mimo_potts_results
export trace_mimo_potts

using LinearAlgebra
using Random
using Base.Threads: @threads, nthreads
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
    noise_coupling::Symbol
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

function _check_reliability_mode(reliability_mode)
    mode = Symbol(reliability_mode)
    mode in (:margin, :whitened) ||
        error("Unsupported reliability_mode $(reliability_mode). Use :margin or :whitened.")
    return mode
end

function _post_equalization_error_variance(H::AbstractMatrix, noise_variance::Real, base_decoder::Symbol)
    A = Symmetric(transpose(H) * H)
    if base_decoder === :mmse
        A = Symmetric(Matrix(A) + Float64(noise_variance) * I(size(H, 2)))
    end
    Ainv = try
        inv(A)
    catch
        pinv(Matrix(A))
    end
    return Float64(noise_variance) .* max.(diag(Ainv), eps())
end

function zf_mmse_preprocess(inst::MIMOPottsInstance; reliability_mode::Symbol=:margin)
    reliability_mode = _check_reliability_mode(reliability_mode)
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
    if reliability_mode === :whitened
        err_var = _post_equalization_error_variance(H, inst.noise_variance, base_decoder)
        reliability ./= err_var
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

function _check_noise_coupling(noise_coupling)
    coupling = Symbol(noise_coupling)
    coupling in (:directional, :common) ||
        error("Unsupported noise_coupling $(noise_coupling). Use :directional or :common.")
    return coupling
end

@inline function _best_adjacent_transition(grad_i::Real, gii::Real, state::Int,
    value::Real, levels::AbstractVector, noise::Real, noise_coupling::Symbol)
    best_state = state
    best_delta = 0.0
    best_true_delta = 0.0
    best_score = 0.0

    # Down is evaluated first, matching the old equality convention.
    @inbounds for proposed in (state - 1, state + 1)
        if 1 <= proposed <= length(levels)
            delta = Float64(levels[proposed]) - Float64(value)
            true_delta = delta * Float64(grad_i) + Float64(gii) * delta * delta
            score = noise_coupling === :directional ?
                true_delta - delta * Float64(noise) :
                true_delta + abs(delta) * Float64(noise)
            if score < best_score
                best_state = proposed
                best_delta = delta
                best_true_delta = true_delta
                best_score = score
            end
        end
    end
    return best_state, best_delta, best_true_delta, best_score
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


_backend_val(backend) = Val(Symbol(backend))
_mimo_backend_available(::Val{:cpu}) = true
_mimo_backend_available(_) = false

function _check_mimo_backend(backend)
    val = _backend_val(backend)
    if !_mimo_backend_available(val)
        error("MIMO Potts backend $(Symbol(backend)) is not available. Use backend=:cpu, or load CUDA.jl before requesting backend=:cuda.")
    end
    return val
end

function _trial_seed(seed, snr_index::Int, instance_index::Int, branch_rank::Int, trial::Int)
    if isnothing(seed)
        return rand(UInt64)
    end
    return UInt64(hash((seed, snr_index, instance_index, branch_rank, trial)))
end

function _potts_energy(G::AbstractMatrix, h::AbstractVector, const_term::Real, values::AbstractVector)
    f = length(values)
    e = Float64(const_term)
    @inbounds for i in 1:f
        vi = values[i]
        row_sum = 0.0
        for j in 1:f
            row_sum += G[i, j] * values[j]
        end
        e += vi * row_sum + h[i] * vi
    end
    return e
end

function _recompute_gradient_energy!(grad::AbstractVector{Float64}, gx::AbstractVector{Float64},
    G::AbstractMatrix, h::AbstractVector, const_term::Real, values::AbstractVector{Float64})
    mul!(gx, G, values)
    f = length(values)
    hdot = 0.0
    @inbounds for i in 1:f
        grad[i] = 2.0 * gx[i] + h[i]
        hdot += h[i] * values[i]
    end
    return dot(values, gx) + hdot + Float64(const_term)
end

function _apply_state_deltas!(states::AbstractVector{Int}, values::AbstractVector{Float64},
    delta_indices::AbstractVector{Int}, delta_values::AbstractVector{Float64},
    new_states::AbstractVector{Int}, delta_count::Int)
    @inbounds for a in 1:delta_count
        ia = delta_indices[a]
        states[ia] = new_states[a]
        values[ia] += delta_values[a]
    end
    return nothing
end

function _apply_deltas!(grad::AbstractVector{Float64}, states::AbstractVector{Int}, values::AbstractVector{Float64},
    G::AbstractMatrix, delta_indices::AbstractVector{Int}, delta_values::AbstractVector{Float64},
    new_states::AbstractVector{Int}, delta_count::Int)
    linear = 0.0
    quad = 0.0
    @inbounds for a in 1:delta_count
        ia = delta_indices[a]
        da = delta_values[a]
        linear += da * grad[ia]
        for b in 1:delta_count
            ib = delta_indices[b]
            quad += da * G[ia, ib] * delta_values[b]
        end
    end

    _apply_state_deltas!(states, values, delta_indices, delta_values, new_states, delta_count)

    f = length(grad)
    @inbounds for row in 1:f
        update = 0.0
        for a in 1:delta_count
            update += G[row, delta_indices[a]] * delta_values[a]
        end
        grad[row] += 2.0 * update
    end
    return linear + quad
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
    noise_coupling::Symbol=:directional,
    batch_rate::Real,
    eoffset::Real,
    rng::AbstractRNG,
    record_trace::Bool=false,
)
    noise_coupling = _check_noise_coupling(noise_coupling)
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
    gdiag = Float64.(diag(G))
    noise_start = Float64(noise_ratio) * _gradient_bound(G, h, levels)

    states = Vector{Int}(undef, f)
    values = Vector{Float64}(undef, f)
    best_states = Vector{Int}(undef, f)
    grad = Vector{Float64}(undef, f)
    gx = Vector{Float64}(undef, f)
    delta_indices = Vector{Int}(undef, f)
    delta_values = Vector{Float64}(undef, f)
    new_states = Vector{Int}(undef, f)
    noise_random = Vector{Float64}(undef, f)
    mask_random = Vector{Float64}(undef, f)

    @inbounds for i in 1:f
        state = rand(rng, 1:k)
        states[i] = state
        values[i] = Float64(levels[state])
        best_states[i] = state
    end

    distance = _recompute_gradient_energy!(grad, gx, G, h, const_term, values)
    initial_states = record_trace ? copy(states) : Int[]
    trace_energy = record_trace ? Vector{Float64}(undef, steps + 1) : Float64[]
    trace_changed_count = record_trace ? zeros(Int, steps) : Int[]
    trace_changed_dimension = record_trace ? zeros(Int, steps) : Int[]
    trace_old_state = record_trace ? zeros(Int, steps) : Int[]
    trace_new_state = record_trace ? zeros(Int, steps) : Int[]
    trace_states = record_trace ? Matrix{Int}(undef, f, steps + 1) : Matrix{Int}(undef, 0, 0)
    trace_noise = record_trace ? Matrix{Float64}(undef, f, steps) : Matrix{Float64}(undef, 0, 0)
    trace_true_delta = record_trace ? zeros(Float64, steps) : Float64[]
    trace_noisy_score = record_trace ? zeros(Float64, steps) : Float64[]
    record_trace && (trace_energy[1] = Float64(distance))
    record_trace && (trace_states[:, 1] .= states)
    best_distance = distance
    step_found = 0
    dau_offset = 0.0
    batch_rate_f = Float64(batch_rate)
    eoffset_f = Float64(eoffset)

    for step in 1:steps
        scale = _noise_scale(noise_stepper, noise_start, steps, step) + dau_offset
        changed_this_step = 0
        changed_dimension = 0
        old_state = 0
        new_state = 0
        step_noisy_score = 0.0

        if optimizer === :batch || optimizer === :batchdau
            delta_count = 0
            rand!(rng, noise_random)
            rand!(rng, mask_random)
            @inbounds for i in 1:f
                noise = scale * (2.0 * noise_random[i] - 1.0)
                proposed, delta, _, score = _best_adjacent_transition(
                    grad[i], gdiag[i], states[i], values[i], levels, noise, noise_coupling)
                if proposed != states[i] && mask_random[i] < batch_rate_f
                    delta_count += 1
                    delta_indices[delta_count] = i
                    delta_values[delta_count] = delta
                    new_states[delta_count] = proposed
                    step_noisy_score += score
                end
            end

            if optimizer === :batchdau
                dau_offset = delta_count == 0 ? dau_offset + eoffset_f : 0.0
            end

            if delta_count > 0
                changed_this_step = delta_count
                if f >= 32 || delta_count > max(4, f >>> 4)
                    _apply_state_deltas!(states, values, delta_indices, delta_values, new_states, delta_count)
                    distance = _recompute_gradient_energy!(grad, gx, G, h, const_term, values)
                else
                    distance += _apply_deltas!(grad, states, values, G, delta_indices, delta_values, new_states, delta_count)
                end
            end
        elseif optimizer === :singleflip || optimizer === :dau
            changed_count = 0
            chosen_index = 0
            chosen_state = 0
            chosen_delta = 0.0
            chosen_score = 0.0
            rand!(rng, noise_random)
            @inbounds for i in 1:f
                noise = scale * (2.0 * noise_random[i] - 1.0)
                proposed, delta, _, score = _best_adjacent_transition(
                    grad[i], gdiag[i], states[i], values[i], levels, noise, noise_coupling)
                if proposed != states[i]
                    changed_count += 1
                    if rand(rng, 1:changed_count) == 1
                        chosen_index = i
                        chosen_state = proposed
                        chosen_delta = delta
                        chosen_score = score
                    end
                end
            end

            if changed_count == 0
                if optimizer === :dau
                    dau_offset += eoffset_f
                end
            else
                dau_offset = 0.0
                changed_this_step = 1
                changed_dimension = free_indices[chosen_index]
                old_state = states[chosen_index]
                new_state = chosen_state
                delta_indices[1] = chosen_index
                delta_values[1] = chosen_delta
                new_states[1] = chosen_state
                distance += _apply_deltas!(grad, states, values, G, delta_indices, delta_values, new_states, 1)
                step_noisy_score = chosen_score
            end
        else
            error("Unsupported MIMO Potts optimizer $(optimizer). Use :batch, :singleflip, :dau, or :batchdau.")
        end

        if distance < best_distance
            exact_distance = _potts_energy(G, h, const_term, values)
            if exact_distance < best_distance
                best_distance = exact_distance
                best_states .= states
                step_found = step
            end
            distance = exact_distance
        end

        if record_trace
            trace_energy[step + 1] = _potts_energy(G, h, const_term, values)
            trace_changed_count[step] = changed_this_step
            trace_changed_dimension[step] = changed_dimension
            trace_old_state[step] = old_state
            trace_new_state[step] = new_state
            trace_states[:, step + 1] .= states
            trace_noise[:, step] .= scale .* (2.0 .* noise_random .- 1.0)
            trace_true_delta[step] = trace_energy[step + 1] - trace_energy[step]
            trace_noisy_score[step] = step_noisy_score
        end
    end

    return (; best_distance = Float64(best_distance), best_free_states = Vector{Int}(best_states),
        final_free_states = Vector{Int}(states),
        step_found = step_found, total_steps = steps, total_gradient_evals = steps,
        trace = record_trace ? (;
            initial_states,
            energy = trace_energy,
            changed_count = trace_changed_count,
            changed_dimension = trace_changed_dimension,
            old_state = trace_old_state,
            new_state = trace_new_state,
            states = trace_states,
            noise = trace_noise,
            true_delta = trace_true_delta,
            noisy_score = trace_noisy_score,
        ) : nothing)
end

"""
    trace_mimo_potts(path; kwargs...)

Run CPU MIMO Potts trials with every real-valued dimension free and record the
exact current objective after every optimizer step. This diagnostic interface
does not record the incumbent/best-so-far trajectory and does not alter update
acceptance. It supports all CPU optimizers.
"""
function trace_mimo_potts(path::AbstractString;
    snr_index::Int=1,
    instance_index::Int=1,
    modulation=nothing,
    trials::Int=32,
    num_cycles::Int=64,
    cycles_scaler::Real=1.0,
    noise_ratio::Real=0.0,
    noise_stepper::Symbol=:linear,
    noise_coupling::Symbol=:directional,
    optimizer::Symbol=:singleflip,
    batch_rate::Real=0.5,
    eoffset::Real=0.0,
    preprocess=:qr,
    seed=20260612,
)
    optimizer in (:batch, :batchdau, :singleflip, :dau) ||
        error("Unsupported trace optimizer $(optimizer)")
    noise_coupling = _check_noise_coupling(noise_coupling)
    trials > 0 || error("trials must be positive")

    inst = load_mimo_potts_instance(path; snr_index, instance_index, modulation)
    prep = zf_mmse_preprocess(inst)
    geometry = _preprocessed_geometry(inst, preprocess)
    free_dims = length(prep.base_states)
    branches = build_mimo_potts_branches(inst, prep;
        free_dims = free_dims,
        fixed_candidates_per_dim = 1,
        max_branches = 1,
        H = geometry.H,
        y = geometry.y,
        const_offset = geometry.const_offset,
    )
    length(branches) == 1 || error("Expected one all-free branch, got $(length(branches))")
    branch = only(branches)
    isempty(branch.fixed_indices) || error("Trace branch unexpectedly contains fixed dimensions")
    coupling_cache = _make_coupling_cache(
        geometry.H, geometry.y, branch.free_indices, branch.fixed_indices,
        geometry.const_offset, true,
    )

    trace_trials = Vector{NamedTuple}(undef, trials)
    for trial in 1:trials
        trial_seed = _trial_seed(seed, snr_index, instance_index, branch.rank, trial)
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
            noise_coupling = noise_coupling,
            batch_rate = batch_rate,
            eoffset = eoffset,
            rng = MersenneTwister(trial_seed),
            record_trace = true,
        )
        initial_values = states_to_values(sub.trace.initial_states, inst.levels)
        final_values = states_to_values(sub.final_free_states, inst.levels)
        trace_trials[trial] = (;
            trial,
            seed = trial_seed,
            initial_states = sub.trace.initial_states,
            final_states = sub.final_free_states,
            initial_energy = mimo_distance(inst.H, inst.y, initial_values),
            final_energy = mimo_distance(inst.H, inst.y, final_values),
            energy = sub.trace.energy,
            changed_count = sub.trace.changed_count,
            changed_dimension = sub.trace.changed_dimension,
            old_state = sub.trace.old_state,
            new_state = sub.trace.new_state,
            states = sub.trace.states,
            noise = sub.trace.noise,
            true_delta = sub.trace.true_delta,
            noisy_score = sub.trace.noisy_score,
        )
    end

    return (;
        source = inst.source,
        snr_index = inst.snr_index,
        instance_index = inst.instance_index,
        ebnodb = inst.ebnodb,
        modulation = inst.modulation,
        nt = inst.nt,
        nr = inst.nr,
        optimizer,
        noise_ratio = Float64(noise_ratio),
        noise_stepper,
        noise_coupling,
        eoffset = Float64(eoffset),
        num_cycles,
        cycles_scaler = Float64(cycles_scaler),
        free_dims,
        preprocess = geometry.preprocess,
        base_distance = Float64(prep.base_distance),
        trials = trace_trials,
    )
end

function _run_potts_subproblem_batch_backend(::Val{:cpu}, inst::MIMOPottsInstance, branch::MIMOPottsBranch, prep;
    H::AbstractMatrix,
    y::AbstractVector,
    const_offset::Real,
    coupling_cache,
    optimizer::Symbol,
    num_cycles::Int,
    cycles_scaler::Real,
    noise_ratio::Real,
    noise_stepper::Symbol,
    noise_coupling::Symbol,
    batch_rate::Real,
    eoffset::Real,
    seeds::AbstractVector{UInt64},
    gpuBatchSize::Int=256,
    gpuFloat::Type{<:AbstractFloat}=Float32,
)
    return [
        _run_potts_subproblem(inst, branch, prep;
            H = H,
            y = y,
            const_offset = const_offset,
            coupling_cache = coupling_cache,
            optimizer = optimizer,
            num_cycles = num_cycles,
            cycles_scaler = cycles_scaler,
            noise_ratio = noise_ratio,
            noise_stepper = noise_stepper,
            noise_coupling = noise_coupling,
            batch_rate = batch_rate,
            eoffset = eoffset,
            rng = MersenneTwister(seed),
        )
        for seed in seeds
    ]
end

function _run_potts_subproblem_branch_batch_backend(backend_val, inst::MIMOPottsInstance, branches::AbstractVector{<:MIMOPottsBranch}, active_trials_by_branch::AbstractVector, prep;
    H::AbstractMatrix,
    y::AbstractVector,
    const_offset::Real,
    coupling_cache,
    optimizer::Symbol,
    num_cycles::Int,
    cycles_scaler::Real,
    noise_ratio::Real,
    noise_stepper::Symbol,
    noise_coupling::Symbol,
    batch_rate::Real,
    eoffset::Real,
    snr_index::Int,
    instance_index::Int,
    seed,
    gpuBatchSize::Int=256,
    gpuFloat::Type{<:AbstractFloat}=Float32,
)
    out = Vector{Any}(undef, length(branches))
    for (i, branch) in enumerate(branches)
        active_trials = active_trials_by_branch[i]
        seeds = [_trial_seed(seed, snr_index, instance_index, branch.rank, trial) for trial in active_trials]
        out[i] = _run_potts_subproblem_batch_backend(backend_val, inst, branch, prep;
            H = H,
            y = y,
            const_offset = const_offset,
            coupling_cache = coupling_cache,
            optimizer = optimizer,
            num_cycles = num_cycles,
            cycles_scaler = cycles_scaler,
            noise_ratio = noise_ratio,
            noise_stepper = noise_stepper,
            noise_coupling = noise_coupling,
            batch_rate = batch_rate,
            eoffset = eoffset,
            seeds = seeds,
            gpuBatchSize = gpuBatchSize,
            gpuFloat = gpuFloat,
        )
    end
    return out
end

function _full_states(prep, branch::MIMOPottsBranch, free_states::Vector{Int})
    states = copy(prep.base_states)
    states[branch.fixed_indices] .= branch.fixed_states
    states[branch.free_indices] .= free_states
    return states
end


function _solve_mimo_potts_batched(path::AbstractString, backend_val;
    snr_index::Int,
    instance_index::Int,
    modulation,
    free_dims::Int,
    fixed_candidates_per_dim::Int,
    max_branches::Int,
    trials::Int,
    num_cycles::Int,
    cycles_scaler::Real,
    noise_ratio::Real,
    noise_stepper::Symbol,
    noise_coupling::Symbol,
    optimizer::Symbol,
    batch_rate::Real,
    eoffset::Real,
    cacheCouplings::Bool,
    preprocess,
    seed,
    gpuBatchSize::Int,
    gpuFloat::Type{<:AbstractFloat},
    return_all_trials::Bool=false,
    reliability_mode::Symbol=:margin,
)
    if !(optimizer in (:batch, :batchdau, :singleflip, :dau))
        error("Unsupported MIMO Potts optimizer $(optimizer). Use :batch, :singleflip, :dau, or :batchdau.")
    end

    inst = load_mimo_potts_instance(path; snr_index, instance_index, modulation)
    prep = zf_mmse_preprocess(inst; reliability_mode = reliability_mode)
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

    current_radius = fill(initial_radius, trials)
    trial_best_distance = fill(initial_radius, trials)
    trial_best_states = [copy(prep.base_states) for _ in 1:trials]
    trial_best_values = [copy(prep.base_values) for _ in 1:trials]
    trial_step_found_best = zeros(Int, trials)
    trial_branch_found_best = zeros(Int, trials)
    trial_total_steps = zeros(Int, trials)
    trial_total_gradient_evals = zeros(Int, trials)
    trial_branches_visited = zeros(Int, trials)
    trial_branches_pruned = zeros(Int, trials)
    trial_radius_updates = zeros(Int, trials)

    elapsed = @elapsed begin
        branch_i = 1
        while branch_i <= length(branches)
            chunk_branches = MIMOPottsBranch[]
            chunk_active_trials = Vector{Vector{Int}}()
            chunk_jobs = 0

            while branch_i <= length(branches) && (isempty(chunk_branches) || chunk_jobs < max(1, gpuBatchSize))
                branch = branches[branch_i]
                active_trials = [trial for trial in 1:trials if branch.lower_bound < current_radius[trial]]
                active_set = Set(active_trials)
                for trial in 1:trials
                    if trial in active_set
                        trial_branches_visited[trial] += 1
                    else
                        trial_branches_pruned[trial] += 1
                    end
                end
                branches_pruned += trials - length(active_trials)
                if !isempty(active_trials)
                    branches_visited += length(active_trials)
                    push!(chunk_branches, branch)
                    push!(chunk_active_trials, active_trials)
                    chunk_jobs += length(active_trials)
                end
                branch_i += 1
            end

            if isempty(chunk_branches)
                continue
            end

            batch_step_offset = total_steps
            trial_batch_step_offset = copy(trial_total_steps)
            trial_chunk_job_index = zeros(Int, trials)
            subs_by_branch = _run_potts_subproblem_branch_batch_backend(backend_val, inst, chunk_branches, chunk_active_trials, prep;
                H = geometry.H,
                y = geometry.y,
                const_offset = geometry.const_offset,
                coupling_cache = coupling_cache,
                optimizer = optimizer,
                num_cycles = num_cycles,
                cycles_scaler = cycles_scaler,
                noise_ratio = noise_ratio,
                noise_stepper = noise_stepper,
                noise_coupling = noise_coupling,
                batch_rate = batch_rate,
                eoffset = eoffset,
                snr_index = snr_index,
                instance_index = instance_index,
                seed = seed,
                gpuBatchSize = gpuBatchSize,
                gpuFloat = gpuFloat,
            )

            job_linear = 0
            for (branch_local_i, branch) in enumerate(chunk_branches)
                active_trials = chunk_active_trials[branch_local_i]
                subs = subs_by_branch[branch_local_i]
                for (local_i, sub) in enumerate(subs)
                    job_linear += 1
                    trial = active_trials[local_i]
                    trial_chunk_job_index[trial] += 1
                    total_steps += sub.total_steps
                    total_gradient_evals += sub.total_gradient_evals
                    trial_total_steps[trial] += sub.total_steps
                    trial_total_gradient_evals[trial] += sub.total_gradient_evals

                    if sub.best_distance < trial_best_distance[trial]
                        candidate_states = _full_states(prep, branch, sub.best_free_states)
                        candidate_values = states_to_values(candidate_states, inst.levels)
                        candidate_distance = mimo_distance(inst.H, inst.y, candidate_values)
                        if candidate_distance < trial_best_distance[trial]
                            trial_best_states[trial] = candidate_states
                            trial_best_values[trial] = candidate_values
                            trial_best_distance[trial] = candidate_distance
                            current_radius[trial] = candidate_distance
                            radius_updates += 1
                            trial_radius_updates[trial] += 1
                            trial_step_found_best[trial] = trial_batch_step_offset[trial] + (trial_chunk_job_index[trial] - 1) * sub.total_steps + sub.step_found
                            trial_branch_found_best[trial] = branch.rank

                            if candidate_distance < best_distance
                                best_states = candidate_states
                                best_values = candidate_values
                                best_distance = candidate_distance
                                step_found_best = batch_step_offset + (job_linear - 1) * sub.total_steps + sub.step_found
                                trial_found_best = trial
                                branch_found_best = branch.rank
                            end
                        end
                    end
                end
            end
        end

        for trial in 1:trials
            if trial_best_distance[trial] < best_distance
                best_states = trial_best_states[trial]
                best_values = trial_best_values[trial]
                best_distance = trial_best_distance[trial]
                trial_found_best = trial
            end
        end
    end

    if return_all_trials
        per_trial_step_time = elapsed / trials
        return [
            let (tber, tser, tfer) = detection_error_rates(trial_best_states[trial], inst.x_true, inst.levels)
                MIMOPottsSolveResult(
                    source = inst.source,
                    snr_index = inst.snr_index,
                    instance_index = inst.instance_index,
                    ebnodb = inst.ebnodb,
                    modulation = inst.modulation,
                    optimizer = optimizer,
                    trials = 1,
                    num_cycles = num_cycles,
                    noise_ratio = Float64(noise_ratio),
                    noise_coupling = noise_coupling,
                    cycles_scaler = Float64(cycles_scaler),
                    free_dims = free_dims,
                    initial_radius = Float64(prep.base_distance),
                    final_radius = trial_best_distance[trial],
                    best_distance = trial_best_distance[trial],
                    zf_distance = Float64(prep.zf_distance),
                    mmse_distance = Float64(prep.mmse_distance),
                    step_found_best = trial_step_found_best[trial],
                    trial_found_best = 1,
                    branch_found_best = trial_branch_found_best[trial],
                    total_steps = trial_total_steps[trial],
                    total_gradient_evals = trial_total_gradient_evals[trial],
                    branches_generated = length(branches),
                    branches_visited = trial_branches_visited[trial],
                    branches_pruned = trial_branches_pruned[trial],
                    radius_updates = trial_radius_updates[trial],
                    ber = tber,
                    ser = tser,
                    fer = tfer,
                    zf_ber = Float64(prep.zf_ber),
                    mmse_ber = Float64(prep.mmse_ber),
                    best_states = trial_best_states[trial],
                    best_values = trial_best_values[trial],
                    step_time = per_trial_step_time,
                    preprocess = geometry.preprocess,
                    cache_couplings = cacheCouplings,
                )
            end
            for trial in 1:trials
        ]
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
        noise_coupling = noise_coupling,
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

function _run_cpu_trial(inst::MIMOPottsInstance, prep, geometry, branches, coupling_cache;
    trial_seed::UInt64,
    optimizer::Symbol,
    num_cycles::Int,
    cycles_scaler::Real,
    noise_ratio::Real,
    noise_stepper::Symbol,
    noise_coupling::Symbol,
    batch_rate::Real,
    eoffset::Real,
)
    rng = MersenneTwister(trial_seed)
    current_radius = Float64(prep.base_distance)
    trial_best_distance = Float64(prep.base_distance)
    trial_best_states = copy(prep.base_states)
    trial_best_values = copy(prep.base_values)
    step_found_best = 0
    branch_found_best = 0
    total_steps = 0
    total_gradient_evals = 0
    branches_visited = 0
    branches_pruned = 0
    radius_updates = 0

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
            noise_coupling = noise_coupling,
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
                step_found_best = step_offset + sub.step_found
                branch_found_best = branch.rank
            end
        end
    end

    return (;
        best_distance = trial_best_distance,
        best_states = trial_best_states,
        best_values = trial_best_values,
        step_found_best = step_found_best,
        branch_found_best = branch_found_best,
        total_steps = total_steps,
        total_gradient_evals = total_gradient_evals,
        branches_visited = branches_visited,
        branches_pruned = branches_pruned,
        radius_updates = radius_updates,
    )
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
    noise_coupling::Symbol=:directional,
    optimizer::Symbol=:batch,
    batch_rate::Real=0.5,
    eoffset::Real=0.0,
    cacheCouplings::Bool=true,
    preprocess=:normal,
    seed=nothing,
    backend=:cpu,
    gpuBatchSize::Int=256,
    gpuFloat::Type{<:AbstractFloat}=Float32,
    cpuThreads::Bool=true,
    reliability_mode::Symbol=:margin,
)
    noise_coupling = _check_noise_coupling(noise_coupling)
    reliability_mode = _check_reliability_mode(reliability_mode)
    backend_val = _check_mimo_backend(backend)
    if Symbol(backend) !== :cpu
        return _solve_mimo_potts_batched(path, backend_val;
            snr_index = snr_index,
            instance_index = instance_index,
            modulation = modulation,
            free_dims = free_dims,
            fixed_candidates_per_dim = fixed_candidates_per_dim,
            max_branches = max_branches,
            trials = trials,
            num_cycles = num_cycles,
            cycles_scaler = cycles_scaler,
            noise_ratio = noise_ratio,
            noise_stepper = noise_stepper,
            noise_coupling = noise_coupling,
            optimizer = optimizer,
            batch_rate = batch_rate,
            eoffset = eoffset,
            cacheCouplings = cacheCouplings,
            preprocess = preprocess,
            seed = seed,
            gpuBatchSize = gpuBatchSize,
            gpuFloat = gpuFloat,
            reliability_mode = reliability_mode,
        )
    end

    old_blas_threads = BLAS.get_num_threads()
    if old_blas_threads != 1
        BLAS.set_num_threads(1)
    end
    try

    inst = load_mimo_potts_instance(path; snr_index, instance_index, modulation)
    prep = zf_mmse_preprocess(inst; reliability_mode = reliability_mode)
    geometry = _preprocessed_geometry(inst, preprocess)
    initial_radius = Float64(prep.base_distance)

    branches = build_mimo_potts_branches(inst, prep;
        free_dims = free_dims,
        fixed_candidates_per_dim = fixed_candidates_per_dim,
        max_branches = max_branches,
        H = geometry.H,
        y = geometry.y,
        const_offset = geometry.const_offset,
    )
    coupling_cache = isempty(branches) ? nothing : _make_coupling_cache(geometry.H, geometry.y, branches[1].free_indices, branches[1].fixed_indices, geometry.const_offset, cacheCouplings)

    seed_rng = isnothing(seed) ? Random.default_rng() : nothing
    trial_seeds = Vector{UInt64}(undef, trials)
    for trial in 1:trials
        trial_seeds[trial] = isnothing(seed) ? rand(seed_rng, UInt64) : _trial_seed(seed, snr_index, instance_index, 0, trial)
    end

    trial_results = Vector{Any}(undef, trials)
    elapsed = @elapsed begin
        if cpuThreads && nthreads() > 1 && trials > 1
            @threads for trial in 1:trials
                trial_results[trial] = _run_cpu_trial(inst, prep, geometry, branches, coupling_cache;
                    trial_seed = trial_seeds[trial],
                    optimizer = optimizer,
                    num_cycles = num_cycles,
                    cycles_scaler = cycles_scaler,
                    noise_ratio = noise_ratio,
                    noise_stepper = noise_stepper,
                    noise_coupling = noise_coupling,
                    batch_rate = batch_rate,
                    eoffset = eoffset,
                )
            end
        else
            for trial in 1:trials
                trial_results[trial] = _run_cpu_trial(inst, prep, geometry, branches, coupling_cache;
                    trial_seed = trial_seeds[trial],
                    optimizer = optimizer,
                    num_cycles = num_cycles,
                    cycles_scaler = cycles_scaler,
                    noise_ratio = noise_ratio,
                    noise_stepper = noise_stepper,
                    noise_coupling = noise_coupling,
                    batch_rate = batch_rate,
                    eoffset = eoffset,
                )
            end
        end
    end

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
    step_offset = 0

    for trial in 1:trials
        tr = trial_results[trial]
        branches_visited += tr.branches_visited
        branches_pruned += tr.branches_pruned
        radius_updates += tr.radius_updates
        total_steps += tr.total_steps
        total_gradient_evals += tr.total_gradient_evals

        if tr.best_distance < best_distance
            best_states = tr.best_states
            best_values = tr.best_values
            best_distance = tr.best_distance
            step_found_best = step_offset + tr.step_found_best
            trial_found_best = trial
            branch_found_best = tr.branch_found_best
        end
        step_offset += tr.total_steps
    end

    ber, ser, fer = detection_error_rates(best_states, inst.x_true, inst.levels)
    result = MIMOPottsSolveResult(
        source = inst.source,
        snr_index = inst.snr_index,
        instance_index = inst.instance_index,
        ebnodb = inst.ebnodb,
        modulation = inst.modulation,
        optimizer = optimizer,
        trials = trials,
        num_cycles = num_cycles,
        noise_ratio = Float64(noise_ratio),
        noise_coupling = noise_coupling,
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
    return result
finally
    if old_blas_threads != 1
        BLAS.set_num_threads(old_blas_threads)
    end
end
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
        noiseCoupling = String(r.noise_coupling),
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
    noise_coupling::Symbol = :directional,
    batch_rate::Real = 0.5,
    eoffset::Real = 0.0,
    cacheCouplings::Bool = true,
    preprocess = :normal,
    seed = nothing,
    jld2file = nothing,
    saveMetadata = nothing,
    showProgress::Bool = true,
    resultFormat = :flat,
    backend = :cpu,
    gpuBatchSize::Int = 256,
    gpuFloat::Type{<:AbstractFloat} = Float32,
    cpuThreads::Bool = true,
    reliabilityMode::Symbol = :margin,
)
    noise_coupling = _check_noise_coupling(noise_coupling)
    reliability_mode = _check_reliability_mode(reliabilityMode)
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

    run_on_cpu = Symbol(backend) === :cpu

    for snr_index in snrs, instance_index in instances, noise_ratio in noise_ratios, cycles_scaler in cycles_scalers, free_dims in free_dims_values
        run_id += 1
        shared_row_added = false

        if run_on_cpu
            for outer_trial in 1:trials
                run_counter += 1
                if showProgress
                    println("MIMO Potts run $(run_counter): runId=$(run_id), snr=$(snr_index), instance=$(instance_index), outerTrial=$(outer_trial), noiseRatio=$(noise_ratio), noiseCoupling=$(noise_coupling), cyclesScaler=$(cycles_scaler), freeDims=$(free_dims), optimizer=$(optimizer)")
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
                    noise_coupling = noise_coupling,
                    optimizer = optimizer,
                    batch_rate = batch_rate,
                    eoffset = eoffset,
                    cacheCouplings = cacheCouplings,
                    preprocess = preprocess,
                    seed = run_seed,
                    backend = backend,
                    gpuBatchSize = gpuBatchSize,
                    gpuFloat = gpuFloat,
                    reliability_mode = reliability_mode,
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
        else
            # Non-CPU backends (e.g. :cuda) batch all `trials` outer trials into a
            # single call instead of looping `trials` times with trials=1: looping
            # re-does load_mimo_potts_instance/branch-building/GPU buffer
            # allocation+H2D transfer from scratch for every trial even though only
            # the RNG seed differs, which measured ~7.5-7.8x slower than batching.
            run_counter += trials
            if showProgress
                println("MIMO Potts run $(run_counter - trials + 1)-$(run_counter): runId=$(run_id), snr=$(snr_index), instance=$(instance_index), trials=$(trials) (batched), noiseRatio=$(noise_ratio), noiseCoupling=$(noise_coupling), cyclesScaler=$(cycles_scaler), freeDims=$(free_dims), optimizer=$(optimizer)")
            end
            run_seed = isnothing(seed) ? nothing : hash((seed, snr_index, instance_index, noise_ratio, cycles_scaler, free_dims))
            backend_val = _check_mimo_backend(backend)
            results = _solve_mimo_potts_batched(instance_path, backend_val;
                snr_index = Int(snr_index),
                instance_index = Int(instance_index),
                modulation = modulation,
                free_dims = Int(free_dims),
                fixed_candidates_per_dim = fixed_candidates_per_dim,
                max_branches = max_branches,
                trials = trials,
                num_cycles = numCycles,
                cycles_scaler = cycles_scaler,
                noise_ratio = noise_ratio,
                noise_stepper = noise_stepper,
                noise_coupling = noise_coupling,
                optimizer = optimizer,
                batch_rate = batch_rate,
                eoffset = eoffset,
                cacheCouplings = cacheCouplings,
                preprocess = preprocess,
                seed = run_seed,
                gpuBatchSize = gpuBatchSize,
                gpuFloat = gpuFloat,
                return_all_trials = true,
                reliability_mode = reliability_mode,
            )

            for (outer_trial, r) in enumerate(results)
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
