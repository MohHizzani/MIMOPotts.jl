module MIMOPottsCUDAExt

using CUDA
using LinearAlgebra: diag

import MIMOPotts
import MIMOPotts: MIMOPottsInstance, MIMOPottsBranch

MIMOPotts._mimo_backend_available(::Val{:cuda}) = CUDA.functional()

@inline function _mix64(x::UInt64)
    x = xor(x, x >>> 30)
    x *= 0xbf58476d1ce4e5b9
    x = xor(x, x >>> 27)
    x *= 0x94d049bb133111eb
    return xor(x, x >>> 31)
end

@inline function _rand_unit(::Type{T}, seed::UInt64, step::Int32, dim::Int32, tag::Int32) where {T<:AbstractFloat}
    x = seed
    x = xor(x, UInt64(step) * 0x9e3779b97f4a7c15)
    x = xor(x, UInt64(dim) * 0xbf58476d1ce4e5b9)
    x = xor(x, UInt64(tag) * 0x94d049bb133111eb)
    bits = (_mix64(x) >>> 40) & 0x0000000000ffffff
    return T(bits) * T(5.960464477539063e-8)
end

function _threads_for_free_dims(f::Int)
    f <= 1024 || error("backend=:cuda currently supports at most 1024 free dimensions per fused block; got $(f).")
    threads = 1
    while threads < f
        threads <<= 1
    end
    return max(32, threads)
end

function _potts_init_kernel!(states, values::AbstractArray{T}, best_states, best_distance, step_found,
    seeds, Gt, h, levels, const_terms, f::Int32, k::Int32) where {T<:AbstractFloat}
    tid = Int32(threadIdx().x)
    job = Int32(blockIdx().x)
    nthreads = Int32(blockDim().x)
    shmem = CuDynamicSharedArray(T, Int(nthreads) + 1)

    if tid <= f
        seed = seeds[job]
        state = Int32(1) + Int32(floor(_rand_unit(T, seed, Int32(0), tid, Int32(11)) * T(k)))
        state = ifelse(state < Int32(1), Int32(1), ifelse(state > k, k, state))
        states[tid, job] = state
        best_states[tid, job] = state
        values[tid, job] = levels[state]
    end
    sync_threads()

    e = zero(T)
    if tid <= f
        vi = values[tid, job]
        row_sum = zero(T)
        @inbounds for j in Int32(1):f
            row_sum += Gt[j, tid] * values[j, job]
        end
        e = vi * row_sum + h[tid, job] * vi
    end
    shmem[tid] = e
    sync_threads()

    offset = nthreads >>> 1
    while offset > 0
        if tid <= offset
            shmem[tid] += shmem[tid + offset]
        end
        sync_threads()
        offset >>>= 1
    end

    if tid == Int32(1)
        best_distance[job] = shmem[1] + const_terms[job]
        step_found[job] = Int32(0)
    end
    return nothing
end

function _potts_step_kernel!(states, values::AbstractArray{T}, best_states, best_distance, step_found, dau_offsets,
    seeds, Gt, gdiag, h, levels, const_terms, f::Int32, k::Int32, step::Int32,
    base_scale::T, batch_rate::T, eoffset::T, use_dau::Bool, single_flip::Bool, directional_noise::Bool) where {T<:AbstractFloat}
    tid = Int32(threadIdx().x)
    job = Int32(blockIdx().x)
    nthreads = Int32(blockDim().x)
    shmem = CuDynamicSharedArray(T, 2 * Int(nthreads) + 1)

    applied = zero(T)
    proposed = Int32(0)
    changed = false
    if tid <= f
        vi = values[tid, job]
        row_sum = zero(T)
        @inbounds for j in Int32(1):f
            row_sum += Gt[j, tid] * values[j, job]
        end
        grad = T(2) * row_sum + h[tid, job]
        scale = use_dau ? base_scale + dau_offsets[job] : base_scale
        noise = scale * (T(2) * _rand_unit(T, seeds[job], step, tid, Int32(23)) - T(1))
        current = states[tid, job]
        proposed = current
        best_score = zero(T)

        if current > Int32(1)
            down = current - Int32(1)
            delta = levels[down] - vi
            true_delta = delta * grad + gdiag[tid] * delta * delta
            score = directional_noise ? true_delta - delta * noise : true_delta + abs(delta) * noise
            if score < best_score
                best_score = score
                proposed = down
            end
        end
        if current < k
            up = current + Int32(1)
            delta = levels[up] - vi
            true_delta = delta * grad + gdiag[tid] * delta * delta
            score = directional_noise ? true_delta - delta * noise : true_delta + abs(delta) * noise
            if score < best_score
                best_score = score
                proposed = up
            end
        end
        changed = proposed != current
    end

    if single_flip
        priority = changed ? _rand_unit(T, seeds[job], step, tid, Int32(37)) : -one(T)
        shmem[tid] = priority
        shmem[Int(nthreads) + Int(tid)] = T(tid)
        sync_threads()

        offset = nthreads >>> 1
        while offset > 0
            if tid <= offset
                left_priority = shmem[tid]
                right_priority = shmem[tid + offset]
                left_index = shmem[Int(nthreads) + Int(tid)]
                right_index = shmem[Int(nthreads) + Int(tid + offset)]
                if right_priority > left_priority || (right_priority == left_priority && right_index < left_index)
                    shmem[tid] = right_priority
                    shmem[Int(nthreads) + Int(tid)] = right_index
                end
            end
            sync_threads()
            offset >>>= 1
        end

        selected = changed && tid == Int32(round(shmem[Int(nthreads) + 1])) && shmem[1] >= zero(T)
        if selected
            states[tid, job] = proposed
            values[tid, job] = levels[proposed]
            applied = one(T)
        end
    elseif changed && (_rand_unit(T, seeds[job], step, tid, Int32(37)) < batch_rate)
        states[tid, job] = proposed
        values[tid, job] = levels[proposed]
        applied = one(T)
    end

    shmem[tid] = applied
    sync_threads()

    offset = nthreads >>> 1
    while offset > 0
        if tid <= offset
            shmem[tid] += shmem[tid + offset]
        end
        sync_threads()
        offset >>>= 1
    end

    if use_dau && tid == Int32(1)
        dau_offsets[job] = shmem[1] > zero(T) ? zero(T) : dau_offsets[job] + eoffset
    end
    sync_threads()

    e = zero(T)
    if tid <= f
        vi = values[tid, job]
        row_sum = zero(T)
        @inbounds for j in Int32(1):f
            row_sum += Gt[j, tid] * values[j, job]
        end
        e = vi * row_sum + h[tid, job] * vi
    end
    shmem[tid] = e
    sync_threads()

    offset = nthreads >>> 1
    while offset > 0
        if tid <= offset
            shmem[tid] += shmem[tid + offset]
        end
        sync_threads()
        offset >>>= 1
    end

    if tid == Int32(1)
        distance = shmem[1] + const_terms[job]
        improved = distance < best_distance[job]
        if improved
            best_distance[job] = distance
            step_found[job] = step
            shmem[2 * Int(nthreads) + 1] = one(T)
        else
            shmem[2 * Int(nthreads) + 1] = zero(T)
        end
    end
    sync_threads()

    if tid <= f && shmem[2 * Int(nthreads) + 1] != zero(T)
        best_states[tid, job] = states[tid, job]
    end
    return nothing
end

function _run_cuda_jobs(G_host, h_host, const_terms_host, levels, seeds::AbstractVector{UInt64};
    optimizer::Symbol,
    num_cycles::Int,
    cycles_scaler::Real,
    noise_ratio::Real,
    noise_stepper::Symbol,
    noise_coupling::Symbol,
    batch_rate::Real,
    eoffset::Real,
    gpuFloat::Type{T},
) where {T<:AbstractFloat}
    f = size(h_host, 1)
    k = length(levels)
    steps = max(1, ceil(Int, cycles_scaler * num_cycles))
    jobs = length(seeds)

    if jobs == 0
        return Any[]
    end

    threads = _threads_for_free_dims(f)
    shmem = (2 * threads + 1) * sizeof(T)
    noise_start = T(Float64(noise_ratio) * MIMOPotts._gradient_bound(G_host, vec(h_host[:, 1]), levels))

    states = CUDA.zeros(Int32, f, jobs)
    values = CUDA.zeros(T, f, jobs)
    best_states = CUDA.zeros(Int32, f, jobs)
    best_distance = CUDA.zeros(T, jobs)
    step_found = CUDA.zeros(Int32, jobs)
    dau_offsets = CUDA.zeros(T, jobs)
    seeds_d = CuArray(UInt64.(seeds))
    Gt = CuArray(T.(transpose(G_host)))
    gdiag_d = CuArray(T.(diag(G_host)))
    h_d = CuArray(T.(h_host))
    levels_d = CuArray(T.(levels))
    const_terms_d = CuArray(T.(const_terms_host))

    CUDA.@sync @cuda threads=threads blocks=jobs shmem=shmem _potts_init_kernel!(
        states, values, best_states, best_distance, step_found,
        seeds_d, Gt, h_d, levels_d, const_terms_d, Int32(f), Int32(k))

    use_dau = optimizer === :batchdau || optimizer === :dau
    single_flip = optimizer === :singleflip || optimizer === :dau
    directional_noise = noise_coupling === :directional
    for step in 1:steps
        base_scale = T(MIMOPotts._noise_scale(noise_stepper, noise_start, steps, step))
        CUDA.@sync @cuda threads=threads blocks=jobs shmem=shmem _potts_step_kernel!(
            states, values, best_states, best_distance, step_found, dau_offsets,
            seeds_d, Gt, gdiag_d, h_d, levels_d, const_terms_d, Int32(f), Int32(k), Int32(step),
            base_scale, T(batch_rate), T(eoffset), use_dau, single_flip, directional_noise)
    end

    best_distance_h = Array(best_distance)
    best_states_h = Array(best_states)
    step_found_h = Array(step_found)

    return [
        (;
            best_distance = Float64(best_distance_h[col]),
            best_free_states = Vector{Int}(best_states_h[:, col]),
            step_found = Int(step_found_h[col]),
            total_steps = steps,
            total_gradient_evals = steps,
        )
        for col in 1:jobs
    ]
end

function _run_cuda_chunk(inst::MIMOPottsInstance, branch::MIMOPottsBranch, prep;
    H,
    y,
    const_offset,
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
    gpuFloat::Type{T},
) where {T<:AbstractFloat}
    f = length(branch.free_indices)
    if f == 0 || isempty(seeds)
        return MIMOPotts._run_potts_subproblem_batch_backend(Val(:cpu), inst, branch, prep;
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
            gpuFloat = gpuFloat,
        )
    end

    G_host, h_vec, const_term = MIMOPotts._subproblem_terms(H, y, branch, const_offset, coupling_cache)
    h_host = repeat(reshape(T.(h_vec), :, 1), 1, length(seeds))
    const_terms = fill(T(const_term), length(seeds))
    return _run_cuda_jobs(G_host, h_host, const_terms, inst.levels, seeds;
        optimizer = optimizer,
        num_cycles = num_cycles,
        cycles_scaler = cycles_scaler,
        noise_ratio = noise_ratio,
        noise_stepper = noise_stepper,
        noise_coupling = noise_coupling,
        batch_rate = batch_rate,
        eoffset = eoffset,
        gpuFloat = gpuFloat,
    )
end

function MIMOPotts._run_potts_subproblem_batch_backend(::Val{:cuda}, inst::MIMOPottsInstance, branch::MIMOPottsBranch, prep;
    H,
    y,
    const_offset,
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
    if !(optimizer in (:batch, :batchdau, :singleflip, :dau))
        error("Unsupported MIMO Potts optimizer $(optimizer). Use :batch, :singleflip, :dau, or :batchdau.")
    end
    if !CUDA.functional()
        error("CUDA.jl is loaded, but no functional CUDA GPU is available.")
    end

    out = Any[]
    sizehint!(out, length(seeds))
    for first in 1:max(1, gpuBatchSize):length(seeds)
        last = min(first + max(1, gpuBatchSize) - 1, length(seeds))
        append!(out, _run_cuda_chunk(inst, branch, prep;
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
            seeds = seeds[first:last],
            gpuFloat = gpuFloat,
        ))
    end
    return out
end


function _fallback_branch_batch(backend_val, inst, branches, active_trials_by_branch, prep;
    H,
    y,
    const_offset,
    coupling_cache,
    optimizer,
    num_cycles,
    cycles_scaler,
    noise_ratio,
    noise_stepper,
    noise_coupling,
    batch_rate,
    eoffset,
    snr_index,
    instance_index,
    seed,
    gpuBatchSize,
    gpuFloat,
)
    out = Vector{Any}(undef, length(branches))
    for (i, branch) in enumerate(branches)
        active_trials = active_trials_by_branch[i]
        seeds = [MIMOPotts._trial_seed(seed, snr_index, instance_index, branch.rank, trial) for trial in active_trials]
        out[i] = MIMOPotts._run_potts_subproblem_batch_backend(backend_val, inst, branch, prep;
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

function MIMOPotts._run_potts_subproblem_branch_batch_backend(::Val{:cuda}, inst::MIMOPottsInstance, branches::AbstractVector{<:MIMOPottsBranch}, active_trials_by_branch::AbstractVector, prep;
    H,
    y,
    const_offset,
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
    gpuFloat::Type{T}=Float32,
) where {T<:AbstractFloat}
    if !(optimizer in (:batch, :batchdau, :singleflip, :dau))
        error("Unsupported MIMO Potts optimizer $(optimizer). Use :batch, :singleflip, :dau, or :batchdau.")
    end
    if isempty(branches)
        return Any[]
    end
    if !CUDA.functional()
        error("CUDA.jl is loaded, but no functional CUDA GPU is available.")
    end

    ref_free = branches[1].free_indices
    ref_fixed = branches[1].fixed_indices
    if any(branch -> branch.free_indices != ref_free || branch.fixed_indices != ref_fixed, branches)
        return _fallback_branch_batch(Val(:cuda), inst, branches, active_trials_by_branch, prep;
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
            snr_index = snr_index,
            instance_index = instance_index,
            seed = seed,
            gpuBatchSize = gpuBatchSize,
            gpuFloat = gpuFloat,
        )
    end

    f = length(ref_free)
    if f == 0
        return _fallback_branch_batch(Val(:cuda), inst, branches, active_trials_by_branch, prep;
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
            snr_index = snr_index,
            instance_index = instance_index,
            seed = seed,
            gpuBatchSize = gpuBatchSize,
            gpuFloat = gpuFloat,
        )
    end

    counts = [length(active_trials) for active_trials in active_trials_by_branch]
    total_jobs = sum(counts)
    if total_jobs == 0
        return [Any[] for _ in branches]
    end

    first_terms = MIMOPotts._subproblem_terms(H, y, branches[1], const_offset, coupling_cache)
    G_host = first_terms[1]
    h_host = Matrix{T}(undef, f, total_jobs)
    const_terms = Vector{T}(undef, total_jobs)
    seeds = Vector{UInt64}(undef, total_jobs)

    job = 1
    for (branch_i, branch) in enumerate(branches)
        _, h_vec, const_term = branch_i == 1 ? first_terms : MIMOPotts._subproblem_terms(H, y, branch, const_offset, coupling_cache)
        for trial in active_trials_by_branch[branch_i]
            h_host[:, job] .= T.(h_vec)
            const_terms[job] = T(const_term)
            seeds[job] = MIMOPotts._trial_seed(seed, snr_index, instance_index, branch.rank, trial)
            job += 1
        end
    end

    flat = Any[]
    sizehint!(flat, total_jobs)
    for first in 1:max(1, gpuBatchSize):total_jobs
        last = min(first + max(1, gpuBatchSize) - 1, total_jobs)
        append!(flat, _run_cuda_jobs(G_host, h_host[:, first:last], const_terms[first:last], inst.levels, seeds[first:last];
            optimizer = optimizer,
            num_cycles = num_cycles,
            cycles_scaler = cycles_scaler,
            noise_ratio = noise_ratio,
            noise_stepper = noise_stepper,
            noise_coupling = noise_coupling,
            batch_rate = batch_rate,
            eoffset = eoffset,
            gpuFloat = gpuFloat,
        ))
    end

    out = Vector{Any}(undef, length(branches))
    first = 1
    for (i, count) in enumerate(counts)
        last = first + count - 1
        out[i] = count == 0 ? Any[] : flat[first:last]
        first = last + 1
    end
    return out
end

end
