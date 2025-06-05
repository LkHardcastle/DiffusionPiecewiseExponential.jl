include("Types.jl")
include("Potential.jl")
include("Updating.jl")
include("SplitMerge.jl")
include("HyperUpdates.jl")
include("Storage.jl")
include("Extrapolation.jl")
include("RJ.jl")
include("Metropolis.jl")

function pem_fit(state0::State, dat::PEMData, priors::Prior, settings::Exact, test_times, burn_in::Int64)
    out1 = pem_sample(state0, dat, priors, settings)
    out2 = pem_sample(state0, dat, priors, settings)
    test_smp1 = cts_transform(cumsum(out1["Smp_θ"], dims = 2), out1["Smp_s_loc"], test_times)
    test_smp2 = cts_transform(cumsum(out2["Smp_θ"], dims = 2), out2["Smp_s_loc"], test_times)
    rhat_ = []
    ess_ = []
    for i in eachindex(test_times)
        diag = MCMCDiagnosticTools.ess_rhat(vcat(test_smp1[1,i,burn_in:end], test_smp2[1,i,burn_in:end]))
        push!(ess_, diag[1])
        push!(rhat_, diag[2])
    end
    diag = MCMCDiagnosticTools.ess_rhat(vcat(out1["Smp_σ"], out2["Smp_σ"]))
    push!(ess_, diag[1])
    push!(rhat_, diag[2])
    return out1, out2, rhat_, ess_
end

function pem_fit(state0::State, dat::PEMData, priors::Prior, settings::Splitting, test_times, burn_in::Int64)
    out1 = pem_sample(state0, dat, priors, settings)
    out2 = pem_sample(state0, dat, priors, settings)
    test_smp1 = cts_transform(cumsum(out1["Sk_θ"], dims = 2), out1["Sk_s_loc"], test_times)
    test_smp2 = cts_transform(cumsum(out2["Sk_θ"], dims = 2), out2["Sk_s_loc"], test_times)
    rhat_ = []
    ess_ = []
    for i in eachindex(test_times)
        diag = MCMCDiagnosticTools.ess_rhat(vcat(test_smp1[1,i,burn_in:end], test_smp2[1,i,burn_in:end]))
        push!(ess_, diag[1])
        push!(rhat_, diag[2])
    end
    diag = MCMCDiagnosticTools.ess_rhat(vcat(out1["Sk_σ"][burn_in:end], out2["Sk_σ"][burn_in:end]))
    push!(ess_, diag[1])
    push!(rhat_, diag[2])
    return out1, out2, rhat_, ess_
end


function pem_sample(state0::State, dat::PEMData, priors::Prior, settings::Splitting)
    ### Setup
    state = copy(state0)
    times = Times(state, settings, priors)
    dyn = Dynamics(state, dat)
    # Set up storage 
    if settings.skel == false
        dyn.ind = 2
    end
    storage = storage_start!(state, settings, dyn, priors, priors.grid)
    AV_calc!(state, dyn, priors)
    println("Starting sampling")
    while dyn.ind <= settings.max_ind
        if settings.verbose
            verbose(dyn, state)
        end
        split_inner!(state, dyn, priors, dat, times, settings)
        store_state!(state, storage, dyn, priors; skel = settings.skel)
    end
    out = sampler_end(storage, dyn, settings)
    println("Final time: ");println(state.t)
    return out  
end

function pem_sample(state0::State, dat::PEMData, priors::Prior, settings::Exact)
    ### Setup
    state = copy(state0)
    times = Times(state, settings, priors)
    dyn = Dynamics(state, dat)
    # Set up storage 
    if settings.skel == false
        dyn.ind = 2
    end
    storage = storage_start!(state, settings, dyn, priors, priors.grid)
    AV_calc!(state, dyn, priors)
    println("Starting sampling")
    while dyn.ind < settings.max_ind
        if settings.verbose
            verbose(dyn, state)
        end
        sampler_inner!(state, dyn, priors, dat, times, settings)
        store_state!(state, storage, dyn, priors; skel = settings.skel)
        store_smps!(state, storage, dyn, times, priors)
        stop = sampler_stop(state, dyn, settings)
        if stop
            break
        end
    end
    out = sampler_end(storage, dyn, settings)
    return out  
end

function sampler_inner!(state::Union{ECMC2, BPS}, dyn::Dynamics, priors::Prior, dat::PEMData, times::Times, settings::Settings)
    ## Evaluate potential at current point to get constants
    Uθt, ∂U = U_new!(state, dyn, priors)
    ## Get next deterministic event and evaluate at that point
    get_time!(dyn, times)
    if !isinf(dyn.t_det)
        U_det, ∂U_det = U_eval(state, dyn.t_det - state.t, dyn, priors)
    else
        U_det, ∂U_det = Inf, Inf
    end
    ## If potential decreasing at that point jump to it and break
    if ∂U_det < 0.0
        t_switch = 0.0
        t_event = dyn.t_det - state.t
    else
        ## Elseif potential decreasing at initialpoint line search for point where gradient begins to increase
        t_switch = 0.0
        if ∂U < 0.0
            t_switch = find_zero(x -> U_eval(state, x + t_switch, dyn, priors)[2], (0.0, dyn.t_det - state.t), A42())
            Uθt, ∂U = U_eval(state, t_switch, dyn, priors)
        end
        ## Generate uniform r.v and check if deterministic time is close enough - if so break
        V = rand()
        if U_det - Uθt < -log(V)
            t_event = dyn.t_det - state.t - t_switch
        else
            dyn.next_event = 5
            ## Generate next time via time-scale transformation
            if isinf(U_det)
                t_event = find_zero(x -> U_eval(state, x + t_switch, dyn, priors)[1] - Uθt + log(V), (0.0, 1), A42(); atol = 0.1)
            else
                t_event = find_zero(x -> U_eval(state, x + t_switch, dyn, priors)[1] - Uθt + log(V), (0.0, dyn.t_det - state.t - t_switch), A42())
            end
        end
    end
    for j in axes(state.x, 1)
        t_event, t_switch = diffusion_time!(state, priors, dyn, priors.diff[j], t_event + t_switch, 0.0, j)
    end
    update!(state, t_switch + t_event)
    event!(state, dat, dyn, priors, times, settings)
end

function split_inner!(state::Union{ECMC2, BPS}, dyn::Dynamics, priors::Prior, dat::PEMData, times::Times, settings::Splitting)
    if settings.h_rate > 0.0
        grid_update!(state, dyn, dat, priors, priors.grid)
    end
    if rand() < 1 - exp(-settings.r_rate*settings.δ*0.5)
        refresh!(state, dat, dyn, priors)
    end
    update!(state, settings.δ*0.5, priors)
    for i in 1:(settings.thin-1)
        λ = max(0, dot(vcat(state.v[state.active], priors.v), vcat(∇U(state, dat, dyn, priors), ∇σ(state, dat, dyn, priors, priors.σ))))
        if rand() < 1 - exp(-settings.δ*λ)
            flip!(state, dat, dyn, priors, settings)
        end
        update!(state, settings.δ, priors)
        if rand() < 1 - exp(-settings.r_rate*settings.δ)
            refresh!(state, dat, dyn, priors)
        end
    end
    λ = max(0, dot(vcat(state.v[state.active], priors.v), vcat(∇U(state, dat, dyn, priors), ∇σ(state, dat, dyn, priors, priors.σ))))
    if rand() < 1 - exp(-settings.δ*λ)
        flip!(state, dat, dyn, priors, settings)
    end
    update!(state, settings.δ*0.5, priors)
    if rand() < 1 - exp(-settings.r_rate*settings.δ*0.5)
        refresh!(state, dat, dyn, priors)
    end
end

function get_time!(dyn::Dynamics, times::Times)
    dyn.t_det, dyn.next_event = findmin([times.next_split, times.next_merge, times.refresh[1], times.hyper[1]])
end


function sampler_stop(state::State, dyn::Dynamics, settings::Settings)
    if dyn.ind >= settings.max_ind
        println("Stopping for max_iter");println("Final time: ");println(state.t)
        return true
    end
    if dyn.smp_ind >= settings.max_smp
        println("Stopping for max_smp");println("Final time: ");println(state.t)
        return true
    end
    if state.t >= settings.max_time
        println("Stopping for max_time");println("Final time: ");println(state.t)
        return true
    end
    return false
end

function Times(state::State, settings::Settings, priors::Prior)
    merge_curr = Inf
    j_curr = CartesianIndex(0,0)
    for j in state.active
        if j[2] > 1
            if rand() < priors.p_split
                merge_cand = merge_time(state, j, priors)
                if merge_cand < merge_curr
                    merge_curr = copy(merge_cand)
                    j_curr = CartesianIndex(j[1],j[2])
                end
            end
        end
    end
    if priors.p_split > 0.0
        T_split = Inf
    else 
        T_split = Inf
    end
    T_smp = exp_vector(settings, settings.smp_rate)
    T_h = exp_vector(settings, settings.h_rate)
    T_ref = exp_vector(settings, settings.r_rate)
    return Times(T_split, merge_curr, j_curr, T_ref, T_h, T_smp)
end


function exp_vector(settings::Settings, rate::Float64)
    if rate > 0.0
        out = cumsum(rand(Exponential(1/rate), trunc(Int, rate*settings.max_time + 100)))
        if out[end] < settings.max_time
            while out[end] < settings.max_time
                push!(out, out[end] + rand(Exponential(1/rate)))
            end
        end
    else
        out = [Inf]
    end
    return out
end

function sampler_end(storage::Storage, dyn::Dynamics, settings::Settings)
    if settings.skel
        out = Dict("Sk_x" => storage.x[:,:,1:(dyn.ind-1)], "Sk_θ" => storage.x[:,:,1:(dyn.ind-1)].*reshape(repeat(storage.σ[1:(dyn.ind-1)], inner = size(storage.x,1)*size(storage.x,2), outer = 1), size(storage.x,1),size(storage.x,2),dyn.ind-1), 
                    "Sk_v" => storage.v[:,:,1:(dyn.ind-1)], "Sk_s" => storage.s[:,:,1:(dyn.ind-1)], "Sk_t" => storage.t[1:(dyn.ind-1)], 
                    "Sk_ω" => storage.ω[:,1:(dyn.ind-1)], "Sk_σ" => storage.σ[1:(dyn.ind-1)], "Sk_Γ" => storage.Γ[1:(dyn.ind-1)], "Sk_γ" => storage.γ[1:(dyn.ind-1)], "Sk_J" => storage.J[1:(dyn.ind-1)], "Sk_s_loc" => storage.s_loc[:,1:(dyn.ind-1)],
                    "Smp_x" => storage.x_smp[:,:,1:(dyn.smp_ind-1)], "Smp_θ" => storage.x_smp[:,:,1:(dyn.smp_ind-1)].*reshape(repeat(storage.σ_smp[1:(dyn.smp_ind-1)], inner = size(storage.x_smp,1)*size(storage.x_smp,2), outer = 1), size(storage.x_smp,1),size(storage.x_smp,2),dyn.smp_ind-1), 
                    "Smp_v" => storage.v_smp[:,:,1:(dyn.smp_ind-1)], "Smp_s" => storage.s_smp[:,:,1:(dyn.smp_ind-1)], 
                    "Smp_t" => storage.t_smp[1:(dyn.smp_ind-1)], "Smp_ω" => storage.ω_smp[:,1:(dyn.smp_ind-1)], "Smp_σ" => storage.σ_smp[1:(dyn.smp_ind-1)], "Smp_Γ" => storage.Γ_smp[1:(dyn.smp_ind-1)], "Smp_γ" => storage.γ_smp[1:(dyn.smp_ind-1)], "Smp_J" => storage.J_smp[1:(dyn.smp_ind-1)], "Smp_s_loc" => storage.s_loc_smp[:,1:(dyn.smp_ind-1)],
                    "Smp_trans" => transform_smps(storage.x_smp[:,:,1:(dyn.smp_ind-1)]), "Eval" => dyn.sampler_eval) 
    else
        out = Dict("Smp_x" => storage.x_smp[:,:,1:(dyn.smp_ind-1)], "Smp_θ" => storage.x_smp[:,:,1:(dyn.smp_ind-1)].*reshape(storage.σ_smp[:,1:(dyn.smp_ind-1)], size(storage.σ_smp[:,1:(dyn.smp_ind-1)],1), 1, size(storage.σ_smp[:,1:(dyn.smp_ind-1)],2)), "Smp_v" => storage.v_smp[:,:,1:(dyn.smp_ind-1)], "Smp_s" => storage.s_smp[:,:,1:(dyn.smp_ind-1)], 
                    "Smp_t" => storage.t_smp[1:(dyn.smp_ind-1)], "Smp_ω" => storage.ω_smp[:,1:(dyn.smp_ind-1)], "Smp_σ" => storage.σ_smp[:,1:(dyn.smp_ind-1)], "Smp_Γ" => storage.Γ_smp[1:(dyn.smp_ind-1)], "Smp_γ" => storage.γ_smp[1:(dyn.smp_ind-1)],
                    "Smp_J" => storage.J_smp[1:(dyn.smp_ind-1)], "Smp_s_loc" => storage.s_loc_smp[:,1:(dyn.smp_ind-1)],
                    "Smp_trans" => transform_smps(storage.x_smp[:,:,1:(dyn.smp_ind-1)]),"Eval" => dyn.sampler_eval) 

    end
    return out
end

function verbose(dyn::Dynamics, state::State)
    println("----------------------")
    print("Iteration: ");print(dyn.ind);print("\n");
    println(dyn.next_event)
    println(state.t)
    println("----------------------")
end