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


function pem_sample(state0::State, dat::PEMData, priors0::Prior, settings::Splitting)
    ### Setup
    state = copy(state0)
    # Sampler mutates priors (σ, v, Γ) so each chain needs its own copy
    priors = deepcopy(priors0)
    dyn = Dynamics(state, dat)
    # Set up storage
    storage = storage_start!(state, settings, dyn, priors, priors.grid)
    AV_calc!(state, dat, dyn, priors)
    println("Starting sampling")
    while dyn.ind <= settings.max_ind
        if settings.verbose
            verbose(dyn, state)
        end
        split_inner!(state, dyn, priors, dat, settings)
        store_state!(state, storage, dyn, priors)
    end
    out = sampler_end(storage, dyn, settings)
    println("Final time: ");println(state.t)
    return out  
end

function split_inner!(state::ECMC2, dyn::Dynamics, priors::Prior, dat::PEMData, settings::Splitting)
    if settings.h_rate > 0.0
        grid_update!(state, dyn, dat, priors, priors.grid)
    end
    if rand() < 1 - exp(-settings.r_rate*settings.δ*0.5)
        refresh!(state, dat, dyn, priors)
    end
    update!(state, settings.δ*0.5, priors)
    for i in 1:(settings.thin-1)
        U_grad = ∇U(state, dat, dyn, priors)
        λ = max(0, dot(vcat(state.v[state.active], priors.v), U_grad))
        if rand() < 1 - exp(-settings.δ*λ)
            flip!(state, priors, U_grad)
        end
        update!(state, settings.δ, priors)
        if rand() < 1 - exp(-settings.r_rate*settings.δ)
            refresh!(state, dat, dyn, priors)
        end
    end
    U_grad = ∇U(state, dat, dyn, priors)
    λ = max(0, dot(vcat(state.v[state.active], priors.v), U_grad))
    if rand() < 1 - exp(-settings.δ*λ)
        flip!(state, priors, U_grad)
    end
    update!(state, settings.δ*0.5, priors)
    if rand() < 1 - exp(-settings.r_rate*settings.δ*0.5)
        refresh!(state, dat, dyn, priors)
    end
end

function sampler_end(storage::Storage, dyn::Dynamics, settings::Settings)
    N = dyn.ind - 1
    # Drop the Inf padding beyond the largest number of knots used
    J_max = maximum(findlast(isfinite, storage.s_loc[:,i]) for i in 1:N)
    θ = storage.x[:,1:J_max,1:N].*reshape(storage.σ[1:N], 1, 1, N)
    # α₀ in column 1 isn't scaled by σ
    θ[:,1,:] = storage.x[:,1,1:N]
    out = Dict("Sk_x" => storage.x[:,1:J_max,1:N], "Sk_θ" => θ,
                "Sk_v" => storage.v[:,1:J_max,1:N], "Sk_s" => storage.s[:,1:J_max,1:N], "Sk_t" => storage.t[1:N],
                "Sk_ω" => storage.ω[:,1:N], "Sk_σ" => storage.σ[1:N], "Sk_Γ" => storage.Γ[1:N], "Sk_γ" => storage.γ[1:N], "Sk_J" => storage.J[1:N], "Sk_s_loc" => storage.s_loc[1:J_max,1:N])
    return out
end

function verbose(dyn::Dynamics, state::State)
    println("----------------------")
    print("Iteration: ");print(dyn.ind);print("\n");
    println(state.t)
    println("----------------------")
end
