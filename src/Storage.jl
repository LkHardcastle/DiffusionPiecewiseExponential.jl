


function storage_start!(state::State, settings::Settings, dyn::Dynamics, priors::Prior, grid::Fixed)
    storage = Storage(fill(Inf,size(state.x, 1),size(state.x, 2), settings.max_ind + 1),
                        fill(Inf,size(state.v, 1),size(state.v, 2), settings.max_ind + 1),  
                        fill(false,size(state.s, 1),size(state.s, 2), settings.max_ind + 1),
                        fill(Inf,size(state.s, 2), settings.max_ind + 1),
                        zeros(settings.max_ind+ 1),
                        zeros(settings.max_ind+ 1),
                        fill(Inf, size(state.x,1), settings.max_ind + 1),
                        fill(Inf, settings.max_ind + 1),
                        zeros(settings.max_ind+ 1),
                        zeros(settings.max_ind+ 1),
                        fill(Inf,size(state.x, 1),size(state.x, 2), settings.max_smp + 1),
                        fill(Inf,size(state.v, 1),size(state.v, 2), settings.max_smp + 1), 
                        fill(false,size(state.s, 1),size(state.s, 2), settings.max_smp + 1),
                        fill(Inf,size(state.s, 2), settings.max_smp + 1),
                        zeros(settings.max_smp + 1),
                        zeros(settings.max_smp + 1),
                        fill(Inf, size(state.x, 1), settings.max_smp + 1),
                        fill(Inf, settings.max_smp + 1),
                        zeros(settings.max_smp + 1),
                        zeros(settings.max_smp + 1))
    store_state!(state, storage, dyn, priors; skel = settings.skel)
    return storage
end

function storage_start!(state::State, settings::Settings, dyn::Dynamics, priors::Prior, grid::Union{Cts,RJ})
    storage = Storage(fill(Inf,size(state.x, 1), grid.max_points, settings.max_ind + 1),
                        fill(Inf,size(state.v, 1),grid.max_points, settings.max_ind + 1), 
                        fill(false,size(state.s, 1),grid.max_points, settings.max_ind + 1),
                        fill(Inf,grid.max_points, settings.max_ind + 1),
                        zeros(settings.max_ind + 1),
                        zeros(settings.max_ind + 1),
                        fill(Inf, size(state.x,1), settings.max_ind + 1),
                        fill(Inf, settings.max_ind + 1),
                        zeros(settings.max_ind+ 1),
                        zeros(settings.max_ind+ 1),
                        fill(Inf,size(state.x, 1),grid.max_points, settings.max_smp + 1),
                        fill(Inf,size(state.v, 1),grid.max_points, settings.max_smp + 1), 
                        fill(false,size(state.s, 1),grid.max_points, settings.max_smp + 1),
                        fill(Inf,grid.max_points, settings.max_smp + 1),
                        zeros(settings.max_smp + 1),
                        zeros(settings.max_smp + 1),
                        fill(Inf, size(state.x, 1), settings.max_smp + 1),
                        fill(Inf, settings.max_smp + 1),
                        zeros(settings.max_smp + 1),
                        zeros(settings.max_smp + 1))
    store_state!(state, storage, dyn, priors; skel = settings.skel)
    return storage
end


function store_state!(state::State, storage::Storage, dyn::Dynamics, priors::Prior; skel = true)
    if !skel
        dyn.ind -= 1
    end
    range = 1:size(state.s_loc,1)
    storage.x[:,range,dyn.ind] = copy(state.x)
    storage.v[:,range,dyn.ind] = copy(state.v)
    storage.s[:,range,dyn.ind] = copy(state.s)
    storage.s_loc[range,dyn.ind] = copy(state.s_loc)
    storage.J[dyn.ind] = copy(state.J)
    storage.t[dyn.ind] = copy(state.t)
    storage.ω[:,dyn.ind] = copy(priors.ω.ω)
    storage.σ[dyn.ind] = copy(priors.σ.σ)
    storage.Γ[dyn.ind] = copy(priors.grid.Γ)
    storage.γ[dyn.ind] = copy(priors.grid.γ)
    dyn.ind += 1
end

function store_smps!(state::State, storage::Storage, dyn::Dynamics, times::Times, priors::Prior)
    ind_end = findfirst(times.smps .> state.t)
    if isnothing(ind_end)
        ind_end = size(times.smps,1)
    end
    range = 1:size(state.s_loc,1)
    ind_end -= 1
    t_old = storage.t[dyn.ind - 2]
    x_old = storage.x[:,range,dyn.ind - 2]
    v_old = storage.v[:,range,dyn.ind - 2]
    s_old = storage.s[:,range,dyn.ind - 2]
    s_loc_old = storage.s_loc[range, dyn.ind - 2]
    J_old = storage.J[dyn.ind - 2]
    for i in 1:ind_end
        storage.t_smp[dyn.smp_ind] = times.smps[i]
        storage.x_smp[:,range,dyn.smp_ind] = x_old + v_old*(times.smps[i] - t_old)
        storage.v_smp[:,range,dyn.smp_ind] = copy(v_old)
        storage.s_smp[:,range,dyn.smp_ind] = copy(s_old)
        storage.ω_smp[:,dyn.smp_ind] = copy(priors.ω.ω)
        storage.σ_smp[dyn.smp_ind] = copy(priors.σ.σ) 
        storage.Γ_smp[dyn.smp_ind] = copy(priors.grid.Γ)
        storage.γ_smp[dyn.smp_ind] = copy(priors.grid.γ)
        storage.J_smp[dyn.smp_ind] = J_old
        storage.s_loc_smp[range, dyn.smp_ind] = s_loc_old
        dyn.smp_ind += 1
    end
    deleteat!(times.smps, 1:ind_end)
end
