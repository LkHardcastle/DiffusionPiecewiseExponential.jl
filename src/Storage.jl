


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
                        zeros(settings.max_ind+ 1))
    store_state!(state, storage, dyn, priors)
    return storage
end

function storage_start!(state::State, settings::Settings, dyn::Dynamics, priors::Prior, grid::Cts)
    storage = Storage(fill(Inf,size(state.x, 1), grid.max_points, settings.max_ind + 1),
                        fill(Inf,size(state.v, 1),grid.max_points, settings.max_ind + 1), 
                        fill(false,size(state.s, 1),grid.max_points, settings.max_ind + 1),
                        fill(Inf,grid.max_points, settings.max_ind + 1),
                        zeros(settings.max_ind + 1),
                        zeros(settings.max_ind + 1),
                        fill(Inf, size(state.x,1), settings.max_ind + 1),
                        fill(Inf, settings.max_ind + 1),
                        zeros(settings.max_ind+ 1),
                        zeros(settings.max_ind+ 1))
    store_state!(state, storage, dyn, priors)
    return storage
end


function store_state!(state::State, storage::Storage, dyn::Dynamics, priors::Prior)
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
