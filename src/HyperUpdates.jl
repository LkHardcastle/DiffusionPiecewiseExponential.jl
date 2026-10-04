
function grid_update!(state::State, dyn::Dynamics, dat::PEMData, priors::Prior, Grid::Fixed)
    
end

function pois_gen(grid::CtsPois, priors::Prior, state::State, window::Float64)
    Pois_new = []
    weight_vec = []
    for k in axes(state.x,1)
        push!(Pois_new, rand(Poisson(window*priors.grid.Γ*(1 - priors.ω.ω[k]))))
        push!(weight_vec, (1 - priors.ω.ω[k]))
    end
    return Pois_new, weight_vec
end

function pois_gen(grid::CtsNB, priors::Prior, state::State, window::Float64)
    Pois_new = []
    weight_vec = []
    # Γ∣J+K ∼ Gamma(α + J + K, β + 1)
    #priors.grid.Γ = rand(Gamma(state.J + priors.grid.α, 1/(priors.grid.β + 1)))
    # Γ ∼ Gamma(α, β) with rate β and the active knots in row k form a PPP(ω_k Γ) on (min_time, max_time),
    # so with K active knots Γ∣K ∼ Gamma(α + K, β + window*Σω_k); column 1 holds the initial values, not knots
    priors.grid.Γ = rand(Gamma(priors.grid.α + sum(state.s[:,2:end]), 1/(priors.grid.β + window*sum(priors.ω.ω))))
    for k in axes(state.x,1)
        push!(Pois_new, rand(Poisson(window*priors.grid.Γ*(1 - priors.ω.ω[k]))))
        push!(weight_vec, (1 - priors.ω.ω[k]))
    end
    return Pois_new, weight_vec
end

function pois_gen(grid::CtsNB2, priors::Prior, state::State, window::Float64)
    Pois_new = []
    weight_vec = []
    # Γ∣J+K ∼ Gamma(α + J + K, β + 1)
    priors.grid.β = rand(Gamma(priors.grid.a, priors.grid.Γ + priors.grid.b))
    priors.grid.Γ = rand(Gamma(sum(state.s) + priors.grid.α, priors.ω.ω[1]/(priors.grid.β + 1)))
    for k in axes(state.x,1)
        push!(Pois_new, rand(Poisson(window*priors.grid.Γ*(1 - priors.ω.ω[k]))))
        push!(weight_vec, (1 - priors.ω.ω[k]))
    end
    return Pois_new, weight_vec
end

function pois_gen(grid::CtsPoisRE, priors::Prior, state::State, window::Float64)
    Pois_new = []
    weight_vec = []
    # Γ∣J+K ∼ Gamma(α + J + K, β + 1)
    Γ_prop = exp(log(priors.grid.Γ) + rand(Normal(0.0,priors.grid.h)))
    A = sum(state.s)*(log(Γ_prop) - log(priors.grid.Γ)) - priors.ω.ω[1]*(Γ_prop - priors.grid.Γ) + logpdf(Normal(priors.grid.μ,priors.grid.σ),Γ_prop) - logpdf(Normal(priors.grid.μ,priors.grid.σ),priors.grid.Γ)
    if rand() < min(1,exp(A))
        priors.grid.Γ = copy(Γ_prop)
    end
    for k in axes(state.x,1)
        push!(Pois_new, rand(Poisson(window*priors.grid.Γ*(1 - priors.ω.ω[k]))))
        push!(weight_vec, (1 - priors.ω.ω[k]))
    end
    return Pois_new, weight_vec
end

function pois_gen(grid::CtsNBmix, priors::Prior, state::State, window::Float64)
    Pois_new = []
    weight_vec = []
    # Γ∣J+K ∼ Gamma(α + J + K, β + 1)
    #error("Update indicators here")
    priors.grid.Γ = priors.grid.γ*rand(Gamma(sum(state.s) + priors.grid.α1, priors.ω.ω[1]/(priors.grid.β1 + 1))) + (1-priors.grid.γ)*rand(Gamma(sum(state.s) + priors.grid.α2, priors.ω.ω[1]/(priors.grid.β2 + 1)))
    for k in axes(state.x,1)
        push!(Pois_new, rand(Poisson(window*priors.grid.Γ*(1 - priors.ω.ω[k]))))
        push!(weight_vec, (1 - priors.ω.ω[k]))
    end
    A =  logpdf(Gamma(sum(state.s) + priors.grid.α2, priors.ω.ω[1]/(priors.grid.β2 + 1)), priors.grid.Γ) - logpdf(Gamma(sum(state.s) + priors.grid.α1, priors.ω.ω[1]/(priors.grid.β1 + 1)), priors.grid.Γ)
    if rand() < min(1, exp(priors.grid.γ*A - (1-priors.grid.γ)*A))
        priors.grid.γ = mod(priors.grid.γ + 1, 2)
    end
    return Pois_new, weight_vec
end

function grid_update!(state::State, dyn::Dynamics, dat::PEMData, priors::Prior, grid::Cts)
    # Column j covers [s_loc[j-1], s_loc[j]), so its increment is the jump at its left end s_loc[j-1]
    s_left = vcat(0.0, state.s_loc[1:(end - 1)])
    rem_ind = findall(sum.(eachcol(state.s)) .!= 0.0)
    state.x, state.v, state.s, state.g, s_left = state.x[:,rem_ind], state.v[:,rem_ind], state.s[:,rem_ind], state.g[:,rem_ind], s_left[rem_ind]
    J_curr = sum(state.s)
    window = priors.grid.max_time - dyn.min_time
    Pois_new, weight_vec = pois_gen(grid, priors, state, window)
    J_new = min(sum(Pois_new), priors.grid.max_points - J_curr)
    weight_vec = weight_vec/sum(weight_vec)
    J_row = rand(Categorical(weight_vec), J_new)
    J_loc = rand(Uniform(dyn.min_time, priors.grid.max_time), J_new)
    g_new = fill(false, size(state.x, 1), J_new)
    for i in 1:J_new
        g_new[J_row[i], i] = true
    end
    # A new candidate starts a zero column at its location, splitting the column it falls in
    ind_new = sortperm(vcat(s_left, J_loc))
    zero_mat = zeros(size(state.x, 1),J_new)
    s_left = vcat(s_left, J_loc)[ind_new]
    # The last column runs to ∞, so its right end only has to keep s_loc sorted
    state.s_loc = vcat(s_left[2:end], max(state.s_loc[end], priors.grid.max_time))
    state.x = hcat(state.x, zero_mat)[:,ind_new]
    state.v = hcat(state.v, zero_mat)[:,ind_new]
    state.s = hcat(state.s, fill(false, size(zero_mat)))[:,ind_new]
    state.g = hcat(state.g, g_new)[:,ind_new]
    state.active = findall(state.s)
    state.J = length(state.s_loc)
    dat_update!(state, dyn, dat)
end

function dat_update!(state::State, dyn::Dynamics, dat::PEMData)
    L = size(dat.UQ, 2)
    J = size(state.s_loc,1)
    W = zeros(L,J)
    δ = zeros(L,J)
    d = zeros(Int, length(dat.y))
    for i in eachindex(dat.y)
        if isnothing(findfirst(state.s_loc .> dat.y[i]))
            d[i] = J
        else
            d[i] = findfirst(state.s_loc .> dat.y[i])
        end
    end
    for l in 1:L
        yl = dat.y[findall(dat.grp .== l)]
        dl = d[findall(dat.grp .== l)]
        δl = dat.cens[findall(dat.grp .== l)]
        for j in 1:J
            if j == 1
                sj1 = 0.0
            else
                sj1 = state.s_loc[j-1]
            end
            W[l,j] = sum(yl[findall(dl .== j)]) .- length(findall(dl .== j))*sj1 + length(findall(dl .> j))*(state.s_loc[j] - sj1)
            δ[l,j] = length(intersect(findall(δl .== 1), findall(dl .== j)))
        end
    end
    dyn.W = copy(W)
    dyn.δ = copy(δ)
end
