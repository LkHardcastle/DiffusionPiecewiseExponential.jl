
function AV_calc!(state::State, dyn::Dynamics, priors::Prior)
    x = copy(state.x)
    #x[:,1] .= x[:,1]./priors.σ.σ
    A = cumsum(x, dims = 2)
    dyn.A = transpose(dat.UQ)*A
    v = copy(state.v)
    #v[:,1] .= v[:,1]./priors.σ.σ
    V = cumsum(v, dims = 2)
    dyn.V = transpose(dat.UQ)*V
end

function U_new!(state::State, dyn::Dynamics, priors::Prior)
    AV_calc!(state, dyn, priors)
    U_, ∂U_ = U_eval(state, 0.0, dyn, priors)
    return U_, ∂U_
end

function U_eval(state::State, t::Float64, dyn::Dynamics, priors::Prior)
    θ = (dyn.A .+ t.*dyn.V).*priors.σ.σ
    U_ = sum((exp.(θ).*dyn.W .- dyn.δ.*θ)) 
    ∂U_ = sum(priors.σ.σ.*dyn.V.*(exp.(θ).*dyn.W .- dyn.δ)) 
    Σθ = cumsum(state.x .+ t.*state.v, dims = 2).*priors.σ.σ
    Σv = cumsum(state.v, dims = 2)
    for j in axes(state.x, 1)
        U_, ∂U_ = U_prior(state, t, j, Σθ, Σv, U_, ∂U_, priors)
    end
    return U_, ∂U_
end

function U_eval_rj(state::State, t::Float64, dyn::Dynamics, priors::Prior)
    U_ = sum((exp.(dyn.A).*dyn.W .- dyn.δ.*dyn.A)) 
    active_j = filter(idx -> idx[1] == j, state.active)
    for k in active_j
        if k != active_j[1]
            U_ -= logpdf(Normal(0.0, 1), (state.x[k] + state.v[k]*t)*priors.σ.σ)
        else
            U_ -= logpdf(Normal(0.0, priors.σ0), (state.x[k] + state.v[k]*t)*priors.σ.σ)
        end
    end
    return U_, ∂U_
end

function U_prior(state::State, t::Float64, j::Int64, Σθ::Matrix{Float64}, Σv::Matrix{Float64}, U_::Float64, ∂U_::Float64, priors::BasicPrior)
    #error("Not ready yet")
    μθ = drift_U(Σθ[j,:], priors.diff[j])
    ∂μθ = drift_deriv_t(Σθ[j,:], priors.diff[j])
    active_j = filter(idx -> idx[1] == j, state.active)
    for k in active_j
        if k != active_j[1]
            U_ -= logpdf(Normal(0.0, 1), (state.x[k] + state.v[k]*t)*priors.σ.σ)
            #U_ += -log(1 + tanh(μθ[k[2]-1]*(state.x[k] + state.v[k]*t)*priors.σ.σ))
            #∂U_ += state.v[k]*(state.x[k] + state.v[k]*t) 
            #∂U_ += -2*priors.σ.σ*(Σv[k[1],k[2] - 1]*(state.x[k] + state.v[k]*t)*∂μθ[k[2]-1] + priors.σ.σ*state.v[k]*μθ[k[2]-1])/(exp(2*(state.x[k] + state.v[k]*t)*μθ[k[2]-1]*priors.σ.σ) + 1)
        else
            U_ -= logpdf(Normal(0.0, priors.σ0), (state.x[k] + state.v[k]*t)*priors.σ.σ)
            #∂U_ += state.v[k]*(state.x[k] + state.v[k]*t)
        end
    end
    return U_, ∂U_
end

function U_prior(state::State, t::Float64, j::Int64, Σθ::Matrix{Float64}, Σv::Matrix{Float64}, U_::Float64, ∂U_::Float64, priors::EulerMaruyama)
    error("Not ready yet")
    μθ = drift_U(Σθ[j,:], priors.diff[j])
    ∂μθ = drift_deriv_t(Σθ[j,:], priors.diff[j])
    active_j = filter(idx -> idx[1] == j, state.active)
    for k in active_j
        if k != active_j[1]
            U_ -= logpdf(Normal(μθ[k[2]-1]*priors.σ.σ^2, priors.σ.σ), state.x[k] + state.v[k]*t)
            ∂U_ += (1/(priors.σ.σ^2))*(state.x[k] + state.v[k]*t - μθ[k[2]-1]*priors.σ.σ^2)*(state.v[k] - Σv[k[1],k[2] - 1]*∂μθ[k[2]-1]*priors.σ.σ^2) 
        else
            U_ -= logpdf(Normal(0.0, priors.σ0), state.x[k] + state.v[k]*t)
            ∂U_ += (state.v[k]/(priors.σ0^2))*(state.x[k] + state.v[k]*t)
        end
    end
    return U_, ∂U_
end

function ∇U(state::State, dat::PEMData, dyn::Dynamics, priors::BasicPrior)
    ∇U_out = zeros(size(state.active))
    AV_calc!(state, dyn, priors)
    # L x J matrix
    U_ind = reverse(cumsum(reverse(exp.(dyn.A.*priors.σ.σ).*dyn.W .- dyn.δ, dims = 2), dims = 2), dims = 2)
    # Convert to p x J matrix
    U_ind = (dat.UQ*U_ind).*priors.σ.σ
    ∇U_out = U_ind[state.active]
    x = copy(state.x)
    #x[:,1] = x[:,1]./priors.σ.σ
    Σθ = cumsum(x, dims = 2).*priors.σ.σ
    μθ = Vector{Vector{Float64}}()
    ∂μθ = Vector{Array{Float64}}()
    for j in axes(state.x, 1)
        push!(μθ, drift(Σθ[j,:], state.s_loc, priors.diff[j]))
        push!(∂μθ, drift_deriv(Σθ[j,:], state.s_loc, priors.diff[j]))
    end
    for i in eachindex(∇U_out)
        ∇U_out[i] += prior_add(state, priors, state.active[i])
        ∇U_out[i] += drift_add(x.*priors.σ.σ, μθ[state.active[i][1]], ∂μθ[state.active[i][1]], priors.diff[state.active[i][1]], state.active[i], priors.σ.σ)
    end
    return ∇U_out
end

function ∇U(state::State, dat::PEMData, dyn::Dynamics, priors::EulerMaruyama)
    ∇U_out = zeros(size(state.active))
    AV_calc!(state, dyn, priors)
    # L x J matrix
    U_ind = reverse(cumsum(reverse(exp.(dyn.A.*priors.σ.σ).*dyn.W .- dyn.δ, dims = 2), dims = 2), dims = 2)
    # Convert to p x J matrix
    U_ind = (dat.UQ*U_ind).*priors.σ.σ
    ∇U_out = U_ind[state.active]
    x = copy(state.x)
    #x[:,1] = x[:,1]./priors.σ.σ
    Σθ = cumsum(x, dims = 2).*priors.σ.σ
    μθ = Vector{Vector{Float64}}()
    ∂μθ = Vector{Array{Float64}}()
    for j in axes(state.x, 1)
        push!(μθ, drift(Σθ[j,:], state.s_loc, priors.diff[j]))
        push!(∂μθ, drift_deriv(Σθ[j,:], state.s_loc, priors.diff[j]))
    end
    for i in eachindex(∇U_out)
            ∇U_out[i] += prior_EM(state, μθ[state.active[i][1]], ∂μθ[state.active[i][1]], priors, state.active[i])
    end
    return ∇U_out
end

function ∇σ(state::State, dat::PEMData, dyn::Dynamics, priors::Prior, σ::Union{PC, InvGamma})
    AV_calc!(state, dyn, priors)
    out = sum(dyn.A.*priors.σ.σ.*(exp.(dyn.A.*priors.σ.σ).*dyn.W .- dyn.δ))
    #println(dyn.A.*priors.σ.σ.*(exp.(dyn.A.*priors.σ.σ).*dyn.W .- dyn.δ))
    #println(out)
    out += ∇σp(priors.σ)
    if prod(size(out)) == 0.0
        out = ∇σp(priors.σ)
    end
    x = copy(state.x)
    #x[:,1] = x[:,1]./priors.σ.σ
    Σθ = cumsum(x, dims = 2).*priors.σ.σ
    μθ = Vector{Vector{Float64}}()
    ∂μθ = Vector{Array{Float64}}()
    for j in axes(state.x, 1)
        push!(μθ, drift(Σθ[j,:], state.s_loc, priors.diff[j]))
        push!(∂μθ, drift_deriv(Σθ[j,:], state.s_loc, priors.diff[j]))
    end
    for k in eachindex(priors.σ.σ)
        out += ∇σ_drift(state.x[k,:].*priors.σ.σ, μθ[k], ∂μθ[k], priors.σ.σ, k, priors.diff[k], priors)
        out += 2*priors.σ.σ*(state.x[k,1]/priors.σ0)^2
    end
    return out
end

function ∇σ(state::State, dat::PEMData, dyn::Dynamics, priors::Prior, σ::FixedV)
    return []
end

function ∇σp(σ::PC)
    return σ.a*σ.σ - 1
end

function ∇σ_drift(x::Vector{Float64}, μθ, ∂μθ, σ::Float64, k, diff::Union{GammaLangevin, GaussLangevin, GompertzBaseline}, priors::BasicPrior)
    out = 0.0
    θ = cumsum(x)
    for j in 2:size(x,1)
        out += (x[j]*(μθ[j - 1] + σ*θ[j-1]*∂μθ[k, j - 1]))*(tanh(σ*x[j]*μθ[j - 1]) - 1)
    end
    return out
end

function ∇σ_drift(x::Vector{Float64}, μθ, ∂μθ, σ::Float64, k, diff::Union{GammaLangevin, GaussLangevin, GompertzBaseline}, priors::EulerMaruyama)
    out = 0.0
    for j in 2:size(x,1)
        out -= (x[j] - μθ[j - 1]*σ^2)*(σ*∂μθ[k, j - 1] + 2*μθ[j - 1])*σ^2
    end
    return out
end

function ∇σ_drift(x::Vector{Float64}, μθ, ∂μθ, σ::Float64, k, diff::RandomWalk, priors::Prior)
    return 0.0
end



############ Random Walk

function drift(θ, t, diff::RandomWalk)
    return zeros(size(θ))
end

function drift_U(θ, diff::RandomWalk)
    return zeros(size(θ))
end

function drift_deriv(θ, t, diff::RandomWalk)
    return zeros(size(θ,1), size(θ,1))
end

function drift_deriv_t(θ, diff::RandomWalk)
    return zeros(size(θ))
end

function drift_add(x, μθ, ∂μθ, diff::RandomWalk, j::CartesianIndex, σ::Float64)
    return 0.0
end

################ GaussLangevin

function drift(θ, t, diff::GaussLangevin)
    return -0.5.*(θ .- diff.μ(t))./diff.σ(t).^2
end

function drift_U(θ, diff::GaussLangevin)
    return -0.5.*(θ .- diff.μ(t))./diff.σ(t).^2
end

function drift_deriv(θ, t, diff::GaussLangevin)
    return fill(-1 ./(2*diff.σ(t).^2), size(θ,1), size(θ,1))
end

function drift_deriv_t(θ, diff::GaussLangevin)
    return fill(-1/(2*diff.σ(t)^2), size(θ))
end

###### GammaLangevin

# Should change this so the tapering function is input as part of the drift....

function drift(θ, t, diff::GammaLangevin)
    return 0.5*(diff.α.*max.(min.(1,t./diff.c),1/diff.c) .- diff.β.*exp.(θ).*max.(min.(1,t./diff.c),1/diff.c))
end

function drift_U(θ, diff::GammaLangevin)
    return zeros(size(θ))
end

function drift_deriv(θ, t, diff::GammaLangevin)
    out = fill(Inf, size(θ,1), size(θ,1))
    for i in 1:size(θ, 1)
        out[i,:] = -0.5.*diff.β.*exp.(θ).*max.(min.(1,t./diff.c),1/diff.c)
    end
    return out
end

function drift_deriv_t(θ, diff::GammaLangevin)
    return -0.5.*diff.β.*exp.(θ).*max.(min.(1,t./diff.c),1/diff.c)
end

############ Gompertz

function drift(θ, t, diff::GompertzBaseline)
    return fill(diff.α, size(θ))
end

function drift_U(θ, diff::GompertzBaseline)
    return fill(diff.α, size(θ))
end

function drift_deriv(θ, t, diff::GompertzBaseline)
    return zeros(size(θ,1), size(θ,1))
end

function drift_deriv_t(θ, diff::GompertzBaseline)
    return zeros(size(θ))
end


##################

function drift_add(x, μθ, ∂μθ, diff::Union{GaussLangevin, GammaLangevin, GompertzBaseline}, j::CartesianIndex, σ::Float64)
    if j[2] > 1
        out = σ*μθ[j[2] - 1]*(tanh(x[j]*μθ[j[2] - 1]) - 1)
    else
        out = 0.0
    end
    if j[2] < size(x,2)
        for k in (j[2] + 1):size(x,2)
            out += σ*x[j[1], k]*∂μθ[j[2], k - 1]*(tanh(x[j[1], k]*μθ[k - 1]) - 1)
        end
    end
    return out
end

function prior_add(state::State, priors::Prior, k::CartesianIndex)
    #return state.x[k]
    if k[2] == 1
        return state.x[k]/(priors.σ0^2)
    else
        return state.x[k]
    end
end

function prior_EM(state::State, μθ, ∂μθ, priors::EulerMaruyama, k::CartesianIndex)
    # Need to update for GammaLangevin
    if k[2] == 1
        return state.x[k]/(priors.σ0^2)
    else
        if k[2] > 1
            out = (state.x[k] - μθ[k[2] - 1]*priors.σ.σ^2)
        else
            out = 0.0
        end
        if k[2] < size(state.x, 2)
            for j in (k[2] + 1):size(state.x,2)
                out -= (state.x[k[1], j] - μθ[j - 1]*priors.σ.σ^2)*∂μθ[k[2], j - 1]*priors.σ.σ^3
            end
        end
        return out
    end
end


function diffusion_time!(state::State, priors::Prior, dyn::Dynamics, diff::Union{RandomWalk, GaussLangevin, GompertzBaseline}, t_end::Float64, t_switch::Float64, j::Int64)
    return t_end, t_switch
end

function diffusion_time!(state::State, priors::Prior, dyn::Dynamics, diff::GammaLangevin, t_end::Float64, t_switch::Float64, j::Int64)
    # Compute end state
    state_curr = copy(state)
    state_curr.x += state.v.*t_switch
    state_new = copy(state)
    state_new.x += state.v.*t_end
    # Compute bound
    Λ = max(λ_diff(state_curr, priors, j), λ_diff(state_new, priors, j)) #+ 10.0
    if Λ > 0.0
        Λ += 2.0
    end
    if Λ > 10_000
        println(Λ)
        t_move = min(rand(Exponential(1/λ_diff(state_curr, priors, j))),0.00001)
        if t_move < t_end
            dyn.next_event = 5
            return t_move, 0.0
        else
            return t_end, t_switch
        end
    end
    t_move = 0.0
    while t_move < t_end 
        t_new = rand(Exponential(1/Λ))
        t_move += t_new
        if t_move < t_end
            state_curr.x += t_new.*state.v
            λ = λ_diff(state_curr, priors, j)
            if λ/Λ > 1.0
                println(λ);println(Λ)
                error("Bad bound")
            end
            if λ/Λ > rand()
                dyn.next_event = 5
                return t_move, 0.0
            end
        end
    end
    return t_end, t_switch
end

function λ_diff(state::State, priors::BasicPrior, j::Int64)
    # Add time movement
    active_j = filter(idx -> idx[1] == j, state.active)
    ∇U_out = zeros(size(active_j))
    Σθ = cumsum(state.x, dims = 2).*priors.σ.σ
    μθ = drift(Σθ[j,:], 0.0, priors.diff[j])
    ∂μθ = drift_deriv(Σθ[j,:], priors.diff[j])
    for i in eachindex(∇U_out)
        ∇U_out[i] += drift_add(state.x.*priors.σ.σ, μθ, ∂μθ, priors.diff[j], active_j[i])
    end
    return max(0, dot(state.v[active_j],∇U_out))
end

function λ_diff(state::State, priors::EulerMaruyama, j::Int64)
    # Add time movement
    active_j = filter(idx -> idx[1] == j, state.active)
    ∇U_out = zeros(size(active_j))
    Σθ = cumsum(state.x, dims = 2).*priors.σ.σ
    μθ = drift(Σθ[j,:], 0.0, priors.diff[j])
    ∂μθ = drift_deriv(Σθ[j,:], priors.diff[j])
    for i in eachindex(∇U_out)
        ∇U_out[i] += drift_add(state.x.*priors.σ.σ, μθ, ∂μθ, priors.diff[j], active_j[i])
    end
    return max(0, dot(state.v[active_j],∇U_out))
end