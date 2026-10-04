
function AV_calc!(state::State, dat::PEMData, dyn::Dynamics, priors::Prior)
    # Column 1 is the initial log-hazard α₀ ~ N(0, σ0²); the increments in later columns are scaled by σ
    θ = copy(state.x)
    θ[:,2:end] .*= priors.σ.σ
    Σθ = cumsum(θ, dims = 2)
    dyn.A = transpose(dat.UQ)*Σθ
    return Σθ, θ
end

function ∇U(state::State, dat::PEMData, dyn::Dynamics, priors::BasicPrior)
    # Gradient for the active entries of x followed by log σ, sharing the terms both need
    Σθ, θ = AV_calc!(state, dat, dyn, priors)
    η = dyn.A
    # L x J matrix
    r = exp.(η).*dyn.W .- dyn.δ
    U_ind = reverse(cumsum(reverse(r, dims = 2), dims = 2), dims = 2)
    # Convert to p x J matrix; the increments enter η scaled by σ
    U_ind = dat.UQ*U_ind
    U_ind[:,2:end] .*= priors.σ.σ
    ∇U_out = U_ind[state.active]
    μθ = Vector{Vector{Float64}}()
    ∂μθ = Vector{Vector{Float64}}()
    for j in axes(state.x, 1)
        push!(μθ, drift(Σθ[j,:], state.s_loc, priors.diff[j]))
        push!(∂μθ, drift_deriv(Σθ[j,:], state.s_loc, priors.diff[j]))
    end
    for i in eachindex(∇U_out)
        ∇U_out[i] += prior_add(state, priors, state.active[i])
        ∇U_out[i] += drift_add(θ, μθ[state.active[i][1]], ∂μθ[state.active[i][1]], priors.diff[state.active[i][1]], state.active[i], priors.σ.σ)
    end
    # Part of η that scales with σ, i.e. all but α₀
    ∂η = η .- transpose(dat.UQ)*state.x[:,1]
    return vcat(∇U_out, ∇σ(state, priors, priors.σ, ∂η, r, θ, μθ, ∂μθ))
end

function ∇σ(state::State, priors::Prior, σ::PC, ∂η::Matrix{Float64}, r::Matrix{Float64}, θ::Matrix{Float64}, μθ, ∂μθ)
    out = sum(∂η.*r)
    #println(∂η.*r)
    #println(out)
    out += ∇σp(priors.σ)
    # α₀ in column 1 has a N(0, σ0²) prior and isn't scaled by σ
    for k in axes(state.x, 1)
        out += ∇σ_drift(θ[k,:], μθ[k], ∂μθ[k], priors.σ.σ, k, priors.diff[k], priors)
    end
    return [out]
end

function ∇σ(state::State, priors::Prior, σ::FixedV, ∂η::Matrix{Float64}, r::Matrix{Float64}, θ::Matrix{Float64}, μθ, ∂μθ)
    return Float64[]
end

function ∇σp(σ::PC)
    return σ.a*σ.σ - 1
end

function ∇σ_drift(x::Vector{Float64}, μθ, ∂μθ, σ::Float64, k, diff::Union{GammaLangevin, GaussLangevin, GompertzBaseline}, priors::BasicPrior)
    # d/dlog σ of -log(1 + tanh(μ(θ[j-1])x[j])); x[j] for j > 1 and θ[j-1] - x[1] scale with σ, α₀ = x[1] doesn't
    out = 0.0
    θ = cumsum(x)
    for j in 2:size(x,1)
        out += (x[j]*(μθ[j - 1] + (θ[j-1] - x[1])*∂μθ[j - 1]))*(tanh(x[j]*μθ[j - 1]) - 1)
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

function drift_deriv(θ, t, diff::RandomWalk)
    return zeros(size(θ))
end

function drift_add(x, μθ, ∂μθ, diff::RandomWalk, j::CartesianIndex, σ::Float64)
    return 0.0
end

################ GaussLangevin

function drift(θ, t, diff::GaussLangevin)
    return -0.5.*(θ .- diff.μ(t))./diff.σ(t).^2
end

function drift_deriv(θ, t, diff::GaussLangevin)
    return fill(-1 ./(2*diff.σ(t).^2), size(θ))
end

###### GammaLangevin

# Should change this so the tapering function is input as part of the drift....

function drift(θ, t, diff::GammaLangevin)
    return 0.5*(diff.α.*max.(min.(1,t./diff.c),1/diff.c) .- diff.β.*exp.(θ).*max.(min.(1,t./diff.c),1/diff.c))
end

function drift_deriv(θ, t, diff::GammaLangevin)
    return -0.5.*diff.β.*exp.(θ).*max.(min.(1,t./diff.c),1/diff.c)
end

############ Gompertz

function drift(θ, t, diff::GompertzBaseline)
    return fill(diff.α, size(θ))
end

function drift_deriv(θ, t, diff::GompertzBaseline)
    return zeros(size(θ))
end


##################

function drift_add(x, μθ, ∂μθ, diff::Union{GaussLangevin, GammaLangevin, GompertzBaseline}, j::CartesianIndex, σ::Float64)
    if j[2] > 1
        out = σ*μθ[j[2] - 1]*(tanh(x[j]*μθ[j[2] - 1]) - 1)
    else
        out = 0.0
    end
    # Later log-hazards depend on x[j] through σ*x[j], or directly for α₀ in column 1
    c = j[2] > 1 ? σ : 1.0
    if j[2] < size(x,2)
        for k in (j[2] + 1):size(x,2)
            out += c*x[j[1], k]*∂μθ[k - 1]*(tanh(x[j[1], k]*μθ[k - 1]) - 1)
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
