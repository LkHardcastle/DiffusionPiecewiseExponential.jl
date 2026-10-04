function merge_time(state::State, j::CartesianIndex, priors::Prior)
    if rand() < priors.p_split
        if (state.x[j] > 0.0 && state.v[j] < 0.0) || (state.x[j] < 0.0 && state.v[j] > 0.0)
            return abs(state.x[j])/abs(state.v[j])
        else
            return Inf
        end 
    else
        return Inf
    end
end

function split_rate(state::State, priors::Prior, k::Int64)
    rate = log(priors.p_split) + log(priors.ω.ω[k]) - log(1 - priors.ω.ω[k]) - 0.5*log(2*pi)
    J = log(2) + log_sphere_area(size(state.active,1) + size(priors.v,1) - 1) - log_sphere_area(size(state.active,1) + size(priors.v,1)) - log(size(state.active,1) + size(priors.v,1))
    return exp(rate + J)
end

function log_sphere_area(d::Int64)
    # Log area of sphere embedded in R^(d+1), using loggamma so large d doesn't overflow
    return log(2) + (0.5*d+0.5)*log(π) - loggamma(0.5*d+0.5)
end

function split!(state::State, priors::Prior)
    k_prob = []
    for k in axes(state.x, 1)
        push!(k_prob, (size(findall(state.g[k,:]),1))*split_rate(state, priors, k))
    end
    k_prob = k_prob/sum(k_prob)
    k = rand(Categorical(k_prob))
    j = findall(state.g[k,:])[rand(DiscreteUniform(1,size(findall(state.g[k,:]),1)))]
    state.s[k,j] = true
    state.g[k,j] = false
    # Add to state.active
    a = split_velocity(state, priors)
    if a == 0.0
        error("Bad split velocity")
    end
    state.active = findall(state.s)
    # New velocities
    state.v[state.active] *= sqrt(1-a^2)
    priors.v *= sqrt(1-a^2) 
    state.v[k,j] = a
end

function split_velocity(state::State, priors::Prior)
    # Draw new velocity 
    return sqrt(1 - rand()^(2/(size(state.active,1) + size(priors.v,1))))*(2*rand(Bernoulli(0.5)) - 1)
end


function merge!(state::State, j::CartesianIndex, priors::Prior)
    state.s[j] = false
    state.g[j] = true
    state.v[j] = 0.0
    state.x[j] = 0.0
    # Remove from state.active
    state.active = findall(state.s)
    nm = norm(vcat(state.v[state.active], priors.v))
    state.v[state.active] /= nm
    priors.v /= nm
end