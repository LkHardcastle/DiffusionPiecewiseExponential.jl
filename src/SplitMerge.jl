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
    J = log(2) + log(sphere_area(size(state.active,1) + size(priors.v,1) - 1)) - log(sphere_area(size(state.active,1) + size(priors.v,1))*(size(state.active,1) + size(priors.v,1)))
    return exp(rate + J)
end

function sphere_area(d::Int64)
    # Area of sphere embedded in R^(d+1)
    return (2*π^(0.5*d+0.5))/gamma(0.5*d+0.5)
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

function split!(state::State, priors::EulerMaruyama)
    k_prob = []
    for k in axes(state.x, 1)
        push!(k_prob, (size(findall(state.g[k,:]),1))*split_rate(state, priors, k))
    end
    k_prob = k_prob/sum(k_prob)
    k = rand(Categorical(k_prob))
    j = findall(state.g[k,:])[rand(DiscreteUniform(1,size(findall(state.g[k,:]),1)))]
    Σθ = cumsum(state.x, dims = 2)
    μθ = drift_U(Σθ[k,:], priors.diff[k])
    p = exp(logpdf(Normal(μθ[j-1]*priors.σ.σ^2 , 1.0),state.x[k,j]) - logpdf(Normal(0.0, 1.0), 0.0))
    acc = rand(Bernoulli(p))
    if acc
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
        state.v[k,j] = a
    end
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

function split_time!(state::State, times::Times, priors::Prior)
    # Update split time
    if priors.p_split > 0.0
        test_time = []
        for k in axes(state.x,1)
            rate = size(findall(state.g[k,:]),1)*split_rate(state, priors, k)
            push!(test_time, rand(Exponential(1/rate)) + state.t)
        end
        times.next_split = minimum(test_time)
    else 
        times.next_split = Inf
    end
end

function merge_time!(state::State, times::Times, priors::Prior)
    # Update merge times
    merge_curr = Inf
    j_curr = CartesianIndex(0,0)
    for j in state.active
        if j[2] > 1
            merge_cand = merge_time(state, j, priors)
            if merge_cand < merge_curr
                merge_curr = copy(merge_cand)
                j_curr = CartesianIndex(j[1],j[2])
            end
        end
    end
    times.next_merge = copy(merge_curr) + state.t
    times.next_merge_index = CartesianIndex(j_curr[1],j_curr[2])
end

function merge_time_ref!(state::Union{BPS,ECMC}, times::Times, priors::Prior)
    merge_time!(state, times, priors)
end

function merge_time_ref!(state::ECMC2, times::Times, priors::Prior)
    #merge_time!(state, times, priors)
end