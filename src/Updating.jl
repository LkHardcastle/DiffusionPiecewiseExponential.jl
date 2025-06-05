
function flip!(state::BPS, dat::PEMData, dyn::Dynamics, priors::Prior)
    ### Calculate gradient
    U_grad = ∇U(state, dat, dyn, priors)
    ### Flip
    state.v[state.active] -= 2*dot(state.v[state.active], U_grad)*U_grad/norm(U_grad)^2
end

function refresh!(state::BPS, dat::PEMData, dyn::Dynamics, priors::Prior)
    state.v[state.active] = rand(Normal(0,1),size(state.active))
    state.v[state.active] /= norm(state.v[state.active])
end


function gram_schmidt(state::State, U_grad::Vector{Float64}, v_hold::Vector{Float64})
    g1 = (I - U_grad*transpose(U_grad))*rand(Normal(0,1),size(v_hold,1))
    g2 = (I - U_grad*transpose(U_grad))*rand(Normal(0,1),size(v_hold,1))
    e1 = g1/norm(g1)
    e2 = (g2 - dot(g1,g2)*g1)
    e2 = e2/norm(e2)
    return e1, e2
end

function flip!(state::ECMC2, dat::PEMData, dyn::Dynamics, priors::Prior, settings::Exact)
    ### Calculate gradient
    U_grad = ∇U(state, dat, dyn, priors)
    v_hold = copy(state.v[state.active])
    v_hold -= 2*dot(v_hold, U_grad)*U_grad/norm(U_grad)^2
    U_grad = U_grad/norm(U_grad)
    if size(state.active,1) > 1.0
        ### Gradient update
        v_0_new = -(1 - rand()^(2/(size(v_hold,1)-1)))^0.5
        v_0_old = dot(U_grad, v_hold)
        v_hold -= v_0_old*U_grad
        v_hold /= norm(v_hold)
        v_hold = ((1-v_0_new^2)^0.5)*v_hold .+ v_0_new*U_grad
        state.v[state.active] = v_hold[1:size(state.active,1)]
        ### Orthogonal update
        if state.b
            ortho_update!(state, dat, dyn, priors, settings)
            state.b = false
        end
    else
        priors.v = v_hold[(end - size(priors.v,1) + 1):end]
        state.v[state.active] = v_hold[1:size(state.active,1)]
    end
end

function flip!(state::ECMC2, dat::PEMData, dyn::Dynamics, priors::Prior, settings::Splitting)
    ### Calculate gradient
    U_grad = vcat(∇U(state, dat, dyn, priors),∇σ(state, dat, dyn, priors, priors.σ))
    v_hold = vcat(state.v[state.active], priors.v)
    v_hold -= 2*dot(v_hold, U_grad)*U_grad/norm(U_grad)^2
    U_grad = U_grad/norm(U_grad)
    if size(state.active,1) > 1.0
        ### Gradient update
        v_0_new = -(1 - rand()^(2/(size(v_hold,1)-1)))^0.5
        v_0_old = dot(U_grad, v_hold)
        v_hold -= v_0_old*U_grad
        v_hold /= norm(v_hold)
        v_hold = ((1-v_0_new^2)^0.5)*v_hold .+ v_0_new*U_grad
        priors.v = v_hold[(end - size(priors.v,1) + 1):end]
        state.v[state.active] = v_hold[1:size(state.active,1)]
        ### Orthogonal update
        if state.b
            ortho_update!(state, dat, dyn, priors, settings)
            state.b = false
        end
    else
        priors.v = v_hold[(end - size(priors.v,1) + 1):end]
        state.v[state.active] = v_hold[1:size(state.active,1)]
    end
end

function ortho_update!(state::ECMC2, dat::PEMData, dyn::Dynamics, priors::Prior, settings::Exact)
    #U_grad = ∇U(state, dat, dyn, priors)
    U_grad = ∇U(state, dat, dyn, priors)
    v_hold = copy(state.v[state.active])
    U_grad = U_grad/norm(U_grad)
    #v_perp = state.v[state.active] - dot(state.v[state.active], U_grad)*U_grad
    v_perp = v_hold - dot(v_hold, U_grad)*U_grad
    g1, g2 = gram_schmidt(state, U_grad, v_hold)
    g1_, g2_ = dot(g1, v_perp), dot(g2, v_perp)
    v_perp += g1_*(g2 - g1) + g2_*(g1 - g2)
    #v_perp *= dot(state.v[state.active] - dot(state.v[state.active], U_grad)*U_grad, v_perp)
    v_perp *= dot(v_hold - dot(v_hold, U_grad)*U_grad, v_perp)
    v_perp /= norm(v_perp)
    #b = dot(state.v[state.active], U_grad)
    b = dot(v_hold, U_grad)
    a = (1 - b^2)^0.5
    #state.v[state.active] = a*v_perp + b*U_grad
    v_hold = a*v_perp + b*U_grad
    state.v[state.active] = v_hold[1:size(state.active,1)]
end

function ortho_update!(state::ECMC2, dat::PEMData, dyn::Dynamics, priors::Prior, settings::Splitting)
    #U_grad = ∇U(state, dat, dyn, priors)
    U_grad = vcat(∇U(state, dat, dyn, priors), ∇σ(state, dat, dyn, priors, priors.σ))
    v_hold = vcat(state.v[state.active], priors.v)
    U_grad = U_grad/norm(U_grad)
    #v_perp = state.v[state.active] - dot(state.v[state.active], U_grad)*U_grad
    v_perp = v_hold - dot(v_hold, U_grad)*U_grad
    g1, g2 = gram_schmidt(state, U_grad, v_hold)
    g1_, g2_ = dot(g1, v_perp), dot(g2, v_perp)
    v_perp += g1_*(g2 - g1) + g2_*(g1 - g2)
    #v_perp *= dot(state.v[state.active] - dot(state.v[state.active], U_grad)*U_grad, v_perp)
    v_perp *= dot(v_hold - dot(v_hold, U_grad)*U_grad, v_perp)
    v_perp /= norm(v_perp)
    #b = dot(state.v[state.active], U_grad)
    b = dot(v_hold, U_grad)
    a = (1 - b^2)^0.5
    #state.v[state.active] = a*v_perp + b*U_grad
    v_hold = a*v_perp + b*U_grad
    priors.v = v_hold[(end - size(priors.v,1) + 1):end]
    state.v[state.active] = v_hold[1:size(state.active,1)]
end

function refresh!(state::ECMC2, dat::PEMData, dyn::Dynamics, priors::Prior)
    state.b = true
end


function update!(state::State, t::Float64)
    if t < 0.0
        error("Time travel")
    end
    state.x += state.v*t
    state.t += t
end

function update_σ(σ::Variance, v::Vector{Float64}, t::Float64)
    return exp.(log.(σ.σ) .+ v.*t)[1]
end

function update_σ(σ::FixedV, v::Vector{Float64}, t::Float64)
    return σ.σ
end

function update!(state::State, t::Float64, priors::Prior)
    if t < 0.0
        error("Time travel")
    end
    priors.σ.σ = update_σ(priors.σ, priors.v, t)
    state.t += t
    t_push = 0.0
    finish = false
    while !finish
        # Find next split time
        # Update split time
        split_time = rand(Exponential(1/(size(findall(state.g),1)*split_rate(state, priors, 1)*priors.p_split)))
        # Find next merge time 
        merge_curr = Inf 
        if size(state.active,1) > priors.J_min && priors.p_split > 0.0 
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
        end
        # Update 
        t_new, event = findmin([t - t_push, split_time, merge_curr])
        state.x += state.v*t_new
        t_push += t_new
        if event == 1
            finish = true
        elseif event == 2
            split!(state, priors)
        else
            merge!(state, j_curr, priors)
        end
    end
end

function event!(state::State, dat::PEMData, dyn::Dynamics, priors::Prior, times::Times, settings::Settings)
    if dyn.next_event == 1
        # Split
        split!(state, priors)
        split_time!(state, times, priors)
        merge_time!(state, times, priors)
    end
    if dyn.next_event == 2
        # Merge 
        #if priors.p_split > rand()
        merge!(state, times.next_merge_index, priors)
        split_time!(state, times, priors)
        merge_time!(state, times, priors)
        #end
    end
    if dyn.next_event == 3
        # Refresh
        refresh!(state, dat, dyn, priors)
        merge_time_ref!(state, times, priors)
        deleteat!(times.refresh, 1)
        dyn.next_event = 0
    end
    if dyn.next_event == 4
        # Hyperparameter update
        hyper_update!(state, dyn, dat, priors)
        split_time!(state, times, priors)
        merge_time!(state, times, priors)
        deleteat!(times.hyper, 1)
    end
    if dyn.next_event == 5
        flip!(state, dat, dyn, priors, settings)
        merge_time!(state, times, priors)
    end
end