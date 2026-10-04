
function gram_schmidt(state::State, U_grad::Vector{Float64}, v_hold::Vector{Float64})
    # Project onto the complement of U_grad without forming I - U_grad*U_grad'
    g1 = rand(Normal(0,1),size(v_hold,1))
    g1 -= dot(U_grad, g1)*U_grad
    g2 = rand(Normal(0,1),size(v_hold,1))
    g2 -= dot(U_grad, g2)*U_grad
    e1 = g1/norm(g1)
    e2 = (g2 - dot(e1,g2)*e1)
    e2 = e2/norm(e2)
    return e1, e2
end

function flip!(state::ECMC2, priors::Prior, U_grad::Vector{Float64})
    ### Gradient at the current point comes from ∇U in split_inner!
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
            ortho_update!(state, priors, U_grad)
            state.b = false
        end
    else
        priors.v = v_hold[(end - size(priors.v,1) + 1):end]
        state.v[state.active] = v_hold[1:size(state.active,1)]
    end
end

function ortho_update!(state::ECMC2, priors::Prior, U_grad::Vector{Float64})
    # U_grad is the unit gradient from flip!
    #U_grad = ∇U(state, dat, dyn, priors)
    v_hold = vcat(state.v[state.active], priors.v)
    # The switch needs two directions orthogonal to U_grad
    if size(v_hold,1) < 3
        return
    end
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
        # Update split time; candidates in row k unstick at split_rate(k), matching the row probabilities in split!
        split_time = rand(Exponential(1/(sum(size(findall(state.g[k,:]),1)*split_rate(state, priors, k) for k in axes(state.x, 1))*priors.p_split)))
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
        state.x .+= state.v.*t_new
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