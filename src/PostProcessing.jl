function post_smps(smps::Array{Float64})
    est = zeros(size(smps[:,:,1],1)*size(smps[:,:,1],2), size(smps,3))
    for i in axes(smps,3)
        est[:,i] = vec(smps[:,:,i])
    end
    return est
end

function survival_plot(t, breaks, h_vec, break_int)
    S_store = []
    for t_ in t
        past_ind = findfirst(breaks .>= t_) - 1
        if past_ind == 0
            push!(S_store, exp(-h_vec[past_ind + 1]*(t_)))
        else
            S_init = exp(-break_int*sum(h_vec[1:past_ind]))
            push!(S_store, S_init*exp(-h_vec[past_ind + 1]*(t_ - breaks[past_ind])))
        end
    end
    return hcat(t, S_store)
end

function cts_transform(x::Array{Float64}, s_loc::Array{Float64}, grid::Vector{Float64})
    out = zeros(size(x,1), size(grid,1), size(x,3))
    for k in 1:size(x,3)
        for i in 1:length(grid)
            if isnothing(findlast(s_loc[:,k] .< grid[i]))
                ind = 1
            else
                ind = findlast(s_loc[:,k] .< grid[i]) + 1
            end
            if ind == (size(s_loc,1) + 1)
                ind -= 1 
            end
            for j in 1:size(x,1)
                if isinf(x[j, ind, k])
                    #out[j,i,k] = x[j, ind - 1, k]
                    out[j,i,k] = x[j, findlast(isinf.(x[j,:,k]) .== false), k]
                else
                    out[j,i,k] = x[j, ind, k]
                end
            end
        end
    end
    return out
end

function pem_survival(λ::Matrix{Float64}, times::Vector{Float64})
    t_ = times[2:end] .- times[1:(end -1)]
    return cumprod(exp.(.- t_'.*λ'), dims = 2)'
end

function get_DIC(out, dat::PEMData, burn::Int64)
    deviance = zeros(size(out["Sk_θ"],3))
    n_param = zeros(size(out["Sk_θ"],3))
    for i in burn:size(out["Sk_θ"],3)
        J = out["Sk_J"][i]
        θ = cumsum(out["Sk_θ"][:,1:J,i],dims = 2)
        s_loc = out["Sk_s_loc"][1:J,i]
        L = size(dat.UQ, 2)
        W = zeros(L,J)
        δ = zeros(L,J)
        d = zeros(Int, length(dat.y))
        for i in eachindex(dat.y)
            if isnothing(findfirst(s_loc .> dat.y[i]))
                d[i] = J
            else
                d[i] = findfirst(s_loc .> dat.y[i])
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
                    sj1 = s_loc[j-1]
                end
                W[l,j] = sum(yl[findall(dl .== j)]) .- length(findall(dl .== j))*sj1 + length(findall(dl .> j))*(s_loc[j] - sj1)
                δ[l,j] = length(intersect(findall(δl .== 1), findall(dl .== j)))
            end
        end
        deviance[i] = 2*sum(exp.(θ).*W .- δ.*θ) 
        n_param[i] = length(findall(out["Sk_θ"][:,1:J,i] .!= 0.0))
    end
    DIC = mean(deviance[findall(.!isnan.(deviance))][burn:end]) + 0.5*var(deviance[findall(.!isnan.(deviance))][burn:end])
    dev_new = mean(deviance[findall(.!isnan.(deviance))][burn:end]) + mean(n_param[burn:end])
    dev_bar = mean(deviance[findall(.!isnan.(deviance))][burn:end])
    return deviance, DIC, dev_new, dev_bar
end

function get_llhood(out::Dict, dat::PEMData, burn_in::Int64)
    # Pointwise log-likelihood δ_i log h(y_i) - ∫_0^{y_i} h(u)du, one row per observation (in dat's sorted order)
    # and one column per stored iteration from burn_in
    N = size(out["Sk_θ"], 3)
    llhood = zeros(dat.n, N - burn_in + 1)
    for ind in burn_in:N
        J = out["Sk_J"][ind]
        s_loc = out["Sk_s_loc"][1:J, ind]
        η = transpose(dat.UQ)*cumsum(out["Sk_θ"][:, 1:J, ind], dims = 2)
        # As in dat_update!, column j covers [s_loc[j-1], s_loc[j]) and the last column runs to ∞
        s_left = vcat(0.0, s_loc[1:(end - 1)])
        # Cumulative hazard at the left end of each column
        H = hcat(zeros(size(η, 1)), cumsum(exp.(η[:, 1:(end - 1)]).*transpose(diff(s_left)), dims = 2))
        for i in eachindex(dat.y)
            j = min(searchsortedlast(s_loc, dat.y[i]) + 1, J)
            l = dat.grp[i]
            llhood[i, ind - burn_in + 1] = dat.cens[i]*η[l, j] - H[l, j] - exp(η[l, j])*(dat.y[i] - s_left[j])
        end
    end
    return llhood
end

function get_looic(out::Dict, dat::PEMData, burn_in::Int64)
    # LOOIC = -2 elpd_loo, estimated by Pareto-smoothed importance sampling
    llhood = get_llhood(out, dat, burn_in)
    est = psis_loo(llhood, chain_index = ones(Int, size(llhood, 2)), source = "mcmc")
    return -2*est.estimates(:cv_elpd, :total)
end



function get_meansurv(haz, s_loc, cov)
    #mean_surv = zeros(size(haz,2))
    #for i in axes(haz,2)
    #    for j in axes(haz, 1)
    #        if j == 1
    #            sj1 = 0.0
    #        else
    #            sj1 = s_loc[j-1]
    #        end
    #        #mean_surv[i] += exp(log(exp(-sj1*exp(haz[j,i])) - exp(-s_loc[j]*exp(haz[j,i]))) - haz[j,i])
    #        mean_surv[i] += (s_loc[j] - sj1)*exp(-sj1*exp(haz[j,i]))
    #    end
    #end
    mean_surv1 = zeros(size(haz,2))
    for i in axes(haz,2)
        s_y = zeros(size(haz,1) + 1)
        s_y[1] = 1
        logS_y = 0.0
        for j in axes(haz, 1)
            if j == 1
                sj1 = 0.0
            else
                sj1 = s_loc[j-1]
            end
            logS_y += exp(haz[j,i])*(s_loc[j] - sj1)
            s_y[j+1] = exp(-logS_y)
            mean_surv1[i] += exp(-haz[j,i])*(s_y[j] - s_y[j+1])
        end
    end
    return mean_surv1
end

function r_hat(x::Vector{Vector{Float64}})
    xbar = mean.(x)
    μhat = mean(reduce(vcat,x))
    s2i = var.(x)
    s2 = mean(s2i)
    B = (1/(size(x,1) - 1))*sum((xbar .- μhat).^2)
    σ2 = s2*(size(x[1],1)-1)/size(x[1],1) + B
    return sqrt(σ2/s2)
end