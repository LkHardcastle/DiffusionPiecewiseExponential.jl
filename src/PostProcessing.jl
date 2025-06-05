function post_estimates(out, dat, t_smp)
    smps = zeros(length(t_smp), length(dat.s))
    for i in eachindex(dat.s)
        est = hcat(out["t"], vec(out["Sk_x"][:,i,:]), vec(out["Sk_v"][:,i,:]))
        ind = 1
        for t in eachindex(t_smp)
            ind = findfirst(out["t"][ind:end] .> t_smp[t]) - 1 + (ind-1)
            t_diff = t_smp[t] - out["t"][ind]
            smps[t, i] = est[ind,2] + est[ind,3]*t_diff
        end
    end
    return smps
end

function post_smps(smps::Array{Float64})
    est = zeros(size(smps[:,:,1],1)*size(smps[:,:,1],2), size(smps,3))
    for i in axes(smps,3)
        est[:,i] = vec(smps[:,:,i])
    end
    return est
end

function transform_smps(smps::Array{Float64})
    out_smps = copy(smps)
    # Faster way of doing this but don't worry for now
    for i in axes(smps, 1)
        out_smps[i,:, :] = cumsum(smps[i, :, :], dims = 1)
    end
    return out_smps
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

function get_WAIC(out, dat::PEMData, burn::Int64)
    lhood = zeros(size(dat.y,1), size(out["Sk_θ"],3))
    for j in burn:size(out["Sk_θ"],3)
        J = out["Sk_J"][j]
        θ = vec(cumsum(out["Sk_θ"][:,1:J,j],dims = 2))
        s_loc = out["Sk_s_loc"][1:J,j]
        Δ = vcat(s_loc[1] ,s_loc[2:end] .- s_loc[1:(end - 1)])
        St = exp.(cumsum(-exp.(θ).*Δ, dims = 2))
        for i in eachindex(dat.y)
            d = Inf
            if isnothing(findfirst(s_loc .> dat.y[i]))
                d = J
            else
                d = findfirst(s_loc .> dat.y[i])
            end
            if d > 1
                lhood[i,j] = exp(dat.cens[i]*θ[d] - St[d-1] - exp(θ[d])*(dat.y[i] - s_loc[d-1]))
            else
                lhood[i,j] = exp(dat.cens[i]*θ[d] - exp(θ[d])*dat.y[i])
            end
        end
    end
    WAIC = -2*(sum(log.(mean(lhood[:,burn:end], dims = 2))) - sum(var(log.(lhood[:,burn:end]), dims = 2)))
    pWAIC = sum(var(log.(lhood[:,burn:end]), dims = 2))
    return WAIC, pWAIC
end

function get_llhood(out, dat::PEMData, burn::Int64)
    lhood = zeros(size(dat.y,1), size(out["Sk_θ"],3))
    for j in burn:size(out["Sk_θ"],3)
        J = out["Sk_J"][j]
        θ = cumsum(out["Sk_θ"][:,1:J,j],dims = 2)
        s_loc = out["Sk_s_loc"][1:J,j]
        Δ = vcat(s_loc[1] ,s_loc[2:end] .- s_loc[1:(end - 1)])
        for i in eachindex(dat.y)
            θ_ = vec(dat.covar[:,i]'*θ)
            St = exp.(cumsum(-exp.(θ_).*Δ))
            d = Inf
            if isnothing(findfirst(s_loc .> dat.y[i]))
                d = J
            else
                d = findfirst(s_loc .> dat.y[i])
            end
            if d > 1
                lhood[i,j] = exp(dat.cens[i]*θ_[d] - St[d-1] - exp(θ_[d])*(dat.y[i] - s_loc[d-1]))
            else
                lhood[i,j] = exp(dat.cens[i]*θ_[d] - exp(θ[d])*dat.y[i])
            end
        end
    end
    return log.(lhood[:, burn:size(out["Sk_θ"],3)])
end



function get_meansurv(smp_x, smp_s_loc, smp_J, cov)
    mean_surv = zeros(size(smp_x,3))
    for i in axes(smp_x)[3]
        J = smp_J[i]
        θ = cumsum(smp_x[:,1:J,i],dims = 2)
        s_loc = smp_s_loc[1:J,i]
        θ_ = cov*θ
        mean_surv[i] = 0.0
        for j in 1:J
            if j == 1
                sj1 = 0.0
            else
                sj1 = s_loc[j-1]
            end
            mean_surv[i] += exp(log(exp(-sj1*exp(θ_[j])) - exp(-s_loc[j]*exp(θ_[j]))) - θ_[j])
        end
    end
    return mean_surv
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