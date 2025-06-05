function init_params(p::Int64, dat::PEMData, v_abs::Vector{Float64})
    x0 = rand(Normal(0.0,0.1), p, length(dat.s))
    v0 = v_abs'.*(2 .*(rand(Bernoulli(0.5),p, length(dat.s))) .- 1.0)
    s0 = fill(true, p, length(dat.s))
    return x0, v0, s0
end

function init_params(p::Int64, dat::PEMData)
    x0 = rand(Normal(0.0,1.0), dat.p, size(dat.W,2))
    v0 = rand(Normal(0,1),size(x0))
    s0 = fill(true, dat.p, size(dat.W,2))
    return x0, v0, s0
end

#function init_data(y, cens, covar, breaks)
#    ind = sortperm(y)#
#    y = y[ind]
#    cens = cens[ind]
#    covar = covar[:,ind]
#    d = zeros(Int, length(y))
#    for i in eachindex(y)
#        d[i] = findfirst(breaks .> y[i])
#    end
#    return PEMData(y, cens, covar, size(covar,1), size(covar, 2), breaks, d)
#end

function init_data(y, cens, covar, breaks)
    ind = sortperm(y)
    y = y[ind]
    cens = cens[ind]
    covar = covar[:,ind]
    p, n = size(covar,1), size(covar, 2)
    # Subgroup identification
    UQ = unique(covar, dims = 2)
    L = size(UQ, 2)
    J = size(breaks,1)
    if size(UQ, 2) == n
        grp = collect(1:n)
        UQ = copy(covar)
    else
        grp = zeros(n)
        for i in axes(covar,2)
            for j in axes(UQ,2)
                if UQ[:,j] == covar[:,i]
                    grp[i] = j
                end
            end
        end
    end
    # Build W, δ
    W = zeros(L,J)
    δ = zeros(L,J)
    d = zeros(Int, length(y))
    for i in eachindex(y)
        d[i] = findfirst(breaks .> y[i])
    end
    for l in 1:L
        yl = y[findall(grp .== l)]
        dl = d[findall(grp .== l)]
        δl = cens[findall(grp .== l)]
        for j in 1:J
            if j == 1
                sj1 = 0.0
            else
                sj1 = breaks[j-1]
            end
            W[l,j] = sum(yl[findall(dl .== j)] .- sj1) + length(findall(dl .> j))*(breaks[j] - sj1)
            δ[l,j] = length(intersect(findall(δl .== 1), findall(dl .== j)))
        end
    end
    return PEMData(y, cens, covar, grp, p, n, δ, W, UQ)
end