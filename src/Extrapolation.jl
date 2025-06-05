function barker_extrapolation(out::Dict, diffs::Diffusion, grid::Grid, t_start::Float64, t_end::Float64, plot_grid::Vector{Float64}, k::Int64, step_size::Float64; typeof = "Sk")
    ## Get end points
    n_smp, Σθ, ω, Γ, σ = get_smps(out, typeof, k)
    initial = Σθ[k,end,:]
    if sum(isinf.(initial)) > 0.0
        for i in 1:n_smp
            initial[i] = Σθ[k, findlast(isinf.(Σθ[k,:,i]) .== false), i]
        end
    end
    ## Get time points
    times = extrapolation_times(out, grid, t_start, t_end, n_smp, ω, Γ.*(σ.^2)./step_size)
    #times = extrapolation_times(out, grid, t_start, t_end, n_smp, ω, Γ)
    ## Simulate dynamics
    paths = Vector{Vector{Float64}}()
    for i in 1:n_smp
        push!(paths, barker_dynamics(initial[i], size(times[i],1), diffs, sqrt(step_size), times[i]))
    end
    output = fill(Inf, length(plot_grid), n_smp)
    for i in 1:n_smp
        for j in 1:length(plot_grid)
            if isnothing(findlast(times[i] .<= plot_grid[j]))
                ind = 1
            else
                ind = findlast(times[i] .<= plot_grid[j])
            end
            output[j, i] = paths[i][ind]
        end
    end
    return output
end

function get_smps(out::Dict, typeof, k::Int)
    if typeof == "Sk"
        n_smp = size(out["Sk_t"],1)
        Σθ = cumsum(out["Sk_θ"], dims = 2)
        ω = out["Sk_ω"][k,:]
        Γ = out["Sk_Γ"]
        σ = out["Sk_σ"]
    elseif typeof == "Smp"
        n_smp = size(out["Smp_t"],1)
        Σθ = cumsum(out["Smp_θ"], dims = 2)
        ω = out["Smp_ω"][k,:]
        Γ = out["Smp_Γ"]
        σ = out["Smp_σ"]
    else
        error("typeof must be one of Sk or Smp")
    end
    return n_smp, Σθ, ω, Γ, σ
end

function extrapolation_times(out::Dict, grid::Fixed, t_start::Float64, t_end::Float64, n_smp::Int64, ω::Vector{Float64})
    times = Vector{Vector{Float64}}()
    # Find candidate times
    cand_times = collect(t_start:grid.step:t_end)
    # Thin
    for i in 1:n_smp
        push!(times, vcat(t_start - grid.step, cand_times[findall(rand(Bernoulli(ω[i]),size(cand_times,1)) .== 1)]))
    end
    return times
end

function extrapolation_times(out::Dict, grid::Cts, t_start::Float64, t_end::Float64, n_smp::Int64, ω::Vector{Float64}, Γ::Vector{Float64})
    times = Vector{Vector{Float64}}()
    for i in 1:n_smp
        times_inner = [0.0]
        time_past = copy(t_start)
        while time_past < t_end
            t_push = rand(Exponential(1/(Γ[i]*ω[i])))
            push!(times_inner, t_push)
            time_past += t_push
        end
        push!(times, t_start .+ cumsum(times_inner))
    end
    return times
end

function barker_dynamics(init::Float64, iters::Int64, diff::Diffusion, step_size::Float64, times::Vector{Float64})
    out_vec = zeros(iters)
    out_vec[1] = init
    for i in 2:iters
        ξ = rand(Normal(0,step_size))
        if rand() < 1/(1 + exp.(-2*ξ*drift(out_vec[i-1], times[i], diff)))
            out_vec[i] = out_vec[i-1] + ξ
        else
            out_vec[i] = out_vec[i-1] - ξ
        end
    end
    return out_vec
end
