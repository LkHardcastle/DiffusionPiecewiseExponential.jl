# Diffusion Piecewise Exponential Models

This package provides an interface for fitting Diffusion Piecewise Exponential models as introduced in [*Diffusion piecewise exponential models for survival extrapolation using Piecewise Deterministic Monte Carlo*](https://arxiv.org/abs/2505.05932).

Calling the `pem_fit` function runs two chains of the model, and outputs model parameters and MCMC diagnostics using [MCMCDiagnosticTools.jl](https://turinglang.org/MCMCDiagnosticTools.jl/stable/).

A simple example based on simulated data is as follows:

```
using Random, Distributions, LinearAlgebra
using DiffusionPiecewiseExponential

# Simulate data
Random.seed!(2352)
n = 100
y = rand(Exponential(1.0), n)
breaks = vcat(0.01,collect(0.26:0.25:1.01))
p = 1
cens = (y .< 1.0)
y[findall(y .> 1.0)] .= 1.0
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)

# Initialise sampler
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 10_000
nsmp = 10
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)

# Hazard times for diagnostics and burn in iterations
test_times = collect(0.2:0.2:1.0)
burn_in = 1_000
# Specify the prior
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(7.0, 1.0, 100.0, 1.1), [GaussLangevin(t -> log(0.29), t-> 0.4)], [0.1], 2)
# PC(1.0, 2, 0.5, Inf) - A penalised complexity prior for \sigma
# CtsPois(7.0, 1.0, 100.0, 1.1) - A Poisson process prior for the knots with intensity 7.0, and maximum knots = 100 on the interval (0.0,1.1)
# [GaussLangevin(t -> log(0.29), t-> 0.4)] - A Gaussian stationary distribution for the log-hazard function with mean = log(0.29) and standard deviation = 0.4

# Run the sampler for two chains
out1 = pem_fit(state0, dat, priors, settings, test_times, burn_in)
```

## Calling the package via R

Analysis for Health Technology Assessment will often use R. This package can be called via R using the [JuliaCall](https://cran.r-project.org/web/packages/JuliaCall/index.html) package. 

Repeating the above example:

```
library(JuliaCall) # Package for calling Julia from R
julia_setup() # Package initialisation
julia_library("DiffusionPiecewiseExponential")
julia_library("Distributions")
julia_library("Random")
julia_library("LinearAlgebra")

set.seed(123)
n = 100
y = rexp(n, 1)
cens = as.numeric(y < 1)
for(i in 1:n){
  y[i] = ifelse(cens[i] == 1, y[i], 1)
}
julia_assign("n", as.integer(n))
julia_assign("y", y)
julia_assign("cens", cens)
julia_command("breaks = vcat(0.01,collect(0.26:0.25:1.01))")
julia_command("p = 1")
julia_command("covar = fill(1.0, 1, n)")

julia_source("Setup.jl")

julia_command("priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(7.0, 1.0, 30.0, 1.1), [GaussLangevin(t -> log(0.29), t-> 0.4)], [0.1], 2)")
julia_command("julia_output = pem_fit(state0, dat, priors, settings, test_times, burn_in)")
R_output = julia_eval("julia_output")
```

## Current work

This package contains all the functionality required to implement these models and generate extrapolations based on the code available [here](https://github.com/LkHardcastle/PEM_extrap). It is still in the early stages of development. Current work is focused on:

* Functions for automatically plotting outputs
* Improved documentation and examples
* Improved interoperability with R and relevant HTA focused packages 
