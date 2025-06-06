module DiffusionPiecewiseExponential

using Distributions
using LinearAlgebra
using MCMCDiagnosticTools
using Optim
using ParetoSmooth
using SpecialFunctions
using Statistics

include("Types.jl")
export ECMC2, Splitting
include("Potential.jl")
include("Updating.jl")
include("SplitMerge.jl")
include("HyperUpdates.jl")
include("Storage.jl")
include("Extrapolation.jl")
include("Sampler.jl")
export pem_fit
include("PreProcessing.jl")
export init_data, init_params
include("PostProcessing.jl")

end # module DiffusionPiecewiseExponential
