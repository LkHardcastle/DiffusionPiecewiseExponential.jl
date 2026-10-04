abstract type State end

mutable struct ECMC2 <: State
    x::Matrix{Float64}
    v::Matrix{Float64}
    s::Matrix{Bool}
    g::Matrix{Bool}
    s_loc::Vector{Float64}
    t::Float64
    J::Int64
    b::Bool
    active::Vector{CartesianIndex{2}}
end

mutable struct Dynamics
    ind::Int64
    A::Matrix{Float64}
    δ::Matrix{Int64}
    W::Matrix{Float64}
end

mutable struct Storage
    x::Array{Float64}
    v::Array{Float64}
    s::Array{Bool}
    s_loc::Array{Float64}
    J::Vector{Int64}
    t::Vector{Float64}
    ω::Array{Float64}
    σ::Vector{Float64}
    Γ::Vector{Float64}
    γ::Vector{Float64}
end

abstract type Variance end

mutable struct FixedV <: Variance
    σ::Float64
end

mutable struct PC <: Variance
    σ::Float64
    a::Float64
end

abstract type Weight end

mutable struct FixedW <: Weight
    ω::Vector{Float64}
end


abstract type Grid end

mutable struct Fixed <: Grid
    step::Float64
end

abstract type Cts  <: Grid end

mutable struct CtsNB <: Cts
    α::Float64
    β::Float64
    Γ::Float64
    γ::Int64
    max_points::Int64
    max_time::Float64
end

mutable struct CtsNB2 <: Cts
    α::Float64
    β::Float64
    a::Float64
    b::Float64
    Γ::Float64
    γ::Int64
    max_points::Int64
    max_time::Float64
end

mutable struct CtsNBmix <: Cts
    α1::Float64
    β1::Float64
    α2::Float64
    β2::Float64
    γ::Int64
    Γ::Float64
    max_points::Int64
    max_time::Float64
end

mutable struct CtsPois <: Cts
    Γ::Float64
    γ::Int64
    max_points::Int64
    max_time::Float64
end

mutable struct CtsPoisRE <: Cts
    Γ::Float64
    μ::Float64
    σ::Float64
    γ::Int64
    h::Float64
    max_points::Int64
    max_time::Float64
end

abstract type Diffusion end

mutable struct RandomWalk <: Diffusion
end

mutable struct GaussLangevin <: Diffusion
    μ::Function
    σ::Function
end

mutable struct GammaLangevin <: Diffusion
    α::Float64
    β::Float64
    c::Float64
end

mutable struct GompertzBaseline <: Diffusion
    α::Float64
end
abstract type Prior end

mutable struct BasicPrior{V<:Variance, W<:Weight, G<:Grid} <: Prior
    σ0::Float64
    σ::V
    ω::W
    p_split::Float64
    grid::G
    diff::Vector{Diffusion}
    v::Vector{Float64}
    J_min::Int64
end

struct PEMData
    y::Vector{Float64}
    cens::Vector{Float64}
    covar::Matrix{Float64}
    grp::Vector{Int64}
    p::Int64
    n::Int64
    δ::Matrix{Int64}
    W::Matrix{Float64}
    UQ::Matrix{Float64}
end

abstract type Settings end

struct Splitting <: Settings
    max_ind::Int64
    h_rate::Float64
    r_rate::Float64
    verbose::Bool
    δ::Float64
    thin::Float64
end

### Functions

function Base.copy(state::ECMC2)
    return ECMC2(copy(state.x), copy(state.v), copy(state.s), copy(state.g), copy(state.s_loc), copy(state.t), copy(state.J), copy(state.b), copy(state.active))
end

Dynamics(state::State, dat::PEMData) = Dynamics(1, copy(state.x), copy(dat.δ), copy(dat.W))

# Converts arguments like the default constructor of a non-parametric struct, e.g. Vector{GaussLangevin} to Vector{Diffusion}
BasicPrior(σ0, σ::Variance, ω::Weight, p_split, grid::Grid, diff, v, J_min) = BasicPrior{typeof(σ), typeof(ω), typeof(grid)}(σ0, σ, ω, p_split, grid, diff, v, J_min)