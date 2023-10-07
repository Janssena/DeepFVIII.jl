import DifferentialEquations: ODEProblem, remake, solve, DiscreteCallback, Tsit5
import Zygote.ChainRules: @ignore_derivatives
import ForwardDiff
import FiniteDiff
import Zygote
import Random
import Optim
import Lux

using LinearAlgebra
using Distributions
using SciMLSensitivity

softplus(x::Real) = log(exp(x) + one(x))

function indicator(n, a, T=Bool)
    Iₐ = zeros(T, n, length(a))
    Zygote.ignore() do
        for i in eachindex(a)
            Iₐ[a[i], i] = one(T)
        end
    end
    return Iₐ
end

# Get vector of lower triangular, non-diagonal elements in matrix
function vecl(A::AbstractMatrix) 
    d = size(A, 1)
    return A[LowerTriangular(ones(Bool, d, d)) .⊻ I(d)]
end

function vecl_to_correlation_matrix(ρ::AbstractVector)
    d = Int(0.5 * (1 + sqrt(1 + 8 * length(ρ))))
    indexes = vecl(reshape(1:(d*d), (d,d)))
    return Symmetric(reshape(indicator(d*d, indexes) * ρ, (d, d)) + I, :L)
end

# We get a positive definite matrix as follows: Symmetric(ω ⋅ C ⋅ ω'), we need to use Symmetric due to numerical inaccuracy.
covariance_matrix(ρ::AbstractVector, σ::AbstractVector) = Symmetric(σ .* vecl_to_correlation_matrix(ρ) .*  σ')


function predict(prob, individual, p; typical=false, interpolate=false, full=false, tmax=maximum(individual.ty), measurement_idx = 1)
    z = typical ? p : p .* exp.([1 0; 0 1; 0 0; 0 0] * individual.eta)
    prob_ = remake(prob, tspan = (prob.tspan[1], tmax), p = [z; zero(p[1])])
    saveat = interpolate ? empty(individual.ty) : individual.ty
    
    return solve(prob_, Tsit5(), save_idxs=full ? (1:length(prob.u0)) : measurement_idx, saveat=saveat, tstops=individual.callback.condition.times, callback=individual.callback)
end

"""Can infer return type."""
function predict_adjoint(prob, t, callback, ζ, η::AbstractVector{T}; measurement_idx = 1) where T<:Real
    z = ζ .* exp.([η; 0.f0; 0.f0])
    prob_ = remake(prob, tspan = (prob.tspan[1], maximum(t)), p = [z; zero(ζ[1])])
    return solve(prob_, Tsit5(), dtmin=1e-10, saveat=t, tstops=callback.condition.times, callback=callback, sensealg=ForwardDiffSensitivity(;convert_tspan=true))[measurement_idx, :]
end

function predict_adjoint(prob, individual::Individual, z::AbstractVector{T}; measurement_idx = 1) where T
    prob_ = remake(prob, tspan = (prob.tspan[1], maximum(individual.ty)), p = [z; zero(T)])
    return solve(
        prob_, Tsit5(), dtmin=1e-10, saveat=individual.ty, 
        tstops=individual.callback.condition.times, callback=individual.callback, 
        sensealg=ForwardDiffSensitivity(;convert_tspan=true)
    )[measurement_idx, :]
end

δyδη(prob, t, callback, p, η) = FiniteDiff.finite_difference_jacobian(eta -> predict_adjoint(prob, t, callback, p, eta), η)

function FO(prob, individual::Individual, ζᵢ, Ω, σ²)
    ŷ = predict_adjoint(prob, individual.ty, individual.callback, ζᵢ, individual.eta)
    Gᵢ = δyδη(prob, individual.ty, individual.callback, ζᵢ, individual.eta)
    Σ = Diagonal(fill(first(σ²), length(ŷ)))
    residuals = individual.y - ŷ
    Cᵢ = Gᵢ * Ω * Gᵢ' + Σ
    return log(det(Cᵢ)) + residuals' * inv(Cᵢ) * residuals
end

function FOCE1(prob, individual::Individual, ζᵢ, Ω, σ²)
    ŷ = predict_adjoint(prob, individual.ty, individual.callback, ζᵢ, individual.eta)
    Gᵢ = δyδη(prob, individual.ty, individual.callback, ζᵢ, individual.eta)
    Σ = Diagonal(fill(first(σ²), length(ŷ)))
    residuals = individual.y - (ŷ + Gᵢ * -individual.eta)
    Cᵢ = Gᵢ * Ω * Gᵢ' + Σ
    return log(det(Cᵢ)) + residuals' * inv(Cᵢ) * residuals
end

function FOCE2(prob, individual::Individual, ζᵢ, Ω, σ²)
    ŷ = predict_adjoint(prob, individual.ty, individual.callback, ζᵢ, individual.eta)
    Gᵢ = δyδη(prob, individual.ty, individual.callback, ζᵢ, individual.eta)
    Σ = diagm(fill(first(σ²), length(ŷ)))
    residuals = individual.y - ŷ
    Cᵢ = Gᵢ * Ω * Gᵢ' + Σ
    return log(det(Cᵢ)) + residuals' * inv(Σ) * residuals + individual.eta' * inv(Ω) * individual.eta
end

function objective(objective_fn, ann, prob, population::Population, st, p::NamedTuple)
    ρ = tanh.(p.gamma)
    ω = softplus.(p.omega)
    Ω = covariance_matrix(ρ, ω)
    σ² = softplus.(p.sigma2)
    
    ζ, _ = Lux.apply(ann, population.x, p.weights, st) # Lux
    
    neg_2LL = zero(eltype(σ²))
    for i in eachindex(population)
        neg_2LL += objective_fn(prob, population[i], ζ[:, i], Ω, σ²)
    end
    
    return neg_2LL
end

function EBE(prob, individual::Individual, ζᵢ, η, Ω_inv, σ²)
    ŷ = predict_adjoint(prob, individual.ty, individual.callback, ζᵢ, η)
    residuals = individual.y - ŷ
    Σ = Diagonal(fill(first(σ²), length(ŷ)))
    
    return log(det(Σ)) + residuals' * inv(Σ) * residuals + η' * Ω_inv * η
end

function optimize_etas!(population::Population, prob, ann, st::NamedTuple, p::NamedTuple, alg=Optim.NelderMead())
    ρ = tanh.(p.gamma)
    ω = softplus.(p.omega)
    Ω_inv = inv(covariance_matrix(ρ, ω))
    σ² = softplus.(p.sigma2)
    
    ζ, _ = Lux.apply(ann, population.x, p.weights, st)

    Threads.@threads for i in eachindex(population)
        opt = Optim.optimize((eta) -> EBE(prob, population[i], ζ[:, i], eta, Ω_inv, σ²), zeros(eltype(ζ), 2), alg)
        population[i].eta .= opt.minimizer
    end
    
    nothing
end

function fixed_objective(ann, prob, population::Population, st::NamedTuple, p::NamedTuple)
    ζ, _ = Lux.apply(ann, population.x, p, st)
    SSE = zero(Float32)
    k = 0
    for i in eachindex(population)
        individual = population[i]
        ŷ = predict_adjoint(prob, individual.ty, individual.callback, ζ[:, i], individual.eta)
        SSE += sum(abs2, individual.y - ŷ)
        k += length(individual.ty)
    end
    return SSE / k
end

################################################################################
##########                                                            ##########
##########                       Normalize layer                      ##########
##########                                                            ##########
################################################################################

struct Normalize{T} <: Lux.AbstractExplicitLayer
    lb::AbstractVector{T}
    ub::AbstractVector{T}
end

Normalize(lb::Real, ub::Real) = Normalize([lb], [ub])
Normalize(ub::Real) = Normalize([ub])
Normalize(lb::AbstractVector, ub::AbstractVector) = Normalize{eltype(lb)}(lb, ub)
Normalize(ub::AbstractVector) = Normalize{eltype(ub)}(zero.(ub), ub)

Lux.initialparameters(rng::Random.AbstractRNG, ::Normalize) = NamedTuple()
Lux.initialstates(rng::Random.AbstractRNG, l::Normalize) = (lb=l.lb, ub=l.ub)

Lux.parameterlength(::Normalize) = 0
Lux.statelength(::Normalize) = 2 # is this correct?

function (l::Normalize)(x::AbstractArray, ps, st::NamedTuple)
    # y = (((x .- st.lb) ./ (st.ub - st.lb)) .- 0.5f0) .* 2.f0 # Normalizes between -1 and 1
    y = (x .- st.lb) ./ (st.ub - st.lb) # Normalizes between 0 and 1, seems to work better against overfitting? Maybe because the bias is initialized further away.
    return y, st
end

################################################################################
##########                                                            ##########
##########                 Add global parameter layer                 ##########
##########                                                            ##########
################################################################################


struct AddGlobalParameters{T, F1, F2} <: Lux.AbstractExplicitLayer
    theta_dim::Int
    out_dim::Int
    locations::AbstractVector{Int}
    init_theta::F1
    activation::F2
end

AddGlobalParameters(out_dim, loc, T=Float32; init_theta=Lux.glorot_uniform, activation=softplus) = AddGlobalParameters{T, typeof(init_theta), typeof(activation)}(length(loc), out_dim, loc, init_theta, activation)

Lux.initialparameters(rng::Random.AbstractRNG, l::AddGlobalParameters) = (theta = l.init_theta(rng, l.theta_dim, 1),)
Lux.initialstates(rng::Random.AbstractRNG, l::AddGlobalParameters{T,F1,F2}) where {T,F1,F2} = (indicator_theta = indicator(l.out_dim, l.locations, T), indicator_x = indicator(l.out_dim, (1:l.out_dim)[Not(l.locations)], T))
Lux.parameterlength(l::AddGlobalParameters) = l.theta_dim
Lux.statelength(::AddGlobalParameters) = 2

function (l::AddGlobalParameters)(x::AbstractMatrix, ps, st::NamedTuple)
    if size(st.indicator_x, 2) !== size(x, 1)
        indicator_x = st.indicator_x * st.indicator_x' # Or we simply do not do this, the one might already be in the correct place following the combine function.
    else
        indicator_x = st.indicator_x
    end
    y = indicator_x * x + st.indicator_theta * repeat(l.activation.(ps.theta), 1, size(x, 2))
    return y, st
end


################################################################################
##########                                                            ##########
##########                  Combine parameters layer                  ##########
##########                                                            ##########
################################################################################

struct Combine{T1, T2} <: Lux.AbstractExplicitLayer
    out_dim::Int
    pairs::T2
end

function Combine(pairs::Vararg{Pair}; T=Float32)
    out_dim = maximum([maximum(pairs[i].second) for i in eachindex(pairs)])
    return Combine{T, typeof(pairs)}(out_dim, pairs)
end

function get_state(l::Combine{T1, T2}) where {T1, T2}
    indicators = Vector{Matrix{T1}}(undef, length(l.pairs))
    negatives = Vector{Vector{T1}}(undef, length(l.pairs))
    for pair in l.pairs
        Iₛ = indicator(l.out_dim, pair.second, T1)
        indicators[pair.first] = Iₛ
        negatives[pair.first] = abs.(vec(sum(Iₛ, dims=2)) .- one(T1))
    end
    return (indicators = indicators, negatives = negatives)
end

Lux.initialparameters(rng::Random.AbstractRNG, ::Combine) = NamedTuple()
Lux.initialstates(rng::Random.AbstractRNG, l::Combine) = get_state(l)
Lux.parameterlength(::Combine) = 0
Lux.statelength(::Combine) = 2

function (l::Combine)(x, ps, st::NamedTuple) 
    indicators = @ignore_derivatives st.indicators
    negatives = @ignore_derivatives st.negatives
    y = .*(broadcast(.+, indicators .* x, negatives)...)
    return y, st
end