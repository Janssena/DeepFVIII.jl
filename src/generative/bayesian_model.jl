import BSON
import Flux
import CSV

include("src/generative/neural_spline_flows.jl");

using Turing
using Bijectors
using DataFrames
using StatsPlots

ckpt_age_ht = BSON.load("checkpoints/age_to_ht_manuscript.bson")
ht_μ = ckpt_age_ht[:re_mean](ckpt_age_ht[:w_mean])
ht_σ = ckpt_age_ht[:re_sigma](ckpt_age_ht[:w_sigma])

ckpt_ht_wt = BSON.load("checkpoints/ht_to_wt_manuscript.bson")
spline = ckpt_ht_wt[:re](ckpt_ht_wt[:w])

ann = restructure(model)

vwf_scale(x::AbstractVector) = exp(4.11f0 + max((x[1] / 45f0), 40f0 / 45f0) * 0.644f0 * 0.8f0) * (0.701f0 ^ x[2])
vwf_scale(x::AbstractMatrix) = exp.(4.11f0 .+ max.((x[:, 1] ./ 45.f0), 40f0 / 45f0) .* 0.644 * 0.8f0) .* (0.701f0 .^ x[:, 2])


"""
Code example for bayesian generative model given that only age is observed.
dcm here refers to the DCM(⋅) object, while individual refers to the Individual(⋅)
object from DeepCompartmentModels.jl
"""

@model function generative_model(dcm, ann, individual, y, age; Ω=Ω, σ=σ, ht_μ=ht_μ, ht_σ=ht_σ, spline=spline)
    height ~ MultivariateNormal(ht_μ([age]), ht_σ([age]))

    b = NeuralSpline(spline(height); order=Quadratic(), B=10.f0)
    weight ~ transformed(Truncated(Normal(0.f0, 1.f0), -10.f0, 10.f0), ckpt_ht_wt[:b2] ∘ b)
    
    w ~ Dirichlet([1 - 0.45f0, 0.45f0])
    vwf_base = LogNormal(0.1578878428424249f0, 0.3478189783243864f0)
    vwf ~ MixtureModel([
        transformed(vwf_base, Bijectors.Scale(vwf_scale([age, 0.f0]))),
        transformed(vwf_base, Bijectors.Scale(vwf_scale([age, 1.f0])))
    ], w)
    
    eta ~ MultivariateNormal(zeros(2), Ω)
    
    ζ = ann([weight; height; vwf])
    z = [ζ .* exp.(dcm.eta_mask * eta); 0.f0]
    if (any(isnan.(z)) || any(isinf.(z)) || any(eta .< -3.) || any(eta .> 3.)) 
        Turing.@addlogprob! -Inf 
        return
    end

    pred = predict_adjoint(dcm, individual, z)
    y ~ MultivariateNormal(pred, σ)
end

"""
Code example for Bayesian generative model when only a patient's blood group 
and vwf are missing.
"""

@model function generative_model(dcm, ann, individual, y, age, weight, height)
    w ~ Dirichlet([1.f0 - 0.45f0, 0.45f0])
    vwf_base = LogNormal(0.1578878428424249f0, 0.3478189783243864f0)
    vwf ~ MixtureModel([
        transformed(vwf_base, Bijectors.Scale(vwf_scale([age, 0.f0]))),
        transformed(vwf_base, Bijectors.Scale(vwf_scale([age, 1.f0])))
    ], w)
    
    eta ~ MultivariateNormal(zeros(Float32, 2), Ω)
    
    ζ = ann([weight; height; vwf])
    z = T[ζ .* exp.(dcm.eta_mask * eta); 0.f0]
    if (any(isnan.(z)) || any(isinf.(z)) || any(eta .< -3.) || any(eta .> 3.)) 
        Turing.@addlogprob! -Inf 
        return
    end

    pred = predict_adjoint(dcm, individual, z)
    y ~ MultivariateNormal(pred, σ)
end

