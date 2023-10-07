import Zygote
import Plots
import Optim
import Flux
import BSON
import CSV

include("src/generative/neural_spline_flows.jl")

using Bijectors
using DataFrames
using StatsPlots
using KernelDensity
using Distributions

file = "data/NHANES_prepped.csv"
df = DataFrame(CSV.File(file))

file2 = "data/NHANES_prepped_fem_wt.csv" # for females
df_fem = DataFrame(CSV.File(file2))

################################################################################
#####                                                                      #####
#####                             wt → ht: easy                            #####
#####                                                                      #####
################################################################################


ann = Flux.Chain(
    x -> (min.(x ./ 70.f0, 120.f0 / 70.0f0) .- 0.5f0) .* 2.f0, # tanh encodes assumptions around extrapolation beyond age = 85
    Flux.Dense(1, 16, Flux.swish),
    Flux.Dense(16, 16, Flux.swish),
    Flux.Dense(16, 1, Flux.sigmoid),
    x -> x .* (210.f0 - 50.f0) .+ 50.f0 # back into ht space, note that support does not extend beyond these values
)

opt = Flux.ADAM(1e-3)
cost() = mean(abs2.([fill(54.7, 10); df.Height] - ann([fill(4.5, 10); df.Weight]')[1, :])) # Average wt + ht of neonate ≈ 54.7 cm & 4.5 kg

p = Flux.params(ann)

for i in 1:5_000
    println("Epoch $i, loss = $(cost())")
    ∇p = Zygote.gradient(cost, p)
    Flux.update!(opt, p, ∇p)
end

ann_sigma = Flux.Chain(
    x -> (max.(min.(x ./ 70.f0, 120.f0 / 70.0f0), 12.f0 / 70.f0) .- 0.5f0) .* 2.f0, # tanh encodes assumptions around extrapolation beyond age = 85
    Flux.Dense(1, 8, Flux.swish),
    Flux.Dense(8, 1, Flux.softplus)
)

opt_sigma = Flux.ADAM(1e-2)
function neg_2ll()
    μ = ann(df.Weight')[1, :]
    σ = ann_sigma(df.Weight')[1, :]
    D = MultivariateNormal(μ, σ)
    return -2 * logpdf(D, df.Height)
end

p_sigma = Flux.params(ann_sigma)

for i in 1:3_000
    println("Epoch $i, loss = $(neg_2ll())")
    ∇p_sigma = Zygote.gradient(neg_2ll, p_sigma)
    Flux.update!(opt_sigma, p_sigma, ∇p_sigma)
end

w, re = Flux.destructure(ann)
w_sigma, re_sigma = Flux.destructure(ann_sigma)

BSON.bson("checkpoints/wt_to_ht.bson", Dict(:re_mean => re, :w_mean => w, :re_sigma => re_sigma, :w_sigma => w_sigma))


################################################################################
#####                                                                      #####
#####                           ht → wt: difficult                          #####
#####                                                                      #####
################################################################################

##### Neural Spline Flows model:
K = 4

ann = Flux.Chain(
    # We want 210cm to be more similar to shorter individuals than what the model currently predicts, we thus enable this AFTER training.
    # x -> x - 0.75f0 .* Flux.relu.(x .- 170.1f0),
    x -> (x .- 40.f0) ./ (210.f0 - 40.f0),
    Flux.Dense(1, 16, Flux.swish),
    Flux.Dense(16, 16, Flux.swish),
    Flux.Dense(16, K + K + K - 1, Flux.swish),
)

ht = convert.(Float32, df.Height')
wt = convert.(Float32, df.Weight)

dist_init = Normal{Float32}(0.f0, 1.f0)

function loss(ann, D, x, y)
    out = ann(x)
    neg_LL = 0.f0
    for i in 1:lastindex(y)
        b₁ = NeuralSpline(out[:, i]; order=Quadratic(), B=10.f0)
        b₂ = inverse(bijector(LogNormal(0.f0, 1.f0)))
        Yᵢ = transformed(D, b₂ ∘ b₁)
        neg_LL += -logpdf(Yᵢ, y[i])
    end
    
    return neg_LL
end

p = Flux.params(ann)
opt = Flux.ADAM(3e-3)

for epoch in 1:1000
    if epoch == 1 || epoch % 50 == 0 
        println("Epoch $epoch, loss = $(loss(ann, dist_init, ht, wt))") 
    end
    ∇p = Zygote.gradient(() -> loss(ann, dist_init, ht, wt), p) # How come the gradient is so slow?
    Flux.update!(opt, p, ∇p)
end

w, _ = Flux.destructure(ann)

ann_ = Flux.Chain(
    x -> x - 0.75f0 .* Flux.relu.(x .- 170.1f0),
    x -> (x .- 40.f0) ./ (210.f0 - 40.f0),
    Flux.Dense(1, 16, Flux.swish),
    Flux.Dense(16, 16, Flux.swish),
    Flux.Dense(16, K + K + K - 1, Flux.swish),
)

_, re = Flux.destructure(ann_)

model = re(w)

BSON.bson("checkpoints/ht_to_wt.bson", Dict(:re => re, :w => w, :dist => dist_init, :order => Quadratic(), :B => 10.0f0, :b2 => inverse(bijector(LogNormal(0.f0, 1.f0)))))
