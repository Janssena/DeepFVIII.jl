import ForwardDiff
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
using Plots.Measures

softplus(x) = log(exp(x) + 1.)

Plots.default(fontfamily="Computer Modern", linewidth=1.4, tickdirection=:out)

file = "data/NHANES_prepped.csv"
df = DataFrame(CSV.File(file))

################################################################################
#####                                                                      #####
#####                                p(age)                                #####
#####                                                                      #####
################################################################################

K = 6
order = Linear()
X = Normal(0.f0, 1.f0)
b₁ = NeuralSpline(K; order=order, B=10.f0)
b₂ = inverse(bijector(Uniform(0.f0 - eps(Float32), 1.f0 + eps(Float32))))
b₃ = Bijectors.Scale(100.f0)
Y = transformed(X, b₃ ∘ b₂ ∘ b₁)

p_init = vcat(params(b₁)...)[Not(K+K+1, K+K+K+1)] # Linear
function loss(p, X, y)
    b₁ = NeuralSpline(p; order=order, B=10.f0)
    b₂ = inverse(bijector(Uniform(0.f0 - eps(Float32), 1.f0 + eps(Float32))))
    b₃ = Bijectors.Scale(100.f0)
    Y = transformed(X, b₃ ∘ b₂ ∘ b₁)
    return sum(-1.f0 .* logpdf.(Y, y))
end

for i in 1:5000
    if i % 25 == 0 
        println("Epoch $(i): loss = $(loss(p_init, X, df.Age))") 
        plt = Plots.histogram(df.Age, bins=100)
        b = NeuralSpline(p_init; order=order, B=10.f0)
        Y = transformed(X, b₃ ∘ b₂ ∘ b)
        Plots.plot!(Plots.twinx(), 0:0.1:100, pdf.(Y, 0:0.1:100), color=:black, linewidth=2)
        display(plt)
    end
    grad = ForwardDiff.gradient(p_ -> loss(p_, X, df.Age), p_init)
    lr = i < 1000 ? 1e-4 : 2e-5
    p_init -= lr .* grad
end

X = Normal(0.f0, 1.f0)
b₁ = NeuralSpline(p_init; order=order, B=10.f0)
b₂ = inverse(bijector(Uniform(0.f0 - eps(Float32), 1.f0 + eps(Float32))))
b₃ = Bijectors.Scale(100.f0)
Y = transformed(X, b₃ ∘ b₂ ∘ b₁)
BSON.bson("checkpoints/age.bson", Dict(:X => X, :Y => Y, :b => [b₁, b₂, b₃]))


################################################################################
#####                                                                      #####
#####                            age → ht: easy                            #####
#####                                                                      #####
################################################################################

Plots.scatter(df.Age, df.Height, color=:black, markersize=2, xlim=(0, 80), ylim=(50, 210))

ann = Flux.Chain(
    x -> (min.(x ./ 85.f0, 1.0f0) .- 0.5f0) .* 2.f0, # tanh encodes assumptions around extrapolation beyond age = 85
    Flux.Dense(1, 24, Flux.relu),
    Flux.Dense(24, 24, Flux.relu),
    Flux.Dense(24, 1, Flux.sigmoid),
    x -> x .* (210.f0 - 40.f0) .+ 40.f0 # back into ht space, note that support does not extend beyond these values
)

rand_heights = rand(Normal(48, 1), 50) # For females

opt = Flux.ADAM(1e-3)
cost() = mean(abs2.(vcat(df.Height, rand_heights) - ann(vcat(df.Age, zeros(length(rand_heights)))')[1, :]))

p = Flux.params(ann)

for i in 1:15_000
    println("Epoch $i, loss = $(cost())")
    ∇p = Zygote.gradient(cost, p)
    Flux.update!(opt, p, ∇p)
end


ann_sigma = Flux.Chain(
    x -> (min.(x ./ 85.f0, 1.0f0) .- 0.5f0) .* 2.f0, # tanh encodes assumptions around extrapolation beyond age = 85
    Flux.Dense(1, 8, Flux.relu),
    Flux.Dense(8, 8, Flux.relu),
    Flux.Dense(8, 1, Flux.softplus)
)

opt_sigma = Flux.ADAM(1e-2)
function neg_2ll()
    μ = ann(df.Age')[1, :]
    σ = ann_sigma(df.Age')[1, :]
    D = MultivariateNormal(μ, σ)
    return -2 * logpdf(D, df.Height)
end

p_sigma = Flux.params(ann_sigma)

for i in 1:1_000
    println("Epoch $i, loss = $(neg_2ll())")
    ∇p_sigma = Zygote.gradient(neg_2ll, p_sigma)
    Flux.update!(opt_sigma, p_sigma, ∇p_sigma)
end


Plots.plot!(0:0.1:90, ann((0:0.1:90)')[1,:], ribbon=1.96 .* ann_sigma((0:0.1:90)')[1,:], linewidth=3, xlim=(0, 90), color=:red, fillcolor=:lightgrey)

w, re = Flux.destructure(ann)
w_sigma, re_sigma = Flux.destructure(ann_sigma)

BSON.bson("checkpoints/age_to_ht.bson", Dict(:re_mean => re, :w_mean => w, :re_sigma => re_sigma, :w_sigma => w_sigma))

################################################################################
#####                                                                      #####
#####                          ht → age: difficult                          #####
#####                                                                      #####
################################################################################

K = 5

ann = Flux.Chain(
    # We want 210cm to be more similar to shorter individuals than what the model currently predicts, we thus enable this AFTER training.
    # x -> x - 0.75f0 .* Flux.relu.(x .- 170.1f0), # Enable after training
    x -> ((x .- 50.f0) ./ (210.f0 - 50.f0) .- 0.5f0) .* 2.f0,
    Flux.Dense(1, 16, Flux.swish),
    Flux.Dense(16, 16, Flux.swish),
    Flux.Dense(16, 16, Flux.swish),
    Flux.Dense(16, K + K + K - 1, Flux.swish),
)

ht = convert.(Float32, df.Height')
age = convert.(Float32, df.Age ./ 85.f0)

dist_init = Normal{Float32}(0.f0, 1.f0)

function loss(ann, D, x, y)
    out = ann(x)
    neg_LL = 0.f0
    for i in 1:lastindex(y)
        bᵢ = NeuralSpline(out[:, i]; order=Quadratic())
        Yᵢ = transformed(D, bᵢ)
        neg_LL += -logpdf(Yᵢ, y[i])
    end
    
    return neg_LL
end

p = Flux.params(ann)
opt = Flux.ADAM(1e-3)

for epoch in 1:1000
    println("Epoch $epoch, loss = $(loss(ann, dist_init, ht, age))")
    ∇p = Zygote.gradient(() -> loss(ann, dist_init, ht, age), p)
    Flux.update!(opt, p, ∇p)


    age_pred = zeros(Float32, length(ht))
    for (i, htᵢ) in enumerate(ht)
        out_ = ann([htᵢ])
        b = NeuralSpline(out_; order=Quadratic())
        Y = transformed(dist_init, b)
        age_pred[i] = rand(Y) * 80.f0
    end

end

w, re = Flux.destructure(ann)

ann_ = Flux.Chain(
    x -> x - 0.75f0 .* Flux.relu.(x .- 170.1f0),
    x -> (x .- 50.f0) ./ (210.f0 - 50.f0),
    Flux.Dense(1, 16, Flux.swish),
    Flux.Dense(16, 16, Flux.swish),
    Flux.Dense(16, 16, Flux.swish),
    Flux.Dense(16, K + K + K - 1, Flux.swish),
)

_, re = Flux.destructure(ann_)

model = re(w)

BSON.bson("checkpoints/ht_to_age.bson", Dict(:re => re, :w => w, :dist => dist_init, :order => Quadratic()))

