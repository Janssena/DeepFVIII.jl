import Optim
import Plots
import CSV

using Bijectors
using StatsPlots
using DataFrames
using Distributions

"""
Difference between BG non-O and BG O in male blood donors.
Data from: 
"Effect of von Willebrand factor Y/C1584 on in vivo protein level and function and
interaction with ABO blood group"

            |   n    |   VWFAg ± one SD  |
BG O        |  1131  |      98 ± 30      |
BG non-O    |  1293  |   127.3 ± 38      |

[BG non-O] = 1.299130612 × [BG O]
non-O groups were combined using https://www.statstodo.com/CombineMeansSDs.php
"""

################################################################################
#####                                                                      #####
#####                       age → vwfag ← bgo: easy                        #####
#####                                                                      #####
################################################################################

"""
Conclusion:

We use the data of Biguzzi et al. to fit an initial model to predict the mean 
effect of age (while having bgo). Next we validate that model on the remaining 
VWF levels. Here we see that fixing the effect for individuals with age < 40 
makes sense (which matches the concept that the increase in VWF with age is due 
to comorbidities rather than age). In addition, based on both data sets, is 
seems that the effect of age is overestimated, so we reduce the fitted θ by a 
factor of 0.8. We thus end up with the following model:

f(x) = exp(4.11 + max(age / 45, 40 / 45) * 0.644 * 0.8) * (0.701 ^ bgo)

We then fit a LogNormal distribution to the data normalized by f(x) resulting in 
the following distribution:

LogNormal(μ = 0.1578878428424249, σ = 0.3478189783243864)

which fits the data quite well. Since the mean of the lognormal is not zero we 
can adjust the model as follows:

g(x) = f(x) / exp(0.1578878428424249)

and use X = LogNormal(μ = 0., σ = 0.34782).

Essentially we take a sample from X, multiply by g(x) and we have our 
sample vwf ~ Y. We use a Scale Bijector to get the correctly scaled 
distribution.
"""
logf(x::AbstractVector, θ) = (θ[1] + (x[1] / 45.) * θ[2]) + log(0.7008693880000001 ^ x[2])
logf(x::AbstractMatrix, θ) = (θ[1] .+ (x[:, 1] ./ 45) .* θ[2]) .+ log.(0.7008693880000001 .^ x[:, 2])

f(x::AbstractVector, θ) = exp(θ[1] + max((x[1] / 45), 40 / 45) * θ[2]) * (0.7008693880000001 ^ x[2])
f(x::AbstractMatrix, θ) = exp.(θ[1] .+ max.((x[:, 1] ./ 45.), 40 / 45) .* θ[2]) .* (0.7008693880000001 .^ x[:, 2])

function sample_vwf(age, bgo)
  X = LogNormal(0.1578878428424249, 0.3478189783243864)
  b = Bijectors.Scale(f([age, bgo], [4.11, 0.644 * 0.8]))
  Y = transformed(X, b)
  return rand(Y)
end

# For learning relationship between age & vwfag
df = DataFrame(CSV.File("data/collected_age_vwfag_bg.csv"))
df_ = DataFrame(CSV.File("data/VWFPlots/Biguzzi_et_al_age_vwfag_bgo.csv"))

x₁ = Matrix(df[:, [:age, :bgo]])
x₂ = Matrix(df_[:, [:age, :bgo]])

neg_2ll(θ) = -2 * logpdf(MultivariateNormal(logf(x₂, θ), θ[end]), log.(df_.vwfag))
opt = Optim.optimize(neg_2ll, rand(3))
# θ = [4.114746612955803, 0.6439406463187992, 0.3564840639436774]
