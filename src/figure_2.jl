import DifferentialEquations.SciMLBase
import WeightInitializers
import Plots
import BSON

include("src/lib/compartment_models.jl");
include("src/lib/model.jl");

using Turing

"""Code to recreate figure 1 (without the data histograms)"""

folder = "checkpoints/neural_network/ffm-vwf"
files = readdir(folder)
filter!(endswith(".bson"), files)

inn = Lux.Chain(
    Lux.BranchLayer(
        Lux.Chain(
            Lux.SelectDim(1, 1),
            Lux.ReshapeLayer((1,)),
            Normalize([90.f0]), # HT: 210, WT:150, FFM: 90
            Lux.Dense(1, 12, smooth_relu), # was smooth_relu
            Lux.Parallel(vcat, 
                Lux.Dense(12, 1, non_zero_relu, init_bias=Lux.ones32),
                Lux.Dense(12, 1, non_zero_relu, init_bias=Lux.ones32)
            )
        ),
        Lux.Chain(
            Lux.SelectDim(1, 2),
            Lux.ReshapeLayer((1,)),
            Normalize([300.f0]),
            Lux.Dense(1, 12, smooth_relu),
            Lux.Dense(12, 1, non_zero_relu, init_bias=Lux.ones32)
        )
    ),
    Combine(1 => [1, 2], 2 => [1]), # Join tuples
    AddGlobalParameters(4, [3, 4]; activation=Lux.softplus)
)

D = 120
dummy_ffm = range(0, 90., D)'
dummy_vwf = range(0, 300., D)'
dummy_x = vcat(dummy_ffm, dummy_vwf)

ffm_on_cl = zeros(length(files), D)
ffm_on_v1 = zeros(length(files), D)
vwf_on_cl = zeros(length(files), D)

for (i, file) in enumerate(files)
    ckpt = BSON.parse(joinpath(folder, file))
    delete!(ckpt, :model)
    ckpt = BSON.raise_recursive(ckpt, Main)

    med_eff_ffm, _ = inn[1].layers[1]([60.; 100;;], ckpt[:parameters][1][1], ckpt[:st][1][1])
    eff_ffm, _ = inn[1].layers[1](dummy_x, ckpt[:parameters][1][1], ckpt[:st][1][1])
    ffm_on_cl[i, :] = eff_ffm[1, :] ./ med_eff_ffm[1]
    ffm_on_v1[i, :] = eff_ffm[2, :] ./ med_eff_ffm[2]
    med_eff_vwf, _ = inn[1].layers[2]([60.; 100;;], ckpt[:parameters][1][2], ckpt[:st][1][2])
    vwf_on_cl[i, :] = (inn[1].layers[2](dummy_x, ckpt[:parameters][1][2], ckpt[:st][1][2])[1] ./ med_eff_vwf)[1, :]
end

q1 = hcat([quantile(ffm_on_cl[:, j], [0.05, 0.95]) for j in 1:D]...)
med_eff_ffm_on_cl = median(ffm_on_cl, dims=1)[1, :]
pltA = Plots.plot(dummy_x[1, :], med_eff_ffm_on_cl, ribbon = (med_eff_ffm_on_cl - q1[1, :], q1[2, :] - med_eff_ffm_on_cl), linewidth=1.4, fillalpha=0.2, label=nothing)

q2 = hcat([quantile(ffm_on_v1[:, j], [0.05, 0.95]) for j in 1:D]...)
med_eff_ffm_on_v1 = median(ffm_on_v1, dims=1)[1, :]
pltC = Plots.plot(dummy_x[1, :], med_eff_ffm_on_v1, ribbon = (med_eff_ffm_on_v1 - q2[1, :], q2[2, :] - med_eff_ffm_on_v1), linewidth=1.4, fillalpha=0.2, label=nothing)

q3 = hcat([quantile(vwf_on_cl[:, j], [0.05, 0.95]) for j in 1:D]...)
med_eff_vwf_on_cl = median(vwf_on_cl, dims=1)[1, :]
pltB = Plots.plot(dummy_x[2, :], med_eff_vwf_on_cl, ribbon = (med_eff_vwf_on_cl - q3[1, :], q3[2, :] - med_eff_vwf_on_cl), linewidth=1.4, fillalpha=0.2, label=nothing)


# Plots D is a forest plot from the results of the Bayesian model:
chain = BSON.load("checkpoints/neural_network/ffm-vwf/bayesian/factor_posterior_post_eta_omega_from_nm.bson")[:chain]
data = group(chain, :factors).value.data[:, :, 1]

pltD = Plots.scatter(median(data, dims=1)[1, :], reverse(collect(1:4)), xerr=1.96 .* std(data, dims=1)[1, :], label=nothing)
Plots.vline!([1.], color=:lightgrey, linestyle=:dash, label=nothing, yticks=(1:4, ["V2", "Q", "V1", "CL"]))

Plots.plot(pltA, pltB, pltC, pltD, layout=(2,2), size=(600, 400))