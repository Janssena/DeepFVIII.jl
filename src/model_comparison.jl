include("src/lib/compartment_models.jl");
# Bjorkman 2012 -> OSA
CL = 193 .* (wt ./ 53).^0.8 .* (1 .- 0.0045 .* (age .- 22))
V1 = 2.22 .* (wt ./ 53).^0.95
Q = fill(147, length(population))
V2 = 0.73 .* (wt ./ 53).^0.76

p_bjorkman = collect(hcat(CL ./ 1000., V1, Q ./ 1000, V2)')

# Nesterov 2014 -> OSA
CL = fill(2.53, length(population))
V1 = 34.6 .* (wt ./ 73).^0.508
Q = fill(0.548, length(population))
V2 = fill(4.94, length(population))

p_nesterov = collect((hcat(CL, V1, Q, V2) ./ 10.)')

# McEneny-King 2019 -> OSA
CL = 0.238 .* (ffm ./ 53).^0.794 .* (1 .- 0.205 .* max.(0., (age .- 21) ./ 21)) .* (1. .+ 0.309 .* bdd)
V1 = 3.01 .* (ffm ./ 53).^1.02 .* (1. .+ 0.159 .* bdd)
Q = fill(0.142, length(population))
V2 = 0.525 .* (ffm ./ 53).^0.787

p_mcenenyking = collect(hcat(CL, V1, Q, V2)')

# Allard 2020 -> OSA
CL = 204 .* (wt ./ 64).^0.75 .* (age ./ 30).^-0.214
V1 = 2640 .* (wt ./ 64).^0.827
Q = fill(135, length(population))
V2 = fill(339, length(population))

p_allard = collect((hcat(CL, V1, Q, V2) ./ 1000.)')

prob = ODEProblem(two_comp!, zeros(2), (-0.01, 1.))

function predict(prob, individual, p)
    prob_ = remake(prob, tspan = (), p = [p; 0.])
    return solve(prob_, save_idxs = 1, saveat = individual.ty, tstops = individual.callback.condition.times, callback = individual.callback).u
end

rmse(population::Population, p::AbstractMatrix) = mean([rmse(population[i], p[:, i]) for i in eachindex(population)])
rmse(individual::Individual, p::AbstractVector) = sqrt(sum(abs2, individual.y - predict(prob, individual, p)) / length(individual.y))
# Complete data set                     |   OSA   |                     GOF plot                 |
subset_ = [indexes_advate; indexes_novo] # Not([indexes_refa; indexes_aafact])
##################################################################
##########                                              ##########
##########                      Ours                    ##########
##########                                              ##########
##################################################################

smooth_relu(x::T; β::T = T(10)) where {T<:Real} = one(T) / β * Lux.softplus(x * β)
non_zero_relu(x::T) where {T<:Real} = Lux.relu(x) + T(1e-3)

# grab models
dir = "checkpoints/neural_network/ffm-vwf"
files = filter!(endswith(".bson"), readdir(dir))

# We have some issues getting the model so here it is:
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
    Combine(1 => [1, 2, 4], 2 => [1]), # Join tuples
    AddGlobalParameters(4, [3]; activation=Lux.softplus)
)

ζs = zeros(length(files), 4, length(population_ann))
for (i, ckpt_file) in enumerate(files)
    ps = BSON.raise_recursive(BSON.parse(joinpath(dir, ckpt_file))[:parameters], Main)
    st = BSON.raise_recursive(BSON.parse(joinpath(dir, ckpt_file))[:st], Main)
    # ckpt = BSON.load(joinpath(dir, ckpt_file))
    ζs[i, :, :], _ = inn(population_ann.x, ps, st)
end

ζ_median = median(ζs, dims=1)[1, :, :]
advate_correction = [1.15003, 0.810127, 0.25661, 2.23407]
csa_to_osa(csa) = max(-3.0689989 + 4.7581 * csa^0.6636609, 0.)

preds = broadcast(x -> zero.(x), population_ann.y)
for i in eachindex(population_ann)
    preds[i] = csa_to_osa.(predict(prob, population_ann[i], ζ_median[:, i] .* advate_correction).u)
end

sqrt(sum(abs2, vcat((population_ann.y - preds)...)) / 125)
# 18.39 without advate correction
# 14.647 <-- with median post eta advate correction + csa -> osa correction
