import Bijectors: Bijector, Inverse, inverse, with_logabsdet_jacobian
import Distributions: Normal
import Distributions: logpdf
import Bijectors

softplus(x::Real) = log(exp(x) + one(x))
softmax(x::AbstractVector) = exp.(x) ./ sum(exp.(x))
sigmoid(x::Real) = one(x) / (one(x) + exp(-x))

abstract type Order end
struct Linear <: Order end
struct Quadratic <: Order end


struct NeuralSpline{O<:Order,T} <: Bijector
    θʷ_::Vector{T} # ∈ ℝᴷ, unnormalized length between each knot point in x
    θʰ_::Vector{T} # ∈ ℝᴷ, unnormalized length between each knot point in y
    δ::Vector{T} # ∈ ℝᴷ⁻¹
    λ::Vector{T} # ∈ ℝᴷ, used for linear splines.
    B::T
end

Base.show(io::IO, b::NeuralSpline{O,T}) where {O<:Linear,T} = print(io, "NeuralSpline{$(T), :linear}(...)")
Base.show(io::IO, b::NeuralSpline{O,T}) where {O<:Quadratic,T} = print(io, "NeuralSpline{$(T), :quadratic}(...)")

function NeuralSpline(K::Integer; order::Order=Linear(), B=3.f0)
    if !isa(order, Linear) && !isa(order, Quadratic)
        return throw(ErrorException("Unsupported `order` argument. Should be either `Linear()` or `Quadratic()`."))
    end
    θʷ_ = rand(Normal(0.f0, 1.f0), K)
    θʰ_ = rand(Normal(0.f0, 1.f0), K)
    δ = rand(Normal(0.f0, 1.f0), K - 1)
    λ = isa(order, Linear) ? rand(Float32, K) : eltype(δ)[]
    return NeuralSpline{typeof(order), eltype(δ)}(θʷ_, θʰ_, δ, λ, convert(eltype(δ), B))
end

"""Constructors for conditional models"""
function NeuralSpline(p::AbstractArray; order=Linear(), B=3.f0)
    if !isa(order, Linear) && !isa(order, Quadratic)
        return throw(ErrorException("Unsupported `order` argument. Should be either `Linear()` or `Quadratic()`."))
    end
    len = length(p) + 1
    K = Integer(order isa Linear ? len / 4 : len / 3)
    return NeuralSpline{order isa Linear ? Linear : Quadratic, eltype(p)}(p[begin:K], p[K+1:2*K], p[2*K+1:3*K-1], p[3*K:end], convert(eltype(p), B))
end

"""For parameter estimation"""
NeuralSpline(θʷ_, θʰ_, δ, λ; B=3.f0) = NeuralSpline{Linear, eltype(δ)}(θʷ_, θʰ_, δ, λ, convert(eltype(δ), B))
NeuralSpline(θʷ_, θʰ_, δ; B=3.f0) = NeuralSpline{Quadratic, eltype(δ)}(θʷ_, θʰ_, δ, eltype(δ)[], convert(eltype(δ), B))

function params(b::NeuralSpline{Linear,T}) where T
    eps_ = T(1e-3)
    θʷ = eps_ .+ (one(T) .- eps_ .* length(b.θʷ_)) .* softmax(b.θʷ_) # Make length sum to one
    θʰ = eps_ .+ (one(T) .- eps_ .* length(b.θʰ_)) .* softmax(b.θʰ_) # Make length sum to one
    δ = [one(T) - eps_; eps_ .+ softplus.(b.δ); one(T) - eps_] # pad with ones (apparently not exactly one from orignal implementation)
    λ = (one(T) - T(0.05)) .* sigmoid.(b.λ) .+ T(0.025)
    return θʷ, θʰ, δ, λ
end

function params(b::NeuralSpline{Quadratic,T}) where T
    eps_ = T(1e-3)
    θʷ = eps_ .+ (one(T) .- eps_ .* length(b.θʷ_)) .* softmax(b.θʷ_) # Make length sum to one
    θʰ = eps_ .+ (one(T) .- eps_ .* length(b.θʰ_)) .* softmax(b.θʰ_) # Make length sum to one
    δ = [one(T) - eps_; eps_ .+ softplus.(b.δ); one(T) - eps_] # pad with ones (apparently not exactly one from orignal implementation)
    return θʷ, θʰ, δ
end

params(b::Inverse{NeuralSpline{O, T}}) where {O,T} = params(b.orig)

function params(b::Union{NeuralSpline{Linear, T}, Inverse{NeuralSpline{Linear, T}}}, inputs) where T
    θʷ_, θʰ_, δ_, λ_ = params(b)
    Nx_, xᵏ_, Ny_, yᵏ_, Iᵏ = handle_knots(b, inputs, θʷ_, θʰ_)    
    return Nx_[Iᵏ], Ny_[Iᵏ], δ_[Iᵏ], δ_[2:end][Iᵏ], λ_[Iᵏ], xᵏ_[Iᵏ], yᵏ_[Iᵏ]
end

function params(b::Union{NeuralSpline{Quadratic, T}, Inverse{NeuralSpline{Quadratic, T}}}, inputs) where T
    θʷ_, θʰ_, δ_ = params(b)
    Nx_, xᵏ_, Ny_, yᵏ_, Iᵏ = handle_knots(b, inputs, θʷ_, θʰ_)    
    return Nx_[Iᵏ], Ny_[Iᵏ], δ_[Iᵏ], δ_[2:end][Iᵏ], xᵏ_[Iᵏ], yᵏ_[Iᵏ]
end

function handle_knots(b::NeuralSpline{O,T}, x, θʷ_, θʰ_) where {O,T}
    Nx_, xᵏ_ = knots(b, θʷ_) # Nx = xᵏ⁺¹ - xᵏ 
    Ny_, yᵏ_ = knots(b, θʰ_) # Ny = yᵏ⁺¹ - yᵏ 

    Iᵏ = knot_mask(x, xᵏ_ .+ T(1e-6)) # This mask will fail when x has values outside if the sequence
    return Nx_, xᵏ_, Ny_, yᵏ_, Iᵏ
end

function handle_knots(b::Inverse{NeuralSpline{O,T}}, y, θʷ_, θʰ_) where {O,T}
    Nx_, xᵏ_ = knots(b, θʷ_) # Nx = xᵏ⁺¹ - xᵏ 
    Ny_, yᵏ_ = knots(b, θʰ_) # Ny = yᵏ⁺¹ - yᵏ 

    Iᵏ = knot_mask(y, yᵏ_ .+ T(1e-6)) # This mask will fail when x has values outside if the sequence
    return Nx_, xᵏ_, Ny_, yᵏ_, Iᵏ
end


"""
Takes the knot lengths in [0, 1] and converts them to their true lengths in [-B, B].
Also returns the knot coordinates (i.e. xᵏ and yᵏ).
"""
function knots(b::NeuralSpline{O, T}, lengths) where {O,T}
    knot_coordinates = [-b.B; (b.B + b.B) .* cumsum(lengths)[1:end-1] .- b.B; b.B]
    return knot_coordinates[begin+1:end] - knot_coordinates[begin:end-1], knot_coordinates
end

knots(b::Inverse{NeuralSpline{O, T}}, lengths) where {O,T} = knots(b.orig, lengths)

function find_knot_idx(value; sequence)
    if all(<(value), sequence) 
        return lastindex(sequence) - 1 
    elseif all(>=(value), sequence) 
        return firstindex(sequence)
    else
        return findfirst(>=(value), sequence) - 1
    end
end

knot_mask(value::Real, sequence) = find_knot_idx(value; sequence)
knot_mask(values::AbstractArray, sequence) = find_knot_idx.(values; sequence)

function monotonic_rational_linear_spline_forward(x, Nx, Ny, δ, δ⁺¹, λ, xᵏ, yᵏ)
    T = eltype(x)
    ξ = (x - xᵏ) ./ Nx
    sₖ = Ny ./ Nx
    λ_minus_ξ = λ - ξ

    wa = one(T) # can be whatever
    wb = sqrt.(δ ./ δ⁺¹) .* wa
    wc = @. (λ * wa * δ + (one(T) - λ) * wb * δ⁺¹) / sₖ
    
    # Calculate y coords of bins
    ya = yᵏ
    yb = Ny + yᵏ
    yc = @. ((one(T) - λ) * wa * ya + λ * wb * yb) / ((one(T) - λ) * wa + λ * wb)

    numerator = @. (wa * ya * λ_minus_ξ + wc * yc * ξ) * (ξ <= λ) + (wc * yc * (one(T) - ξ) + wb * yb * (ξ - λ)) * (ξ > λ)
    denominator = @. (wa * λ_minus_ξ + wc * ξ) * (ξ <= λ) + (wc * (one(T) - ξ) + wb * (ξ - λ)) * (ξ > λ)

    y = numerator ./ denominator

    jac_num = @. (wa * wc * λ * (yc - ya) * (ξ <= λ) + wb * wc * (one(T) - λ) * (yb - yc) * (ξ > λ)) / Nx
    logabsdet = log.(jac_num) - T(2.) .* log.(abs.(denominator))
    # Make the adjustment for elements outside of the [-B, B] bounding box
    return y, logabsdet
end

function monotonic_rational_linear_spline_back(y, Nx, Ny, δ, δ⁺¹, λ, xᵏ, yᵏ)
    T = eltype(y)
    sₖ = Ny ./ Nx

    wa = one(T) # can be whatever
    wb = sqrt.(δ ./ δ⁺¹) .* wa
    wc = @. (λ * wa * δ + (one(T) - λ) * wb * δ⁺¹) / sₖ
    
    # Calculate y coords of bins
    ya = yᵏ
    yb = Ny + yᵏ
    yc = @. ((one(T) - λ) * wa * ya + λ * wb * yb) / ((one(T) - λ) * wa + λ * wb)

    numerator = @. (λ * wa * (ya - y)) * (y <= yc) + ((wc - λ * wb) * y + λ * wb * yb - wc * yc) * (y > yc)
    denominator = @. ((wc - wa) * y + wa * ya - wc * yc) * (y <= yc) + ((wc - wb) * y + wb * yb - wc * yc) * (y > yc)
    
    ξ = numerator ./ denominator
    x = ξ .* Nx + xᵏ

    jac_num = @. (wa * wc * λ * (yc - ya) * (y <= yc) + wb * wc * (one(T) - λ) * (yb - yc) * (y > yc)) * Nx
    logabsdet = log.(jac_num) - T(2.) .* log.(abs.(denominator))
    # Make the adjustment for elements outside of the [-B, B] bounding box
    return x, logabsdet
end


function monotonic_rational_quadratic_spline_forward(x, Nx, Ny, δ, δ⁺¹, xᵏ, yᵏ)  # Note this is the forward.
    T = eltype(x)
    ξ = (x - xᵏ) ./ Nx
    sₖ = Ny ./ Nx
    # ξ_one_minus_ξ = ξ .* (one(T) .- ξ)
    ξ_one_minus_ξ = max.(ξ .* (one(T) .- ξ), zero(T))
    numerator = Ny .* (sₖ .* abs2.(ξ) + δ .* ξ_one_minus_ξ)
    denominator = sₖ + ((δ + δ⁺¹ - T(2.) .* sₖ) .* ξ_one_minus_ξ)

    y = yᵏ + numerator ./ denominator

    jac_num = abs2.(sₖ) .* (δ⁺¹ .* abs2.(ξ) + T(2.) .* sₖ .* ξ_one_minus_ξ + δ .* abs2.(one(T) .- ξ))
    logabsdet = log.(jac_num) - T(2.) .* log.(denominator)

    # # Set values outside of the bounds to x with a logabsdet of 0. 25 -> 33 allocs
    mask = iszero.(ξ_one_minus_ξ)
    mask_neg = .!mask
    
    return y .* mask_neg + x .* mask, logabsdet .* mask_neg
end

function monotonic_rational_quadratic_spline_back(y, Nx, Ny, δ, δ⁺¹, xᵏ, yᵏ) 
    T = eltype(y)
    sₖ = Ny ./ Nx
    
    a = @. Ny * (sₖ - δ) + (y - yᵏ) * (δ⁺¹ + δ - T(2.) * sₖ)
    b = @. (Ny * δ) - (y - yᵏ) * (δ⁺¹ + δ - T(2.) * sₖ)
    c = @. -sₖ * (y - yᵏ)
    
    ξ = @. (T(2.) * c) / (-b - sqrt(abs2(b) - T(4.) * a * c))
    x = ξ .* Nx + xᵏ

    ξ_one_minus_ξ = ξ .* (one(T) .- ξ)
    denominator = @. sₖ + ((δ + δ⁺¹ - T(2.) * sₖ) * ξ_one_minus_ξ)

    jac_num = @. abs2(sₖ) * (δ⁺¹ * abs2(ξ) + T(2.) * sₖ * ξ_one_minus_ξ + δ * abs2(one(T) - ξ))
    logabsdet = -(log.(jac_num) - T(2.) .* log.(denominator))

    return x, logabsdet
end


forward(b::NeuralSpline{Linear, T}, x) where T = monotonic_rational_linear_spline_forward(x, params(b, x)...)
forward(b::Inverse{<:NeuralSpline{Linear, T}}, y) where T = monotonic_rational_linear_spline_back(y, params(b, y)...)

forward(b::NeuralSpline{Quadratic, T}, x) where T = monotonic_rational_quadratic_spline_forward(x, params(b, x)...)
forward(b::Inverse{<:NeuralSpline{Quadratic, T}}, y) where T = monotonic_rational_quadratic_spline_back(y, params(b, y)...)

(b::NeuralSpline)(x) = forward(b, x)[1]
(b::Inverse{<:NeuralSpline})(y) = forward(b, y)[1]

function Bijectors.with_logabsdet_jacobian(b::NeuralSpline, y::Real) 
    x, logabsdet = forward(b, y)
    return first(x), first(logabsdet)
end

function Bijectors.with_logabsdet_jacobian(b::Inverse{<:NeuralSpline}, y::Real) 
    x, logabsdet = forward(b, y)
    return first(x), first(logabsdet)
end
