module MvNormals

import Distributions: Distributions, ContinuousMultivariateDistribution, logpdf
import LinearAlgebra: LinearAlgebra, I, UpperTriangular, Diagonal, Hermitian, Cholesky 
import LinearAlgebra: cholesky!, cholesky, logdet
import ChainRulesCore: ChainRulesCore, Tangent, NoTangent, ZeroTangent, ProjectTo, rrule, @thunk

export MvNormal, IsoMvNormal, μ, Σ, product, logpdf, logpdfnan

abstract type AbstractMvNormal{T} <: ContinuousMultivariateDistribution end

Base.eltype(::AbstractMvNormal{T}) where T = T

Base.length(x::AbstractMvNormal) = x.n

Base.:+(d₁::AbstractMvNormal, d₂::AbstractMvNormal) = MvNormal(μ(d₁) + μ(d₂), cholesky(Σ(d₁) + Σ(d₂)).U)

Base.:(≈)(d₁::AbstractMvNormal, d₂::AbstractMvNormal) = (μ(d₁) ≈ μ(d₂)) && (Σ(d₁) ≈ Σ(d₂))

Base.rand(d::AbstractMvNormal, n::Int64) = [rand(d) for _ in 1:n]

function Base.:&(d₁::AbstractMvNormal, d₂::AbstractMvNormal)
    c = cholesky!(Σ(d₁) + Σ(d₂))
    L⁻¹ = inv(c.L)
    Σ₁, Σ₂ = L⁻¹ * Σ(d₁), L⁻¹ * Σ(d₂)
    μ₁, μ₂ = L⁻¹ * μ(d₁), L⁻¹ * μ(d₂)
    return MvNormal(Σ₂'μ₁ + Σ₁'μ₂, cholesky!(Σ₁'Σ₂).U)
end

function product(d₁::AbstractMvNormal, ds...)
    if isempty(ds) return d₁ end
    Λ = d₁ |> cholesky |> inv
    m = Λ * μ(d₁)
    for d in ds
        Λᵢ = d |> cholesky |> inv
        m .+= Λᵢ * μ(d)
        Λ .+= Λᵢ
    end
    Σ = Λ |> Hermitian |> cholesky! |> inv
    return MvNormal(Σ * m, cholesky!(Σ).U)
end

include("utilities.jl")
include("mvn.jl")
include("isomvn.jl")

end # MvNormals