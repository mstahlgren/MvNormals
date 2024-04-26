using LinearAlgebra: I, UniformScaling, cholesky, logdet, Diagonal
import ChainRulesCore: ChainRulesCore, Tangent, NoTangent, ZeroTangent

export MvNormal, IsoMvNormal, μ, Σ, logpdf

# TODO: Make compatible with Distributions package

struct MvNormal{T, S}
    n::Int
    μ::T
    Σ::S
end

const IsoMvNormal = MvNormal{Zero, UniformScaling{Bool}}

MvNormal(n::Integer) = MvNormal(n, Zero(), I)

MvNormal(mu, sigma) = MvNormal(length(mu), mu, sigma)

μ(x::MvNormal) = x.μ

Σ(x::MvNormal) = x.Σ

Base.size(x::MvNormal) = x.n

Base.:+(d₁::MvNormal, d₂::MvNormal) = MvNormal(size(d₁), μ(d₁) + μ(d₂), Σ(d₁) + Σ(d₂))

Base.:+(v::AbstractVector, d::MvNormal) = MvNormal(size(d), v + μ(d), Σ(d))

Base.:*(m::AbstractMatrix, d::MvNormal) = MvNormal(size(d), m * μ(d), m * Σ(d) * m')

Base.:(≈)(d₁::MvNormal, d₂::MvNormal) = (μ(d₁) ≈ μ(d₂)) && (Σ(d₁) ≈ Σ(d₂))

function Base.:&(d₁::MvNormal, d₂::MvNormal)
    c = cholesky(Σ(d₁) + Σ(d₂))
    L⁻¹ = inv(c.L)
    Σ₁, Σ₂ = L⁻¹ * Σ(d₁), L⁻¹ * Σ(d₂)
    μ₁, μ₂ = L⁻¹ * μ(d₁), L⁻¹ * μ(d₂)
    return MvNormal(size(d₁), Σ₂'μ₁ + Σ₁'μ₂, Σ₁'Σ₂)
end

function logpdf(d::MvNormal, x::AbstractVector)
    c = d |> Σ |> cholesky
    ld = log(2π)*size(d) + 2*logdet(c.U)
    le = c.U\(x - d.μ)
    return -0.5*(ld + le'le)
end

function ChainRulesCore.rrule(::typeof(logpdf), d::MvNormal, x::AbstractVector)
    c = d |> Σ |> cholesky
    z = x - μ(d)
    cz = c\z
    ld = log(2π)*size(d) + 2*logdet(c.U)
    A = inv(c) .- cz .* cz'
    mvn_dll(s) = (NoTangent(), Tangent{MvNormal}(μ = s .* cz, Σ = -0.5 .* s .* (2A .- Diagonal(A))), -s .* cz)
    return -0.5*(ld + z'cz), mvn_dll
end

function logpdf(d::IsoMvNormal, x::AbstractVector)
    return -0.5*(log(2π)*size(d) + x'x)
end

function ChainRulesCore.rrule(::typeof(logpdf), d::IsoMvNormal, x::AbstractVector)
    isomvn_dll(s) =  (NoTangent(), ZeroTangent(), -s .* x)
    return logpdf(d, x), isomvn_dll
end

Base.:rand(x::MvNormal, n::Int64) = [μ(x) + cholesky(Σ(x)).L * randn(size(x)) for _ in 1:n]

Base.:rand(x::IsoMvNormal, n::Int64) = [randn(size(x)) for _ in 1:n]