using LinearAlgebra: I, Hermitian, cholesky, logdet
import StatsBase

export MvNormal, ZeroMeanMvNormal, IsoMvNormal 
export logpdf, sample, mu, sigma, size

abstract type AbstractMvNormal <: Distribution end

abstract type AbstractZeroMeanMvNormal <: AbstractMvNormal end

struct IsoMvNormal <: AbstractZeroMeanMvNormal
    n::Int64
end

struct ZeroMeanMvNormal <: AbstractZeroMeanMvNormal
    Σ::Matrix
end

struct MvNormal <: AbstractMvNormal
    μ::Vector
    Σ::Matrix
 end

Base.:size(x::AbstractMvNormal) = size(sigma(x), 1)

Base.:size(x::IsoMvNormal) = x.n

mu(x::AbstractMvNormal) = x.μ

mu(x::ZeroMeanMvNormal) = x |> size |> zeros # Zero struct could be useful

sigma(x::AbstractMvNormal) = x.Σ

sigma(x::IsoMvNormal) = 1.0I(x |> size)

Base.:+(d₁::AbstractMvNormal, d₂::AbstractMvNormal) = MvNormal(mu(d₁) .+ mu(d₂), sigma(d₁) .+ sigma(d₂))

Base.:+(d₁::AbstractZeroMeanMvNormal, d₂::AbstractMvNormal) = MvNormal(mu(d₂), sigma(d₁) .+ sigma(d₂))

Base.:+(d₂::AbstractMvNormal, d₁::AbstractZeroMeanMvNormal) = d₁ + d₂

Base.:+(d₁::AbstractZeroMeanMvNormal, d₂::AbstractZeroMeanMvNormal) = ZeroMeanMvNormal(sigma(d₁) .+ sigma(d₂))

Base.:+(v::Vector, d::AbstractMvNormal) = MvNormal(v .+ mu(d), sigma(d))

Base.:+(v::Vector, d::AbstractZeroMeanMvNormal) = MvNormal(v, sigma(d))

Base.:*(m::Matrix, d::AbstractMvNormal) = MvNormal(m*mu(d), m*sigma(d)*m')

Base.:*(m::Matrix, d::AbstractZeroMeanMvNormal) = ZeroMeanMvNormal(m*sigma(d)*m')

Base.:*(m::Matrix, ::IsoMvNormal) = ZeroMeanMvNormal(m*m')

Base.:(==)(x::AbstractMvNormal, y::AbstractMvNormal) = (mu(x) == mu(y)) && (sigma(x) == sigma(y))

function Base.:&(d₁::AbstractMvNormal, d₂::AbstractMvNormal)
    c = (sigma(d₁) + sigma(d₂)) |> Hermitian |> cholesky
    Σ₁ = c.L\sigma(d₁); Σ₂ = c.L\sigma(d₂)
    μ₁ = c.L\mu(d₁); μ₂ = c.L\mu(d₂)
    return DenseMvNormal(Σ₂'*μ₁ .+ Σ₁'*μ₂, Σ₁'*Σ₂)
end

function logpdf(x::AbstractMvNormal, o::Vector)
    C = x |> sigma |> Hermitian |> cholesky
    ld = log(2π)*length(o) + 2*logdet(C.U)
    le = C\(o-x.μ)
    return -0.5*(ld + le'le)
end

logpdf(x::IsoMvNormal, o::Vector) = -0.5*(log(2π)*size(x) + o'o)

StatsBase.:sample(x::IsoMvNormal, n::Int64) = [randn(size(x)) for _ in 1:n]
