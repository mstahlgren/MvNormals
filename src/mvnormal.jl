using LinearAlgebra: I, Hermitian, cholesky, logdet
import StatsBase

export DenseMvNormal, DenseZeroMeanMvNormal, IsoMvNormal
export logpdf, sample, mu, sigma, size

abstract type MvNormal <: Distribution end

abstract type ZeroMeanMvNormal <: MvNormal end

struct IsoMvNormal <: ZeroMeanMvNormal
    n::Int64
end

struct DenseZeroMeanMvNormal <: ZeroMeanMvNormal
    Σ::Matrix
end

struct DenseMvNormal <: MvNormal
    μ::Vector
    Σ::Matrix
 end

Base.:size(x::MvNormal) = size(sigma(x), 1)

Base.:size(x::IsoMvNormal) = x.n

mu(x::MvNormal) = x.μ

mu(x::ZeroMeanMvNormal) = x |> size |> zeros # Zero struct could be useful

sigma(x::MvNormal) = x.Σ

sigma(x::IsoMvNormal) = 1.0I(x |> size)

Base.:+(d₁::MvNormal, d₂::MvNormal) = MvNormal(mu(d₁) .+ mu(d₂), sigma(d₁) .+ sigma(d₂))

Base.:+(d₁::ZeroMeanMvNormal, d₂::MvNormal) = MvNormal(mu(d₂), sigma(d₁) .+ sigma(d₂))

Base.:+(d₁::MvNormal, d₂::ZeroMeanMvNormal) = d₂ + d₁ 

Base.:+(d₁::ZeroMeanMvNormal, d₂::ZeroMeanMvNormal) = ZeroMeanMvNormal(sigma(d₁) .+ sigma(d₂))

Base.:+(v::Vector, d::MvNormal) = MvNormal(v .+ mu(d), sigma(d))

Base.:+(v::Vector, d::ZeroMeanMvNormal) = MvNormal(v, sigma(d))

Base.:*(m::Matrix, d::MvNormal) = MvNormal(m*mu(d), m*sigma(d)*m')

Base.:*(m::Matrix, d::ZeroMeanMvNormal) = ZeroMeanMvNormal(m*sigma(d)*m')

Base.:*(m::Matrix, ::IsoMvNormal) = ZeroMeanMvNormal(m*m')

Base.:(==)(x::MvNormal, y::MvNormal) = (mu(x) == mu(y)) && (sigma(x) == sigma(y))

function Base.:&(d₁::MvNormal, d₂::MvNormal)
    c = (sigma(d₁) + sigma(d₂)) |> Hermitian |> cholesky
    Σ₁ = c.L\sigma(d₁); Σ₂ = c.L\sigma(d₂)
    μ₁ = c.L\mu(d₁); μ₂ = c.L\mu(d₂)
    return DenseMvNormal(Σ₂'*μ₁ .+ Σ₁'*μ₂, Σ₁'*Σ₂)
end

function logpdf(x::MvNormal, o::Vector)
    C = x |> sigma |> Hermitian |> cholesky
    ld = log(2π)*length(o) + 2*logdet(C.U)
    le = C\(o-x.μ)
    return -0.5*(ld + le'le)
end

logpdf(x::IsoMvNormal, o::Vector) = -0.5*(log(2π)*size(x) + o'o)

StatsBase.:sample(x::IsoMvNormal, n::Int64) = [randn(size(x)) for _ in 1:n]
