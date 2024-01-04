module MvNormals

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

DenseMvNormal(x::IsoMvNormal) = DenseMvNormal(x |> mu, x |> sigma)

Base.:size(x::MvNormal) = size(sigma(x), 1)

Base.:size(x::IsoMvNormal) = x.n

mu(x::MvNormal) = x.μ

mu(x::ZeroMeanMvNormal) = x |> size |> zeros # Zero struct could be useful

sigma(x::MvNormal) = x.Σ

sigma(x::IsoMvNormal) = 1.0I(x |> size)

Base.:+(d₁::MvNormal, d₂::MvNormal) = DenseMvNormal(mu(d₁) .+ mu(d₂), sigma(d₁) .+ sigma(d₂))

Base.:+(d₁::ZeroMeanMvNormal, d₂::MvNormal) = DenseMvNormal(mu(d₂), sigma(d₁) .+ sigma(d₂))

Base.:+(d₁::MvNormal, d₂::ZeroMeanMvNormal) = d₂ + d₁ 

Base.:+(d₁::ZeroMeanMvNormal, d₂::ZeroMeanMvNormal) = DenseZeroMeanMvNormal(sigma(d₁) .+ sigma(d₂))

Base.:+(v::Vector, d::MvNormal) = DenseMvNormal(v .+ mu(d), sigma(d))

Base.:+(v::Vector, d::ZeroMeanMvNormal) = DenseMvNormal(v, sigma(d))

Base.:*(m::Matrix, d::MvNormal) = DenseMvNormal(m*mu(d), m*sigma(d)*m')

Base.:*(m::Matrix, d::ZeroMeanMvNormal) = DenseZeroMeanMvNormal(m*sigma(d)*m')

Base.:*(m::Matrix, ::IsoMvNormal) = DenseZeroMeanMvNormal(m*m')

Base.:(==)(x::MvNormal, y::MvNormal) = (mu(x) == mu(y)) && (sigma(x) == sigma(y))

function Base.:&(d₁::MvNormal, d₂::MvNormal)
    c = (sigma(d₁) + sigma(d₂)) |> Hermitian |> cholesky
    Σ₁ = c.L\sigma(d₁); Σ₂ = c.L\sigma(d₂)
    μ₁ = c.L\mu(d₁); μ₂ = c.L\mu(d₂)
    return DenseMvNormal(Σ₂'*μ₁ .+ Σ₁'*μ₂, Σ₁'*Σ₂)
end

function logpdf(x::MvNormal, o::AbstractVector)
    C = x |> sigma |> Hermitian |> cholesky
    ld = log(2π)*length(o) + 2*logdet(C.U)
    le = C\(o-x.μ)
    return -0.5*(ld + le'le)
end

logpdf(x::IsoMvNormal, o::AbstractVector) = -0.5*(log(2π)*size(x) + o'o)

Base.:rand(x::IsoMvNormal, n::Int64) = [randn(size(x)) for _ in 1:n]

#@btime logpdf(mvn, zeros(10))
#  1.333 μs (19 allocations: 1.61 KiB)
#-9.189385332046726

end # MvNormals