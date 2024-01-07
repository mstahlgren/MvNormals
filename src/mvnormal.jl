using LinearAlgebra: I, UniformScaling, cholesky, logdet

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

Base.:(==)(d₁::MvNormal, d₂::MvNormal) = (μ(d₁) == μ(d₂)) && (Σ(d₁) == Σ(d₂))

function Base.:&(d₁::MvNormal, d₂::MvNormal)
    c = cholesky(Σ(d₁) + Σ(d₂))
    L⁻¹ = inv(c.L)
    Σ₁, Σ₂ = L⁻¹ * Σ(d₁), L⁻¹ * Σ(d₂)
    μ₁, μ₂ = L⁻¹ * μ(d₁), L⁻¹ * μ(d₂)
    return MvNormal(size(d₁), Σ₂'μ₁ + Σ₁'μ₂, Σ₁'Σ₂)
end

function logpdf(x::MvNormal, o::AbstractVector)
    c = cholesky(Σ(x))
    ld = log(2π)*length(o) + 2*logdet(c.U)
    le = c.U\(o - x.μ)
    return -0.5*(ld + le'le)
end

function logpdf(::IsoMvNormal, o::AbstractVector)
    ld = log(2π)*length(o)
    return -0.5*(ld + o'o)
end

Base.:rand(x::MvNormal, n::Int64) = [μ(x) + cholesky(Σ(x)).L * randn(size(x)) for _ in 1:n]

Base.:rand(x::IsoMvNormal, n::Int64) = [randn(size(x)) for _ in 1:n]