abstract type AbstractMvNormal{T} end

μ(d::AbstractMvNormal) = d.μ

σ(d::AbstractMvNormal) = d.σ

Σ(d::AbstractMvNormal) = d |> σ |> AAᵀ

Base.eltype(::AbstractMvNormal{T}) where T = T

Base.length(d::AbstractMvNormal) = d |> μ |> length

Base.:+(x::AbstractVector, d::T) where T <: AbstractMvNormal = T(x + μ(d), σ(d))

function Base.rand(d::AbstractMvNormal)
    N, T = length(d), eltype(d)
    x = isbits(μ(d)) ? randn(SVector{N, T}) : randn(T, N)
    return μ(d) + σ(d) * x
end

function Base.:&(d₁::AbstractMvNormal, d₂::AbstractMvNormal)
    c = cholesky(Σ(d₁) + Σ(d₂))
    Σ₁, Σ₂ = c.L \ Σ(d₁), c.L \ Σ(d₂)
    μ₁, μ₂ = c.L \ μ(d₁), c.L \ μ(d₂)
    return MvNormal(Σ₂'μ₁ + Σ₁'μ₂, cholesky(Σ₁'Σ₂).L)
end

function logpdf(d::AbstractMvNormal, x)
    s, c = log(2π)*length(d), Cholesky(σ(d), :L, 0)
    return -0.5*(s + logdet(c) + AᵀA(c.L \ (x - d.μ)))
end
