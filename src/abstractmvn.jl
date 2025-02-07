abstract type AbstractMvNormal{T} end

μ(d::AbstractMvNormal) = d.μ

σ(d::AbstractMvNormal) = d.σ

Σ(d::AbstractMvNormal) = d |> σ |> AAᵀ

Base.eltype(::AbstractMvNormal{T}) where T = T

Base.length(d::AbstractMvNormal) = d |> μ |> length

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
    z, s, c = x - d.μ, log(2π)*length(d), Cholesky(σ(d), :L, 0)
    return -0.5*(s + logdet(c) + AᵀA(c.L \ z))
end

function logpdfnan(d::AbstractMvNormal, x, nums)
    if !any(nums) return eltype(d)(0) end
    if all(nums) return logpdf(d, x) end
    return logpdf(d[nums], x[nums])
end

function ChainRulesCore.rrule(::typeof(logpdfnan), d::AbstractMvNormal, x, nums)
    if !any(nums) return eltype(d)(0), Δy -> NoTangent(), ZeroTangent(), ZeroTangent() end
    if all(nums) return rrule(logpdf, d, x) end
    y, y_pb = rrule(logpdf, d[nums], x[nums])
    logpdfnan_pb(Δy) = begin
        _, Δd, Δx = y_pb(Δy)
        x₁ = @thunk begin x₀ = zeros(eltype(d), d |> length); view(x₀, nums) .= Δx; x₀ end
        μ₁ = @thunk begin μ₀ = zeros(eltype(d), d |> length); view(μ₀, nums) .= Δd.μ; μ₀ end
        σ₁ = @thunk begin σ₀ = zeros(eltype(d), d.σ |> size); view(σ₀, nums, nums) .= Δd.σ; U₀ end
        (NoTangent(), Tangent{typeof(d)}(μ = μ₁, σ = σ₁), x₁)
    end
    return y, logpdfnan_pb
end
