import LinearAlgebra: UpperTriangular, Diagonal, Hermitian, Cholesky, cholesky, cholesky!, logdet
import ChainRulesCore: ChainRulesCore, Tangent, NoTangent, ZeroTangent, ProjectTo, rrule, @thunk
import Distributions: Distributions, ContinuousMultivariateDistribution, logpdf

export MvNormal, IsoMvNormal, μ, Σ, product, logpdf, logpdfnan

struct MvNormal{V, M} <: ContinuousMultivariateDistribution
    n::Int
    μ::V
    U::M
end

const IsoMvNormal = MvNormal{Zero, Identity}

MvNormal(n::Integer) = MvNormal(n, Zero(), Identity())

MvNormal(μ, Σ::Matrix) = MvNormal(length(μ), μ, cholesky(Σ).U)

MvNormal(μ, U) = MvNormal(length(μ), μ, U)

μ(x::MvNormal) = x.μ

Σ(x::MvNormal) = AᵀA(x.U)

Base.length(x::MvNormal) = x.n

Base.:+(d₁::MvNormal, d₂::MvNormal) = MvNormal(μ(d₁) + μ(d₂), Σ(d₁) + Σ(d₂))

Base.:+(v::AbstractVector, d::MvNormal) = MvNormal(v + μ(d), d.U)

Base.:*(m::AbstractMatrix, d::MvNormal) = MvNormal(m * μ(d), AᵀA(d.U * m'))

Base.:(≈)(d₁::MvNormal, d₂::MvNormal) = (μ(d₁) ≈ μ(d₂)) && (Σ(d₁) ≈ Σ(d₂))

function Base.:&(d₁::MvNormal, d₂::MvNormal)
    c = cholesky!(Σ(d₁) + Σ(d₂))
    L⁻¹ = inv(c.L)
    Σ₁, Σ₂ = L⁻¹ * Σ(d₁), L⁻¹ * Σ(d₂)
    μ₁, μ₂ = L⁻¹ * μ(d₁), L⁻¹ * μ(d₂)
    return MvNormal(Σ₂'μ₁ + Σ₁'μ₂, cholesky!(Σ₁'Σ₂).U)
end

function product(d₁::MvNormal, ds...)
    if isempty(ds) return d₁ end
    Λ = Cholesky(d₁.U, :U, 0) |> inv
    μ = Λ * d₁.μ 
    for d in ds
        Λᵢ = Cholesky(d.U, :U, 0) |> inv
        μ .+= Λᵢ * d.μ
        Λ .+= Λᵢ
    end
    Σ = Λ |> Hermitian |> cholesky! |> inv
    return MvNormal(Σ * μ, cholesky!(Σ).U)
end

# Logpdf

function Distributions.logpdf(d::MvNormal{T,S}, x::AbstractVector) where {T,S}
    ld = log(2π)*length(d) + 2*logdet(d.U)
    le = d.U'\(x .- d.μ)
    return -0.5*(ld + le'le)
end

function ChainRulesCore.rrule(::typeof(logpdf), d::MvNormal{T,S}, x::AbstractVector) where {T,S}
    z = x .- d.μ
    C = Cholesky(d.U, :U, 0)
    L⁻¹z = d.U'\z
    Σ⁻¹z = C\z
    ld = log(2π)*length(d) + 2*logdet(d.U)
    logpdf_pb(Δy) = begin
        Σₜ = @thunk ProjectTo(d.U)(Δy .* (L⁻¹z * Σ⁻¹z' .- inv(Diagonal(d.U))))
        (NoTangent(), Tangent{MvNormal}(n = ZeroTangent(), μ = Δy .* Σ⁻¹z, U = Σₜ), -Δy .* Σ⁻¹z)
    end
    return -0.5*(ld + AᵀA(L⁻¹z)), logpdf_pb
end

function Distributions.logpdf(d::IsoMvNormal, x::AbstractVector)
    return -0.5*(log(2π)*length(d) + x'x)
end

function ChainRulesCore.rrule(::typeof(logpdf), d::IsoMvNormal, x::AbstractVector)
    isomvn_dll(Δy) = (NoTangent(), ZeroTangent(), -Δy .* x)
    return logpdf(d, x), isomvn_dll
end

# Logpdfnan

function logpdfnan(d::MvNormal{T,S}, x::AbstractVector, nums = .!isnan.(x)) where {T,S}
    if !any(nums) return 0.0 end
    if all(nums) return logpdf(d, x) end
    return logpdf(MvNormal(d.μ[nums], S(S <: Diagonal ? d.U.diag[nums] : d.U[nums,nums])), x[nums])
end

function ChainRulesCore.rrule(::typeof(logpdfnan), d::MvNormal{T,S}, x::AbstractVector, nums = .!isnan.(x)) where {T,S} 
    if !any(nums) return 0.0, Δy -> NoTangent(), ZeroTangent(), ZeroTangent() end
    if all(nums) return rrule(logpdf, d, x) end
    y, y_pb = rrule(logpdf, MvNormal(d.μ[nums], S(S <: Diagonal ? d.U.diag[nums] : d.U[nums,nums])), x[nums])
    logpdfnan_pb(Δy) = begin
        _, Δd, Δx = y_pb(Δy)
        μ₁ = @thunk begin μ₀ = zeros(size(d.μ)); view(μ₀,nums) .= Δd.μ; μ₀ end
        U₁ = @thunk begin U₀ = zeros(size(d.U)); view(U₀,nums,nums) .= Δd.U; ProjectTo(d.U)(U₀) end
        x₁ = @thunk begin x₀ = zeros(size(x)); view(x₀,nums) .= Δx; x₀ end
        (NoTangent(), Tangent{MvNormal}(n = NoTangent(), μ = μ₁, U = U₁), x₁)
    end
    return y, logpdfnan_pb
end

# Rand

Base.rand(d::MvNormal, n::Int64) = [rand(d) for _ in 1:n]

Base.rand(d::MvNormal) = d.μ .+ d.U' * randn(length(d))