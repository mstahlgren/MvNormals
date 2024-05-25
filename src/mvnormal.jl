using LinearAlgebra: I, UniformScaling, cholesky, logdet, Diagonal
import ChainRulesCore: ChainRulesCore, Tangent, NoTangent, ZeroTangent, ProjectTo, rrule, @thunk

export MvNormal, IsoMvNormal, μ, Σ, logpdf, logpdfnan

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

# Logpdf

function logpdf(d::MvNormal{T,S}, x::AbstractVector) where {T,S}
    c = d |> Σ |> cholesky
    ld = log(2π)*size(d) + 2*logdet(c.U)
    le = c.U\(x .- d.μ)
    return -0.5*(ld + le'le)
end

function ChainRulesCore.rrule(::typeof(logpdf), d::MvNormal{T,S}, x::AbstractVector) where {T,S}
    c = d |> Σ |> cholesky
    z = x - μ(d)
    cz = c\z
    ld = log(2π)*size(d) + 2*logdet(c.U)
    logpdf_pb(Δy) = begin
        Σₜ = @thunk begin
            A = inv(c) .- cz .* cz'
            ProjectTo(d.Σ)(-0.5 .* Δy .* (2A .- Diagonal(A)))
        end
        (NoTangent(), Tangent{MvNormal}(μ = Δy .* cz, Σ = Σₜ), -Δy .* cz)
    end
    return -0.5*(ld + z'cz), logpdf_pb
end

function logpdf(d::IsoMvNormal, x::AbstractVector)
    return -0.5*(log(2π)*size(d) + x'x)
end

function ChainRulesCore.rrule(::typeof(logpdf), d::IsoMvNormal, x::AbstractVector)
    isomvn_dll(s) =  (NoTangent(), ZeroTangent(), -s .* x)
    return logpdf(d, x), isomvn_dll
end

# Logpdfnan

function logpdfnan(d::MvNormal{T,S}, x::AbstractVector, nums = .!isnan.(x)) where {T,S}
    if !any(nums) return 0.0 end
    if all(nums) return logpdf(d, x) end
    if S <: Diagonal return logpdf(MvNormal(μ(d)[nums], Diagonal(Σ(d).diag[nums])), x[nums]) end
    return logpdf(MvNormal(μ(d)[nums], Σ(d)[nums, nums]), x[nums])
end

function ChainRulesCore.rrule(::typeof(logpdfnan), d::MvNormal{T,S}, x::AbstractVector, nums = .!isnan.(x)) where {T,S} 
    if !any(nums) return 0.0, Δy -> NoTangent(), ZeroTangent(), ZeroTangent() end
    if all(nums) return rrule(logpdf, d, x) end
    y, y_pb = rrule(logpdf, MvNormal(μ(d)[nums], S <: Diagonal ? Diagonal(Σ(d).diag[nums]) : Σ(d)[nums, nums]), x[nums])
    logpdfnan_pb(Δy) = begin
        _, Δd, Δx = y_pb(Δy)
        μ₁ = @thunk begin μ₀ = zeros(size(μ(d))); view(μ₀,nums) .= Δd.μ; μ₀ end
        Σ₁ = @thunk begin Σ₀ = zeros(size(Σ(d))); view(Σ₀,nums,nums) .= Δd.Σ; ProjectTo(Σ(d))(Σ₀) end
        x₁ = @thunk begin x₀ = zeros(size(x)); view(x₀,nums) .= Δx; x₀ end
        (NoTangent(), Tangent{MvNormal}(μ = μ₁, Σ = Σ₁), x₁)
    end
    return y, logpdfnan_pb
end

# Rand

Base.:rand(x::MvNormal, n::Int64) = [μ(x) + cholesky(Σ(x)).L * randn(size(x)) for _ in 1:n]

Base.:rand(x::IsoMvNormal, n::Int64) = [randn(size(x)) for _ in 1:n]