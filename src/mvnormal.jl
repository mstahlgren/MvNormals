using LinearAlgebra: LowerTriangular, Symmetric, I, UniformScaling, Cholesky, cholesky, logdet, Diagonal, Hermitian
import ChainRulesCore: ChainRulesCore, Tangent, NoTangent, ZeroTangent, ProjectTo, rrule, @thunk

export MvNormal, IsoMvNormal, μ, Σ, logpdf, logpdfnan
export product, product2

struct Identity <: AbstractMatrix{Float64} end

struct MvNormal2{V <: AbstractVector, M <: AbstractMatrix} 
    n::Int
    μ::V
    L::M
end

const IsoMvNormal2 = MvNormal2{Zero, Identity}

MvNormal2(n::Integer) = MvNormal2(n, Zero(), Identity())

MvNormal2(μ, Σ::Matrix) = MvNormal2(length(μ), μ, cholesky(Σ).L)

MvNormal2(μ, D::Diagonal) = MvNormal2(length(μ), μ, LowerTriangular(D))

MvNormal2(μ, L) = MvNormal2(length(μ), μ, L)

μ(x::MvNormal2) = x.μ

Σ(x::MvNormal2) = x.L * x.L'

Base.size(x::MvNormal2) = x.n

Base.:+(d₁::MvNormal2, d₂::MvNormal2) = MvNormal2(μ(d₁) + μ(d₂), Σ(d₁) + Σ(d₂))

Base.:+(v::AbstractVector, d::MvNormal2) = MvNormal2(v + μ(d), d.L)

Base.:*(m::AbstractMatrix, d::MvNormal2) = MvNormal2(m * μ(d), m * d.L)

Base.:(≈)(d₁::MvNormal2, d₂::MvNormal2) = (μ(d₁) ≈ μ(d₂)) && (Σ(d₁) ≈ Σ(d₂))

function Base.:&(d₁::MvNormal2, d₂::MvNormal2)
    c = cholesky(Σ(d₁) + Σ(d₂))
    L⁻¹ = inv(c.L)
    Σ₁, Σ₂ = L⁻¹ * Σ(d₁), L⁻¹ * Σ(d₂)
    μ₁, μ₂ = L⁻¹ * μ(d₁), L⁻¹ * μ(d₂)
    return MvNormal2(Σ₂'μ₁ + Σ₁'μ₂, Σ₁'Σ₂)
end

function product(d₁::MvNormal2, ds...)
    Σ = d₁.L |> Cholesky |> inv
    μ = Σ * d₁.μ 
    for d in ds
        Σᵢ = d₁.L |> Cholesky |> inv
        μ .+= Σᵢ * d.μ
        Σ .+= Σᵢ
    end
    Σ⁻¹ = Σ |> cholesky |> inv
    return MvNormal2(Σ⁻¹ * μ, cholesky(Σ⁻¹).L)
end

# Logpdf

function logpdf(d::MvNormal2{T,S}, x::AbstractVector) where {T,S}
    ld = log(2π)*size(d) + 2*logdet(d.L)
    le = d.L\(x .- d.μ)
    return -0.5*(ld + le'le)
end

function ChainRulesCore.rrule(::typeof(logpdf), d::MvNormal2{T,S}, x::AbstractVector) where {T,S}
    z = x - μ(d)
    cz = d.L\z
    ld = log(2π)*size(d) + 2*logdet(d.L)
    logpdf_pb(Δy) = begin
        Σₜ = @thunk begin
            A = inv(c) .- cz .* cz'
            ProjectTo(d.Σ)(-0.5 .* Δy .* (2A .- Diagonal(A)))
        end
        (NoTangent(), Tangent{MvNormal2}(μ = Δy .* cz, Σ = Σₜ), -Δy .* cz)
    end
    return -0.5*(ld + z'cz), logpdf_pb
end

function logpdf(d::IsoMvNormal2, x::AbstractVector)
    return -0.5*(log(2π)*size(d) + x'x)
end

function ChainRulesCore.rrule(::typeof(logpdf), d::IsoMvNormal2, x::AbstractVector)
    isomvn_dll(s) =  (NoTangent(), ZeroTangent(), -s .* x)
    return logpdf(d, x), isomvn_dll
end

# Logpdfnan

function logpdfnan(d::MvNormal2{T,S}, x::AbstractVector, nums = .!isnan.(x)) where {T,S}
    if !any(nums) return 0.0 end
    if all(nums) return logpdf(d, x) end
    if S <: Diagonal return logpdf(MvNormal(μ(d)[nums], Diagonal(Σ(d).diag[nums])), x[nums]) end
    return logpdf(MvNormal(μ(d)[nums], Σ(d)[nums, nums]), x[nums])
end

function ChainRulesCore.rrule(::typeof(logpdfnan), d::MvNormal2{T,S}, x::AbstractVector, nums = .!isnan.(x)) where {T,S} 
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

Base.:rand(d::MvNormal2, n::Int64) = [μ(x) + d.L * randn(size(d)) for _ in 1:n]

Base.:rand(d::IsoMvNormal2, n::Int64) = [randn(size(d)) for _ in 1:n]