struct MvNormal{T, V, M <: AbstractMatrix} <: AbstractMvNormal{T}
    n::Int
    μ::V
    U::M
end

MvNormal(μ, Σ::Matrix) = MvNormal(μ, cholesky(Σ).U)

MvNormal(μ, U) = MvNormal{eltype(μ), typeof(μ), typeof(U)}(length(μ), μ, U)

μ(x::MvNormal) = x.μ

Σ(x::MvNormal) = AᵀA(x.U)

Base.:+(v::AbstractVector, d::MvNormal) = MvNormal(v + d.μ, d.U)

Base.:*(m::AbstractMatrix, d::MvNormal) = MvNormal(m * d.μ, cholesky(AᵀA(d.U * m')).U)

Base.rand(d::MvNormal) = d.μ .+ d.U' * randn(eltype(d), length(d))

LinearAlgebra.cholesky(d::MvNormal) = Cholesky(d.U, :U, 0)

# Logpdf

function Distributions.logpdf(d::MvNormal, x::AbstractVector)
    ld = log(2π)*length(d) + 2*logdet(d.U)
    le = d.U'\(x .- d.μ)
    return -0.5*(ld + le'le)
end

function ChainRulesCore.rrule(::typeof(logpdf), d::MvNormal, x::AbstractVector)
    C = cholesky(d)
    z = x .- d.μ
    L⁻¹z = d.U'\z
    Σ⁻¹z = C\z
    ld = log(2π)*length(d) + 2*logdet(d.U)
    logpdf_pb(Δy) = begin
        Uₜ = @thunk ProjectTo(d.U)(Δy .* (L⁻¹z * Σ⁻¹z' .- inv(Diagonal(d.U))))
        (NoTangent(), Tangent{MvNormal}(n = NoTangent(), μ = Δy .* Σ⁻¹z, U = Uₜ), -Δy .* Σ⁻¹z)
    end
    return -0.5*(ld + AᵀA(L⁻¹z)), logpdf_pb
end

# Logpdfnan

function logpdfnan(d::MvNormal{T,V,M}, x::AbstractVector, nums = .!isnan.(x)) where {T,V,M}
    if !any(nums) return 0.0 end
    if all(nums) return logpdf(d, x) end
    return logpdf(MvNormal(d.μ[nums], M(M <: Diagonal ? d.U.diag[nums] : d.U[nums,nums])), x[nums])
end

function ChainRulesCore.rrule(::typeof(logpdfnan), d::MvNormal{T,V,M}, x::AbstractVector, nums = .!isnan.(x)) where {T,V,M}
    if !any(nums) return 0.0, Δy -> NoTangent(), ZeroTangent(), ZeroTangent() end
    if all(nums) return rrule(logpdf, d, x) end
    y, y_pb = rrule(logpdf, MvNormal(d.μ[nums], M(M <: Diagonal ? d.U.diag[nums] : d.U[nums,nums])), x[nums])
    logpdfnan_pb(Δy) = begin
        _, Δd, Δx = y_pb(Δy)
        μ₁ = @thunk begin μ₀ = zeros(size(d.μ)); view(μ₀,nums) .= Δd.μ; μ₀ end
        U₁ = @thunk begin U₀ = zeros(size(d.U)); view(U₀,nums,nums) .= Δd.U; ProjectTo(d.U)(U₀) end
        x₁ = @thunk begin x₀ = zeros(size(x)); view(x₀,nums) .= Δx; x₀ end
        (NoTangent(), Tangent{MvNormal}(n = NoTangent(), μ = μ₁, U = U₁), x₁)
    end
    return y, logpdfnan_pb
end