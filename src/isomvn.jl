struct IsoMvNormal{T} <: AbstractMvNormal{T}
    n::Int
end

IsoMvNormal(T, n) = IsoMvNormal{T}(n)

IsoMvNormal(n::Integer) = IsoMvNormal{Float64}(n)

μ(d::IsoMvNormal) = zeros(d.n)

Σ(d::IsoMvNormal) = 1.0I(d.n)

Base.:+(v::AbstractVector, d::IsoMvNormal) = MvNormal(v , d.n |> ones |> Diagonal)

Base.:*(m::AbstractMatrix, d::IsoMvNormal) = MvNormal(zeros(d.n), cholesky(AAᵀ(m)).U)

Base.rand(d::IsoMvNormal) = randn(eltype(d), d.n)

LinearAlgebra.cholesky(d::IsoMvNormal) = Cholesky(d.n |> ones |> Diagonal, :U, 0)

function Distributions.logpdf(d::IsoMvNormal, x::AbstractVector)
    return -0.5*(log(2π)*length(d) + x'x)
end

function ChainRulesCore.rrule(::typeof(logpdf), d::IsoMvNormal, x::AbstractVector)
    isomvn_dll(Δy) = (NoTangent(), ZeroTangent(), -Δy .* x)
    return logpdf(d, x), isomvn_dll
end

function logpdfnan(d::IsoMvNormal, x::AbstractVector, nums = .!isnan.(x))
    if !any(nums) return 0.0 end
    if all(nums) return logpdf(d, x) end
    return logpdf(IsoMvNormal(sum(nums)), x[nums])
end

function ChainRulesCore.rrule(::typeof(logpdfnan), d::IsoMvNormal, x::AbstractVector, nums = .!isnan.(x))
    if !any(nums) return 0.0, Δy -> NoTangent(), ZeroTangent(), ZeroTangent() end
    if all(nums) return rrule(logpdf, d, x) end
    y, y_pb = rrule(IsoMvNormal(sum(nums)), x[nums])
    logpdfnan_pb(Δy) = begin
        x₁ = @thunk begin x₀ = zeros(d.n); view(x₀,nums) .= y_pb(Δy); x₀ end
        (NoTangent(), NoTangent(), x₁)
    end
    return y, logpdfnan_pb
end