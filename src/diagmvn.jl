struct DiagMvNormal{T, M, S} <: AbstractMvNormal{T}
    μ::M
    σ::S
end

DiagMvNormal(μ, σ::Diagonal) = DiagMvNormal{eltype(μ), typeof(μ), typeof(σ)}(μ, σ)

DiagMvNormal(μ, σ::AbstractVector) = DiagMvNormal(μ, Diagonal(σ))

Base.getindex(d::DiagMvNormal, idxs) = MvNormal(d.μ[idxs], Diagonal(diag(d.σ)[idxs]))

function ChainRulesCore.rrule(::typeof(logpdf), d::DiagMvNormal, x)
    s = log(2π)*length(d)
    L⁻¹z = (x - d.μ) ./ d.σ.diag
    logpdf_pb(Δy) = begin
        dx = @thunk -Δy .* L⁻¹z ./ d.σ.diag
        dμ = @thunk Δy .* L⁻¹z ./ d.σ.diag
        dσ = @thunk Diagonal(Δy .* (L⁻¹z .^2 .- 1.0) ./ d.σ.diag)
        (NoTangent(), Tangent{DiagMvNormal}(μ = dμ, σ = dσ), dx)
    end
    return -0.5*(s + 2*logdet(d.σ) + AᵀA(L⁻¹z)), logpdf_pb
end