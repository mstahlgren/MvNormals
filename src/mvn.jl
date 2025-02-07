struct MvNormal{T, M, S} <: AbstractMvNormal{T}
    μ::M
    σ::S
end

MvNormal(μ, σ::LowerTriangular) = MvNormal{eltype(μ), typeof(μ), typeof(σ)}(μ, σ)

MvNormal(μ, σ) = MvNormal(μ, LowerTriangular(σ))

Base.getindex(d::MvNormal, idxs) = MvNormal(d.μ[idxs], d.σ[idxs, idxs])

function ChainRulesCore.rrule(::typeof(logpdf), d::MvNormal, x)
    z, s, c = x - d.μ, log(2π)*length(d), Cholesky(d.σ.data, :L, 0)
    L⁻¹z, Σ⁻¹z = c.L\z, c\z
    logpdf_pb(Δy) = begin
        dx = @thunk -Δy .* Σ⁻¹z
        dμ = @thunk Δy .* Σ⁻¹z
        dσ = @thunk ProjectTo(d.σ)(Δy .* (Σ⁻¹z .* L⁻¹z' .- inv(Diagonal(d.σ))))
        (NoTangent(), Tangent{MvNormal}(μ = dμ, σ = dσ), dx)
    end
    return -0.5*(s + logdet(c) + AᵀA(L⁻¹z)), logpdf_pb
end

function logpdftest(d::MvNormal, x, h = 0.00001)
    ll, llΔx = logpdf(d, x), zeros(length(d))
    llΔμ = zeros(length(d))
    llΔσ = LowerTriangular(zeros(length(d), length(d))) 
    for i in 1:length(d)
        Δμ = copy(d.μ); Δμ[i] += h
        llΔμ[i] = (logpdf(MvNormal(Δμ, σ(d)), x) - ll)/h
        for j in 1:i
            Δσ = copy(d.σ); Δσ[i,j] += h
            llΔσ[i,j] = (logpdf(MvNormal(μ(d), Δσ), x) - ll)/h
        end
        Δx = copy(x); Δx[i] += h
        llΔx[i] = (logpdf(d, Δx) - ll)/h
    end
    return (Tangent{MvNormal}(μ = llΔμ, σ = llΔσ), llΔx)
end