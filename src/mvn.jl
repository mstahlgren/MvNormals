struct MvNormal{T, M, S} <: AbstractMvNormal{T}
    μ::M
    σ::S
end

MvNormal(μ, σ::AbstractMatrix) = MvNormal(eltype(μ), μ, LowerTriangular(σ))

MvNormal(μ, σ::AbstractVector) = MvNormal(eltype(μ), μ, Diagonal(σ))

MvNormal(T, μ, σ) = MvNormal{T, typeof(μ), typeof(σ)}(μ, σ)

function logpdf(d::MvNormal{T, M, S}, x, b) where {T, M, S <: LowerTriangular}
    c = Cholesky(d.σ.data, :L, 0)
    b .= x .- d.μ
    BLAS.trsv!('L', 'N', 'N', d.σ.data, b)
    return -0.5*(log(2π)*length(d) + logdet(c) + AᵀA(b))
end

function logpdf(d::MvNormal{T, M, S}, x, b) where {T, M, S <: Diagonal}
    b .= (x .- d.μ) ./ d.σ.diag
    return -0.5*(log(2π)*length(d) + 2*sum(log(σᵢ) for σᵢ in d.σ.diag) + AᵀA(b))
end

function δlogpdfδσ(d::MvNormal{T, M, S}, x, b) where {T, M, S <: Diagonal}
    b .= (x .- d.μ) ./ d.σ.diag
    logpdf_pb(Δy) = begin b .= Δy .* (b.^2 .- 1.0) ./ d.σ.diag; Diagonal(b) end
    return -0.5*(log(2π)*length(d) + 2*sum(log(σᵢ) for σᵢ in d.σ.diag) + AᵀA(b)), logpdf_pb
end

function δlogpdfδμ(d::MvNormal{T, M, S}, x, b) where {T, M, S <: Diagonal}
    b .= (x .- d.μ) ./ d.σ.diag
    logpdf_pb(Δy) = begin b .= Δy .* b ./ d.σ.diag; b end
    return -0.5*(log(2π)*length(d) + 2*sum(log(σᵢ) for σᵢ in d.σ.diag) + AᵀA(b)), logpdf_pb
end

function ChainRulesCore.rrule(::typeof(logpdf), d::MvNormal{T, M, S}, x) where {T, M, S <: LowerTriangular}
    z, s, c = x - d.μ, log(2π)*length(d), Cholesky(d.σ.data, :L, 0)
    Σ⁻¹z = c\z; L⁻¹z = d.σ'*Σ⁻¹z
    logpdf_pb(Δy) = begin 
        dx = @thunk -Δy .* Σ⁻¹z
        dμ = @thunk Δy .* Σ⁻¹z
        dσ = @thunk ProjectTo(d.σ)(Δy .* (Σ⁻¹z .* L⁻¹z' .- inv(Diagonal(d.σ))))
        (NoTangent(), Tangent{MvNormal}(μ = dμ, σ = dσ), dx)
    end
    return -0.5*(s + logdet(c) + AᵀA(L⁻¹z)), logpdf_pb
end

function ChainRulesCore.rrule(::typeof(logpdf), d::MvNormal{T, M, S}, x) where {T, M, S <: Diagonal}
    s, c = log(2π)*length(d), Cholesky(d.σ, :L, 0)
    L⁻¹z = c.L\(x - d.μ)
    logpdf_pb(Δy) = begin
        dx = @thunk -Δy .* L⁻¹z ./ d.σ.diag
        dμ = @thunk Δy .* L⁻¹z ./ d.σ.diag
        dσ = @thunk Diagonal(Δy .* (L⁻¹z.^2 .- 1.0) ./ d.σ.diag)
        (NoTangent(), Tangent{MvNormal}(μ = dμ, σ = dσ), dx)
    end
    return -0.5*(s + logdet(c) + AᵀA(L⁻¹z)), logpdf_pb
end

function test(::typeof(logpdf), d::MvNormal{T, M, S}, x, h = 0.00001) where {T, M, S <: LowerTriangular}
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
    return (NoTangent(), Tangent{MvNormal}(μ = llΔμ, σ = llΔσ), llΔx)
end