import LinearAlgebra: LinearAlgebra, Hermitian, Diagonal, UpperTriangular, I, Cholesky
import LinearAlgebra: cholesky, cholesky!, logdet
import ChainRulesCore: ChainRulesCore, ZeroTangent, NoTangent, Tangent, ProjectTo, rrule, @thunk
import Distributions: Distributions, logpdf

struct MvNormal{T, N, M, S, U}
    μ::M
    Σ::S
    C::U
end

MvNormal(T, N, μ, Σ, C) = MvNormal{T, N, typeof(μ), typeof(Σ), typeof(C)}(μ, Σ, C)

MvNormal(T, N; μ = nothing, Σ = nothing, C = nothing) = MvNormal(T, N, μ, Σ, C)

MvNormal(N; kwargs...) = MvNormal(Float64, N; kwargs...)

Base.length(::MvNormal{T, N}) where {T, N} = N

Base.eltype(::MvNormal{T, N}) where {T, N} = T

Σ(d::MvNormal{T, N}) where {T, N} = begin
    if !isnothing(d.Σ) return d.Σ end
    if !isnothing(d.C) return AᵀA(d.C) end
    return T(1)I(N)
end

Λ(d::MvNormal{T, N}) where {T, N} = begin
    if !isnothing(d.C) return inv(Cholesky(d.C, :U, 0)) end
    if !isnothing(d.Σ) return inv(cholesky(d.Σ)) end
    return T(1)I(N)
end

LinearAlgebra.cholesky(d::MvNormal{T, N}) where {T, N} = begin
    if !isnothing(d.C) return Cholesky(d.C, :U, 0) end
    if !isnothing(d.Σ) return cholesky(d.Σ) end
    return Cholesky(T(1.0)I(N), :U, 0)
end

# Operators

Base.:+(d₁::MvNormal{T, N}, d₂::MvNormal{T, N}) where {T, N} = begin
    return MvNormal(T, N, d₁.μ + d₂.μ, Σ(d₁) .+ Σ(d₂), nothing)
end

Base.:+(v::AbstractVector{T}, d::MvNormal{T, N}) where {T, N} = 
    return MvNormal(T, N, v .+ d.μ, d.Σ, d.C)

Base.:+(d::MvNormal, v::AbstractVector) = v + d

Base.:*(m::AbstractMatrix{T}, d::MvNormal{T, N}) where {T, N} = begin
    Σ = if !isnothing(d.C) AᵀA(d.C * m')
    else m * d.Σ * m' end
    return MvNormal(T, N, m * d.μ, Σ, nothing)
end

Base.:*(d::MvNormal, m::AbstractMatrix) = m' * d

Base.:(≈)(d₁::MvNormal, d₂::MvNormal) = begin
    if length(d₁) != length(d₂) return false end
    return d₁.μ ≈ d₂.μ && d₁.Σ ≈ d₂.Σ && d₁.C ≈ d₂.C
end

Base.:&(d₁::MvNormal{T, N}, d₂::MvNormal{T, N}) where {T, N} = begin
    c = cholesky!(Σ(d₁) + Σ(d₂))
    #Σ₁, Σ₂ = c.L \ Σ(d₁), c.L \ Σ(d₂)
    #μ₁, μ₂ = c.L \ d₁.μ, c.L \ d₂.μ
    L⁻¹ = LinearAlgebra.inv!(c.L)
    Σ₁, Σ₂ = L⁻¹ * Σ(d₁), L⁻¹ * Σ(d₂)
    μ₁, μ₂ = L⁻¹ * d₁.μ, L⁻¹ * d₂.μ
    return MvNormal(T, N, Σ₂'μ₁ + Σ₁'μ₂, Σ₁'Σ₂, nothing)
end

function product(d₁::MvNormal{T,N}, ds...) where {T,N}
    if isempty(ds) return d₁ end
    Λₛ = Λ(d₁)
    m = Λₛ * d₁.μ
    for d in ds
        Λᵢ = Λ(d)
        m += Λᵢ * d.μ
        Λₛ += Λᵢ
    end
    Σ = Λₛ |> Hermitian |> cholesky! |> inv
    return MvNormal(T, N, Σ * m, Σ, nothing)
end

# Sampling

Base.rand(d::MvNormal) = d.μ + cholesky(d).L * randn(eltype(d), length(d)) # This is too ineffective

function Base.rand(d::MvNormal, n::Int64)
    d = MvNormal(eltype(d), length(d), d.μ, nothing, cholesky(d).U)
    return [rand(d) for _ in 1:n]
end

# Logpdf

function Distributions.logpdf(d::MvNormal, x)
    z, s = x - d.μ, log(2π)*length(d)
    if isnothing(d.C) && isnothing(d.Σ) return -0.5*(s + z'z) end
    c = cholesky(d)
    return -0.5*(s + logdet(c) + AᵀA(c.U' \ z))
end

function ChainRulesCore.rrule(::typeof(logpdf), d::MvNormal, x)
    z, s = x .- d.μ, log(2π)*length(d)
    isovar_pb(Δy) = begin
        dx = -Δy .* z
        dμ = !isnothing(d.μ) ? -dx : NoTangent()
        (NoTangent(), Tangent{MvNormal}(μ = dμ, Σ = NoTangent(), C = NoTangent()), dx) 
    end 
    if isnothing(d.C) && isnothing(d.Σ) return -0.5*(s + z'z), isovar_pb end
    c = cholesky(d)
    Σ⁻¹z, L⁻¹z = c\z, c.U'\z
    mvn_pb(Δy) = begin
        dx = -Δy .* Σ⁻¹z
        dμ = !isnothing(d.μ) ? -dx : NoTangent()
        dC = if !isnothing(d.C) @thunk ProjectTo(d.C)(Δy .* (L⁻¹z * Σ⁻¹z' .- inv(Diagonal(d.C)))) else NoTangent() end
        dΣ = if !isnothing(d.Σ) @thunk ProjectTo(d.Σ)((x->2x.-Diagonal(x))(-0.5(inv(c) .- AAᵀ(Σ⁻¹z)))) else NoTangent() end
        (NoTangent(), Tangent{MvNormal}(μ = dμ, Σ = dΣ, C = dC), dx)
    end
    return -0.5*(s + logdet(c) + AᵀA(L⁻¹z)), mvn_pb
end

function logpdftest(d::MvNormal{T,N}, x, h = 0.00001) where {T,N}
    ll, llΔx = logpdf(d, x), zeros(length(d))
    llΔμ = if !isnothing(d.μ) zeros(length(d)) else NoTangent() end
    llΔΣ = if !isnothing(d.Σ) zeros(length(d), length(d)) else NoTangent() end
    llΔC = if !isnothing(d.C) UpperTriangular(zeros(length(d), length(d))) else NoTangent() end 
    for i in 1:length(d)
        if !isnothing(d.μ)
            Δμ = copy(d.μ)
            Δμ[i] += h
            llΔμ[i] = (logpdf(MvNormal(T, N, Δμ, d.Σ, d.C), x) - ll)/h
        end
        for j in 1:length(d)
            if !isnothing(d.Σ)
                ΔΣ = copy(d.Σ)
                ΔΣ[i,j] += h
                if (i != j) ΔΣ[j,i] += h end
                llΔΣ[i,j] = (logpdf(MvNormal(T, N, d.μ, ΔΣ, d.C), x) - ll)/h
                if (i != j) llΔΣ[j,i] = llΔΣ[i,j] end
            end
            if !isnothing(d.C) && i <= j
                ΔC = copy(d.C)
                ΔC[i,j] += h
                llΔC[i,j] = (logpdf(MvNormal(T, N, d.μ, d.Σ, ΔC), x) - ll)/h
            end
        end
        Δx = copy(x)
        Δx[i] += h
        llΔx[i] = (logpdf(d, Δx) - ll)/h
    end
    return (Tangent{MvNormal}(μ = llΔμ, Σ = llΔΣ, C = llΔC), llΔx)
end

# Logpdfnan

function logpdfnan(d::MvNormal{T,N,M,S,U}, x, nums = .!isnan.(x)) where {T,N,M,S,U}
    if !any(nums) return T(0) end
    if all(nums) return logpdf(d, x) end
    μ = !isnothing(d.μ) ? d.μ[nums] : nothing
    C = if !isnothing(d.C) U(U <: Diagonal ? d.C.diag[nums] : d.C[nums,nums]) else nothing end
    Σ = if !isnothing(d.Σ) S <: Diagonal ? S(d.Σ.diag[nums]) : d.Σ[nums,nums] else nothing end
    return logpdf(MvNormal(T, sum(nums), μ, Σ, C), x[nums])
end

function ChainRulesCore.rrule(::typeof(logpdfnan), d::MvNormal{T,N,M,S,U}, x, nums = .!isnan.(x)) where {T,N,M,S,U}
    if !any(nums) return T(0), Δy -> NoTangent(), ZeroTangent(), ZeroTangent() end
    if all(nums) return rrule(logpdf, d, x) end
    μ = !isnothing(d.μ) ? d.μ[nums] : nothing
    C = if !isnothing(d.C) U(U <: Diagonal ? d.C.diag[nums] : d.C[nums,nums]) else nothing end
    Σ = if !isnothing(d.Σ) S <: Diagonal ? S(d.Σ.diag[nums]) : d.Σ[nums,nums] else nothing end
    y, y_pb = rrule(logpdf, MvNormal(T, sum(nums), μ, Σ, C), x[nums])
    logpdfnan_pb(Δy) = begin
        _, Δd, Δx = y_pb(Δy)
        μ₁ = isnothing(d.μ) ? NoTangent() : @thunk begin μ₀ = zeros(length(d)); view(μ₀,nums) .= Δd.μ; μ₀ end
        U₁ = if U <: Nothing NoTangent()
        elseif U <: Diagonal @thunk begin U₀ = zeros(length(d)); view(U₀,nums) .= Δd.C.diag; U(U₀) end
        else @thunk begin U₀ = zeros(size(d.C)); view(U₀,nums,nums) .= Δd.C; U(U₀) end end
        Σ₁ = if S <: Nothing NoTangent()
        elseif S <: Diagonal @thunk begin Σ₀ = zeros(length(d)); view(Σ₀,nums) .= Δd.Σ.diag; S(Σ₀) end
        else @thunk begin Σ₀ = zeros(size(d.Σ)); view(Σ₀,nums,nums) .= Δd.Σ; end end
        x₁ = @thunk begin x₀ = zeros(length(d)); view(x₀,nums) .= Δx; x₀ end
        (NoTangent(), Tangent{MvNormal}(μ = μ₁, Σ = Σ₁, U = U₁), x₁)
    end
    return y, logpdfnan_pb
end