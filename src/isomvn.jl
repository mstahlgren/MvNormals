struct IsoMvNormal{N, T} <: AbstractMvNormal{T} end

IsoMvNormal(N, T = Float64) = IsoMvNormal{N, T}()

μ(::IsoMvNormal{N, T}) where {N, T} = zeros(SVector{N, T})

σ(::IsoMvNormal{N, T}) where {N, T} = Diagonal(ones(SVector{N, T}))

Σ(d::IsoMvNormal) = d |> σ

Base.length(::IsoMvNormal{N, T}) where {N, T} = N

Base.rand(::IsoMvNormal{N, T}) where {N, T} = rand(SVector{N})

logpdf(d::IsoMvNormal, x) = -0.5*(log(2π)*length(d) + x'x)

function ChainRulesCore.rrule(::typeof(logpdf), d::IsoMvNormal, x)
    mvn_pb(Δy) = (NoTangent(), NoTangent(), -Δy .* x)
    return -0.5*(log(2π)*length(d) + x'x), mvn_pb
end