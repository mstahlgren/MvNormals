import StatsBase

export Mixture, logpdf, sample

# Should be UniformMixture
struct Mixture{D} <: Distribution
    dists::Vector{D}
end

Base.:+(m::Mixture{D₁}, d::D₂) where {D₁ <: MvNormal, D₂ <: MvNormal} = Mixture([dᵢ + d for dᵢ in m.dists])

Base.:*(mat::Matrix, m::Mixture{D}) where {D <: MvNormal} = Mixture([mat * d for d in m.dists])

Base.:(==)(x::Mixture, y::Mixture) = [i[1] == i[2] for i in zip(x.dists, y.dists)] |> all

StatsBase.:sample(m::Mixture, n::Int) = [sample(d, 1) for d in rand(m.dists, n)]

logpdf(m::Mixture, o) = sum([logpdf(d, o) for d in m.dists]) / length(m.dists)
