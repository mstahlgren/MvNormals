import StatsBase

export Mixture, logpdf, sample

# Should be UniformMixture
struct Mixture{D} <: Distribution
    dists::Vector{D}
end

Base.:+(m::Mixture{D}, d::D) where {D <: MvNormal} = Mixture([dᵢ + d for dᵢ in m.dists])

sample(m::Mixture, n::Int) = [sample(d, 1) for d in rand(m.dists, n)]

logpdf(m::Mixture, o) = sum([logpdf(d, o) for d in m.dists]) / length(m.dists)
