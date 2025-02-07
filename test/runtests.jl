import Distributions: Distributions, MultivariateNormal
import ChainRulesCore: unthunk

using MvNormals
using LinearAlgebra
using Test

N = 2

x  = randn(N)
μ  = randn(N)
σ₁ = exp.(randn(N))
σ₂ = (x->x*x')(randn(N, N))

iso = IsoMvNormal(N)
dia = DiagMvNormal(μ, σ₁)
cov₁ = MvNormal(μ, cholesky(σ₂).L)
cov₂ = MultivariateNormal(μ, σ₂)

@testset "product" begin
    @test diag(σ(iso & iso)) == 1.0 ./ sqrt.(fill(2.0, N))
end

@testset "logpdf" begin
    @test logpdf(cov₁, x) ≈ Distributions.logpdf(cov₂, x)
end

@testset "rrules" begin
    dσ = MvNormals.rrule(logpdf, cov₁, [1.3, 0.5])[2](1.0)
    Δσ = MvNormals.logpdftest(cov₁, [1.3, 0.5], 1e-8)
    @test isapprox(Δσ[2], unthunk(dσ[3]), atol = 1e-3)
    @test isapprox(Δσ[1][:μ], unthunk(dσ[2][:μ]), rtol = 1e-3)
    @test isapprox(Δσ[1][:σ], unthunk(dσ[2][:σ]), rtol = 1e-3)
end
