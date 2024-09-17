import Distributions: MultivariateNormal
import Zygote: pullback, gradient

using MvNormals
using LinearAlgebra
using Test

n = 2
v = randn(n)
s = randn(n,n) |> MvNormals.AᵀA
r = cholesky(s).U

i₀ = MvNormal(n)
c₀ = MvNormal(n; μ = v, C = r)
σ₀ = MvNormal(n; μ = v, Σ = s)
x₀ = MultivariateNormal(v, s)

@testset "rand" begin
    @test MvNormal(Float32, n) |> rand |> eltype <: Float32
    @test rand(i₀, 5) |> length == 5
end

@testset "addition" begin
    @test i₀ + c₀ + σ₀ isa MvNormal
end

@testset "multiplication" begin
    @test Diagonal(ones(n)) * c₀ ≈ σ₀
end

@testset "product" begin
    @test i₀ & c₀ ≈ product(i₀, σ₀)
end

@testset "logpdf" begin
    @test logpdf(c₀, v) ≈ logpdf(x₀, v)
    @test logpdf(c₀, v) ≈ logpdf(σ₀, v)
    @test logpdf(c₀, v) ≈ pullback(logpdf, c₀, v)[1]
    @test logpdfnan(MvNormal(3), [3.1, 4.1, NaN]) ≈ logpdf(i₀, [3.1, 4.1])
    @test logpdfnan(σ₀, [3.1, NaN]) ≈ pullback(logpdfnan, σ₀, [3.1, NaN])[1]
    @test logpdfnan(c₀, [3.1, NaN]) ≈ pullback(logpdfnan, c₀, [3.1, NaN])[1]
end

@testset "rrules" begin
    dσ = pullback(logpdf, σ₀, [1.3, 0.5])[2](1.0)
    Δσ = MvNormals.logpdftest(σ₀, [1.3, 0.5], 1e-8)
    dc = pullback(logpdf, c₀, [1.3, 0.5])[2](1.0)
    Δc = MvNormals.logpdftest(c₀, [1.3, 0.5], 1e-8)
    @test isapprox(Δc[2], dc[2], atol = 1e-3)
    @test isapprox(Δc[1][:μ], dc[1][:μ], rtol = 1e-3)
    @test isapprox(Δσ[1][:Σ], dσ[1][:Σ], atol = 1e-3)
    @test isapprox(Δc[1][:C], dc[1][:C], rtol = 1e-3)
end
