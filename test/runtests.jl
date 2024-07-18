import Distributions: MultivariateNormal
import Zygote: pullback, gradient

using MvNormals
using LinearAlgebra
using Test

n = 2
v = randn(n)
r = randn(n,n) |> MvNormals.AAᵀ

i₀ = IsoMvNormal(n)
d₀ = MvNormal(zeros(n), Diagonal(ones(n)))
f₀ = MvNormal(zeros(n), 1.0Matrix(I, n, n))
x₀ = MultivariateNormal(zeros(n), Diagonal(ones(n)))

d₁ = MvNormal(ones(n), Diagonal(sqrt(2)*ones(n)))
f₁ = MvNormal(ones(n), UpperTriangular(sqrt(8)*Matrix(I, n, n)))

f₂ = MvNormal(v, r)
x₂ = MultivariateNormal(v, r)

f₃ = MvNormal(randn(3), randn(3,3) |> MvNormals.AAᵀ)

@testset "rand" begin
    @test IsoMvNormal(Float32, n) |> rand |> eltype <: Float32
    @test rand(i₀, 5) |> length == 5
end

@testset "addition" begin
    @test i₀ + i₀ ≈ d₀ + f₀
    @test i₀ + d₀ + d₁ + d₁ + d₁ ≈ 2.0ones(n) + f₁
    @test (d₀ + d₁).U isa Diagonal
    @test zeros(n) + i₀ ≈ d₀
end

@testset "multiplication" begin
    @test zeros(n) + Diagonal(ones(n)) * i₀ ≈ d₀
    @test 2.0ones(n) + 2.0Matrix(I, n, n) * i₀ ≈ d₁ + d₁
end

@testset "product" begin
    @test i₀ & d₀ ≈ product(i₀, f₀)
end

@testset "logpdf" begin
    @test logpdf(i₀, v) ≈ logpdf(d₀, v)
    @test logpdf(f₀, v) ≈ logpdf(x₀, v)
    @test logpdf(f₂, v) ≈ logpdf(x₂, v)
    @test logpdf(f₂, v) ≈ pullback(logpdf, f₂, v)[1]
    @test logpdfnan(IsoMvNormal(3), [3.1, 4.1, NaN]) ≈ logpdf(d₀, [3.1, 4.1])
    @test logpdfnan(f₃, [3.1, 4.1, NaN]) ≈ pullback(logpdfnan, f₃, [3.1, 4.1, NaN])[1]
end

@testset "rrules" begin
    df, dx = pullback(logpdf, f₂, [1.3, 0.5])[2](1.0), pullback(logpdf, x₂, [1.3, 0.5])[2](1.0)
    @test df[2] ≈ dx[2]
    @test df[1][:μ] ≈ dx[1][:μ]
    @test df[1][:U] ≈ dx[1][:Σ].chol.factors
    # @test typeof(df[1][:U]) == typeof(dx[1][:Σ].chol.factors) Fails as chol rrule for chol seems bugged
end
