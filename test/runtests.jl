using MvNormals
import Distributions: MultivariateNormal
using LinearAlgebra
using Test

n = 2

mvn₀ = MvNormal(n)
mvn₁ = MvNormal(ones(n), Diagonal(ones(n)))
mvn₂ = MvNormal(ones(n), Matrix(2.0I(n)))
mvn₃ = MvNormal(ones(n), (x->x'x)(randn(n,n)))

dvn₀ = MultivariateNormal(zeros(n), Matrix(1.0I(n)))
dvn₁ = MultivariateNormal(ones(n), Diagonal(ones(n)))
dvn₂ = MultivariateNormal(ones(n), Matrix(2.0I(n)))
dvn₃ = MultivariateNormal(ones(n), (x->x'x)(mvn₃.U))

nanvn₁ = MvNormal(ones(3), Diagonal(ones(3)))
nanvn₂ = MvNormal(ones(3), Matrix(2.0I(3)))

@testset "operators" begin
    @test mvn₀ + mvn₁ == mvn₂
    @test mvn₂ & mvn₂ ≈ mvn₁
    @test product(mvn₁, mvn₁, mvn₁) isa MvNormal
end

@testset "logpdf" begin
    @test logpdf(mvn₀, ones(n)) ≈ logpdf(dvn₀, ones(n))
    @test logpdf(mvn₁, ones(n)) ≈ logpdf(dvn₁, ones(n))
    @test logpdf(mvn₂, ones(n)) ≈ logpdf(dvn₂, ones(n))
    @test logpdf(mvn₃, ones(n)) ≈ logpdf(dvn₃, ones(n))
    @test logpdfnan(nanvn₁, [1.0, 1.0, NaN]) ≈ logpdf(mvn₁, ones(2))
    @test logpdfnan(nanvn₂, [1.0, 1.0, NaN]) ≈ logpdf(mvn₂, ones(2))
end

@testset "rrules" begin
    
end