using MvNormals
using LinearAlgebra
using Test

n = 2

mvn₀ = MvNormal(n)
mvn₁ = MvNormal(ones(n), Diagonal(ones(n)))
mvn₂ = MvNormal(ones(n), Matrix(2.0I(n)))

@testset "MvNormal" begin
    @test mvn₀ + mvn₁ == mvn₂
    @test ones(n) + Matrix(1.0I(n)) * mvn₀ == mvn₁
    @test mvn₂ & mvn₂ == mvn₁
    @test μ(mvn₁ & mvn₀) == Zero()
end