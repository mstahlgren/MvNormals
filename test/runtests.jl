using ComposableDistributions
using LinearAlgebra
using Test

n = 2
Z = n |> zeros
I₂ = Matrix(1.0I(n))

mvn = DenseMvNormal(Z, I₂)
imvn = IsoMvNormal(n)

@testset "MvNormal" begin
    @test Z + I₂*(imvn + imvn) == mvn + mvn
end