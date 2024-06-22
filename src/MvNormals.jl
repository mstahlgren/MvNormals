module MvNormals

include("zero.jl")
include("mvnormal.jl")
include("mvnormal2.jl")

import LinearAlgebra: I, LowerTriangular

mu3 = randn(3)
mu10 = randn(10)
mu100 = randn(100)

sd3f = (x->x'x)(randn(3,3))
sd3d = Diagonal(exp.(randn(3)))
sd10f = (x->x'x)(randn(10,10))
sd10d = Diagonal(exp.(randn(10)))
sd100f = (x->x'x)(randn(100,100))
sd100d = Diagonal(exp.(randn(100)))

const d1_full3 = MvNormal(mu3, sd3f)
const d1_diag3 = MvNormal(mu3, sd3d)
const d1_full10 = MvNormal(mu10, sd10f)
const d1_diag10 = MvNormal(mu10, sd10d)
const d1_full100 = MvNormal(mu100, sd100f)
const d1_diag100 = MvNormal(mu100, sd100d)

const d2_full3 = MvNormal2(mu3, sd3f)
const d2_diag3 = MvNormal2(mu3, sd3d)
const d2_full10 = MvNormal2(mu10, sd10f)
const d2_diag10 = MvNormal2(mu10, sd10d)
const d2_full100 = MvNormal2(mu100, sd100f)
const d2_diag100 = MvNormal2(mu100, sd100d)

export d1_full3, d1_diag3, d1_full10, d1_diag10, d1_full100, d1_diag100
export d2_full3, d2_diag3, d2_full10, d2_diag10, d2_full100, d2_diag100

end # MvNormals