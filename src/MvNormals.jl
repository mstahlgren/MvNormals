module MvNormals

import StaticArrays: SVector
import ChainRulesCore: ChainRulesCore, ZeroTangent, NoTangent, Tangent, ProjectTo, rrule, @thunk
import LinearAlgebra: BLAS, Cholesky, LowerTriangular, Diagonal, cholesky, logdet, inv

include("utilities.jl")

include("abstractmvn.jl")
export μ, σ, Σ, logpdf, δlogpdfδσ

include("mvn.jl")
export MvNormal, test

include("isomvn.jl")
export IsoMvNormal

end # MvNormals