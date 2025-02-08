module MvNormals

import StaticArrays: SVector
import ChainRulesCore: ChainRulesCore, ZeroTangent, NoTangent, Tangent, ProjectTo, rrule, @thunk
import LinearAlgebra: Cholesky, LowerTriangular, Diagonal, cholesky, logdet, inv, diag

include("utilities.jl")

include("abstractmvn.jl")
export μ, σ, Σ, logpdf

include("mvn.jl")
export MvNormal, logpdftest

include("diagmvn.jl")
export DiagMvNormal

include("isomvn.jl")
export IsoMvNormal

end # MvNormals