module ComposableDistributions

abstract type Distribution end

include("mvnormal.jl")

include("mixture.jl")

end
