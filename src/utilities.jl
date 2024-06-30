AᵀA(A) = A' * A

AAᵀ(A) = A * A'


struct Zero end

Base.:+(::Zero, x::AbstractArray) = x

Base.:+(x::AbstractArray, ::Zero) = x

Base.:*(::Zero, ::AbstractArray) = Zero()

Base.:*(::AbstractArray, ::Zero) = Zero()

Base.:\(::AbstractArray, ::Zero) = Zero()


struct Identity end

Base.:+(x::Identity, y) = Matrix(1.0I, size(y)) .+ y

Base.:+(x, y::Identity) = y + x

Base.:*(x::Identity, y) = y

Base.:*(x, y::Identity) = x

Base.:*(x::Identity, y::Identity) = x

Base.adjoint(x::Identity) = x