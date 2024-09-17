AᵀA(A) = A' * A

AAᵀ(A) = A * A'

Base.:+(::Nothing, ::Nothing) = nothing

Base.:+(::Nothing, x) = x

Base.:+(x, ::Nothing) = x

Base.:-(::Nothing, x) = -x

Base.:-(x, ::Nothing) = x

Base.:*(::Nothing, x) = nothing

Base.:*(x, ::Nothing) = nothing

Base.:\(x, ::Nothing) = nothing

Base.:≈(::Nothing, ::Nothing) = true

Base.:≈(::Nothing, x) = false

Base.:≈(x, ::Nothing) = false