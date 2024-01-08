struct Zero end

Base.:+(::Zero, x) = x

Base.:+(x, ::Zero) = x

Base.:*(::Zero, x) = Zero()

Base.:*(x, ::Zero) = Zero()

Base.:\(x, ::Zero) = Zero()