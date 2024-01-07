struct Zero end

Base.:+(::Zero, x₂) = x₂

Base.:+(x₁, ::Zero) = x₁

Base.:*(::Zero, x₂) = Zero()

Base.:*(x₁, ::Zero) = Zero()