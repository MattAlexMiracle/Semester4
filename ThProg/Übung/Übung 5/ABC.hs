data Term = A Term | B Term | C Term | X
    deriving (Show, Eq)





tes (A (B x)) = B x
tes (C (A x)) = A (C x)
tes (C (C x)) = B x
tes (B (A x)) = B x
tes (C (B x)) = B (C (x))
tes (A x) = A (tes x)
tes (B x) = B (tes x)
tes (C x) = C (tes x)
tes x = x

runtes f x = if (tes x) == x then x
    else f (tes x)
