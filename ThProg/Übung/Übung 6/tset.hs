{-n0::(Int->Int)->Int->Int
n0 = \f a -> a
n1= \f a -> f( n0 f a)
n2= \f a -> f(f( n0 f a))
n3= \f a -> f(f(f( n0 f a)))
n4= \f a -> f(f(f(f( n0 f a))))
n5= \f a -> f(f(f(f(f( n0 f a)))))
n6= \f a -> f(f(f(f(f(f( n0 f a))))))
n7= \f a -> f(f(f(f(f(f(f( n0 f a)))))))
n8= \f a -> f(f(f(f(f(f(f(f( n0 f a))))))))
n9= \f a -> f(f(f(f(f(f(f(f(f( n0 f a)))))))))
true = \x y -> x
false = \x y -> y
zero = \f a -> a
one = \f a -> f a
two = \f a -> f(f a)
succ n  =\f a->  f (n (f a))
pred n = \f a -> n (\g h -> h (g f)) (\u -> a) (\u -> u)

-}
import Data.List
data Term = 
	TVar String
	| TAbstr String Term
	| TApp Term Term
	deriving (Eq, Ord, Show)
--substitution
subst::String->Term->Term->Term
subst x (TVar v) newVal
	| x==v = newVal
	| otherwise = TVar v
subst x (TAbstr y t1) newVal
	| x== y = TAbstr y t1
	--test sonst würde ich eine variable reinsubst die gecaptured wird
	| x/=y && (notElem y (freeVars newVal)) = TAbstr y ( subst x t1 newVal) 
	| otherwise = error "cannot substitute"
subst x (TApp t1 t2) newVal = TApp (subst x t1 newVal) (subst x t2 newVal)
-- Freie Variablen rekursiv
freeVars::Term->[String]
freeVars (TVar x) = [x]
freeVars (TAbstr x t) =freeVars t \\ [x] --explizit ausschließen von geb. var
freeVars (TApp t1 t2) = freeVars t1 ++ freeVars t2

--Freie variablen in pseudo-De-Brujin form
eval::Term->Term
eval (TApp (TAbstr x t12) v2@(TAbstr _ _)) = subst x t12 v2 --beta-elim mit Term
eval (TApp v1@(TAbstr x _) t2) = let t2' = eval t2 in  TApp t2' v1 -- beta-elim
eval (TApp t1 t2) = let t1' =  eval t1 in TApp t1' t2
eval x= error "Keine Regel"
-- irgendwie de-brujin indizes draufstampfen
data TermBrujin = 
	Const
	|TVarB Int
	|LamB TermBrujin
	|TAppB TermBrujin TermBrujin
	deriving (Eq, Ord,Read,Show)
evalB::TermBrujin->TermBrujin
evalB (TAppB tm applikant) = case evalB tm of
	LamB rest -> evalB (substB 0 applikant rest)
evalB v = v

substB::Int->TermBrujin->TermBrujin->TermBrujin
substB n tm Const = Const
substB n tm (TVarB m) = case compare m n of
	LT-> TVarB m
	EQ->tm
	GT->TVarB (m-1)
substB  n tm (LamB rest) = LamB (substB (n+1) tm rest)
substB n tm (TAppB tm' applikant) = TAppB (substB n tm tm') (substB n tm applikant)

