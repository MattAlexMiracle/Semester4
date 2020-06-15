import Control.Monad.Fix


data Logic = Not Logic| And Logic Logic | 
 Or Logic Logic| Implies Logic Logic| 
 Forall Logic Logic| Exists Logic Logic| Var Int
 deriving (Show, Eq, Ord)

-- The number in Var is only for easier debuging, there's no rule matching against it
-- and variables never occur more than once in our example, so no variables can be accidentally merged
-- (because there are no merges)
to_zero::Logic -> Logic
to_zero (Not (Not x)) = x
to_zero (And x y )= Not (Or (Not x)  (Not y))
to_zero (Not (And x y)) = (Or (Not x)  (Not y))
to_zero (Implies x y) = Or (Not x) y
to_zero (Or x (Or y z)) = Or (Or x y) z
to_zero (Forall x y) = Not (Exists x (Not y))
to_zero x = x

firstArg :: Logic -> Logic
firstArg (And x _) =  x
firstArg (Or x _) =  x
firstArg (Implies x _) = x
firstArg (Forall x _) =  x
firstArg (Not x) =  x
firstArg (Var _) = Var (-1)
firstArg (Exists  x _) =x

secondArg::Logic -> Logic
secondArg (And _ x) =  x
secondArg (Or _ x) = x
secondArg (Implies _ x) = x
secondArg (Forall _ x)= x
secondArg (Exists  _ x) =x
-- technically those don't matter as they are invariant to
-- reduction and ignored when using connector.
-- they also save me the hassle of wrapping/unwrapping Maybes
secondArg (Not _) = Var (-1)
secondArg (Var _ )= Var (-1)


connector::Logic->Logic->Logic->Logic
connector (Not _)  x _ = Not x
connector (And _ _) x y = And x y 
connector (Or _ _) x y = Or x y 
connector (Forall _ _) x y = Forall x y 
connector (Exists _ _) x y = Exists x y 
connector (Implies _ _) x y = Implies x y 
connector (Var z ) x y = Var z

-- Kontextabgeschlossen (to_zero ist bereits stabil)
to::Logic -> Logic
-- to (Not x) = let x' = to_zero (Not x) in if x' == x then to_zero x else x'
to x
	| not (x==x1) = x1 
	| not (x==x2) = x2
	| not (x==x3) = x3
	| otherwise = x
 where
 	con = connector x
 	x1 = to_zero x
 	x2 = to_zero$con (to$firstArg x) (secondArg x)
 	x3 = to_zero$con (firstArg x) (to$secondArg x) 


to_star::Logic -> Logic
to_star x = let x' = to x in if x'==x then x else to_star x'

build_word::Int->Logic
build_word x = case x of
	0->(Not $Var 1)
	1->(And (Var 1) (Var 1) )
	2->(Or (Var 1) (Var 1) )
	3->(Implies (Var 1) (Var 1))
	4->(Forall (Var 1) (Var 1))

check_sentences::Int->Int->Int -> Bool
check_sentences first second third = case first of
	0->to_star (Not first_arg) == to_star (Not normalised_first)
	1->to_star (And first_arg second_arg ) ==to_star (And normalised_first second_arg )
	 && to_star (And first_arg second_arg ) ==to_star (And first_arg normalised_second )
	2->to_star (Or first_arg second_arg )== to_star (Or normalised_first second_arg )
	  && to_star (Or first_arg second_arg )== to_star (Or first_arg normalised_second )
	3 ->to_star (Implies first_arg second_arg) == to_star (Implies normalised_first second_arg)
	  && to_star (Implies first_arg second_arg) == to_star (Implies first_arg normalised_second)
	4->to_star (Forall first_arg second_arg) ==  to_star (Forall normalised_first second_arg)
	 && to_star (Forall first_arg second_arg) ==  to_star (Forall first_arg normalised_second)

	where
		first_arg = build_word second
		second_arg = build_word third
		normalised_first = to_star first_arg
		normalised_second = to_star second_arg

build_sentences::Int->Int->Int -> Logic
build_sentences first second third = case first of
	0-> (Not first_arg) 
	1-> (And first_arg second_arg )
	2-> (Or first_arg second_arg )
	3-> (Implies first_arg second_arg)
	4-> (Forall first_arg second_arg)
	where
		first_arg = build_word second
		second_arg = build_word third