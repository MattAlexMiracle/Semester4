{-# LANGUAGE GADTs #-}
import Prelude hiding (length, drop, reverse, elem, maximum)

data List a where
	Nil::List a
	Cons::a->List a->List a
	deriving (Show, Eq)
length::List a-> Int
length Nil = 0
length (Cons x y) = 1+  length y
snoc::List a-> a->List a
snoc Nil x = Cons x Nil
snoc (Cons x y) z = Cons x (snoc y z)
reverse::List a-> List a
reverse (Cons x y)= snoc ( reverse y) x
reverse Nil = Nil
drop::Eq a=>a->List a-> List a
drop x Nil = Nil
drop x (Cons z y) =  if x==z then drop x y else Cons z $drop x y
elem::Eq a=>a->List a->Bool
elem x Nil =False
elem x (Cons z y) = if x==z then True else  elem x y
maximum::List Int->Int
maximum Nil = 0
maximum (Cons x y) = if maximum y> x then maximum y else x


main:: IO ()
main = do
	let x = Cons 2$Cons 1$ Cons 2$ Cons 3 $ Nil
	print$ Main.length x
	print $reverse x
	print $drop 2 x
	print $ elem 4 x
	print $elem 2 $ drop 2 x
	print $ maximum x

