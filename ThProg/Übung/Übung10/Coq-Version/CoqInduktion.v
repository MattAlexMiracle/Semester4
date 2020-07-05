Section List.

Variable A:Type.
Variable x:A.


Inductive  List:=
 | Nil : List
 | Cons : A->List->List.
Print List.

Fixpoint length (x:List) := 
 match x with
 | Nil => O
 | Cons x xs => S (length xs)
end.

Eval compute in (length (Cons x (Cons x Nil))).

Fixpoint snoc (x:List) (y:A) := 
 match x with
 | Nil => Cons y Nil
 | Cons x xs => Cons x (snoc xs y)
end.
Print snoc.

Fixpoint reverse (x:List) := 
 match x with
 | Nil => Nil
 | Cons x xs => snoc (reverse xs) x
end.

Fixpoint concat (xs:List) (ys:List) :=
 match xs with
 | Nil=>ys
 | Cons x xs => Cons x (concat xs ys)
end.
Infix "++" := concat (at level 60, right associativity). 


Fixpoint reverse' (xs ys:List):=
 match xs with
 | Nil=>ys
 | Cons x xs=> reverse' xs (Cons x ys)
end.


Theorem CCNil: forall (xs:List),xs ++ Nil =xs.
intro B; elim B.
simpl.
reflexivity.
intros.
simpl.
rewrite -> H.
reflexivity.
Qed.


Theorem SnocPlus: forall (x:A), forall (xs ys:List), snoc (xs++ys) x = xs ++ (snoc ys x).
intros.
elim xs.
simpl.
reflexivity.
simpl.
intros.
rewrite ->H.
reflexivity.
Qed.

Theorem RevSnoc: forall (x: A),forall (xs:List), reverse (snoc xs x) = Cons x (reverse xs).
induction xs. (* hab ich grad erst herausgefunden, dass es die Tactic auch gibt*)
simpl.
reflexivity.
simpl.
rewrite -> IHxs.
reflexivity.
Qed.

Theorem DoubleInverse: forall (xs:List), reverse (reverse xs)=xs.
induction xs.
simpl.
reflexivity.
simpl.
rewrite RevSnoc.
rewrite -> IHxs.
reflexivity.
Qed.

Theorem SingleAddFromBack: forall (x:A), forall (xs:List), xs ++ (Cons x Nil)=snoc xs x.
induction xs.
simpl.
reflexivity.
simpl.
rewrite <- IHxs.
reflexivity.
Qed.

Theorem RevConcat: forall (xs ys:List), reverse (xs++ys) = (reverse ys) ++ (reverse xs).
induction xs.
intros.
simpl.
rewrite CCNil.
reflexivity.
intro.
simpl.
rewrite IHxs.
rewrite SnocPlus.
reflexivity.
Qed.

Lemma CCAssoc: forall (xs ys zs:List), (xs ++ ys)++zs = xs ++(ys++zs).
induction xs.
simpl.
reflexivity.
simpl.
intros.
rewrite IHxs.
reflexivity.
Qed.

Lemma Rev'Eq: forall (xs ys : List), reverse' xs ys = reverse' xs Nil ++ ys.
induction xs.
intro.
simpl.
reflexivity.
simpl.
rewrite IHxs.
intro.
rewrite CCAssoc.
rewrite IHxs.
simpl.
reflexivity.
Qed.

Theorem RevEqRev': forall (xs :List), reverse xs = reverse' xs Nil.
induction xs.
simpl.
reflexivity.
simpl.
rewrite -> Rev'Eq.
rewrite SingleAddFromBack.
rewrite IHxs.
reflexivity.
Qed.

End List.

Definition chain {A B C:Type} (f: B->C) (g:A->B) (n:A) :C := f (g n).

(* das ist jetzt hier in Coq nicht optimal, weil . schon verwendet wird, deswegen dieser Hack*)

Infix "(.)" := chain (at level 60, right associativity).

Eval compute in (length nat (.) reverse nat) (Cons nat 1 (Nil nat)).


Fixpoint map {A B:Type} (f: A->B) (xs:List A): (List B):=
 match xs with
 | Nil _ =>Nil _
 | Cons _ x xs => Cons _ (f x) (map f xs)
end.


Theorem MapChain: forall (A B C:Type), forall (f: B->C), forall (g: A->B), forall (xs:List A), ((map f) (.) (map g)) xs = map (f(.)g) xs.
induction xs.
simpl.
unfold chain.
simpl.
reflexivity.
unfold chain.
simpl.
unfold chain in IHxs.
rewrite IHxs.
reflexivity.
Qed.


Theorem MapCC: forall (A B:Type), forall (f:A->B), forall (xs ys: List A), map f ( concat A xs ys) = concat B (map f xs) (map f ys).
induction xs.
intros.
simpl.
reflexivity.
simpl.
intro.
rewrite IHxs.
reflexivity.
Qed.

Section Bäume.

Variable A:Type.

Inductive BinTree :=
 | Leaf:BinTree
 | Bin: BinTree -> A-> BinTree-> BinTree
.

Fixpoint mirror (tree:BinTree) :=
 match tree with
 |Leaf=> Leaf
 |Bin l x r => Bin (mirror r) x (mirror l)
end.

Fixpoint inorder (tree: BinTree):List A :=
 match tree with
 | Leaf=>Nil A
 | Bin l x r=> concat A (inorder l) (Cons A x (inorder r))
end.

Theorem DoubleMirror: forall (t: BinTree), mirror (mirror t) = t.
induction t.
simpl.
reflexivity.
simpl.
rewrite IHt1.
rewrite IHt2.
reflexivity.
Qed.

Lemma ShiftSnoc: forall (B:Type), forall (xs:List B), forall (ys:List B), forall (x:B), concat B (snoc B xs x) ys = concat B xs (Cons B x ys).
induction xs.
intros.
simpl.
reflexivity.
intros.
simpl.
rewrite IHxs.
reflexivity.
Qed.

Theorem MirroredFlatten: forall (t: BinTree), inorder (mirror t) = reverse A (inorder t).
induction t.
simpl.
reflexivity.
simpl.
rewrite IHt1.
rewrite IHt2.
rewrite RevConcat.
simpl.
rewrite ShiftSnoc.
reflexivity.
Qed.







