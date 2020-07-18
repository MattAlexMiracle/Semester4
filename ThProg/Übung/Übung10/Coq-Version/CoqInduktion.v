Section List.

Inductive List A:=
 | Nil : List A
 | Cons : A->List A->List A.
Print List.

Fixpoint length {A} (x:List A):nat := 
 match x with
 | Nil _ => O
 | Cons _ x xs => S (length xs)
end.

Eval compute in (length (Cons nat 0 (Cons nat 0 (Nil nat)))).

Fixpoint snoc {A} (x:List A) (y:A) := 
 match x with
 | Nil _ => Cons _ y (Nil _)
 | Cons _ x xs => Cons _ x (snoc xs y)
end.
Check snoc.

Fixpoint reverse {A} (x:List A) := 
 match x with
 | Nil _ => Nil _
 | Cons _ x xs => snoc (reverse xs) x
end.

Fixpoint concat {A} (xs:List A) (ys:List A) :=
 match xs with
 | Nil _=>ys
 | Cons _ x xs => Cons _ x (concat xs ys)
end.
Infix "++" := concat (at level 60, right associativity). 


Fixpoint reverse' {A} (xs ys:List A):=
 match xs with
 | Nil _=>ys
 | Cons _ x xs=> reverse' xs (Cons _ x ys)
end.


Theorem CCNil {A}: forall (xs:List A),xs ++ (Nil A) =xs.
intro B; elim B.
simpl.
reflexivity.
intros.
simpl.
rewrite -> H.
reflexivity.
Qed.


Theorem SnocPlus {A}: forall (x:A), forall (xs ys:List A), snoc (xs++ys) x = xs ++ (snoc ys x).
intros.
elim xs.
simpl.
reflexivity.
simpl.
intros.
rewrite ->H.
reflexivity.
Qed.


Theorem RevSnoc {A}: forall (x: A),forall (xs: (List A)), reverse (snoc xs x) = Cons A x (reverse xs).
induction xs. (* hab ich grad erst herausgefunden, dass es die Tactic auch gibt*)
simpl.
reflexivity.
simpl.
rewrite -> IHxs.
reflexivity.
Qed.

Theorem DoubleInverse {A}: forall (xs:List A), reverse (reverse xs)=xs.
induction xs.
simpl.
reflexivity.
simpl.
rewrite RevSnoc.
rewrite -> IHxs.
reflexivity.
Qed.

Theorem SingleAddFromBack {A}: forall (x:A), forall (xs:List A), xs ++ (Cons A x (Nil A))=snoc xs x.
induction xs.
simpl.
reflexivity.
simpl.
rewrite <- IHxs.
reflexivity.
Qed.

Theorem RevConcat{A}: forall (xs ys:List A), reverse (xs++ys) = (reverse ys) ++ (reverse xs).
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

Lemma CCAssoc{A}: forall (xs ys zs:List A), (xs ++ ys)++zs = xs ++(ys++zs).
induction xs.
simpl.
reflexivity.
simpl.
intros.
rewrite IHxs.
reflexivity.
Qed.

Lemma Rev'Eq{A}: forall (xs ys : List A), reverse' xs ys = reverse' xs (Nil A) ++ ys.
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

Theorem RevEqRev'{A}: forall (xs :List A), reverse xs = reverse' xs (Nil A).
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

Eval compute in (length (.) reverse) (Cons nat 1 (Nil nat)).


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


Theorem MapCC {A B}: forall (f:A->B), forall (xs ys: List A), map f ( concat xs ys) = concat (map f xs) (map f ys).
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

Inductive BinTree A :=
 | Leaf:BinTree A
 | Bin: BinTree A -> A-> BinTree A-> BinTree A
.

Fixpoint mirror {A} (tree:BinTree A) :=
 match tree with
 |Leaf _=> Leaf _
 |Bin _ l x r => Bin _ (mirror r) x (mirror l)
end.

Fixpoint inorder {A} (tree: BinTree A):List A :=
 match tree with
 | Leaf _ =>Nil _
 | Bin _ l x r=> concat (inorder l) (Cons _ x (inorder r))
end.

Theorem DoubleMirror {A}: forall (t: BinTree A), mirror (mirror t) = t.
induction t.
simpl.
reflexivity.
simpl.
rewrite IHt1.
rewrite IHt2.
reflexivity.
Qed.

Lemma ShiftSnoc {A}: forall (xs:List A), forall (ys:List A), forall (x:A), concat (snoc xs x) ys = concat xs (Cons A x ys).
induction xs.
intros.
simpl.
reflexivity.
intros.
simpl.
rewrite IHxs.
reflexivity.
Qed.

Theorem MirroredFlatten{A}: forall (t: BinTree A), inorder (mirror t) = reverse (inorder t).
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

Print Assumptions MirroredFlatten.







