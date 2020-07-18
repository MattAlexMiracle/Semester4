
Inductive List A:=
  | Nil: List A
  |Cons:A->List A->List A
.

Fixpoint snoc {A} (xs:List A) (y:A) := match xs with
 | Nil _=> Cons _ y (Nil _)
 | Cons _ x xs => Cons _ x (snoc xs y)
end.

Fixpoint concat {A} (xs:List A) (ys:List A) :=
 match xs with
 | Nil _=>ys
 | Cons _ x xs => Cons _ x (concat xs ys)
end.
Infix "++" := concat (at level 60, right associativity).



Variable Sprite:Type.
Variable blankSprite:Sprite.
Variable s1 s2 s3 s4 s5 s6:Sprite.

(* Die defintion in Coq läuft über eine unendliche anwendung einer Regel,
 * advance/sprite werden extra als CoFixpunkte implementiert.
 * Für diese aufgabe macht es mehr sinn das ganze als positiv coinductiv zu sehen (wir haben einen Konstruktor)
 * Für die nächest wird dann das negative verfahren benutzt, was seit Coq 8 stand der Technik ist.
*)
CoInductive Animation:= 
 loop:List Sprite->Animation
.

Definition sprite (x:Animation) := match x with
 |loop (Nil _) => blankSprite
 |loop (Cons _ x xs) => x
end.

Definition advance (x:Animation) := match x with
 | loop (Nil _) => loop (Nil _)
 | loop (Cons _ s ss)=> loop (snoc ss s)
end.

Check advance.
Check sprite.

Eval compute in sprite(advance (loop (Cons _ s1 (Cons _ s2 (Nil _))))).

Fixpoint prepend (x:List Sprite) (a1:Animation):= match x, a1 with
 | Nil _, loop z=> a1
 | Cons _ y ys, loop z=> prepend ys a1
end.



Inductive Bool :=
 |True:Bool
 |False: Bool
.

Variable Compatible: Sprite -> Sprite -> Bool.

Definition transition (a1: Animation) (a2: Animation):Animation:= 
 match Compatible (sprite a1)  (sprite a2) with
 | True => advance a2
 | False=> a1
end.

Fixpoint not (x:Bool) := match x with 
 | True=>False
 | False=>True
end.
Lemma BoolComplete: not True = False.
simpl.
reflexivity.
Qed.

(* Hier kann man nochmal besondes schön sehen: Ich brauche keine Bisimulation*)

Theorem IncompatibleStreams: forall (s t :Sprite), forall (ts: List Sprite),
 Compatible s t = False -> transition (loop (Cons _ s (Nil _))) (loop (Cons _ t ts)) = loop (Cons _ s (Nil _)).
intros.
unfold transition.
simpl.
rewrite ->H.
reflexivity.
Qed.

Theorem TransitionToPrepend: forall (s t:Sprite), forall (ss ts: List Sprite), Compatible s t = True ->
 transition (loop (Cons _ s ss)) (loop (Cons _ t ts)) = prepend (Cons _ s (Nil _)) (loop (snoc ts t)).
intros.
unfold transition.
simpl.
rewrite -> H.
reflexivity.
Qed.


CoInductive ITree A := Tree { inner : A; left : ITree A; right : ITree A }.


Check ITree.
Check left.

Check Tree.
Check inner.


(*Ich krieg LISP PTSD*)
CoFixpoint itadd (a: ITree nat) (b:ITree nat) := Tree _ ((inner _ a)+(inner _ b)) (itadd (left _ a) (left _ b)) (itadd (right _ a) (right _ b)) .

CoInductive Stream A:= Seq {tl:Stream A; hd: A}.

CoFixpoint flip (s: Stream Bool ):= Seq _ (flip (tl _ s)) (not( hd _ s)).

CoFixpoint choose {A} (s: Stream Bool) (t:ITree A):= if hd _ s then
 Seq _ (choose (tl _ s) (left _ t)) (inner _ t)
 else Seq _ (choose (tl _ s) (right _ t)) (inner _ t).

CoFixpoint mirror {A} (tree: ITree A) := Tree _ (inner _ tree) (mirror (right _ tree)) (mirror (left _ tree)).


CoInductive BisimTree {A} (t1 t2: ITree A):Prop := bisim{
  eq_inner : inner _ t1 = inner _ t2;
  eq_left: BisimTree (left _ t1) (left _ t2);
  eq_right: BisimTree (right _ t1) (right _ t2);
}.



Ltac infiniteproof f:= cofix f; constructor; [clear f| simpl; try (apply f; clear f)].
Lemma Plus_commut: (forall (t1 t2: nat), t1+t2 = t2+t1).
Admitted.

(*hier erst mal ohne meta-taktik infiniteproof*)
Theorem AssociativeTreeAdd: forall (t1 t2: ITree nat), BisimTree (itadd t1 t2) (itadd t2 t1).
cofix f.
constructor.
simpl.
rewrite <- Plus_commut.
reflexivity.
apply f.
apply f.
Qed.

(*Bisimilarität bei Streams*)

CoInductive BisimStream {A} (s1 s2:Stream A):Prop := bisimSt{
  eq_hd : hd _ s1 = hd _ s2;
  eq_tl : BisimStream (tl _ s1) (tl _ s2);
}.

Lemma FlippableIf {B}: forall (s:Bool), forall (y x:B), (if not s then x else y) = (if s then y else x).
induction s; simpl; reflexivity.
Qed.
Lemma ExtractableIf {B C}: forall (f: B->C), forall (b:Bool), forall (inp1 inp2: B), f (if b then inp1 else inp2) = if b then (f (inp1)) else (f (inp2)).
induction b;reflexivity.
Qed.

Print choose.

Print well_founded.


Theorem FlippedPath {A}: forall (s: Stream Bool), forall (t: ITree A), BisimStream (choose (flip s) (mirror t)) (choose s t).
cofix f.
intros.
constructor.
simpl.
case (hd Bool s);
simpl;
reflexivity.
simpl.
induction (hd Bool s).
simpl.
apply f.
simpl.
apply f.
Qed.















