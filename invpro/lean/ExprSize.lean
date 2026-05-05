/-
Expression size utilities for Lean 4 `Expr`.

Size is the number of printed components: we use the same pretty-printer as display (Expr → Format),
then count each `Format.text` node, excluding brackets `()` so they don't contribute.
-/
import Lean.Meta

namespace invpro.lean.ExprSize

open Lean

/-- Count Format nodes that contribute printed output (each `text` is one component; brackets don't count). -/
partial def formatNumComponents (f : Format) : Nat :=
  match f with
  | .nil => 0
  | .line => 0
  | .align _ => 0
  | .text s => if s == "(" || s == ")" then 0 else 1
  | .nest _ g => formatNumComponents g
  | .append f1 f2 => formatNumComponents f1 + formatNumComponents f2
  | .group g _ => formatNumComponents g
  | .tag _ g => formatNumComponents g

/-- Max depth of a contributing node (same notion as formatNumComponents; .nest increases depth). -/
partial def formatMaxDepthAt (f : Format) (depth : Nat) : Nat :=
  match f with
  | .nil => 0
  | .line => 0
  | .align _ => 0
  | .text s => if s == "(" || s == ")" then 0 else depth
  | .nest _ g => formatMaxDepthAt g (depth + 1)
  | .append f1 f2 => Nat.max (formatMaxDepthAt f1 depth) (formatMaxDepthAt f2 depth)
  | .group g _ => formatMaxDepthAt g depth
  | .tag _ g => formatMaxDepthAt g depth

/-- Max nesting depth among contributing printed components (root = 0). -/
def formatMaxDepth (f : Format) : Nat := formatMaxDepthAt f 0

/-- Number of printed components for the expression (same Format tree as display; runs in MetaM for ppExpr). -/
def exprSyntaxSize (e : Expr) : MetaM Nat := do
  let fmt ← Meta.ppExpr e
  pure (formatNumComponents fmt)

/-- Max nesting depth for the expression (same Format tree; runs in MetaM for ppExpr). -/
def exprMaxDepth (e : Expr) : MetaM Nat := do
  let fmt ← Meta.ppExpr e
  pure (formatMaxDepth fmt)

end invpro.lean.ExprSize
