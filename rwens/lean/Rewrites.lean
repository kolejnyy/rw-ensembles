/-
Tactic `rwcnc`: finds applicable rewrites (single- or multi-step) and shows them as "Try this".
Does not apply any rewrite. Uses Lean's core Meta layer (Lean.Meta.Tactic.Rewrites).

Syntax: `rwcnc 10 2 at h` = at each step consider top 10 rewrites, do up to 2 steps at <loc>.
- First number: max suggestions per step (default 20).
- Second number: number of rewrite steps (default 1).
- Third: optional `true` (default false). When `true`, only show rewrites that do not increase complexity or depth.
- Fourth: optional `reverse` (default false). When set, traverse rewrite suggestions in reverse order.
- Location: optional (goal or `at h`).

Compatible with Lean 4.9.0-rc2.
-/
import Lean.Elab.Tactic.Location
import Lean.Meta.Tactic.Replace
import Lean.Meta.Tactic.Rewrites
import rwens.lean.ExprSize

namespace rwens.lean.Rewrites

open Lean
open Lean.Meta (withMCtx)
open Lean.Meta.Rewrites
open Lean.Parser.Tactic
open Lean.Elab (throwUnsupportedSyntax)
open Lean.Elab.Tactic
open rwens.lean.ExprSize

/-- Extract head function from an application (Lean 4.9 compatibility - core Expr may not have getAppFn). -/
private def getAppFn (e : Expr) : Expr :=
  match e with
  | .app f _ => getAppFn f
  | _ => e

/-- Build "rw [← n1, n2] at h" (or without " at h") from steps and optional location name.
For non-constant rules (e.g. local hypotheses), use pretty-printed expression name.
-/
def rwTacticString (steps : List (Expr × Bool)) (locName? : Option Name) : TacticM String := do
  let item (e : Expr) (symm : Bool) : TacticM String := do
    let base ←
      match getAppFn e with
      | .const n _ => pure (toString n)
      | _ =>
          let fmt ← Lean.Meta.ppExpr e
          pure fmt.pretty
    pure ((if symm then "← " else "") ++ base)
  let parts ← steps.mapM (fun (e, s) => item e s)
  let rwCore := "rw [" ++ (", ".intercalate parts) ++ "]"
  match locName? with
  | some n => pure (rwCore ++ " at " ++ toString n)
  | none => pure rwCore

/-- True when rewrite expression refers to a local hypothesis (fvar-headed application). -/
private def isLocalHypRewriteExpr (e : Expr) : Bool :=
  match getAppFn e with
  | .fvar _ => true
  | _ => false

/-- True when every rewrite step in the sequence is from local hypotheses. -/
private def stepsAreOnlyLocalHyps (steps : List (Expr × Bool)) : Bool :=
  steps.all (fun (e, _) => isLocalHypRewriteExpr e)

syntax rwcncOnlySimplify := "true"
syntax rwcncReverse := "reverse"
syntax (name := rwcnc) "rwcnc" (num)? (num)? (ppSpace rwcncOnlySimplify)? (ppSpace rwcncReverse)? (ppSpace location)? : tactic

@[tactic rwcnc] def elabRwcnc : Tactic := fun stx => do
  let tk := stx
  let maxPerStep := if stx[1].isNone then 20 else
    match stx[1][0].isNatLit? with
    | some n => n
    | none => 20
  let depth := if stx[2].isNone then 1 else
    match stx[2][0].isNatLit? with
    | some n => n
    | none => 1
  let filterOnlySimplify := !stx[3].isNone
  let reverseOrder := !stx[4].isNone
  let moduleRef ← createModuleTreeRef
  let forbidden : NameSet := ∅
  reportOutOfHeartbeats `findRewrites tk
  let goal ← getMainGoal

  let locOpt := if stx[5].isNone then none else some (stx[5][0])

  withLocation (expandOptLocation (Lean.mkOptionalNode locOpt))
    (fun f => do
      let some a ← f.findDecl? | return
      if a.isImplementationDetail then return
      let target ← instantiateMVars (← f.getType)
      let hyps ← localHypotheses (except := [f])
      if depth = 0 then return

      let origComplexity ← exprSyntaxSize target
      let origDepth ← exprMaxDepth target
      let rec goHyp (mctx : MetavarContext) (stepsLeft : Nat) (goal : MVarId) (target : Expr)
          (origCompl : Nat) (origD : Nat) (acc : List (Expr × Bool)) (hypsAcc : Array (Expr × Bool × Nat)) : TacticM Unit := do
        let results ← withMCtx mctx (findRewrites hypsAcc moduleRef goal target forbidden (side := Lean.Meta.Rewrites.SideConditions.solveByElim) (stopAtRfl := Bool.false) (max := maxPerStep))
        reportOutOfHeartbeats `rewrites tk
        if acc.isEmpty && results.isEmpty then
          throwError "Could not find any lemmas which can rewrite the hypothesis {← f.getUserName}"
        let orderedResults := if reverseOrder then results.reverse else results
        for r in orderedResults do
          let steps := acc ++ [(r.expr, r.symm)]
          let (typeStr, newComplexity, newDepth) ← withMCtx r.mctx do
            let fmt ← Lean.Meta.ppExpr r.result.eNew
            pure (fmt.pretty, rwens.lean.ExprSize.formatNumComponents fmt, rwens.lean.ExprSize.formatMaxDepth fmt)
          let onlyLocalHyps := stepsAreOnlyLocalHyps steps
          let showIt := !filterOnlySimplify || onlyLocalHyps || (newComplexity <= origCompl && newDepth <= origD)
          if showIt then
            let locName ← f.getUserName
            let rwStr ← rwTacticString steps (some locName)
            Lean.Meta.Tactic.TryThis.addSuggestion tk
              { suggestion := .string rwStr
                messageData? := some (m!"{rwStr}\n-[rwcnc]- {typeStr}\n-[rwcnc]- complexity: {origCompl}->{newComplexity}, depth: {origD}->{newDepth}") }
              (origSpan? := ← getRef)
          if stepsLeft > 1 then
            let (newGoal, newMctx) ← withMCtx r.mctx do
              let replaceResult ← goal.replaceLocalDecl f r.result.eNew r.result.eqProof
              pure (replaceResult.mvarId, ← getMCtx)
            let newHyps ← withMCtx newMctx (localHypotheses (except := [f]))
            goHyp newMctx (stepsLeft - 1) newGoal r.result.eNew origCompl origD steps newHyps

      let mctx0 ← getMCtx
      goHyp mctx0 depth goal target origComplexity origDepth [] hyps)
    (do
      let target ← instantiateMVars (← goal.getType)
      let hyps ← localHypotheses
      if depth = 0 then return

      let origComplexity ← exprSyntaxSize target
      let origDepth ← exprMaxDepth target
      let rec goGoal (mctx : MetavarContext) (stepsLeft : Nat) (goal : MVarId) (target : Expr)
          (origCompl : Nat) (origD : Nat) (acc : List (Expr × Bool)) (hypsAcc : Array (Expr × Bool × Nat)) : TacticM Unit := do
        let results ← withMCtx mctx (findRewrites hypsAcc moduleRef goal target forbidden (side := Lean.Meta.Rewrites.SideConditions.solveByElim) (stopAtRfl := Bool.true) (max := maxPerStep))
        reportOutOfHeartbeats `rewrites tk
        if acc.isEmpty && results.isEmpty then
          throwError "Could not find any lemmas which can rewrite the goal"
        let orderedResults := if reverseOrder then results.reverse else results
        for r in orderedResults do
          let steps := acc ++ [(r.expr, r.symm)]
          let (typeStr, newComplexity, newDepth) ← withMCtx r.mctx do
            match r.newGoal with
            | none => pure ("(no goals)", 0, 0)
            | some e =>
              if e.isLit then pure ("(no goals)", 0, 0) else do
                let fmt ← Lean.Meta.ppExpr e
                pure (fmt.pretty, rwens.lean.ExprSize.formatNumComponents fmt, rwens.lean.ExprSize.formatMaxDepth fmt)
          let onlyLocalHyps := stepsAreOnlyLocalHyps steps
          let showIt := !filterOnlySimplify || onlyLocalHyps || (newComplexity <= origCompl && newDepth <= origD)
          if showIt then
            let rwStr ← rwTacticString steps none
            Lean.Meta.Tactic.TryThis.addSuggestion tk
              { suggestion := .string rwStr
                messageData? := some (m!"{rwStr}\n-[rwcnc]- {typeStr}\n-[rwcnc]- complexity: {origCompl}->{newComplexity}, depth: {origD}->{newDepth}") }
              (origSpan? := ← getRef)
          if stepsLeft > 1 then
            let (newGoalMvar, newMctx) ← withMCtx r.mctx do
              let g ← goal.replaceTargetEq r.result.eNew r.result.eqProof
              pure (g, ← getMCtx)
            let newHyps ← withMCtx newMctx localHypotheses
            goGoal newMctx (stepsLeft - 1) newGoalMvar r.result.eNew origCompl origD steps newHyps

      let mctx0 ← getMCtx
      goGoal mctx0 depth goal target origComplexity origDepth [] hyps)
    (fun _ => throwError "Failed to find a rewrite for some location")

end rwens.lean.Rewrites
