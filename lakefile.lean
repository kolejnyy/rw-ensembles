import Lake
open Lake DSL

package «invpro» where
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.9.0-rc2"

@[default_target]
lean_lib «invpro» where
