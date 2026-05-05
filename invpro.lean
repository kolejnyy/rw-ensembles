import Mathlib

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

theorem thm1 (x y : ℕ) (h₀ : Nat.lcm x y = 12)
  (h₁ : Nat.gcd x y = 2) : x * y = 24 := by
  rw [← Nat.gcd_mul_lcm x y]
  norm_num [h₀, h₁]
