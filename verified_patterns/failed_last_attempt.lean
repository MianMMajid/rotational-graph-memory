import Mathlib.Data.Int.Basic

theorem sum_of_even_integers_is_even (p q : ℤ) (hp : Even p) (hq : Even q) : Even (p + q) :=
begin
  rcases hp with ⟨k, hk⟩,
  rcases hq with ⟨l, hl⟩,
  use k + l,
  rw [hk, hl],
  ring,
end
