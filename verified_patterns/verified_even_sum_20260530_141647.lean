def Even (n : Int) : Prop := ∃ k : Int, n = 2 * k

theorem even_sum_of_even {p q : Int} (hp : Even p) (hq : Even q) : Even (p + q) :=
  match hp, hq with
  | ⟨k1, hk1⟩, ⟨k2, hk2⟩ =>
    ⟨k1 + k2, by rw [hk1, hk2, Int.mul_add]⟩
