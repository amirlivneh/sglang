/-
  Streaming-session admission protocol of SGLang
  (python/sglang/srt/session/session_controller.py).

  `inflight` is `Session._inflight` (fixed: the in-flight turn itself).
  We prove that with the fix, at most one turn runs on the session's KV slot
  and the slot is never freed under a running turn, for every reachable state
  and any number of requests. We also replay the TLC counterexample against
  the buggy step relation to show the invariant fails there.
-/

inductive Status where
  | new | accepted | rejected | done
  deriving DecidableEq, Repr

structure St where
  status   : Nat → Status
  inflight : Option Nat
  closed   : Bool
  freed    : Bool

def upd (f : Nat → Status) (r : Nat) (v : Status) : Nat → Status :=
  fun x => if x = r then v else f x

def init : St := ⟨fun _ => .new, none, false, false⟩

/-- One protocol step. `buggy` selects the code as written: retiring a
rejected turn clears the in-flight flag unconditionally. -/
inductive Step (buggy : Bool) : St → St → Prop where
  | createAccept (s : St) (r : Nat) :
      s.status r = .new → s.closed = false → s.inflight = none →
      Step buggy s { s with status := upd s.status r .accepted, inflight := some r }
  | createReject (s : St) (r : Nat) (q : Nat) :
      s.status r = .new → s.closed = false → s.inflight = some q →
      Step buggy s { s with status := upd s.status r .rejected }
  | retireRejected (s : St) (r : Nat) :
      s.status r = .rejected →
      Step buggy s { s with status := upd s.status r .done,
                            inflight := if buggy then none else s.inflight }
  | endTurn (s : St) (r : Nat) :
      s.status r = .accepted →
      Step buggy s { s with status := upd s.status r .done, inflight := none }
  | close (s : St) :
      s.closed = false → s.inflight = none →
      Step buggy s { s with closed := true, freed := true }

inductive Reachable (buggy : Bool) : St → Prop where
  | init : Reachable buggy init
  | step {s t : St} : Reachable buggy s → Step buggy s t → Reachable buggy t

/-- Inductive invariant of the fixed protocol. -/
def Inv (s : St) : Prop :=
  (∀ r, s.status r = .accepted ↔ s.inflight = some r) ∧
  (s.freed = true → s.closed = true ∧ s.inflight = none)

theorem inv_init : Inv init := by
  refine ⟨?_, ?_⟩
  · intro r; simp [init]
  · intro h; simp [init] at h

theorem inv_step {s t : St} (hi : Inv s) (hs : Step false s t) : Inv t := by
  obtain ⟨h1, h2⟩ := hi
  cases hs with
  | createAccept r hn hc hnone =>
    refine ⟨?_, ?_⟩
    · intro x
      simp only [upd]
      by_cases hx : x = r
      · subst hx; simp
      · have := h1 x
        rw [hnone] at this
        simp only [hx, if_false, this]
        constructor
        · intro h; cases h
        · intro h; injection h with h; exact absurd h.symm hx
    · intro hf
      have := (h2 hf).1
      rw [hc] at this; cases this
  | createReject r q hn hc hq =>
    refine ⟨?_, ?_⟩
    · intro x
      simp only [upd]
      by_cases hx : x = r
      · subst hx
        simp only [if_true]
        constructor
        · intro h; cases h
        · intro h
          have := (h1 x).2 h
          rw [hn] at this; cases this
      · simp only [hx, if_false]; exact h1 x
    · intro hf; exact h2 hf
  | retireRejected r hrej =>
    refine ⟨?_, ?_⟩
    · intro x
      simp only [upd, Bool.false_eq_true, if_false]
      by_cases hx : x = r
      · subst hx
        simp only [if_true]
        constructor
        · intro h; cases h
        · intro h
          have := (h1 x).2 h
          rw [hrej] at this; cases this
      · simp only [hx, if_false]; exact h1 x
    · intro hf
      simp only [Bool.false_eq_true, if_false] at hf ⊢
      exact h2 hf
  | endTurn r hacc =>
    have hr := (h1 r).1 hacc
    refine ⟨?_, ?_⟩
    · intro x
      simp only [upd]
      by_cases hx : x = r
      · subst hx; simp
      · simp only [hx, if_false]
        constructor
        · intro h
          have := (h1 x).1 h
          rw [hr] at this
          injection this with this
          exact absurd this.symm hx
        · intro h; cases h
    · intro hf
      exact ⟨(h2 hf).1, rfl⟩
  | close hc hnone =>
    refine ⟨?_, ?_⟩
    · intro x; exact h1 x
    · intro _; exact ⟨rfl, hnone⟩

theorem inv_reachable {s : St} (h : Reachable false s) : Inv s := by
  induction h with
  | init => exact inv_init
  | step _ hs ih => exact inv_step ih hs

/-- With the fix, two different requests are never both running. -/
theorem one_turn_on_slot {s : St} (h : Reachable false s) (a b : Nat)
    (ha : s.status a = .accepted) (hb : s.status b = .accepted) : a = b := by
  have ⟨h1, _⟩ := inv_reachable h
  have := (h1 a).1 ha
  rw [(h1 b).1 hb] at this
  cases this; rfl

/-- With the fix, the session's KV is never freed under a running turn. -/
theorem no_free_under_running_turn {s : St} (h : Reachable false s)
    (hf : s.freed = true) (r : Nat) : s.status r ≠ .accepted := by
  have ⟨h1, h2⟩ := inv_reachable h
  intro ha
  have := (h1 r).1 ha
  rw [(h2 hf).2] at this; cases this

/-- The buggy relation reaches a state with two running turns (TLC's trace:
r1 accepted, r2 rejected, r2 retired, r3 accepted). -/
theorem buggy_two_turns : ∃ s, Reachable true s ∧
    s.status 1 = .accepted ∧ s.status 3 = .accepted := by
  let s1 : St := { init with status := upd init.status 1 .accepted, inflight := some 1 }
  let s2 : St := { s1 with status := upd s1.status 2 .rejected }
  let s3 : St := { s2 with status := upd s2.status 2 .done, inflight := none }
  let s4 : St := { s3 with status := upd s3.status 3 .accepted, inflight := some 3 }
  have r1 : Reachable true s1 := .step .init (.createAccept init 1 rfl rfl rfl)
  have r2 : Reachable true s2 := .step r1 (.createReject s1 2 1 (by simp [s1, upd, init]) rfl rfl)
  have r3 : Reachable true s3 := .step r2 (.retireRejected s2 2 (by simp [s2, upd]))
  have r4 : Reachable true s4 :=
    .step r3 (.createAccept s3 3 (by simp [s3, s2, s1, upd, init]) rfl rfl)
  exact ⟨s4, r4, by simp [s4, s3, s2, s1, upd], by simp [s4, upd]⟩
