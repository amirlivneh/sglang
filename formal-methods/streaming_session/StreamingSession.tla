---------------------------- MODULE StreamingSession ----------------------------
(* Streaming-session admission and close protocol of SGLang
   (python/sglang/srt/session/session_controller.py, streaming_session.py,
    managers/scheduler_components/request_retirer.py).

   A streaming session runs at most one turn at a time on its KV slot. The
   guard is Session._inflight: create_req rejects a new turn while it is set,
   and SessionController._close defers the KV release while it is set.

   BUGGY = TRUE models the code as written: retiring ANY aborted request of the
   session calls Session.abort_req(), which clears _inflight unconditionally,
   including a request that create_req itself rejected because another turn was
   in flight.  BUGGY = FALSE models the fix: only the in-flight turn clears it. *)
EXTENDS FiniteSets, Naturals

CONSTANTS Reqs, BUGGY

VARIABLES
    status,     \* r -> "new" | "accepted" | "rejected" | "done"
    inflight,   \* Session._inflight
    slotFreed,  \* the session's KV slot (row + pages) went back to the pool
    closed      \* the session was closed (explicitly or by the timeout reaper)

vars == <<status, inflight, slotFreed, closed>>

Running == {r \in Reqs : status[r] = "accepted"}

Init ==
    /\ status = [r \in Reqs |-> "new"]
    /\ inflight = FALSE
    /\ slotFreed = FALSE
    /\ closed = FALSE

(* Session.create_req: reject while a turn is in flight. *)
Create(r) ==
    /\ status[r] = "new"
    /\ ~closed
    /\ IF inflight
         THEN /\ status' = [status EXCEPT ![r] = "rejected"]
              /\ UNCHANGED inflight
         ELSE /\ status' = [status EXCEPT ![r] = "accepted"]
              /\ inflight' = TRUE
    /\ UNCHANGED <<slotFreed, closed>>

(* The rejected request is queued with to_finish set; find_active_slot or
   RequestRetirer._detach then calls session.abort_req(). *)
RetireRejected(r) ==
    /\ status[r] = "rejected"
    /\ status' = [status EXCEPT ![r] = "done"]
    /\ inflight' = IF BUGGY THEN FALSE ELSE inflight
    /\ UNCHANGED <<slotFreed, closed>>

(* The running turn finishes (finish_req) or aborts (abort_req); either way it
   stops reading the slot. *)
EndTurn(r) ==
    /\ status[r] = "accepted"
    /\ status' = [status EXCEPT ![r] = "done"]
    /\ inflight' = FALSE
    /\ UNCHANGED <<slotFreed, closed>>

(* SessionController._close / maybe_reap: release now unless a live reader
   exists (_has_live_readers == _inflight for a text-only streaming session). *)
Close ==
    /\ ~closed
    /\ ~inflight
    /\ closed' = TRUE
    /\ slotFreed' = TRUE
    /\ UNCHANGED <<status, inflight>>

Next ==
    \/ \E r \in Reqs : Create(r) \/ RetireRejected(r) \/ EndTurn(r)
    \/ Close

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ status \in [Reqs -> {"new", "accepted", "rejected", "done"}]
    /\ inflight \in BOOLEAN /\ slotFreed \in BOOLEAN /\ closed \in BOOLEAN

(* At most one turn decodes on the slot at a time. *)
OneTurnOnSlot == Cardinality(Running) <= 1

(* The slot is never freed under a running turn (KV use-after-free). *)
NoFreeUnderRunningTurn == slotFreed => Running = {}
==============================================================================
