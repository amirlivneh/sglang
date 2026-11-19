# Formal-methods checks

## Streaming session: a rejected turn releases the running turn's hold (bug)

`Session.abort_req()` cleared `_inflight` for *any* aborted request of a
streaming session. A turn that `create_req` rejects because another turn is in
flight ("Streaming session already has an active request.") is itself queued as
an abort, and its retirement (`find_active_slot` or `RequestRetirer._detach`)
called `abort_req()`, clearing the flag the running turn holds. After that:

1. a third turn is accepted while the first still runs, borrows the same
   `slot.kv`, and `_share_token_arrays` rewrites in place the token arrays the
   running turn still reads; its `_free_tail` frees the running turn's tail KV;
2. `close_session` (or the timeout reaper) sees no live reader and calls
   `release_session`, returning the running turn's row and KV pages to the
   pool while it decodes on them. When that turn finishes,
   `try_cache_finished_req` hits `assert slot is not None` and the scheduler
   crashes.

Trigger: two concurrent requests on one streaming session (a client retry is
enough).

- `streaming_session/StreamingSession.tla`: TLC model.
  `StreamingSession.cfg` (`BUGGY = TRUE`) violates `OneTurnOnSlot` in 4 steps
  and `NoFreeUnderRunningTurn` in 4 steps; `StreamingSessionFixed.cfg` passes
  (67 distinct states, 3 requests).
- `streaming_session/StreamingSession.lean`: Lean 4 proof that the fixed
  protocol keeps both invariants for every reachable state and any number of
  requests (`one_turn_on_slot`, `no_free_under_running_turn`), plus the buggy
  counterexample replayed as a theorem (`buggy_two_turns`).
- Regression tests: `TestStreamingSessionRejectedTurn` in
  `test/registered/unit/mem_cache/test_session_token_share_unit.py` (CPU only;
  both fail before the fix).

Run:

```bash
java -cp tla2tools.jar tlc2.TLC -deadlock -config StreamingSession.cfg StreamingSession.tla
java -cp tla2tools.jar tlc2.TLC -deadlock -config StreamingSessionFixed.cfg StreamingSession.tla
lean StreamingSession.lean   # Lean 4.15, no Mathlib
```

## Radix cache and KV allocator (no bug found)

`radix_cache/explore_radix.py` drives the real `RadixCache` with a real
`TokenToKVPoolAllocator` / `PagedTokenToKVPoolAllocator` on CPU through random
interleavings of admit (match, lock, alloc), decode, checkpoint, finish, abort
and evict over 4 concurrent requests, and checks after every step: every KV
slot is owned exactly once (free list, a tree node, or a request's own tail),
each request's tree-owned row prefix equals its locked path, `lock_ref` equals
the number of requests locking through each node, `evictable_size_` /
`protected_size_` / `evictable_leaves` match a recount. 3,600 runs (page size
1 and 2, EAGLE bigram keys, LRU / LFU / priority eviction) found no violation.

```bash
PYTHONPATH=python python formal-methods/radix_cache/explore_radix.py 200
```
