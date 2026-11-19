"""Bounded random state-space exploration of the real RadixCache + allocator.

Drives the request lifecycle (match -> lock -> alloc -> checkpoint* -> finish|abort)
interleaved with eviction across several concurrent requests, and checks the
invariants of the TLA+ model described in formal-methods/README.md after every step.
"""

import random
import sys
import unittest.mock
from array import array

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

MAX_LEN = 8
NUM_ROWS = 4


class ReqToTokenPool:
    def __init__(self):
        self.req_to_token = torch.zeros((NUM_ROWS, MAX_LEN + 4), dtype=torch.int64)

    def write(self, indices, values):
        self.req_to_token[indices] = values


class Sim:
    def __init__(self, seed, page_size, is_eagle, policy):
        self.rng = random.Random(seed)
        self.ps = page_size
        size = 64
        if page_size == 1:
            self.alloc = TokenToKVPoolAllocator(
                size=size, dtype=torch.float16, device="cpu", kvcache=None, need_sort=False
            )
        else:
            self.alloc = PagedTokenToKVPoolAllocator(
                size=size, page_size=page_size, dtype=torch.float16, device="cpu",
                kvcache=None, need_sort=False,
            )
        self.total_slots = self.alloc.available_size()
        self.cache = RadixCache.create_simulated(mock_allocator=self.alloc, page_size=page_size)
        self.cache.is_eagle = is_eagle
        if policy != "lru":
            from sglang.srt.mem_cache.utils import get_eviction_strategy
            self.cache.eviction_strategy = get_eviction_strategy(policy, None)
        self.pool = ReqToTokenPool()
        self.cache.req_to_token_pool = self.pool
        self.free_rows = list(range(NUM_ROWS))
        self.reqs = []
        self.trace = []

    # ---- request lifecycle -------------------------------------------------
    def new_req(self):
        if not self.free_rows:
            return
        prompt_len = self.rng.randint(1, 5)
        prompt = array("q", [self.rng.choice([1, 2]) for _ in range(prompt_len)])
        row = self.free_rows.pop()
        key = RadixKey(array("q", prompt), None, is_bigram=False)
        m = self.cache.match_prefix(MatchPrefixParams(key=key))
        prefix = self.cache.path_device_indices(m.last_device_node)
        plen = len(prefix)
        lock = self.cache.lock(m.last_device_node)
        # allocate the row up to the page-aligned max length
        need = -(-(MAX_LEN - plen) // self.ps) * self.ps
        if self.alloc.available_size() < need:
            self.cache.evict(EvictParams(num_tokens=need))
        new = self.alloc.alloc(need)
        if new is None:
            self.cache.unlock(lock)
            self.free_rows.append(row)
            self.trace.append(("admit_fail", list(prompt)))
            return
        self.pool.req_to_token[row, :plen] = prefix
        self.pool.req_to_token[row, plen : plen + need] = new
        kv = ReqKvInfo(req_pool_idx=row, cache_protected_len=plen)
        kv.set_row_len(plen + need)
        req = unittest.mock.Mock(
            kv=kv, extra_key=None, cache_salt=None, priority=self.rng.randint(0, 2),
            last_node=m.last_device_node, lock=lock,
        )
        req.origin_input_ids = prompt
        req.output_ids = array("q")
        req.full_untruncated_fill_ids = array("q", prompt)
        req.seq_len = len(prompt)
        req.rid = len(self.trace)
        self.reqs.append(req)
        self.trace.append(("admit", req.rid, list(prompt), plen))

    def decode(self, req):
        if req.seq_len >= MAX_LEN - 1:
            return
        t = self.rng.choice([1, 2])
        req.output_ids.append(t)
        req.full_untruncated_fill_ids.append(t)
        req.seq_len += 1
        self.trace.append(("decode", req.rid, t))

    def checkpoint(self, req):
        self.trace.append(("checkpoint", req.rid, req.seq_len))
        self.cache.checkpoint(req, up_to=req.seq_len)

    def finish(self, req, insert=True):
        self.trace.append(("finish" if insert else "abort", req.rid, req.seq_len))
        if insert:
            self.cache.checkpoint(req, up_to=req.seq_len)
        self.cache.free_kv_row(req.kv, [(req.kv.cache_protected_len, req.kv.kv_allocated_len)])
        self.cache.unlock(req.lock)
        req.lock = None
        self.free_rows.append(req.kv.req_pool_idx)
        self.reqs.remove(req)

    def evict(self):
        n = self.rng.randint(1, 6)
        self.trace.append(("evict", n))
        self.cache.evict(EvictParams(num_tokens=n))

    # ---- invariants ---------------------------------------------------------
    def tree_nodes(self):
        out, st = [], [self.cache.root_node]
        while st:
            n = st.pop()
            if n is not self.cache.root_node:
                out.append(n)
            st.extend(n.children.values())
        return out

    def check(self):
        c = self.cache
        nodes = self.tree_nodes()
        owners = {}

        def own(slot, who):
            assert slot not in owners, f"slot {slot} owned by {owners[slot]} and {who}"
            owners[slot] = who

        for s in self.alloc.free_pages.tolist() if self.ps == 1 else []:
            own(s, "free")
        if self.ps > 1:
            for p in self.alloc.free_pages.tolist():
                for s in range(p * self.ps, (p + 1) * self.ps):
                    own(s, "free")
        for n in nodes:
            for s in n.value.tolist():
                own(s, f"node{n.id}")
        for r in self.reqs:
            row = self.pool.req_to_token[r.kv.req_pool_idx]
            for s in row[r.kv.cache_protected_len : r.kv.kv_allocated_len].tolist():
                own(s, f"req{r.rid}")
            # tree-owned part of the row must match the locked path
            path = c.path_device_indices(r.lock.node)
            assert r.kv.cache_protected_len <= len(path), (r.kv.cache_protected_len, len(path))
            assert torch.equal(row[: r.kv.cache_protected_len], path[: r.kv.cache_protected_len]), "row/tree mismatch"
        assert len(owners) == self.total_slots, f"leak: {self.total_slots - len(owners)} slots unaccounted"
        # lock refs
        expect = {}
        for r in self.reqs:
            n = r.lock.node
            while n is not c.root_node:
                expect[n.id] = expect.get(n.id, 0) + 1
                n = n.parent
        for n in nodes:
            assert n.lock_ref == expect.get(n.id, 0), f"lock_ref node{n.id}={n.lock_ref} expect {expect.get(n.id, 0)}"
        ev = sum(len(n.key) for n in nodes if n.lock_ref == 0)
        pr = sum(len(n.key) for n in nodes if n.lock_ref > 0)
        assert c.evictable_size_ == ev, f"evictable_size_ {c.evictable_size_} != {ev}"
        assert c.protected_size_ == pr, f"protected_size_ {c.protected_size_} != {pr}"
        leaves = {n for n in nodes if n.lock_ref == 0 and not n.children}
        assert c.evictable_leaves == leaves, "evictable_leaves mismatch"
        for n in nodes:
            assert len(n.key) == len(n.value)
            assert len(n.key) % self.ps == 0

    def step(self):
        acts = ["new"] * 2 + ["evict"]
        if self.reqs:
            acts += ["decode"] * 3 + ["checkpoint", "finish", "abort"]
        a = self.rng.choice(acts)
        r = self.rng.choice(self.reqs) if self.reqs else None
        if a == "new":
            self.new_req()
        elif a == "evict":
            self.evict()
        elif a == "decode":
            self.decode(r)
        elif a == "checkpoint":
            self.checkpoint(r)
        elif a == "finish":
            self.finish(r, True)
        else:
            self.finish(r, False)
        self.check()


def main():
    nseeds = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    configs = [(ps, eagle, pol) for ps in (1, 2) for eagle in (False, True)
               for pol in ("lru", "lfu", "priority")]
    fails = 0
    for cfg in configs:
        for seed in range(nseeds):
            sim = Sim(seed, *cfg)
            try:
                for _ in range(40):
                    sim.step()
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL cfg={cfg} seed={seed}: {type(e).__name__}: {e}")
                for t in sim.trace[-12:]:
                    print("   ", t)
                break
    print("done, failing configs:", fails)


if __name__ == "__main__":
    main()
