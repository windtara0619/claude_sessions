---
name: kvproj-fusion-race-condition-fix
description: "has_kvproj=True race condition: cache READ started before cache WRITE; two-part fix in kernel.py"
metadata: 
  node_type: memory
  type: project
  originSessionId: 43e7d40c-f46f-436b-ae56-41baa7e1974c
---

## Root cause of has_kvproj=True numerical failures

For `has_kvproj=True`, `bq_idx=0` eagerly computes K/V for ALL `bkv_sz` rows (not just its own `bq_sz` tokens) and writes them to the KV cache via `start_update_kv_cache`. Later `bq_idx` tiles read those pages back from cache.

**The bug**: `prefetch_next_bkv()` started the cache READ DMA (HBM→VMEM) before `start_update_kv_cache()` started the cache WRITE DMA (VMEM→HBM). TPU uses **separate DMA engines** for HBM→VMEM reads and VMEM→HBM writes — they run concurrently regardless of submission order, so the read completed with stale (all-zero) data.

**Why it only triggers with ≥2 bq tiles**: With 1 bkv tile per bq, the prefetch for `bq_idx=1` uses a DIFFERENT semaphore slot (1) than the write for `bq_idx=0` (slot 0). The existing `wait_update_kv_cache(bkv_sem_idx)` only guards the SAME slot, missing the cross-slot race.

## Two-part fix (kernel.py)

**Fix 1** — `compute_with_bkv` (~line 1216): For `has_kvproj`, move `prefetch_next_bkv` to AFTER `start_update_kv_cache`. This ensures `bkv_update_ids_ref[other_slot+4]` is non-zero (write has been recorded) when `_fetch_bkv` checks it. For non-kvproj, keep early prefetch for pipeline overlap.

**Fix 2** — `_fetch_bkv` (not-wait branch, ~line 637): For `has_kvproj` when `bkv_sz_frm_cache > 0`, explicitly wait for the OTHER semaphore slot's pending write before starting the cache read DMA:
```python
if has_kvproj:
    @pl.when(bkv_sz_frm_cache > 0)
    def _wait_prior_write():
        other_sem_idx = lax.select(bkv_sem_idx == 0, jnp.int32(1), jnp.int32(0))
        wait_update_kv_cache(other_sem_idx)
```

Both fixes are needed: Fix 1 ensures the write metadata is recorded before Fix 2 tries to wait on it.

**Why**: `wait_update_kv_cache` checks `bkv_update_ids_ref[sem_idx+4]` — this is only non-zero if `start_update_kv_cache` was called first (Fix 1). Then it explicitly waits for the async write DMA to complete (Fix 2), preventing the concurrent read from seeing stale data.

## Related bugs fixed in same session

- `compute_q_rope_norm` (line 870): `qn_scale_vmem_ref[:head_dim][None, :]` — needed explicit `[None, :]` for rank-promotion safety under `jax_numpy_rank_promotion='raise'`
- `compute_kv_from_x_bkv` (line 922): same `[None, :]` fix for `kn_scale_vmem_ref`
- Test file: flax `.value` → `[...]` for 5 weight accesses in `_build_qkv_baseline`
- Test file: baseline uses `ref_ragged_paged_attention` (not JIT) to avoid `no_tracing` re-trace error

## Test results

4 new fusion tests all pass after fix:
- `test_ragged_paged_attention_qkvproj_fusion0` (bq_sz=64, bkv_sz=256, q_len=384)
- `test_ragged_paged_attention_qkvproj_fusion1` (bq_sz=256, bkv_sz=1024, q_len=384)
- `test_ragged_paged_attention_qkvproj_multiseq_fusion0` (bq_sz=64, bkv_sz=256, multi-seq)
- `test_ragged_paged_attention_qkvproj_multiseq_fusion1` (bq_sz=256, bkv_sz=1024, multi-seq)

[[qproj-fusion]], [[has_kvproj-design]]
