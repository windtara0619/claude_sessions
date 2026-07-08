---
name: bkv-update-ids-xla-buffer-bug
description: "mega_kernel=True hang at input-len=992: XLA aliases shared init_bkv_update_ids HBM buffer causing scalar_prefetch DMA to read garbage; fix: move sem_ids/bo_ids/bkv_update_ids to pltpu.SMEM scratch"
metadata: 
  node_type: memory
  type: project
  originSessionId: 43e7d40c-f46f-436b-ae56-41baa7e1974c
---

## Bug: hang at input-len=992 (not 991) with MEGA_KERNEL=true

`MEGA_KERNEL=true python3 examples/tpu_profiling.py --model Qwen/Qwen3-4B --input-len 992 --output-len 1 --batch-size 16 --num-iters-warmup 2` hangs; 991 does not.

## Root cause

`init_bkv_update_ids = jnp.full((6,), -1, jnp.int32)` was created ONCE at `ragged_paged_attention` scope and shared as `scalar_prefetch` across all `run_rpa_kernel` calls (TC0@894 1-seq kernel AND TC0@900 15-seq kernel for iter2).

With `PrefetchScalarGridSpec`, each pallas_call invocation fires the kernel body TWICE:
1. First fire (preamble): scalar ops only, SMEM has stale values from the previous invocation
2. DMA loads fresh values from the HBM `init_bkv_update_ids` buffer into SMEM
3. Second fire (main body): all ops, SMEM has freshly DMA'd values

For TC0@900 (15-seq, first MIXED invocation): XLA had **already reused the `init_bkv_update_ids` HBM buffer** for other computations (TC0@894's 36 layers ran first and exhausted XLA's liveness estimate). The DMA reads garbage values like `[-1135428526, ..., 1012939872]`. The `wait_update_kv_cache` check `update_sz > 0` fires on the garbage `update_sz = 1012939872` → waits on a DMA that never completes → **deadlock**.

**Why 991 works**: at input-len=991, TC0@894 and TC0@900 execute in a different schedule where TC0@900's scalar_prefetch DMA still reads the valid buffer.

**Why "move inside run_rpa_kernel" didn't fix it**: XLA CSE'd the identical `jnp.full` constants back to one shared node, so all `run_rpa_kernel` calls still shared the same aliased HBM buffer.

**Why `jax.debug.print(init_bkv_update_ids)` accidentally fixed it**: the print created a data dependency on `init_bkv_update_ids` at TC0@900's call site, preventing XLA from reusing the buffer.

## Fix (kernel.py) — SMEM scratch approach

Moved `sem_ids`, `bo_ids`, `bkv_update_ids` from `scalar_prefetch` (HBM DMA → SMEM) to `pltpu.SMEM` scratch (on-chip scalar memory, no HBM buffer, no XLA aliasing). The kernel body initializes them directly (scalar SMEM stores run in BOTH the preamble and main body firings), ensuring fresh -1s before the loop runs in the main body.

```python
# scratch_shapes — last 3 entries:
pltpu.SMEM((4,), jnp.int32),  # sem_ids
pltpu.SMEM((4,), jnp.int32),  # bo_ids
pltpu.SMEM((6,), jnp.int32),  # bkv_update_ids

# scalar_prefetches — reduced from 7 to 4:
scalar_prefetches = (kv_lens, page_indices, cu_q_lens, distribution)

# _ragged_paged_attention_kernel — initialize at top of every firing:
sem_ids_ref[0] = jnp.int32(0); ...  (4 entries)
bo_ids_ref[0] = jnp.int32(-1); ...  (4 entries)
bkv_update_ids_ref[0] = jnp.int32(-1); ...  (6 entries)
```

**input_output_aliases** changed from `{7: 0, 9: 1}` to `{4: 0, 6: 1}` (q shifted 7→4, kv_cache shifted 9→6, due to 3 fewer scalar_prefetch items).

**Arg layout after fix (27 total)**:
- scalar_prefetch (4): kv_lens(0), page_indices(1), cu_q_lens(2), distribution(3)
- inputs (7): q(4), kv(5), kv_cache(6), x(7), w_qkv(8), qn_scale(9), kn_scale(10)
- outputs (2): o(11), updated_kv_cache(12)
- scratch (14): bkv_x2(13), bq_x2(14), bo_x2(15), sems(16), l(17), m(18), acc(19), rope(20), x_tile(21), x_bkv(22), x_proj_sems(23), **sem_ids(24), bo_ids(25), bkv_update_ids(26)**

## Debugging evidence

- TC0@894: shows 2 KERNEL START prints per invocation (PrefetchScalarGridSpec double-fire)
- TC0@900 without fix: first fire sees `[-1135428526,...,1012939872]` → hang
- TC0@900 with fix: both fires see `[-1,-1,-1,-1,-1,-1]` from in-kernel initialization → no hang

**How to apply**: Any future mega_kernel hang where `wait_update_kv_cache` fires spuriously (first kernel invocation seeing garbage `bkv_update_ids_ref`) — check if init arrays are being shared as constants across multiple `run_rpa_kernel` calls and getting aliased by XLA. The solution is `pltpu.SMEM` scratch initialized in the kernel body, NOT `scalar_prefetch` HBM constants.
