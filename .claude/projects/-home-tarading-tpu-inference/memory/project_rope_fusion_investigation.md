---
name: rope-fusion-investigation
description: Full findings from the FUSE_ROPE_INTO_ATTN_KERNEL regression investigation on the mega_kernel branch
metadata: 
  node_type: memory
  type: project
  originSessionId: 43e7d40c-f46f-436b-ae56-41baa7e1974c
---

Investigated the performance regression from `FUSE_ROPE_INTO_ATTN_KERNEL=true` (kernel `has_rope=True`). The kernel is `RPAm-p_128-bq_256_128-bkv_1024_512` (ragged paged attention prefill), source `kernel.py:2069`.

**Why:** Regression was measured at +273.93µs/layer (RPAm: 320.68→594.61µs) in a single-step offline trace (old trace) and +195.30µs/call at 2048-token microbenchmark on v6e-1.

**How to apply:** When optimizing RoPE fusion, the three root causes are (1) sin/cos recompute per bq-tile, (2) Q rotation inside the bkv loop (4× redundant), (3) K rotation of full bkv tiles including cached rows repeated per bq-tile.

Root causes in `kernel.py`:
- `load_rope_sincos` (line ~856): called 16×/layer (8 bq_tiles × 2 bq_chunks); standalone CSEs across 36 layers
- `load_bq` (line ~881): Q rotation inside bkv inner loop → 4× redundant per Q token
- `rotate_inplace_bkv_k` (line ~981): rotates full bkv_sz=1024 tile including cached rows, called num_bq × num_new_bkv times

Ablation results at q_len=2048 fresh prefill (v6e-1, RPAm baseline=448.23µs, no_rope=252.93µs, delta=195.30µs):
- skip sin/cos compute: saves 129µs
- skip apply Q rotation: saves 123µs
- skip apply K rotation+writeback: saves 80µs
- skip all: recovers no_rope baseline exactly

Correct framing for FALSE vs TRUE cost:
- FALSE (standalone apply_rope): DMA-bound — Q=110.5µs, K=32.4µs for 2048 tokens
- TRUE (kernel internal): compute-bound — Q=123µs (1.1× standalone), K=80µs (2.5× standalone)
- The "12×" apparent gap was a scale mismatch: old trace "8µs" was for ~256-token Q, not 2048

The 27µs (256-token bench) vs 273µs (real trace) gap: explained by 8× more bq_tiles (2048-token vs 256-token input) giving 8× sin/cos + 4× Q + 16× K work, plus 1.27× device speed factor on the serving machine.
