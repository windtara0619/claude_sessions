---
name: project-qproj-fusion
description: has_qproj=True implementation in RPA v3 kernel — fused Q projection to eliminate Q HBM round-trip
metadata: 
  node_type: memory
  type: project
  originSessionId: 43e7d40c-f46f-436b-ae56-41baa7e1974c
---

## Status: IMPLEMENTED AND TESTED (2026-06-16)

`has_qproj=True` is fully working in `tpu_inference/kernels/ragged_paged_attention/v3/kernel.py` on the `mega_kernel` branch.

**What it does:** When `has_qproj=True`, the kernel accepts `x` (hidden states, `[T, H]`) and `w_q` (Q projection weight, `[H, N*D]`) instead of pre-projected Q, and computes `q = x @ W_q` inside the bq outer loop. This eliminates:
- copy.3 write of Q to HBM (~40.7 µs)
- Q read by kernel from HBM (~73 µs)
- Target: ~113 µs improvement at serving shape (2×4096 tokens, Qwen3-4B)

**Key implementation decisions:**

1. **x HBM shape = `[max_num_tokens, hidden_size//128, 128]`** (3D, NOT 2D `[T, H]`). Reason: for a 2D bf16 array, the Mosaic HBM tile is (8, 128) and the outermost dim gets tile=8. Any dynamic token offset must then be a multiple of 8. By making it 3D, the token dimension becomes the outermost dim with tile=1 (no alignment constraint), allowing arbitrary `q_len_start` offsets.

2. **VMEM x tile = `[2, bq_sz, hidden_size//128, 128]`** (matches HBM 3D layout). Flattened to `[bq_sz, hidden_size]` for the matmul via `reshape`.

3. **W_q stays 2D `[hidden_size, N*head_dim]`** in HBM/VMEM. Loaded once into VMEM in the prologue (W_q is constant across all bq tiles).

4. **`preferred_element_type=jnp.float32`** on `jnp.dot` — Mosaic requires 32-bit accumulation for bf16 matmul.

5. **Decode always uses `has_qproj=False`** — bq_sz=1 violates VMEM tile alignment.

6. **`has_qproj` added to `static_argnames`** in `@jax.jit` decorator — otherwise JAX traces it as a dynamic bool causing TracerBoolConversionError.

7. **Correctness**: result is bit-for-bit identical to pre-projected baseline when using bf16-quantized inputs for both paths.

**Why:** hidden_size must be divisible by 128 (Qwen3-4B: 2560 = 20×128 ✓).

**Files changed:**
- `tpu_inference/kernels/ragged_paged_attention/v3/kernel.py`
- `tests/kernels/ragged_paged_attention_kernel_v3_test.py` (new test `test_ragged_paged_attention_qproj_fusion`)

**has_kvproj compilation**: Adding `has_kvproj=True` (k_proj + k_norm + v_proj) causes Mosaic's scheduler to take ~5-10 minutes to compile (vs 22s for has_qproj alone). This is because Mosaic's memory scheduling is super-linear in the number of matmuls. The compilation DOES succeed (verified via background run). JAX caches the result so it's a one-time cost. The KV projection reuses the Q x-tile (requires bq_sz == bkv_sz).

**has_kvproj architecture**: `w_k` and `w_v` are concatenated into `w_kv = [W_k | W_v]` (shape [D, 2*K*H]) to do a single fused matmul per bkv tile instead of two. The result is split and k-norm is applied to the K slice. Results written to bkv_x2_ref via per-head strided_store (same as store_bkv_k).
