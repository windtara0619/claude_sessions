# Memory Index

- [Profiling runs are user-run](feedback_profiling_runs.md) — don't launch tpu_profiling.py; user runs it and pastes errors
- [Batched decode mega_kernel](project_decode_batched_mega_kernel.md) — bq_sz counts sequences; batch-head Q/KV GEMM design + list of pre-existing test failures on mega_kernel branch

- [RoPE fusion investigation](project_rope_fusion_investigation.md) — Full findings from FUSE_ROPE_INTO_ATTN_KERNEL regression on mega_kernel branch; root causes, ablations, cost accounting
- [Q projection fusion (has_qproj)](project_qproj_fusion.md) — has_qproj=True implementation: fused Q projection in RPA kernel, 3D x HBM layout trick, key gotchas
- [KV projection fusion race condition fix](project_kvproj_fusion.md) — has_kvproj=True bug: cache READ before WRITE race; two-part fix: move prefetch after write + explicit cross-slot wait
- [mega_kernel XLA buffer aliasing bug](project_bkv_update_ids_xla_buffer_bug.md) — hang at input-len=992: shared init_bkv_update_ids constant aliased by XLA; fixed by moving init arrays inside run_rpa_kernel
