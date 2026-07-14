---
name: feedback-profiling-runs
description: "User runs examples/tpu_profiling.py themselves and pastes errors back; don't launch it from the session"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 43e7d40c-f46f-436b-ae56-41baa7e1974c
---

The user prefers to run `python3 examples/tpu_profiling.py ...` (and other engine-level runs) themselves, then paste the resulting error/trace location back into the chat.

**Why:** They rejected my attempts to launch the profiler via Bash multiple times (2026-07-14) and each time ran it themselves, pasting the EngineCore traceback or telling me where the profile landed (/tmp/tpu_profiler).

**How to apply:** After making a fix that needs an end-to-end engine run to verify, say the fix is ready and let the user run the profiling command; analyze the trace/log they point to. Kernel-level pytest runs on the TPU are fine to run directly. Related: [[decode-batched-mega-kernel]]
