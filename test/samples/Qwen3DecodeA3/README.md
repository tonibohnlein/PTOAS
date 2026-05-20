Qwen3 decode PTO kernels for A3, generated from `pypto-lib/models/qwen3/32b/qwen3_32b_decode.py` at `05127d2`.

Scope:
- compile-regression inputs for `ptoas`
- board-validation inputs with per-case custom golden

Notes:
- This directory vendors the 14 raw `.pto` fragments emitted by the latest PyPTO PTO backend for the A3 lowering.
- The upstream kernel topology changed from the old 17-case `qwen3_decode_incore_*` layout to a mixed set including `rmsnorm`, `rope_kv_cache`, `out_proj_residual`, `post_rmsnorm`, and `down_proj_residual`.
- `runop.sh` defaults these cases to `--pto-level=level3`.
- `runop.sh` skips this directory on A5 / Ascend950 targets.
- Each current fragment has a sibling `<case>_golden.py`; shared reference logic lives in `qwen3_decode_golden_lib.py`.
