DeepSeek V4 decode PTO kernels for A5, generated from `hw-native-sys/pypto-lib` `models/deepseek/v4` at commit `be3c7942420b48fbab4ab1150edbc4ca8a125b94`.

Scope:
- compile-regression inputs for `ptoas`
- board-validation inputs for direct `.pto` kernels

Notes:
- This directory vendors the primary raw `.pto` fragments emitted from these source modules:
  - `decode_attention_csa.py`
  - `decode_attention_hca.py`
  - `decode_attention_swa.py`
  - `decode_csa.py`
  - `decode_hca.py`
  - `decode_sparse_attn.py`
  - `decode_swa.py`
- The `.pto` file contents are copied directly from PyPTO raw PTO backend output and are not hand-edited.
- `runop.sh` defaults these cases to `--pto-arch=a5 --pto-level=level3`.
- Board-validation uses custom `*_golden.py` references for the standalone rope-pack kernel and full-buffer sizing/default block args wired in `generate_testcase.py`.
