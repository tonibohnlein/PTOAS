# Pinned PyPTO-lib survey inputs

These are physical-storage PTO immediately before `pto-insert-sync`, with an
explicit A3 module wrapper. Copyright/provenance comments are the only fixture
changes from the raw snapshots. Sources and commands are pinned in
`docs/designs/frontier-pypto-region-survey-20261019.json`.

Frontend: PyPTO `fb107da927f40a65e0e3ca7198e47f40d9e075d5`;
pypto-lib `f3167ecbf8838ecff0e169ee8d854baafaa0236c`.
Native lowering: `e54d19e91cdf7c54a6b6738f0f0bd1b67cdb9f44` plus the
manifested workspace changes. Target A3, PYPTO memory planner; native memory
replanning disabled to retain the frontend physical addresses.

- `qwen-down`: Qwen3-14B `prefill_fwd.py::down_core`, nested stage-2 K pipeline,
  HIDDEN=5120, MLP=17408, band=4352, M=128, K chunk=256, L0 chunk=64.
- `qwen-finalize`: same source, `attention_finalize_phase`, work stride=48,
  dynamic final-token guard, vector divide/cast/store.
- `deepseek-rms`: DeepSeek V4 Pro `rmsnorm.py::rms_norm`, D=7168, token tile=8,
  reduce/apply tile=1792, two tiles per lowered paired reduction visit.

Scalar bindings, bounded reduction prefixes and child-context extraction are
explicit in `survey_pypto_regions.py`. They are finite validation instances,
not production simplifications. The independent interpreter consumes original
views/types/access operands; scratch full-tile envelopes and matrix accumulator
protection remain supplied model premises, not device-qualified footprints.
Different GM arguments are disjoint only under the named `may-not-alias`
assumption. No indirect addressing is admitted by this interpreter.

Unavailable GM selectors are expected limitations, never empty answers.
The survey test tool refuses finite-cell fallback on an unrepresented GM domain.
The saved false-empty probe reproducer is an interface/diagnostic finding, not a
native miscompilation claim. No production analysis/allocation is modified.
