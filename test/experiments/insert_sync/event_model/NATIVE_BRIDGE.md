# Native handoff facts: first implementation commit

This directory retains Event Lab v0.2 as a reference. Its `evidence/` and
`STATUS.json` contain the supplied historical model results, not current native
compiler or device results. The native integration starts at compiler
`0387c6f17a4a13712a3d606349d0bdabc0c76e99`.

`--insert-sync-handoff-facts-dir=DIR` exports actual pass-entry facts before
lifecycle selection. The equivalent `pto-test-opt` pass option is
`--pto-insert-sync="handoff-facts-dir=DIR ..."`. Export inspects a clone and
does not change ordinary or revised synchronization. Unsupported analysis still
produces a status record. Files are written atomically, named by symbol, input
and module-context digests. The MD5 values are identity checks, not security
certificates. Directory/write failures are reported as compilation errors.

The schema `ptoas.handoff-facts.v1` contains original PTO, target/module context,
deterministic physical phase identities, native access records, retained typed
local obligations and a qualified reference `model`. No input annotation is
required or trusted. Loops use original induction variables, not first/middle/
last graph nodes. The initial expression subset admits signed unit-step loops,
constant/invariant bounds, supported comparisons/boolean choices and qualified
slot residues. Incomplete translator effects and helper calls without imported symbol
contracts reject the projection. Unsupported or wrapping arithmetic, unsigned loops, hidden
physical phases and unavailable local addresses prevent the projection.

The reference model is explicitly a **conservative local ordering projection**:

- Local intervals retain native allocation bounds. Every exported write has
  `definite=false`; a full descriptor alone is not definite byte production.
- Native descriptor qualification is retained separately for inspection.
- GM effects remain visible as omitted native requirements. No artificial GM
  addresses or independent address spaces are invented.
- ACC resource ordering remains a typed native obligation outside the ordinary
  reference memory relations. MMAD is not converted to full M completion.
- A reference result is neither full native synchronization proof nor an
  authorization to remove residual or external-resource synchronization.

With the matching built Python package on `PYTHONPATH`, run:

```sh
python check_native_facts.py /path/to/capture.json \
  --arguments '[0,16,-1]' --output /path/to/parity.json
```

Use arguments for that captured function; the example is the lit loop fixture.
Integer parameters use signed bit-pattern interpretation (`i1` true is `-1`).
The checker independently replays captured scalar control, compares every
physical occurrence/lane and translated footprint, checks dense conflicts and
retained native memory obligations. It explicitly counts resource obligations
outside the reference. These are finite interchange checks, not symbolic
native verification or device tests.

The reference can use a specifically selected shared library via
`PTOAS_EVENT_MODEL_ISL_LIBRARY=/absolute/path/to/libisl.so`. No automatic package
installation is performed. The local runtime is isl 0.24; the supplied evidence
used 0.27. Literal `0 % 2` and `1 % 2` expressions in the unrolled example are
constant-folded in this integration to preserve their values with the older
parser. The environment selector and this input spelling change are recorded
integration differences; the original bundle remains unchanged. Use
`python preflight_native.py` to test representative parser/API features.

The second commit must make native handoff decisions and realize complete
protocols. This first bridge deliberately does not claim an optimization.

## Recorded native acceptance

[NATIVE_BRIDGE_RESULTS.json](NATIVE_BRIDGE_RESULTS.json) records the final
pre-commit worktree experiment: 8 unchanged kernels, 16 compiler invocations,
identical export-off/on output, and 24 supported finite occurrence checks at
bounds 0/1/2/16. Historical GEMM and Q projection retain explicit unsupported
bound/step status. The native lit check also covers full, partial and unknown
descriptors; the parity test detects six deliberately corrupted captures.
The 60 reference unit tests pass. The supplied revision campaign passes while
retaining its documented unproved nested-three-slot-scaled result.

Use the matching built PTOAS Python interpreter/package for
`capture_native_examples.py --python-root BUILD/python --output FRESH_DIRECTORY`.
It runs serially and records source hashes, commands, compiler fingerprint and
per-invocation elapsed time. `test_native_parity.py CAPTURE` runs the corruption
tests against the native lit loop capture. No device results are claimed.

`UPSTREAM_SOURCE_MANIFEST.json` preserves the supplied manifest.
`SOURCE_MANIFEST.json` hashes the integrated tree, excluding itself and caches.
Imported reference sources retain their MIT notices and this directory's MIT
license. Repository CANN contribution headers have been added as required by
the repository policy; they do not remove the original MIT attribution.
Native bridge files are repository additions under the root CANN license.
