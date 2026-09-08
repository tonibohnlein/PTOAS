# Ascend Event Lab v0.2

**Native integration:** [NATIVE_BRIDGE.md](NATIVE_BRIDGE.md) documents the new
read-only pass-entry exporter and parity checker. The supplied evidence and
`STATUS.json` below remain historical standalone-model results.

A runnable synchronization-algorithm reference for an already scheduled, single
physical-context program. Inputs are explicit physical access/control facts;
outputs are symbolic handoffs and checked finite realizations. **This is not a
PTO parser, native InsertSync patch, or device simulator.**

v0.2 fixes the reproduced redundant-barrier, slot-key-candidate, byte-enumeration,
and unrolled-query failures. See [REVISION.md](REVISION.md) for changes, evidence,
and the remaining explicit unproved case. [ALGORITHM.md](ALGORITHM.md) describes
the algorithm. [INTEGRATION.md](INTEGRATION.md) keeps the native migration boundary.

## Run

Requires Python 3.10+ and a system libisl runtime exposing the APIs used by the
wrapper (tested here with Python 3.13.5 and isl-0.27-GMP). No pip package is needed.

```sh
python -m unittest discover -s tests -v
python run_campaign.py --output /tmp/eventlab-original
python run_revision_campaign.py --output /tmp/eventlab-v02

python -m eventlab plan examples/review_combined_gemm.json \
  --output /tmp/gemm-model-symbolic.json
python -m eventlab realize examples/review_combined_gemm.json \
  --params O=1 K=1 --pool 6 --verify \
  --output /tmp/gemm-model-finite.json
```

`--allocation finite` selects the bounded chain-cover oracle rather than the
symbolic key-function assignment. A finite pass does not prove arbitrary-trip
behavior. The symbolic rules are proved over exactly their declared context.
`LIMIT` and `UNPROVED_OR_INVALID` are never relabeled as verified plans.

The input JSON format is unchanged from v0.1. Read/write outputs in concrete
reports now use normalized half-open intervals, not lists of bytes. Mathematical
parameter/range conditions must be established from native IR or actual contracts
before being exported by a compiler. This prototype does not add runtime overflow
guards or infer unknown-root noalias.

## Files

- `eventlab/symbolic.py`: occurrence dependencies, readiness frontiers, exact mixed
  completion queries, and deletion-only barrier refinement.
- `eventlab/storage_keys.py`: candidate key functions derived from shared physical
  conflict write footprints. Every candidate still needs its full causal proof.
- `eventlab/reuse.py`: symbolic rearm proof and bounded candidate search.
- `eventlab/intervals.py`: exact compact interval sets for independent finite checks.
- `eventlab/physical.py`: issue/finish/submit/fire/consume model, bounded allocator,
  dense physical hazard checks, and exhaustive finite exploration.
- `eventlab/islwrap.py`: ownership-safe system libisl binding; exact relation queries.
- `tests/`: original tests plus review regressions and negative controls.
- `evidence/`: fresh v0.2 outputs, with explicit test scope and input provenance.

## Applying the experiment patch

The incremental patch supplied alongside this ZIP upgrades an exact v0.1 experiment
at `test/experiments/insert_sync/event_model/`:

```sh
git apply --check /path/to/ascend-event-lab-v0.1-to-v0.2.patch
git apply /path/to/ascend-event-lab-v0.1-to-v0.2.patch
```

The cumulative patch adds that experiment directory where it does not yet exist.
Use only one route. Neither patch modifies existing native C++, CMake, passes,
options, or native tests. Neither changes PTOAS synchronization when compiled.
No remote branch, commit, or push was performed for this delivery.

Keep the same short-lived child/worktree of the existing InsertSync branch. Do not
start a clean-main rewrite. Native fact export and synchronization emission remain
NOT_IMPLEMENTED; native and device experiments remain NOT_RUN.
