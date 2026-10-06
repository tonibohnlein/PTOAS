# Automatic versus hand-authored synchronization

Compare the **expert**, **frontier**, and **existing InsertSync** plans with the
same computation, physical storage, and payload schedule. Record correctness,
device time, compiler time, synchronization operations, and event IDs. More
SET/WAIT operations alone do not establish a worse plan.

`benchmarks.json` is the registry. It distinguishes prepared PTO inputs from
source candidates and exclusions. A source containing manual flags is evidence
of an authored protocol, not evidence that it is faster or device-qualified.
This collection is an opt-in benchmark, not part of lit or pytest discovery.

## Cases and readiness

The two earlier GEMMs and **nine additional kernel families** now have prepared
PTO IR pairs. Every pair is an explicit benchmark port with recorded source
pins and configuration. See [EXPORT_STATUS.md](EXPORT_STATUS.md) for the
per-kernel table and [AUDIT.md](AUDIT.md) for translation decisions.

Baseline C++ generation (before the subsequent analysis changes): **24/33 variants succeed**. Expert and existing InsertSync
succeed for all11 cases. Frontier succeeds for the two earlier GEMMs and rejects
all9 new ports during analysis, before allocation. No new-port device timing or
correctness result is claimed. CANN compilation and launch wrappers remain to
be supplied on the device machine.

The original2 GEMM entries specialize their protocols to M256,K4096,N256 on
one cube block, with the corrected NZ L1 layout; they were tested in an earlier
pinned device bundle. The new persistent GEMM uses the full default dimensions
and persistent mapping. GroupNorm is serial-only, mHC is forward-only, and the
lossless cast port uses the generic64×64 scale-block configuration, not max4.

RMSNorm and online softmax remain source-only automatic-sync comparisons.
FlashAttention, sparse FlashAttention and MoE permutation remain deferred
because of cross-core communication. The existing235-module device task and
its denominator are unchanged.

TileLang's `target="pto"` emits PTO-ISA C++, not PTOAS IR. These ports are
benchmark translations; they are not a reusable frontend converter. All three
plans use identical ported payload/control/storage. Native upstream performance
must be reported separately until backend/port differences are qualified.

## Sources

New TileLang examples are pinned to
[`cbc67f2697e8ee18357adc447272691432b3948d`](https://github.com/tile-ai/tilelang-ascend/tree/cbc67f2697e8ee18357adc447272691432b3948d/examples).
Unmodified source snapshots under `sources/` retain their byte hashes and
upstream MIT license. The `.txt` suffix prevents importing or collecting these
archived examples as local tests. Use a full checkout at the pinned revision to
execute upstream examples, including their helper imports and dependencies.

The older GEMM source revisions are pinned separately in the manifest, preserving
the sources used in the existing comparison. Only their prepared PTO ports are
included here. The new registry does not alter the already-pinned 235-module
device task or add cases silently to its denominator.

## Prepare the currently available comparisons

Verify all local hashes, parse archived Python without executing it, and list
every entry with its status:

```sh
python3 test/npu_validation/benchmarks/auto_sync/prepare.py
```

Generate fresh C++ for all prepared cases with all three plans:

```sh
python3 test/npu_validation/benchmarks/auto_sync/prepare.py \
  --pto-test-opt .local/modeled-access/pto-test-opt \
  --ptoas .local/shared-alias/ptoas \
  --output .local/expert-sync-comparison
```

Use binaries built from the revision being tested. The script records binary
hashes, the manifest hash, commands, logs, pending cases, and one-shot wall times
in `generation.json`. It exits unsuccessfully on compilation/allocation failure.
It does not repair ID scarcity. The output directory must be new. This is C++
generation; a CANN device build and launcher are still required.

To check deterministic regeneration of all nine additional ports:

```sh
python3 test/npu_validation/benchmarks/auto_sync/export_ports.py \
  --pto-test-opt .local/modeled-access/pto-test-opt
```

Use `--install` only when intentionally updating their IR and registry hashes.

## Turn a pinned candidate into a comparison

1. Export the original kernel with hand-written synchronization preserved and
   additional upstream automatic synchronization disabled. Preserve its schedule,
   layouts, views, buffer addresses, masks, and loop/control structure.
2. Obtain the corresponding synchronization-free planned PTO input. Remove only
   the protocol being replaced; do not erase cross-core operations or invocation
   requirements. Audit the two representations after removing intra-core
   synchronization and normalizing SSA names and nonsemantic provenance. A
   different tiling or memory plan is a separate comparison.
3. Feed this **same** planned input to frontier and existing InsertSync. Compile
   all three plans with the same PTOAS/PTO-ISA/CANN versions and automatic sync
   and memory planning disabled in the final backend. Keep native upstream
   performance as a separately labeled measurement if its backend differs.
4. Add the two audited PTO files and hashes to the entry's `prepared` fields,
   then change its status to `prepared_pto`. Keep the source snapshot and record
   every specialization or compatibility edit. Add the launch ABI, shapes,
   independent golden, tolerances, and mutated-state reset to the device task.
   Registering a source alone does not make it launchable.

## Device measurements

- Build each variant separately. Record PTO compilation and CANN compilation
  separately; for comparable compile times use five forced rebuilds with the
  same cache policy. The preparer's single observations are diagnostic only.
- Check each variant against the independent mathematical reference, including
  mutated state and auxiliary outputs. Agreement with the expert alone is
  insufficient. Preserve upstream tolerances and justify any changes.
- Reset mutable state outside timed regions. Use identical data and launch
  geometry, warm up each variant, and collect at least 30 interleaved samples
  after 10 warmups. Report median and dispersion; keep failures and exclusions.
- Record target device, CANN/PTO-ISA versions, source/compiler/input hashes,
  compile logs, generated plans and C++, launch configuration, correctness, and
  runtimes. Report frontier/expert and frontier/existing ratios per case.
- Existing eligibility is IDs 0..5. Some upstream examples use ID 6; qualify
  that API's hardware mapping before claiming an equivalent resource contract.
  Do not silently renumber the authored protocol or add scarcity repair.

Cross-core candidates remain excluded until that scope is explicitly opened.
