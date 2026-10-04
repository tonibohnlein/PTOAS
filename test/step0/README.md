# Step 0: shared synchronization input

Step 0 collects the information used by the analysis routes. It does not select
an exact route, compute demands, insert synchronization, or allocate event IDs.

Its reusable components are `SyncInput`, `PhaseIndex`, and `SyncStorageEffects`:

- Reuse InsertSync's translator, pipe assignments and buffer records.
- Preserve every shared phase read/write record and every operation-interface
  effect declaration, including declarations without a translated pipe phase.
- Preserve source operation anchors, original regions, loop bounds and branch
  conditions. Unsupported expressions retain their SSA identity.
- Retain planned addresses and view ranges. Reuse the existing alias/range query
  for possible conflicts, with the selected GM alias policy.
- Attach symbolic descriptor maps and slot selectors where available. A missing
  map is an unresolved description, not an empty access or an extraction failure.

The read-only `pto-sync-input-test --step0-json INPUT` audit checks declarations
against the original MLIR interfaces, records against the shared phase vectors,
loop domains, SSA availability and unchanged source IR. It reports declarations
without a pipe phase separately. These include descriptor metadata and existing
pipe/communication operations; preserving them does not assign new semantics to
them. No cross-core synchronization analysis is added.

`--existing-check INPUT` independently runs the existing InsertSync pass on a
prepared module. This verifies pass acceptance at the same pipeline boundary;
it is not a C++ code-generation or device check.

## Corpus runner

Run serially with the selected workspace Python and freshly built test tool:

```sh
.venv/bin/python test/step0/run_corpus.py \
  --tool /path/to/pto-sync-input-test \
  --manifest /path/to/manifest.json \
  --output /path/to/new-results-directory
```

The manifest contains `cases`, each with `id`, `family`, `input` (prepared IR),
`sha256` and the expected number of `functions`. Optional `source` and
`source_sha256` bind the snapshot to the original frontend input. Record snapshot
preparation provenance alongside the manifest. Paths are local files selected by
the operator, not remote downloads or shell fragments.

For each input, the runner checks hashes, runs existing InsertSync and audits
Step 0 with both GM alias policies. It requires identical collected information
under both policies and preserves commands, logs, timing, hashes and failures.
The output directory must be new. The tool disables MLIR multithreading; the
runner launches one child at a time and kills its process group on timeout.

## Claim supported by this test

Success establishes preservation of the shared compiler input for the tested
snapshots. It does not establish completeness of native instruction effect
semantics, exact accessed-byte sets, a compact normal form for every expression,
or acceptance by the tractable-case recognizers. Those stronger requirements
remain explicit inputs to later analysis routes. In particular, buffer-range
information never establishes a complete overwrite by itself.

## Shared resolved accesses

`SyncInput::accesses()` owns the resolved records used by existing InsertSync
and the demand-analysis clients. Buffer records remain allocation/view geometry;
instruction selections refine individual reads and writes without modifying those
buffer records. Multiple exact declarations denote their union.

A known enclosing physical range is retained as `UpperBound` when no exact
selection is available. `Unknown` denotes unavailable geometry, not an empty
access. Neither permits definite-overwrite reasoning. Exact symbolic maps remain
exact descriptions even when overlap cannot be decided without iteration context.

Existing InsertSync obtains dependency buffer pairs from the shared access query.
Its hazard rules, insertion, motion and event-ID allocation are unchanged. The
`--existing-dump` test option prints the resulting IR for insertion regressions.

The shared-range refinement was checked on the existing 784-module prepared
corpus (851 functions): both GM policies passed existing InsertSync and the
Step 0 audit, with no failures. All 37,524 access records were preserved.

| Resolved precision | Before refinement | After refinement |
|---|---:|---:|
| Exact | 1,970 | 5,008 |
| Conservative physical bound | 27,660 | 27,579 |
| Unresolved access geometry | 7,894 | 4,937 |

The same prepared-input manifest was used. Inputs and the test binary were
hash-checked throughout the run. The two GEMM fixtures retained 17/17 and 21/21
exact accesses. The focused diagnostic suite passed 61 RUN checks; six full-CLI
checks were skipped. These are pass-level checks; they do not establish generated
C++ correctness, device behavior, or exact effects for the remaining accesses.

### Explicit access domains

Shared `MemoryEffectOpInterface` parameters now also accept version-2
`pto.access_region` selections. Version 1 remains unchanged. Version 2 has
`extents` (an affine map with no dimensions), `coordinates`, `symbol_operands`,
`addressing` (`coordinates` or `bytes`) and `byte_width`, in addition to its
version field. Both maps use the same scalar operand references. The number of
selection dimensions equals the number of extent results.

An empty extent list denotes a singleton access; an extent of zero denotes an
empty access. Logical coordinates compose with the operand's layout. Byte
selections have one result and a positive width, and are relative to the
operand's physical base, including existing view offsets exactly once. Logical
selections have zero byte width because the descriptor supplies element size.

The domain and selection are an exact declaration by the producer. Runtime
extents must be nonnegative on valid executions. Invalid parameters or
unresolved geometry retain conservative effects. Packed sub-byte writes are
not promoted to byte-wide definite overwrites. Exact symbolic maps are retained
when concrete interval materialization is unavailable. No serialized operation
syntax or consumer instruction whitelist is added.

### Instruction declarations using the shared interface

The following refinements are produced by the instructions' existing
`MemoryEffectOpInterface`; neither synchronization consumer has an opcode table:

- `TGETVAL`/`TSETVAL`: one scalar in VEC storage, at the operand's physical base
  plus the element offset narrowed to unsigned 32 bits. Supported views contribute
  their offset once. Dynamic offsets remain symbolic.
- `pto.load`/`pto.store`: one scalar or a one-dimensional, fixed power-of-two
  vector, relative to a typed pointer. Offsets and `addptr` strides count complete
  pointee objects. Odd-length vectors, scalable vectors and packed element types
  retain the existing conservative declaration.
- Ordinary aligned VEC-to-VEC `TINSERT`: reads the source valid rectangle and
  writes that rectangle at the destination row/column offsets, narrowed to
  unsigned 16 bits. The initial exact path requires known aligned allocation
  bases, row-major noncompact layouts, matching supported element types and
  native transfer lengths/gaps that fit. Other layouts, conversion forms,
  unaligned transfers and unavailable geometry retain conservative ranges.

The scalar tile offset follows `Tile::GetValue`/`SetValue` in the PTO-ISA tile
interface. The insert rectangle and block bounds follow the aligned ND paths
in `common/arch/memory/tinsert_common.hpp` and `npu/a5/TInsert.hpp`. Pointer
selections follow complete-object indexing in the EmitC and LLVM lowerings;
non-power-of-two vectors are excluded because allocation stride can exceed
stored width. These declarations describe valid executions; they do not add
runtime bounds checks.

The shared A3 `TEXTRACT` alignment check also uses `scf.for` induction values:
an aligned lower bound and aligned step imply that every executed offset is
aligned. This supports nested loops and dynamic upper bounds; unknown lower
bounds, unknown steps and unrelated loop-carried arguments remain conservative.
The selected region still uses the destination dimensions and source layout,
including separated physical strips for blocked matrix layouts.

After this alignment refinement, all 784 prepared corpus modules still pass
existing InsertSync under both GM alias policies and the Step 0 audit. Of the
same 37,524 access records, 5,772 are exact, 26,815 retain conservative physical
bounds and 4,937 have unresolved geometry. The focused suite now passes 63 RUN
checks, with the same six full-CLI checks skipped. In the nested `down_proj`
kernel, all 12 TEXTRACT operations now have exact source and destination
selections; other conservative or unresolved accesses remain explicit.

The existing `TEXTRACT` and transfer declarations continue through the same
resolver. Partial `TLOAD`/`TSTORE` refinement is deferred. A declaration of a
precise symbolic region does not imply that every compact recognizer accepts
its expression form.

## Program structure and recognition

The registered `pto-frontier-analysis` pass builds an owning MLIR
`FrontierAnalysis`: shared Step 0 input followed by the structure and recognition
results. The analysis owns the shared records for as long as the results borrow
them; MLIR invalidates it after IR-changing passes. The current arithmetic class
is fixed to 8 pipes, 8 coordinates, period 2 and coefficient bound 8.

`pto-sync-input-test --recognize input.pto` runs this real pass followed by a
test consumer of its cached result. The consumer verifies structure against the
original MLIR and emits the existing recognition text plus one JSON object per
function. It also checks that the pass leaves the IR unchanged. There is no
separate program-recognition mode. This is the analysis stage under development;
it does not yet generate demands or insert synchronization.
`--gm-alias=may-alias` and `--gm-alias=may-not-alias` select the existing shared
policy. The pass is callable in a standard MLIR pipeline as
`builtin.module(func.func(pto-frontier-analysis))`.

The reusable `frontiersynch::recognizeProgram` API follows the original MLIR
region structure without unrolling. Sequences contain maximal adjacent leaf
runs, counted loops, and conditionals. Each conditional retains separate then
and else sequences, including an empty else arm. Unsupported structured control
is preserved and marked; its descendants are not silently accepted as an
independent executable plan. Payload-bearing anchors link to their original
shared phase, pipe and access-record IDs exactly once.

Loop nodes retain the actual bounds, step and induction value. Descendants
retain outer-to-inner loop coordinates. A shared guard chain records each branch
choice, SSA availability before the branch, and availability before enclosing
loops. These are availability observations, not a proof that every future
synchronization endpoint can evaluate its guard. Empty-trip behavior stays in
the loop bounds; no dynamic occurrences are instantiated.

The report tries explicit recognition on leaf runs, finite-guarded recognition
on sequences, and rotating/guarded-rotating recognition on counted loops. It
also runs the existing arithmetic producer once on the whole function; that
producer does not yet offer a per-subregion adapter. Attempts preserve their
`applicable`, `not-applicable` or `missing-premise` status and diagnostics, with
operation names and source locations where the recognizer supplies an anchor.
All candidate results remain separate: acceptance of an inner loop does not
certify composition through the outer loop. `analysis_ready` is therefore
false; generator extraction, reduction and endpoint construction are later work.

Node, payload and effect IDs are references, not dynamic occurrence ranks.
Sequence child lists preserve source order; payload order within a leaf run
preserves source order. Reports are deterministic for unchanged IR. The
structural walk is linear in the IR plus the stored context/record links;
recognizer work is additional (including repeated subtree checks and arithmetic
primitive construction), so the complete report has no blanket linear bound.

To compare recognition with a completed Step 0 corpus run:

```sh
python test/step0/run_recognition.py \
  --tool /path/to/pto-sync-input-test \
  --manifest /path/to/manifest.json \
  --baseline /path/to/step0-results \
  --output /path/to/new-recognition-results
```

The runner is serial, checks input and executable hashes, and verifies that each
function retains all shared phases and effect IDs exactly once. Its summary
counts candidate attempts only on regions containing payloads; overlapping
candidate regions are not independent kernels and their counts must not be
interpreted as whole-program acceptance. The per-function JSON includes access
precision, descriptor/access-map presence and concrete-range availability to
help distinguish missing effects from limitations of a recognizer's adapter.

The pass-based structural audit preserved all 18,100 shared phases and 37,524
effects across 784 modules (851 functions). The independent checker compared
each result with the original MLIR, including source order, parentage, loop and
branch ancestry, empty arms and shared-record ownership. The focused suite
passed 67 RUN checks, with six full-CLI checks skipped; a separate may-alias
invocation also passed. This establishes structural correctness on these inputs
and input preservation, not broad exact-route coverage: no payload-bearing rotating
candidate was accepted, and both arithmetic-accepted corpus functions had no
shared payload phases. The reports expose the remaining access-precision,
view-adapter, loop-normalization and control-support gaps.
