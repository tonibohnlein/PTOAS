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

The shared-range integration was checked on the existing 784-module prepared
corpus (851 functions): both GM policies passed existing InsertSync and the
Step 0 audit. All 37,524 access records were preserved: 1,970 exact, 27,660 with
conservative physical bounds, and 7,894 with unresolved access geometry. The two
GEMM fixtures retained 17/17 and 21/21 exact accesses. These are pass-level checks;
they do not establish generated C++ correctness or device behavior.

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

The existing `TEXTRACT` and transfer declarations continue through the same
resolver. Partial `TLOAD`/`TSTORE` refinement is deferred. A declaration of a
precise symbolic region does not imply that every compact recognizer accepts
its expression form.
