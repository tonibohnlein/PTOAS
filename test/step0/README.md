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

The registered `pto-frontier-analysis` pass coordinates analysis and logical
insertion at module scope. Each function's `FrontierAnalysis` owns shared Step 0
records and structural recognition results. Closed delegation wrappers retain
their calls; their callees are analyzed independently with an explicit completion
interface. Physical event allocation remains a separate function pass.

`pto-sync-input-test --recognize input.pto` performs read-only recognition and
checks structure against the original MLIR. It emits recognition text and one
JSON object per function, plus `closed-callee-json` candidate reports where
applicable. Candidate recognition does not establish callee closure or allocation
success. The probe leaves IR unchanged and runs no demand backend.
`--gm-alias=may-alias` and `--gm-alias=may-not-alias` select the shared root policy;
called functions use may-alias so actual argument aliases are preserved.
The analysis pass is callable as `builtin.module(pto-frontier-analysis)`.

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

### Recognizer contracts and validation

| Recognizer | Current accepted representation | Limits kept explicit |
|---|---|---|
| Explicit | Adjacent original leaves with exact materialized physical ranges | No unresolved control, multiple phases or unrepresented value prerequisites |
| Finite guarded | Finite if/else tree with exact materialized ranges and guard DAG | Entry guard availability is reported separately; it is not an endpoint-code certificate |
| Rotating | Fixed loop body, disjoint known slot families, exact fixed within-slot byte unions and proved selectors | Constant nonnegative lower bound and positive step; no carried state or nested body |
| Guarded rotating | The rotating contract with invariant branch participation and parameter offsets | Conditions/offset parameters must be entry values or have pure speculatable entry recipes; varying guards require another route |
| Arithmetic bundle | Complete exact primitive relations in the configured difference, octagon or bounded-coefficient class | Every piece, parameter, residue and coefficient is checked; no backend is run |
| Arithmetic IR producer | Counted nests, exact shared maps with symbolic origins and fixed local byte unions | Constant nonnegative lower bounds, positive steps dividing the fixed period, proved affine upper bounds, period at most two, dimension at most eight, exactly represented branch predicates |

Rotating recognition now consumes Step 0's exact symbolic maps. It subtracts
the selected slot's shared physical origin and requires the remaining byte map
and extents to be independent of iteration/parameter symbols. The shared range
materializer then supplies the byte union, preserving fixed subview offsets
and gaps. All endpoints within a slot family define a common exact partition.
Records for the same payload, selector and atom are consolidated while retaining
every original effect ID and both read/write modes. The subsequent generator
bounds count these normalized fragments; atomization can increase their number. This adds no instruction-specific
footprint rules. `unsupported-slot-footprint` distinguishes a missing fixed-slot
representation from an inexact access declaration.

Loop normalization retains the original IR and expresses the slot pattern in
the iteration ordinal `j`, where `iv = lower + step*j`. Products are widened
before modular reduction. Selector arithmetic that lacks a proof under machine
semantics still reports `index-arithmetic`. Unsigned power-of-two remainders also
admit same-width wrapping add/subtract/constant-multiply expressions, because
machine wrap preserves those residues. Signed remainders and other moduli still
need the shared scalar analysis's nonnegative/no-wrap proof.

Guarded offsets retain their immutable SSA parameters and modular expression.
A pure speculatable definition recomputed inside the loop can be exported as an
operand-first entry recipe; the original IR is not changed, and actual entry
availability remains separately reported. This is not endpoint code.

The arithmetic producer retains original induction coordinates, substitutes
fixed residues before classifying linear constraints, and emits exact physical
byte relations from the same shared access maps. This is primitive extraction,
not general integer projection or minimum-demand reduction. Pure metadata is
classified through the shared leaf interface. Loop-carried scalar values are
accepted only when the shared scalar analysis proves their recurrence in loop
coordinates; opaque carried state and loop-result-dependent geometry remain
unmet obligations. No loop is unrolled to establish this contract.

Index comparisons and Boolean combinations qualify occurrence, access, order
and native-order primitives at each endpoint's own iteration coordinates.
Entry Boolean parameters carry their 0/1 domain. Shared Boolean expressions
are memoized; duplicate conjunction rows are removed. The producer accepts only
bounded explicit guard unions (64 expression levels, 4096 pieces and 4096 rows
per piece); exceeding a limit rejects the route without dropping alternatives.
Unsigned comparisons, data-dependent guards and opaque branch-result geometry
remain unsupported. Guard construction is charged separately from scanning the
resulting arithmetic rows.

All routes share the phase index's detection of payload-result prerequisites.
An unrepresented SSA dependency, including one used by control or yielded from
a region, cannot be silently discarded by an otherwise exact storage route.
Such cases report `additional-prerequisite`; this does not manufacture a storage
conflict or claim that the required prerequisite is impossible to support.

Regression checks cover positive and negative cases for every recognizer,
including symbolic rotation, views, invariant/varying guards, physical aliasing,
changing within-slot offsets and additional scalar prerequisites. Arithmetic
exports are evaluated against independently enumerated finite executions of
nested reset, triangular and sibling-loop examples, including zero and negative
trip bounds. Integer-row normalization is also checked against direct integer
evaluation. Structural verification continues to compare against original MLIR
and the Step 0 corpus baseline. Full-CLI and device validation are separate.

### Remaining draft contracts

The current recognition layer does not implement the counted-pattern
recognizers, supplied periodic-certificate interfaces or finite-overlay
contracts. Arithmetic production is whole-function with the guarded input
contract above; the supplied primitive-bundle checker has its own broader contract. Additional
forward prerequisites are detected but not extracted. These limitations remain
distinct from the generator/reducer and composition backends, which recognition
does not provide. No recognized input is reported as Section 8 Ready.

### Guarded leaves in compact class composition

Compact class composition accepts a loop-free conditional as a guarded leaf
through the existing finite-guarded scan and rank reducer. The leaf retains its
original occurrence identities, presence predicates, query owner, boundary
selectors and endpoint recipes. Empty and unequal arms do not require padding.
Repeating the enclosing body still requires the existing invariant-invocation
contract; this does not admit arbitrary iteration-dependent control or loops
inside the conditional.

Arithmetic diagnostics distinguish an unsupported condition from a supported
comparison whose operands cannot be represented without changing machine-integer
semantics. In particular, an unsigned-divided trip count does not establish
that a signed tail subtraction is non-wrapping. Such operands report
`index-arithmetic` at the defining operation.

`sync_compact_class_repetition.pto` checks the guarded boundary queries against
independent unfolded executions, varying branch choice independently of trip
counts. `sync_regional_arithmetic_context.pto` includes an unsigned tail reduced
from the prefill benchmark and verifies rejection of an unjustified affine
interpretation.

### Exact demands with unavailable exports

The logical dispatcher retains successful whole-function explicit, numerical,
sequence and arithmetic demand results independently of endpoint preparation.
It may try another exact route, but cannot replace known exact demands with
compact bounding demands merely because endpoint code is unavailable. That
case reports `unmet-exports`. A successful proper child does not block fallback
for its parent. Arithmetic results are cached with the shared input and cleared
when its GM alias policy changes.

`sync_compact_bounding_pipeline.pto` checks a guard computed after the source
cut: exact guarded reachability survives failed endpoint preparation, the IR
stays unchanged, and the dispatcher reports the missing export. The same test
also checks arithmetic cache reuse and alias-policy invalidation.

### Machine-integer boundary partitions

The shared scalar congruence normalizer preserves fixed-width addition,
subtraction and constant multiplication modulo the data-layout width. Boundary
composition uses its 64-bit index form to partition signed comparisons with a
constant into modular truth intervals. Counted-loop ordinals and entry values
remain symbolic; neither runtime iterations nor parameter valuations are
expanded. Zero trips and signed wrap are included. Comparisons without this
representation still report an adapter obligation.

This cut-list adapter supports at most 64 modular bands and checked i128
endpoint circuits. These are representation limits, not evidence that an input
is outside every arithmetic class. Cuts with the same affine numerator
coefficients and divisor retain their known order. Both boundary consumers use
one partition builder, specializing branch predicates on nonempty slices.

The compact-pipeline checks compare slice predicates against APInt evaluation,
including signed extrema, negative metadata lengths, nonunit loop steps and
large ordinals. The affine-boundary test compares inserted command closures
against independently unfolded accesses across signed wrap. These checks do
not establish complete logical-plan support for the prefill benchmark: its
remaining storage and endpoint exports are separate obligations.

Rotating physical origins accept factored constant scales while retaining the
modulo expression as an atom. This handles the prefill corpus's
`(bank*4 + offset)*256` addresses without adding instruction footprint rules.
The extended-rotation regression compares factored and distributed addresses
and requires them to merge into the same physical slot family.
