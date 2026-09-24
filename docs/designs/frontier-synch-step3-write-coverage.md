# Phase A, step 3: qualified full-cell writes

> **Integration update, 2026-09-24:** this increment has passed local native
> validation and independent review at its stated scope. See the
> [integrated review](frontier-synch-steps2-6-review.md) for fixes and evidence.
> The candidate/pending statements below record the original patch submission,
> not the current integration status. Later Phase A gates remain open.

Base: `0a38c8e9f4bd4852177c1dd6f6e0d0ee26b6ca16` on
`codex/handoff-foundation`, including the accepted factored core at `9bba65520`.
Draft: v0.44, Sections 2.1, 3.2, 3.4, 5.2 and Appendix I.1. Its operative
Phase A contract is unchanged from v0.43.

**Status: bounded implementation candidate; native build and independent
acceptance remain pending.** This is the coverage prerequisite, not the later
conditional-provenance integration, occurrence decoder or complete Phase A.
The older inventory's statement that *all* imported writes lack coverage is
superseded for the qualified cases below. Its broader parity gaps remain.

## Semantic audit and interface

The existing `MemoryEffectOpInterface` identifies may-reads and may-writes of
values. `PTOIRTranslator::UpdatePTOOpInfoWithPipeline` turns those into common
`useVec`/`defVec` entries. `BaseMemInfo::allocateSize` is a bounding allocation
or segment size, not a statement that an instruction overwrites all those bytes.
The existing importer consequently sets every imported `definiteWrite` false.
The existing marginal and factored transfers already have strong-update consumers.

The new shared `TileWriteCoverageOpInterface::writesAllValidTileElements(value)`
is an additive, per-written-value semantic refinement. True guarantees that
this whole operation writes each current valid logical element of that tile.
False withholds that guarantee; it does not mean no write, no old-content read,
unsupported instruction, or failure of synchronization insertion. The per-value
query can later qualify multiple destinations without changing the interface.

| Instruction / variant | Existing effects | New guaranteed extent |
| --- | --- | --- |
| Ordinary `tload` | source read; destination write | Every valid destination element; no guarantee of an inactive row or padding byte |
| `tadd` | both inputs read; destination write | Every valid destination element; aliased input/output still contributes both modes |
| `taxpy` | source/scalar/destination read; destination write | Every valid destination element; destination read remains authoritative |
| TLOAD with extra value operands, explicit padding mode, or enabled output initialization | Unchanged | Not qualified by this initial semantic rule |
| Other writers, including store, extraction/insertion, scatter/gather, reductions/scratch, matrix/macro operations and other elementwise operations | Unchanged | No new semantic certificate in this increment |

A default or explicit `init_out_buffer=false` is not a reason to reject the
ordinary TLOAD refinement. Attribute presence alone would make the result
sensitive to how TableGen materializes defaults. Extra value operands are
conservatively declined even when a particular zero-valued operand might be
proved harmless. Those overloads need their own positive coverage rules.

This is an explicit initial semantic fragment, **not an audit claiming full
coverage semantics for every PTO instruction**. The unannotated operations
still enter both modes through the same effects and translator. No opcode
switch or instruction admission whitelist is added to Phase A.

### Source basis

The pinned shared effect definitions are in:

- `lib/PTO/IR/PTOInterfaces/PTOSubViewAndCoreEffects.cpp`: TLOAD and the
  common read/write effect helpers. These helpers do not establish full-cell
  overwrite coverage.
- `lib/PTO/IR/PTOInterfaces/PTOMemoryAndElementwiseEffects.cpp`: the TADD
  binary effects and explicit TAXPY destination read/write. This patch does
  not change either implementation's R/W effects.
- `lib/PTO/Transforms/InsertSync/PTOIRTranslator.cpp`: allocation/segment
  footprints, conservative alias forwarding, dynamic selection and subview
  parent-relative addresses.
- `include/PTO/Transforms/FrontierSynch/SyncTileDescriptorState.h`: effective
  valid dimensions at each original use, including metadata updates and joins.

The valid-element semantics were checked against the PTO-ISA instruction
references (accessed 2026-09-24):

- https://pto-isa.gitcode.com/docs/isa/TLOAD/
- https://pto-isa.gitcode.com/docs/isa/TADD/
- https://pto-isa.gitcode.com/docs/isa/TAXPY/

In particular, TLOAD padding is not a promise to overwrite a complete bounding
allocation. The proof below uses only the guaranteed valid elements, even when
an implementation also writes some padding. These references are semantic
premises, not evidence of native compilation or device validation of this patch.

## Physical qualification

`OriginalWriteCoverage` combines the shared guarantee with the imported
geometry and the descriptor state **at this original operation**. It requires:

1. A whole original instruction, not one of its internal translated phases.
2. One translated destination location with no conservative alias ambiguity.
3. Known current valid dimensions within the allocated tile shape.
4. A supported byte-addressable, rank-two, NoneBox layout and checked strides.
5. An original constant local allocation address, requalified through the
   checked scalar evaluator, with matching root, address and translated size.
6. For a strong update, a canonical exact cell wholly contained in the
   guaranteed write pattern.

The byte pattern is `(begin, blocks, blockBytes, strideBytes)`. Its construction
and membership tests check addition/multiplication overflow and represent holes
without enumerating rows or elements. Row-major and column-major dimensions
are mapped to the corresponding major/minor coordinates. RowPlusOne physical
padding contributes to stride but not to guaranteed written bytes. Packed,
fractal, opaque and unqualified layouts withhold coverage.

A one-level static subview spanning one major-axis segment is qualified using
its **parent's** stride and checked offsets. Its translated segment size and
address must agree. Nested, multi-segment and dynamic subviews remain
conservative in this increment. They have not been relabelled as a draft open
problem; extending their positive geometry rules is further implementation
work. Dynamic address/bank selectors and selected/loop-carried buffer aliases
likewise retain may-footprints without gaining a must-write certificate.
Constant multi-tile selection, symbolic addresses and GM write coverage are
not added by this initial adapter.

The canonical partition itself is unchanged. A partial write can strongly
update an already represented exact subcell that it completely covers, but
cannot kill provenance for a larger cell crossing unwritten bytes. The patch
does not expand every valid row into a new physical cell. An overlap witness
is never made definite, even when its contributing write has a valid-element
certificate. A bounding interval alone never sets `definiteWrite`.

The immutable `WriteCoverage` record accompanies every resulting write
incidence, together with its original translated `memory` and address relation.
It distinguishes missing semantics, descriptor, layout, location and geometry
premises. `Qualified` does not necessarily imply a full-cell write: a qualified
empty/partial pattern can cover no whole cell. Neither status supplies completion.

This refinement does not certify the entire legacy may-alias implementation.
The draft's complete/conservative shared-effect and may-footprint contract
remains a prerequisite. Rechecking scalar arithmetic for a must proof must not
be described as a general repair of all shared address extraction.

## Provenance consumer and RMW correction

`appendCanonicalStorage` is the only point that promotes a guaranteed byte
pattern to `Access::definiteWrite`. `OriginalLifetimes` then uses the existing
strong-update transfer; its requirements come from the input state at the write,
not from its replacement output state. Earlier demands and source subscriptions
are not deleted.

Enabling full RMW writes also exposes a marginal-transfer detail: after clearing
the old state, a definite RMW should become the new writer, not additionally a
*pure reader of the old generation*. The patch restricts reader insertion to
reads without a definite overwrite. Possible/partial RMW still adds both modes
and preserves earlier writers/readers. Forward transfer suppresses that post-write reader; backward transfer retains
the RMW read because it precedes the overwrite. The native next-reader assertion
covers this direction-specific distinction. The factored core already has the old-state-before-kill
rule; its later conditional/control integration is not modified here.

## Tests and evidence

### Executed here

`test/frontier_synch/run_write_coverage_standalone.sh` compiles the production
`WriteCoverage.h` and `StorageWitnesses.h` with a minimal non-MLIR data-record
adapter. It compares the byte coverage query against independent explicit byte
sets for small rectangles, empty extents, holes and cells crossing gaps. It
also checks overflow, enormous compact domains, per-cell partial coverage,
RMW mode preservation and non-definite conservative alias witnesses.

Both executions completed with **1,383,576 checks passed**:

```sh
CXX=clang++ EXTRA_CXXFLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  bash test/frontier_synch/run_write_coverage_standalone.sh
CXX=g++ bash test/frontier_synch/run_write_coverage_standalone.sh
```

Compilation used C++17 with `-Wall -Wextra -Werror`. The Clang execution used
AddressSanitizer and UndefinedBehaviorSanitizer. This is execution of the
coverage/partition core, **not an MLIR build, native importer test or pass run**.

The original complete `StorageWitnesses.h` used by this harness was verified
against pinned Git blob `04a07846b7dba899cc85143954d5618ec770658e` before applying
the changes. The patch is assembled from pinned source ranges for the other
modified files; this environment did not have a complete checkout. Patch syntax
was parsed with `git apply --numstat`; a clean-application check in the complete
repository remains part of the native gate.

### Added, but not executed here

`test/lit/pto/frontier_synch_write_coverage.pto` goes through `SyncInput`,
`importOriginalStructure` and public `ProgramAnalysis` using real PTO. Its
`test.strong`, `test.coverage`, `test.raw`, `test.war` and `test.waw` assertions
are checked in the existing analysis runner. They test:

- Real full writes and full RMW, including pre-update RAW/WAW/WAR requirements
  and elimination of replaced origins only at subsequent consumers.
- Partial writes and partial RMW preserving older writers and readers.
- Known, changed, unknown and empty valid dimensions at distinct original uses.
- A proved single-row subview overwrite and an unqualified strided subview;
  untouched parent bytes retain their earlier writer.
- An uncertain selected address; an unannotated writer still producing real
  requirements; masked padding; a TLOAD overload with no semantic certificate.

The runner retains its original-IR equality check. Its existing TAXPY and
factored-provenance tests remain present. The new assertions reject an importer
that returns all false as well as one that blindly treats all allocations as
full writes.

`test/lit/pto/frontier_synch_write_coverage_modes.pto` exercises both actual
pass modes. Existing mode must emit flag synchronization. FrontierSynch must
reach its explicit construction-not-implemented diagnostic; that diagnostic is
not counted as successful synthesis. This is a smoke test, not a claim of
unchanged output on every existing-mode input.

Run in a configured checkout:

```sh
git apply --check phase-a-step3-write-coverage.patch
git apply phase-a-step3-write-coverage.patch
cmake --build build --target pto-test-opt pto-frontier-analysis-test --parallel 2
lit -j 1 -v build/test/lit --filter 'frontier_synch_write_coverage|frontier_synch_taxpy_effects|frontier_synch/'
build/tools/pto-test-opt/pto-frontier-analysis-test --factored-self-test
```

Then run the full existing synchronization regression suite. No native tests,
five-kernel rerun, full compiler build, numerical/device checks or performance
measurements were executed here: the environment has no LLVM/MLIR development
dependencies, and the repository could only be read through the connector.

## Review record and remaining acceptance gates

The implementation was self-audited for physical-domain identity, whole-phase
qualification, partial/empty coverage, default attributes, overflow, subview
stride, witness lifetime and old-state RMW demands. Corrections included
preserving false-valued TLOAD defaults, suppressing the post-strong-update RMW
pure-reader origin, and retaining aggregate-initializer compatibility for the
new evidence fields. This is not an independent reviewer decision.

Before accepting this increment: regenerate TableGen, compile the native code,
run the focused and existing-mode regressions, inspect public results on the
five kernels, and obtain the independent review required by the implementation
plan. Broader per-instruction/geometry coverage must receive its own explicit
rules and tests. Step 4's original-use integration and the later D1–D4 and
preparation gates are still required; passing these coverage tests would not
establish complete Phase A parity.
