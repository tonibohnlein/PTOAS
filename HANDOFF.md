# FrontierSynch first-pass port

Branch: `codex/handoff-foundation`.
Committed foundation: `e7537ad90`; scalar/descriptor source port: `2516cdd4d`;
physical/control source port: `38c2f995c`; occurrence/reader source port:
`ff2224d18`. Stage 4 is uncommitted.
Base: upstream `master`, `f5eff3ee249697f6157088f649c6434fcc9d7c5b`.
Donor: `371fdb344d2783b92d6c39424c507b2ce082e08c`.
Draft checked: paper repository `7e3f59c`, revision 0.40, Section 3.

## Current increment: original lifetimes and requirement subscriptions

`frontiersynch::run` imports original structure from the shared `SyncInput`, queries
the Stage 3 reader/occurrence facts, and now indexes original storage demands:

- Donor-style previous/next writer and reader provenance flows retain partial
  histories and the possibility of uninitialized storage.
- RAW/WAR/WAW requirements have stable original source-after and target-before
  positions. Source subscriptions are indexed before construction traversal.
- Original first/last-use may frontiers and affected producer-to-reuse intervals
  retain readers, reloads, bypass/re-entry paths, and other-cell requirements.
- Unknown full-write coverage, occurrence matching and executable endpoint
  guards remain unresolved; no analyzed demand grants selected completion.

The analysis is read-only. Records borrow original IR and instruction pointers;
those owners must outlive the result. Unsupported control reports failure and
leaves the caller's output unchanged. The handoff mode still reports that
construction is not implemented after completing this analysis.

`algorithm=existing` retains the upstream InsertSync construction path and the
same shared instruction input. This increment changes handoff analysis only.

[Port sequence, donor crosswalk, and gate inventory](docs/designs/frontier-synch-first-pass-port.md).

## Scope and remaining limits

The tree is original control, not a selected occurrence refinement. Stage 3's
restricted D1/D2/D4 certificates remain unchanged. Stage 4's marginal histories
and support intervals are complete may facts, not exact generation or protocol
certificates. All imported writes remain non-definite until an effect-coverage
proof is ported; a reload therefore retains older possible writers. Source
subscriptions identify original positions, not ordered command-word gaps or
source snapshots under selected waits. The shared translator remains responsible
for effect completeness.

Next: complete owner/occurrence-qualified first-conflict and last-use queries,
full-write coverage certificates and endpoint-gap predicate availability before
claiming the complete draft Phase A. The first constructor draft then consumes
these indexed requests and evaluates selected causal credit separately.

## Validation

Current handoff/import and translator sources compile in a focused C++17 probe
with the project's warning flags. The probe links existing LLVM 19 and generated
PTO dialect dependencies from the donor build; this is not a fresh full compiler
build. The earlier full CMake configuration remains blocked by the upstream
Python-extension dependency when Python bindings are disabled.

Stage 2 port probes remain at
`/home/toni/work/pypto3_sync_more/handoff-builds/phase-a-stage2/`.
Stage 3 probe source, input cases and build logs are retained at
`/home/toni/work/pypto3_sync_more/handoff-builds/phase-a-stage3/`. Its build is
serial and links current importer/occurrence code with the shared translator and
existing generated LLVM/PTO dialect dependencies. No old constructor is linked.

The induction and carried-selector examples both retain separate periods 2 and
3 despite unrelated carried state. GM-alias and nested if/for/while inputs pass.
The probe also checks nontransitive overlap witnesses, cyclic root propagation,
unchanged source IR, and one tree occurrence per translated phase. The
partially unknown-address input retains the independent period-3 relation and
an unknown footprint. The multi-block CFG is explicitly refused as expected
(exit 6 in the probe). `results.log` and `source-manifest.txt` pin the outcomes
and analyzed sources. `git diff --check` passes.

The Stage 3 focused probe passes optional sibling readers, a late unavailable
predicate, exact fixed visit, nested child interval versus unknown whole-loop
participation, and independent noninjective/exact bank relations. The handoff
entry reports analysis counts and still refuses construction. Focused probe
source/build logs are retained at the Stage 3 artifact path.

The Stage 4 serial probe and inputs are at
`/home/toni/work/pypto3_sync_more/handoff-builds/phase-a-stage4/`. The current
source build and importer probe pass two optional readers followed by reuse and
a reload between children. It checks all affected readers, RAW/WAR/WAW
subscriptions, retention of older possible writers after an unproved reload,
and the affected producer interval. The existing Stage 3 inputs still pass.
This is focused source-port evidence, not a complete linked compiler/corpus run.

No sanitizer, corpus, device or synchronization-quality campaign was run.
Separate generality acceptance of the complete first pass remains pending.

The changed-code scanner (`--base HEAD`) reports ten G.FMT.11-CPP brace
errors in the new lifetime file. Each cited `if` has an explicit brace body;
the scanner's greedy line regex mistakes nested condition parentheses for the
end of the control statement. These are inspected false positives, not
suppressed findings. It reports no other finding. New port files retain donor
license headers.
