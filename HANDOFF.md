# FrontierSynch first-pass port

Branch: `codex/handoff-foundation`.
Committed foundation: `e7537ad90`; scalar/descriptor source port: `2516cdd4d`.
Base: upstream `master`, `f5eff3ee249697f6157088f649c6434fcc9d7c5b`.
Donor: `371fdb344d2783b92d6c39424c507b2ce082e08c`.
Draft checked: paper repository `7e3f59c`, revision 0.40, Section 3.

## Current increment: physical storage and original control

`frontiersynch::run` now imports original structure from the shared `SyncInput`:

- Original sequence/choice/for/while structure, stable source owners, and every
  translated instruction phase.
- Possible storage roots through structured SSA, finite dependency-sliced
  addresses, canonical intervals and conservative pairwise overlap witnesses.
- Effective descriptor dimensions and original loop-domain qualification from
  the previous source port.

The import is read-only. Records borrow original IR and instruction pointers;
those owners must outlive the result. Unsupported control reports failure and
leaves the caller's output unchanged. The handoff mode still reports that
construction is not implemented after completing this analysis.

`algorithm=existing` retains the upstream InsertSync construction path and the
same shared instruction input. This increment changes handoff analysis only.

[Port sequence, donor crosswalk, and gate inventory](docs/designs/frontier-synch-first-pass-port.md).

## Scope and remaining limits

The tree is original control, not an occurrence refinement. Scalar address
periods do not establish physical predecessor-use relationships. Carried or
unknown origin geometry remains conservative. Definite overwrite coverage is
not inferred from a bounding interval. The shared translator remains responsible
for effect completeness; read-only root analysis does not recreate missing
translated effects.

Next: stage 3, port guarded original read summaries, occurrence correspondence
and child/re-entry relationships into these records. Then stage 4 ports
lifetimes and constructor-facing requirements/source milestones. Keep this a
source extraction, recording necessary adapter changes against the working
specification.

## Validation

Current handoff/import and translator sources compile in a focused C++17 probe
with the project's warning flags. The probe links existing LLVM 19 and generated
PTO dialect dependencies from the donor build; this is not a fresh full compiler
build. The earlier full CMake configuration remains blocked by the upstream
Python-extension dependency when Python bindings are disabled.

Port probes and commands are retained at:
`/home/toni/work/pypto3_sync_more/handoff-builds/phase-a-stage2/`.
The build script compiles the current importer, SyncInput, translator, alias
analyzer, shared macro/common/debug implementation, and probe with at most two
workers. No old constructor is linked.

The induction and carried-selector examples both retain separate periods 2 and
3 despite unrelated carried state. GM-alias and nested if/for/while inputs pass.
The probe also checks nontransitive overlap witnesses, cyclic root propagation,
unchanged source IR, and one tree occurrence per translated phase. The
partially unknown-address input retains the independent period-3 relation and
an unknown footprint. The multi-block CFG is explicitly refused as expected
(exit 6 in the probe). `results.log` and `source-manifest.txt` pin the outcomes
and analyzed sources. `git diff --check` passes.

No sanitizer, corpus, device or synchronization-quality campaign was run.
Separate generality acceptance of the complete first pass remains pending.

The changed-code scanner reports brace errors on expressions with multiple
parentheses; inspected bodies all have braces. These are false positives in its
greedy line regex (G.FMT.11-CPP), not suppressed findings. No other finding was
reported. New port files use explicit braces and retain donor license headers.
