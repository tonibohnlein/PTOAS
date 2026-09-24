# FrontierSynch first-pass port

Branch: `codex/handoff-foundation`.
Committed foundation: `e7537ad90`; scalar/descriptor source port: `2516cdd4d`;
physical/control source port: `38c2f995c`. Stage 3 is uncommitted.
Base: upstream `master`, `f5eff3ee249697f6157088f649c6434fcc9d7c5b`.
Donor: `371fdb344d2783b92d6c39424c507b2ce082e08c`.
Draft checked: paper repository `7e3f59c`, revision 0.40, Section 3.

## Current increment: occurrence correspondence and guarded reader frontiers

`frontiersynch::run` imports original structure from the shared `SyncInput`, then
queries guarded reader frontiers and original physical occurrence relations:

- Guarded first/final reader expressions compose through sequence, choice and
  qualified counted loops. Independent readers and alternative readers remain
  distinct. Write-delimited segments retain original owner/interval IDs.
- Predicate availability is checked at represented read sites. Future command
  gaps require their own availability check before an executable endpoint.
- Restricted fixed-visit and physical bank queries consume the original control
  graph and Stage 2 address relations. Exact bank matching requires a single
  disjoint address per residue and mandatory uses.
- Child queries retain source operation membership and skip/repeat possibilities.

The analysis is read-only. Records borrow original IR and instruction pointers;
those owners must outlive the result. Unsupported control reports failure and
leaves the caller's output unchanged. The handoff mode still reports that
construction is not implemented after completing this analysis.

`algorithm=existing` retains the upstream InsertSync construction path and the
same shared instruction input. This increment changes handoff analysis only.

[Port sequence, donor crosswalk, and gate inventory](docs/designs/frontier-synch-first-pass-port.md).

## Scope and remaining limits

The tree is original control, not a selected occurrence refinement. Stage 3
adds an exact predecessor distance only for its restricted D2 permutation
certificate; a scalar period or finite may-footprint alone does not provide it.
The D1 query handles fixed, noncyclic original visits; guarded alternative
origins remain future work. D4 retains original child relationships but does
not transport selected credit across re-entry. Carried or unknown origin
geometry and definite overwrite coverage remain conservative. The shared
translator remains responsible for effect completeness.

Next: stage 4, port generation/support intervals and constructor-facing typed
requirements/source milestones. Qualify predicates at their actual endpoint
gaps and keep source subscriptions distinct from selected causal credit.

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

No sanitizer, corpus, device or synchronization-quality campaign was run.
Separate generality acceptance of the complete first pass remains pending.

The changed-code scanner reports brace errors on expressions with multiple
parentheses; inspected bodies all have braces. These are false positives in its
greedy line regex (G.FMT.11-CPP), not suppressed findings. No other finding was
reported. New port files use explicit braces and retain donor license headers.
