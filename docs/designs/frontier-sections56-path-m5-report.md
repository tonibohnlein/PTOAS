# M5 — Realized-order loss and final conformance

Baseline: `338cc1923` (accepted M4); the complete-plan corpus comparison uses
`d6a920bf`. This implements the qualified loss capability in the
[plan](frontier-sections56-path-completion-plan.md). The
[conformance table](frontier-sections56-conformance.md) separates supported
production paths from unavailable exports and stronger unqualified contracts.

## Retained certificate

`CompletionRegion` retains an immutable CFG partition. Each atom identifies its
original root interval by function postorder ordinals, its fixed payload count
or requested actual visit count, and whether internal realized order is equal
to the required order. Source ordinals come from the shared original inventory;
inserted commands and staging clones cannot change their meaning.

`CompletionLossCertificate` validates vertex/atom alignment, successors, entry,
exit and interval shape before publishing a diagnostic. For an admitted finite
walk, the numeric fold appends `(u, U_R)` to `(n, U)` as
`(n+u, U+U_R+n*u)`. The default cross lower count is zero. Opaque children use
`u*(u-1)/2`; zero requires the existing concrete realization-order classifier,
including its same-pipe placement premises. Mathematical minimum demands alone
do not authorize zero. Fixed counts and CFG edges are checked; branch outcomes
and opaque counts remain supplied execution evidence, not a runtime analyzer.

The 256-bit fold checks products and sums before adopting them. Overflow leaves
the accumulated numbers intact and returns an explicit unavailable result.
Zero visits and the final false while-condition visit are distinct CFG paths.
`ResultGuarantees` exports this certificate only for accepted completion owners.

The existing periodic excess engine remains authoritative for its supported
upper/lower graphs. Its fold adapter additionally requires equality between the
realized order and the supplied upper graph. This adapter is tested; general
production completion children still use the coarse bound unless the provider
establishes that stronger premise. This is not a new repeated-child executable
import or a claim of universal linear loss.

## Independent checks

The loss fold is compared with explicit CI closure differences over 4,000
randomized finite partitions, plus invalid walks/counts, empty/repeated visits,
final-false conditions, overflow and missing periodic support. Existing periodic
oracles check finite-prefix excess and Gamma under their own premises.

Two production fixtures exercise retained finite children through actual staged
command insertion. The independent graph observes emitted local/ALL barriers,
native start/start and completion/completion order, and physical-cell hazards.
The adjacent child has actual excess 0 with bound 0. The nonadjacent local-barrier
child has excess 1 with bound 3. The latter must not receive a zero certificate.
The first test run missed a barrier nested in a constant-true `scf.if`; the oracle
was corrected to validate that shape and inspect its body. No production
predicate, placement or expectation was weakened.

## Full-gate integration correction

The first broad run exposed a standalone bounded-preparation adapter that built
a new `PhaseIndex` after recognition. M3's prerequisite-scope check correctly
rejected it. The sole caller now builds one index and passes it through both
recognition and preparation; the adapter no longer rebuilds the index. Foreign
or rebuilt scope rejection remains unchanged. The original baseline passes the
physical-refresh oracle. The interrupted pre-fix campaign is retained separately
from final qualification, not counted as a passing gate.

The first completed broad run passed 338/344. Its sequence cache test also
created a separate index, despite testing identity-scoped cache reuse. It now
passes one index to recognition, root/child analysis and the cache retry, while
retaining foreign-input rejection. The phase test had expected comparison
specialization to fail; it now checks the exact representative result and
separately rejects the false period-two claim for `i<2`. Unsupported quotient
roots remain rejected through both eager and lazy exports. These corrections
were independently reviewed; no production prerequisite or oracle predicate was
weakened. The CLI shared compiler and Python core were incrementally relinked
before repeating the legacy CLI cases. Three legacy InsertSync expectations
still fail after that relink: slot-transitive, same-region two-gets and two-region
two-gets. Running the ordinary memory-planning/InsertSync pass sequence with
both the pinned original and candidate optimizer yields byte-identical output
for all three. Their expectations are unchanged; they remain open legacy
failures rather than Frontier acceptance successes.

## Measured construction and corpus behavior

Three serial interleaved baseline/candidate pairs compare M5 with M4 through
logical insertion and ordinary allocation using IDs {2,5}. Candidate/baseline
median paired ratios are 1.009 for nested children and 1.015, 1.005, 1.005 for
8, 32, 128 children. These are near-neutral results, not a speedup claim.
Command and arithmetic inventories are unchanged. The geometric certificate
inventories contain 37, 109, 397 atoms respectively: linear in this supplied
CFG. Diagnostics add serialized metadata, not executed control arithmetic.

The 49 primary PyPTO-lib modules compile logically and allocate within six IDs
against both the original `d6a920bf` baseline and the candidate. Thirty outputs
are byte-identical; eleven more differ only in recognition diagnostics. The
remaining eight contain M4's imported exact children within fallback, so their
command arrays deliberately differ. They are not counted as equivalent outputs.

For those eight modules, 32 fixed executions covering 1,012 payloads preserve
the payload sequence, pass the independent physical occupancy/causal-reuse
checker, and add no payload order relative to the original fallback. These
checks include skipped and active paths and executions with 132 payloads. They
do not independently recover arbitrary production storage effects; demand
coverage remains separately checked by the finite and completion semantic
oracles. No corpus-wide semantic-equivalence claim follows from stage success.

## Qualification status

The fresh complete synchronization gate passes **341/344**, with no timeout.
All selected Frontier tests pass. The three failures are the unchanged legacy
InsertSync expectations documented above. This is a fresh complete run, not a
sum of repaired subsets. The machine-readable
[receipt](frontier-sections56-path-m5-results.json) records the failures, hashes,
paired timings, corpus outcomes and fixed-execution checks. No device compilation or
execution is part of this campaign. Combined physical output for completion,
unrestricted eventful/while child imports, coordinator completion and runtime
streaming remain outside the qualified implementation.

## Build and review discipline

All compilation reused the existing LLVM build. The native test tools and CLI
shared libraries were updated incrementally, with at most two aggregate workers.
No full toolchain rebuild, test-limit increase or expectation suppression was
used. The changed-code prefilter's brace reports were inspected: all indicated
control bodies have braces; nested parentheses/multiline conditions confuse the
regular expression. The new overlong call was wrapped. The rebuilt unchanged
`SessionProducers` retains its existing covered-switch-default warning; no new
warning was introduced by M5. `git diff --check` passes.

Architecture and correctness reviewers accepted the source, ownership correction,
phase/cache oracle updates and stated scope. Both reviewers also accepted the final
runtime receipts, including the fresh 341/344 gate and explicitly retained
legacy failures. The machine-readable receipt retains the
final tool hashes, test outcomes, production comparisons and timing pairs.
