# Step 8: D2 periodic physical correspondence

**Integration update (2026-09-25): accepted at this component's scoped v0.44 gate.**
See [current integration evidence and corrections](frontier-synch-steps7-13-review.md).
Patch-delivery validation/status statements below describe the original submission.


**Patch base:** `4122dd1531fbdb2859bd7b27772dba56930be393`, branch
`codex/handoff-foundation`, with steps 2–6 integrated.

**Specification:** supplied draft v0.44, Section 3.6 D2 (p. 12), the physical
selection/finite-footprint distinction in Section 3.4 (pp. 9–10), the
owner/occurrence contract in Sections 3.2/3.6, and the noninjective/re-entry
qualification boundary of Appendix I.1. This is a bounded Phase A change.
D1 and D4 construction, exact reader-frontier derivation, descriptors,
subscriptions and selected synchronization are not implemented by this patch.

**Status:** implementation and regression patch prepared; native compilation,
full-checkout application validation and independent reviewer acceptance remain
pending. The executed checks below are not described as independent acceptance.

## Procedure and interface

`PhysicalPermutation.h` is the MLIR-independent production core. It checks an
explicit population of nonempty, nonoverlapping physical intervals and an
injective successor function, constructs the inverse function, and follows each
permutation cycle once to derive its length. It handles one-bank cycles and
multiple cycles of different lengths. There is no modulus expansion or dynamic
trace unrolling in production.

`OccurrenceQueries::bank` now requalifies the original address expression using
`SyncSlotMapping::derive/evaluate`, including closure of the complete relevant
scalar dependency state. It verifies every recorded physical address against
that expression and the translated effect offset before accepting the physical
permutation. A repeated output address alone is insufficient. Its cached result
is stamped by the original snapshot. Static incidences and optional/repeated
incidences are reported separately: a geometric permutation is not an
exactly-once certificate for every reader.

`OccurrenceQueries::periodic` performs the role query. It retains the complete
original interval key, orders the actual static roles by original structure,
and compares their physical streams rather than requiring equal relation IDs or
one shared instruction. Different reader sites stay distinct. Reads between a
producer and another reader do not terminate that relationship; an intervening
write prevents this D2 relationship from being certified. Fixed, partial,
noninjective or differently periodic interfering accesses remain visible and
produce an obstruction rather than inheriting a convenient period.

The query also accepts a direct allocation whose shared address analysis already
supplies a qualified explicit finite address population. This path re-derives the
original transition within that population; constant addresses are its identity
permutation. Both paths reuse the existing scalar and value analyses. They do
not add another instruction-effect registry or infer full-write coverage.

For each relevant physical bank, the result contains a source/target role link
and four domains: predecessor and initial at the target, successor and final at
the source. A same-visit writer-to-reader link has distance zero; an ordinary
reader-to-next-writer link has the bank cycle length. Shifted selectors can have
bank-specific distances. The source and target domains use their own phases.
The initial case is **no local predecessor**, not completed/fresh storage. The
final case retains the enclosing continuation, not an automatic drain. Entry,
exit and the zero-trip bypass have explicit original cuts/observations.

Endpoint predicates use equality against the **original address SSA value** and
the existing width-qualified `LoopHasPrevious`/`LoopHasNext` recipes. There is no
new runtime ordinal counter or ambiguous reinterpretation of `LoopResidue`.
Every directional domain retains its own selector/boundary qualifications and
any independently dischargeable completion prerequisites. Legal original cuts
are checked; internal noninsertable phases are not silently replaced by another
point. The semantic ordinal oracle uses subtraction/range tests so a successor
check cannot overflow by speculatively adding the distance.

Disjoint selector populations are excluded by their existing may-footprints,
without forming an LCM/product history. A canonical cell query restricts both
returned links and interference to that cell's banks. Conservative overlap
witnesses are not discarded; actual per-effect address proofs may qualify a
local relation without upgrading that witness to a full-write proof.

## Public consumers and unchanged obligations

`ProgramAnalysis::periodicUseFor(OriginalObligationId)` calls this service directly
from the step-6 immutable obligation universe. It uses lazy membership, not
compatibility-pair enumeration. The returned local placement relationship does
not replace the obligation's whole-prefix scope, applicability or incoming cases.
A request about an incoming or typed source retains its separate interface.

`ProgramAnalysis::interpretAt` uses the same service for the compatibility view.
`PeriodicSameRole` remains available for self-role recurrence; `PeriodicRoles`
represents distinct roles. The complete periodic record is authoritative. The
legacy scalar distance fields are only a uniform-distance convenience and must
not be used instead of the per-bank domains.

No requirement, source/consumer effect witness, source identity, allocation,
payload operation or original control decision is removed. No selected
completion, event ID, receipt or protocol is introduced.

## Parity inventory delta

| Entry | Procedure now supplied | Evidence and remaining gate |
| --- | --- | --- |
| PA-043 | Explicit permutation/inverse/cycle core; original scalar/address requalification; one-bank identity support | Core permutation oracle executed; native induction/carried fixtures added but not run |
| PA-044 | Both occurrence maps and four directional domains; legal source/target cuts; incoming/final/bypass records | Finite core oracle executed, including zero/short/final and overflow cases; native endpoint recipes await execution |
| PA-045 | Distinct producer/readers and role-sensitive interference; public lazy-ID and compatibility consumers | Core multi-reader and interfering-write checks executed; native public-API assertions await execution |
| PA-046 | Independent physical populations; noninjective selection retains may information but no D2 proof | Core negatives/independent-family checks executed; imported footprint-preservation fixture awaits execution |

These are implementation/evidence updates, not a declaration that the old
checklist's entire integrated system has passed its final parity gate.

## Executed validation

The standalone test compiles the **same production core** used by the native
adapter. Its oracle constructs concrete finite access words independently and
scans them for matching physical predecessors/successors. It does not use the
returned distances to select those oracle occurrences.

- All 5,913 permutations on one through seven explicit banks: inverse edges and
  individual cycle lengths checked by independent cycle walks.
- 1,340 finite access traces: writer/readiness, two distinct readers, returns,
  writer recurrence, phase-shifted and non-rotation reader selection.
- Initial/final complements, first/last uses, empty traces, malformed/overlapping
  banks, noninjective A,A,B,B selection, interfering writes, harmless readers,
  cell-specific interference, independent periods and UINT64_MAX boundaries.
- A supplied 1,025-bank case checks that the proof core has no 256-state trial
  cutoff on an explicit population.

The test reports **412,407 always-active assertions**. It was run with Clang 17
at `-O2 -DNDEBUG -Wall -Wextra -Werror`, GCC 14 at the same settings, and Clang
AddressSanitizer/UndefinedBehaviorSanitizer. All completed successfully.

The patch's context fixture contains the retrieved original source windows;
the complete `OccurrenceQueries.h/.cpp` originals were additionally checked
against their Git blob SHA-1s. Patch application to that fixture is not a
checkout-wide application check. The checkout-side verifier distributed with
the patch checks the pinned original files and every old hunk before applying.

## Native checks supplied, not executed here

`pto-frontier-periodic-test` builds 23 real PTO/SCF fixtures through the shared
`SyncInput` importer and public `ProgramAnalysis` API. Expected answers include
induction/carried-selector equivalence, periods 1/2/3/5, a stride-two residue
cycle, phase-shifted readers, real zero/short/final invocation bounds, two reader
sites, independent periods, noninjective physical selection, interfering writes,
a fixed interfering bank, and an optional-reader qualification obstruction.
It checks domain recipes against explicit accesses, lazy obligation identity,
compatibility-consumer agreement, full interval distinctions, stale snapshot
refusal, retained footprints and unchanged IR. CMake, lit tool substitutions and
`check-pto` dependencies include both new test binaries.

This environment has no LLVM/MLIR/PTOAS build installation and could not clone a
complete repository. Therefore these native tests, the existing-mode suite and
the five development kernels were **not** compiled or run. Run them with the
repository toolchain before accepting this increment:

```sh
cmake --build build --target pto-frontier-periodic-core-test pto-frontier-periodic-test
build/tools/pto-test-opt/pto-frontier-periodic-core-test
build/tools/pto-test-opt/pto-frontier-periodic-test
cmake --build build --target check-pto
```

## Supported limits and review gate

The local pairing procedure requires a counted owner and a qualified, mandatory
single-visit source/target role word. Transparent sequence wrappers are allowed.
Optional/repeated participant frontiers need their D3 qualifications; a boundary
inside an unfinished use or a relation crossing owner re-entry needs step 9's D4
transport. Those are not treated as fresh entries or guessed distances. The
geometric bank proof is retained separately where available.

The existing `SyncSlotMapping` import-time discovery fragment and its default
256-state exploration limit are unchanged. Once a finite population is supplied,
the new proof/requalification uses that explicit size, not the default limit.
This patch does not claim discovery of an arbitrarily large implicitly encoded
bank population or a general recurrence solver. An importer-side discovery
obstruction is distinct from a refuted physical permutation.

Self-review checked the distinction between geometry and role multiplicity,
query-cell filtering, source/target phase alignment, distance-zero domains,
positive-length self recurrence, complete writer interference, arithmetic-safe
successor domains, whole-prefix obligation preservation and versioned caches.
The cell-specific interference and explicit-large-population regressions were
added during that review. **Independent review is still pending.** Acceptance
requires native build/test results and an independent examination of PA-043–046;
step 14 must still review the integrated Phase A system.
