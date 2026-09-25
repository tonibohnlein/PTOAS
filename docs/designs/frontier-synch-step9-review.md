# Phase A step 9: D4 composition and supported re-entry

**Integration update (2026-09-25): accepted at this component's scoped v0.44 gate.**
See [current integration evidence and corrections](frontier-synch-steps7-13-review.md).
Patch-delivery validation/status statements below describe the original submission.


## Baseline and review status

Patch base: `4122dd1531fbdb2859bd7b27772dba56930be393` on
`tonibohnlein/PTOAS`, branch `codex/handoff-foundation`. This is the integrated
steps 2–6 snapshot, not the older `0a38c8e9f` baseline.

Specification: supplied draft v0.44, Section 3.6 D4 (page 12), the original
owner/continuation contract in Sections 3.2/3.4/3.6, and Appendix I.1
(pages 78–80). The original-versus-selected distinction is unchanged.

**Candidate increment; independent reviewer acceptance is pending.** The portable
production core has executable semantic evidence. Native integration code and its
PTO fixture are supplied but were not compiled or run in the preparation
environment. This report is a scoped delta for PA-047–PA-049; it neither replaces
the full parity inventory nor records full Phase A acceptance.

## Implementation and actual consumers

| Component | Procedure and consumer | Meaning |
| --- | --- | --- |
| `OriginalProgramPoints.h` | `normalizeOriginalSequences`, called by the `ProgramAnalysis` constructor before values, lifetimes, occurrence or reader indexes | Remove anonymous sequence wrappers consistently for all consumers; retain named scopes, choice arms, for bodies and both while regions. No original IR changes. |
| `D4Composition.h` | `D4ChildSummary`, `composeD4Children`, `buildD4Composition`; called by `FactoredProvenance` | Compile each child using the existing `FactoredUseBuilder`, then substitute shared incoming/following writer and reader roots in the enclosing arena. The existing `ProgramAnalysis::originalUsesAt` and `applyOriginalUsesAt` paths consume this composition. |
| `D4Relations.h` | `composeD4Relations` | Compose already qualified complete-use child profiles in one physical owner/interval; retain their D1/D2 correspondence objects and previous/next/initial/final domains. Supply cross-child previous/next role references, not a guessed new distance. |
| `D4Relations.h` | `composeD4ReadFragments` | Apply the D3 sequence equations to first/last reader references inside the same unfinished enclosing use. Empty paths have no reader frontier. No complete-use transition is exported for a read fragment. |
| `D4ProgramQueries.h` | Public facet over `ProgramAnalysis`: `fixedChild`, `noUseChild`, `completeUses`, `readFragments` | Supply a narrow native D1 child or an All exclusion; validate full original interval, physical witnesses, legal cuts, coverage and child order before composing supplied profiles. Results retain access to the unchanged obligation universe. |

No changes are made to `SyncInput`, shared instruction effects, the accepted
`FactoredUseBuilder` transfer equations, the original obligation-ID model, or the
existing synchronization emitter. D4 does not publish, consume or reset events.

## Composition contract

Each reusable child summary has four distinct formal DAG roots: incoming writers,
incoming readers, following writers and following readers. Forward application
substitutes the current original histories; backward application substitutes the
continuation. Substitution memoizes the template and shares actual input roots
without traversing or copying their histories. Access nodes retain their immutable
translated-effect witnesses. Shared original Boolean identities are not renamed
by child position.

A definite write updates subsequent provenance but cannot erase requirements
already formed by that child or an earlier child. Partial writes and RMW use the
existing conservative/old-state rules. The result retains unresolved repeated
interfaces and their incoming histories. An All-excluded repeat preserves this
physical projection only, not any selected causal state.

The fixed-use transfer must not be applied twice to represent two dynamic visits
of one static access. Such duplicate-site composition is rejected. Re-entry is
instead represented by the separately qualified occurrence profiles below.

## Complete uses, unfinished uses and re-entry

`D4ExistingCorrespondence` is an immutable, retained input premise. It includes the
rule, full original interval, first/last role references and participation, plus
any qualified distances/domains. D4 checks that the child profile names exactly
those qualified boundaries and that same participation. It does not accept a
profile merely because it contains `rule = D2` or an address has a modulus.

For complete uses, a forward shared expression retains the last participating
use from preceding children; the backward dual retains the next participating
use. A no-use child leaves these physical references unchanged. An unqualified
child inserts an explicit unknown dependency, preventing a later child from
silently bypassing it as though it were access-free. The initial/final expression
references denote the actual owning interfaces, not fresh storage.

A reset of a child-local selector never resets this composition state. The child
qualifier must supply the correct physical first/last use in the enclosing
interpretation. The retained D2 distance remains local to its qualified domain;
a cross-child link retains role references rather than applying that distance
across a reset. Opaque occurrence descriptor IDs are not new runtime counters.

Inside an unfinished read episode, all child premises must retain the same
qualified enclosing-use identity. An explicit incoming use is allowed only under
its named incoming interface. The first/last equations retain optional empty
paths and the enclosing owner. A different writer/use identity rejects this
same-episode composition. The native read adapter additionally rejects an
intervening write in its original interval and keeps separate reader-engine
queries.

## Supported premises and remaining gates

The following are distinct, and must not be conflated at review:

- **Implemented composition procedure:** same-owner fixed-use child transfers;
  composition of supplied qualified D1/D2 complete-use boundary profiles;
  composition of qualified unfinished-read profiles; physical no-use identity;
  explicit missing-premise propagation and unchanged original obligations.
- **Native positive supplier provided here:** a D1 child with exactly two related
  cell accesses, a qualified definite writer and pure reader, current matching
  applicability and legal immediate cuts; and complete All exclusions. The
  remaining native D1/D2 suppliers belong to steps 7–8. Their absence is not
  reclassified as a draft-open problem or filled with an invented relation.
- **Not implemented by this increment:** derivation of arbitrary reset/carried
  selector coordinate maps, general mixed-write repeated transfer summaries,
  general conditional recurrence matching, or transporting selected publications.
  An opaque repeated projection stays opaque unless another qualified service
  supplies its relation; the original marginal obligations remain available.
- **Separate endpoint/constructor gates:** a complete original first/last
  composition is not a proof that its newly combined guard is available at an
  earlier endpoint. Steps 10/12 and the existing original-value service must
  qualify executable endpoint conditions. No result here grants selected
  completion, event-generation matching or an E6/F8 induction proof.

The native facet deliberately rejects a body-visit predicate arena used for an
interval that crosses a backedge. It also rejects a whole-loop interval being
interpreted as one fixed body visit. Those are missing occurrence interpretations,
not permission to discard their conflicts.

## Executable evidence

`pto-frontier-d4-test.cpp` tests the production headers, not a reimplementation of
the composition. Its small-input oracle independently scans concrete forward
writer/reader histories, old-state RAW/WAR/WAW demands and backward next-use
histories. It also compares the composed result with the unchanged monolithic
`FactoredUseBuilder`.

Covered cases include all 256 four-access combinations over the chosen effect
modes; conditional writers, shared guards, RMW and partial writes; splits around
an overwrite and inside an unfinished use; explicit incoming histories;
transparent wrappers; real choice/for/while boundaries; unknown and All-excluded
repeats; stale snapshots, different owners/intervals and wrong expression sorts;
missing or mismatched retained child premises; both directions of conditional
complete-use transport; and finite nested selector-reset oracles.

The nested selector tests supply their qualified child profiles explicitly.
They compare physical predecessors with finite executions of `j % 2` and
`(3*i+j) % 2` across children. They do **not** claim that the native frontend has
derived those general coordinate maps.

All three executed configurations returned:

```text
D4 checks=16611 concrete-scans=530 PASS
```

Configurations: Clang 17 release with `-DNDEBUG`; GCC 14.2 release with `-DNDEBUG`;
and Clang 17 address/undefined-behavior sanitizers with leak detection. All used
C++17, `-fno-exceptions`, and `-Wall -Wextra -Wpedantic -Werror`. Assertions remain
active in release builds.

Large independent-reader formation does not enumerate valuations:

| Optional readers | Child summaries | Added expression nodes | Visited substitution nodes |
| ---: | ---: | ---: | ---: |
| 64 | 66 | 1,429 | 454 |
| 256 | 258 | 5,653 | 1,798 |
| 1,024 | 1,026 | 22,549 | 7,174 |
| 4,096 | 4,098 | 90,133 | 28,678 |

These are formation counters for the tested fixed-use family, not compiler
latency, a native whole-program bound, or a bound on arbitrary symbolic matching.
Summary templates and applications are both charged. Native interval scans,
original syntax indexes, qualification and materialized query output remain
additional work; this patch makes no production compilation-time improvement
claim.

## Native and integrated checks still required

The added native executable uses verified PTO through `SyncInput`, the physical
importer and `ProgramAnalysis`. It adds anonymous representation wrappers before
analysis, then checks native D1/All child qualification, complete-use transport,
rejected omissions/reordering/duplicate uses, wrong cuts, missing correspondence,
unchanged obligation identities and unchanged original IR. CMake targets and the
lit driver `test/lit/pto/frontier_synch_d4.pto` are included.

**This native executable has not been compiled or executed here.** The preparation
environment had no PTOAS/LLVM/MLIR development build. Existing native tests, both
autosync modes, the five development kernels and device tests were not rerun.
No separate reviewer was available; self-review and independent concrete oracles
are not independent reviewer acceptance.

Run against an already configured project build, substituting its build path:

```sh
cmake --build build --target pto-frontier-d4-test pto-frontier-d4-native-test \
  pto-frontier-analysis-test pto-factored-provenance-test \
  pto-frontier-interval-test pto-frontier-interval-core-test
build/bin/pto-frontier-d4-test
build/bin/pto-frontier-d4-native-test
build/bin/pto-frontier-analysis-test --factored-self-test
```

Then run the existing focused lit/standalone tests using that checkout's runner,
followed by the integrated Phase A and existing-mode regressions. In particular,
review the new native relation-premise adapter and cross-step 7–8 supplier contract
before accepting PA-047–PA-049. Step 14 remains a separate integrated parity gate.

## Patch preparation and application check

GitHub connector reads supplied the pinned source. Direct cloning/downloading and
a native build were unavailable. The complete reconstructed base copies of
`OriginalProgramPoints.h`, `FactoredUse.h`, `FactoredProvenance.h` and the tool
`CMakeLists.txt` were checked against their upstream Git blob hashes. The small
`ProgramAnalysis.cpp` constructor hunk and parity-index pointer hunk use their
exact retrieved upstream contexts. Patch application was checked against those
retrieved files/contexts, not against an entire cloned PTOAS checkout.

Apply with `git apply --check` in the actual branch checkout before `git apply`.
A later branch head may need a rebase. No remote commit or push is part of this
patch preparation.
