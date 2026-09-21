# KDA first-write investigation handoff

2026-09-21. **WIP research checkpoint, not a merge-ready fix.** The user
requested that this state, including the failing regression, be preserved and
pushed to `codex/kda-pipeline-parallelism`. Do not merge into
`codex/oahs-clean-m1` or use this compiler for a performance claim.

## Start here

Read `AGENTS.md`, this document, `oahs-kda-pipeline-investigation.md`, and
`oahs-selected-plan.md`. The local package is
`/opt/pypto/oahs-kda-handoff-20260921/`; its report is also published at
`/opt/pypto/docs/oahs-kda-handoff-20260921.md`. The package contains an exact
prepared input, selected plans, diagnostic logs, source snapshot, Git patch,
source identity and SHA-256 manifest. It excludes compiler executables, build
trees, dependencies and large device archives. Rebuild before continuing.

Worktree: `/tmp/PTOAS-kda-fix`. Starting HEAD was
`a513ffe300449c6b14173250f8a0ebcc4b314fe4`, a feedback commit atop
`bec7dfe37e7c6a6bcde4860ba878f65ab1b1abae`. The package's `identity.txt` records
the final WIP commit. All implementation changes are in the development
workstream. Frozen benchmark compilers and working runtimes were not modified.

## What is established

The historical controlled full-runtime KDA measurement is for compiler
`2cc458cbe1e5aff6f77ffc6fea62b13686f38a94`, **not this WIP**:
existing 241.810 us, OAHS 318.160 us (+31.57%), identical-existing control
238.9995 us. Alternating rounds and correctness passed. See the earlier
investigation and committed device feedback for measurement scope and limits.
No new device jobs, numerical tests or latency measurements were run for this
first-write extension. Static AIC traces are not complete coupled AIC/AIV runs.

The exact prepared KDA input SHA-256 is
`832a68408907b9d100d7af9bc56b3b8b363907c76ee1b65d4bbcd87826bd7e04`.
Its original server location is
`/opt/pypto/oahs-full-20260921/expanded/lib-pairs/models__glm5_3_flash__kda_projection/handoff/ptoas/kda_projection.prepared.pto`.
The package copies it to `inputs/kda_projection.prepared.pto`.

## Mechanisms implemented so far

1. `CyclicFrontiers.cpp`: allow mixed acyclic/cyclic producer episodes, still
   requiring at least one cyclic writer, complete reader coverage and existing
   participation/safety checks. Pure one-shot placement stays unchanged.
   The linked portable regression passes nine finite paths, missing-channel
   negatives and cached-versus-prefix-only replay. Full payload order is equal.
   **This change alone leaves the native KDA plan unchanged.**
2. `selected_lookahead_test.cpp`: a portable first-conflicting-write witness
   supplies original first-visit information. A single receipt protects later
   writes without gating an earlier outward publication. Three finite paths
   (1/2/5 visits) pass full-order checks. Missing acquisition is rejected.
   Hoisting to entry is cold-safe but adds forbidden outward ordering.
3. `NativeFirstConsumer.h`: extend the existing native occurrence vocabulary
   to first writes with a prior foreign-reader witness from the original
   `StorageFrontierAnalysis`. Require known nonempty positive-step leaf loops,
   an unconditional straight prefix, known/nonexclusive cells and an inactive
   source engine in the region. Preserve the original `lower + step` predicate.
   This supplies vocabulary, not completion credit. Conditional/bypassable
   first writes remain unsupported. Future-only readers do not qualify.
4. `SelectedGroups.cpp`: require source/target correspondence in both directions
   for shared command words, not only first-prefix acquisitions. A publication
   must not execute on an unmatched later/final visit. This avoids two observed
   unbalanced-publication/acquisition failures, but is conservative and is not
   a complete rearming or parallelism certificate.
5. Native tests, the standalone `test/oahs/first_write_outward.pto` reproducer,
   read-only `--explain` diagnostics and three host diagnostic scripts were
   added. No kernel-name recognizer or new authoritative completion state.

## Current acceptance status

| Check | Result / qualification |
| --- | --- |
| Earlier mixed-only checkpoint | 24/24 portable suites and native suite passed; KDA unchanged |
| Simple native first-write cases | Positive nonunit/single-visit cases and conservative negatives passed |
| Latest native suite | **FAIL**, new outward-publication case; later tests short-circuit |
| Latest exact KDA AIC/AIV | Construction and emitted reconstruction both pass |
| Latest portable suite | Not rerun; chained run stopped at native failure |
| Latest 88-input corpus | Not run; `check_corpus.py` is prepared tooling, not a result |
| Device correctness/performance | Not run for WIP |

The two new native missing-support/guard mutations occur after the failing
outward case and were not reached in the final suite. Earlier passing suite
logs must not be presented as final validation.

## KDA plan changes: not a parallelism win

Baseline selected PTO (bec7/mixed-only):
`c73b9089f4a3895f38efbc6bdc137037d28eeb51b348c2badd5039bcf3de5308`.
Candidate selected PTO:
`813c5f331490a8335a302fa1fd89bc96f3dad02289a8e35b09fe609998fb2745`.

| Explicit operations per AIC invocation | Baseline | WIP |
| --- | ---: | ---: |
| MTE2 fences | 67 | 50 |
| MTE1 fences | 196 | 196 |
| M fences | 196 | 196 |
| FIX fences | 4 | 2 |
| FIX to M pairs | 76 | 74 |
| M to FIX pairs | 77 | 73 |
| All SET/WAIT pairs | 1,453 | 1,246 |
| All fences, including terminal ALL | 464 | 445 |

The explicit-command graph diagnostic compares 1,727 payloads, with start and
completion vertices. It finds **14 relations removed and 2,552 added**
(5,953,709 before, 5,956,247 after). Added examples include MTE1 extraction
vertices 40–43 gating matrix vertices 44–45. These are dynamic vertex IDs,
not native operation IDs. Fewer fences/pairs do not establish more parallelism.
The correspondence guard is a plausible source of broader common-cut fallback;
this attribution has not been isolated by an ablation.

Scope: finite scalar execution, explicit events/fences, imported engine map,
queue pushes treated as opaque FIX payloads. No peer protocol, lowering-internal
instructions, ACC target edges, numerical execution or memory-conflict oracle
is added by `compare_order.py`. Native reconstruction separately checks payload
preservation and selected-plan safety. Do not call this a coupled FIFO proof.
All changes above are combined; no isolated benefit is established.

## Failure that must be addressed next

`test/oahs/first_write_outward.pto` has an outer repeated entry, an external
MTE3 read, and a nonunit-step inner loop: V reads B, MTE2 overwrites B, then V
writes A. The earlier V-to-MTE2 release must not be gated by A's first-write
receipt. The fixture also has a genuine shared destination/source GM dependency
between the external store and inner load. A packaging-time check with the
frozen `a0c1d761e7624b42a0f494853cc70eecaabe9c44` driver passes construction
and reconstruction. Its executable hash is
`6d72cfc82b7c596f7818b818c07f8f6fcaffc3cb56a0e8bfa3e2cf3f48a43e7d`.
This establishes a regression relative to a0, but does not isolate which of
the combined later changes causes it. The exact bec7 closed baseline still
needs a controlled check. Logs/plans are `handoff-outward-a0.*`.

Current construction rejects it with `latest consumption does not precede this
publication`. The saved explain ledger ends with V-to-MTE2 publication and
acquisition at cut 15, key 0 (IDs 10/11). Emission is transactional; the failed
plan output is empty and must not be counted or timed. This is a compact
reproducer, not a claim of global minimality or isolated first-write attribution.

Earlier KDA attempts failed must-empty publication at cut 99/106, then rearming
at 428, then must-full acquisition at 105. A global reader list admitted
unnecessary boundaries; it was replaced with prior-reader witnesses. Attempts
to split all prefix words were reverted. Their logs are historical debugging
evidence, not reproducible final variants; exact intermediate source snapshots
were not retained. `first-write-shared-*` is the passing KDA checkpoint;
`first-write-final-tests.log` contains the new native failure.

## Reproduction and next actions

Build from this checkout with GCC 15, Release, LLVM/MLIR 19.1.7. Exact local
configuration and compiler executable hash are in the package. Use two workers:

```bash
CCACHE_DISABLE=1 cmake --build build-kda --target pto-oahs-selected-test --parallel 2
build-kda/tools/pto-test-opt/pto-oahs-selected-test
build-kda/tools/pto-test-opt/pto-oahs-selected-test --construct test/oahs/first_write_outward.pto
build-kda/tools/pto-test-opt/pto-oahs-selected-test --explain test/oahs/first_write_outward.pto
build-kda/tools/pto-test-opt/pto-oahs-selected-test --construct PACKAGE/inputs/kda_projection.prepared.pto
```

For static counts/order, use `test/benchmarks/kda_projection/count_trace.py` and
`compare_order.py`. Scalar bindings are
`{"%arg16":67,"%arg17":67,"%arg18":67,"%arg19":0,"%arg20":5}`.
Imported engine map is `tload:MTE2`, `textract:MTE1`, `tmov:MTE1`,
`tmatmul:M`, `tmatmul.acc:M`, `tpush:FIX`, `tinsert:FIX`.
The package's `reproduce.sh` has the complete diagnostic commands.

Next: compare the small reproducer against the exact bec7 closed baseline; trace the
exact key generation and selected consumption-support path; isolate native
qualification from the shared-word guard and mixed-producer change. Preserve
useful publication prefixes rather than accepting a late common-cut fallback
merely because it is safe. Review qualification cost (additional original
storage analysis, repeated origin queries) and shared-occurrence cross-product
cost before promotion. Resolve the native failure and added-order examples,
then run cached/cold comparisons, full portable/native tests and hash-verified
corpus checks. Only then qualify unchanged coupled payloads on-device and
measure with alternating arms and identical-binary controls through TaskQueue.

Review verdict: **not merge-ready**. The WIP push is an explicitly requested
handoff checkpoint. Do not hide the failing test, claim the old 24/24 applies
to this tree, or equate this partial mechanism with a completed KDA fix.
