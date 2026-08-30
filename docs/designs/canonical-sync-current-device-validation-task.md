# Device task: current CanonicalSync mode validation and performance attribution

## Dispatch status

`READY` after the branch push containing this document completes.

Do not start from the moving branch name. Check out the exact compiler-source
SHA below. The documentation commit containing this task is allowed to sit on
top because it changes no compiler source.

## Objective

Validate the corrected CanonicalSync implementation on the same 28-case corpus
and historical GEMM used by the prior campaign. The experiment must answer:

1. Do the newly generated plans execute correctly, including historical GEMM
   cases that failed with AI-core exception `507015` on the old refined build?
2. Does production action-first CanonicalSync improve device time over
   InsertSync, and on which kernels?
3. How much value comes from transitive pair coverage, as distinct from the
   synthesized lifecycle/ownership mechanisms?
4. Does the serialization-first objective predict device performance better,
   or does its localized `PIPE_ALL` fallback make it worse?
5. When the older performance branch loses to InsertSync, are dynamically
   executed body `PIPE_ALL` barriers the cause, a contributor, or unrelated?

Correctness and structural equivalence are hard gates. Static synchronization
count is diagnostic; it is not a runtime verdict.

## Exact provenance

### Current candidate

```text
remote:  git@github.com:tonibohnlein/PTOAS.git
branch:  codex/canonical-sync-refined       # provenance only
sha:     bf89e4e89e3f5640d0e490846e09e60934a4c8b0
subject: fix(sync): preserve changed-core repair provenance
```

The local compile campaign used:

```text
manifest SHA256: fdc6a4e08b822bb6ee2fff7ebcc3b99855a481f885ac01d004d07f8b098ceb00
GEMM fixture SHA256: e21a921e552c04d85af3e1a814a7d0a6d5b0c708dd9a52a19d92c3484f2219a9
results.json SHA256: c452c7a09ca851bc2aad825e1364c7319172cb29d117b42fa0ec37bb06f6c9f6
committed input/results manifest SHA256: c0cde7d5ad6d494f2357c93d4f7ef322c49d43efa9dbc3a460823f3057cb7447
```

Use `canonical-sync-mode-ablation-results.md` as the committed authoritative
29-input manifest; its appendix records every source and SHA256. The raw
`results.json` is a separate campaign handoff artifact with the hash above.
Copy or regenerate each pre-sync PTO input, then require its SHA256 to match.
Do not substitute a similarly named kernel.

### Performance-branch diagnostic revision

```text
remote:  git@github.com:tonibohnlein/PTOAS.git
branch:  codex/canonical-sync-covering-performance  # provenance only
sha:     5eab65e919ca053cc93a96ab09efd3ffa434a70b
```

Build this revision in a separate checkout. Its primary causal comparison is
its own CanonicalSync against InsertSync built from the same revision. A
cross-revision timing difference is valid only if normalized non-sync PTO,
backend code, ABI, and launch geometry are identical.

Record both checkouts' clean status, merge base, submodule SHAs, LLVM, CANN,
PTO-ISA, compiler, Python, runtime, driver, firmware, board model, and exact
resolved `ptoas` paths.

## Resource and isolation policy

- Follow the device host's standing resource policy and record actual worker
  counts. Do not copy the laptop's local worker ceiling into this task.
- Build each compiler revision once. Compile every arm from the same frozen
  pre-sync PTO for that case.
- Use private checkout, build, cache, generated-code, binary, result, and
  device-output directories.
- Use finite compile and launch timeouts.
- Run only one timed workload on a device at a time. Do not compile or profile
  concurrently on the timing device.
- Restore mutable inputs before every correctness launch and timing block.
- Do not modify, commit, or push compiler source from the device host.

## Current-candidate arms

All canonical arms use event budget eight. The 28 corpus cases use
`--canonical-sync-assume-distinct-gm-args-noalias`; historical GEMM uses
`--canonical-sync-assume-all-gm-accesses-noalias` after verifying that the
runtime ABI satisfies that stronger contract.

### `insert_current`

```text
--enable-insert-sync
```

### `full_direct`

Full synthesized mechanism catalog, transitive pair catalog disabled:

```text
--enable-canonical-sync
--canonical-sync-mechanism-families=all
--canonical-sync-pattern-mode=direct
--canonical-sync-selection-strategy=action-aware-singleton
--canonical-sync-selection-objective=action-first
```

### `full_pair_singleton`

Full catalog and transitive pair coverage, one greedy mechanism at a time:

```text
--enable-canonical-sync
--canonical-sync-mechanism-families=all
--canonical-sync-pattern-mode=direct-pair
--canonical-sync-selection-strategy=action-aware-singleton
--canonical-sync-selection-objective=action-first
```

### `full_pair_lookahead`

Production/default action-first configuration:

```text
--enable-canonical-sync
--canonical-sync-mechanism-families=all
--canonical-sync-pattern-mode=direct-pair
--canonical-sync-selection-strategy=pair-lookahead
--canonical-sync-selection-objective=action-first
```

The local campaign found `full_pair_singleton` and `full_pair_lookahead`
byte-identical on all 29 cases. Compile both and prove this again. If their
post-sync PTO and backend binaries remain identical, time only
`full_pair_lookahead` and classify pair lookahead as `NO_EMITTED_EFFECT` for
this corpus.

### `full_serialization_first` (diagnostic)

```text
--enable-canonical-sync
--canonical-sync-mechanism-families=all
--canonical-sync-pattern-mode=direct-pair
--canonical-sync-selection-strategy=pair-lookahead
--canonical-sync-selection-objective=serialization-first
```

This is not a proposed production arm. Locally it used 62 body `PIPE_ALL`
barriers in case 10 and 45 in case 25 after exhausting the repair budget before
running a repair trial. Run it for correctness and performance attribution,
but report that fallback prominently.

### `gemm_ownership`

Historical GEMM only:

```text
--enable-canonical-sync
--canonical-sync-mechanism-families=l0-operand-ownership+basic-ownership+boundary-ownership+hierarchical-ownership
--canonical-sync-pattern-mode=direct
--canonical-sync-selection-strategy=action-aware-singleton
--canonical-sync-selection-objective=action-first
```

Do not run the restricted direct-only catalog on general corpus cases. It is a
fail-closed completeness ablation and locally compiled only 7/29 cases.

## Performance-revision arms

For every executable case compile byte-identical pre-sync PTO with:

1. `insert_performance_revision`: InsertSync from SHA `5eab65e...`.
2. `canonical_performance_revision`: production CanonicalSync from the same
   SHA and the same sound alias contract.

This within-revision pair is the primary experiment for the older performance
branch. Do not compare it only with InsertSync from the current revision.

## Local structural reference

The current candidate compiled and freshly verified all 29 cases in both full
objectives. Body counts exclude the automatic final tail `PIPE_ALL`.

| Arm | Compiled | Set | Wait | Targeted barrier | Body PIPE_ALL | Body total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `insert_current` | 29/29 | 446 | 446 | 165 | 0 | 1057 |
| `full_direct` | 29/29 | 302 | 302 | 272 | 0 | 876 |
| `full_pair_singleton` | 29/29 | 291 | 291 | 258 | 0 | 840 |
| `full_pair_lookahead` | 29/29 | 291 | 291 | 258 | 0 | 840 |
| `full_serialization_first` | 29/29 | 337 | 337 | 221 | 107 | 1002 |

Pair coverage was active in 15 cases, with 78 surviving direct pairs and 338
extra covered rows. Enabling the pair catalog reduced the aggregate body count
by 36 versus `full_direct`, but pair lookahead changed no emitted output beyond
singleton greedy. Preserve and explain any drift.

Historical GEMM locally produced:

| Arm | Set | Wait | Targeted barrier | Body PIPE_ALL | Body total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `insert_current` | 44 | 44 | 21 | 0 | 109 |
| `gemm_ownership` | 52 | 52 | 0 | 0 | 104 |
| `full_pair_lookahead` | 46 | 46 | 4 | 0 | 96 |
| `full_serialization_first` | 50 | 50 | 6 | 0 | 106 |

`gemm_ownership` selected four ownership protocols and no active pair.
`full_pair_lookahead` selected three ownership protocols, four direct events,
and four loop-carry drains; it also selected no active pair. This is direct
evidence that ownership/fusion synthesis, not ordinary pair composition, makes
the GEMM coverable.

For GEMM, the local physical-ID instruction distributions were:

```text
gemm_ownership:       ID0=24, ID1=22, ID2=3, ID3=3 sets; waits identical
full_pair_lookahead:  ID0=22, ID1=20, ID2..ID5=1 each; waits identical
```

Validate allocation per directed event domain, not merely by global numeric
ID. Different domains may legally reuse the same ID. The old broken refined
binary's concentration of 110/118 actions on ID0 is not an acceptable
reference.

## Phase 0: host gates

1. Verify exact SHAs, clean trees, submodules, toolchains, and input hashes.
2. Build each revision once using the device host's approved parallelism.
3. On the current candidate run all registered SyncCover/CanonicalSync unit
   tests and all discovered `canonical_sync` lit tests. Record discovered and
   passed counts.
4. Reproduce every current-candidate arm and the aggregate table above.
5. Preserve pre-sync PTO, full argv, stdout/stderr, comparison JSON, post-sync
   PTO, generated backend source, object/binary, and SHA256 values.
6. Require every current full-catalog row to report fresh verification true,
   balanced sets/waits, event IDs in `[0,7]`, and valid allocation.
7. Count body and automatic-tail `PIPE_ALL` separately. Any action-first body
   `PIPE_ALL` is `PLAN_DRIFT` and blocks its device timing until explained.

## Phase 1: structural equivalence

For every compared pair, normalize only synchronization operations and
pass-owned synchronization metadata. Require equality of:

- all non-sync operations and their order;
- addresses, memory spaces, tile types, shapes, and constants;
- branches, loop bounds, recurrence structure, and launch geometry;
- backend lowering path, ABI, and static shape.

A difference is `STRUCTURE_BLOCKED`; do not attribute its runtime to the sync
plan. Cross-revision comparisons need this proof independently from the
within-revision comparisons.

Per canonical row report:

- enabled family mask/list and selection objective;
- candidate and selected mechanism origins;
- active pairs and their extra coverage;
- selected mechanism details and parent/witness provenance;
- sets/waits by directed domain and loop depth;
- barriers by type, pipe, loop depth, and guarded execution condition;
- static and measured-trip dynamic synchronization counts;
- event allocation, maximum pressure, repairs, work budgets, and fallback;
- serialization breadth, event lifetime area, and action profiles.

## Phase 2: silicon correctness

Route A3 and A5 cases to matching supported devices. A missing suitable device
is `DEVICE_BLOCKED`, not a skipped row.

For every arm admitted to timing:

1. use an independent golden that checks every output;
2. use at least three deterministic seeds;
3. restore inputs and sentinel-fill outputs before each launch;
4. exercise startup, parity/alternation, steady state, and drain trip counts
   when the ABI exposes them;
5. run at least 25 repeated launches per seed/shape/device;
6. record complete-output signatures, max absolute/relative error, NaN/Inf,
   unwritten output, timeout, runtime status, and device logs.

For historical GEMM first run M=N=4096, K=512 on devices 4, 5, and 6 if they
remain assigned and healthy, then M=N=K=4096. The old refined plan failed both
shapes on all three devices with exception `507015`. Do not reuse that binary.

Any wrong result, nondeterminism, timeout, hang, or AI-core exception is
`CORRECTNESS_FAILED`; stop timing that arm and preserve the first failing
artifacts. If every arm fails the same frozen input, classify
`HARNESS_OR_DEVICE_BLOCKED`.

## Phase 3: balanced device timing

Time only correctness-closed and structurally comparable arms. Use one quiet
device per paired comparison and the exact binaries validated in Phase 2.

- warm every arm;
- use at least 20 balanced ABBA/BAAB blocks;
- retain at least 50 positive launch samples per arm per block;
- make each arm appear first equally often;
- revalidate output after every block;
- preserve raw samples and the deterministic seeded analysis script.

Report separately:

1. `device_kernel`;
2. `host_resident_e2e` with tensors resident;
3. `host_full_e2e` including allocation, transfers, kernel, and output check.

For every pair report the median difference, ratio, throughput where defined,
and a seeded paired block-bootstrap 95% confidence interval.

Primary current-revision comparisons:

1. `full_pair_lookahead` versus `insert_current` on all 29 runnable cases;
2. `full_pair_singleton` versus `full_direct` on the 16 cases whose local
   output changed, to measure transitive pair value;
3. historical `gemm_ownership` versus `full_pair_lookahead` versus
   `insert_current`;
4. `full_serialization_first` versus action-first on cases 10, 25, and 28,
   plus cases 6 and 23 as high objective-sensitivity controls.

Primary performance-revision comparison:

```text
canonical_performance_revision versus insert_performance_revision
```

Run it on all executable corpus cases, not only the historical GEMM.

## Phase 4: test the PIPE_ALL hypothesis

Static presence alone cannot establish causality. For the performance revision:

1. record each body `PIPE_ALL` anchor, guard, enclosing loop depths, and the
   exact dynamic execution count at the measured shape;
2. stratify paired device deltas into zero-`PIPE_ALL` and positive-`PIPE_ALL`
   cases, then by dynamic `PIPE_ALL` count;
3. report wins and losses in both strata—losses without `PIPE_ALL` disprove it
   as a complete explanation;
4. profile at least one loss with `PIPE_ALL`, one win with `PIPE_ALL`, one loss
   without `PIPE_ALL`, and one matched no-`PIPE_ALL` control;
5. separately profile current cases 10 and 25 under action-first and
   serialization-first. These share one compiler revision and directly expose
   the cost of the latter's localized fallback, although other selected
   mechanisms may also differ.

Collect real in-core evidence:

- total device span and critical path;
- work and idle/stall cycles by CUBE, MTE2, MTE1, V/M, and FIX pipe;
- set/wait stall cycles by directed domain;
- targeted-barrier and `PIPE_ALL` drain cycles;
- overlap lost immediately before and after each dynamically executed barrier;
- non-sync instruction counts and cycles.

Classify the hypothesis:

```text
PIPE_ALL_CAUSAL:       within-case drain stalls explain the loss and matched
                       no-PIPE_ALL controls do not show it
PIPE_ALL_CONTRIBUTES:  drain stalls are material but other placement/coverage
                       effects remain
PIPE_ALL_NOT_PRIMARY:  losses persist without PIPE_ALL or traces identify a
                       different critical-path cause
INCONCLUSIVE:          structural equivalence or profiling cannot be closed
```

Do not infer causality from unweighted instruction count or a cross-kernel
correlation alone.

## Required deliverables

Return one verified archive containing:

- `REPORT.md` with provenance, host gates, structural table, correctness,
  paired timing/CIs, pair attribution, objective attribution, and PIPE_ALL
  verdict;
- `HANDOFF.md` with blockers and exact next commands;
- all frozen inputs and hashes;
- compiler reports, post-sync PTO, normalized traces, backend sources,
  binaries, and logs;
- event allocations and mechanism provenance;
- raw timing samples and deterministic analysis scripts;
- in-core profiles and cleaned traces;
- a SHA256 manifest covering every archive member.

Verify the archive from a pristine extraction. Report missing and unmanifested
file counts, leave source trees clean, and leave devices idle and healthy.

Final verdicts:

```text
SOURCE_PROVENANCE:       CLOSED | BLOCKED
HOST_REPRODUCTION:       CLOSED | PLAN_DRIFT | FAILED
STRUCTURAL_EQUIVALENCE:  CLOSED | BLOCKED
CURRENT_CORRECTNESS:     CLOSED | FAILED | NOT_RUN
PAIR_VALUE:              WIN | TIE | REGRESSION | INCONCLUSIVE
OWNERSHIP_VALUE:         WIN | TIE | REGRESSION | INCONCLUSIVE
OBJECTIVE_VALUE:         ACTION_FIRST | SERIALIZATION_FIRST | MIXED | INCONCLUSIVE
CURRENT_PERFORMANCE:     WIN | TIE | REGRESSION | MIXED | NOT_RUN
PIPE_ALL_HYPOTHESIS:     CAUSAL | CONTRIBUTES | NOT_PRIMARY | INCONCLUSIVE
```
