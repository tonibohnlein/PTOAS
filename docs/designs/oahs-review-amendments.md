# Exact proposal words and replay comparison cost

Implemented locally on `6b1b32f46`, 2026-09-21. These are two bounded fixes to
existing construction and replay, with no new policy mode or hardware premise.

## Exact proposal words

The old recurring candidate appended requests in order, while guarded commitment
partitioned publications before acquisitions. A return that followed a forward
WAIT in the checked candidate could precede it after commitment and lose the
consumption evidence needed for rearming. Final validation remained mandatory;
no unsafe native emission was demonstrated.

`recurring()` now materializes canonical ordered endpoints first. Mandatory
protocol checking, producer-support and resource admission read those words.
Omission trials use the same materializer; accepted trials retain their checked
endpoint list. Commitment assigns channel IDs without changing endpoint order.
The existing ledger remains ahead of the appended proposal.

The linked regression has two reciprocal exchanges on one key per direction.
Both the causal checker and independent command graph accept WAIT-before-return
and reject the publication-first mutation. The optional constructor now rejects
that mutation before changing ledger version, endpoints, reservations, channels
or contextual-replay state. Ordinary fallback succeeds. Positive commitment and
accepted omission tests check endpoint order and request-to-channel identity.
Linking this regression against the old production `CyclicFrontiers.cpp` fails
at the expected mandatory-admission assertion.

## Replay comparison cost

The sibling invalidation walk is unchanged. Normal certified sibling replay no
longer computes the old reusable prefix solely to report how many additional
components were reused. That computation remains available with `--trace-replay`
and remains authoritative for `--prefix-replay` and ordinary prefix replay.

`prefix_queries` and `prefix_span_examinations` charge the actual legacy work.
`sibling_comparison=0` means the extra-sibling comparison count was not measured;
it does not mean no sibling reuse occurred. Total reused components and causal
evaluation counts retain their meaning.

On the linked 512-word chain, explicit comparison examines 526,338 spans; the
normal invalidation walk visits 1,026 sites and performs no legacy-prefix query.
Complete checkpoints and endpoint aggregates match traced, untraced, prefix-only
and cold execution. The existing sibling tests also cover real events, shared
words, deletion, failure and recovery in both tracing modes. Restoring the old
unconditional call makes the new regression fail.

## Validation

- 23/23 portable suites, built with strict warnings; no sanitizer run.
- Native diagnostic suite and both retained-input native/FileCheck fixtures pass.
- Serial repository hooks and `git diff --check` pass.
- 88/88 corpus modules (97 functions) construct and reconstruct successfully.
  Every output is byte-identical to the retained-cohort baseline at `6b1b32f46`.
  Selected updates, causal replay, recurring/trial and rejection counts match.
- Traced and untraced partial attention emit identical PTO:

| Function | Legacy span examinations, traced | Normal mode | Causal evaluations, both modes |
| --- | ---: | ---: | ---: |
| Partial AIC | 24,827 | 0 | 102,810 |
| Partial AIV | 177,875 | 0 | 608,848 |

These are compiler-work counts, not wall-time or device speedups. No device
measurement was run; the corpus plans are unchanged. The source-complete device
packages already dispatched remain immutable.

Artifacts are under `/home/toni/work/pypto3_sync_more/review-fixes-work/`: build/test/hook logs,
linked negative controls, source hashes, serial corpus reproducer/results, and
traced partial-attention output. The general frontier-motion certificate and
conditional future-key-reuse test remain separate follow-ups.
