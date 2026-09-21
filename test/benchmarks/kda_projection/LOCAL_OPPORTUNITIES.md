# KDA local synchronization opportunities — 2026-09-21

## Initial diagnostic: separate the two operand-bank readiness deadlines

On the rebased branch's default OAHS plan before the choice-frontier change, the first prologue extracts
bank A, then bank B, then publishes one broad MTE1 readiness receipt before the
first conditional matrix operation. That operation reads only A. The second
matrix operation reads B, and needs its own readiness if A's receipt moves early.

A diagnostic fixed-plan transformation preserves all payloads and existing
reuse transfers:

1. Publish MTE1->M key1 after A's two extracts, before B's first extract.
2. Retain the first matrix operation's key1 acquisition in both original arms.
3. Replace the old late publications with key3 publications for B.
4. Acquire key3 at the second matrix operation, preserving its M barrier.

The imported native Program and actual changed command words pass `verify()`.
Deleting the B acquisition fails event validation; deleting the entire B
transfer fails uncovered-completion validation. Thus B readiness is not silently
lost. This is a diagnostic protocol, not yet constructor-selected output or
native emission/reconstruction qualification of an implemented optimization.

Independent ordinary issue/completion graph, exact KDA scalar bindings
arg16=67, arg17=67, arg18=67, arg19=0, arg20=5:

| Metric | Default | Split prologue readiness |
| --- | ---: | ---: |
| Payload operations | 1,727 | 1,727 |
| Strict payload relations | 5,953,709 | 5,953,693 |
| Relations removed / added | — | 16 / 0 |
| Executed event pairs | 1,453 | 1,455 |

The graph checks matching, emptiness and consumption-before-republication.
Queues remain opaque; no peer progress, hidden lowering effects, target ACC
ordering, or latency is inferred. This first probe changes only the first
prologue. It demonstrates avoidable OAHS serialization; existing already has
separate readiness in this prologue, so it does not establish superiority to
existing or explain the complete device latency gap.

## Implemented: original-choice consumer frontiers

Ordinary construction now selects an earlier source prefix when both original
choice arms start with consumers requiring that completion. The receipt is
placed once at the original choice entry. This uses existing control, physical
access and source-frontier views; it introduces no guards or observation product.
See [qualification and tests](../../../docs/designs/oahs-choice-consumer-frontiers.md).

The constructor improves **two** KDA boundaries, including the first prologue
above. Native construction and emitted-command reconstruction both pass.
For the same finite bindings:

| Metric | Choice frontier disabled | Enabled |
| --- | ---: | ---: |
| Payload operations | 1,727 | 1,727 |
| Strict payload relations | 5,953,709 | 5,953,677 |
| Relations removed / added | — | **32 / 0** |
| Executed event pairs | 1,453 | 1,457 |
| Original analysis sites | 679 | 679 |
| Selected updates | 183 | 187 |
| Replay site evaluations | 184,538 | 189,813 |
| Choice staged checks / site evaluations | 0 / 0 | 2 / 5,144 |

Choice preparation visits 140 sites on AIC. These are compiler work counts,
not a host timing comparison. The finite check includes event rearming and
complete explicit payload relation sets; its queue/target limitations above
still apply. This is not a general least-order guarantee or a device speedup.
The unfinished first-write experiment is disabled in both arms.

All 25 portable suites and both native suites pass. A paired comparison on
88 historical corpus modules passes construction and reconstruction in both
arms, with all plans byte-identical. Archived scalar load/store syntax is
migrated identically, and input/binary hashes are recorded. The KDA input is a
separate changed native witness; the unchanged corpus is regression evidence.

## Redundant commands are not necessarily lost overlap

Offline diagnostic group deletions on the same imported native program:

| Deleted barrier group | Native fixed-plan verification |
| --- | --- |
| All MTE1 barriers (26 analytical entries, 14 emitted sites) | Pass |
| All MTE2 barriers (11 analytical entries) | Fail: uncovered original completion requirements |
| All M barriers (25 analytical entries) | Fail: uncovered original completion requirements |

The accepted MTE1 deletion removes 196 executed barriers on the finite trace,
but removes **zero** payload relations and adds zero. Completion is already
implied by other selected paths. This may reduce command overhead; it is not
evidence of recovered pipeline overlap. No deletion is installed in production.

## Remaining opportunities

The implemented choice rule preserves separate producer prefixes and consumer
deadlines. Broader control corridors and tighter key capacity still use ordinary
fallback. The unfinished first-write refinement remains separate and still fails
full-KDA rearming validation; it is not promoted by this change.

Other leads remain unproved: acquiring inactive FIX completion only at the first
conflicting ACC write, and constructing complete MAT readiness/reader-return
support across prologue/body/tail before local MTE2 repair. Keep their required
paths until a complete replacement is checked.

## Reproduction and provenance

Local artifact directory: `../kda-first-write-work/` from repository root.

- `bank-split-probe.cpp`, `build-bank-split.py`, `bank-split-probe.log`: current
  native importer/constructor and fixed-plan verification, including negatives.
- `kda.base.pto`, `kda.bank-split.pto`, `kda.bank-split.order.json`.
- `kda.no-mte1-barriers.pto`, `kda.no-mte1-barriers.order.json`.
- `kda.default.latest.pto`: regenerated current default, byte-identical to base.
- `compare_order.py` in this directory supplies the independent finite check.

Input is the archived KDA prepared input with eight obsolete scalar-load
spellings migrated to current syntax, identically for all arms. These are local
host experiments, not new device results. The artifacts are local, not a portable
committed device bundle.

Implemented milestone artifacts in the same local directory:

- `kda.choice-final.pto`, `kda.choice-final.log`, `kda.choice-final.order.json`.
- `choice-core-tests.log`, `choice-native-final-suite.log`,
  `choice-native-analysis-suite.log`.
- `choice-corpus-final/summary.json` and per-arm plans/logs, produced by
  `check_corpus.py` with the same frozen driver and baseline flag
  `--no-choice-consumer-frontiers`.
