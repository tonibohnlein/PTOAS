# Reused keys at an early source gap

Local continuation of `d1bf07ee55025d27bc66f508a3fa46727af7ffd0`, 2026-09-21.
The source-gap option remains disabled by default. This extends its existing
word-start certificate; it adds no completion state, observation or key pool.

## Reproduced missed placement

The linked constructor regression uses one key per direction and disjoint
physical X, Y, Z and W cells:

```
Q writes W
P reads W                 // first Q -> P key generation is consumed
R reads Y
P writes X
Q copies X into Z
Q overwrites Y
P overwrites X            // needs another Q -> P transfer
```

P -> Q readiness for X can carry the earlier Q -> P consumption back to Q.
At the post-copy source gap, Q can therefore publish X's release before the
unrelated R -> Q acquisition protecting Y. The original virgin-only query
refuses that binding and appends the release after the Y acquisition.

Before the change, `reusedSourceGap()` fails with
`reused-key X release imported unrelated Y completion`. After the change, the
actual selected plan reuses key zero and removes four strict payload
launch/finish relations (67 -> 63), adding none. Removing actual P -> Q
support makes both the production causal checker and independent graph reject.
This is a portable constructor result, not a native speedup.

## Certificate and binding

`reusableAtStart()` requires:

1. A straight acyclic source-to-consumer corridor and unique source/target words.
2. A directionally eligible key outside closed and recurring reservations.
3. `canPublish()` in the **incoming state before the source word**: the key is
   empty and its preceding consumption is known at the publishing engine.
4. Every active selected endpoint on this key is strictly before the source
   gap on a straight acyclic corridor, with a unique word. A use in the source
   word itself or any later/branch-shared use rejects this candidate.

The last condition is a conservative neighboring-use certificate. It avoids
moving a new publication across an existing selected generation. It does not
implement general key lifetime reuse across branches or recurring scopes.

Coverage and physical binding use the same word-start state. Binding rechecks
the ledger version, eligible key and actual required coverage before insertion.
The publication is prepended; its acquisition stays at the consumer deadline.
Normal replay, finalized-requirement checks and the final cold certificate
remain authoritative. No future return or post-word receipt supplies credit at
an earlier gap.

The query scans active ledger records for each eligible key, as the previous
virgin-only test did. It adds no candidate solve or deletion trial. This is not
a compiler-time optimization or a general optimal-allocation claim.

## Validation

- Five replay-derived boundary cases: prior real return; absent return; return
  inside the source word; selected next generation inside that word; selected
  next generation after it. Only the first is admitted.
- One-key linked positive, complete payload-relation subset check, and removed
  support negative using both causal and independent graph checks.
- Existing virgin-key/outward-export tests and 200 finite acyclic option cases.
- 23/23 portable suites passed; focused placement suite rerun after extending
  the common-frontier export-position test to nine cases.
- Native selected diagnostics passed. Default and `--source-gaps` each
  construct/reconstruct down, gate/up, LM head, post-RMSNorm, partial attention,
  both single-block attention variants and Shenggan GEMM: 16 module runs,
  22 function constructions/reconstructions. Every plan is byte-identical to
  the saved `d1bf07ee5` output. This is a targeted set, not a new full-corpus run.

Artifacts: `../source-reuse-quality-work/` relative to the checkout, including
`placement-tests.log`, `core-tests.log`, `native-tests.log`, `corpus/`, `gemm.json`
and the diagnostic attention traces. No device task is restarted; the dispatched
candidate remains frozen. Interior gaps, reused keys with later selected uses,
recurring gaps and a useful changed native workload remain follow-ups.
