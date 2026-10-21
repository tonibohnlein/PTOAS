# Experimental conservative services: commit acceptance

The source-pipe prototype, scalar native-order refinement and contract-based
operation admission were reviewed together against
`29cd96a2af94fd9d9b5c5711561a0222df67b37a`. Both the implementation review and
an independent correctness/generality review accept this package for its
documented experimental scope. No blocking findings remain.

## Reviewed contracts

The review covered source-pipe projection, IN-before-refresh selection,
class-wide KILL/GEN transfers, structured control, effect coverage, synchronous
scalar results and the qualification of every scalar phase before native
prefix discharge. It also covered separate GM read-visibility admission,
directional forward/acknowledgment roles, ordered command placement,
transactional staging, matching and causal physical-ID reuse.

The services depend on supplied effect and target contracts. They contain no
kernel-name recognition, benchmark-derived bounds or ID-pool increase. Scalar
and qualified accumulator protection retain their existing scope; no blanket
FIX-pipe exemption is introduced. Production dispatch and allocation algorithms
are unchanged. The experimental emitter is a standalone whole-invocation
construction, not a new production fallback or general physical-child import.

## Validation accepted

- The final source-pinned focused gate passes **33/33** tests, including
  compact-K/replay, zero-inner-body, completion-child, scalar and accumulator
  protection checks. This completed campaign is reused; acceptance did not
  rerun the unchanged gate.
- Recurrent pointer-load examples retain runtime control values and pass
  bounded access-coverage, matching, acyclicity and causal-reuse checks.
- Original Delta emits physical IR and C++ under `may-not-alias`, with the
  required distinct-root caller contract and coherent-input entry premise.
  Under `may-alias`, three scalar GM reads retain explicit unproved visibility
  obligations. This refusal is not a capacity or infeasibility claim.
- Nine production kernel/policy cases succeed at four retained stages. Against
  the immediate scalar checkpoint, 32/36 outputs are byte-identical. The three
  differing Persistent GEMM PTO artifacts agree under the documented scoped
  conjunction comparison; the differing C++ has no separate equivalence proof.
- During acceptance, Ruff F checks pass for all eight changed Python helpers,
  all twenty changed source/build files have the required license headers,
  and `git diff --check` passes. The existing brace-prefilter receipt records
  25 individually reviewed false positives and no unresolved finding.

Independent checks consume declared layouts or retained reservations as stated
in their receipts. They do not establish native footprint accuracy, numerical
Delta behavior or device correctness. No device compilation/execution,
sanitizer or full static-analyzer qualification is claimed.

## Remaining nonblocking work

Add the alias-policy comparison for `--current-cut-contracts` to lit. It is
currently covered by the pinned fresh checkpoint runs, while the operation lit
driver exercises emission and its admission refusals. This is a diagnostic
regression improvement, not a missing correctness premise for this acceptance.

The earlier reports describe their historical uncommitted checkpoints; their
source and artifact hashes are preserved rather than rewritten during commit.
Evidence and reproducible commands are in:

- [Source-pipe prototype](frontier-conservative-services-prototype-20261021.md)
- [Scalar native order](frontier-scalar-native-order-20261021.md)
- [Operation admission](frontier-operation-admission-20261021.md) and its
  [machine-readable receipt](frontier-operation-admission-20261021.json)
