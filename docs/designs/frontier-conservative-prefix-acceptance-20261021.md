# Conservative prefix services: review and commit acceptance

The jointly certified pruning checkpoint and its subsequent typed production
integration were reviewed together against `7facb6ea859ca06399f02e8438fd8aaa6aebad64`.
An independent synchronization-correctness review and a separate architecture/
generality review both accept the changes. No blocking findings remain.

The reviews covered full-source-pipe eligibility, block-local proposals,
ordered service deletion, joint raw-storage delivery certification, protected
scalar/control paths, typed SCF input, whole acknowledgment quartets, constant
directional roles, causal rearming, native/visibility/readiness qualifications,
reserved directional pools, baseline retention and module-owned rollback.
They confirmed that both live entry points use the same typed library service,
that JSON does not select live masks/roles/cuts, and that superseded driver
implementation bodies were removed. No kernel-name recognition, benchmark
trip bound or inferred caller restriction was found.

The reviewers performed read-only source and receipt audits; they did not claim
fresh independent compiler/device runs. The implementation's source hashes
were rechecked against its receipt before packaging. Existing serial validation
is reused because review required no source fix:

- Fresh **34/34** focused tests for the exact intermediate pruning snapshot.
- Fresh **35/35** focused tests for the final production tree, including
  compact-K/replay, zero-inner-body, child-import and scalar/accumulator checks.
- Actual emitted-command checks on 18 fixtures and 32 Delta scenarios; matching,
  modeled coverage, acyclicity and causal physical reuse pass.
- Delta reproduces 933→577 commands with identical checked payload order.
  The disjoint-buffer fixture separately demonstrates strict order containment
  and completion-to-start excess 11→7.
- All 12 default artifacts for the three original kernel checks are byte-identical
  to the immediate baseline; separately pinned C++ generation passes.
- Ruff F checks and `git diff --check` pass. The recorded brace-prefilter findings
  are individually reviewed false positives, with no unresolved finding.

Commit packaging preserves two independently qualified states: optional
pruning first, then shared typed ownership and opt-in production integration.
The preserved intermediate source snapshot supplies the first package; the
final reviewed sources supply the second. Historical implementation reports
retain their uncommitted-checkpoint identities rather than being rewritten as
post-commit validation.

Acceptance is limited to the documented standalone ownership and supplied
modeled-access/native contracts. The production option reports conservative
coverage, not minimum demands or realization-order equality. Retained-child
composition, automatic fallback selection and device qualification remain
outside this checkpoint. Allocation algorithms, expert protocols, established
hardware exemptions and default dispatch are unchanged. No push is requested.

Evidence: [pruning report](frontier-service-pruning-20261021.md),
[production report](frontier-conservative-prefix-production-20261021.md), and
[production receipt](frontier-conservative-prefix-production-20261021.json).

Commit packaging removes one extra blank line at the end of the pruning report.
The historical receipt retains the hash of that report before this whitespace-only
normalization; all reviewed implementation bytes remain unchanged.
