# Prefix-child composition acceptance

Reviewed the changes from `5af91f65434fb4dc0ea8f909c7bf50fd2d0ffa30`
for the explicitly requested, opt-in import of one qualified eventful whole
counted-loop child into a conservative prefix parent. The implementation and
its validity argument are in
[the construction report](frontier-prefix-child-composition-20261022.md).

Two independent read-only reviews accepted the change with no blocking
findings. The correctness review followed whole-loop ownership, original
coordinates and guards, every grouped record label, assignment equivalence,
directional palette ownership, causal entry handshakes, pending effects and
module rollback. The architecture review confirmed that production contains
no kernel-name or benchmark-bound special case, new allocator/backend,
automatic selection or downgrade of retained exact mathematics.

The correctness reviewer suggested labeling standalone Qwen receipt rows by
caller applicability. That metadata-only change is included and was accepted
on follow-up review: multiple outer visits are structural reset stress with
unqualified caller/layout applicability; the pinned original caller and bounded
zero/short layout derivatives are distinguished. No production behavior,
validation assertion, watchdog or tested configuration changed.

The implementation campaign passed 36/36 focused tests and two separate
supplementary tests. Acceptance validation reran `sync_prefix_children`,
`sync_closed_child_adapter` and `sync_scalar_shared_contracts` serially after the
receipt-label correction: 3/3 pass. These are separate runs, not a claim of a
new full-suite campaign. The reviewers inspected source and recorded evidence;
they did not independently execute builds or tests.

Acceptance commands from the worktree root:

```bash
source .local/prefix-child-composition-20261022/linux-env.sh
.venv/bin/python /home/toni/work/llvm19/llvm-project/build-shared/bin/llvm-lit -j1 \
  --filter '(sync_prefix_children|sync_closed_child_adapter|sync_scalar_shared_contracts)\.pto$' \
  -o .local/prefix-child-review-20261022/final-gate.json \
  .local/section8-refactor/source-pinned/candidate-build/test/lit
ruff check --select F test/lit/pto/Inputs/check_prefix_children.py \
  test/lit/pto/Inputs/check_prefix_qwen_child.py
git diff --check
pre-commit validate-config .pre-commit-config.yaml
python3 /home/toni/work/pypto3_sync_more/PTOAS-handoff/.agents/skills/enforce-ptoas-code-compliance/scripts/check_changed_code.py \
  --repo . --base HEAD
```

The compliance prefilter returns 16 brace-rule matches, all reviewed false
positives caused by nested calls/conditions with braced bodies. The evidence
lists each source context; no suppression was added. Ruff and whitespace
checks pass. Pre-commit configuration is valid with its inherited deprecated
OAT stage warning; no Git hook is installed in this worktree's common Git
directory. No hook was bypassed. Full analyzer, sanitizer and complexity metric
qualification are not claimed.

The companion machine-readable receipt pins the dirty source, Linux build,
Python environment and original campaign. Its `acceptance_review` entry records
the additional review and fresh test evidence. Local review artifacts are in
`.local/prefix-child-review-20261022/`; source/build campaign artifacts remain
in `.local/prefix-child-composition-20261022/`.

Acceptance retains the documented limits: matching and within-visit assignment
remain qualified provider contracts; one whole `scf.for` is supported; the
parent still inspects its static child CFG; child-aware pruning and refreshed
baselines are refused. Static control and compilation cost grow. The measured
ordering improvement over the previous prefix route does not imply dominance
over the qualified completion route. This is host-model validation, with no
device qualification.
