# Qualified pending-child composition receipt

The fifth construction-first milestone accepts private wrappers with no local
event commands whose selective descendants already own closed physical resources.
Such a wrapper may return with pending payload effects. Its parent imports those
effects through the existing SCC-local completion transfer; importing the child
does not add a boundary drain.

Matching closure, completed payloads and resource closure are separate facts.
`Empty`, `ClosedDescendants` and `CompletedExit` identify the supported resource
ownership cases. The original completed-exit subclass remains available.
Qualification refuses a wrapper that would need an internal completion command,
local event ownership, recursive calls or an unsupported child contract. This
receipt does not accept arbitrary pending children with local event resources.

A qualification request detects recursion and memoizes shared descendants. Its
owned summary retains descendant proofs, and validation checks both the original
wrapper and its descendants for mutation. Module coordination freezes qualified
bodies and rejects incompatible callers. No runtime dependence analysis or
additional ordering at a child boundary is introduced.

Correctness and architecture reviewers accepted this scope after reviewing the
resource ownership, SCC transfer, refusal paths and independent test oracles.
The source-pinned incremental build completed with two aggregate workers. On that
build, `sync_closed_completion.pto`, `sync_completion_children.pto` and
`sync_completion_transfer.pto` passed. Tests cover pending-state and command-cut
oracles, guarded/nonunit invocation, shared descendants, mutation, recursion,
local-event refusal and preservation of completed exits.

Execution receipts are in `.local/milestone-loop/upgrade/`: the source and tool
manifests pin this build, and `focused.out` records all three passes. The broader
21-test focused run has 15 passes and six failures in other active construction
and lowering changes; it is not a green overall gate. Production corpus refresh
and broad validation remain part of milestone six. This receipt establishes
qualified implementation acceptance, not successful device compilation or a new
allocation result for GroupNorm, prefill, decode or Delta.
