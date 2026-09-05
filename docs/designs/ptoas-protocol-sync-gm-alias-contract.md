# ProtocolSync GM alias contract

Status: implementation direction agreed on 2026-09-05; not an implemented CLI
option or a target-visibility qualification.

## Two explicit modes

The default is **may-alias**. Distinct GM argument roots may overlap unless
authoritative ranges or an explicit alias contract establish otherwise.

An opt-in **assume-disjoint-arguments** mode promises that distinct GM argument
roots do not overlap. This is a caller/frontend contract, not a conclusion
derived from SSA names. A caller that violates the promise is outside that
mode's supported input contract. The assumption must appear in diagnostics,
reproducibility records and the verifier's semantic inputs.

Neither mode treats separate views of the same argument as disjoint. Casts,
subviews, pointer arithmetic, loop-carried arguments and path joins must retain
their possible original roots. If a root set cannot be recovered, the opt-in
mode must not invent disjointness. The option does not weaken local physical
alias checks, queue ownership, communication, ACC/proxy or cross-core rules.

## May-alias is an obligation, not automatic rejection

For each potentially overlapping access pair:

1. Determine whether its paths and iteration instances can both execute in the
   relevant order. Ordinary read/read pairs need no memory-race repair.
2. Derive the required effect: completion/reclamation/overwrite ordering, or
   write-to-read visibility. Keep alias uncertainty separate from effect kind.
3. Check the complete selected world's transitive, control- and
   iteration-qualified ordering before proposing another synchronization.
4. Repair only requirements that remain uncovered and whose target mechanism
   is qualified. Otherwise retain an explicit unsupported obligation or use
   attributed whole-function fallback.

This preserves the user's expected optimization: local storage and compute
dependencies may already order GM accesses. It does not assume that every
correctly synchronized local pipeline establishes GM publication.

Current F already has part of this behavior: `memoryHazardCovered` consults
the completion graph for uncertain-alias WAR/WAW and the separate visibility
graph for uncertain-alias RAW. The existing linear load/compute/store tests
demonstrate that distinct GM arguments do not universally prevent admission.
The missing work is an explicit alias contract, precise source/root and range
facts, target-qualified visibility supply, and canonical typed obligations.
Relabeling every unknown GM RAW as ordinary completion would bypass the missing
visibility proof and is not the proposed fix.

## Integration and acceptance tests

Implement the contract once in shared semantic analysis, then make residual
interpretation, OneShot/ReadyRelease admission and concrete verification
consume it consistently. The concrete verifier may use the caller's original
alias contract; it must not infer that contract from planner coverage tags.

For both modes, test:

- distinct arguments with a genuinely unordered GM hazard;
- an existing transitive local-dependency chain that already supplies the
  required completion, with no redundant GM event required;
- same-root and overlapping subviews, casts and dynamically selected roots;
- aliasing through branch and loop block arguments;
- write/read pairs with completion but no qualified visibility;
- loop backedges and exit completion;
- identical byte-identical input snapshots and per-kernel overlapping blocker
  matrices, including kernels still rejected for unrelated causes.

Report default-safe and assumed-disjoint admission separately. Neither static
admission nor fewer emitted actions establishes device correctness or speed.
Target publication claims still require PTO-ISA/CA-model evidence and focused
device tests with intentionally aliasing inputs in safe mode.
