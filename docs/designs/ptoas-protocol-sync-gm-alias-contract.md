# ProtocolSync GM alias contract

Status: the explicit argument-root contract is implemented and host-tested.
General may-alias repair, access-region precision and target visibility
qualification remain separate, incomplete work.

## Two explicit modes

The default is **may-alias**. Distinct GM argument roots may overlap unless
authoritative ranges or an explicit alias contract establish otherwise.

An opt-in **assume-disjoint-arguments** mode promises that the sets of bytes
accessed through distinct GM argument roots are disjoint, including derived
views and pointer offsets. Different base-pointer numbers alone do not satisfy
that promise. This is a caller/frontend contract, not a conclusion derived
from SSA names. A caller that violates the promise is outside that
mode's supported input contract. The assumption must appear in diagnostics,
reproducibility records and the verifier's semantic inputs.

Neither mode treats separate views of the same argument as disjoint. Casts,
subviews, pointer arithmetic, loop-carried arguments and path joins must retain
their possible original roots. If a root set cannot be recovered, the opt-in
mode must not invent disjointness. The option does not weaken local physical
alias checks, queue ownership, communication, ACC/proxy or cross-core rules.

### Compiler and IR interface

```bash
ptoas --pto-arch=a3 --pto-level=level3 --protocol-sync-mixed \
  --protocol-sync-fallback=fail --protocol-sync-gm-alias=may-alias kernel.pto
ptoas --pto-arch=a3 --pto-level=level3 --protocol-sync-mixed \
  --protocol-sync-fallback=fail \
  --protocol-sync-gm-alias=assume-disjoint-arguments kernel.pto
```

The pass option is `gm-alias`; both A2 and A3 use the shared policy. The CLI
option requires an active ProtocolSync mode. A function can explicitly record
the semantic contract as `pto.gm_alias = "assume-disjoint-arguments"` or
`pto.gm_alias = "may-alias"`. Without an option override, a valid recorded
contract is inherited; without either, the mode is may-alias. An explicit
option overrides a valid recorded contract. Invalid strings and non-string
attributes fail, even when an override is provided.

Analysis-only mode does not modify the input attribute or body. Emission
records an explicit override on the disposable clone, and transfers it to
the original function only when the entire module transaction succeeds.
Concrete verification inherits this semantic input, not planner coverage
tags. An explicit may-alias override can audit emitted IR under the stronger
aliasing requirement; an earlier assumed-disjoint acceptance is not evidence
that such an audit must pass.

Root recovery follows supported pointer/view forwarding, pointer-to-pointer
casts, `arith.select`, both arms of `scf.if`, and both the initial and yielded
values of `scf.for` arguments/results. Cycles use a visited-value fixed point.
Pointer/integer round trips, opaque definitions, unsupported forwarding and
traces exceeding 256 values remain incomplete. Disjointness requires two
complete, nonempty root sets with no common argument. This does not recover
exact byte ranges or compare offsets relative to unrelated GM bases.

Schedule dumps include the effective `gm-alias` mode and recovered `gm-roots`
with a completeness flag. Statistics include `gm_alias_mode`. Campaign
metadata and rows record the requested override independently of the effective
per-function mode.

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
The explicit alias contract and bounded source/root recovery are now shared by
residual interpretation, OneShot, ReadyRelease and concrete verification.
The missing work remains precise ranges, target-qualified visibility supply,
canonical typed obligations and broader native repair.
Relabeling every unknown GM RAW as ordinary completion would bypass the missing
visibility proof and is not the proposed fix.

## Integration and acceptance tests

The contract is implemented once in shared semantic analysis. The concrete
verifier may use the caller's original
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

### Current regression evidence

`protocol_sync_gm_alias_contract.pto` checks strict direct/mixed/OneShot
emission of two disjoint-argument copies, default-safe rejection of their
possible intervening GM write/read hazard, recorded-contract inheritance,
safe override of emitted IR, read-only analysis, invalid contracts and
module-transaction rollback. `protocol_sync_gm_alias_roots.pto` checks
same-root hazards, branch unions, loop init/yield unions and self-carried
fixed points, dynamic pointer/view forwarding, pointer casts and integer
round-trip rejection. Existing linear OneShot/direct tests continue to pass
under the default may-alias policy.

These are host admission/provenance regressions, not a new device campaign
or proof of general symbolic-range and recurring-GM synchronization.
