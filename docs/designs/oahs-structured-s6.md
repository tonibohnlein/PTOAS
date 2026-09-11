# Structured OAHS S6: complete local refinement and startup-site coalescing

Base: `13cfaf396fc512ee235b2589bd4c15be181e16e7` (integrated S4/S5).
This increment changes the existing `structured` path. It introduces no pass,
selector, general relation backend, or new hardware-ordering premise.

## Scope

S6 implements the complete-local-region refinement and startup-site part of the
S6 implementation brief, with opt-in mechanism accounting. It adds a small,
explicitly synthetic split-K GEMM native regression. It does **not** implement
admission of the frozen historical GEMM. Runtime-strided outer wrappers,
physical-section retirement, last-sensitive original prefetch payloads and
factored independent storage periods remain outside the current fragment.

No UnitFlag enablement, shape-dependent accumulator elision, blanket synchronous
matrix-pipeline rule, expanded event pool, or guessed GM visibility rule is added.
The existing target semantics and original conservative requirements are retained.
The default planner remains unchanged.

## 1. Complete local-region deletion

`refineRegionHandoffs` takes an already constructed and allocated **complete**
local model and its plan. It validates both the endpoints and the complete input
before examining deletions. An invalid plan cannot disappear into an empty
rendered action list on an empty requirement ledger.

There is one reverse sweep over the originally selected handoff population. A
trial removes one complete set/wait family, preserves every surviving endpoint,
key, barrier and relative same-cut order, and invokes `verify` on the full model.
This includes the startup/prelude, periodic tail, exit/empty-tail obligations,
matching and consumption-before-rearm. A pair used only to acknowledge a live
recurring event cannot be removed merely because memory hazards stay covered.

The native adapter applies this only to unwrapped local regions. Dynamic
empty/nonempty startup cases are independently checked in their original guarded
domains. No local projection of an S3 invocation plan is pruned under an old
cross-invocation certificate. Re-entrant plans retain S5 behavior.

Cost is bounded by one verifier call per original handoff plus input validation;
it is not a repeated convergence loop. It can still be expensive with large
dense models. The normal compiler campaign must measure the added cost with
fresh emitted checking enabled; portable suite runtime is not that campaign.

## 2. Original-cut site identities

For a period-one startup model, `SiteOrigin` records whether an atom is the
initial occurrence or a steady occurrence and its **original physical-phase
identity**. Initial and steady operations in different original branches are
not identified merely because their opcode, address, or operands look similar.

A generated command can be emitted once across the two phases only when:

* Both atoms refer to the same original operation and before/after cut.
* They are its exhaustive initial and steady domains, with no enclosing re-entry.
* Kind, direction, key, and side of the cut are identical.
* Participation is `Every`, with no next-use displacement, first/last/existence
  condition, residue selection, or invocation predicate.
* Relative command order in each individual phase is preserved.

A coalesced set is still after that exact original producer, and a coalesced
wait is still before that exact original consumer. No publication is moved after
independent work, and no wait moves earlier. The phase-union command occurs only
when the original operation executes; the original payload/control is untouched.
The dynamic empty/nonempty case predicate, when present, remains explicit.

`startupEmissionSites` builds an ordered alignment at each original cut using a
longest common subsequence of matchable initial and steady commands. This avoids
crossing equal commands and changing a same-key FIFO. Nonmatching commands keep
their phase guards. Each original virtual action appears in exactly one site.
Its cost is the sum of products of the two command-list lengths at each cut,
not a function of the runtime trip count.

## 3. Checked key alignment, separately from deletion

Different initial and steady key assignments can prevent otherwise compatible
commands from sharing a site. `coalesceStartupHandoffs` makes one pass over
boundary pairs. It may adopt a key already used by a distance-zero recurring
handoff of the same direction with at least one identical original endpoint.
No logical handoff is added, deleted, or moved by this operation.

A candidate is considered only if it decreases the emitted site count. It is
accepted only after full-region verification of the new mixed key population.
The original source-prefix and consumer boundary are preserved even when only
one of the pair's endpoints has a common original site. Matching is established
from the full event population, not from equal numeric keys alone.

The native adapter explicitly allows S4's mixed boundary/periodic proof mode for
this optimization even when the initial allocation fitted without sharing. This
is an explicit realization-policy change, not a claim that hardware capacity
forced it. The emitted checker uses the same policy and reconstructs the actual
commands independently. Unsupported candidates simply retain the original keys.

## 4. Emission and reconstruction

The emitter uses `EmissionSite` only as an emission plan. A two-member group emits
one command and drops only its complementary initial/steady predicates.

Reconstruction does **not** consume group metadata. At an original body cut, an
absent phase guard is accepted only if re-extraction finds exactly one initial
and one steady atom at that original operation, period one, no wrappers, and no
other participation predicates. It expands the actual command back into those
two occurrence domains, then runs the complete checker and the original-cut/
same-boundary-order fingerprint check. A missing guard at a singleton site, or a
changed first/last/existence predicate, is not accepted as coalescing.

The existing clone/verify/snapshot/physical-retranslation transaction is retained.
A malformed or mutated output does not replace the original function. Static
set and wait **site counts** may now differ: an initial and a steady set can share
one site while mutually exclusive waits remain at different original operations.
Correctness requires dynamic participation, not equal static instruction counts.

## 5. Diagnostic mode

Set `PTOAS_STRUCTURED_PLAN_JSON=1` to print a line prefixed `OAHS_PLAN ` with schema
`oahs.s6.plan.v1`. Reports are produced only after emitted reconstruction and
physical-preservation checks. They include:

* Original operation ids, lanes, roles, may-access ranges and overlap-relevant
  source/target facts; geometry is encoded as decimal strings.
* Immutable local/carried requirements, occurrence distances, selected command
  participation, event directions/keys, and static-site membership.
* For each local handoff, original requirement indices that lose a completion
  proof upon deletion, and a separate event-protocol result with the memory
  ledger cleared. These are proof diagnostics, not minimality certificates.

The report exposes source/target physical inventories rather than inventing a
unique byte-level hazard witness for an aggregated requirement. It is not a new
exact-value-flow analysis. Re-entrant deletion audits are explicitly not run.

`test/experiments/insert_sync/logical_plan/s6_report.py` validates accounting,
exhaustive site membership, equivalence and phase order. It is **not** a second
hardware or semantic verifier. The native gate compiles a separate audited copy
and requires identical output bytes and matching static instruction accounting.
Diagnostic audit work is intentionally excluded from compile-time samples.
Both the native gate and paired benchmark runner clear diagnostic/trace environment
variables from their normal timed children.

## 6. Evidence and acceptance

The portable suite retains S1-S5 tests and adds 129 complete-region/startup models,
including a hand-transcribed Q-projection footprint model. It tests full deletion,
key alignment, original-cut grouping, independent finite asynchronous checking,
invalid plans, crossing site orders, distinct original branch identities and an
event-only acknowledgment that must remain.

For the Q-projection **portable model**, the result is:

```
S5-shaped construction: 35 pairs + 7 named barriers + 1 drain = 78 sites
Full-region deletion:   34 pairs + 7 named barriers + 1 drain = 76 sites
Startup coalescing:     13 shared command sites, one key alignment = 63 sites
```

At 32 total original iterations, the model's executed commands change from 1204
to 1144 through deletion. Coalescing reduces static sites and generated phase
tests, not the dynamic synchronization commands represented by its members.
These are not native importer or device measurements. The native gate requires
unchanged Q projection to improve the 78-site baseline (at most 76 sites) and to
actually coalesce sites; it does not assume exact agreement with the hand model.

All seven original population cases remain in the native gate. Added native
mutations narrow a shared command to initialization, duplicate it, and move a
publication after another load. All must be rejected atomically. Existing
first-panel-prefix, ordinal, startup, boundary and invocation tests remain.

The new `structured_inputs/s6_startup_gemm.pto` is a synthetic ordinary split-K
matrix multiply with M=16, N=256, K=2048 and 32 K panels. It uses the unchanged
raw loop `0..64 step 2`, first-iteration initialization, L1/L0 operand reuse and
final output. It is neither a substitute nor a relabeling of the historical
GEMM. The new native test checks preservation, participation and reconstruction;
it is not a numerical device test. The original recovery utility continues to
pin the historical source for the later admission milestone.

Patch-preparation validation is recorded in the distributed package. The
portable builds, finite-oracle runs and Python utility tests do not substitute
for a pinned MLIR native build, seven-input native gate, paired whole-compiler
campaign, or device correctness/performance qualification. Run those before
promoting the implementation or reporting a native Q-projection/GEMM result.
