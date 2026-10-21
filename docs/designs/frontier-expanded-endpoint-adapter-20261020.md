# Stage 4: original-coordinate endpoint adapter

The stage-1 requests reach the same unconditional rejection as Elementwise:
`prepareFiniteGuardedLogicalInsertion()` rejects `expandedProgram`, although it
retains exact fixed original coordinates, presence, reduced records and legal
original payload cuts. This is an implementation/export gap, not new mathematics.

The proposed repair reuses `FiniteGuardedState::prepare`, the shared direct
publication policy and the existing contextual family compiler. Each expanded
record keeps its static label and zero local ordinal; the original source and
consumer points select **their own** fixed coordinates. Enclosing invocation
identities remain the existing composition wrappers' responsibility. No runtime
loop is expanded further and no new allocation construction is introduced.

Before accepting a recipe, require every loop internal to the selected original
region to have exactly one consistent fixed coordinate for that site. Require
those loops to enclose the original payload, retain source/target cuts, and
establish predicate availability independently at each cut. Unsupported negative
points and missing frames refuse explicitly. The source point predicate and
consumer point predicate are different expressions; a source coordinate must not
be evaluated in the consumer's scope. Static record identity, not equal source
and consumer IV values, establishes matching. Original parent control remains.

Publication uses the existing policy, including its selected-single-publication
and terminal-history checks if early publication is needed. Actual command
activity includes original point selection. Allocation activity is the uniform
occurrence predicate returned by that policy, not the current-cut IV test.
Retained mathematics, queries and successful storage domains survive failure.
The allocator and all existing barriers are unchanged.

Validation must compare the actual emitted commands with original occurrences,
including skipped prefetch and enclosing resets; it must not infer correctness
from successful preparation alone. Physical certification is a separate stage.


## Implementation and validity

`FiniteGuardedAnalysis.cpp` exposes qualified prepare/filtered-prepare callbacks
on the retained expanded regional view. `FiniteGuardedInsertion.cpp` consumes
the original program's fixed coordinates instead of rejecting every expanded
result. It rejects foreign or missing frames, duplicate/negative points and
predicates requiring unqualified intra-visit history. A failed preparation does
not replace retained mathematics. This is an export adapter for an already
bounded expansion; it neither raises its limits nor makes expansion compact in
trip count.

`originalFixedPointGuard` in `Shared/RegionalEvents.cpp` checks that each frame
properly encloses its cut and constructs exact own-side IV equalities. Within
the producer-certified counted-loop domain, the selected fixed tuple denotes
one occurrence; every other visit's equality is false. The source and consumer
use their own tuples, while the retained record label and existing namespace
and enclosing-invocation wrappers retain matching identity. These equalities
are command-site selection, not extra physical-ID tuple coordinates. Original
control, cuts and within-cut order remain authoritative. Availability and
publication policy still qualify separately.

Independent full-parent checking found a second, shared adapter defect:
`GeneralArithmeticEndpointRecipeBuilder` retained a fixed first-iteration
consumer in its mathematics but omitted its own-side point mask when emitting
a sequence crossing WAIT. One SET consequently had multiple WAITs. The repair
uses the same checked point service for arithmetic/DBM endpoint pieces and
retains source/target coordinate metadata. The rejected intermediate plan and
trace are kept under `.local/compact-class-checkpoints/stage4/rejected-intermediate`;
compilation success for that intermediate is not acceptance evidence.

No allocator, SCC transfer, recognition backend, expert protocol or barrier
policy changed. Unsupported storage selectors still refuse separately.

## Independent checks and outcomes

Six generic initialized load/add/store fixtures cover zero/one/two visits,
nonunit starts/steps, nested frames and reset reuse. The actual retained covers
are independently compared (2, 5, 5 and 11 in the nonempty cases), together with
all four event-kind queries and emitted logical matching/coverage. Missing,
duplicate and negative coordinates and changing per-visit filter history
refuse; the original owner remains reusable after refusal. Empty filters
produce an empty inventory.

Twenty production/derivative checks cover Elementwise and GroupNorm under both
alias policies, zero/singleton/short/original trips and the taken/skipped lane
branches. The full kernels are compiled with their original hardware-coordinate
inputs; coordinates are bound only for checking the resulting commands.
Independent byte intervals and original scalar dependencies establish coverage,
logical matching and acyclicity. The possible-alias upper graph treats distinct
GM roots as potentially overlapping without comparing their relative offsets
as absolute addresses. Scratch footprints remain supplied allocation enclosures.
Both ports require disjoint GM arguments; the additional Elementwise C=A check
is explicitly an out-of-port structural stress, not an admitted ABI execution.

Seventeen of these twenty cases receive and independently pass six-ID causal
reuse checks. Three early-specialized derivatives (Elementwise one trip under
both policies and GroupNorm two trips under may-alias) retain logical success
but refuse the existing allocation export. They are not capacity impossibility
claims. All original Elementwise/GroupNorm policy combinations allocate and
emit C++ successfully. C++ source generation is not device qualification.

| Original kernel / policy, bounded launch | Previous explicit-completion excess C→I | Current ordinary excess | Removed / added pairs | Executed commands, previous→current |
|---|---:|---:|---:|---|
| Elementwise / may-not-alias | 43 | 0 | 43 / 0 | 16→25 |
| Elementwise / may-alias | 43 | 22 | 21 / 0 | 16→27 |
| GroupNorm / may-not-alias | 250 | 2 | 248 / 0 | 78→96 |
| GroupNorm / may-alias | 250 | 32 | 218 / 0 | 78→98 |

This is a policy comparison, **not** a same-policy speedup. Previous ordinary
may-alias insertion refused for both kernels, so there is no prior ordinary
command relation to compare there. For both may-not-alias originals, previous
and current ordinary prerequisite relations agree in the checked execution.
The original Elementwise explicit-completion relation is also unchanged.
Greater command counts can accompany less payload order; no device latency
claim follows. The report's root guarantee remains coverage, not uniform
realization-order equality.

The three original preservation kernels (Persistent GEMM, GroupNorm and Qwen
down projection) retain logical/allocation/C++ success under the tested ordinary
and explicit-completion policies, except the unchanged ordinary Qwen may-alias
35-second timeout. Explicit completion succeeds there. A fresh debugger sample
was unavailable because host ptrace is prohibited; this checkpoint makes no
new exclusive-cost attribution for that timeout and changes no Qwen algorithm.
Compact-K/replay and zero-inner-body checks are recorded in the final campaign;
zero-inner-body extra barrier order must remain.

## Reproduction and evidence

Run from the worktree, with the pinned Linux build:

```sh
.venv/bin/python test/lit/pto/Inputs/check_expanded_endpoint_adapter.py \
  --optimizer .local/section8-refactor/source-pinned/candidate-build/tools/pto-test-opt/pto-test-opt \
  --trace-tool .local/section8-refactor/source-pinned/candidate-build/tools/pto-test-opt/pto-sync-input-test \
  --output .local/compact-class-checkpoints/stage4/adapter
.venv/bin/python test/lit/pto/Inputs/check_expanded_production_adapter.py \
  --optimizer .local/section8-refactor/source-pinned/candidate-build/tools/pto-test-opt/pto-test-opt \
  --trace-tool .local/section8-refactor/source-pinned/candidate-build/tools/pto-test-opt/pto-sync-input-test \
  --baseline-optimizer .local/compact-class-checkpoints/start/tools/pto-test-opt \
  --output .local/compact-class-checkpoints/stage4/production-semantics
```

The [receipt](frontier-expanded-endpoint-adapter-20261020.json) records commands,
outputs, hashes and outcomes. Final complete source/tool identities and the
focused gate are in the [aggregate checkpoint](frontier-practical-compact-coverage-20261020.md).

| Established result | Remaining premise / limitation | Appropriate manuscript claim |
|---|---|---|
| Original-frame adapter connects already-retained expanded mathematics to existing commands | Only qualified fixed frames and stable predicates; expansion limits unchanged | Missing executable export can be an implementation gap rather than a new mathematical class |
| Independent actual-cover and command checks pass bounded instances | Native footprints and uniform realization equality remain separate | Exact modeled reduction plus qualified executable coverage |
| Original kernels allocate with six eligible IDs | Some specialized derivatives lack allocation exports | Feasibility of these assigned protocols, not a universal/minimum budget |
| Checked parent relations become smaller than explicit completion | Command counts increase; no device measurements | Bounded ordering-quality improvement, not a device-speed claim |
