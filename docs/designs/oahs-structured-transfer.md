# OAHS: obligation repair and finite structured transfer

Initial hardening base: `b0c241f12fbb3d9532746bc47e966d3978ccc9f9`.
Budget-independent revision base: `2c072d1a11b2e400f615ca104c42708e52aec71c`.
This series keeps `algorithm=existing|handoff`. It does not port the old planner
stack or add a persistent-lifetime recognizer.

## Obligation import

`closeStructuredSyncOrigins` is shared, read-only analysis called after the
production translator and before OAHS trusts its phases. It closes SSA storage
origins over for initial/backedge/result edges, while initial/condition/backedge/
result edges, and forwarding operations. A loop-derived view covers its reaching
whole roots: this is deliberate widening, not an exact offset recurrence solver.
The records are append-only and original phase effects are then refreshed.
Unknown origins gain unknown-range coverage. Root propagation uses a delta
worklist to completion: each new node/root fact crosses each forwarding edge
once. No computational cutoff changes the origin result.
This hook is not enabled for the legacy planner by this series.

Each native physical effect is retained even when it has no other static conflict
partner. Repeated uses of the identical immutable footprint record share an own
cell; distinct records retain separate identities. Pairwise overlap cells supplement
these records without taking a transitive alias closure. This preserves both self
recurrence and non-transitive may-alias relations. Missing absolute local addresses,
zero-size or overflowing local geometry are unknown, not distinct-root noalias.

Alias queries now depend on distinct footprint groups rather than repeated effect
pairs. There is no million-pair threshold or flag that removes overlap obligations.
Worst-case distinct-footprint overlap remains quadratic; grouping is not a claim
of a universally linear alias algorithm.

## Actual placement

Native extraction identifies the original successor instruction and containing
block of each new command. A branch terminator is not an arbitrary substitute
for a payload cut. Commands are detached only for payload/contract comparison and
restored at their actual saved block/successor, not at a reinterpreted global cut.
The immutable original control tree determines which commands execute. Mutation
checks move a required command into the other branch and move final retirement
into an optional arm; failure must leave the input function unchanged.

## One state transfer

`OAHS/Transfer.h` is the common candidate-analysis and final-checking interpreter.
It is not a symbolic occurrence expansion. Static operation classes stand for
all dynamic visits of that operation and all its imported conservative accesses.
The state contains per-observer possibly pending classes, live publication
remainders, possible empty/full balance, validity of publication receipts, and
must-knowledge of the latest consumption of each actual directional key.

Issue adds its class to all pending sets and all live remainders. Publication
captures the source's outstanding remainder minus source-engine classes; it does
not gate later source issue. Acquisition intersects the target's pending set with
that remainder. A new consumption replaces the old consumption knowledge and
invalidates that key in every saved acknowledgment. Barriers do not consume flags.

Sequences pass state. Choices union possible histories/balances and intersect
must facts. One worklist on the static control graph computes ascending finite
invariants containing entry and backedge state. A site is re-enqueued only when
its incoming facts change. Unreachable input is distinct from fresh quiescent
input. Preconditions are checked once at the stabilized site states, not in an
optimistic discovery iteration or by recursively resolving every nested loop. For includes the zero-trip edge; while exits from
the before-region. A live publication can cross a child region when the original
control and receipt preconditions justify it; no region-entry reset is inserted.
The full proof is relative to the documented prefix/event contract, not a claim
of new hardware validation. Finite tests are supporting evidence, not a
machine-checked simulation proof.

## Construction

One deterministic repair loop analyzes the same state, supplies missing memory
completion, and then checks the complete protocol. The first placement policy
retains early producer cuts in one-visit words; general structured cases use
matched commands at the consumer's actual cut. This is intentionally conservative
placement, not a second barrier-only semantic planner.

Key choice tries unused eligible keys, then proposes reuse. Neither choice is an
acceptance fact. The complete memory plan is present before recurrence is tested,
so real storage-release handoffs can acknowledge readiness without extra replies.
Only a failed causal-consumption check can introduce a reply. Overlapping logical
balances can force an earlier publication to its consumer cut. Unresolved repair
selects a fully checked ALL realization when available. A repair slot indexed by
(consumer, source) can be created once; its packet can be moved to the consumer
once and acquire at most one acknowledgment. Every continuing repair strictly
advances these finite states. An unchanged candidate is not retried. There is no
arbitrary construction-step allowance and no verification-work allowance. Actual
target scarcity and a failed finite protocol repair can still require conservative
synchronization. There is no legacy fallback, minimum-command claim, or global
optimality claim.

## Shared target and operation contracts

`PTO/IR/SyncTargetProfile.h` exposes separate A3 AIV/AIC lane/direction tables.
The source-qualified directions and static-library-safe pool 0..5 are taken from
`c73c04fb3b7a76d10ab55b48d4dcbe0270e36da2`,
`StructuredSyncCore.cpp`, `Target::supports/event/barrier`, and its header.
In particular, the AIC scalar directions and MTE2/MTE3--FIX directions excluded by
that profile are not invented. IDs 6/7 are not asserted absent from hardware.
This is a deliberately selected conservative source contract, not a newly
measured device guarantee. Source classification is distinct from hardware
qualification; new targets must supply their own reviewed profile.

The adapter resolves the core from the original kernel-kind attribute and actual
pipeline/local-storage facts and rejects contradictory mixed contexts. It keeps
the pre-existing production alias/ordinary-GM compatibility premise; it does not
claim a new general visibility theorem.

Ordinary operation admission now uses the shared translator's production
pipe/memory-effect contract and semantic accounting report. There is no
external-model registration list for vector arithmetic. Optional restrictions
for special load/store variants live with those operations; ordinary FIX stores
use the existing source-dependent pipeline declaration. See
[shared semantic extraction](oahs-shared-semantics.md) for completeness checks,
compatibility assumptions, and remaining protocol gaps.

## Remaining coverage work (not claimed complete)

This is not the full production-population semantic-accounting milestone.
Macro import and its may-effects/must-completion boundary, authored event
composition, explicit visibility recipes, queue/resource ownership protocols,
additional operation variants, mixed physical sections, and additional targets
remain unsupported and are rejected. Reservations and resource-exclusion cells
are handled; generic resource/visibility fields are structurally validated but
are not silently treated as byte completion. No 100% corpus claim follows.
Distinct-footprint pair materialization and full candidate reanalysis remain
scalability tasks. Budget independence does not imply cheap analysis on all input
families. Ordinary allocation failure/cancellation must not be treated as a proof.

## Tests and required native gate

The standalone CMake target compiles the exact production Plan.cpp and its
finite transfer header. The graph oracle independently builds launch/finish
vertices, checks all original conflicts and command-graph acyclicity, and proves
consume-before-republication by reachability WITHOUT inserting those edges as
assumptions. It expands bounded for/while/choice traces only in tests.

The native tests cover carried roots, self-recurrence, while forwarding, actual
endpoint placement, atomic rollback, generic ordinary interfaces, dynamic local
address aliasing, and 730 repeated loads without budget-induced ALL insertion. They must be built with the repository's configured
MLIR/PTOAS toolchain. Adding their source does not mean they were executed.
See the patch package validation record for commands actually run.


## Termination and scaling obligations

For a fixed candidate, incoming site states grow by join in a finite product:
pending/remainder and occupancy facts grow, while must-valid and consumption
knowledge can only be lost at a join. Primitive transfer may establish facts, but
the stored incoming states still change monotonically. Work continues until the
queue is empty, then all original preconditions are checked on those states.
This eliminates exponential re-traversal caused solely by recursive nesting.
It does not complete the domain's hardware simulation proof.

Construction has at most N*P memory-repair slots, where N is the static physical
phase count and P is the fixed lane count. Each is created at most once, and each
packet can be relocated and acknowledged once. All branches that cannot change
one of those states terminate in a checked fallback or failure. Thus successful
edits have a structural bound (at most 3*N*P), not an empirical attempt budget.
Each candidate still invokes complete analysis; no near-linear whole-constructor
complexity is asserted. A future incremental analyzer must invalidate affected
receipt facts after edits rather than silently reusing old candidate state.

The test-only concrete trace enumerator retains its own enumeration limit. It is
not used by production analysis and cannot select a production synchronization
plan. Intrinsic container/identity size checks remain, with no claim of infinite
memory availability.
