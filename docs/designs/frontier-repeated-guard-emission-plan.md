# Repeated guard emission: architectural implementation plan

This is an implementation plan, not a completed emission capability. Its goal
is to make the prepared Gated Delta input reach public logical synchronization
under MayNotAlias through general endpoint mechanisms. Section 5.4 producers
and reducers, original memory semantics, GM alias policies and the allocation-only
policy remain unchanged. MayAlias mathematical gaps and event scarcity are
separate work.

## Evidence and paper correspondence

The reference implementation is `4d1767ae22a97f739d329a1702dd91561a705a5f`.
Gated Delta has exact demands for 29/29 original regions under MayNotAlias and
23/29 under MayAlias. The bounded loop at input line 144 has 60 potential sites
and certified span three. Its recorded first emission failure is site 2 to
site 7, displacement zero, with an unavailable non-replayable guard input.

That diagnostic does not identify the failing endpoint. `CircuitEndpoints::add`
tries the consumer guard before preparing the SET. The bounded caller reports
the expression/predicate error and does not preserve the shared publication
error. Consequently the existing receipt does not establish whether this pair
fails consumer recovery, source recovery, or the cleanup qualifier. Precise
diagnosis is the first milestone, not a premise inferred from the error text.

The current paper working tree supplies three relevant constructions:

- `sections/direct_realization.tex`, lines 64-82 and 287-317: guards must use
  available values; early publication has complementary terminal cleanup.
  Applying cleanup to loops requires retained branch outcomes when original
  outcomes cease to be available. A branch join is not automatically a legal
  replacement for terminal cleanup.
- `appendices/explicit_reduction_details.tex`, lines 80-116, label
  `app:incoming-cut-queries`: known endpoint maps can obtain exact retention
  guards from latest incoming sources, the preceding native start, and
  restricted queries. The query count does not bound selector construction.
- `appendices/online_prefix.tex`, label `sec:online-prefix`: a separate
  realization publishes per executed source and acquires source prefixes using
  counters and watermarks. It explicitly extends the control contract with
  private scalar bookkeeping and still requires executable incoming selectors.
  Its later incremental demand-analysis construction is not part of this plan.

Section 8 (`sections/combined_algorithm.tex`, lines 226-240 and 1120-1162)
keeps mathematical demands separate from executable matching, checks value
availability, retains mathematics after emission failure, and stages fragments
until acceptance. Uniform mathematical analysis remains the default. Emission
bookkeeping must not silently select stateful demand evaluation.

The compiler expert's recommendation is diagnostics, contextual recovery and
incoming-cut queries first. If genuine recurring publication remains, prefer
the paper's prefix construction to an unbounded per-record history log. Do not
claim that counters solve an unavailable incoming selector.

Paper SHA-256 pins at planning time:

| Source | SHA-256 |
| --- | --- |
| `sections/direct_realization.tex` | `0dc7baafde580f0c4f87a99ab00e71c6d987f6341e8247f42139b6eb56fd8385` |
| `sections/combined_algorithm.tex` | `1043c9ea53225148462ea19417b16c6798aaa04158fa0be84fdc7957b4fff393` |
| `appendices/explicit_reduction_details.tex` | `8594316956e82ac6afa48341ffa9ce777a5e7115af3a82fb39da3cff3a635426` |
| `appendices/online_prefix.tex` | `b9bc66015ee06397cca1c494d30b7158f1ee47ab6ae245babf7b983154026ff8` |

Recheck these passages before each implementation milestone and record relevant
changes. The paper and implementation must use the same control assumptions.

## 1. Identify exact endpoint obligations

Introduce a typed emission obligation identifying source guard, consumer guard,
matching identity, terminal cleanup, or target-lowering failure. Include original
operation/phase anchors, pipes, visit coordinates, the retained record, exact
guard root, unavailable original SSA leaves and their requested evaluations.
Preserve errors from `CircuitEndpoints`, `IterationPredicates` and the shared
publication constructor without overwriting the responsible stage.

An explicit audit request should collect failures across all retained records;
ordinary compilation may stop on its first failure. Audit failed recipes in
isolated detached fragments and reuse retained mathematics. Record whether each
leaf is current and available, known from enclosing control, replayable, a past
evaluation, or a genuinely unavailable future evaluation. Classification must
use the original coordinate frame, not static SSA identity alone. Record visit
displacement and definition/cut dominance separately: a current-visit condition
defined after the SET cut is unavailable there without belonging to a future
iteration.

Acceptance: map the actual site 2/7 operations and failing cut, serialize enough
of the guard DAG to reproduce the failure, and account for every remaining
blocked record in the token loop. Keep 29/29 exact demand results and original
IR unchanged. Capture a small general regression from the actual mechanism.

## 2. Normalize context before recovering predicate values

Create one evaluation-aware endpoint preparation path. Relate current-visit
predicate placeholders to their original SSA evaluations, apply proved branch
facts at the original cut, simplify the shared DAG, and recover only live
inputs. Today `IterationPredicates::recover` enumerates inputs before the
contextual emitter applies its branch cofactor; investigate this ordering as a
possible unnecessary recovery, not an already confirmed cause of the benchmark.

Keep distinct visits distinct. Preserve select-based masking of inactive arms,
off-domain coordinates and poison. An unknown or future evaluation is never
assigned a convenient truth value. Reuse deterministic scalar arithmetic only
under the existing replay contract; never replay `tgetval`, loads or payloads.
Cache by session, evaluation coordinates and cut/control context. Avoid DNF
conversion or enumeration of Boolean valuations.

Acceptance: an enclosing branch fact eliminates unnecessary recovery, genuinely
unavailable inputs remain explicit obligations, and zero-trip/nonunit-step
cases preserve original coordinates. Independent command graphs and retained
demand identities are unchanged.

## 3. Supply consumer-causal matching selectors

Use the selected backend's existing endpoint maps and native queries to expose
exact incoming source selectors at consumer cuts. Apply the paper's restricted
incoming-query construction where it avoids unavailable retention expressions.
Keep the backend representation; do not lower every query to integer relations.

Certify that each emitted consumer selector depends only on legally available
current or past evaluations. Forward-order/prefix locality can justify removing
irrelevant future dependencies, but the implementation needs an equivalence
certificate for that removal. A guessed future value is insufficient. The
result must agree with the retained exact demand relation for every admitted
valuation, including absent endpoints.

If necessary, add an owned adapter that retains original predicate outcomes and
source ordinals across a certified finite lookback. Store validity with each
evaluation and preserve enclosing invocation coordinates and loop resets.
Bounded generator span alone is not a proof that every global query or selector
has bounded history. Establish the actual lookback before choosing a ring or
loop-carried SSA tuple; otherwise return the specific unmet interface.
History storage also requires the explicit private-state target capability,
representable indices/state, and transactional structured-control edits described
below for prefix counters. It is not an unrestricted extension of the
original-value endpoint interface.

Acceptance: compare selector results with an independently unfolded oracle on
all small guard sequences, including skipped sources and consumers. Snapshot
original evaluations without extra memory reads or synchronization. First
decide whether milestones 1-3 already close Gated Delta's emission gap. If they
do, omit the optional constructor below from this fix.

## 4. Add shared prefix publication if still required

Implement the paper's separate logical realization only after executable exact
incoming selectors and the private-bookkeeping target contract are established.
This consumes static retained demands; it does not run a streaming lifetime
scan or reducer at runtime.

For each participating source pipe `p`, maintain its executed-source count
`N[p]`. For each selected direction `p -> q`, maintain the prefix of emitted
acquisitions `w[q,p]`. A consumer's exact selector returns the executed-source
ordinal `r[p,b]`, rather than assuming that iteration number equals source
number when branches skip sources.

| Position | Construction |
| --- | --- |
| After each executed source on `p` | Increment `N[p]`; publish once per selected destination using `(invocation, p, q, N[p])`. |
| Before a consumer `b` on `q` | Acquire indices `w[q,p]+1` through `r[p,b]`; advance the watermark by maximum. |
| Whole-invocation terminal epilogue | Acquire the remaining indices through final `N[p]`. |

The watermark records emitted WAITs, not physical completion. Native completion
order makes earlier publications in an acquired prefix redundant prerequisites
of the latest required source. Require the paper's coverage, no-extra-order,
matching and progress proofs. Keep the existing exact same-pipe barrier rules.
Do not insert iteration-end or branch-join drains.

Use one invocation-owned coordinator for a direction across child regions and
sequence crossings. Select a single protocol owner for all demands in that
direction; do not mix incompatible per-record and prefix identities. Child
recipes carry original occurrence provenance and are linked only after the
selected graph is finalized. Existing direct guarded and singleton terminal
cleanup recipes remain cheaper choices when applicable.

Add typed preparation recipes for private helper state and structured control
edits. The current detached arithmetic blocks alone cannot express loop-carried
state and acquisition loops. Stage/rebuild affected SCF control on an owned
clone, verify it, and attach atomically; failed preparation must leave original
IR and retained mathematical owners intact. Prove counter representability in
a supported target integer type and target control readiness. Unsupported
counts, scalar lowering, selector history or scope return explicit obligations.

Acceptance: exactly one consumption per publication, terminal flags empty,
acyclic command graphs, and payload closure equal to the retained exact model
for exhaustive small executions. State crosses nested/sibling regions correctly;
zero trips and skipped arms create no phantom sources. No unbounded branch
history log, payload replay, automatic serialization or kernel recognizer is
introduced.

## 5. Integrate, validate and review

Route accepted endpoint recipes through `PrepareLogical` without rerunning
producers/reducers. Scope caches by selected mathematical result, control policy
and helper-state protocol. Preserve detached ownership, composition provenance
and physical-section mappings. Report the actual realization and its stage
costs, helper-state size, selector work and command counts.

If prefix emission is added, introduce an explicit protocol/certificate kind in
logical metadata and serialized validation. Existing allocation certificates
for per-record matching cannot certify its changed command graph. Reject an
unsupported allocation export while retaining the completed logical plan; do
not weaken validation, assign cyclic IDs from counters, or implement scarcity
repair. Finite fixtures may use an independently validated explicit allocation
view if that adapter is actually implemented.

Focused checks cover current/past/future guards, inactive-arm poison, arbitrary
small branch patterns, zero/one/multiple trips, nonzero lower bounds and steps,
skipped sources, two consumers needing the same prefix, cross-visit and sibling
crossings, nested resets, terminal tails, counter limits, rollback and cache
isolation. Independently check payload reachability, exact matching and cleanup;
command counts alone are insufficient. Validate helper arithmetic/control
lowering and serialized rejection of corrupted protocol metadata.

Run Gated Delta on the optimized source-pinned build under both alias policies.
MayNotAlias acceptance is unchanged 29/29 exact demands plus successful public
logical insertion with unchanged original payload/control behavior. MayAlias
must retain its existing exact results and state any remaining mathematical
obligations; this work does not promise to close them. Allocation may remain
explicitly unsupported. Then replay the 11 prepared TileLang inputs serially
and compare exact-demand, logical and allocation outcomes with the pinned
starting revision. Explain regressions rather than interpreting timeouts as
class exclusions.

Keep aggregate intensive work at two workers or fewer, corpus compilation
serial, and use existing watchdog/memory safeguards as measurement limits.
Do not store builds or campaign artifacts in RAM-backed `/tmp`. Each implemented
milestone needs accepting architecture and correctness reviews before its
commit. No implementation or commit is implied by this planning document.

Implementation evidence and subsequent paper-pin changes are recorded in
[the endpoint audit](frontier-repeated-guard-emission-audit.md).
