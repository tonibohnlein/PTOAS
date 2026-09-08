# Concrete algorithm: occurrence dependencies → prefix frontiers → verified event reuse

This is our adaptation, not a theorem or algorithm claimed by one of the cited
papers. The executable implementation is `eventlab/`. Source-derived ideas and
important differences are listed in `SOURCES.md`.

## v0.2 implementation changes

See `REVISION.md` for exact tests and limitations. v0.2 adds proof-checked
same-lane barrier subset deletion, physical-conflict-footprint key candidates,
compact finite interval checks, and the unrolled-key-query reformulation.
It does not add native effect import/emission or a qualified MMAD intrinsic rule.

## 1. Problem and scope

Input: an already scheduled, single-core physical program. Do not change its
payload operations, physical addresses, tiling, prefetches, or per-lane issue
order. We may change only synchronization and its scalar participation control.

An occurrence is `(static phase, enclosing iteration vector)` under parameter
and path conditions. A schedule assigns it an injective lexicographic time.
This time specifies original issue order; it does **not** assert completion.
A lane is a physical engine in one execution context, not a buffer slot.

For supported effects, preserve all required completion-before-issue relations.
Prefer not to add ordering beyond what the primitives necessarily impose. Do not
minimize a sum of flags, barriers and PIPE_ALL. Keep command count, added payload
ordering, event capacity and eventual device time distinct.

The prototype accepts integer-set domains and constant-size byte intervals. It
uses the installed isl C API. Per-input declared parameter constraints are part
of the analyzed problem. A compiler may export a constraint only when established
by existing IR/target facts or an authorized input contract. A runtime fast-path
condition requires an actual fallback construction; this prototype does not emit
such native guards.

## 2. Three deliberately separate objects

1. **Semantic facts and requirements:** reads/writes, source schedule and alias
   relationships. They do not depend on the current synchronization plan.
2. **Logical handoffs:** a symbolic relation mapping a publication occurrence to
   its matching acquisition occurrence, without concrete hardware IDs.
3. **Physical realization:** actual nonblocking submissions, delayed flag fires,
   consuming waits, same-pipe barriers and key reuse.

The last two objects can change while the first remains fixed. No `alreadySync`
shortcut is allowed to erase the semantic obligation from the reference.

## 3. Dependence discovery

Let R and W map operation occurrences to the bytes read/written. Let `<T` be the
original schedule order. For exact writes:

- RAW: reaching prior writes to each read, using isl's flow interface.
- WAW: reaching prior writes to each write.
- WAR: the next writes after each read, obtained by the same flow query using
  reversed schedule order, then reversing the dependence relation.

This computes occurrence relationships directly. No fixed buffering distance,
loop-parent equality, or specific parity syntax is built into the algorithm.
Partial disjoint writes can both supply a later whole-range read.

May-write is not a kill. The prototype deliberately falls back to all earlier
conflicting pairs when a may-write is present; it does not claim a precise reaching
producer for that case. A production implementation can use full must/may flow.
Live-in reads are reported; the reference does not establish their initialization.

Separately retain the dense memory contract:

```
D = ((W ; inverse(R)) ∪ (R ; inverse(W)) ∪ (W ; inverse(W))) ∩ <T
```

Here `A ; B` means relational composition, first A then B. This complete pairwise
contract is an independent symbolic check on the sparse frontier seed. No closure
is needed just to discover the sparse dependencies. An early experiment that
computed that redundant closure exhausted the quota on nested slot resets;
removing it reduced work without weakening the final contract.

Ordinary RAR creates no payload-memory demand. Unmodeled target/resource hazards
are **not** silently treated as ordinary RAR. Cross-pipe GM RAW is rejected as a
visibility case. Cross-pipe ACC read/read not already covered by represented
memory hazards is rejected pending its qualified target contract.

## 4. Concrete handoff construction: a prefix staircase

For each directed lane pair P→Q, consider the sparse requirements whose source
is on P and target on Q.

For a destination occurrence v, compute:

```
a(v) = latest source occurrence on P whose completion v requires
```

This is a lexicographic maximum of **required sources**, not all preceding P work.
The source pipeline's publication at a(v) necessarily includes its earlier work.
There is no need to name every such earlier operation as another token.

Scanning Q's consumers conceptually in schedule order:

```
acquired_prefix = none
for each actual destination occurrence v:
    if a(v) advances acquired_prefix:
        publish after a(v)
        acquire immediately before v
        acquired_prefix = a(v)
```

The implementation does this symbolically, without enumerating iterations. Define
`≤P` as inclusive order between P occurrences and `<Q` as strict Q order. Let A
map each selected latest source to the targets needing it. Then:

```
H_PQ = A \ (≤P ; A ; <Q)
```

The subtracted relation identifies requirements already dominated by a demand at
an earlier executing Q occurrence for an equal/later P prefix. Such domination
chains terminate in each finite launch. Their earliest representative retains a
handoff. Both H and inverse(H) must be single-valued: physical flags are not
silently treated as broadcast futures.

This derives first-use waits, last-reader releases and compatible input bundling
from dependencies and lane order. It does not recognize a GEMM arrangement.

### Independent preloads

```
P: LoadA; LoadB
Q: UseA; UseB
```

Requirements A→UseA and B→UseB give two prefix increases. Handoff A is published
before LoadB completion, even if UseA is inside a later loop. A single handoff
after B is memory-safe but imposes B→UseA; the quality checker detects that.

### Compatible bundle

```
P: ExtractLeft; ExtractRight
Q: MatrixReadsBoth
```

The consumer needs the rightmost source prefix already containing both extracts.
One handoff suffices. Adding an earlier Left-only consumer creates a separate
readiness deadline; the algorithm retains two handoffs.

## 5. Whole-plan completion and residual same-lane repair

One direct handoff relation H_PQ supplies the payload completion relation:

```
C_H = ≤P ; H_PQ ; ≤Q
```

A named barrier before a target b on P supplies all earlier P completions to b
and to later P operations. The prototype places unresolved barriers immediately
before their demand, conservatively; it does not claim optimal same-lane cuts.

Combine all directed handoffs and propagate completion by **actual relation
composition**. Do not use a union of singleton coverage sets. The prototype uses
bounded repeated squaring:

```
C := primitive supply
repeat up to a fixed budget:
    C_next := C ∪ (C ; C)
    stop on equality
```

Every included path is exact; this is an underapproximation if stopped early.
v0.2 can stop as soon as the specific queried requirements are covered, without
claiming a global fixed point. A rejection-only first-edge bound avoids expensive
futile closure on impossible deletion trials; positive acceptance still requires
exact supply. The broader nested/scaled query in the revision campaign remains
UNPROVED at its configured work budget.
No approximate transitive-closure result is treated as a proof. Failure to cover
a requirement returns unproved; the code never guesses coverage.

First construct cross-lane handoffs. Then add a conservative same-lane repair
seed only for dense requirements not supplied by their combination. In v0.2,
selected barrier completion also proposes deletion of dominated occurrence
subsets, followed by bounded whole-site-family deletion. Each complete trial is
checked against the original dense contract without the removed actions.
Surviving payload and event cuts are unchanged; primitive supply can only shrink.
This can derive a first-use-only barrier for one repeated static read site. This removes the redundant WAW
barrier in load→compute→next-same-slot-load round trips. Complete directed
handoff families may be reverse-deleted only if the remaining combined plan still
implies D. Final acceptance requires `D ⊆ C` for the entire declared parameter
context. Requirements are retained after any deletion.

This is a conservative construction heuristic, not a minimum-cardinality solver,
not a full backward code-motion framework, and not a proof of maximum overlap.

## 6. Branches, preloads, loops and updates

Control is encoded in occurrence domains and schedule relations. For a panel
produced once per outer iteration and read by an inner loop, the implementation
actually derives:

```
J>0: Load[o] → Extract[o,0]
J>0: Extract[o,J-1] → Load[o+1]
J=0: residual same-lane ordering of successive Load occurrences
```

Independent work after the last Extract does not enter the release publication
prefix. Empty consumer regions do not manufacture a consumption. A production
with no future consumer creates no unused readiness token, but it still needs
completion at exit or before an overlapping overwrite.

Nested slot resets retain the entire `(outer, inner)` identity. There is no
flattened guessed distance. The test with two slots and three inner iterations
has `C[0,2] → L[1,0]`, not a presumed universal `i→i+2` across the boundary.

RMW is represented as both read and write. ACC initialization/update/store chains
retain their internal dependencies. This prototype has no qualified MMAD intrinsic
model, so it deliberately keeps conservative internal barriers.

A native emitter must translate H and its inverse into available scalar guards,
loop-carried logical events and complete consuming operations. Guard availability,
code size, actual scalar cost and emitted-IR rechecking remain part of native
integration. Symbolic relations alone do not establish that arbitrary such code
has already been emitted or is cheap.

## 7. The Ascend-style physical model

A physical payload operation has separate `Issue(v)` and `Finish(v)` nodes.
For an event e, distinguish `Submit(e)`, `Fire(e)` and `Consume(e)`.

```
Issue(v) → Finish(v)
source command order includes Submit(e), NOT Fire(e)
Submit(e) and prior source Finish nodes → Fire(e)
Fire(e) → Consume(e)
Consume(e) gates later destination commands
```

There is **no** automatic edge from `Finish(v)` to issue of the next operation on
that lane. That would incorrectly serialize asynchronous pipelines. A SetFlag
submission does not itself drain later source work. Completion of all prior
source effects is required before its flag fires.

Finite access specialization stores exact normalized half-open intervals. It
enumerates occurrence/endpoint tuples rather than every accessed byte, and tests
hazards independently with interval intersections. Operation/state budgets remain.

The finite explorer interleaves lane-command progress, payload completion and
flag firing independently. Physical memory requirements are checked when an
operation issues, not used as hidden enabling conditions. Thus an omitted wait
can expose a violation. It also detects wrong-generation consumption, rearm,
undrained flags and deadlock.

The explorer's terminal rule separately requires all represented payload effects
complete. This is an abstract exit obligation, **not** code that removes or adds
an Ascend ABI barrier. Initial flags are assumed empty in the abstract model.
Target instruction semantics, queue behavior, cache visibility, reservations and
initialization must be qualified when integrating into real PTOAS.

## 8. Finite keys: two complementary algorithms

### 8.1 Symbolic candidate functions, proved across the parameter context

For each directed domain, v0.2 first recovers local write footprints shared by
actual source/target handoffs. A bounded descriptor range (space/start/extent) is
projected across the declared parameter context and ranked. Rank modulo k provides
storage-derived candidates, including middle-loop panel slots and preloads.
This descriptor selection is proposal evidence, not a safety certificate.

The older constant, static-rank and selected iteration-coordinate functions remain
fallback candidates. Search k up to the configured pool. No candidate is accepted
on matching shape, slot identity alone, or sampled success.

For a function K, let Next_K(a,b) say b is the next source publication after a
that uses the same key, in the same execution. Let H(a)=c be a's matching consumer.
A sufficient reuse condition is:

```
Next_K ⊆ H ; C
```

C here is the selected plan's full payload completion relation. This means:

```
Consume(a) → Issue(c) → Finish(c) → Issue(b.source) → Submit(b)
```

The check is stronger than necessary: other causal acknowledgements may establish
safe reuse without completion of c. Nevertheless it is a sound sufficient rule in
the stated target model, and it does not add any wait. The acquisition key is
computed by `inverse(H) ; K`, preserving the matched generation rather than
assuming the source and target use the same iteration counter.

This produces and proves `i mod Nslots` for the simple rings, including their
nested-reset test. It also proves constant reuse for supported panel lifecycles.
It does not turn the finite sample's coloring into an arbitrary-trip assertion.
Failure means the candidate search did not establish an assignment, not that the
hardware necessarily requires serialization.

### 8.2 Exact finite chain-cover oracle under a causal criterion

For a concrete specialization, event a may precede event b on one hardware key
only if `Consume(a) → Submit(b)` is already implied by the full physical graph.
For events in each directed domain, these compatibility edges form a DAG.

The minimum number of chains is:

```
number_of_events - maximum_bipartite_matching_size
```

One chain uses one key. A chain cover induces a matching through its consecutive
links, and any matching in the acyclic compatibility relation induces disjoint
chains; this proves optimality **within this fixed plan and sufficient criterion**.
It is not a globally optimal synchronization plan or a symbolic arbitrary-trip
allocator. The prototype caps this reference search's domain size.

Both assignments are checked by the same independent asynchronous explorer on
finite scenarios. No scarcity fallback is silently inserted if either search
fails. Production may later use the existing qualified explicit recovery policy.

## 9. Quality and correctness are different checks

Compare two plans only with identical payload occurrences, effects, schedule and
lane assignment. Compute payload ordering `Finish(a) → Issue(b)` from their
physical graphs. Report new and removed relationships separately, along with
sets/waits and named barriers.

The default sufficient quality condition is no new mandatory payload ordering.
It does not price additional scalar tests or event overhead and is not a latency
prediction. A safe-but-broad preload plan deliberately passes the safety checker
and fails this quality condition. Compatible input coalescing removes a pair while
preserving the same payload-order relation.

v0.2 enforces no-added-primitive-order for its deletion-only barrier refinement.
Every new key assignment is proved on the resulting plan. Symbolic quality checks
for arbitrary movement/splitting, instruction costs, and device timing are still
not implemented. The initial constructor itself
preserves distinct per-pair prefix deadlines; its completeness and global
optimality are deliberately not claimed.

## 10. Native integration, not another replacement pass

Implement a read-only physical-facts exporter at the current pass entry using the
existing translator, actual selector/range facts and guard identities. Compare
its required relations with the prototype and with retained native requirements.
Do not independently parse PTO op names to guess effects.

Migrate one looping relationship through the common requirement/resolution
interface. Consume the prototype's proven prefix-frontier rule there; reuse the
native boundary emitter, target tables and independent reconstruction. Remove the
superseded decision for that family instead of layering another planner alongside
it. The unchanged QK case and equivalent forms are an effectiveness gate; the
current test reductions are not a substitute for that native result.
