# Stage 2 contract: static reaching-access research prototype

This prototype is a finite **may-analysis over original static sites**, not the
paper team's token construction and not an executable synchronization provider.
It consumes the existing `CompletionStorage` class/mode summaries and original
SCF structure through a test-only export. Ordinary dispatch, SCC completion and
allocation stay unchanged.

## Invariant and transfers (before implementation)

At each CFG point, retain a set of `(class, mode, original site)` facts. A fact
means an earlier visit to that site **may remain relevant** on some path reaching
this point. It does not mean that this is the selected last writer or a unique
occurrence. All modeled effects must be covered by the class memberships. An
unknown address retains every class it could touch; distinct GM roots are not
assumed disjoint.

For an executed payload a, request a completion-to-start obligation from every
incoming fact sharing a possibly accessed class when either mode writes. Reads
add their own read facts. Writes add their own write facts. In the baseline,
**neither a possible nor a partial write kills any earlier fact**. RMW reads its
incoming state before adding its write facts. Native protection and scalar
readiness are separate interfaces; this prototype conservatively retains
same-pipe storage obligations rather than inferring readiness from address data.

Sequential composition passes OUT to IN; empty regions are identities. A branch
joins the two OUT sets by union, including the unchanged input for an absent
else. This is static possibility, not execution of both arms or a token join.
A loop header unions entry and backedge states; solve to the finite least fixed
point. The exit includes the zero-trip header path. Nested loops and counter
resets do not clear the state. Forwardness is interpreted dynamically: a static
self-pair denotes earlier visits only, never a cycle within one occurrence.

Optional refresh requires a supplied proof that **this executed payload fully
writes the entire class**, and first acquires the incoming conflicting facts.
Only its transfer may replace that class's facts with its own write fact. A
skipped branch preserves the old facts through the other arm. The invariant then
allows a killed fact to be represented by the refresh completion through the
already generated obligation. Merely having finitely many buffers is not this
proof. Baseline Delta receives no automatic refresh facts.

There are finitely many static facts, independently of loop trip counts.
Monotone union/GEN-KILL transfers terminate at their least fixed point. The
induction argument is storage coverage: every conflicting earlier dynamic access
is represented directly, or through a proved refresh that acquired it. This is
not a minimum-demand claim.

## Explicit realization boundary

A static source-site set does not choose which dynamic visit to wait for, retain
branch history, establish an executable earlier publication cut or assign flags.
Bounded validation expands earlier occurrences solely in the checker. It must
not be mistaken for compact command generation. General lowering remains
unavailable until the existing endpoint interface can supply those identities,
available guards, legal cuts and consuming-notification lifetime certificates.
The existing completion fallback remains the production implementation.

The two-class skipped-writer cut witness is retained: publishing a release at a
later writer can order an unrelated earlier reader before that writer. Moving
the cut earlier needs an independently proved availability argument. The
prototype must report this gap rather than infer order equality from dataflow
coverage.

## Implemented research boundary and results

`CompletionDataflowInput.h` exports original SCF nodes, physical reservation
summaries and the existing production `CompletionStorage` memberships only for
an explicitly annotated `--step1-json` test request. It performs no recognition,
allocation or endpoint construction. `reaching_access_research.py` solves that
finite graph with a worklist. The independent checker derives all conflicts from
reservation overlap and read/write effects, without reading the class partition
or prototype obligations; it separately checks class coverage of those overlaps.
Unknown GM roots remain may-alias. This is coverage relative to the supplied
reservation model, **not independent native footprint validation**. Scalar-value
readiness is outside this storage-only experiment.

The original Gated Delta graph has 78 static sites, three coarse classes and 89
CFG nodes. It reaches its fixed point in 272 node visits, retains 13376 facts
across CFG outputs and inspects 11856 incoming facts, producing 5497 static
source/target pairs. These numbers do not depend on the chosen runtime token
count: the same solved graph is reused for every bounded execution. This simple
may-analysis is not promised linear in static site count; its pair inventory
can be quadratic.

90 bounded control valuations cover zero through three tokens, both softplus
outcomes, initial-state load/skips, prefetch tails, and the original outer guard
`iv39<1` (later outer visits remain skipped). Floating outcomes are explicit
control-abstraction inputs: this is not a floating-point numerical execution
check. All reservation conflicts are covered; the maximum measured excess is
211 completion-to-start pairs relative to the finer reservation reference.
This excess comes from the coarse classes and static path/origin union, before
any SET/WAIT realization. No refresh fact is inferred for Delta.

400 separately labeled synthetic skipped-writer/repeated-child cases check
carried accesses across empty visits and counter resets, with and without a
proved full four-byte-class overwrite. A dominating refresh reduces retained
facts 6→4 and expanded obligations 3→2 in its straight-line witness; excess stays
zero in both. Thus the demonstrated refresh improvement is representation work,
not a claimed ordering improvement. A read-only site is explicitly rejected as
a refresh proof. In a branch, the skipped path preserves the earlier writer.

The two-class publication-cut witness adds exactly `U0→P1`, `U0→Q1`, `U0→U1`.
It demonstrates why storage coverage does not settle executable order quality.
No protocol from the separate token research was modified or promoted to a
production theorem.

Reproduce:

```sh
.venv/bin/python test/lit/pto/Inputs/check_reaching_access_research.py \
  --trace-tool .local/section8-refactor/source-pinned/candidate-build/tools/pto-test-opt/pto-sync-input-test \
  --input test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto \
  --output .local/compact-class-checkpoints/stage2
```

The [receipt](frontier-conservative-dataflow-prototype-20261020.json) pins the
input, test tool, source and command. Production dispatch, existing completion
protocols and allocation are unchanged.

| Region | Candidate class | Recognition result | Demand guarantee | Executable export | Allocation | First remaining obligation |
|---|---|---|---|---|---|---|
| Delta token/control graph | Finite reaching-access may-analysis | Test graph admitted | Bounded reservation-conflict coverage; no minimum claim | Not provided | Not requested | Dynamic occurrence selection, available guards, safe publication cuts and notification lifetimes |
| Skipped writer/reset child | Same prototype + separately proved refresh | 400 bounded cases pass | Storage coverage across joins/backedges | Not provided | Not requested | Same realization obligations |
| Two-class publication cut | Scoped ordering counterexample | Checked finite graph | Later cut covers but strengthens by three pairs | General exact cut unavailable | Not requested | Guard availability at an earlier publication cut |

| Established result | Remaining premise | Appropriate manuscript claim |
|---|---|---|
| Finite static dataflow covers declared may-conflicts without a lifetime bound | Native effects and scalar readiness remain supplied | Conservative storage obligation generation is constructive |
| Optional refresh replaces facts after acquiring them | Must prove full-class overwrite on that executed path | Refresh can reduce retained work; no automatic quality claim |
| Dynamic source/cut mapping is unresolved | No implicit branch-history state or completion tokens | No new uniform Section 6 SET/WAIT realization theorem |

Oracle scope: the reservation reference includes storage conflicts even between
S-pipe accesses. Native scalar readiness/protection is not independently modeled
here. The reported excess is relative to that declared reservation-conflict graph,
not a production exact-F* or complete native-readiness measurement.
