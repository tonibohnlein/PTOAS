# Integer difference construction and attribution

This source checkpoint reduces arrangement work during exact integer-union
difference. Compilation, focused execution and paired production measurements
are pending. It does not establish a prefill allocation or a speedup.

The earlier prefill watchdog reached integer-union subtraction while constructing
logical arithmetic demands, before allocation export. Its stack passed through
`arrange`, arithmetic `complete`, and `analyzeArithmeticRegionDemands`.
The [historical evidence](frontier-family-pipeline-history.md) retains the
watchdog observations. A timeout at this stage does not show that the later
coordinate-allocation formula is unavailable.

## Exact arrangement reduction

An arrangement axis fixes affine or modular membership atoms of the subtracted
union. Each atom records its owning conjunction. Once a conjunction is rejected
on a branch, later atoms cannot make it true again. If every atom on the next
axis belongs to a rejected conjunction, every slice of that axis has the same
remaining membership. Their union is the unchanged current region.

The implementation skips that axis before constructing any of its slices. It
preserves the region and the remaining-atom counts of live conjunctions. Axes
used by any live conjunction retain the existing exact construction. No new
solver, admission limit or approximate answer is introduced.

Every pushed arrangement frame has a live conjunction with at least one
remaining atom: covered cells and cells with all conjunctions rejected do not
push a frame. Each remaining atom belongs to exactly one current or later axis.
A skipped axis has no live atom, so some later live axis must stop the skip
before the axis list ends. Initial empty unions and zero-atom conjunctions return
before constructing the frame stack.

Optional structural counters record difference requests, arrangement seeds,
axes, skipped axes, constructed slices and arrangement output pieces. Slice
feasibility counters exclude prerequisite seed, implication and axis-normalization
checks. Early returns without an arrangement do not contribute arrangement
output pieces. Counters saturate instead of wrapping.

## Regional demand profile

The session's optional profile context now reaches the actual regional
arithmetic-demand producer. It records generators, reduction, native and
required closure, required difference, cover composition and difference, and
adjacency. Nested composition and difference scopes use the existing exclusive
time accounting. The reduction record includes structural counter deltas for
its own lifetime.

This is regional logical-demand attribution, not full compiler profiling.
Generic mathematical callers remain unprofiled by default. A disabled context
does not read a clock, create stage records or consume request IDs. Profiling
does not change operation order, mathematical answers or admission.

## Validation boundary

The focused integer-relation oracle enumerates points independently for coupled
affine and modular clauses, shared live axes, multiple seeds and reordered
unions. Scaling cases add 8, 32 and 128 irrelevant axes; another case includes a
rejected congruence with a 130-bit modulus. Mathematical integer relations do
not have machine-wrap or undefined-value semantics.

The arithmetic-demand checks compare enabled and disabled profiles against the
existing independent physical-conflict oracle, check identical structural work,
and verify exclusive-time nesting without asserting a timing improvement.
The entry points are `sync_integer_relations.pto` and
`sync_general_arithmetic_demands.pto`. Production acceptance still requires
fresh serial paired prefill runs through materialization, followed by the
ordered milestone validation campaigns.
