# Compact exact selector equality

This checkpoint changes the representation of qualified slot conflicts inside
the existing constructor. It does not increase the work allowance, introduce
an approximation, or complete the buffering compile-time acceptance gate.

## Computation and trust boundary

When both accesses have the same original ordered physical slot table, equal
scope and footprint, independently qualified disjoint slots, and full-domain
in-range selectors, physical overlap is exactly selector equality. The new
query adds that equality to the original ordered occurrence relation instead
of building and unioning one occurrence domain per physical slot.

Different, shifted, permuted and partially overlapping tables retain general
physical interval comparison. Unknown or out-of-range selectors retain the
conservative requirements. Equal selector numbers alone do not prove overlap
or disjointness across different physical mappings.

The filtered query's cache is valid only for the full immutable native order
between its two points. It must not be reused for arbitrary previously filtered
relations. Range proofs and geometry comparisons have separate scoped caches.
Unsupported queries never become empty conflict relations; exhaustion remains
an explicit outcome.

An initial mixed-representation probe made the reversed mapping time out even
though its requirements remained exactly equal. The corrected constructor first
uses union-find over the existing physical-overlap candidate edges and freezes
representation eligibility per storage component. Different or unresolved
geometry selects the existing precise general slot calculation for that whole
component. An unrelated homogeneous component remains eligible for equality.
Unknown scalar ranges still retain conservative requirements independently of
this geometry-based representation choice.

This prepass adds no new pair population. Rank and path halving bound graph
work; geometry is compared only when joining distinct homogeneous roots. Beyond
the existing candidate enumeration, its work is
`O(A + (E+M) alpha(M) + M log M + compared table elements)` for A accesses,
E candidate edges and M memory descriptors. The logarithmic term includes
existing ordered-map lookups. This is a preprocessing bound, not a complexity
claim about the subsequent Presburger solver.

Fresh reconstruction reimports every used mapping and checks each selector's
value under independently checked original point, loop and parameter bindings.
Preserving pairwise selector equality alone would miss a common permutation of
both selectors. The emitted check does not use that weaker criterion.

The new scalar-equality query qualifies coefficient mass and expression depth
with a memoized DAG traversal shared by its two operands. It charges actual
node/edge visits and bounds possible expansion before invoking MLIR dimension
substitution and flattening. This fixes that cost contract for the new query;
it does not claim to replace every older scalar-import traversal.

## Acceptance evidence

The rebuilt mandatory focused CTest passed in **57.48 seconds**, including all
46 scalar/equality tests, six strict slot fixtures and corruption challenges,
five new compact-slot scaling fixtures and the strict mixed-component test.
Its artifact directory is
`insertsync-builds/PTOAS-oahs-clean/test-results/oahs/focused-m89e9obx` in the
parent workspace. The compiler library and five native targets were rebuilt
with at most two workers. Native test invocations were serial.
This is worktree-build evidence based on `93167f5ca`; the final native driver's
SHA-256 is `7ef62f181bab8ccb7ec02e8a95202f082aa374663f78b081e4e53af6fc71d80f`.
Commands, input hashes and the worktree state are retained in the campaigns.

The scalar tests compare full equality, filtering a qualified relation, and
their libisl reference, including nested occurrences, negative/nonunit loops,
guards, empty domains, shared/distinct parameters, unavailable precision and
exhausted budgets. Shared-expression and depth tests challenge the query's
qualification; they do not establish general importer DAG complexity.

The production scaling population keeps four access phases and four loop
iterations while varying the physical table through 2, 3, 4, 8 and 16 slots.
Every case constructs strictly, checks concrete present and absent conflicts,
preserves independent readiness, and reaches fresh reconstruction. Each records
exactly eight range builds, twelve equality builds and one geometry comparison,
and **zero per-slot domain builds**. Counts include reconstruction. Four trips
do not exercise a complete rotation of the larger tables; the test establishes
query-population scaling for this stated family, not universal linear time.

The common-permutation corruption changes both D2 descriptor selectors by
`1-s`, preserving tables, payload operation kinds/operands and event counts.
The original-descriptor contract rejects it and preserves the untouched input.
That earlier safeguard necessarily fires before the later selector-meaning
comparison; this test is not reported as coverage of that later branch.

The mandatory mixed-component fixture contains a homogeneous table at
`[8192,8704]` and a disjoint reversed pair `[0,512]` / `[512,0]`. Per-mapping
production receipts verify that only the former remains compact. Strict
construction checks 20 payload occurrences and 61 concrete RAW/WAR/WAW
obligations, unchanged payload/allocation/view/ABI, balanced events and
retirement. The homogeneous component executes four trips and the reversed
component one. The initial four/four fixture correctly declined allocation;
its artifact remains `slots-m2/compact-components1`, with no effectiveness claim.

## Full slot campaign

The final component-consistent campaign completed all twelve unchanged inputs:
**ten strict constructions, two controlled refusals, no timeouts and no oracle
mismatches**. All twelve typed requirement sets are exactly equal, using libisl,
to the previous general-slot implementation at `93167f5ca`. The comparison also
checks the original input hash, physical access/phase identities and occurrence
point domains; coordinate counts alone are insufficient.

| Input | Previous strict outcome / seconds | Compact outcome / seconds |
| --- | --- | --- |
| Rotating D2 | Applied / 47.80 | Applied / 3.19 |
| Rotating D3 | Timeout at 90 | Applied / 4.07 |
| Equivalent mask D2 | Applied / 50.12 | Applied / 3.21 |
| Explicit-base D2 | Applied / 49.13 | Applied / 3.21 |
| Shifted overlap | Lowering refusal / 16.71 | Lowering refusal / 16.51 |
| Reversed overlap | Allocation refusal / 41.21 | Allocation refusal / 41.16 |
| Additional reader | Applied / 87.65 | Applied / 5.42 |
| Common consumer | Applied / 42.75 | Applied / 3.49 |
| Skipped first reader | Timeout at 90 | Applied / 5.70 |

Unknown, out-of-range and negative selectors also construct and retain their
checked conservative dependencies. D2's erased-wait, swapped-load and swapped-key
mutations are rejected. Static D2 remains three set/wait sites and a terminal
full drain; the first reader retains its independent readiness.

These are single serial native test-driver invocations, including discovery
export and verification, from separate campaigns. They demonstrate the current
outcomes and large observed cost reduction; they are not a repeated paired
performance experiment or the ordinary two-/three-buffer `<=2x` compiler gate.
Rotating D3 is distinct from the three-buffer regression kernel.

Artifacts in the parent workspace:

- `insertsync-builds/campaign/logical-plan/slots-m2/exact-guards-original12`
  is the frozen general reference, including complete discovery sidecars.
- `.../compact-original12-final1` preserves the mixed-form regression: ten
  constructed, one lowering refusal, and the reversed mapping timed out.
- `.../compact-original12-final2` is the corrected full campaign above; its
  aggregate status is **partial**, with checked refusals separate from the ten
  emitted successes. The runner's checks passed; that is not twelve successes.

## Preserved accepted kernels

The final rebuilt four-kernel campaign passed in
`insertsync-builds/campaign/logical-plan/slots-m2/compact-four-final2`, including
strict constructor attribution, unchanged payload/allocation/view/ABI,
scalar replay, useful completion boundaries, C++ emission and negative checks.

| Kernel | Sets / waits | Body barriers | Terminal `PIPE_ALL` |
| --- | ---: | --- | ---: |
| One buffer | 4 / 4 | None | 1 |
| Online softmax | 12 / 12 | V: 20 | 1 |
| Q projection | 23 / 21 | M: 5 | 1 |
| QK | 17 / 17 | M: 2 | 1 |

These preserve the preceding checkpoint. Static set/wait site counts can differ
under complementary guards; executed matching and participation are checked.
Device correctness and timing remain separate and have not been run locally.

The next cost task is exact publication succession over actual selected event
domains, followed by the ordinary two-/three-buffer acceptance and repeated
compile-cost comparison. Finishing this checkpoint does not authorize skipping
those gates before later allocation, MMAD and GEMM work.
