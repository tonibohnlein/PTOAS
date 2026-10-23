# Qualifying upstream access improvements for Frontier

Starting source: `3c7a5839aff5844101c7c6af3b37b4ef43183fc0`, clean.
The four incoming fixes are `6f5e16afa` (dynamic local aliases), `d30269595`
(ACC columns), `dea14616b` (disjoint row backedges), and `60c1a767f` (boxed L1
and TEXTRACT ranges). This checkpoint concerns their shared access/interface
benefits, not a new demand reducer or allocation algorithm.

## Existing connections

`SyncInput` now owns the translator's instruction-specific access records.
Ordinary instructions keep those records in the pipe-envelope view.
`SyncStorageEffects` consumes their narrowed intervals as enclosing bounds;
they are not promoted into declared exact instruction selections. The existing
16-/32-bit TEXTRACT contract and boxed descriptor maps remain in place.
Unknown physical addresses still cannot establish disjoint local allocations.
Completion's memory-space treatment remains conservative.

The upstream row-backedge hook belongs to legacy `InsertSyncAnalysis`.
Calling it from an occurrence-free overlap query would be invalid: a proof for
different iterations cannot remove a real conflict within one iteration or
across a counter reset.

## First source-level missing connection and proposed repair

The shared vector operation qualification rejects every `pto.subview` in
`plainVectorAccessTile`, even a complete contiguous row whose address and
alignment follow directly from an already supported parent. Consequently a
precise descriptor map can still yield a whole-parent access enclosure. A
new loop-backedge pruning hook alone cannot repair that loss.

Qualify a narrow complete-row view in the existing operation-effect service:
plain row-major, noncompact, unpadded VEC parent; matching element/layout/space;
one complete child row; zero column offset; a directly allocated parent with
256-byte-aligned base and complete 256-byte row extents; and either a constant
in-bounds row or the induction variable of a static nonnegative unit-step counted loop bounded by the parent. Existing
dimension and native burst/alignment restrictions continue to apply.

The address is `parent_base + row * parent_row_bytes`. The row extent and
column offset make its physical footprint identical to the selected child
rectangle. The range proof prevents overflow and leaving the parent; base and
row-byte alignment prove the existing operation's alignment premise. The
complete vector row also contains A5 unmasked read fringes after a supported
valid-shape reduction. The operation still supplies its read/write role and valid-shape qualification.
Original row expressions and enclosing occurrence coordinates remain in the
shared map. This proves footprint geometry, not independence of two visits.
No transitive closure of alias metadata or new hardware ordering is assumed.

Other views retain existing enclosures. In particular, unknown row indices,
unsupported layouts, nonunit/unbounded loops, misalignment and incompatible
views are not admitted through this qualification. Existing exact whole-region
results are retained on unavailable exports.

## Validation contract

Exercise actual shared access recovery and ordinary Frontier requests on small
L1/TEXTRACT, ACC-window, unknown-address and row-loop fixtures. Independently
enumerate supplied physical layouts and compare local access coverage, selected
model covers, concrete required-order coverage, queries and selectors. Keep
enclosure precision distinct from a declared exact effect. Use actual emitted
commands for matching, acyclicity and physical reuse checks where endpoints
and six-ID allocation qualify. Include zero/short loops, overlap, skipped
branches and repeated row counters. A same-iteration or cross-reset conflict
must survive even when different rows within one visit are independent.

The pre-rebase tools are preserved with hashes matching the accepted
`fa0214857` campaign. A fresh post-rebase baseline build precedes the repair.
No stale tool output counts as validation of the rebased source. Commands,
source/tool pins and results belong in the companion receipt. No device or
native numerical validation is available; physical layout and instruction
contracts remain explicit premises.

The row fixtures supply ordinary ND source/destination views with explicit
shape and strides. This qualifies the existing DMA effect contract; it is not
an assumption inferred for an opaque partition-view argument in an upstream
reproducer. Original upstream inputs with missing view information must still
retain their conservative effects. The synthetic fixture's initialized local
inputs and may-not-alias GM roots are explicit contracts.

## Additional integration gap found by the singleton regression

A one-row visit now legitimately selects `expanded-finite-guarded` instead of
an enclosure-based sequence. Logical preparation succeeded, but session
allocation ignored `expandedFiniteDemands` and returned no certificate. The
older standalone finite insertion function already uses the expanded regional
query adapter with the common regional certificate builder.

`AnalysisRequest.cpp::allocationRegion` now forwards that same retained query
interface, owning the original input. `allocateRetained` admits this producer
to the existing regional certificate construction. No reduction, coloring,
resource budget, mapping semantics or synchronization protocol is changed.
The allocation export additionally requires a closed expanded occurrence frame:
every counted-loop ancestor of every payload must have its retained fixed
coordinate, and enclosing while scopes refuse. The check also precedes
allocation candidate selection and cached-certificate use, so requesting local
queries first cannot bypass it. A proper inner region retains its mathematics,
queries and logical recipe after refusal; no cross-visit rearming is inferred.
Unavailable proof still returns no certificate. This closes a real downstream
regression rather than making the row-footprint qualification contingent on
whether a less precise representation happened to allocate.

## Measured result

The comparisons distinguish accepted pre-rebase `fa0214857`, fresh rebased
`3c7a5839a` before this repair, and the dirty candidate. The receipt separately
pins the initial candidate tools and the final review-corrected build; the final
build repeats the same eight production comparisons. All use the same
LLVM build, prepared fixtures, A3 target, six-ID pool and explicit may-not-alias
GM-root contract. Each row below is one identical bounded execution. Counts
are executed SET/WAIT/barrier commands; excess counts completion-to-start pairs
relative to the independent supplied-layout model.

| Fixture | Pre-rebase commands / excess | Rebased commands / excess | Candidate commands / excess |
|---|---:|---:|---:|
| Disjoint INT8 L1/TEXTRACT | 5 / 2 | 3 / 0 | 3 / 0 |
| Overlapping INT8 L1/TEXTRACT | 5 / 0 | 5 / 0 | 5 / 0 |
| Two TEXTRACT windows from one source | 7 / 3 | 5 / 0 | 5 / 0 |
| Disjoint ACC columns | 9 / 5 | 7 / 0 | 7 / 0 |
| Overlapping ACC columns | 9 / 0 | 9 / 0 | 9 / 0 |
| Three distinct rows | 10 / 4 | 10 / 4 | 8 / 0 |
| Three rows, two counter-reset visits | 15 / 14 | 15 / 14 | 11 / 2 |
| Three visits reusing the same row | 10 / 0 | 10 / 0 | 10 / 0 |

The companion receipt records both directions of prerequisite comparison.
Across these checks the candidate removes prerequisites without adding any;
coverage, command matching, acyclicity and causal flag reuse pass. This is
bounded relation containment, not a uniform order-equality proof. In particular,
the repeated-row-counter case retains two excess pairs, and production still
reports `realization_order=covers-selected-order`.

The newly qualified original row effects retain two symbolic row regions,
including the original induction variable, rather than whole-parent writes.
They do not gain disjointness across resets. Unknown local addresses continue
to overlap known local allocations under both GM alias policies.

## Qualification and limits

The focused gate passes **22/22** tests, including compact-K/replay,
zero-inner-body, Qwen interval/import/reuse, retained prefix children,
expanded-coordinate endpoints, storage-export coverage, scalar contracts and
hardware protections. The new test exercises **14 positive fixtures**, **six
negative cases**, **3,644 all-event reachability comparisons** and **162 local
boundary probes**, including genuine absence. Two additional allocation-frame
probes compare a closed whole expansion over two visits with a scoped inner
region under the same visits. The former allocates; the latter retains queries,
preparation and its mathematical owner but refuses allocation. Twelve fixtures
compare actual retained covers against independent all-conflict reduction. Two repeated INT8
fixtures select arithmetic records that the existing test consumer cannot
unfold; that cover-check limitation is explicit. The test allows this refusal
only for those two named synthetic fixtures, so a later route change cannot
silently remove cover checks elsewhere. Their access, all-event
queries, selectors and emitted-command coverage/reuse checks still pass.

One negative allocation address is rejected by the existing 32-byte alignment
verifier. Five valid but unsupported row descriptions retain enclosures:
nonunit steps, unknown parent address, unknown row index, a merely DMA-aligned
base, and a short vector row. No verifier or admission limit is relaxed.

The references enumerate supplied plain/fractal physical layouts independently
of compiler cell partitions, generators and retained demand records. They
reuse existing generic graph-closure and physical-command checkers. They do
not establish native footprint semantics independently. GM effects are included
in the selected conflict model, but the new access-agreement and selector
checks qualify local storage only; complete exact GM recovery is not claimed.

No ordinary BF16 move contract is added. The original upstream row reproducer
also uses opaque GM partition arguments; their missing layout/stride evidence
must not be inferred from this synthetic fixture. Its complete Frontier result
is not claimed. Completion's coarser storage classes can still merge adjoining
ranges, so narrower shared bounds do not promise improvement in every fallback.

## Scaling and reproduction

Synthetic derivatives vary both row count and backing extent: 8, 64 and 256.
Original effect count stays **8**, and original symbolic row-region count stays
**2**. Mathematics-only diagnostic requests retain 26/138/522 shared arena
nodes and 11/67/259 root ports; storage-boundary construction grows with the
physical row partition. Ordinary production logical requests retain **3 SET
sites, 3 WAIT sites, 2 barriers and 39 arithmetic operations** at all three
sizes. Allocation is disabled in this scaling request. Single wall times around
0.26 seconds are descriptive; no paired compile-time or device-speed claim is
made. The small physical cases separately allocate under six IDs.

Use the pinned Linux environment rather than the stale checkout `env.sh`:

```sh
source .local/prefix-child-composition-20261022/linux-env.sh
cmake --build "$PTO_BUILD_DIR" --target pto-test-opt pto-sync-input-test --parallel 2
python test/lit/pto/Inputs/check_frontier_upstream_access.py \
  --optimizer "$PTO_BUILD_DIR/tools/pto-test-opt/pto-test-opt" \
  --trace-tool "$PTO_BUILD_DIR/tools/pto-test-opt/pto-sync-input-test" \
  --output .local/frontier-upstream-benefits-20261023/recheck
```

The build uses the separate Linux source snapshot. Copy changed files there
with fresh timestamps before rebuilding; the companion receipt pins their
hashes. Run the build alone under the machine-wide two-worker limit. The full
focused command and raw comparison/scaling artifacts are under
`.local/frontier-upstream-benefits-20261023/`; tool hashes and compact results
are in the companion JSON. The comparison uses production requests, not an
all-form audit. No fresh whole-suite, native C++ backend build, device compile,
device execution or sanitizer campaign is claimed.

| Established evidence | Remaining premise or limitation | Appropriate claim |
|---|---|---|
| Frontier consumes refined ACC and instruction-specific L1 ranges | Supplied native footprint/layout contract; enclosures retain that precision label | Shared access refinements can reduce selected conflicts and emitted ordering |
| Complete aligned row views retain original symbolic coordinates | Narrow declared bounds/alignment contract; unsupported views stay conservative | Finite/regional services can exploit exact modeled row effects without an occurrence-free independence exemption |
| Closed expanded finite plans use the existing regional allocation certificate | Unbound enclosing visits require a separate rearming contract; individual proof and budget checks can still refuse | A supported mathematical representation can share existing allocation exports |
| Bounded commands cover requirements and obey six-ID causal reuse | Realization equality is not uniform; two repeated arithmetic cover expansions unavailable to this checker | Host-model evidence for selected executions and qualified resource feasibility |
| Constant emitted row-control size on the scaling derivatives | Boundary interfaces still grow with physical rows | Compact executable control does not imply constant total construction size |

## Acceptance review

Independent correctness and generality reviews accepted the package after two
findings were addressed. The first tightened the test's two documented
arithmetic-cover limitations so they cannot silently spread to other fixtures.
The second required the closed-frame allocation guard described above, including
protection against pre-exported-query and cached-certificate bypasses. The paired
frame probes verify both successful closed allocation and refusal with useful
interfaces retained.

The review-corrected tools pass a fresh **22/22** focused gate (108.82 seconds),
including the additional frame probes in the access-benefits test. The final
build introduces no compiler warnings; the changed-code prefilter, Ruff F checks
and diff whitespace checks pass. The final eight production comparisons retain
the ordering and command counts above. The receipt keeps the initial and final
build identities separate and records both reviews' accepted scope. No review
or host-model test establishes device behavior or a general open-child reuse
certificate.
