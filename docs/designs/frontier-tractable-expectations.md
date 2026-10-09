# Frontier tractable-class correspondence and source expectations

This ledger precedes changes to recognizer decisions. The working-tree paper
and eleven prepared inputs are pinned in `frontier-tractable-catalog.json`.
Recognition failure is not a proof that no tractable representation exists.
Paper sources are external to this checkout under
`/home/toni/work/synchronization_draft/paper`; the catalog records content
hashes rather than treating the external path as a portable build dependency.
The scope of any negative conclusion must name the catalog, arithmetic profiles,
and normalization policy. A missing extraction or certificate procedure is an
unresolved obligation, never an outside-class witness.

## Correspondence

| Paper construction | Premises to certify | Mathematical result and composition interface |
| --- | --- | --- |
| Finite occurrences (§5) | Complete finite ordered sites, pipes, modeled effects and prerequisites | Exact nonnative covers; native/order queries and storage selectors |
| Finite guarded occurrences (§5, §8) | Finite potential sites; exact occurrence/access/prerequisite predicates | Guarded covers and exact queries/selectors; late guards affect code availability separately |
| Bounded lifetime (§5.4) | Fixed potential sites/pipes, exact predicates, explicitly represented cells, supplied generator-span certificate or sufficient refresh certificate | Shared window circuit for exact covers; window rows alone do not certify arbitrary regional queries |
| Periodic generators (§5.4) | Finite periodic skeleton; complete reference-forward generator union; uniform exact coverage certificate | Periodic reduction of merged generators; queries/selectors require their supported mathematical adapters |
| Common-stride storage (§5.4) | Exact disjoint within-slot atoms; one constant stride per family; shift permutes cells; writer orbit refresh; complete external and prerequisite treatment | Producer of periodic generators, not a separate reducer |
| Mixed fixed strides (§5.4) | Exact fragments; pair period b/gcd(s1−s2,b); complete coverage/refresh and exception treatment | Charged numerical phase expansion produces periodic generators; no common-stride encoded-size bound is claimed |
| Immutable guarded rotation (§5.4, §8) | Common-stride storage under an invocation-invariant guard valuation; complete guarded generators | Shared guarded reduction and the corresponding guarded mathematical interfaces |
| Restricted arithmetic (§5.4) | Complete primitive relations in declared fixed k,D,P,C; parameters and auxiliary coordinates counted; exact machine semantics | Difference, octagon or bounded-coefficient exact covers, queries and requested access selectors |
| Counted writer/readers (counted composition appendix) | A persistent cell; writer pipe p and reader pipe q with p≠q; immutable N,c≥0 and fixed a≥0; affine-length read-only visits, only stated effects and no additional prerequisites (extras require separately certified composition) | First-reader/last-reader and empty-visit crossings, retaining internal demands |
| Persistent rotating inner storage (counted composition appendix) | Child compact storage interface; exact re-entry certificate | Internal demands plus jointly reduced complete crossings |
| Sequence/conditional (§8) | Exact child queries and selectors under shared entry bindings; all storage, native and supplied-prerequisite crossings | Combined exact order; child success alone is insufficient |
| Certified base plus finite exceptions (§8) | Supplied complete decomposition; exact base covers and all-event queries; every exception is one actual forward edge on existing endpoints | Refilter base covers with exception crossings; combined queries; whole-region storage selectors need independent certification |
| Repetition (§8) | In paper order: rotating visits, invariant phase expansion, repeating boundary/startup-suffix certificate, shared numerical repetition, finite visit types, bounded startup/drain | Route-specific exact demands; finite visit types do not promise general selectors or executable matching |
| Small fixed counts (§8) | Charged expansion within declared caps; exact coordinate/carried-state/effect/cut mapping | Optional representation of the original region; reported class is the class of that representation |

## Independent source inspection

Node numbers refer to the original structural tree. Bounds below come from
constant definitions and `scf.for` operands in the prepared inputs. They are
not inferred from recognizer output. The two runtime bounds are derived from scalar values
loaded before their loop; signed casts and possible wrapping require proof. A runtime bound does not by itself exclude any class.

| Kernel | Loop node(s), trip counts | Source fact and required distinction |
| --- | --- | --- |
| tilelang_gemm | 2:2, 5:16, 11:4 | L1 banks alternate modulo two. Inner L1 extraction advances by 64 elements, so a single invariant within-slot inner footprint is false. Expanding the four inner visits makes their individual slices fixed. Initial MATMUL and final-prefetch guards must retain their original coordinates. |
| pypto_gemm | 2:2, 5:16, 9:4 | Two carried banks update as `(bank+1) mod 2`, initially one. Same moving L1 slices as GEMM above. Equivalent normalized bank spelling must receive equivalent class decisions; inability to recover the recurrence is unresolved. |
| persistent_gemm | 2:13, 9:32, 15:4 | Task index is `20*wave+core`, admitted under task<256; quotient and remainder choose the output tile. Four advancing inner L1 slices can be expanded, but task guards and outer re-entry require separate certificates. |
| group_norm | 6:4, 8:4 | Two-bank UB accesses and last-prefetch exception. Subblock query is an entry binding for a regional loop. Finite four-trip expansion can represent the changing guard; direct immutable-guard certification cannot use an iteration-dependent guard. |
| causal_conv1d_prefill | 7:runtime | Bound is unsigned `(end-start+3)/4` after signed i32-to-index casts; intended nonnegative lengths and absence of wrapping require proof. Tail and next-prefetch predicates vary with iteration. Do not call the immutable-guard class applicable; check bounded span/refresh or complete arithmetic predicates, including integer casts. |
| causal_conv1d_decode | none | Finite potential sites and conditionals, including late scalar-loaded guards. Late guard availability is distinct from finite guarded mathematical membership. |
| gdn_chunk_cumsum | 2:8, 6:127 | Inner scalar read/write indices advance through a 128-element prefix. Direct fixed within-slot footprint is false; exact arithmetic accesses or charged finite expansion must preserve byte offsets and prerequisites. |
| mhc_head_mix | 2:3, 5:4 | Outer starts at one; scalar writes use `4*outer+inner`. Twelve visits are finite. Exact expansion must preserve the nonzero origin; byte coefficients need the declared arithmetic profile, not an implicit universal C=8. |
| elementwise_pipeline | 2:4 | Two-slot UB rotation, unsigned `i<3` next-prefetch guard, iteration-private GM output. Four-trip normalization is finite. Unsigned comparison requires exact nonnegative-range proof or exact unsigned semantics, not a signed reinterpretation. |
| gated_delta_rule | 2:2, 11:runtime | Inner bound is cast end-start; intended nonnegative lengths and absence of wrapping require proof; floating-point comparison guards state updates. These are not integer-arithmetic predicates. Bounded-lifetime applicability still depends on every written cell's refresh/span certificate, which must include skipped updates. |
| lossless_block_cast | 6:2, 8:64, 22:4, 9:2, 11:2, 12:64, 25:4, 13:64, 26:128 | Scalar access coordinates advance and include quotient/packing expressions. The final 64×128 nest exceeds a cheap full-expansion policy; that policy limit is not a class rejection. Check the complete arithmetic relation after exact quotient lowering, including hardware entry bindings. |

These are source expectations for particular premises, not an assertion that
all complex parents are recognized. For every enclosing sequence/conditional,
check complete crossings and context bindings independently. The catalog also
records the source line, induction SSA name and original loop-bound operands
beside every node ID. For roots, check
GM alias policy and scalar prerequisites as well. Neither a matched child nor
an ancestor's expanded template establishes a direct match for another node.

## Normalization counter scope

The retained outer loop is not expanded by the numerical-template producer.
The visit counter charges every visited original operation in the expanded
inner body, including scalar/control operations and terminators; it is not
simply the number of loop iterations. The payload counter charges shared phase
instances. The preflight walks both conditional arms conservatively; emission
of the normalized representation keeps the selected arm. Fragment charging
counts prepared access fragments, and depth bounds nested control traversal.
All counters are local to one proposed representation. Lossless's final nest
has 8,192 inner iterations and at least 16,384 payload instances; the latter
exceeds the 4,096 payload cap even though the iteration count is below 65,536.
A capped alternative is unavailable, not outside a tractable class.

## Review and implementation gates

Freeze positive certificates and negative witnesses only when fully derived.
An incomplete certificate remains unresolved in the expectation manifest.
Before a final acceptance claim, compare each original region to this ledger,
exercise equivalent spellings and small explicit cover/query oracles, and run
both alias policies serially. Record paper-hash changes before each milestone.
No emission, allocation or device-runtime correctness is claimed by this work.

## Verified certification checkpoint

The optimized local build uses `-O2`, C++17 and enabled assertions; these runs
are correctness audits, not release performance comparisons. The source is
an uncommitted candidate based on `254a9de15`. Paper and input hashes match the
catalog above. Corpus compilation runs serially with a 60-second external
watchdog per input/policy; that watchdog is not a class criterion.

The audit covers all 240 original structural nodes in the eleven inputs under
each policy. Every invocation verifies unchanged IR, no logical preparation or
allocation export, and a repeated certification request with no new mathematical
attempts or arithmetic contracts. Exact-result presence, class certification,
query availability and selector availability are recorded separately in JSON.

| Input | May-not-alias recognized / nodes | May-not-alias unresolved node IDs | May-alias recognized / nodes | May-alias unresolved node IDs |
| --- | --- | --- | --- | --- |
| tilelang_gemm | 20 / 22 | 7, 11 | 18 / 22 | 3, 5, 7, 11 |
| pypto_gemm | 13 / 17 | 3, 5, 7, 9 | 13 / 17 | 3, 5, 7, 9 |
| persistent_gemm | 30 / 34 | 0, 2, 11, 15 | 26 / 34 | 0, 2, 3, 5, 6, 9, 11, 15 |
| group_norm | 23 / 23 | none | 18 / 23 | 0, 2, 3, 6, 8 |
| causal_conv1d_prefill | 46 / 48 | 0, 7 | 46 / 48 | 0, 7 |
| causal_conv1d_decode | 7 / 7 | none | 7 / 7 | none |
| gdn_chunk_cumsum | 8 / 9 | 4 | 8 / 9 | 4 |
| mhc_head_mix | 8 / 8 | none | 8 / 8 | none |
| elementwise_pipeline | 10 / 10 | none | 8 / 10 | 0, 2 |
| gated_delta_rule | 23 / 29 | 0, 2, 3, 5, 6, 11 | 23 / 29 | 0, 2, 3, 5, 6, 11 |
| lossless_block_cast | 33 / 33 | none | 30 / 33 | 0, 2, 3 |

Totals are **221/240** and **205/240**, respectively; whole roots are recognized
for **8/11** and **5/11** inputs. All 22 invocations finish without timeout.
These counts establish implemented certificates, not full catalog completeness.
No unresolved node is reported as outside the catalog.

The remaining source obligations include advancing GEMM slice footprints and
carried bank state in direct regional requests; persistent outer re-entry;
exact cast/unsigned length semantics in prefill; and predicate-dependent refresh
with overlapping physical families in gated delta. Regional arithmetic coefficient
limits are genuine failures of those declared profiles, not universal exclusion
from arithmetic or other classes. Additional GM aliases require complete crossing
and storage certificates; the may-alias losses are not silently ignored.

The source ledger remains provisional for complete positive/negative certificates.
General finite expansion of each original child, complete bounded-span certificates,
exhaustive outside-catalog witnesses remain
unfinished. Parent recognition through a normalized representation does not certify
every original descendant directly.

Validation passes 26 focused checks and nine additional library checks, including
independent arithmetic, numerical-periodic, guarded-rotation and finite-guarded
oracles, exact-result retention after export failure, profile/context isolation,
cache reuse and unchanged public-pass IR. Direct library emission is exercised
with `--insert-logical-library`; the public pass still deliberately stops before
emission and allocation. Both architecture and correctness source reviews accept
this checkpoint; neither is a claim that the broader refactoring is complete.

Detailed JSON, per-run logs, exact compilation commands and binary/source hashes
are retained locally under `.local/section8-refactor/tractable-certification/`
(`corpus-results.json`, `corpus-provenance.json`, `validation-results.json`,
`additional-results.json`, `commands.json`).

## Finite-visit session milestone

Finite-visit types now have an owned session result and share the unchanged
structural index. The general dispatcher retains its earlier applicable routes;
`analyzeFiniteVisit(request)` explicitly requests this backend through the same
cache. Exact type-pair templates survive unavailable arbitrary-word queries,
selectors and synchronization exports. No emission or allocation is added.

The focused fixture verifies four complete types and sixteen ordered pairs,
child identity, root/child owner reuse, absence of child-only whole-region
evidence, a parent with an external payload, export-failure retention, repeated
requests, one structural index and both alias policies. The independent whole-visit
oracle checks common prefix/middle/suffix, shared atomic-node reuse, correlated
choices, projected visit-owned storage and the explicit type-expansion limit.
The fixture also admits bounded-lifetime analysis; generic dispatch correctly
keeps that earlier route rather than forcing finite-visit selection.

This milestone passes 29 focused/oracle checks and all 22 serial corpus
invocations, with unchanged recognition totals and no timeouts or IR mutations.
It does not resolve moving-slice GEMM children: those require a cached finite
guarded expansion preserving original coordinates and carried bank substitutions.

## Original-region finite occurrence expansion

The demands-only session can request one cached finite occurrence expansion
for a genuine original loop or conditional, or the complete function. Synthetic
sequence and explicit-run anchors are rejected: an anchor alone does not name
those regions. Expansion never clones IR or rebuilds the shared access model.
Each occurrence retains its original phase, branch path and complete fixed IV
tuple. Actual IV values preserve nonzero origins and nonunit positive steps;
shared scalar evolution supplies only proved carried recurrences.

The adapter specializes the original physical access maps before affine
normalization. Its integer overlap generator is a representation adapter, not
a restricted-arithmetic class claim. Runtime entry parameters remain symbolic.
Original prerequisite identities are paired within matching represented visits;
scoped hardware protection additionally requires matching scope and ancestor
coordinates. Guarded rank reduction owns the resulting exact demands.

The pinned alternative charges 65,536 scalar/control/payload visits, 4,096
payloads, depth 16 and 65,536 primitive fragments. A separate 65,536-pair bound
covers potential ordered payload pairs and compatible read/write or write/write
physical-piece joins; read/read joins are excluded. Products are checked before
construction or bounded by the fragment cap. Exhaustion makes this optional
representation unavailable, never outside a tractable class. Costs retain
actual visits, sites, access fragments, joins, generator edges and circuit nodes.

Repeated original anchors do not yet have certified query, storage-selector or
endpoint adapters. All three exports are explicitly unavailable; failed exports
retain the identical owned demand result and do not rerun expansion or reduction.
Unsupported predicate/scalar adapters and carried payload/control prerequisites
remain unresolved. This milestone does not implement emission or allocation.

Validation of this milestone passes all 31 focused/oracle checks and all 22
serial invocations of the eleven pinned inputs under both alias policies.
The independent oracle verifies byte footprints and dependency closure/covers
for all three four-visit GEMM children, including carried bank state and
accumulator reset/protection. Exact original-region certificates increase from
221/240 to 224/240 with MayNotAlias and from 205/240 to 211/240 with MayAlias.
Every run preserves original IR, performs no endpoint/allocation construction,
and finishes within the external measurement limit. Unresolved regions remain
unresolved; no outside-catalog conclusion is inferred from adapter failure.

Finite-only generator import omits projection of read primitives with no writer
in the same physical storage identity. This is the existing conflict-join
partition; complete original primitives remain owned for future interfaces.
Candidate validation uses fresh compilation of all 248 linked Frontier/tool
translation units at the recorded optimized flags, based on 69ece94a0371.
Commands, binary/input hashes, per-case timings and outcomes are retained in
`.local/section8-refactor/tractable-certification/` (`commands.json`,
`corpus-provenance.json`, `corpus-results.json`, `validation-results.json`).

## Exact root-list normalization

A sequence body, conditional arm or explicit run now supplies its exact adjacent
original operations to finite normalization. The owned context records those
roots and uses the first operation as the entry cut. The adapter never replaces
a body/arm with its enclosing synthetic anchor. Unsupported contexts, empty
lists and malformed/nonadjacent lists remain unavailable.

The collector charges one aggregate budget across the complete list. Incoming
prerequisites are classified against all selected roots; edges between selected
roots remain internal. Enclosing loop IVs are actual entry bindings, not new
occurrence coordinates. Proper inner fixed loops alone are expanded. Parent
branch presence is an entry condition, whereas nested branches retain their
own exact presence circuits. The mapped query/selector/emission obligations
of finite expansion remain explicit.

The Section 8, minimum-demands and counted-composition reference hashes were
rechecked before this milestone and match the pinned catalog.

Symbolic modulo-two access terms can select a parameter-residue adapter before
integer overlap projection. The same original parameter residues constrain
occurrences, accesses, native edges and prerequisites. Predicate circuits use
the retained period; byte residues are preserved as well. This avoids repeated
quotient projection when a symbolic bank has both readers and writers.

The original P1 primitive description is constructed first and supplies an
overflow-safe upper bound on the complete P2 description, including every
parameter and byte residue. P2 is attempted only when its whole description
fits the existing per-adapter fragment cap. P1 remains owned until P2 succeeds;
normalization failure restores it. Sites and parameter identities are built
once. Actual fragment counters charge both adapter constructions; the two
bounded primitive descriptions are separate from the once-only occurrence
expansion. Every admitted residue tuple is charged before construction.

The independent oracle checks both parities, negative and positive entry
parameters, two correlated symbolic banks, nonunit original steps, exact byte
sets and dependency covers. A smaller fragment allowance verifies that P1
remains available when P2 is too large.

This milestone passes the 31 focused/oracle checks and all 22 serial pinned
corpus invocations. Compared with the finite-expansion checkpoint, no case
loses a certificate: default coverage rises from 224/240 to 227/240 (the three
GEMM body sequences), and MayAlias coverage rises from 211/240 to 212/240
(a group-normalization sequence). Every run preserves IR and builds no
synchronization or allocation.

The first root-list prototype timed out during GEMM's symbolic bank overlap
projection. Parity normalization resolves that regression: the full TileLang
case takes approximately 1.5 seconds, and the slowest final corpus case takes
13.8 seconds. External timeouts remain measurement limits. Optimized binaries
are built from current sources based on 64e162fe1, with all 248 linked Frontier
and test-tool translation units refreshed; repaired units and final links
passed. Commands, hashes, stage outcomes and per-case timings remain in the
tractable-certification artifact directory.

## Regional entry expressions

After preserving existing supported integer and Boolean guard forms, the
arithmetic producer can bind an otherwise unsupported guard to its original
i1 result when it is a deterministic expression over actual values available
before the original regional entry cut. These leaves include values supplied
by preceding payloads and enclosing block arguments. Existing index parameter
handling remains unchanged. Unsupported integer-comparison predicates or
operands still report their existing normalization obligations. Interior operations
must still be deterministic, memory-effect-free and speculatable; their SSA
results remain the parameter bindings. Floating predicates are not translated
into integer predicates. Incoming completion prerequisites remain in the
original shared model, and executable replay is a separate export obligation.
The memo belongs to one producer and its unchanged region context; its key
also distinguishes legacy entry checks from the regional-leaf fallback.

Focused finite-expansion checks compare an external scalar with a distinct
zero threshold under ordered-greater and unordered-equal predicates, binding
negative/positive finite values, NaN and both signed zeros to the original
Boolean result. They also check that an enclosing IV is available for one
body invocation but cannot make an opaque predicate invariant for the whole
loop; an internal payload result and a non-speculatable integer divide remain
unavailable. Existing original-coordinate, alias, prerequisite and retained
export checks remain enabled. All three pinned paper hashes were rechecked
and remain unchanged for this milestone.

An initial eager-binding prototype added opaque parameters to affine GEMM
integer guards and exceeded the existing pair budget. The final selection keeps
those integer/Boolean normal forms and uses the regional-leaf check only at the
guard fallback. Rebuilding all five consumers of the private producer header
and both optimized test binaries passes. The final 31 focused/oracle checks and
22 serial corpus runs pass, with no per-case loss, timeout, IR mutation or
endpoint/allocation construction. Exact coverage remains 227/240 under
MayNotAlias and 212/240 under MayAlias. The initial prototype is not committed.

## Retained repeating-boundary demands

Affine-length rotating visits now have their own session backend. Its owned
mathematical result contains the entire child quotient and startup, seam and
suffix crossing covers. It also records the original recognized node; the
shared input and recognition tree supply the original phases, loops and
access identities. Root and regional requests reuse one canonical loop result.
The whole-invocation adapter still checks that no payload lies outside that
loop before recording whole-region evidence.

The independent raw-result request builds no arbitrary-event query, storage
selector, endpoint code or allocation. Requests for these still-unmigrated
exports report an explicit obligation without discarding or regenerating
mathematics. Existing exports remain available through sequence analysis;
the next integration step separates their query and selector
attempts as well. This chunk does not claim to complete that export migration.

Sequence repetition now tries the repeating-boundary adapter after rotating
and invariant-phase attempts. It consumes the retained session certificate;
a failed session producer cannot silently rerun analysis through the legacy
optional-certificate path. Existing outside payloads and unsupported additional
cross-visit prerequisites still require their own complete treatment.
The three pinned paper hashes are unchanged at this milestone boundary.

The repeating-boundary session probe checks all three unsupported exports and
retries, one producer reduction across root/child requests, no whole-region
promotion over external payloads, and retention across an alias-policy reset.
Both outer and inner payload-produced carried prerequisites are rejected on
valid VEC scalar-read inputs. The existing independent varying command oracle
checks 20 explicit execution closures, physical allocations and arbitrary-event
queries, including short visits and partial periods.

All 248 linked Frontier/tool translation units and both optimized binaries were
rebuilt successfully; the interrupted build resumed from its 75 verified units,
and the final CLI test edit was rebuilt before validation. All 32 focused
checks, the 20-execution varying oracle and 22 serial pinned corpus invocations
pass. Exact coverage remains 227/240 (MayNotAlias) and 212/240 (MayAlias), with
no per-case loss, timeout, IR mutation or synchronization/allocation construction
in certification runs. The slowest corpus invocation takes 13.9 seconds.
Compiler-generated dependency records now cover all 248 units for subsequent
header-aware rebuilds; command flags, hashes and detailed results remain in the
tractable-certification artifact directory.

## Independent repeating-boundary exports

The next milestone separates repeating-boundary event queries from physical
storage selectors. `AnalysisOutcome.regionalExports` carries an immutable
capability snapshot independently of the mathematical demand owner. Original
root, regional and sequence requests share the canonical loop provider and its
once-only query and selector attempts. A query-only request builds the port
transfer index but no storage selectors, detached commands or allocation.
Selector extension publishes a new snapshot including both native entry and
exit payloads; previously returned query snapshots remain unchanged.

Selector failures retain the query snapshot and original exact demands.
Foreign modeled-input or alias contexts are rejected before caching an export
attempt. Session providers retain their shared input and mathematical certificate
across session reset. Legacy standalone adapters retain their existing borrowed
input lifetime contract and now use the same split exporter. Synchronization
requests instantiate detached fragments from cached factories, while allocation
remains deferred. The three pinned paper hashes are unchanged.

Validation for the split exporter passes all 32 focused checks, the independent
20-execution varying command/allocation/query oracle and 22 serial pinned corpus
runs. Every case preserves its prior exact coverage: 227/240 under MayNotAlias,
212/240 under MayAlias, with no timeout or original IR mutation. The slowest run
is 14.5 seconds. The session probe additionally retains two independently
prepared detached plans, destroys one, and checks the other's surviving code,
unchanged original IR and once-only mathematical/query/selector construction.
Both the original-context retry after foreign-input rejection and old export
ownership across session reset are covered under both alias policies.
