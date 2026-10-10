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
instances. The shared description retains both conditional arms; each producer prunes
proved-dead arms before charging its occurrence enumeration. Fragment charging
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

## Finite-expansion queries in one invocation

Expanded finite occurrences now export arbitrary start/completion reachability
and presence within one invocation of their original selected roots. Each type
retains the original phase and its complete fixed loop-coordinate tuple; the
local event ordinal is zero. Enclosing SSA bindings remain parameters of this
invocation. Nonzero local ordinals are absent, malformed identities are rejected,
and external visit coordinates are rejected rather than silently substituted.
This does not claim a cross-invocation repetition interface.

The session caches this query-only snapshot independently of exact demands.
Stronger selector or synchronization requests retain both owners and report the
missing capability. Expanded physical storage boundaries and endpoint recipes
remain unsupported until their coordinate adapters are implemented. The legacy
finite-guarded exporter remains blocked for expansions to prevent it from
claiming those capabilities. Query callbacks retain the expanded program,
shared rank state and modeled input across session reset. A foreign input owner
cannot be attached to the snapshot. The three pinned paper hashes are unchanged.

The independent expansion oracle compares all start/completion query pairs with
closed execution graphs under both alias policies, including absent and reflexive
events, zero trips, non-unit steps, symbolic parity, nested coordinates and the
three pinned GEMM inner loops. The session probe checks fixed-coordinate maps,
strong-export failure retention, nonzero-ordinal absence, malformed identities,
foreign input rejection and query lifetime after dropping the mathematical handle
and resetting the session. All 32 focused checks, the 20 varying command oracle
cases and 22 serial corpus runs pass. Every case retains its previous coverage
(227/240 and 212/240); no timeout or original IR mutation occurs. The slowest
corpus invocation is 14.3 seconds. Optimized affected-unit builds and both links,
changed-code compliance and diff checks pass.

## Fixed branch proofs during finite normalization

The original-coordinate finite-expansion adapter now evaluates a conditional
only when its complete scalar expression has concrete operands in the current
fixed loop-coordinate environment. It uses the original operation widths and
scoped data layout, with dialect folders operating on detached copies. The
memo is local to that environment. Unknown operands, invalid folds, poison
attributes, overflow or fast-math promises and unsupported index widths retain
the existing guarded attempt and its obligations.

A proved inactive arm consumes no payload or pair budget. The selected arm
retains its original branch identity, arm and coordinates, together with a
proof that its predicate holds in that environment. This proof avoids a second
interpretation through mathematical integer predicates. Original control
prerequisites and selected leaf contracts still apply. Runtime conditionals
retain both potential arms. Folding work and pruned arms are recorded separately
from visited operations and relation fragments. This extends finite normalization;
it introduces no additional tractable class or kernel-specific recognizer.
The three pinned paper hashes remain unchanged.

The initial branch-pruning corpus audit exposed a MayAlias elementwise timeout
in the general arithmetic importer's quotient projection. An optimized
line-table profile located the expensive construction before overlap joins.
Finite demand construction now has a general common-translation adapter:
within one `(space, base)` family, every nonempty access must provide complete
raw guarded rows and the same parameter-only translation. It substitutes old
byte = translated byte + that translation, reuses the selected residue period,
and regenerates the normalized relations. Missing or differing recipes leave
the family unchanged. Alias requirements between different bases remain intact.
Original physical primitives remain retained for queries and future selectors.

This optional representation uses the existing finite fragment cap independently
of the original normalization. Its attempted fragments are charged separately,
including a failed adapter followed by the original description. No external
watchdog becomes a class criterion. Recipes follow the original primitive table
through P1/P2 construction and rollback. The independent checks compare original
and translated generator/native unions, actual selected dependency graphs,
runtime guards, signed bindings, division/remainder origins and P2 residues;
missing coverage and differing translations exercise the unchanged-family path.

Translation expressions are canonically simplified after checked construction,
so visit-dependent constant offsets do not hide a common parameter translation.
No wrapping or guard constraint is removed by that comparison. The previously
timing-out elementwise MayAlias case completes in 0.08 seconds and gains one
exact regional certificate. All 22 serial pinned corpus invocations pass with
227/240 and 213/240 recognized regions; no case loses coverage, times out or
mutates the original IR. The slowest invocation is 14.1 seconds.

### Exact mapped integer predicates

The coordinate adapter for future finite storage exports evaluates affine maps
of original signed inputs before applying the relation's period quotient and
residue. This order preserves negative offsets, residue carries and mathematical
coordinates outside signed 64-bit range. It retains the existing integer-system
representation and shared expression arena; it does not introduce a demand
representation or relational conversion.

Acceptance checks every coordinate intermediate and relation expression against
a uniform signed-128-bit bound. Unsupported maps or unsafe bounds return a
separate diagnostic without changing or poisoning the expression arena. Folding
uses exact integers, emission uses signed 128-bit arithmetic, substitution retains
the maps, and expression identity includes them. Generic relational lowering
explicitly reports that this adapter is not implemented there.

The regression compares folding, substitution and emitted arithmetic against an
independent integer oracle at both signed 64-bit extrema and negative/positive
samples, including floor, ceil and Euclidean remainder. It checks overflow refusal,
invalid dimensions/symbols, zero-dimensional predicates and both polarities of
the explicit relational-lowering failure. This prerequisite does not yet change
regional recognition coverage or advertise finite storage exports. Section 8 and
the pinned paper hashes were rechecked and remain unchanged.

### Finite expanded storage exports

A stronger finite-region request now constructs and caches exact storage
selectors separately from the retained demands and query snapshot. It exports
byte-family support, first/last writers, readers before the first writer and
after the last writer per pipe, and native/per-site payload extrema. Original
phase/effect identities and fixed induction coordinates remain public; read/write
occurrences do not count as readers before or after their own write.

The demand producer retains its successful common-translation alternative and
applied-family metadata once. Storage construction reuses that owner; physical
byte queries apply byte minus the family's translation before quotient/residue
conversion. Untranslated families and native boundaries keep ordinary integer
predicates. All support and boundary predicates are preflighted in a private
arena, with symbolic physical bytes and the actual parameter SSA inputs. Failed
native or physical bounds checks cannot poison the cached query arena. Checks
and attempted builds are counted even on failure and are not repeated on retry.

The storage certificate is a general symbolic family certificate, not a uniform
atom certificate or an additional crossing adapter. Different potentially
aliasing GM bases return unavailable; no-alias bases and different spaces remain
disjoint. Discharged-effect storage exports and synchronization preparation remain
explicitly unavailable. A missing stronger export leaves exact demands and the
weaker immutable query snapshot available.

The independent storage oracle covers 20 cases across both alias policies,
including zero trips, runtime/fixed branches, P2 bank selection, read/write
accumulators, signed translated origins and division/remainder origins. It requires
all 32 requested physical byte probes, checks every selector against source-derived
reads/writes, and checks native/physical preflight refusal and unchanged query
snapshots. Session checks cover cache reuse, foreign input rejection and callbacks
surviving session destruction. Optimized selector measurements on the three GEMMs
complete in approximately 0.15 seconds; elementwise MayAlias completes in 0.08
seconds after previously exceeding the 30-second measurement watchdog. The
MayNotAlias discharged-effect export remains explicitly unavailable. These are
focused export measurements, not the final paired corpus campaign.

The optimized rebuild, 32 focused checks and 12 common-translation graph cases
pass. All 22 pinned corpus invocations preserve coverage at 227/240 under
MayNotAlias and 213/240 under MayAlias, with no per-case loss, timeout or source
mutation. The slowest invocation is 13.8 seconds. The three pinned paper hashes
remain unchanged.

### Request-driven regional arithmetic exports

The canonical regional arithmetic producer now retains only the accepted program
and its exact reduced demands. Demand-only requests do not initialize coordinate
geometry, import query relations, project storage, build selectors or prepare
commands. Queries and selectors have separate attempted/success/error caches;
stronger requests reuse the same mathematical owner without repeating reduction.
Sequence composition uses the same canonical cache through its export resolver.

Each export snapshot has independent callback state in the common expression
arena. Query-only construction imports occurrence/context primitives and skips
access projection and selector construction. A stronger snapshot imports its own
program-relative pointers and builds storage interfaces. A rejected transaction
publishes no callbacks or IDs, rolls back appended expressions and diagnostics,
and preserves prior queries. Original input owners survive session reset. The
finite-only compatibility probe keeps its early symbolic-storage deferral before
reduction; specialized entry bindings remain in the compatibility adapter.

Logical preparation consumes the retained demands independently and creates
fresh detached fragments. Deferred handoff allocation also uses a transaction,
including legacy and composed crossing recipes. A missing allocation certificate
leaves the logical fragments and existing query arena unchanged. Successful
allocation publishes its complete summary before committing the new expressions.

The focused regression checks empty optional state after demand construction,
query-only interfaces, separate query/selector retries through the session,
stateful-evaluation refusal, unchanged mathematical owners, selector failure
followed by weaker queries, common-arena identity, surviving session reset and
isolated input ownership. It compares disjoint detached operations in two live
plans, destroys one, rejects a wrong enclosing visit context, and checks original
IR and the remaining plan. Both allocation success and deliberately unavailable
reuse support preserve logical fragments and query results.

The stable optimized rebuild and final three-unit rebuild pass. All 32 focused
checks and 22 serial pinned-input invocations pass, with no per-case coverage
loss, source mutation or timeout. Coverage remains 227/240 under MayNotAlias and
213/240 under MayAlias; the slowest invocation is 13.7 seconds. Compliance reports
zero errors and warnings. The three paper hashes were rechecked and unchanged.
This milestone separates regional arithmetic capabilities; it does not complete
normalization ordering, specialized-context migration or the final paired corpus
campaign.

### Original-occurrence scalar prerequisite composition

The structural index now records whether every traced SSA path is direct data
flow. Control paths, conditional results, loop bounds and carried values cannot
acquire that certificate by merging with a direct path. A producer/consumer pair
keeps its original identities and synchronous/native classification. This is a
dependency certificate, not an instruction replay contract.

Regional relation exports preserve prerequisites whose producers lie outside
the region. Sequential composition retains these through intermediate merges,
then resolves an entering producer against the original consumer occurrence.
Direct prerequisites join exact occurrence domains at shared parameter bindings;
native prerequisites enter native order and asynchronous ones enter the complete
crossing requirements before joint reduction. Native export closure reuses the
existing arithmetic relation engine rather than rerunning its producer/reducer.

Cartesian crossing construction now explicitly checks all represented source and
target scopes, including accumulated children. Whole-left-before-whole-right
order requires distinct sequential scopes in the already certified supported
structured control. Shared dynamically enumerated loops, reversed siblings,
overlapping scopes and opposite conditional arms are rejected. Body-local
siblings may share a bound outer invocation; fiberwise repetition composition
requires a separate adapter. Missing mapping support leaves retained exact
mathematics available.

The stable optimized rebuild and final one-unit rebuild pass. All 35 focused
checks pass, including 18 source-derived provenance checks across both alias
policies and the independent all-event native closure oracle. All 22 serial
pinned-input runs preserve per-case coverage: 227/240 under MayNotAlias and
213/240 under MayAlias, with no timeout or source mutation. The slowest run is
13.7 seconds. Compliance and whitespace checks pass; the three paper hashes
remain unchanged. This is focused milestone validation, not the final paired
optimized corpus campaign.

### Retained finite-expansion preflight

Complete small-count expansion now separates its bounded control traversal from
access and relation construction. The immutable preflight retains original
roots, payload identities, fixed coordinates, selected branch proofs and
charged visit/folding counts. Existing visit, depth, payload and pair limits
bound this optional normalization; exhausting them leaves other exact routes
available and does not assert that the region is outside a tractable class.

The session caches one preflight per unchanged original region. Demand
construction consumes that description, and query/selector retries reuse it.
Standalone compatibility calls perform the same preflight locally. A retained
plan belongs to its exact modeled input and structural index; materialization
with another input or index is rejected before relation construction.

This split supplies counts for subsequent candidate ordering. It does not yet
change backend order, make numerical-template construction lazy, or select
between original and expanded descriptions by estimated cost.

Both optimized rebuilds and links pass. All 35 focused checks and 22 serial
pinned-input runs pass, retaining per-case coverage at 227/240 under MayNotAlias
and 213/240 under MayAlias. There is no timeout or source mutation; the slowest
run is 14.0 seconds. Compatibility mappings, foreign input/index refusal, failed
preflight retries, export retries and alias-context resets are checked. Compliance
and whitespace checks pass, and the paper hashes are unchanged. This is focused
validation rather than the final paired corpus campaign.

### Lazy numerical-template preflight

Structural recognition no longer enumerates numerical-template bodies or builds
their effects. A bounded preflight records selected original payload identities
and fixed inner-loop coordinates, while preserving the existing conservative
checks and visit/payload accounting across both constant conditional arms. It
constructs no access fragments, storage atoms, demands, endpoints or allocation.

An explicit numerical analysis request materializes effects from the retained
coordinate environment, then uses the unchanged numerical periodic producer and
reducer. Original-region session requests cache the preflight and materialized
result independently of later exports. Foreign modeled-input or structural-index
identities are rejected before materialization. Specialized geometry/control
compatibility calls use the same split locally; their borrowed callback contexts
are not merged into the original-region cache.

This milestone removes eager numerical construction. Estimated candidate order
and specialized composition caches remain separate work. Section 8 and the two
associated paper sections were reread; their pinned hashes remain unchanged.

Optimized builds (19 translation units, then one final test-driver unit) and
links pass. All 35 focused checks pass, including the 21 numerical source cases
under both alias policies, successful/failed-plan retries, foreign-context
refusal and alias reset. All 22 serial pinned-input runs preserve every region's
exact-result, certification, class, representation and interface availability:
227/240 under MayNotAlias and 213/240 under MayAlias, with no timeout or IR
mutation; the slowest run is 14.0 seconds. Compliance and whitespace checks pass.
This is focused milestone validation, not the final paired optimized corpus
campaign.

### Arithmetic reducer ranking before generator construction

Arithmetic reducer selection now estimates work from the certified primitive
schemas, access-pair joins, residue pieces, normalized row sizes, coordinate
dimensions and requested interfaces. Ranking performs no generator projection
or reduction. Counts use checked arithmetic; unknown or overflowing estimates
remain available in stable method order after known estimates. These are
selection heuristics, not certified output-size or complexity bounds.

Root requests retain the implemented periodic-conversion and arithmetic paths.
A periodic skeleton failure retains its adapter diagnostic separately from
cost availability. Regional requests list the implemented arithmetic adapter;
regional periodic conversion remains unsupported. Ranking caches include region
and requested capabilities/mode, and regional construction records consult the
canonical cache shared with composition. Alias-context reset clears all ranks.

A failed sequence attempt no longer scans every finite-visit candidate. The
common dispatcher requests the relevant region's cached finite-visit analysis;
the explicit compatibility operation remains available when intentionally asked.
This milestone does not implement original-versus-expanded candidate ordering.
Section 8 was reread and its pinned content hash remains unchanged.

Both optimized builds and links pass (12 translation units, then two final
units). All 35 focused checks pass, including zero producer construction during
repeated root/regional ranking, stronger-request separation, overflow/unknown
ordering, one shared generator construction across reducers and alias reset.
All 22 serial pinned-input runs preserve per-region classes, representations,
exact-result status and query/selector availability: 227/240 under MayNotAlias,
213/240 under MayAlias. No timeout or source mutation occurs; the slowest run is
14.0 seconds. Compliance and whitespace checks pass. Final paired corpus
performance acceptance remains a separate integration gate.

### Owned original-region numerical mathematics

Numerical analysis now retains its regional template and periodic result in
one owned mathematical handle. The canonical loop cache is shared by direct
regional requests, sequence child construction and eligible whole-function
requests. Regional template validity is independent of root eligibility: a root
request still requires one original loop covering all payloads and the separate
outside-loop checks. Surrounding payloads cannot promote child success to a
whole-function certificate.

Preflight keys distinguish whole-scope compatibility diagnostics from regional
analysis. Specialized phase/substitution callbacks remain private contexts.
When the session resolver is present, sequence construction consumes its owned
numerical form and reduction; an unavailable cached attempt cannot trigger a
second raw numerical recognition. Standalone compatibility calls keep their
existing local construction.

Original phase/coordinate/cut mapping is separated from command recipes.
Sequence children retain the numerical form and instantiate an immutable recipe
only during preparation, then create fresh detached command fragments on each
call. Mathematical requests construct neither command recipes nor physical
allocation. Numerical regional storage-selector and enclosing-visit emission
adapters remain explicit unsupported capabilities; their failure preserves the
owned periodic demands while later routes remain available.

Section 8 was reread and all three pinned paper hashes remain unchanged.

Whole-invocation preparation uses the same insertion validator as the legacy
adapter, preserving bounds, empty-invocation, scope, preexisting-sync, recipe,
layout and legal-cut checks. Recipe correspondence validation reconstructs the
expected recipe; it does not rerun the numerical producer or periodic reducer.

Verification: optimized builds and both links pass (142 affected translation
units, then 33 for shared validation, then one test-driver unit). All 37 focused
checks pass. The new fixture checks retained demands after export failure,
independent root eligibility, sequence reuse, zero trips, alias isolation,
parentless/disjoint detached operations, surviving code after destroying a
sibling plan, unchanged original IR and direct preexisting-sync rejection.
The existing sequence oracle's library portion passes 32 actual command-graph
closure cases plus unavailable-cut replay. Its later public insertion check
reaches the intentional recognition-only stopping boundary; the complete old
script is not reported as passing. A numerical allocation-session check passes
with one cached allocation export after logical preparation.

All 22 serial pinned-input runs preserve every region's class, representation,
exact-result status and query/selector availability: 227/240 under MayNotAlias,
213/240 under MayAlias. No timeout or source mutation occurs; the slowest run is
13.9 seconds. Compliance and whitespace checks pass. This is a milestone
regression check, not the final paired optimized corpus campaign.

### Shared normalized control input with backend occurrence construction

Numerical and finite adapters now consume one immutable control description per
original region in the unchanged session. It retains original operations,
both conditional arms and original loop identities. Optional expansion uses
constant lower bounds and positive steps, preferring innermost loops. Planning
visits each original descriptor once before constructing an expanded view.
There is no speculative nested expansion and compact retry. If the optional
view exceeds its size limit, the compact original description remains valid.
The structural description is not a demand representation or a certificate
that all backend interfaces are available.

Each producer constructs occurrences in its own representation. Retained
finite domains can become known under inherited outer coordinates; interpreting
these domains is charged producer work, not another source normalization.
Coordinates merge by original loop identity and conflicting bindings fail.
Runtime arms remain guarded. Proven-dead arms are pruned before producer
enumeration limits apply. Numerical depth and visit limits remain effective
when a caller supplies a description made with larger limits.

This slice centralizes input normalization and its lifetime. It does not yet
select original versus expanded exact forms by cost, centralize specialized
phase contexts, or replace the existing order of exact attempts. Section 8's
SmallCountCandidates and ExactForm requirements were reread; their pinned
content is unchanged.

Verification: optimized builds and both links pass (56 affected translation
units, then two units for fallback counts and two for defensive context
validation). All 39 focused checks pass, including both alias policies,
null/duplicate/displaced root selection, exact finite visit boundaries,
independent numerical depth/visit limits, zero/one/non-unit coordinates,
dependent retained bounds, a dead 5,000-trip arm and four nested large loops
under a dead guard. Compact counts match the shared phase/effect records;
planning visits equal the original structural descriptor size. Neither
mathematical adapter prepares commands or allocation.

All 22 serial pinned-input runs preserve every region's class, representation,
exact-result status and query/selector availability: 227/240 under MayNotAlias
and 213/240 under MayAlias. No timeout or source mutation occurs; the slowest
run is 13.7 seconds. Compliance reports zero errors/warnings and whitespace
checks pass. These focused milestone results are not the final paired corpus
performance campaign.

### Original and expanded exact-form candidates

The session ranks the original description and its single optional small-count
alternative before mathematical construction. A compact fallback is the original
candidate, not a second alternative. The original description wins equal estimates;
known work precedes unknown work, then representation size breaks ties. Unknown
estimates leave the candidate eligible. Interface requests have separate cost-cache
keys while sharing the immutable normalized input and numerical control preflight.

Numerical estimates count the occurrences enumerated by the retained preflight,
including residual dependent finite domains, and shared effects at those occurrences.
Normalization work is charged separately. Original rotating descriptions receive a
known estimate only when their complete rotating form has been certified; other
original descriptions remain eligible with unknown work. These estimates select
implemented alternatives and do not certify class membership or impose admission
limits.

The original attempt cannot silently construct a numerical result for the same
expanded region through Sequence. It may reuse mathematical evidence already
retained by an earlier attempt. Proper original descendants can still select their
own one-time alternatives. Root promotion shares normalization and finite mathematics
only when the unchanged function contains a complete admitted loop with no external
payload; original scope checks remain independent of this sharing.

Expanded finite exports share the session expression context. Predicate import and
reduction publish only after their expression transaction commits. Ownership and
rollback are checked by source review and the generic transaction tests; this
milestone does not inject a finite reducer failure after predicate import.

A supported relational adapter preserves parameter-free fixed original occurrences
as distinct sites, with no added dynamic relation dimensions. Repeated phase anchors
must have disjoint fixed bindings. Scalar prerequisites pair compatible visits;
mixed fixed/dynamic bindings of the same loop and parameterized fixed-occurrence
relations remain explicit unavailable adapters. This limitation retains the exact
child mathematics and permits later exact attempts. Fixed-occurrence preparation
remains unsupported and is tested to leave original IR unchanged.

Sequential relational composition also carries shared modeled uniform conflicts
between potentially aliasing physical bases for flat occurrence types. Dynamic
uniform crossings require a first/last occurrence adapter and remain an explicit
unavailable export until that adapter is implemented. Their occurrence-domain products join
the byte-level crossings before the existing complete crossing reduction. It retains
the shared scalar protection policy, deferred-effect obligations and original
sequential-scope checks. No physical address precision admission rule is added.

Candidate-order milestone verification: 39 focused checks, 32 independently checked
command closures and unavailable-cut replay, and the regional expression suite
(including generic transaction rollback) pass. All 22 pinned runs preserve source
IR and finish without a timeout. Per-region comparison finds no coverage loss:
MayNotAlias remains 227/240, and MayAlias increases from 213/240 to 214/240.
This is recognition/mathematical coverage; the public insertion boundary remains
unchanged. The three pinned paper section hashes remain unchanged.

## Uniform crossing extrema and deferred access scopes

Dynamic uniform conflicts now compose through exact occurrence extrema. The
relational adapter joins the source's last-site selector and target's first-site
selector, preserving parameter residues, rational witness equalities and original
visit coordinates. Byte bridges and native crossings still participate in one
complete crossing reduction. Missing extrema remain an explicit obligation.

The shared circuit adapter accepts the same uniform conflicts directly from
active or deferred access boundaries. It retains deferred identities and extrema
through nested composition, avoiding finite byte projection and relational
conversion when the existing circuit interface suffices. A conflicting deferred
nonuniform access still requires an exact byte adapter. Uniform alias predicates
come from the shared access model; this change invents no physical addresses.

A discharged GM writer can occur in multiple selected slices of its original
loop. The adapter revalidates the existing disjoint-visit footprint proof before
accepting distinct constant ordinal intervals of the same effect and outermost
original loop. Fixed coordinates, nested visit frames and overlapping intervals
do not use this exemption. The proof consumes the identical full-invocation
access model on both sides; detached composition needs no new shared-input
construction. The elementwise pipeline's split store slices exercise this path.

Exact integer inclusion now avoids feasibility projection for syntactically
implied atoms. A mismatching fixed residue still checks source emptiness. Seeded
union subtraction removes only right-hand atoms syntactically implied by the
seed; every arrangement frame remains restricted to that seed. These changes
remove redundant work without introducing analysis budgets or class exclusions.

Regression checks independently enumerate dynamic non-unit trip domains and
uniform required order/minimum crossings, including zero trips and denominator
scaling. Direct circuit checks remove relational providers, require no projected
finite crossing pairs, test missing extrema and nonuniform refusal, and repeat
the oracle after nested composition. Actual guarded regional exports test
disjoint writer slices and refusal of overlapping slices. Integer tests cover
uncached empty sources, structural inclusion and seeded differences.

The three paper hashes remain unchanged. The public pass retains its requested
recognition-and-analysis boundary; synchronization insertion and allocation are
not enabled by this milestone. Parameterized fixed-occurrence relational
conversion and general deferred nonuniform crossing adapters remain explicit
unsupported capabilities.

Validation: all 39 focused checks pass. The 22 serial pinned corpus runs retain
all previously exact regions, preserve input IR and introduce no timeouts.
Coverage increases from 227/240 to 228/240 under MayNotAlias and from 214/240 to
218/240 under MayAlias. The added exact regions are `gdn_chunk_cumsum` node 4
under both policies and `lossless_block_cast` nodes 0, 2 and 3 under MayAlias.
The slowest final run is 14.2 seconds. This is focused milestone validation,
not the final paired optimized cross-project corpus campaign.

### Specialized phase mathematics and exports

Phase and invariant-body composition now request specialized arithmetic
mathematics through their owning analysis session. The cache includes the original
region, configured arithmetic profiles and the complete observed entry-constant
lookup set, including unavailable values. A changed constant or newly available
value cannot reuse an incompatible reduction. The retained mathematical handle
owns its shared input and structural index; it contains no expression-arena IDs.

Exports instantiate fresh query and selector expressions in the requesting arena.
They validate every consumed entry constant and preserve the specialized counted
loop origin and step in the inverse mapping to original occurrences. Failed
exports roll back their expression IDs. Ordinary arithmetic adapters reject
specialized handles rather than dropping those assumptions. Alias-context reset
clears the cache while already returned handles retain their original inputs.

The regression uses nonzero lower bounds 1 and 3, a step of 2, unknown and known
upper bounds, zero and reversed trip domains, exact first/last writer ordinals,
contradictory and missing bindings, export retry after rollback, profile isolation,
and export of an old retained handle after context reset. This does not supply a
general bounded-span proof or enable synchronization insertion in the public pass.
