# Section 5 implementation update

Reference: current `restart/sections/minimum_demands.tex`, including the October 7 working-tree revisions.

## Ordered milestones

1. Automatic rotating-child boundary certificates and affine varying-length repetition. Derive first/last selectors and stable residue types from normalized rotating accesses and the numerical quotient. Preserve startup, zero trips and original coordinates; construct crossing covers without unfolding visits.
2. Numerical chain-interface sweeps and crossing rank buckets. Reuse ordered numerical ports; retain the symbolic query path for symbolic contexts.
3. Finite guarded scan and rank reduction. Preserve native prerequisites and hardware protections, and replace the dense closure representation with guarded rank queries.
4. General bounded-lifetime analysis. Check supplied refresh/span certificates, construct local guarded windows and expose the analysis separately from endpoint availability.

Every route uses the shared modeled storage input. No address-precision admission category is introduced. Recognition, demand construction, regional exports and endpoint preparation remain separate capabilities. No event scarcity repair or SDMA extension is included.

## Validation

Compare demand sets and query answers against independently unfolded conflict graphs on short executions, including empty visits, residue transitions and conditional presence. Retain corpus regression coverage. Build serially with the existing isolated toolchain; do not rerun unrelated full suites.

## Status

Varying-length repetition now exports regional queries, boundary selectors and logical endpoint recipes. Bounded-lifetime loops now recover iteration-indexed predicates and prepare matching endpoint guards at original cuts. Physical allocation for these new representations is separate; neither this update nor recognition alone claims a hardware-ID assignment.

## Implemented contracts and remaining exports

- `RotatingBoundary` constructs head/tail selectors, residue-indexed boundary queries, startup/seam/suffix crossing demands, and keeps child demands as quotient records. `VaryingRotatingRecognition` checks a single rotating child with `K(t)=a*t+b`, positive numerical `a`, nonnegative numerical `b`, unchanged storage mapping, and machine-arithmetic validity. It does not relax the existing invariant-child recognizer. `VaryingRotatingRegional` exports arbitrary-event queries, first/last storage and native selectors, full nested occurrence identities, and detached endpoint recipes. Sequence composition consumes this export. Cached generator certificates are reused. Internal demands use each visit's actual trip count; startup, transition and suffix crossings have disjoint guards. Increasing lengths allow an empty first visit; no later empty visits are inferred.
- `ChainInterface` builds both numerical threshold directions with `O(kP)` semantic queries. It reduces a deduplicated crossing list in `O(kP+k^2*r)` word work after identity normalization and sorting. Rotating boundary certificates use this index; eligible two-child numerical sequence crossings use its reducer. Symbolic sequences retain their existing query implementation. This change does not establish the complete numerical hierarchy/query bound for all sequence results.
- `GuardedRanks` constructs conditional rank rows and cover guards. Finite guarded analysis now generates lifetime predicates per cell and uses these rows for both reduction and all-event queries. Target-protected pairs and phases of one macro operation retain their modeled pair generators when the ordinary lifetime-chain proof does not apply. Native first/last selectors use linear scans. Endpoint preparation keeps its existing guard-availability checks.
- `BoundedLifetime` accepts a finite relative predicate window and a supplied uniform lifetime-span contract, then returns covers sourced in its first iteration. Its query rows concern that window, not the global profile. A separate producer proves refresh from unconditional rotating writers covering every writable orbit; the IR recognizer permits iteration-varying participation. `IterationPredicates` now distinguishes a predicate's evaluations at different iteration ordinals. It uses the existing shared scalar-replay contract and preserves original integer operations/widths. Current-iteration values may be consumed when already available; other iterations require safe deterministic replay. A memory read or unavailable carried value is never speculated. `BoundedLifetimeInsertion` substitutes the source coordinate at SET and its inverse at WAIT and invokes the shared detached circuit emitter. Both endpoints must pass availability; failure is transactional. The production whole-loop route uses this constructor. The window's local rank rows are not advertised as global regional queries. Streaming profile state and arbitrary global-query exports for these nonperiodic guarded loops remain separate work.

Numerical residue, boundary-cell and startup enumeration are explicit representation costs. The current IR adapter uses the existing regional expansion limit and reports an unmet representation obligation if it is exceeded; it does not report a mathematical class exclusion. No experiment below certifies arbitrary refresh from observed repetitions.

## Verification (October 7)

- Serial incremental compilation of 101 affected translation units and relinking both local test tools; no new compiler warnings. The new test helper initially needed its direct `BuiltinTypes` include; that was corrected and rebuilt.
- Rotating boundary certificates: 360 comparisons against independently unfolded all-conflict graphs, including empty/short visits, non-unit strides, startup and residue changes.
- Guarded ranks: 320 guard valuations checked against independent event-graph closure, covers, and every all-event query.
- Bounded windows: 192 source-anchored windows checked against the complete invocation graph. Refresh tests distinguish covered and uncovered stride orbits.
- Recognition regression: affine varying-length nested input accepted; potentially wrapping scalar bound excluded; irregular participation accepted by the refresh recognizer without claiming entry guard availability.
- Existing finite guarded checks: 64 valuations, same-pipe barrier placement and two transactional endpoint rejections passed.
- Existing sequence checks: 32 actual command closures and trip-independent insertion passed. Updated a stale expectation: the whole dispatcher can use scalar replay, while the direct sequence adapter still reports its unavailable endpoint.
- Existing repeated-region checks: 14 nested command closures, zero trips and compact representation size passed.
- `git diff --check` passed. Changed-code prefilter reports only 44 `G.FMT.11-CPP` matches: its regex stops at nested condition parentheses. Each flagged statement was inspected with balanced-parenthesis parsing and has a braced body. The line-length findings were corrected. No warning suppression was added. Full EChecker/duplication metrics and device performance were not run.

The measurements above preceded the independent reviews below. No full corpus campaign or push was performed.

## Endpoint/export completion

The varying-region query index composes numerical boundary transfers. A transfer contains an internal child path, one native or storage crossing, and another internal path. Products of these Boolean matrices therefore preserve exactly the paths through successive visits. The startup is finite; the suffix has the certified period. Binary powers apply whole suffix cycles without expanding the outer count. Query results are masked by endpoint presence. Within one visit the original child quotient answers directly. Repeated source-to-port computations are memoized.

With `P` exported event ports, `s` startup visits, `q` suffix types and a `w=64` index word, this implementation charges dense transfer construction and matrix products explicitly: at most `O((s+q²+qw)P³)` work and `O((s+qw)P²)` matrix words, beyond child analysis and bridge extraction. A new source/target-visit query context uses `O((s+q(w+q))P²)` circuit work; different target events share it. These are adapter bounds, not a claim that the numerical chain hierarchy's sharper bounds are implemented here. Endpoint code contains retained internal and crossing recipes, not these query matrices.

The two new adapters share `CircuitEndpoints`, which owns detached preparation, matching member tuples, per-cut expression caches and namespace normalization. It does not flatten variable-length visits, allocate IDs, insert boundary drains, or clone payload computations. Nonadjacent local demands retain the established barrier-before-consumer policy; exact demand reduction need not imply order-exact barrier realization in that case.

Bounded-window predicates use common-stride storage renaming to identify relative cells; runtime predicates remain indexed by the actual iteration. Shared scalar and accumulation protection is applied to storage pairs, with reset and dynamic-scope distinctions preserved. Native prerequisites remain separate from storage hazards.

Remaining contracts are explicit: the automatic varying producer requires positive affine visit lengths and an unchanged modeled storage mapping, and currently leaves uniform residual relationships and protected cross-visit boundary adapters to other routes. General bounded-lifetime endpoint emission requires a supplied/derived uniform refresh certificate and executable predicates at both cuts. No opaque future branch outcome is treated as available, and no address-precision admission gate was added.

### Completion validation

- Actual command traces: 30 cases covering startup, the transition, repeated suffixes, zero outer trips, zero first-inner trips, non-unit slopes/steps, narrow-integer wrapping, and independently changing guards. Varying-loop cases match required payload order exactly. Bounded cases check exact cross-pipe cover pairs and local cover consumers, plus safety of the barrier-before-consumer realization.
- Arbitrary start/completion queries from the new varying regional export were compared with independent all-conflict graphs for short and suffix-spanning executions.
- Guarded rank checks: 320 valuations; bounded-window checks increased to 768 anchored windows, including protected accumulation writers, reset targets, scalar protection, and ordinary storage.
- An unavailable non-speculatable indexed predicate rejects endpoint preparation without modifying IR.
- Existing finite-guarded (64 valuations), sequence (32 closures), and repeated-region (14 nested closures) regressions passed.
- Changed translation units were compiled serially and both tools linked; this is not a new full build, full-corpus measurement, external review, or device evaluation.

The varying adapter's emitted preparation size was unchanged when the outer bound increased from 16 to one billion. The final changed-code prefilter has only 66 nested-parenthesis brace false positives; balanced-condition inspection confirmed that every reported body is braced. Its line-length findings were fixed. `git diff --check` is clean. Full analyzer duplication/complexity metrics were not run.

## Review corrections

Two independent code reviews accepted the implementation after these corrections:

- Thread the shared scalar-protection policy through rebuilt rotating-child quotients and cross-visit storage crossings. Native edges and supplied prerequisites remain unchanged.
- Preserve dynamic operation identities in bounded windows. Simultaneous macro envelopes have no internal storage demand, and affected cells keep raw external pairs rather than using a writer-chain shortcut.
- Construct varying-boundary transfers as two Boolean matrix products, avoiding a separate quadratic product for every crossing edge.
- Preserve child-certificate diagnostics. Failure of the sufficient unconditional-writer refresh producer is an unmet obligation, not exclusion from the bounded-lifetime class.

The review is static; the focused regression results are recorded separately. Allocation and complete end-to-end coverage remain separate from recognition and logical preparation. The six-ID limit does not turn a missing reuse proof into a proven scarcity result.

After those corrections, the serial incremental rebuild and both test-tool links passed. The independent checks passed 720 rotating-boundary comparisons, 320 guarded valuations, 960 anchored bounded windows (including simultaneous macro envelopes), and 30 actual endpoint/query cases. The lit expectation was updated to the expanded boundary count. Both reviewers accepted the corrected code; the allocation extensions are a separate change.

## Allocation adapters

Whole-function general arithmetic now checks reuse between every earlier/later pair in each retained endpoint family. Both handoffs share the same parameter context; exact integer projection preserves congruences, and exact union subtraction proves the required completion-to-next-start relation. A dedicated ID per family is a sufficient assignment, not a minimum-pool theorem. Failure to establish reuse leaves the allocation certificate absent. The dispatcher may use this whole-function certificate when regional allocation is unavailable.

Whole-loop bounded-lifetime insertion now certifies cyclic source-ordinal palettes using paths through guaranteed-present occurrences. An optional target attaches to the mandatory periodic graph, which supplies a return threshold to the next source. Alternative retained partners of one source and destination pipe share a palette; different sources reserve disjoint palettes. Shared hardware protection, simultaneous macro envelopes, and merged RMW modes are preserved. Hidden macro IDs are excluded. This certificate applies to one whole-loop invocation; it is not exported as a nested-region allocation summary.

Both use the existing allocation-only pass and IDs 0 through 5. Failure of a sufficient palette strategy is reported without claiming that no six-ID assignment exists. Neither adapter inserts scarcity repairs or extra payload ordering.

Remaining end-to-end gaps include allocation for varying-length repetition, typed arithmetic allocation summaries for regional composition, and global-query/profile exports for general bounded lifetimes. Arithmetic recognition and endpoint emission also retain their configured dimensional, residue and machine-integer contracts. Recognizer acceptance alone does not imply full end-to-end support for every theorem class.

Two independent reviewers accepted these allocation adapters and their integration on static review. Serial incremental compilation and linking passed. General arithmetic checks cover exact union coverage, residue transitions, shared parameter contexts, nested coordinates, and rejection of a witness that holds only for a short prefix. Bounded checks cover optional endpoints, absent mandatory return paths, palette grouping, empty demands, and split/merged RMW equivalence. The 30-case compact endpoint suite now checks physical reuse on its ten bounded cases and transactional failure with one eligible ID. Ten nested arithmetic command traces also passed physical reuse checks with six eligible IDs. The changed-code prefilter's 27 nested-parenthesis brace matches were inspected; every body is braced. No full corpus campaign, full static-analyzer metrics, or device run was performed for this change.

## Full-path completion milestones

The October 7 draft audit selected this order:

1. Shared numeric event-ID contract and finite allocation: greedy reuse, then minimum chain partition if needed.
2. Numerical periodic shared allocation through minimum-weight cycle covers; prefix-safe modular emission.
3. Composable lane exports and global allocation across explicit, arithmetic, guarded and repeated regions.
4. Allocation for varying-length repetition, preserving startup, residue, source identity and zero-trip conditions.
5. Arithmetic endpoint/query adapters, uniform reuse certification, interval-to-periodic and mixed-stride adapters.
6. Bounded-lifetime provenance, storage lanes, executable guards and finite overlays.
7. Numerical hierarchy and sparse frontier performance paths, followed by the pinned corpus validation.

Each milestone needs two code reviews and a separate commit. Scarcity repair stays out of scope: an
allocation failure must distinguish a proved capacity shortage from an unsupported or merely
sufficient assignment. Shared modeled access ranges remain input; no new precision admission gate
or kernel-specific recognizer is introduced.

### Milestone 1

Allocation certificates now use version 2 and one shared numeric-ID pool. Old direction-local
certificates are rejected. Finite explicit plans export the strict WAIT-before-SET relation;
nonadjacent local barriers are included in this fixed-plan order. Same-pipe command order can
establish reuse without a consumer-completion edge. The allocator tries deterministic greedy
reuse and then exact bipartite matching when the greedy result exceeds capacity. A proved
shortage reports the fixed-plan minimum. Guarded compatibility remains a sufficient coloring.

Regional palette conflicts and macro reservations apply across directions. Existing cyclic
certificates temporarily devote disjoint palettes to directions; milestone 2 adds sharing with
periodic reuse evidence. No logical demands or commands are deleted to fit capacity.

The independent command checker now tracks live state by numeric ID and checks direction and
logical generation on consumption. Finite relation construction uses O(h²) queries; validation
and matching take O(h³) worst-case time and O(h²) storage. This first implementation does not yet
provide the draft's inexpensive query-driven greedy fast path; milestone 7 retains that obligation.

M1 validation: 2,048 finite orders matched an independent brute-force chain partition;
20 guarded/explicit traces passed global causal reuse checks; physical prefix, zero-trip,
ID collision and transactional failure checks passed; family emission passed 23 independent
ordering cases and allocation passed 205 phase evaluations plus 21 invalid-interface checks.
Both reviewers accepted the production integration and the updated shared-pool fixtures.

The stricter shared-pool check exposes a regional allocation gap: the two-loop composition
fixture does not fit the current devoted whole-child palettes. It now reports a sufficient
assignment failure, not a minimum-capacity claim; lane-level composition in M3 must recover
sharing. The finite guarded rank circuit also needs integer guard simplification to recover
one-ID sharing between mutually exclusive covers. These are recorded obligations, not class
mismatches. No device timing or full corpus run has been performed for this milestone.

### Milestone 2

Unconditional periodic plans now construct a WAIT-to-SET phase matrix and solve its
minimum-weight cycle cover using exact integer Hungarian assignment. Different directions
may share one cycle. Each record emits its certified source-period modulo rule; grouped
command sites select the original record's rule even when member palettes or strides differ.
The uniform-palette fast path remains. The solver uses O(c³) integer operations and O(c²)
input storage without expanding numerical distances or trip counts.

Finite prefixes retain both endpoints of every emitted handoff. Nonnegative cycle links
place every intermediate consumer no later than the next source; restricting the uniform
chains therefore preserves reuse, including zero trips and partial final iterations.
Local nonadjacent barrier order is reconstructed only when its startup guard permits a
uniform predecessor edge; otherwise the required-order certificate remains sufficient and
is explicitly marked order_exact=false. No arbitrary indexed guards are admitted by this
producer, and no universal hardware-capacity optimum is claimed.

Validation: 19,864 phase matrices matched an independent permutation oracle, including
missing edges, zero cycles and integer overflow. Physical command checks passed empty,
short and longer prefixes and caught forced global ID collisions. Family allocation passed
259 phase evaluations (including mixed palettes at UINT64_MAX ordinal) and 21 rejected
interfaces; family emission passed 23 independent ordering checks; six logical insertion
checks passed. Two independent reviewers accepted the implementation. The two-slot
readiness/release test uses two shared IDs rather than four devoted IDs, with static IDs
where the selected cycles permit them. No device measurements were run.

### Milestone 3

Regional summaries now export guarded first/last handoffs per certified ID lane,
while retaining each endpoint record and its original modulo formula once. The
sequence and conditional adapters remap these selectors together with the payload
identities and guards. Numerical periodic children use the shared cycle-cover
assignment; mixed-direction cycles retain the actual pipes on each record.
Physical allocation matches lanes globally when the proved relation is a strict
order, and otherwise uses pairwise guarded compatibility. It never closes a
conditional relation through a possibly absent intermediate lane. Hidden-ID
exclusions apply to every member of a shared chain.

Repeated unit lanes use the same numerical cycle-cover solver across directions.
The same-pipe WAIT-before-SET query uses native start order; other pipe pairs use
the required completion-to-start query. Equal activity, finite-prefix closure and
partial-period phase restrictions remain part of the certificate. An unavailable
reuse proof remains distinct from a capacity lower bound.

Storage exports can lift outer-invariant readers disjoint from every body writer.
A finite symbolic support can alternatively supply exact per-byte selectors for
the repeated crossing computation, with its expansion charged explicitly and
bounded by the existing regional representation limit. Residual and uniform
relationships remain in the shared access boundary; omitted conflicting extrema
are an unavailable interface. This is not a new precision gate on shared effects.
General symbolic phased intervals still need a selector-clipping adapter; the
ordinary invariant lift is not silently reused for those intervals.

For L exported lanes, the current regional assignment uses O(L²) boundary-pair
checks and O(L³) worst-case relation validation/matching. Each boundary check also
pays for its guarded selector pairs and regional reachability queries. The exact
shared minimum theorem is not claimed for the guarded compatibility fallback or
for an incomplete sufficient reuse relation. No payload ordering is added.

Constant child lanes can still coalesce before repetition. The adapter remaps
only coordinate-independent formulas, proves compatibility against every user
of the chosen lane, preserves each record's direction and tuple arity, and
rebuilds its selectors. Dynamic formulas keep their original mapping.

A historical test correction is necessary: the multi-phase nested allocation
fixture predates the shared numeric-ID contract and formerly ignored conflicts
between different directions. The current period-two proof graph has six
internal lanes plus one epilogue lane with no proved sharing; this does not prove
that the payload plan needs seven hardware IDs. Readiness and release have
different activity predicates (positive versus at least two inner trips), so
the unconditional cycle certificate cannot simply join them. The guarded
uniform-query/allocation adapter remains an explicit follow-up obligation.
The test continues to check all logical command closures and requires allocation
failure to be transactional; it does not accept an incorrect reused numeric ID.

M3 validation: 15 independent regional command-reuse traces passed (including
adjacent loops within six IDs and a guarded one-ID case); family emission checked
259 phase evaluations and rejected 22 malformed interfaces. Symbolic storage
checks cover invariant readers, finite persistent writers, zero trips, aliased
ordinary/deferred writers, and missing extrema. Repeated storage passed seven
production closures; phased composition passed 32 exact command closures and
compact-size checks. Nested allocation passed nine physical traces and twelve
transactional unavailable-certificate cases, all retaining independent logical
closure checks. Both code reviewers accepted the corrected implementation and
the historical test correction. Serial incremental compilation/linking passed.
The compliance prefilter's 43 nested-parenthesis brace matches were inspected;
every body is braced. Line-length findings were fixed, and git diff --check is
clean. Full static-analyzer metrics, corpus and device measurements were not run
for this milestone.

### Milestone 4

Varying-length repetition exports sufficient allocation palettes and guarded
first/last handoffs for each lane. The original source identity remains the
inner ordinal and outer visit. Startup and seam records are singletons; suffix
records use their actual source-visit residue. A suffix lane's first collision
is separated by the least common multiple of its record period and palette
width. Numerical boundary transfers certify reuse without unfolding visits.

Internal child cycles reuse the numerical shared allocator when their width
divides the selector period and their endpoint residues lie within the proved
boundary fringe. Other internal records use a dedicated cyclic palette when
its return path is certified. All groups enter the existing global six-ID
allocator. These are sufficient strategies: failure is not a minimum-capacity
result, and no payload order is added to make a palette fit.

The universal query checks exact short visits and proves that long-class
attachments remain constant under the certified length residue. It retains the
existing startup/suffix matrices. For P ports, an uncached query uses O(P)
quotient attachments and O((s+q+64)P²) Boolean work, where s is the startup
length and q the suffix period. Existing matrix construction is charged
separately. The implication counter counts proof requests, including cache hits;
it does not assign one common cost to numerical and guarded queries.

Two repeated computations exposed by this integration are removed: once two
successive squared transfer matrices agree, every higher square is the same
idempotent matrix; a single conditional application replaces those bits.
Guard specialization visits only the requested expression's ancestors, avoiding
work and generated negations for unrelated queries in the shared arena.

Forced unsigned comparisons against constants now contribute exact bounds on
the same expression. No bounds are propagated through wrapping arithmetic or
signed comparisons. This proves startup facts such as T>4 implying T>3, which
the former Boolean-only proof treated as independent predicates. Numerical
sequence specialization also checks its constant-guard contract before querying
children; symbolic guards retain the general guarded reduction.

M4 validation: all 20 varying command/physical-allocation cases passed, including
zero trips, empty first visits, slope and step changes, startup and suffix
boundaries, exact all-event queries, and unchanged preparation size at a billion
outer iterations. One-ID failure is transactional. Serial incremental builds
and links passed. The expression checks pass the new 22,500 unsigned-bound
implications and wraparound counterexamples. The representative sequence query
fell from a 90-second timeout to about 0.15 seconds after the two local fixes;
this is a diagnostic measurement, not a corpus performance claim.

Two previously outstanding allocation tests remain visible. The broad repeated
region suite reaches an unsupported clipped period-two assignment (M5's active
successor obligation). The bounded-lifetime physical suite exceeds six with its
dedicated palettes (M6); its allocator has been unchanged since M1 corrected
the shared pool. The varying and bounded tests have separate invocations, and
both remain in lit with their original success assertions. No test was changed
to call these proof gaps hardware scarcity. The repeated-reuse oracle was
corrected to use native start order for same-pipe WAIT-to-SET reuse and retains
completion-to-start checks across pipes.


### Milestone 5

Arithmetic allocation now selects exact extrema from the retained handoff
relation and certifies widths one through six over its active source order.
The immediate-successor relation is the strict source order minus its square;
its powers count executed handoffs, including holes in the iteration domain.
For widths above one, a separate proof checks that matching preserves consumer
order. Every join retains the same parameter and residue context. Completed
DBM demands and required order can be converted exactly to the integer backend
for this proof without repeating recognition or reachability construction.

Whole-function arithmetic emission uses two bounded SSA counters per certified
family, one for publications and one for consumptions. Only executed commands
advance them. Structured loop and branch results carry the counters while
preserving all original results and payloads. Empty loops and untaken arms keep
the incoming state. Width-one families need no counters. Palettes are disjoint
within the one numeric pool; failure to fit is a failure of this sufficient
assignment, not a minimum hardware-capacity claim. Hidden IDs are excluded.
The transformation validates a detached function copy before committing it.
Regional arithmetic exports currently use width-one families with exact
first-source and last-consumer selectors; counter scope is not silently reset
at region boundaries.

Mixed-stride accesses use a numerical phase adapter. For each slot family,
relative strides determine a common period by GCD/LCM arithmetic. Expanding
only the analysis skeleton makes the strides equal modulo that family size,
so the existing rotating extractor and quotient apply. Endpoint recipes remain
at the original payload cuts, select the correct phase, and independently test
both endpoints against the actual finite prefix. Work and representation are
charged to the expanded skeleton using the existing numerical-template limits.
This adapter currently handles one whole fixed-body loop; it does not claim
regional storage exports or compact treatment of an arbitrarily large encoded
joint period.

Supplied arithmetic endpoint relations may use other representable fixed
periods; emission no longer has a period-two-only gate. The automatic arithmetic
producer and dispatcher retain their configured period-one/period-two residue
classes. This is distinct from the mixed-stride phase adapter.


Repeated allocation now clips absent phases before constructing its reuse
matrix. Conditional reuse edges propose an assignment; every pair that shares
a numeric label then receives a direct collision check. Thus a skipped
intermediate handoff cannot justify reuse. When one shared cycle is unavailable,
a deterministic palette partition retries the same direct proof for each
insertion. It costs O(6² n³) comparisons beyond the producer's reuse queries
and remains a sufficient assignment. Guarded per-label first/last endpoints
preserve clipping, phase offsets, and delayed consumers for outer composition.

M5 validation: 360 independent mixed-stride graph comparisons and 30 emitted
logical/physical traces passed, including incomplete final periods, reversed
stationary/rotating accesses, and nonzero lower bounds/steps. The command oracle
first checks exact cover endpoints, then applies the agreed consumer-adjacent
barrier policy for nonadjacent local demands. Arithmetic proof checks include
sparse active sources, exact residue/parameter contexts, reversed-consumer
rejection, DBM fallback and supplied period-three selectors. The counter IR
interpreter checks nested loops, branch results, original carried values, zero
trips, skipped handoffs, original matching coordinates and transactional ID
shortage. Ten full arithmetic traces and 21 nested physical traces passed.
Repeated checks cover 123,552 event pairs, 27,264 phased queries, and 1,583
member envelopes/reuse chains. Allocation testing uses one pool of six IDs.
Both independent code reviews accepted the arithmetic/counter and mixed-stride
changes; final repeated-lane review and selector checks are recorded below.

The final repeated-lane review accepted the guarded partition and extrema
proof. Its numerical oracle also passed the exported per-label selector checks
across clipping, absent phases and eight-period ID wraps. Serial incremental
builds and links passed. The compliance prefilter reported 55 nested-parenthesis
brace matches; all corresponding bodies are braced, and line-width findings
were corrected. No full static-analyzer or device claim is made here. The
pinned corpus campaign remains milestone 7.
