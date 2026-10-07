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
