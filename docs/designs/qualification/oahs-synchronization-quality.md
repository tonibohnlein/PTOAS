# Controlled OAHS synchronization quality

## Input and hardware controls

The first matrix used the compiler binary from `0b54eff6f71699b5be9fc8d2e564b197d367d9c1`.
Artifacts are in `../oahs-coverage-work/quality-matrix-0b54eff6f-r1` relative
to this checkout. This is a baseline measurement, not validation of later edits.
Each arm uses the same original payload, physical assignments, A/B/C pairwise
alias attributes, `may-alias` pass setting, architecture and five launches.
Only the following scalar metadata is added for the qualified input:

| Argument | Minimum | Maximum | Multiple |
| --- | ---: | ---: | ---: |
| 3 | 0 | 2147483520 | 128 |
| 4 | 256 | 2147483136 | 256 |
| 5 | 512 | 2147482624 | 512 |

These maxima round the recorded caller-contract bounds down to their stated
multiples. The original input contains no scalar precondition attribute.
The original and qualified inputs do not differ in alias guarantees.

| Input | Planner | Hardware | Discovered/selected lifetimes | SET | WAIT | Named barriers | Terminal ALL |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original | Composition, precision on | Conservative | 5/0 | 56 | 59 | 21 | 1 |
| Original | Composition, precision on | Qualified MMAD | 5/0 | 56 | 59 | 10 | 1 |
| Qualified | Composition, precision on | Conservative | 7/7 | 54 | 54 | 8 | 1 |
| Qualified | Composition, precision on | Qualified MMAD | 7/7 | 54 | 54 | 0 | 1 |
| Qualified | InsertSync | Existing | — | 44 | 44 | 21 | 1 |
| Original (additional control) | InsertSync | Existing | — | 44 | 44 | 21 | 1 |

Disabling MMAD credit does **not** lose the qualified input's lifetime plan.
The eight additional static barriers are on M. Readiness/release handoff
observations match exactly between the two qualified-input arms for all five
launches. MMAD ordering is not credited for operand release or event reuse.

| Launch | SET = WAIT, qualified input | Conservative named barriers | MMAD named barriers |
| --- | ---: | ---: | ---: |
| Empty grid | 7 | 0 | 0 |
| One panel | 31 | 4 | 0 |
| Two panels | 53 | 8 | 0 |
| Three panels | 75 | 12 | 0 |
| Distributed tiles | 99 | 16 | 0 |

Each execution additionally performs one terminal ALL. These are bounded
scalar replay observations, not numerical device measurements.

Both original-input arms reject the persistent proposal at node 47, the initial
MTE2 load into MAT cell 3. Its GM source cell is read-only; this refusal is not
GM publication or a missing noalias promise. Without the scalar geometry facts,
the original program discovers five operand lifetimes but no ACC lifetime.
Its guarded FIX/M entry provider prevents general residual activation in the
baseline. The final failure string does not distinguish local MTE1-read reuse
from MTE2 self-write ordering; matching a cell alone cannot establish that edge.

## Static endpoint asymmetry

The original GEMM emits **56 SET sites and 59 WAIT sites**, not 56 pairs.
Its FIX→M EVENT_ID1 has one SET and four WAIT sites. In the original
conservative endpoint report these are SET site 1 and WAIT sites 16, 22, 77, 83.
Only WAIT 16 is reachable. WAIT 22 requires both first and non-first iteration;
WAITs 77 and 83 require the first iteration in the odd-parity branch. The three
dead alternatives account for the entire static difference.

The reachable entry acquisition primes the first tile and orders a later tile
after the preceding FIX store. M→FIX EVENT_ID1 at tile end returns consumption
acknowledgment and readiness under the nonempty-panel guard. This entry family
has no final cleanup acquisition. Empty grid executes no generation; the
one-, two- and three-panel cases each execute one generation, and distributed
tiles execute two. Every published generation is consumed once.

Qualified persistent GEMM instead uses FIX→M EVENT_ID0 to prime before the
outer loop, acquire before each tile, publish release after each store, and
consume the final release after the loop. Empty grid consumes the priming
generation at cleanup. Distributed tiles consume three generations: prime,
first tile release and second tile release. These roles must not be conflated
with the original input's dead guarded alternatives.

Buffering has another legitimate source of unequal static counts: a release
SET has a next-visit WAIT and a separate guarded exit-cleanup WAIT. One buffer
has two such families; two/three buffers and four-use have per-slot families.
The fresh benchmark reports 4/6, 8/10, 12/15 and 20/22 static SET/WAIT sites.
For n loop visits their executed SET and WAIT populations are respectively
4n, 4n, 4n and 10n, including n=0. These are reachable consumption alternatives.

`quality_matrix.py` records per-direction/key endpoints with enclosing loops
and guards, and per-execution generation inventories and prefix boundaries.
Concrete SET/WAIT balance alone does not prove asynchronous reuse. Consumption
receipt propagation and final retirement are recorded separately; mandatory
compiler reconstruction and mutation tests remain the correctness boundary.
The initial matrix did not pin the newly written matrix runner's own hash;
its input/output and binary hashes are present. Subsequent runs additionally
pin that runner and all three observer modules and enforce their identity.

## Softmax and QK obligations

The fresh conservative benchmark has softmax 12/12/20 versus OAHS 17/17/21,
and QK 21/21/2 versus OAHS 18/18/4 (SET/WAIT/named barriers). Every arm also has
one terminal ALL, excluded from these triples.

QK's extra M barriers precede the first matmul of each half, before incoming
MTE1→M and FIX→M waits. Both halves reuse ACC [0,16384). The FIX release is
downstream of M readiness; its actual prefix may already satisfy the self-lane
hazard. The separate barriers before accumulator updates remain necessary
under the conservative profile and must not receive operand-release credit.

Softmax's extra V barrier precedes its second final divide and an incoming
MTE3→V acquisition. The first result uses UB [8448,12544), then a converted
bf16 store uses [8448,10496), before the second divide reuses that storage.
This is another completion-reuse candidate, requiring proof of the receipt.

Two late tmul consumers per loop instead receive closed V→MTE2→V exchanges.
They read newly loaded UB [4128,4160) plus V output [4224,4256), writing
[4256,4288). InsertSync publishes after the third load and waits at the late
consumer. OAHS's reverse-first exchange additionally waits for preceding V
work on MTE2. These exchanges explain 30 of the 33 extra executed SET and WAIT
operations in the n=16 scenario; three further entry/exit pairs remain.
Earlier preload acquisitions may be redundant after later load-prefix waits,
but that is an unproven provider/generation hypothesis.

A trial that processed incoming demands before self-lane demands lost an
existing deferred-ring selection in a focused test and was reverted. The
extra softmax/QK mechanisms are not declared necessary or fixed on that basis.

## General construction changes

The new residual transaction retries at most once without optional retained
completion groups after combined allocation fails. It restores symbolic
persistent handoffs, reconstructs every residual obligation, and shares the
original residual allowance across attempts. Observation exhaustion retains
its actual `persistent-observation` stage and reason.

Recognized First/NonEmpty entry providers can now be replaced through the same
residual interface even when their obligations lie outside selected lifetimes.
Arbitrary guarded/deferred participation is still unsupported. An independent
zero-trip reproduction exposed syntactic barrier deletion before combined
verification: the apparent return exchange could carry an older prefix.
That deletion was removed. The complete residual population is now verified
before existing bounded optional cleanup can remove any mechanism.

The focused allocation tests distinguish actual capacity exhaustion from an
injected successful retry. A real two-key example retries and returns the
verified fallback; a third key permits construction. A test-only fault after
partial numbering exercises successful retry, compares its commands with
independent construction without completion groups, and verifies the result.
This is not evidence of a naturally occurring capacity salvage in the corpus.
Zero observation allowance preserves the exact observation stage and baseline.

## Final local results

The final compiler measurements use the dirty implementation above base
`0b54eff6f71699b5be9fc8d2e564b197d367d9c1`, not the original base binary.
`../oahs-coverage-work/quality-matrix-final-r1/summary.json` records the source
diff, compiler and runner hashes, input/output identities, commands, scalar
guards, physical keys, hardware profiles and launch arguments. Companion
`.witness.json` and `.endpoints.json` files retain discovered/selected families,
rejections, structural cuts and concrete per-generation endpoint inventories.
The final `libPTOASCompiler.so` SHA-256 is
`748b1929e269f2ab7cb9628164cfbf1aae12bf9d9444b730ccdb5847a560d920`.
Later documentation edits do not relabel an earlier compiler run.

| Input | Hardware | Discovered/selected | SET | WAIT | Named barriers |
| --- | --- | ---: | ---: | ---: | ---: |
| Original | Conservative | 5/5 | 61 | 61 | 10 |
| Original | Qualified MMAD | 5/5 | 60 | 60 | 2 |
| Qualified | Conservative | 7/7 | 54 | 54 | 8 |
| Qualified | Qualified MMAD | 7/7 | 54 | 54 | 0 |

Both InsertSync controls remain 44/44/21. Every arm retains one terminal ALL.
Both qualified composition arms preserve the baseline's exact useful handoff
observations. The original-input conservative result retains eight M barriers,
one MTE2 barrier and one FIX barrier; the MMAD arm retains the latter two.
All discovered lifetimes are now selected without a provider rejection.

| Original conservative launch | Previous SET/WAIT/named | Final SET/WAIT/named | Later/earlier cross-lane observations |
| --- | --- | --- | --- |
| Empty grid | 0/0/0 | 5/5/0 | 0/0 |
| One panel | 27/27/12 | 33/33/6 | 1/0 |
| Two panels | 52/52/19 | 57/57/10 | 1/14 |
| Three panels | 73/73/29 | 81/81/14 | 3/20 |
| Distributed tiles | 104/104/38 | 109/109/20 | 3/33 |

This is mixed quality. The first operand extraction now waits for the second
load's prefix, and a later prefetch can wait for a later accumulator update.
Unconditional lifetime priming also adds five publications/consumptions on the
empty launch. More lifetime participation and fewer barriers do not establish
dominance, lower latency or full quality acceptance.

All OAHS matrix executions establish publication participation, consumption
receipts before rearm and terminal payload retirement in the concrete model.
Both existing GEMM controls have 0/1/9/10/18 rearm sites without a receipt proof
in the five scenarios. Their tokens still balance and retire. This is a model
proof gap, not a demonstrated device failure or a claim that InsertSync is
incorrect. Native emitted verification remains separate from this observer.

### Eight conservative benchmarks

`../oahs-coverage-work/benchmarks-quality-final-r1` reruns the same eight inputs
and alias contracts with existing, composition without precision, and
composition with precision (`demands`). All 24 compiler and subsequent C++
emission results pass. The table describes the precision-enabled arm only;
the artifact preserves all three arms separately.

| Case | Previous SET/WAIT/named | Final SET/WAIT/named |
| --- | --- | --- |
| One buffer | 4/6/0 | 4/6/0 |
| Two buffers | 8/10/0 | 8/10/0 |
| Three buffers | 12/15/0 | 12/15/0 |
| Four-use | 20/22/0 | 20/22/0 |
| Softmax | 17/17/21 | 17/17/19 |
| QK | 18/18/4 | 21/21/2 |
| Q projection | 17/17/6 | 17/17/6 |
| Original GEMM | 56/59/21 | 61/61/10 |

Every case retains one terminal ALL, excluded from named counts.
`../oahs-coverage-work/quality-boundaries-final-r1` includes old, final and
InsertSync endpoint inventories and generation replay for every benchmark
scenario. Buffering and Q projection have unchanged cross-lane observations.

At n=16, softmax executes 157/157/243 named, versus previous 129/129/259 and
InsertSync 96/96/258. QK executes 276/276/32, versus previous 243/243/64 and
InsertSync 261/261/32. The terminal ALL is additional in every triple.
The extra QK first-matmul barriers disappear while the conservative accumulator
update barriers remain. Softmax loses the final storage-reuse barrier and one
barrier per loop visit. Guarded provider replacement adds event participation;
unchanged static softmax event counts conceal 28 extra executed publications
and consumptions in the long-loop scenario.

The endpoint inventory explains the repetition exactly. Softmax changes from
eight loop SET sites executed 15 times plus nine outside sites (129) to ten
loop sites plus seven outside (157). V→MTE2 publications increase by 30 and
MTE2→V decrease by two; output directions are unchanged. QK changes from
15 loop sites executed 16 times plus three outside (243) to 17 loop sites
plus four outside (276). M→MTE1 increases by 18, FIX→M by one and MTE2→MTE1
by 14; other directions are unchanged. These are repeated release and readiness
costs of the migrated provider population, not new payload operations.
In the final softmax PTO, recurring scalar-slot releases follow consumption
at lines 122/140 and 206/224 alongside the retained tile-load release. In QK,
the MTE2→MTE1 SET/WAIT at lines 60–61 replaces guarded preload acquisition;
it accounts for all 15 extra executed publications relative to InsertSync
at n=16. QK's lines 40–43 and 139–142 are priming and cleanup cuts, including
four publications/consumptions on zero-trip execution.

Softmax's n=2 and n=16 cross-lane observations match the previous plan. With
no loop body, its first final divide, conversion and store now wait for all
six initial loads instead of the first three. QK has one later first-extraction
observation per iteration; n=16 has 16 later and 30 earlier observations.
Softmax's final closed exchange at lines 252–253 imposes the later load prefix;
the earlier publication still exists but cannot undo that preceding wait.
QK's later observations arise at the preload exchange described above; its
earlier observations are M→MTE1 improvements at the first two extractions of
subsequent iterations. They measure separate constraints and do not cancel.
The generic constructor still needs better preservation of guarded entry
prefixes and completion reuse. These tradeoffs are not a kernel-specific
admission rule or a claim that fewer barriers compensate for more events.

Compilation timing is telemetry from one paired trial without warmup, with
one other serial corpus worker running. Precision-arm compilation ranges
0.417–0.611 seconds and paired ratios to existing range 0.983–1.171. These
measurements are not a stable performance estimate or a ratio acceptance gate.

### Independent coverage attribution

`../oahs-coverage-work/campaign-quality-final-r1` replays the frozen 363 rows
(336 distinct input/contract runs). Admissions remain 253/363: 79 PTOAS,
7 PyPTO and 167 pypto-lib, with no gains or losses against the immediate
baseline. All original 86 publication refusals remain. Generated synchronization,
authored-protocol preservation and no-op success remain separate artifact fields.
No caller alias contract or device publication capability was introduced.

`../oahs-coverage-work/publication-flow-quality-final-r1` adds bounded structured
control-flow attribution to those 86 reports. Among 166 consumer/cell-matching
writer candidates, 39 may precede without a backedge, 113 may precede through
a backedge, and 14 cannot precede **within one invocation**. The exclusions
affect six real corpus inputs. Every refusal still has at least one possible
candidate. None of these reports exhausts the bounded analysis.

The analysis includes loop zero-trip paths, while before/after flow, choices
and ordered macro access phases. Unknown or exhausted graphs remain unknown.
It does not solve branch predicates, reaching generations, intervening-write
kills, publication kills or cross-invocation dependencies. A may-predecessor
is not a proven causal writer, and even the excluded intra-invocation pairs
do not establish a cross-invocation disjointness guarantee. No compiler
admission is credited to this diagnostic improvement.

### Validation and remaining acceptance

The final targeted build uses at most two aggregate workers. Four focused
CTest groups pass, including 2,433,767 core assertions, endpoint receipt and
non-implication tests, bounded writer-flow tests and coverage witness tests.
`../oahs-coverage-work/composition-quality-final-r1` records 22 positive native
cases, 191 mutation checks, nine expected refusals and two frontend cases.
The qualified GEMM54 reconstruction and tested prototype boundaries pass.

Architect, algorithms/performance and correctness reviews are recorded with
the artifact index. Local correctness tests pass; qualified GEMM and existing
buffering boundaries are preserved; barrier counts improve in three general
regressions. Overall synchronization quality is still mixed because event
traffic and some prefix requirements increase. Structured-reference refusal
and absent device validation remain open gates. General guarded/deferred
precision, softmax/QK completion reuse, exact reaching-writer attribution,
genuine caller alias imports and device GM publication qualification remain
separate work. Existing remains the default.
