# Joint first-consumer and final-reader prefixes

Implementation on `2a130aefe`, following the
[concrete down_proj diagnosis](oahs-blocked-down-last-reader.md).
This extends default handoff qualification; there is no additional pass mode.

## Qualification and construction

The native first-consumer importer collects both sets of endpoint requirements
on the original owner before changing its observed graph. It uses existing
physical-cell effects, original loop control and producer/reader identities.
It keeps the existing B0 acquisition and finds A0's last read in the same
unconditional prefix. Every body path is checked for later A0 reads and writes;
unknown/exclusive storage, active producing pipes, empty/unknown bounds and
nested reader loops retain the conservative path.

For constant signed index bounds `0 <= lower < upper` and positive `step`,
first and final observations are respectively:

```
iv < lower + step
upper - iv <= step
```

Admission checks the first threshold and final increment against signed overflow.
Down_proj visits 0 and 128, so the final threshold is 128, not one. A single
visit has both endpoint predicates true. Non-divisible bounds are covered too.
No event state or hidden runtime counter determines participation.

`refineReaderVisits` creates one joint control view. Only requested endpoint
words gain predicates; the original physical operations and all conditional
suffix command identities stay shared. First and interior visits share the
suffix graph. The final continuation has separate analytical sites to preserve
its exit correspondence through the branches. Sharing those sites as well
would lose final-versus-backedge correlation. This is a bounded two-continuation
representation, not a product of the suffix's branch conditions or new emitted
matrix control. Its analysis cost is explicitly measured below.

Reader-region cycle selection uses the qualified original-step distance and
its existing nearest-use/participation checks. It places the actual return at
the post-A0-read word. The complete recurring protocol is materialized and
validated before commitment; ordinary construction then places the unrelated
B0 reuse wait later. Completion and rearming still come from actual selected
transfers. Neither the observation nor a child exit supplies causal credit.

The emitted word includes the original MTE1-to-M readiness publication and,
on the final visit, the MTE1-to-MTE2 return. The native emitter currently emits
separate final/nonfinal guards for words with different endpoint populations;
this guard cost is deliberately retained for measurement. Genuine M/FIX/ALL
fences remain. No source-name or opcode-pattern recognition is added.

## Validation

The portable joint fixture includes two inputs, separate first/final endpoints,
a conditional suffix, single/multiple visits, changing child lengths and
repeated parent entries. It checks original payload/word identity, real selected
endpoints, an independent event/physical-effect graph on 132 finite traces, and
missing-support failures. Invalid/empty/refined owners and conditional endpoint
requests decline.

Native tests cover nonzero bounds, step 128 and unit step, single visits,
non-divisible bounds, unknown/empty/negative bounds, overflowing final increment,
suffix reads and suffix regeneration. An edited emitted final predicate is
rejected transactionally. The unchanged native checker reconstructs the actual
guards and words for down_proj and every accepted corpus input.

The down_proj independent graph checks 18 paths (chunks 0/1/2/3/4/17 and output
tiles 0/1/2), 73,449 local conflicts per arm. Complete payload relation sets lose
214 relations with zero additions. Executed synchronization and fence counts
are identical on every path. For 17 chunks the one-/two-tile paths remove
56/112 relations. The change is larger than the original A0-only supplied
mutation because the same qualifier also advances other admitted child releases.

Reproduce after generating plans from the frozen baseline and candidate:

```sh
python3 test/oahs/check_joint_reader_trace.py candidate.pto \
  --baseline baseline.pto --output ordering.json
```

All 23 portable suites, both native test executables, the live down_proj pass
regression and direct C++ lowering pass. Six mutations of the emitted down_proj
plan reject missing/first-visit/every-visit release, wrong unit-distance guard,
missing readiness and missing consumption. These are local/event checks, not
device numerical execution.

### Corpus and transfer checks

All 88 inputs (97 function instances) pass construction and reconstruction.
The initial paired run was followed by a final 88-candidate rerun against the
same frozen baseline. Eleven projection modules (prefill rows 0 through 10)
change; 77 remain byte-identical, including attention, post-RMSNorm and Shenggan.
Minimizing the first-prefix copy leaves all 88 initially tested candidate plans
byte-identical.

The six representative projection cases have 37 finite paths and 511,187
checked local conflicts per arm. Complete payload relation sets are compared,
not only their sizes. Executed SET/WAIT and all fence counts remain equal on
every path.

| Case | Paths | Full relations removed | Added |
| --- | ---: | ---: | ---: |
| Down | 18 | 214 | 0 |
| Gate/up | 5 | 520 | 0 |
| KV | 5 | 288 | 0 |
| Q | 3 | 144 | 0 |
| Out AIC | 3 | 144 | 0 |
| LM head | 3 | 104 | 0 |
| Total | 37 | 1,414 | 0 |

The sums contain distinct tested executions and aliases; they are not a count
for one kernel invocation. Reproduce with paired `CASE/{baseline,candidate}/plan.pto`
files:

```sh
python3 test/oahs/check_joint_reader_family.py CORPUS_DIRECTORY --output family.json
```

### Explicit compiler work

| Work | Down baseline | Down candidate | LM baseline | LM candidate |
| --- | ---: | ---: | ---: | ---: |
| Analytical sites | 181 | 262 | 132 | 186 |
| Selected updates | 16 | 17 | 11 | 11 |
| Replay evaluations | 13,563 | 16,950 | 5,014 | 5,794 |
| Mandatory proposal evaluations | 656 | 926 | 350 | 485 |
| Final-check evaluations | 819 | 963 | 442 | 507 |
| Loop-entry trial evaluations | 2,624 | 5,556 | 1,050 | 1,455 |
| Recurring channels | 22 | 22 | 20 | 20 |

Across 97 function instances replay increases 3,234,019 ->3,262,799 (28,780,
about 0.89%). Recurring channels stay 383; recurring/helper trial counts stay
65/55. Down's replay rises about 25%, so this is explicitly not a compiler-cost
improvement. Shared command identities do not make the extra final-continuation
state free. First-prefix minimization reduced the intermediate down figure
17,814 ->16,950 without changing any plan. Per-run wall times are retained in
the artifacts; concurrent local jobs make them unsuitable as a timing claim.

Raw evidence: `../joint-reader-work/` (outside the source checkout), especially
`corpus-final/results.json`, `compiler-work.json`, `final-ordering.json`,
`final-family-ordering.json`, native/core logs, lowered C++, `negatives/` and
`manifest.json`. The frozen baseline has production code at `2a130aefe`; its
additional read-only diagnostic does not alter construction.

## Scope and next action

This does not qualify guarded first/last participating readers, arbitrary
interior word gaps, active-producer reader cycles or general recurring-key reuse.
Native first/last requests at the same word are conservatively declined; this
case has distinct physical anchors. Standalone last-reader qualification keeps
its previous restricted admission.

There is no device latency claim. The
[short matched down_proj task](../../test/benchmarks/joint_reader/DEVICE_TASK.md)
uses the existing qualified harness, correctness before timing, warmups and
20 samples per arm. Commit/push the implementation before dispatching that task.
