# Stage 1: indexed finite loops and the practical arithmetic boundary

Arithmetic earns a **qualified regional-interface role** in these inputs. Its
retained static access relations avoid the chosen production sequence's cell
and port enumeration. It adds no new whole-kernel acceptance in this experiment,
and does not demonstrate fewer synchronization commands: the investigated
scalar-only interiors have zero nonnative covers under the supplied PIPE_S
storage-protection and SSA-readiness semantics. Detailed general arithmetic
construction belongs in the appendix; retain the compact-interface distinction
in the main Section 5 discussion.

The [receipt](frontier-compact-classes-stage1-20261020.json) pins the starting
**nine dirty files**, their patch, the frozen baseline tools, original inputs,
new diagnostic source hashes, commands and results. The source base is
`147cb72edd014a0cc7950ae50e1d870a370a0253`; that hash alone is insufficient.
The Linux snapshot `candidate-source`/`candidate-build` was used, not the stale
macOS workspace metadata. All requests are serial with a 35-second watchdog;
no admission limit, allocation algorithm or production dispatcher changed.

## Actual access problems

All byte offsets below are local VEC physical addresses, independent of SSA root
names. Scalar reads/writes are exact four-byte declarations. This statement does
not qualify enclosing vector instructions' native footprints.

* Lossless block cast overlays f32/i32 views at byte 0 and i32/i16 views at byte
  2048 (also the mask temporaries at 86080 and 88128). For `0<=i<64,0<=j<4`, the
  reduction reads/writes accumulator `86016+4*floor(j/2)` and reads exponent
  `2048+4*(8*i+j)`. The correction reads that exponent and accumulator and writes
  the exponent. The expansion `0<=j<128` reads
  `2048+4*(8*i+floor(j/32))`, writes `53248+4*(128*i+j)`. The lane-zero wrapper
  is original control; diagnostic derivatives bind block/subblock zero only.
  TSHRS, TAND and TMOV still use allocation enclosures. Therefore exact reduction
  of this whole selected model is **not exact original physical demand recovery**.
* Cumsum initializes output at 4096, loads input at 0, and processes eight chunks
  of 128 elements. Chunk `c` copies input `[128*c]` to output `[128*c]`; inner
  `1<=j<128` reads output `[128*c+j-1]` and input `[128*c+j]`, then writes output
  `[128*c+j]`. All offsets scale by four bytes. Floating-point values are not
  access indices. Each write's SSA producers are explicit native readiness
  prerequisites in the reference. Output initialization and the original layout
  are retained in diagnostic production derivatives.

## Fresh route comparison

| Region | Candidate | Recognition / guarantee | Executable export | Allocation | First remaining obligation |
|---|---|---|---|---|---|
| Cast reduction/correction | Arithmetic | Bounded coefficients; D=4, P=0, C=32; 3 sites, 54 primitive relations, 0 nonnative pieces | Existing endpoint request succeeds | Not requested | Whole vector-footprint qualification remains separate |
| Cast expansion | Arithmetic | D=4, P=0, C=512; 2 sites, 30 primitive relations, 0 nonnative pieces | Succeeds | Not requested | Large scalar-to-vector parent boundary may still enumerate cells |
| Cumsum chunk loop | Arithmetic | D=4, P=0; C depends on configured width, 5 sites, 125 primitive relations | Succeeds | Not requested | No new command reduction established |
| All four interiors | Explicit numerical diagnostic | `symbolic-storage-partition` refusal | Unavailable | Not requested | Indexed scalar footprints do not satisfy this importer's supplied numerical partition |
| All four interiors, small | Finite expansion | Exact modeled demands, queries and selectors | Original-coordinate adapter unavailable | Not requested | Existing expanded endpoint contract, addressed in stage 4 |
| All four interiors | Ordinary production | Regional sequence; queries/selectors agree on tested occurrences | Succeeds | Not requested | Generic retained-leaf record checker cannot unpack these numerical leaves; not an empty-cover claim |

The original whole inputs freshly reach logical insertion (cast 3.22 s; cumsum
0.316 s, descriptive single process runs). Their explicit whole-function
arithmetic import stops at `index-arithmetic` before relation construction,
including hardware-index casts. This is not a restricted-class mathematical
rejection. Whole original allocation/C++ were not rerun here; the existing
production-coverage receipt remains separate evidence for cumsum.

## Independent checks and representation scaling

303 scalar effects agree with independently computed physical byte intervals on
oracle-only finite slices. Four arithmetic and four expanded regional results
agree on actual retained covers, all four event-kind queries and requested
boundary selectors, including proved absence at an untouched byte. Production
queries/selectors agree too; unavailable production record decoding is retained
as an explicit test limitation. Large expansion query samples include all
independently selected boundary occurrences; reference reachability is computed
on the full word before projecting onto the sample.

The test consumer previously dropped outer coordinates of arithmetic events.
It now reconstructs every supplied ordinal frame with checked arity/overflow;
unsupported quotient frames remain unavailable. It also consumes the request's
actual regional export rather than inferring capability from a provider name.
These are diagnostic repairs, not changes to production mathematics.

| Original-sized region | Production chosen top interface | Arithmetic retained description |
|---|---|---|
| Cast reduction, 64 rows | 262 ports, 258 cells, 1089 expression nodes | 3 sites / 54 primitive relations |
| Cast expansion, 64 rows | 8704 ports, 8448 cells, 34937 nodes | 2 sites / 30 primitive relations |
| Cumsum, 8 chunks x 128 | 3064 ports, 2048 cells, 2146 nodes | 5 sites / 125 primitive relations |

Arithmetic counts stay fixed when row/chunk counts change at fixed width.
Finite expansion refuses the larger instances under unchanged limits. The
chosen production interfaces grow with materialized scalar cells. This is a
mathematics/interface comparison, not a bound on every later query or emitted
recipe. Changing cumsum width also changes the observed coefficient class;
those runs are not one fixed-C asymptotic family. Most arithmetic process times
are near the approximately 65-ms sampling granularity; no speedup is claimed.

Reproduce from the worktree (all artifacts are disk-backed):

```sh
.venv/bin/python test/lit/pto/Inputs/check_compact_arithmetic_coverage.py \
  --optimizer .local/section8-refactor/source-pinned/candidate-build/tools/pto-test-opt/pto-test-opt \
  --trace-tool .local/section8-refactor/source-pinned/candidate-build/tools/pto-test-opt/pto-sync-input-test \
  --output .local/compact-class-checkpoints/stage1/final --scaling
```

| Established result | Remaining premise | Appropriate paper claim |
|---|---|---|
| Indexed scalar relations have constant static inventories at fixed dimension/coefficient class | Parent selection and scalar-to-vector exports can enumerate | Restricted arithmetic can supply compact regional interfaces |
| Bounded covers and all event queries agree for modeled scalar effects | Native scalar readiness/protection is a supplied contract | Exactness relative to the qualified input model |
| Original logical insertion succeeds | Physical feasibility and device correctness are separate | No new whole-kernel or device-performance claim |

This table records the stage-1 checkpoint. Stage 4 subsequently qualifies the
expanded original-coordinate endpoint adapter; see
[the adapter checkpoint](frontier-expanded-endpoint-adapter-20261020.md).
The historical queries/selectors and scaling measurements above remain separate.
