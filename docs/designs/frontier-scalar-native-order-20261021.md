# Native scalar order in experimental current-cut realization

The experimental adapter now discharges qualified scalar self-prefix obligations
using native order. It emits recurrent scalar/vector examples with their original
GETVAL/SETVAL operations, scalar-derived branches and enclosing loops intact.
The source-pipe fixed point, selected services, role maps, production dispatch and
allocation algorithms are unchanged. Whole-Delta emission remains unavailable
at a separate primitive-admission boundary.

The [receipt](frontier-scalar-native-order-20261021.json) pins the actual starting
dirty tree at `29cd96a2af94fd9d9b5c5711561a0222df67b37a`, final source contents,
tools, commands and disk-backed evidence under
`.local/scalar-native-order-20261021/`. This continues the uncommitted
[source-pipe checkpoint](frontier-conservative-services-prototype-20261021.md);
its source/report contents were preserved. No device was used. These changes
remain experimental and uncommitted.

## Native contracts and preserved hardware facts

| Contract | Scope and evidence | Treatment in this task |
|---|---|---|
| Scalar storage-pair protection | Existing `ptoStorageProtection()` protects same-PIPE_S RAW/WAR/WAW. The official [CANN PipeBarrier restrictions](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0271.html) state that scalar synchronization is automatic and PIPE_S barriers cause a hardware error. | Existing provider remains unchanged. No scalar barrier is emitted. |
| Scalar completion-to-next-start order | Supplied task premise, scoped here to A3/AIV, synchronous noncompound `pto.tgetval`/`pto.tsetval` on vector tile storage. Getter result readiness also uses existing `resultAvailability`. Completion means the modeled scalar access has completed, not that an asynchronous vector producer has completed. | New prototype-only capability records `C_i -> I_next` for the next executed scalar operation. Other scalar contracts remain unavailable. |
| Accumulator-chain protection | Existing A3 Mmad qualification checks effective dimensions, in-place accumulation, accumulator type/physical atoms, PIPE_M and `(m/16)*(n/16) >= 10`. Initializers, interference and visit scope are handled by `HardwareProtection`/`InvocationProtection`. | Unchanged and regression-tested. This is storage-specific protection, not universal matrix-pipe serialization. A FIX-pipe store reading the accumulator interrupts the chain. |
| Cross-pipe visibility | Existing A3 capability checks and ordered forward/acknowledgment services. | Scalar serialization does not discharge these services. |

The accumulator exemption is implemented on the matrix accumulator chain; it is
not a blanket PIPE_FIX exemption. This distinction preserves the existing useful
rule without extending it to unrelated operations. Seven existing protection,
insertion, capability and allocation files are byte-identical to HEAD; hashes
are recorded in `protection-preservation.json`.

## Implementation and preservation argument

[CurrentCutNativeOrder.h](../../tools/pto-test-opt/CurrentCutNativeOrder.h) owns
the small experimental native-prefix query. It reuses existing target presence,
result-availability and scalar storage-protection checks. Unknown target,
storage domain, operation or compound phase does not become serialized merely
because its nominal pipe is S. Non-S pipes remain overlapping for this
whole-prefix query; their existing scoped storage protection is unaffected.

[CurrentCutServiceChecks.cpp](../../tools/pto-test-opt/CurrentCutServiceChecks.cpp)
consumes the same generated structured plan as before. A selected self-prefix
barrier is counted, then either discharged natively or emitted through the
existing barrier service. If neither is available, the request refuses.
Cross-pipe commands keep their original cuts and FSET/FWAIT/RSET/RWAIT order.
Existing staging, physical-map emission and directional reservation checks are
reused. No SET-before-WAIT batching is introduced.

For a qualified scalar consumer, the previous executed scalar completion
precedes its start. Native completion order puts every earlier scalar completion
before that previous completion. Thus every selected scalar self-prefix
obligation already holds. Removing the proposed scalar barrier changes no
payload reachability in this corrected native graph. Skipped branches, empty
visits and counter resets do not affect the argument: the predecessor is the
previous **executed** scalar operation, not the previous static site or loop
iteration. This is realization of unchanged obligations, not service pruning or
a new dataflow algorithm.

The isolated invocation must have no pre-existing commands or hidden event
users, must pass existing readiness/physical preflight, and ends with the
existing ALL completion. Each service closes its notifications; ALL completes
pending work rather than consuming an unmatched flag. Eligible IDs remain in
the six-ID pool. Tests use 3 and 5, while the existing allocator's directional
tail reservations are checked independently of those fixture choices.

## Independent bounded validation

The Python reference explicitly supplies scalar native edges; it does not call
the C++ capability predicate. Its byte/layout interpreter extracts conflicts
from original accesses rather than compiler cell partitions or generators.
Real scalar reads observe independently evaluated memory contents. These
observations are attached only after lowering for the bounded command trace;
they are not compiler-time replacement values or guard qualifications.
Original scalar/control readiness edges are included in both reference and
command graphs. Shared hardware/layout premises remain premises, not device
validation.

* 21 positive executions cover consecutive scalar accesses, vector-to-scalar
  visibility, real scalar-derived branches, taken/skipped producers and
  consumers, zero/short trips, repeated visits and counter resets: 170 payload
  occurrences in total.
* All selected obligations and source-pipe states agree with the independent
  origin solver. Matching closes; command graphs are acyclic; eligible IDs and
  causal consumption-before-republication pass for nonempty flag traces.
* Across all four payload event-kind combinations, physical lowering adds and
  removes zero pairs relative to the selected native-aware plan. The checked
  plans also have zero excess completion-to-start pairs against these fixtures'
  concrete reference. This is finite evidence, not a general exactness claim.
* An oracle-only graph retaining the proposed abstract scalar barriers has the
  same full payload relation as actual native discharge: zero added/removed
  pairs in all 21 cases. No illegal scalar barrier is emitted for this check.
* Three scalar-only cases need no flags or scalar barriers and retain one
  terminal ALL site. The largest executes 19 payloads, including changed scalar
  observations after writes.
* Negative controls detect a removed necessary vector barrier, a removed
  vector-to-scalar four-command service, and omitted scalar native edges.
  A nominal scalar SETVAL on unsupported ACC storage is refused.
* The preceding 24 non-scalar emitted cases remain covered. The expanded gate
  passes **32/32**; a final refreshed six-test subset passes **6/6**. Hardware
  checks include 304 independent scans, 19 cross-region contracts, 11 composed
  traces and 26 structured/reset/empty checks. Scalar protection also passes
  303 scans, 392 rotating prefixes and 64 structured profiles.

Physical PTO IR and C++ generation succeed for the recurrent scalar/vector
example; C++ is 2,700 bytes and retains GETVAL/SETVAL and original branching.
The receipt pins the existing compatibility CLI/native libraries used for this
generation. It is not device compilation or device execution.

## Scaling and cost

| Vector stages | Inner trips tested | OUT facts | Selected self-prefixes | Native discharges | Static SET/WAIT sites | Barriers including return ALL | Added constant ops |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1, 256 | 45 | 5 | 3 | 20 | 3 | 40 |
| 2 | 1, 256 | 49 | 6 | 3 | 24 | 4 | 48 |
| 4 | 1, 256 | 57 | 8 | 3 | 32 | 6 | 64 |
| 8 | 1, 256 | 73 | 10 | 3 | 48 | 10 | 96 |

Each paired trip count retains the same graph, obligations and command sites.
The extra scalar IR consists of constant map operands; there is no added runtime
ID arithmetic or selector circuit. Original scalar control remains. Text sizes
range from 4,244 to 9,411 bytes (the changed trip literal accounts for a byte
difference). Access recovery, phase indexing, descriptor/CFG construction,
solving, plan serialization, map preparation and emission/verification have
separate counters/timings in the raw receipt. They are descriptive single runs,
not speedup evidence; inclusive stages must not be summed indiscriminately.
Native discharge adds constant-time capability work per selected obligation,
within the existing O(mk) command construction bound.

## Delta and production preservation

The original Delta inventory now marks site 9 (line 121) and site 20 (line 153),
both actual TGETVAL, as serialized. All IN/OUT states, selected services,
logical/physical abstract plans and role maps equal the preceding checkpoint.
The solver and plan-generator sources are also byte-identical to task start.

Whole-Delta experimental emission first refuses at line 100, site 0:
`current-cut: unsupported command/compound/control contract for pto.load`.
A minimal pointer-load-to-loop-bound input reproduces this admission gap while
the input exporter recovers its GM read effect and source-pipe obligations.
`pto.load` can produce scalar or vector values; it was not silently folded into
the synchronous vector-tile scalar contract. This is an operation/admission
boundary, not PIPE_S-barrier unavailability, a proved capacity obstruction, or
mathematical intractability. Other unadmitted Delta operations may follow; no
whole-Delta realization claim is made.

The ordinary production optimizer is byte-identical to the task-start binary.
Nine kernel/alias preservation cases succeed through logical, allocation-support,
physical and C++ stages. Of 36 historical artifact comparisons, 32 are
byte-identical. The four differences concern Persistent GEMM may-not-alias
against an older optimizer; the already recorded scoped nonescaping conjunction
comparison covers its three PTO artifacts. This is not a new golden update or
a claim of byte equality for that C++ output. The 12 artifacts whose historical
receipt pins the actual current production optimizer are all byte-identical.

Provenance caveat: the reused preservation runner writes to its previous fixed
artifact directory and refreshed those raw receipts/timings. Current copies are
under this task's directory. Earlier report/source files remain unchanged; the
earlier report's raw preservation-receipt hash is historical. The new receipt
does not pass that overwritten artifact off as immutable earlier evidence.

## Reproduction and paper consequences

Use the pinned Linux snapshot build (the worktree's legacy metadata points at a
different platform). Copy changed files to `candidate-source` before rebuilding
the two native test tools with `--parallel 2`; run lit with `-j1`. Exact commands,
tool hashes, the start-content snapshot and per-case commands are in the receipt.
The standalone scalar driver accepts explicit `--trace-tool`, `--output`,
`--cli` and workspace `--python` arguments. Watchdogs remain 35 seconds per
bounded subprocess; the machine-wide two-intensive-worker limit was respected.

Ruff F checks with the workspace's Python 3.12 target and `git diff --check`
pass. All 17 brace-prefilter findings were individually reviewed: their complete
conditions have braced bodies; nested calls confuse the regex. Evidence and
dispositions are retained. Native builds succeed without new warnings. No full
static-analyzer, sanitizer, device or broad production-performance claim is made.

| Section | Established consequence | Limit |
|---|---|---|
| 3 | Distinguish overlapping pipes from qualified serial pipes. The weaker generic scalar model requested an unnecessary unavailable primitive. | Operation and target scope must be stated. |
| 6 | Select obligations first; realize them through native order or commands. The source-pipe dataflow need not change. | Experimental admission remains restricted; no general token realization theorem is established here. |
| 7 | Resources are required for remaining cross-pipe services, not a fictitious scalar barrier. | Bounded reuse checks and qualified invocation ownership do not prove all Delta plans feasible. |
| 5 | Exact reduction must use the correct native graph; added native edges can change covers. | Existing exact/core-chain paths and accumulator protection are untouched; their support for stronger native graphs is not inferred. |
