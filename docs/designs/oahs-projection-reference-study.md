# Projection reference-plan study

Local artifact directory: `../../../projection-reference-work/` relative to
this design directory (workspace sibling of the repository). Artifact filenames
below refer to that directory.

Date: 2026-09-19. Compiler baseline: `495fb9cbda7649f15a7fc09d6b1d91ffb7737d54`.
The device campaign remains pinned to that revision. This study changes no
production constructor, importer, payload, hardware contract, or device task.

## Result

A shared remaining placement defect is confirmed in down, gate/up, KV, q/out
and LM head: readiness for a MAT generation is sometimes published after later
unrelated MAT loads, immediately before its B extraction. A source-local
publication and one acquisition at the first B extraction remove that ordering.
Moving the acquisition to loop entry instead is safe in the example but adds
B-load completion before the independent first A extraction.

The generated reference PTO files implement that narrower placement. They are
manually derived diagnostic plans, not new autosync output. All 37 paired paths
pass strict local memory, matching and consumption-before-republication checks;
511,187 physical-conflict checks were exercised. Payload instructions, concrete
effects and original contexts are identical, and no finish-to-issue relation is
added. Six reference modules also parse and verify in production pto-test-opt.
Parsing is not native all-path protocol acceptance. No numerical or device
measurements were made, and these plans are not claimed optimal.

## Exact down_proj deadlines

`down-acquisitions.csv` inventories every ordinary decision's acquisition,
physical class and consumer. `pypto_lib__prefill_fwd__0/current.json` also records
all recurring endpoints and command words. Cuts and cells refer to this exact
166-site imported program, not dynamic operation counts.

| Required completion | Physical interval (bytes) | Original producer / first consumer | Current placement | Finding |
| --- | --- | --- | --- | --- |
| A0 readiness | MAT [0,65536) | TLOAD cut 35 / TEXTRACT 56 | publication 36, acquisition 54 | Early source retained; first consumer is also first MTE1 operation. |
| B0 readiness | MAT [65536,196608) | TLOAD 42 / TEXTRACT 58 | publication and acquisition 58, plus reverse acknowledgment | Publication observes later A1/B1 loads; repeats on every inner visit. |
| A1 readiness | MAT [196608,262144) | TLOAD 46 / TEXTRACT 84 | publication and acquisition 82 | Broad second-child entry prefix includes B1; still open after this experiment. |
| Tail A readiness | MAT [262144,327680) | TLOAD 118 / TEXTRACT 128 | publication 119, acquisition 126 | Separate source retained. Shares bytes with part of main-body B1. |
| Tail B readiness | MAT [65536,196608) | TLOAD 125 / TEXTRACT 130 | publication and acquisition 130 | Repeated receipt although producer is outside child; same first-use opportunity. |
| Prior ACC reader | ACC [0,131072) | FIX read / M initializer 69 or 141 | child entry 54 or 126 | Real cross-output-tile reuse obligation; unchanged. |
| Result readiness | ACC [0,131072) | M prefix / FIX store 158 | exchange at 158 | Unchanged. |

B1 occupies [262144,393216), represented by cells 10 and 11 because the odd
K tail reuses its first half for A. Allocation names therefore are not physical
generation identities. Initial A0 and tail-A overwrite returns are separately
recorded in the CSV; the reference changes neither them nor L0 bank protocols.

In the two-chunk, one-tile trace, payloads 0..3 are A0, B0, A1, B1 loads;
payloads 4 and 5 are first A0 and B0 extracts. Current edges 2->5 and 3->5
are absent in the reference. A child-entry B wait instead adds 1->4.

## Reference construction and transfer

After the original enclosing B load, publish the existing forward key. At B's
first extraction, guard its acquisition and the existing reverse acknowledgment
with the original induction-variable-equals-lower-bound condition. Later reads
retain the acquired completion. No new event key or runtime history is added.
All BF16 M fences, MTE2 fences, L0 protocols, other waits and payloads remain.

The admitted reference children have constant positive bounds and steps; their
B generation is invariant and first B read unconditional. This does not admit a
producer followed by an arbitrary zero-reader region. Whole-episode skips and
varying nonempty lengths are covered separately by the reference regression.

LM head needs physical alias matching: its B load and extraction use different
alloc_tile SSA values at the same address/extent (one uses a reshaped view).
The diagnostic generator recognizes that equality; this is not a substitute
for native layout/effect import.

| Representative checked path | Current pairs | Reference pairs | Removed finish-to-issue relations | Added |
| --- | ---: | ---: | ---: | ---: |
| down, 17 chunks, one tile | 359 | 341 | 184 | 0 |
| down, 17 chunks, two tiles | 712 | 676 | 368 | 0 |
| gate/up, one active phase | 418 | 398 | 230 | 0 |
| gate/up, both active phases | 830 | 790 | 460 | 0 |
| KV, one active arm | 418 | 398 | 230 | 0 |
| q/out, one active phase | 418 | 398 | 230 | 0 |
| LM, bounded one tile | 393 | 363 | 235 | 0 |
| LM, bounded two tiles | 780 | 720 | 470 | 0 |

Counts include invocation endpoints. LM starts 744/768/792 are original
output-loop start parameters, not device core IDs; full 33-tile dispatch is
not covered. Raw paths, counts and relations are in `reference-ordering.json`.
Do not interpret every later-load edge listed there as avoidable: prior required
receipts and source-prefix semantics can force some of them.

## Discriminating negatives

`negative-results.json` and `negative_checks.py` record:

- Current down plan fails the early-B ordering assertion; reference passes.
- Removing B readiness fails local memory completion.
- Consuming its single token on every iteration fails matching.
- Acquiring at child entry passes safety but adds B-load -> first A-extract.

Repository tests now include a portable first-consumer command-graph regression
and an opt-in `check_projection_trace.py --require-early-mat` target. That flag
is deliberately not enabled on the current compiler's lit invocation: current
production still has the diagnosed defect. It is not evidence of a compiler fix.

## Separate fence opportunity

One diagnostic cold causal check per completed current plan omits all MTE2
fences at once, preserving every event. No production deletion pass is proposed.

| Module | Removed static fences | Unchanged all-path checker |
| --- | ---: | --- |
| down | 2 | Rejects: unresolved original byte completion at cut 42 (B0 load). |
| gate/up | 2 | Accepts. |
| KV | 2 | Accepts. |
| LM | 1 | Accepts. |
| out | 1 | Accepts. |
| q | 1 | Accepts. |

This establishes redundancy in those *completed imported plans*, not why the
fence was necessary in a partial checkpoint. Investigate selecting actual MAT
readiness/release support before the overwrite repair. Down's refusal requires
its own incoming/tail/zero-trip analysis. Do not generalize the other verdicts
to down or remove fences based solely on a finite trace. These verdicts are
separate from the early-B experiment; its reference files retain all fences.

## What an efficient reference should preserve

Use the device-measured existing plan on identical payloads as the first timing
baseline. It is not a proved optimum or a universal correctness oracle. Preserve
original tiling, dtype, layouts, core assignment and instruction sequence while
constructing a competing reference. Require exact first-use readiness, preceding
same-bank reuse, actual consumption evidence, necessary BF16 M ordering and
invocation closure. Measure latency/timelines, not only event counts or payload
reachability. The previous GEMM campaign showed that reachability alone can miss
the performance effect of named drains.

External references inspected on 2026-09-19:

- [PTO A2/A3 GEMM implementation](https://github.com/hw-native-sys/pto-isa/blob/main/kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp):
  useful placement reference. ProcessKIteration publishes after each MAT load,
  acquires at that operand's first extraction and releases MAT after its last
  extraction. Its default FP16 contract and tiling differ from the BF16 Qwen
  projections; neither its M-barrier omissions nor latency transfer directly.
- [Its benchmark description](https://github.com/hw-native-sys/pto-isa/blob/main/kernels/manual/a2a3/gemm_performance/README.md):
  candidate for a separately pinned stripped-sync/manual benchmark. Not run here.
- [PTO simple matmul](https://github.com/huawei-csl/pto-kernels/blob/main/csrc/kernel/kernel_simple_matmul.cpp):
  useful small correctness control, not the deep-K pipelined projection target.
- [PTO flash-attention example](https://github.com/hw-native-sys/pto-isa/blob/main/kernels/manual/common/flash_atten/README.md):
  later reference candidate; full queue and cross-core contract must be checked.

These main-branch links are a survey, not immutable benchmark pins. Before
adopting an external fixture, pin the source revision and archive its payload,
manual protocol, lowering flags and validation contract.

## Next production mechanism

`loopEntryFrontier()` currently requires every first observer operation to need
the transferred classes. That correctly rejects moving B's wait before A.
`sourceFrontier()` requires an acyclic frontier; native counted specialization
also excludes these non-unit-step loops. The missing interface is the **first
participating consumer of an invariant physical generation**, not engine entry.

Implement a qualified first-consumer boundary with these premises:

1. Original control proves a nonempty reader region and identifies its first
   participating use; unsupported/optional reader paths remain conservative.
2. No intervening write regenerates the relevant physical bytes. Alias views
   share the physical generation; a reload creates a new one.
3. Publication retains the earliest sufficient source prefix. Acquisition stays
   at the first actual consumer, including ordered-word publication checks.
4. Exactly one acquisition matches each selected publication. Keep actual
   acknowledgments or prove an already-required return covers rearming.
5. Later reads inherit real acquired credit through the region. No receipt is
   seeded merely because the interface was recognized.
6. Compose entry/backedge/exit without a product of full nested loop histories.
   Count admitted prefix/interface size and replay separately.

The constructor must reproduce the reference before claiming this fixed. Do not
weaken the existing loop-entry rule or install a completed-plan rewrite. After
that, treat completed MAT-cycle support and remaining second-child A1 readiness
as distinct experiments. Revalidate GEMM, all six attention plans and the corpus
when production code changes. As a baseline check in this study, all six
attention modules were reconstructed and remained byte-identical; the current
GEMM passed one/two/four-tile checks at 200/394/782 pairs, zero named barriers
and one terminal ALL. No production code changed.

## Reproduction

From `/home/toni/work/pypto3_sync_more`:

```
python3 projection-reference-work/make_reference.py
python3 projection-reference-work/compare_reference.py
python3 projection-reference-work/negative_checks.py
python3 PTOAS-oahs-clean-m1/test/oahs/check_projection_trace.py \
  projection-reference-work/pypto_lib__prefill_fwd__0/reference.pto --require-early-mat
ctest --test-dir oahs-m1-core-build -R oahs_first_consumer_reference --output-on-failure -j1
```

The scripts intentionally depend on the archived current plans in
`projection-overlap-work/corpus`. Diagnostics use a reporting wrapper compiled
against the current production core; `diagnostic/build.py`, source, compiler log
and per-module current.json/current.log are retained here. Each regenerated
current.pto was byte-identical to its baseline. Reference parse logs and exact
edit records accompany each module. This workspace is local; any remote task
must package these files explicitly.
