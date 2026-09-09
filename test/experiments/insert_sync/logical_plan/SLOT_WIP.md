# Occurrence-dependent slots: review checkpoint

This is a **WIP checkpoint**, following accepted Steps 1–3. Step 4 is not
accepted: the full two-/three-buffer campaign, adverse-selector campaign and
compile-cost gate remain open. Allocation/GEMM development has not resumed.
The default planner and 384-million work allowance are unchanged. No device
correctness or runtime measurement is claimed.

## Changes available for review

- Optional scalar projections reuse the original occurrence importer and its
  loop/parameter identities. Unsupported expressions retain conservative
  conflicts. Recursive visits are charged, including failed expressions in
  shared SSA DAGs; definite failures are cached after definition domains are
  complete. Exhaustion is an explicit analysis limit.
- A shared multi-tile layout utility is used by physical qualification and
  existing address lowering. Actual slot addresses retain their original
  index order. Qualification rejects a raw-versus-aligned stride mismatch
  before coarse interval pruning can lose a real conflict.
- Local conflicts intersect the original ordered relation with exact source
  and target selector domains for physically overlapping slots. Selector
  numbers alone do not prove disjointness. Full in-range coverage is required;
  dynamic out-of-range selection must retain conservative conflicts because
  lowering defaults to slot zero rather than taking a remainder.
- Fresh emitted reconstruction rebuilds physical mappings and the scalar
  domains used by discovery. It retains the original requirements and checks
  actual event matching, reuse, payload boundaries and retirement.
- Physical-slot overlap uses an output-sensitive interval sweep, with sorting
  and actual output charged before allocation. Literal selection still pays
  for qualification of its full root table. Fixed-coordinate extraction scans
  each constraint row once instead of repeatedly rescanning it.
- Difference checking records qualification, partition and implication costs
  separately. Boolean partition emptiness eliminates unit-defined variables
  on a disposable copy before integer sampling. This exact simplification
  never returns a projected occurrence relation; nonunit parity constraints
  remain intact.

## Native precision result

The unchanged `inputs/slots/rotating_d2.pto` executes two loads and two readers
per iteration, using actual `multi_tile_get` selectors. The independent bounded
oracle interprets original selectors and byte intervals, checks every local
RAW/WAR/WAW relation (including absent requirements), and replays emitted
completion and token consumption. Compared with the frozen Step 3 output:

| Observation | Step 3 | This WIP |
|---|---:|---:|
| First readers acquiring the independent second preload | 17 | 0 |
| Payload completion-prefix constraints removed | — | 34 |
| Payload completion-prefix constraints strengthened | — | 0 |
| Static set/wait pairs | 2 | 3 |
| Static MTE2 barriers | 1 | 0 |
| Terminal ALL | 1 | 1 |
| Executed pairs at bound 7 | 13 | 20 |
| Executed MTE2 barriers at bound 7 | 7 | 0 |

Scalar-operation counts are unchanged; replay steps rise with the extra event
commands. This is an overlap/ordering result, not a command-count win or a
device speedup. The fresh native invocation took **47.11 seconds**. The older
Step 3 conservative plan took about 0.22 seconds in its recorded probe; the
precision gain is currently expensive.

The unchanged two-buffer input also constructs strictly in the latest probe,
taking **30.25 seconds**. Of that, reconstruction is 11.94 seconds. Subtraction
records 17.84 seconds, including 9.49 seconds in qualification and 8.31 seconds
in partition work. Timers are nested, not additive exclusive measurements.
The previous attribution probe took 51.62 seconds and ran beside another
single-worker job; it is not a controlled timing baseline. The query counts
are unchanged (337 subtractions, 383 qualifications, 1,108 partitions).

The last three-buffer probe timed out at 120 seconds **before** the latest
emptiness optimization. It has not been rerun in this pause checkpoint. A
successful two-buffer probe is not full buffering acceptance.

## Validation and remaining gates

The checkpoint build uses the existing Release `-O1`/assertions configuration
and pinned LLVM 19.1.7. The focused gate passed in **41.04 seconds**, including
183 MLIR/libisl relation
checks, 18 scalar-domain cases, 15 physical-mapping cases, interval scaling
and budget refusal, and the existing authored-sync, retirement, endpoint,
occurrence, reconstruction-mutation and native scaling checks. A newly added
mapping fixture initially used 64 slots; the dialect rejects counts above 16.
The fixture was corrected to the legal maximum without changing the verifier.

A separate final-source four-kernel run also passes strict construction,
payload/allocation/view/ABI preservation, C++ emission and the existing
readiness observations for one buffer, online softmax, Q projection and QK.
Their static inventories remain those reported for Step 3. This run uses
`check_constructor.py --focused --case one_buffer --case online_softmax
--case q_proj --case qk_matmul`; `--focused` deliberately excludes the still
unaccepted buffering campaign. Its evidence is in `pause-checkpoint-four`.

The twelve slot inputs and `check_slots.py` are retained as the next campaign.
Only rotating D2 has completed the end-to-end precision check at this checkpoint.
The full overlap, additional-reader, common-consumer, skipped-reader, equivalent
predicate, unknown-selector and out-of-range native cases remain unrun. The
slot-specific emitted mutation challenges and parameter-binding rejection
regression are also pending. Their
presence is not claimed as passing coverage. In particular, missing optional
facts and reconstructed parameter-binding differences must remain conservative
or cause rejection; they must never authorize an omission.

All three reviewers accept this as a tested WIP checkpoint. The performance
reviewer retains a block on full Step 4 acceptance; this is not approval of
broader supported coverage. Further semantic expansion and allocation/GEMM remain
gated on full native acceptance and cost results. Fresh checks are retained.

Local evidence is under `insertsync-builds/campaign/logical-plan/step4-buffering/`:
`pause-checkpoint-d2`, `pause-checkpoint-profile`, the retained `step3-baseline`,
and earlier `probe1`/`attribution1`. Focused runs are under the build's
`test-results/oahs/`. These are local campaign artifacts, not remote device
artifacts. Tested pre-commit binary SHA-256:

- Compiler: `78688282554a7213eb8445bb0ee5cc7ac5f6602a5694e8a2074e30a7633ed128`.
- Native driver: `5a8d929a5ba5a00ac37d3134e41f2177b198ac588992f701634e82fc4252db21`.
