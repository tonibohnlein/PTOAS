# Hand-tuned versus InsertSync follow-up v2

This campaign compares the explicit local synchronization in each hand-tuned
fixture with the synchronization inserted from its sync-free paired input. It
also reruns the original InsertSync compiler for context. The A2 and A3 results
are identical.

The candidate is the native compiler built from the follow-up v2 working tree,
with SHA-256
`b60849b397f67efe50a26ae64702f4dcbbc299406c81a23f84fc768e285f73b2`.
All input and compiler fingerprints were checked before measurement. Both
combined and staged follow-up modes emit PTO IR and C++ for all 11 inputs and
produce identical counts. All 33 manual/combined/staged rows pass on each
architecture. The original compiler also emits both forms for all 11 inputs.

## Static local synchronization

Each cell is **set/wait pairs / named-pipe barriers / PIPE_ALL**. These
are three separate inventories, never a summed performance score. All displayed
set and wait counts balance; the pair count is the count on either side, not
a count of distinct event IDs or a proof of event-lifetime correctness.

| Fixture | Hand | Original | Follow-up v2 |
| --- | ---: | ---: | ---: |
| One buffer | 6 / 0 / 1 | 6 / 0 / 1 | 6 / 0 / 1 |
| Two buffers | 12 / 0 / 1 | 12 / 2 / 1 | 12 / 2 / 1 |
| Three buffers | 18 / 0 / 1 | 18 / 3 / 1 | 18 / 3 / 1 |
| Four-use producer | 24 / 0 / 1 | 24 / 2 / 1 | 24 / 2 / 1 |
| GEMM | 53 / 0 / 0 | 44 / 21 / 1 | 44 / 24 / 1 |
| TopK | 10 / 22 / 0 | 15 / 16 / 1 | 15 / 16 / 1 |
| Conv2D | 24 / 0 / 0 | 0 / 8 / 1 | 0 / 8 / 1 |
| FlashAttention | 18 / 0 / 1 | 27 / 0 / 1 | 13 / 0 / 1 |
| Triangular inverse | 278 / 105 / 0 | 277 / 0 / 1 | 277 / 0 / 1 |
| GDN | 16 / 6 / 2 | 32 / 14 / 2 | 32 / 14 / 2 |
| KDA | 17 / 7 / 2 | 35 / 12 / 2 | 35 / 14 / 2 |

All v2 `PIPE_ALL` sites carry `pto.auto_sync_tail_barrier` and are outside
loops: one per function, or two static sites for the vector/cube functions in
GDN/KDA. There are no in-loop `PIPE_ALL` sites in these automatic outputs.
The manual GDN/KDA `PIPE_ALL` sites are inside their vector work loops.

Conv2D, FlashAttention, triangular inverse, GDN, and KDA carry
`pto.insert_sync.effect_coverage = "gap-legacy-retained"`. This is a result from
the translator-coverage check added inside the revised InsertSync pass by commit
`60db1026a`: it could not certify the production translator's existing
synchronization summaries for those operations. That is not evidence that
production InsertSync omitted required synchronization, and the table does not
classify those rows as failures. The check may be exposing a real summary
omission, or it may lack the contract needed to recognize the production pass's
established handling. Distinguishing those cases requires operation-specific
analysis or device correctness testing.

The follow-up changes only GEMM (+3 barriers), FlashAttention (-14 sets and
-14 waits), and KDA (+2 barriers) relative to the original compiler. Its
opt-in completed-barrier pruning removes zero barriers from all 11 inputs.

## Representative executed counts

These are bounded scalar replays of concrete scenarios. They count executed
local synchronization operations; they do not model latency or device time.
GDN and KDA totals combine one cube peer and both vector stripes for 16 full
chunks.

Each cell is **executed pairs / named-pipe barriers / PIPE_ALL**.

| Fixture / scenario | Hand | Original | Follow-up v2 |
| --- | ---: | ---: | ---: |
| One buffer, 16 trips | 66 / 0 / 1 | 66 / 0 / 1 | 66 / 0 / 1 |
| Two buffers, 16 trips | 68 / 0 / 1 | 68 / 16 / 1 | 68 / 16 / 1 |
| Three buffers, 16 trips | 70 / 0 / 1 | 70 / 16 / 1 | 70 / 16 / 1 |
| Four-use producer, 16 trips | 164 / 0 / 1 | 164 / 16 / 1 | 164 / 16 / 1 |
| GEMM, 4096³ core 0 | 3921 / 0 / 0 | 3437 / 1606 / 1 | 3437 / 1782 / 1 |
| TopK, 16 groups | 130 / 352 / 0 | 180 / 256 / 1 | 180 / 256 / 1 |
| Triangular inverse, 16 matrices | 4418 / 1680 / 0 | 4402 / 0 / 1 | 4402 / 0 / 1 |
| GDN, 16 full chunks, all peers | 320 / 160 / 64 | 476 / 368 / 3 | 476 / 368 / 3 |
| KDA, 16 full chunks, all peers | 352 / 160 / 64 | 572 / 272 / 3 | 572 / 304 / 3 |

Conv2D and FlashAttention have static-only measurements because their helper
and FIFO domains are outside the bounded replay interpreter. Across all other
fixtures, the campaign evaluates 102 declared scenarios per arm. Payload and
allocation traces match the hand-tuned arm for every replayed original and
follow-up result.

## Other synchronization mechanisms retained in the pair

The local counts above deliberately exclude mechanisms that remain in both the
manual and sync-free paired inputs:

- FlashAttention retains three FIFO initializations, two `talloc`, two
  `tpush`, one `tpop`, and one `tfree` operation.
- GDN retains two `set_ffts`, four `sync.wait`, and four `sync.set` operations.
- KDA retains two `set_ffts`, six `sync.wait`, and four `sync.set` operations.

These are persistent queue or cross-core protocols. InsertSync neither inserts
nor removes them in this campaign, so adding them to only one side would
distort the local synchronization comparison.

## Interpretation

The first revision objective is to reduce `PIPE_ALL` where the completion
obligation can be established with narrower synchronization or is already
satisfied. Distinguish body barriers from the exit drain; deleting a drain
without proving completion is not an optimization. This ladder has no automatic
in-loop `PIPE_ALL` fallback to remove. The existing optional pruner only targets
named barriers and explicitly retains `PIPE_ALL`, so it does not address that
objective.

For GEMM, hand tuning uses 53 static pairs and no barriers. Both automatic
versions use 44 pairs and one exit `PIPE_ALL`. Original InsertSync has 21 named
barriers; v2 has 24. The difference is exactly three MTE2 sites, not more
`PIPE_ALL` or more pairs:

| GEMM barrier type | Original sites | v2 sites | Original executions | v2 executions |
| --- | ---: | ---: | ---: | ---: |
| PIPE_MTE2 | 0 | 3 | 0 | 176 |
| PIPE_M | 18 | 18 | 1408 | 1408 |
| PIPE_MTE1 | 2 | 2 | 176 | 176 |
| PIPE_FIX | 1 | 1 | 22 | 22 |
| PIPE_ALL, exit | 1 | 1 | 1 | 1 |

Executions refer to 4096³/core 0. The hand-tuned arm has zero in every barrier
row. Its 3921 executed pairs compare with 3437 in each automatic arm. These
mechanisms cannot be added to decide which schedule is faster.

GDN/KDA further illustrate why placement matters: their 64 executed manual
`PIPE_ALL` barriers occur in vector loops; automatic output has three exit
barriers across one cube and two vector peers. The fixed cross-core protocol
is retained. The count difference alone does not prove equivalent completion
or a device speedup.

The results establish restored compiler admission and identify synchronization
changes worth investigating. They do not establish performance rankings or
show that fewer pairs compensate for extra barriers. Prioritize proving the
exit-drain requirement for this ladder, and investigate GEMM's MTE2 barriers
separately; retain a fallback reproducer when studying allocation-induced
in-loop `PIPE_ALL`.

Machine-readable artifacts are under
`insertsync-builds/campaign/manual-vs-followup-v2-counts`: `comparison.json`
contains all static and dynamic measurements, `static-counts.csv` contains 88
arm rows, and `dynamic-counts.csv` contains all 816 scenario/arm rows. The
individual compiler outputs, diagnostics, inputs, commands, and fingerprints
are retained beside them. No device execution or timing was performed.
