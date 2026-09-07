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

A barrier includes `PIPE_ALL`. “Hand-tuned” means synchronization explicitly
present in the manual fixture; InsertSync does not process that arm.

### Totals

| Fixture | Hand | Original | v2 | v2 − hand |
| --- | ---: | ---: | ---: | ---: |
| One-buffer control | 13 | 13 | 13 | 0 |
| Two-buffer control | 25 | 27 | 27 | +2 |
| Three-buffer control | 37 | 40 | 40 | +3 |
| Four-use producer | 49 | 51 | 51 | +2 |
| GEMM | 106 | 110 | 113 | +7 |
| TopK | 42 | 47 | 47 | +5 |
| Conv2D interior | 48 | 9 | 9 | −39 |
| FlashAttention cube | 37 | 55 | 27 | −10 |
| Triangular inverse | 661 | 555 | 555 | −106 |
| GDN WY | 40 | 80 | 80 | +40 |
| KDA WY | 43 | 84 | 86 | +43 |

### Set / wait / barrier breakdown

| Fixture | Hand | Follow-up v2 |
| --- | ---: | ---: |
| One-buffer control | 6 / 6 / 1 | 6 / 6 / 1 |
| Two-buffer control | 12 / 12 / 1 | 12 / 12 / 3 |
| Three-buffer control | 18 / 18 / 1 | 18 / 18 / 4 |
| Four-use producer | 24 / 24 / 1 | 24 / 24 / 3 |
| GEMM | 53 / 53 / 0 | 44 / 44 / 25 |
| TopK | 10 / 10 / 22 | 15 / 15 / 17 |
| Conv2D interior | 24 / 24 / 0 | 0 / 0 / 9 |
| FlashAttention cube | 18 / 18 / 1 | 13 / 13 / 1 |
| Triangular inverse | 278 / 278 / 105 | 277 / 277 / 1 |
| GDN WY | 16 / 16 / 8 | 32 / 32 / 16 |
| KDA WY | 17 / 17 / 9 | 35 / 35 / 16 |

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

### Executed totals

| Fixture / scenario | Hand | Original | v2 | v2 − hand |
| --- | ---: | ---: | ---: | ---: |
| One buffer, 16 trips | 133 | 133 | 133 | 0 |
| Two buffers, 16 trips | 137 | 153 | 153 | +16 |
| Three buffers, 16 trips | 141 | 157 | 157 | +16 |
| Four-use producer, 16 trips | 329 | 345 | 345 | +16 |
| GEMM, 4096³ core 0 | 7842 | 8481 | 8657 | +815 |
| TopK, 16 groups | 612 | 617 | 617 | +5 |
| Triangular inverse, 16 matrices | 10516 | 8805 | 8805 | −1711 |
| GDN, 16 chunks, all peers | 864 | 1323 | 1323 | +459 |
| KDA, 16 chunks, all peers | 928 | 1419 | 1451 | +523 |

### Executed set / wait / barrier breakdown

| Fixture / scenario | Hand | Follow-up v2 |
| --- | ---: | ---: |
| One buffer, 16 trips | 66 / 66 / 1 | 66 / 66 / 1 |
| Two buffers, 16 trips | 68 / 68 / 1 | 68 / 68 / 17 |
| Three buffers, 16 trips | 70 / 70 / 1 | 70 / 70 / 17 |
| Four-use producer, 16 trips | 164 / 164 / 1 | 164 / 164 / 17 |
| GEMM, 4096³ core 0 | 3921 / 3921 / 0 | 3437 / 3437 / 1783 |
| TopK, 16 groups | 130 / 130 / 352 | 180 / 180 / 257 |
| Triangular inverse, 16 matrices | 4418 / 4418 / 1680 | 4402 / 4402 / 1 |
| GDN, 16 chunks, all peers | 320 / 320 / 224 | 476 / 476 / 371 |
| KDA, 16 chunks, all peers | 352 / 352 / 224 | 572 / 572 / 307 |

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

On GEMM and TopK, follow-up v2 does not reduce the number of synchronization
operations relative to the original compiler. GEMM becomes worse by three
static barriers and by 176 executed barriers in the 4096³ core-0 scenario.
Compared with hand tuning, it executes 815 more local synchronization operations
in that scenario. TopK remains five operations over the hand-tuned plan.

The results support the compatibility goal of the follow-up: all paired inputs
compile again. They do not demonstrate the desired synchronization reduction.
Lower operation counts in Conv2D, FlashAttention, and triangular inverse are
valid count reductions. Like every result in this host benchmark, they are not
device performance measurements. Establishing whether they are wins requires
device correctness and timing; the coverage-checker status alone is
not a reason to declare production InsertSync incomplete.

Machine-readable artifacts are under
`insertsync-builds/campaign/manual-vs-followup-v2-counts`: `comparison.json`
contains all static and dynamic measurements, `static-counts.csv` contains 88
arm rows, and `dynamic-counts.csv` contains all 816 scenario/arm rows. The
individual compiler outputs, diagnostics, inputs, commands, and fingerprints
are retained beside them. No device execution or timing was performed.
