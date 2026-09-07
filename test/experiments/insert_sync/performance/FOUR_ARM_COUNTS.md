# Four-arm synchronization reference

Frozen fixtures at 9e061dcf; before the separate FlashAttention entry-aware TFREE repair.

Each cell is set/wait pairs / named-pipe barriers / PIPE_ALL. Original results are reused from the preceding campaign. Revised results are reused from the MMAD off/on campaign.

| Fixture | Hand-tuned | Original InsertSync | Revised MMAD off | Revised MMAD on |
| --- | ---: | ---: | ---: | ---: |
| one_buffer | 6 / 0 / 1 | 6 / 0 / 1 | 6 / 0 / 1 | 6 / 0 / 1 |
| two_buffer | 12 / 0 / 1 | 12 / 2 / 1 | 12 / 2 / 1 | 12 / 2 / 1 |
| three_buffer | 18 / 0 / 1 | 18 / 3 / 1 | 18 / 3 / 1 | 18 / 3 / 1 |
| four_use | 24 / 0 / 1 | 24 / 2 / 1 | 24 / 2 / 1 | 24 / 2 / 1 |
| historical_gemm | 53 / 0 / 0 | 44 / 21 / 1 | 44 / 24 / 1 | 44 / 10 / 1 |
| topk_128 | 10 / 22 / 0 | 15 / 16 / 1 | 15 / 16 / 1 | 15 / 16 / 1 |
| conv2d_interior | 24 / 0 / 0 | 0 / 8 / 1 | 0 / 8 / 1 | 0 / 8 / 1 |
| flash_attention_cube | 18 / 0 / 1 | 27 / 0 / 1 | 13 / 0 / 1 | 13 / 0 / 1 |
| triangular_inverse_16 | 278 / 105 / 0 | 277 / 0 / 1 | 277 / 0 / 1 | 277 / 0 / 1 |
| gdn_wy | 16 / 6 / 2 | 32 / 14 / 2 | 32 / 14 / 2 | 32 / 14 / 2 |
| kda_wy | 17 / 7 / 2 | 35 / 12 / 2 | 35 / 14 / 2 | 35 / 14 / 2 |

The A2/A3 counts agree. MMAD changes only GEMM PIPE_M: 18 -> 4 sites, or 1,408 -> 176 executed barriers on the primary core-0 replay path. No PIPE_ALL changes.

GEMM named sites: original M=18, MTE1=2, FIX=1; revised off additionally MTE2=3; revised on M=4, MTE1=2, MTE2=3, FIX=1. All remaining per-pipe counts are preserved in the companion JSON.

The separate FlashAttention entry-aware TFREE device repair changes the source population and revised pair inventory to 15. Do not substitute that figure into this frozen comparison.
