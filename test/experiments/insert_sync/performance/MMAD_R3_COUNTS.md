# MMAD r3: all eleven local fixtures

Compiler source: `9e061dcf91f3707dd7d5fc38ddf922186c841dce`. MMAD-chain inference is opt-in.
Native SHA-256: `69cc06f0f879b2e5828daca1923f2003604bf941a762d8e713fe32af4d3fefec`.

Both A2 and A3, combined and staged modes were run with the flag off and on. All 132 compile/replay rows pass. Each row emits both PTO and C++ (264 successful compiler invocations). Scalar replay preserves payload traces for every available off/on scenario. Conv2D and FlashAttention have static measurements only; device correctness remains a separate check.

The table shows A3 combined mode. The static inventories agree with staged mode and A2. Set and wait sites are shown as a pair inventory only when their counts balance; this does not imply that they execute equally on every path. No total or performance score is formed.

| Fixture | Arm | Pairs | PIPE_M | PIPE_V | PIPE_MTE1 | PIPE_MTE2 | PIPE_MTE3 | PIPE_FIX | PIPE_ALL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| one_buffer | Hand | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| one_buffer | MMAD off | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| one_buffer | MMAD on | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| two_buffer | Hand | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| two_buffer | MMAD off | 12 | 0 | 0 | 0 | 0 | 2 | 0 | 1 |
| two_buffer | MMAD on | 12 | 0 | 0 | 0 | 0 | 2 | 0 | 1 |
| three_buffer | Hand | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| three_buffer | MMAD off | 18 | 0 | 0 | 0 | 0 | 3 | 0 | 1 |
| three_buffer | MMAD on | 18 | 0 | 0 | 0 | 0 | 3 | 0 | 1 |
| four_use | Hand | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| four_use | MMAD off | 24 | 0 | 0 | 0 | 0 | 2 | 0 | 1 |
| four_use | MMAD on | 24 | 0 | 0 | 0 | 0 | 2 | 0 | 1 |
| historical_gemm | Hand | 53 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| historical_gemm | MMAD off | 44 | 18 | 0 | 2 | 3 | 0 | 1 | 1 |
| historical_gemm | MMAD on | 44 | 4 | 0 | 2 | 3 | 0 | 1 | 1 |
| topk_128 | Hand | 10 | 0 | 22 | 0 | 0 | 0 | 0 | 0 |
| topk_128 | MMAD off | 15 | 0 | 14 | 0 | 0 | 2 | 0 | 1 |
| topk_128 | MMAD on | 15 | 0 | 14 | 0 | 0 | 2 | 0 | 1 |
| conv2d_interior | Hand | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| conv2d_interior | MMAD off | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 1 |
| conv2d_interior | MMAD on | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 1 |
| flash_attention_cube | Hand | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| flash_attention_cube | MMAD off | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| flash_attention_cube | MMAD on | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| triangular_inverse_16 | Hand | 278 | 0 | 105 | 0 | 0 | 0 | 0 | 0 |
| triangular_inverse_16 | MMAD off | 277 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| triangular_inverse_16 | MMAD on | 277 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| gdn_wy | Hand | 16 | 0 | 6 | 0 | 0 | 0 | 0 | 2 |
| gdn_wy | MMAD off | 32 | 0 | 13 | 1 | 0 | 0 | 0 | 2 |
| gdn_wy | MMAD on | 32 | 0 | 13 | 1 | 0 | 0 | 0 | 2 |
| kda_wy | Hand | 17 | 0 | 7 | 0 | 0 | 0 | 0 | 2 |
| kda_wy | MMAD off | 35 | 0 | 10 | 1 | 2 | 1 | 0 | 2 |
| kda_wy | MMAD on | 35 | 0 | 10 | 1 | 2 | 1 | 0 | 2 |

Only GEMM changes with the MMAD option: `PIPE_M` sites fall from 18 to 4. Pairs remain 44; its exit `PIPE_ALL` remains 1. On the primary core-0 scalar path, executed `PIPE_M` barriers fall from 1,408 to 176. This is a count reduction, not a measured speedup.

FlashAttention uses the repaired entry-aware `TFREE` in both pair members. That fixture repair raises the automatic inventory from the historical 13 pairs to 15 with the MMAD flag either off or on. It is not an MMAD regression. The historical input remains reproducible at `9e061dcf` and in the archived campaign.

Device adapter synchronization is outside this table: FlashAttention has a fixed vector peer (two stripes), and the GDN adapter drains two final cross-core credits on each vector stripe. Full-group device accounting must include those mechanisms separately.

The one-buffer / 16-trips result is 66 executed pairs, zero named barriers, one `PIPE_ALL`; `MANUAL_VS_FOLLOWUP_V2.md` already contains that corrected value in the current checkout.

Raw commands, post-pass PTO, generated C++, placement inventories and replay traces are retained in the campaign directories. The companion JSON contains explicit set/wait counts and all replay scenarios.
