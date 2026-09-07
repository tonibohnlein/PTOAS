# Frontier r4: all eleven local fixtures

Compiler source: `ca2647e71f0a069aa300c9ed9af3b498b35e741c`. Frontier refinement and MMAD-chain inference are opt-in.
Native SHA-256: `584f330df079bc6c9e3bee144ee129ec3b65d5c67138d80e4e9cd88ec070885c`.

The matched frontier-off/frontier-on campaigns cover A2 and A3, MMAD off and on, and combined and staged traversal. All 264 case/arm rows pass, producing 528 successful PTO/C++ compiler invocations. Every available scalar replay preserves the hand-tuned payload trace.

The table shows A3 combined mode. A2 and staged inventories agree. Set/wait sites are reported as a pair inventory only when their static counts balance; the barrier pipes and PIPE_ALL remain separate.

| Fixture | Arm | Pairs | PIPE_M | PIPE_V | PIPE_MTE1 | PIPE_MTE2 | PIPE_MTE3 | PIPE_FIX | PIPE_ALL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| one_buffer | Hand | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| one_buffer | MMAD off, frontier off | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| one_buffer | MMAD off, frontier on | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| one_buffer | MMAD on, frontier off | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| one_buffer | MMAD on, frontier on | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| two_buffer | Hand | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| two_buffer | MMAD off, frontier off | 12 | 0 | 0 | 0 | 0 | 2 | 0 | 1 |
| two_buffer | MMAD off, frontier on | 12 | 0 | 0 | 0 | 0 | 2 | 0 | 1 |
| two_buffer | MMAD on, frontier off | 12 | 0 | 0 | 0 | 0 | 2 | 0 | 1 |
| two_buffer | MMAD on, frontier on | 12 | 0 | 0 | 0 | 0 | 2 | 0 | 1 |
| three_buffer | Hand | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| three_buffer | MMAD off, frontier off | 18 | 0 | 0 | 0 | 0 | 3 | 0 | 1 |
| three_buffer | MMAD off, frontier on | 18 | 0 | 0 | 0 | 0 | 3 | 0 | 1 |
| three_buffer | MMAD on, frontier off | 18 | 0 | 0 | 0 | 0 | 3 | 0 | 1 |
| three_buffer | MMAD on, frontier on | 18 | 0 | 0 | 0 | 0 | 3 | 0 | 1 |
| four_use | Hand | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| four_use | MMAD off, frontier off | 24 | 0 | 0 | 0 | 0 | 2 | 0 | 1 |
| four_use | MMAD off, frontier on | 24 | 0 | 0 | 0 | 0 | 2 | 0 | 1 |
| four_use | MMAD on, frontier off | 24 | 0 | 0 | 0 | 0 | 2 | 0 | 1 |
| four_use | MMAD on, frontier on | 24 | 0 | 0 | 0 | 0 | 2 | 0 | 1 |
| historical_gemm | Hand | 53 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| historical_gemm | MMAD off, frontier off | 44 | 18 | 0 | 2 | 3 | 0 | 1 | 1 |
| historical_gemm | MMAD off, frontier on | 44 | 18 | 0 | 2 | 3 | 0 | 1 | 1 |
| historical_gemm | MMAD on, frontier off | 44 | 4 | 0 | 2 | 3 | 0 | 1 | 1 |
| historical_gemm | MMAD on, frontier on | 44 | 4 | 0 | 2 | 3 | 0 | 1 | 1 |
| topk_128 | Hand | 10 | 0 | 22 | 0 | 0 | 0 | 0 | 0 |
| topk_128 | MMAD off, frontier off | 15 | 0 | 14 | 0 | 0 | 2 | 0 | 1 |
| topk_128 | MMAD off, frontier on | 15 | 0 | 14 | 0 | 0 | 2 | 0 | 1 |
| topk_128 | MMAD on, frontier off | 15 | 0 | 14 | 0 | 0 | 2 | 0 | 1 |
| topk_128 | MMAD on, frontier on | 15 | 0 | 14 | 0 | 0 | 2 | 0 | 1 |
| conv2d_interior | Hand | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| conv2d_interior | MMAD off, frontier off | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 1 |
| conv2d_interior | MMAD off, frontier on | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 1 |
| conv2d_interior | MMAD on, frontier off | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 1 |
| conv2d_interior | MMAD on, frontier on | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 1 |
| flash_attention_cube | Hand | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| flash_attention_cube | MMAD off, frontier off | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| flash_attention_cube | MMAD off, frontier on | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| flash_attention_cube | MMAD on, frontier off | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| flash_attention_cube | MMAD on, frontier on | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| triangular_inverse_16 | Hand | 278 | 0 | 105 | 0 | 0 | 0 | 0 | 0 |
| triangular_inverse_16 | MMAD off, frontier off | 277 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| triangular_inverse_16 | MMAD off, frontier on | 277 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| triangular_inverse_16 | MMAD on, frontier off | 277 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| triangular_inverse_16 | MMAD on, frontier on | 277 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| gdn_wy | Hand | 16 | 0 | 6 | 0 | 0 | 0 | 0 | 2 |
| gdn_wy | MMAD off, frontier off | 32 | 0 | 13 | 1 | 0 | 0 | 0 | 2 |
| gdn_wy | MMAD off, frontier on | 32 | 0 | 13 | 1 | 0 | 0 | 0 | 2 |
| gdn_wy | MMAD on, frontier off | 32 | 0 | 13 | 1 | 0 | 0 | 0 | 2 |
| gdn_wy | MMAD on, frontier on | 32 | 0 | 13 | 1 | 0 | 0 | 0 | 2 |
| kda_wy | Hand | 17 | 0 | 7 | 0 | 0 | 0 | 0 | 2 |
| kda_wy | MMAD off, frontier off | 35 | 0 | 10 | 1 | 2 | 1 | 0 | 2 |
| kda_wy | MMAD off, frontier on | 35 | 0 | 10 | 1 | 2 | 1 | 0 | 2 |
| kda_wy | MMAD on, frontier off | 35 | 0 | 10 | 1 | 2 | 1 | 0 | 2 |
| kda_wy | MMAD on, frontier on | 35 | 0 | 10 | 1 | 2 | 1 | 0 | 2 |

Frontier r4 changes no static synchronization inventory in these fixtures and reports zero removed or overflow-guarded barriers. In particular, MMAD-on GEMM remains at 44 pairs, four PIPE_M, two PIPE_MTE1, three PIPE_MTE2, one PIPE_FIX and one PIPE_ALL. MMAD-off GEMM remains at 18 PIPE_M sites.

This is a negative performance result for r4's current supported proof domain, not a correctness failure. The compiler, replay, architecture and traversal cross-checks all pass. Device execution and timing were not run.

Raw commands, emitted PTO/C++, diagnostics and replay traces remain in the campaign root supplied to the summarizer. The companion JSON records campaign-relative output paths, table rows and cross-check totals; the raw results record every validated row.
