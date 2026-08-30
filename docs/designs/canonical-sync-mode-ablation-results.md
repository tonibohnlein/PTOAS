# Current CanonicalSync mode ablation results

- Branch/commit: `codex/canonical-sync-refined` / `bf89e4e89e3f5640d0e490846e09e60934a4c8b0`
- Cases: 29 (28-case corpus plus historical GEMM)
- At most two compiler processes ran concurrently.
- Body counts exclude the automatic final PIPE_ALL.
- Historical GEMM uses the all-GM-accesses no-alias contract; other cases use distinct GM arguments.

| Mode | Compiled | Verified | Set | Wait | Targeted barrier | Body PIPE_ALL | Body total | PTOAS s | Median ms | Serialization | Event area | Local fallback cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| insert-sync | 29/29 | - | 446 | 446 | 165 | 0 | 1057 | 14.062 | 516.6 | 0 | 0 | 0 |
| D | 7/29 | 7/7 | 45 | 45 | 9 | 0 | 99 | 3.072 | 394.5 | 396 | 348 | 0 |
| D+P | 7/29 | 7/7 | 44 | 44 | 9 | 0 | 97 | 3.047 | 388.1 | 388 | 342 | 0 |
| D+O | 8/29 | 8/8 | 97 | 97 | 9 | 0 | 203 | 4.674 | 471.6 | 81938 | 1924 | 0 |
| D+O+P | 8/29 | 8/8 | 96 | 96 | 9 | 0 | 201 | 5.197 | 471.4 | 81930 | 1918 | 0 |
| FULL-D | 29/29 | 29/29 | 302 | 302 | 272 | 0 | 876 | 15.986 | 397.7 | 317194 | 2093 | 0 |
| FULL-P1 | 29/29 | 29/29 | 291 | 291 | 258 | 0 | 840 | 19.253 | 401.8 | 315378 | 2069 | 0 |
| FULL-A | 29/29 | 29/29 | 291 | 291 | 258 | 0 | 840 | 24.493 | 532.9 | 315380 | 2069 | 0 |
| FULL-S | 29/29 | 29/29 | 337 | 337 | 221 | 107 | 1002 | 28.349 | 538.4 | 191407 | 3396 | 2 |

## Per-case body synchronization delta versus InsertSync

| Case | insert-sync | D | D+P | D+O | D+O+P | FULL-D | FULL-P1 | FULL-A | FULL-S |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 00:auto_fuse_matmul_aic | 8 | FAIL | FAIL | FAIL | FAIL | 8 (+0) | 8 (+0) | 8 (+0) | 10 (+2) |
| 01:auto_fuse_matmul_aiv_seed | 2 | 2 (+0) | 2 (+0) | 2 (+0) | 2 (+0) | 2 (+0) | 2 (+0) | 2 (+0) | 2 (+0) |
| 02:residual_rms_cast | 46 | 52 (+6) | 52 (+6) | 52 (+6) | 52 (+6) | 27 (-19) | 27 (-19) | 27 (-19) | 51 (+5) |
| 03:auto_fuse_pointwise | 4 | 4 (+0) | 4 (+0) | 4 (+0) | 4 (+0) | 4 (+0) | 4 (+0) | 4 (+0) | 4 (+0) |
| 04:auto_fuse_pointwise_chain | 5 | 5 (+0) | 5 (+0) | 5 (+0) | 5 (+0) | 5 (+0) | 5 (+0) | 5 (+0) | 5 (+0) |
| 05:rms_recip | 34 | FAIL | FAIL | FAIL | FAIL | 29 (-5) | 23 (-11) | 23 (-11) | 34 (+0) |
| 06:down_proj | 91 | FAIL | FAIL | FAIL | FAIL | 45 (-46) | 49 (-42) | 49 (-42) | 75 (-16) |
| 07:post_rms_reduce | 20 | FAIL | FAIL | FAIL | FAIL | 25 (+5) | 21 (+1) | 21 (+1) | 22 (+2) |
| 08:silu | 36 | FAIL | FAIL | FAIL | FAIL | 30 (-6) | 30 (-6) | 30 (-6) | 38 (+2) |
| 09:final_rmsnorm | 34 | FAIL | FAIL | FAIL | FAIL | 37 (+3) | 28 (-6) | 28 (-6) | 28 (-6) |
| 10:rope_kv_cache | 92 | FAIL | FAIL | FAIL | FAIL | 86 (-6) | 84 (-8) | 84 (-8) | 62 (-30) |
| 11:mlp_out_cast | 8 | 8 (+0) | 8 (+0) | 8 (+0) | 8 (+0) | 5 (-3) | 5 (-3) | 5 (-3) | 5 (-3) |
| 12:attention_finalize | 9 | FAIL | FAIL | FAIL | FAIL | 14 (+5) | 10 (+1) | 10 (+1) | 19 (+10) |
| 13:qk_matmul | 44 | FAIL | FAIL | FAIL | FAIL | 47 (+3) | 47 (+3) | 47 (+3) | 59 (+15) |
| 14:sv_matmul | 52 | FAIL | FAIL | FAIL | FAIL | 51 (-1) | 51 (-1) | 51 (-1) | 71 (+19) |
| 15:rope_kv_cache | 32 | FAIL | FAIL | FAIL | FAIL | 37 (+5) | 37 (+5) | 37 (+5) | 37 (+5) |
| 16:post_rmsnorm | 32 | FAIL | FAIL | FAIL | FAIL | 27 (-5) | 22 (-10) | 22 (-10) | 32 (+0) |
| 17:q_head_proj | 30 | FAIL | FAIL | FAIL | FAIL | 34 (+4) | 31 (+1) | 31 (+1) | 31 (+1) |
| 18:input_rmsnorm | 24 | FAIL | FAIL | FAIL | FAIL | 16 (-8) | 15 (-9) | 15 (-9) | 20 (-4) |
| 19:s2_k_hadamard_matmul | 8 | FAIL | FAIL | FAIL | FAIL | 11 (+3) | 11 (+3) | 11 (+3) | 11 (+3) |
| 20:decode_back_incore_0 | 30 | FAIL | FAIL | FAIL | FAIL | 34 (+4) | 31 (+1) | 31 (+1) | 31 (+1) |
| 21:prefill_back_incore_0 | 12 | 12 (+0) | 12 (+0) | 12 (+0) | 12 (+0) | 7 (-5) | 7 (-5) | 7 (-5) | 7 (-5) |
| 22:rms_norm | 44 | FAIL | FAIL | FAIL | FAIL | 31 (-13) | 28 (-16) | 28 (-16) | 40 (-4) |
| 23:exp_gate_mm | 70 | FAIL | FAIL | FAIL | FAIL | 67 (-3) | 55 (-15) | 55 (-15) | 112 (+42) |
| 24:mix_x | 34 | FAIL | FAIL | FAIL | FAIL | 13 (-21) | 11 (-23) | 11 (-23) | 19 (-15) |
| 25:prefill_c4_kv_score_proj | 117 | FAIL | FAIL | FAIL | FAIL | 61 (-56) | 78 (-39) | 78 (-39) | 45 (-72) |
| 26:quant | 14 | 16 (+2) | 14 (+0) | 16 (+2) | 14 (+0) | 16 (+2) | 14 (+0) | 14 (+0) | 14 (+0) |
| 27:qkv_rope_rows | 16 | FAIL | FAIL | FAIL | FAIL | 11 (-5) | 10 (-6) | 10 (-6) | 12 (-4) |
| 28:matmul_kernel_ABt_autosync | 109 | FAIL | FAIL | 104 (-5) | 104 (-5) | 96 (-13) | 96 (-13) | 96 (-13) | 106 (-3) |

## Selected mechanism origins

### D

- `direct-distance-zero-event`: 27
- `direct-release-recurrence-protocol`: 9
- `direct-targeted-barrier`: 9

### D+P

- `direct-distance-zero-event`: 26
- `direct-release-recurrence-protocol`: 9
- `direct-targeted-barrier`: 9

### D+O

- `basic-ownership-alternating-l1-protocol`: 1
- `basic-ownership-l0-operand-protocol`: 1
- `basic-ownership-stable-l1-protocol`: 1
- `boundary-guarded-accumulator-protocol`: 1
- `direct-distance-zero-event`: 27
- `direct-release-recurrence-protocol`: 9
- `direct-targeted-barrier`: 9

### D+O+P

- `basic-ownership-alternating-l1-protocol`: 1
- `basic-ownership-l0-operand-protocol`: 1
- `basic-ownership-stable-l1-protocol`: 1
- `boundary-guarded-accumulator-protocol`: 1
- `direct-distance-zero-event`: 26
- `direct-release-recurrence-protocol`: 9
- `direct-targeted-barrier`: 9

### FULL-D

- `basic-ownership-alternating-l1-protocol`: 1
- `basic-ownership-l0-operand-protocol`: 1
- `boundary-guarded-accumulator-protocol`: 1
- `completion-frontier-event`: 1
- `direct-distance-zero-event`: 154
- `direct-release-recurrence-protocol`: 1
- `direct-targeted-barrier`: 101
- `loop-carry-pipe-drain`: 67
- `repair-source-local-pipe-drain`: 2
- `repair-source-prefix-pipe-drain`: 1
- `repair-target-local-pipe-drain`: 29
- `source-local-completion-event`: 74
- `source-local-pipe-drain`: 22
- `source-prefix-pipe-drain`: 16
- `target-completion-certificate-event`: 2
- `target-local-fence-event`: 27

### FULL-P1

- `basic-ownership-alternating-l1-protocol`: 1
- `basic-ownership-l0-operand-protocol`: 1
- `boundary-guarded-accumulator-protocol`: 1
- `direct-distance-zero-event`: 152
- `direct-release-recurrence-protocol`: 1
- `direct-targeted-barrier`: 96
- `loop-carry-pipe-drain`: 66
- `repair-source-local-pipe-drain`: 5
- `repair-source-prefix-pipe-drain`: 1
- `repair-target-local-pipe-drain`: 22
- `source-local-completion-event`: 66
- `source-local-pipe-drain`: 21
- `source-prefix-pipe-drain`: 11
- `target-completion-certificate-event`: 2
- `target-local-fence-event`: 27

### FULL-A

- `basic-ownership-alternating-l1-protocol`: 1
- `basic-ownership-l0-operand-protocol`: 1
- `boundary-guarded-accumulator-protocol`: 1
- `direct-distance-zero-event`: 152
- `direct-release-recurrence-protocol`: 1
- `direct-targeted-barrier`: 96
- `loop-carry-pipe-drain`: 66
- `repair-source-local-pipe-drain`: 5
- `repair-source-prefix-pipe-drain`: 1
- `repair-target-local-pipe-drain`: 22
- `source-local-completion-event`: 66
- `source-local-pipe-drain`: 21
- `source-prefix-pipe-drain`: 11
- `target-completion-certificate-event`: 2
- `target-local-fence-event`: 27

### FULL-S

- `basic-ownership-alternating-l1-protocol`: 1
- `basic-ownership-l0-operand-protocol`: 1
- `boundary-guarded-accumulator-protocol`: 1
- `direct-distance-zero-event`: 134
- `direct-release-recurrence-protocol`: 29
- `direct-targeted-barrier`: 99
- `localized-pipe-all`: 107
- `loop-boundary-source-prefix-protocol`: 8
- `loop-carry-pipe-drain`: 20
- `repair-frontier-barrier`: 10
- `repair-frontier-event`: 8
- `repair-source-local-pipe-drain`: 4
- `repair-target-local-pipe-drain`: 31
- `source-local-completion-event`: 57
- `source-local-pipe-drain`: 17
- `source-prefix-pipe-drain`: 1
- `target-completion-certificate-event`: 1
- `target-local-fence-event`: 21

## Input manifest

The four `pypto/build_output` inputs are generated fixtures and must be
regenerated or transferred by hash. All other paths are relative to the PTOAS
repository root.

| Case | Family | Phase | Kernel | Arch | Source | SHA256 |
| --- | --- | --- | --- | --- | --- | --- |
| 00 | pypto | fusion | auto_fuse_matmul_aic | a3 | `pypto/build_output/auto_fuse_matmul/kernels/aic/fused_0.pto` | `1ea37254975908034fb430598c0a41101eece5b420229360cc3447e0e6c9aaa1` |
| 01 | pypto | fusion | auto_fuse_matmul_aiv_seed | a3 | `pypto/build_output/auto_fuse_matmul/kernels/aiv/fused_0_seed.pto` | `15e07be4533a23384560615d3248c14d01453ebe2f98eed99502451ee86ed683` |
| 02 | qwen3_14b | decode | residual_rms_cast | a3 | `test/samples/Qwen3_14BDecodeA3/kernels/aiv/residual_rms_cast.pto` | `56288eeafda7adf46a8b14a721e5760765626ce539684cfa9cbf8d2ffe9164df` |
| 03 | pypto | fusion | auto_fuse_pointwise | a3 | `pypto/build_output/auto_fuse_pointwise/kernels/aiv/fused_0.pto` | `cedffebc4c8b84ed0d0a7ac71ef72dac6e861b0cc9a79a7c38357e39cb159106` |
| 04 | pypto | fusion | auto_fuse_pointwise_chain | a3 | `pypto/build_output/auto_fuse_pointwise_chain/kernels/aiv/fused_0.pto` | `1c52d9dcd688f13a1fd31d771dc6fdd6b1cc2e3a748d7dccc333f6ca7ef67deb` |
| 05 | qwen3_14b | decode | rms_recip | a3 | `test/samples/Qwen3_14BDecodeA3/kernels/aiv/rms_recip.pto` | `b9484af18c68b515d06195d0d947e6607f5f6708a19ac7f0ac601d3a0e542843` |
| 06 | qwen3_14b | decode | down_proj | a3 | `test/samples/Qwen3_14BDecodeA3/kernels/aic/down_proj.pto` | `41e4472d4363bdac287d392bd7ccc2a1380196d583215360c3fdd95b88434305` |
| 07 | qwen3_14b | decode | post_rms_reduce | a3 | `test/samples/Qwen3_14BDecodeA3/kernels/aiv/post_rms_reduce.pto` | `042159528f83aa8a58ea8bdfa67dfb67d8e6f968a6395b8390fdc56afa676ae8` |
| 08 | qwen3_14b | decode | silu | a3 | `test/samples/Qwen3_14BDecodeA3/kernels/aiv/silu.pto` | `9a6d736bc9e5508b0436ae330a7aa05d6c28648e0bbacf3fffa4819db93a5162` |
| 09 | qwen3_14b | prefill | final_rmsnorm | a3 | `test/samples/Qwen3_14BPrefillA3/kernels/aiv/final_rmsnorm.pto` | `f41f27f5d17ba60012cf2f8810a33c3cfd3aadf3848a0180032f31fd670af8d9` |
| 10 | qwen3_14b | prefill | rope_kv_cache | a3 | `test/samples/Qwen3_14BPrefillA3/kernels/aiv/rope_kv_cache.pto` | `f0a1d1bf80761b6c3cdc9ee5f509048ce57f8e08bb297bdd0bbdb249232476b0` |
| 11 | qwen3_14b | prefill | mlp_out_cast | a3 | `test/samples/Qwen3_14BPrefillA3/kernels/aiv/mlp_out_cast.pto` | `ffda1a8c89ec2480fc900652d4abe1414e3d8abd5e145da5f9d8ed719600f4b4` |
| 12 | qwen3_14b | prefill | attention_finalize | a3 | `test/samples/Qwen3_14BPrefillA3/kernels/aiv/attention_finalize_phase.pto` | `c6df030cd53d199070b9a251e1c9bbfcdec84fca01d165a8f54da547aab13f36` |
| 13 | qwen3_32b | decode | qk_matmul | a5 | `test/samples/Qwen3DecodeA5/kernels/qk_matmul.pto` | `07b8075d2825037fbec61079c246eb3052693b3e886345b367ff4b811bfa87ae` |
| 14 | qwen3_32b | decode | sv_matmul | a5 | `test/samples/Qwen3DecodeA5/kernels/sv_matmul.pto` | `1d1eaf3dd7c5c971748e6e9ff2379ac4f895946777a864f665fd8dc08aa4b439` |
| 15 | qwen3_32b | decode | rope_kv_cache | a5 | `test/samples/Qwen3DecodeA5/kernels/rope_kv_cache.pto` | `9e25a0642f3a2289a710071c68998ab63a1ebb3d757deb68b9b9178322e23f40` |
| 16 | qwen3_32b | decode | post_rmsnorm | a5 | `test/samples/Qwen3DecodeA5/kernels/post_rmsnorm.pto` | `28e8fcdb7c95a904910b8d602e4f1a7a67f8efe9511207a2925b663ec11f5ecd` |
| 17 | deepseek_v3_2 | decode | q_head_proj | a5 | `test/samples/DeepseekV3_2DecodeFrontA5/kernels/q_head_proj.pto` | `1eba0e0ec442c619581eb1b468756750556a2eeeadd1062928cad5a645fcc0ed` |
| 18 | deepseek_v3_2 | decode | input_rmsnorm | a5 | `test/samples/DeepseekV3_2DecodeFrontA5/kernels/input_rmsnorm.pto` | `294e017fd238e2721b2199e971f624c323fffa834dc0646e15123958cacc0b58` |
| 19 | deepseek_v3_2 | decode | s2_k_hadamard_matmul | a5 | `test/samples/DeepseekV3_2DecodeFrontA5/kernels/s2_k_hadamard_matmul.pto` | `7865edd92f7241af92c139d20f140504bfd17b750461a5c3b950c464c13b10c2` |
| 20 | deepseek_v3_2 | decode | decode_back_incore_0 | a5 | `test/samples/DeepseekV3_2DecodeBackA5/kernels/deepseek_v3_2_decode_back_layer_incore_0.pto` | `089dab6da66ad8726dac69f930288a9da68e522dfab26e8999803df410c953c7` |
| 21 | deepseek_v3_2 | prefill | prefill_back_incore_0 | a5 | `test/samples/DeepseekV3_2PrefillBackA5/kernels/deepseek_v3_2_prefill_back_layer_incore_0.pto` | `d746e1119db43a6017c5ae6816a3896436e8887813a6e7061d81a8bb70745243` |
| 22 | deepseek_v4_mtp | decode | rms_norm | a5 | `test/samples/DeepseekV4DecodeA5/kernels/aiv/rms_norm.pto` | `2ca5791622b884a169b7df9bf2a528d1b9f961c0f844234a8c9c08ad77c1e1a0` |
| 23 | deepseek_v4_mtp | decode | exp_gate_mm | a5 | `test/samples/DeepseekV4DecodeA5/kernels/aic/exp_gate_mm.pto` | `4ec7a01bae9267464fde44117642142c6c871944abb17dd65dc280be44d49a24` |
| 24 | deepseek_v4_mtp | decode | mix_x | a5 | `test/samples/DeepseekV4DecodeA5/kernels/aiv/mix_x.pto` | `a271b3d7a25af3127e1aaf0e9224841dd06d2752dc60417a6fea589e038c0ba8` |
| 25 | deepseek_v4_mtp | prefill | prefill_c4_kv_score_proj | a5 | `test/samples/DeepseekV4FlashMtpPrefillA5/kernels/aic/prefill_c4_kv_score_proj.pto` | `ebae60743a6f3fc8499f695ccef94aa068aad50c37def52ffe0e58dbd8351d00` |
| 26 | deepseek_v4_pro | decode | quant | a5 | `test/samples/DeepseekV4ProDecodeA5/kernels/aiv/quant.pto` | `cf25292007a19a8df66d4200942e5f2e757f0fcf7e8f76758c395d8f3583c097` |
| 27 | deepseek_v4_pro | prefill | qkv_rope_rows | a5 | `test/samples/DeepseekV4ProPrefillA5/kernels/aiv/qkv_rope_rows.pto` | `993e9a76bcf9191b6cedd9601d151910dd33888f429e5cb4311ab0938ab78812` |
| 28 | historical | matmul_guide | matmul_kernel_ABt_autosync | a3 | `test/lit/pto/canonical_sync_historical_gemm_ownership.pto` | `e21a921e552c04d85af3e1a814a7d0a6d5b0c708dd9a52a19d92c3484f2219a9` |
