# Frontier Step2 regional validation

The scoped regional proof/adapter milestones are committed through `2a32c688d`.
All 240 original regions of the eleven prepared inputs were checked under
both alias policies. Every input hash and all 480 source-anchor references
match the earlier independent source audit. Expected families below come
from that audit; they were not inferred from the selected backend.

The serial paired campaign uses the pinned optimized baseline described in
`frontier-regional-validation.json` and the optimized candidate, both with
`-O2 -UNDEBUG` and C++17. Baseline source is exactly committed revision
`d5d65f79d`; its recorded changed paths are the committed delta from
`889457c2b`, not an uncommitted source overlay. Recorded rebuilt-source
hashes were checked against Git. All 44 requests completed without lost
exact demands, timeout or source mutation.
MayNotAlias improves from 230/240 to 233/240; MayAlias from
218/240 to 222/240. The public CLI still stops after Step2.

A successful cell means that exact mathematical demands were constructed.
Q and S record query and storage-selector availability separately. This is
not an end-to-end emission or allocation claim. Failed regions retain an
unmet obligation; none has been proved outside every paper class. The full
machine-readable report records every region, provenance, construction
counters and diagnostic hashes. Raw diagnostics remain in the campaign
artifacts and are not duplicated here.

## Paired results

| Kernel | Regions | MNA baseline → candidate | MA baseline → candidate | MNA seconds baseline → candidate | MA seconds baseline → candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| tilelang_gemm | 22 | 22 → 22 | 19 → 20 | 0.870 → 0.883 | 1.075 → 1.084 |
| pypto_gemm | 17 | 15 → 17 | 14 → 15 | 0.844 → 0.800 | 0.951 → 1.055 |
| persistent_gemm | 34 | 34 → 34 | 27 → 28 | 0.315 → 0.356 | 5.615 → 5.610 |
| group_norm | 23 | 23 → 23 | 23 → 23 | 0.056 → 0.058 | 0.497 → 3.856 |
| causal_conv1d_prefill | 48 | 46 → 46 | 46 → 46 | 0.072 → 0.072 | 0.072 → 0.063 |
| causal_conv1d_decode | 7 | 7 → 7 | 7 → 7 | 0.050 → 0.050 | 0.038 → 0.038 |
| gdn_chunk_cumsum | 9 | 9 → 9 | 9 → 9 | 0.271 → 0.254 | 0.257 → 0.265 |
| mhc_head_mix | 8 | 8 → 8 | 8 → 8 | 0.083 → 0.075 | 0.078 → 0.084 |
| elementwise_pipeline | 10 | 10 → 10 | 9 → 10 | 0.047 → 0.035 | 0.069 → 0.434 |
| gated_delta_rule | 29 | 23 → 24 | 23 → 23 | 0.060 → 0.072 | 0.059 → 0.065 |
| lossless_block_cast | 33 | 33 → 33 | 33 → 33 | 13.031 → 3.523 | 12.917 → 4.007 |

Times are whole-request wall times, not isolated stage timers. The report
records actual construction counters separately. Additional finite export
work increases some request times. Group-norm MayAlias increases from
0.497 to 3.856 seconds: node 3 now succeeds through original regional
sequence composition instead of the finite normalized fallback. Its
mathematical-attempt count decreases from 168 to 165; isolated stage cost
was not measured. Elementwise MayAlias also performs new successful
composition work. Lossless-cast conversion becomes substantially cheaper.
No claim is made about release performance relative to the stopped `-Og`
campaign. Existing-success/manual-sync filters and external watchdogs are
not mathematical class criteria.

## Remaining obligations

- Prefill nodes 0 and 7 remain unresolved under both policies. Literal
  machine arithmetic permits reversed loaded endpoints; a wrapping-aware
  exact adapter is still unavailable. No endpoint ordering is assumed.
- Gated delta nodes 0, 2, 3, 5 and 6 remain unresolved under both policies.
  Node 11 has bounded-lifetime exact demands under MayNotAlias, but lacks
  arbitrary-region queries/selectors; its parents cannot inherit success.
  Under MayAlias node 11 also lacks a complete refresh/overlap certificate.
  Per-visit floating predicates remain opaque, not integer predicates.
- MayAlias additionally leaves GEMM nodes 3 and 5 unresolved in both GEMM
  variants, and persistent-GEMM nodes 0, 2, 3, 5, 6 and 9. Complete child
  interfaces/crossing proofs are still required. Successful normalized
  whole-function results do not establish these original child regions.

These are explicit remaining capabilities, not a claim that every region
is recognized or that no tractable class applies. Streaming, finite-visit
emission and scarcity repair remain outside this work.

## Original-region ledger

Expected-family codes: F = finite occurrences; G = finite guarded or
composition; FN = finite normalization; PN = periodic normalization;
BR = boundary rotation; PP = phased periodic; A = arithmetic;
C = structured composition or certified whole-region alternative;
U = runtime proof still required. Normalization is a representation
alternative, not an additional tractable mathematical class.

Each outcome gives `class [representation]; Q/S`, where `1` means available.
`unmet` denotes an unresolved construction obligation.

### tilelang_gemm

| Node / source line | Kind | Expected | MayNotAlias | MayAlias |
| --- | --- | --- | --- | --- |
| 0 / [L10](../../test/npu_validation/benchmarks/auto_sync/prepared/tilelang/input.pto#L10) | sequence | C | periodic-storage [small-count-expanded]; 1/0 | periodic-storage [small-count-expanded]; 1/0 |
| 1 / [L11](../../test/npu_validation/benchmarks/auto_sync/prepared/tilelang/input.pto#L11) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 2 / [L30](../../test/npu_validation/benchmarks/auto_sync/prepared/tilelang/input.pto#L30) | loop | PN | periodic-storage [small-count-expanded]; 1/0 | periodic-storage [small-count-expanded]; 1/0 |
| 3 / [L30](../../test/npu_validation/benchmarks/auto_sync/prepared/tilelang/input.pto#L30) | sequence | C | regional-sequence [original]; 1/1 | unmet |
| 4 / [L31](../../test/npu_validation/benchmarks/auto_sync/prepared/tilelang/input.pto#L31) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 5 / [L39](../../test/npu_validation/benchmarks/auto_sync/prepared/tilelang/input.pto#L39) | loop | PP | regional-repetition [original]; 1/1 | unmet |
| 6 / [L86](../../test/npu_validation/benchmarks/auto_sync/prepared/tilelang/input.pto#L86) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 7 / [L39](../../test/npu_validation/benchmarks/auto_sync/prepared/tilelang/input.pto#L39) | sequence | C | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 8 / [L40](../../test/npu_validation/benchmarks/auto_sync/prepared/tilelang/input.pto#L40) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 9 / [L43](../../test/npu_validation/benchmarks/auto_sync/prepared/tilelang/input.pto#L43) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 10 / [L58](../../test/npu_validation/benchmarks/auto_sync/prepared/tilelang/input.pto#L58) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 11 / [L65](../../test/npu_validation/benchmarks/auto_sync/prepared/tilelang/input.pto#L65) | loop | FN | finite-guarded-occurrences [small-count-expanded]; 0/0 | finite-guarded-occurrences [small-count-expanded]; 0/0 |
| 12 / [L43](../../test/npu_validation/benchmarks/auto_sync/prepared/tilelang/input.pto#L43) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 13 / [L43](../../test/npu_validation/benchmarks/auto_sync/prepared/tilelang/input.pto#L43) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 14 / [L65](../../test/npu_validation/benchmarks/auto_sync/prepared/tilelang/input.pto#L65) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 15 / [L44](../../test/npu_validation/benchmarks/auto_sync/prepared/tilelang/input.pto#L44) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 16 / [L66](../../test/npu_validation/benchmarks/auto_sync/prepared/tilelang/input.pto#L66) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 17 / [L79](../../test/npu_validation/benchmarks/auto_sync/prepared/tilelang/input.pto#L79) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 18 / [L79](../../test/npu_validation/benchmarks/auto_sync/prepared/tilelang/input.pto#L79) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 19 / [L79](../../test/npu_validation/benchmarks/auto_sync/prepared/tilelang/input.pto#L79) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 20 / [L80](../../test/npu_validation/benchmarks/auto_sync/prepared/tilelang/input.pto#L80) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 21 / [L82](../../test/npu_validation/benchmarks/auto_sync/prepared/tilelang/input.pto#L82) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |

### pypto_gemm

| Node / source line | Kind | Expected | MayNotAlias | MayAlias |
| --- | --- | --- | --- | --- |
| 0 / [L10](../../test/npu_validation/benchmarks/auto_sync/prepared/pypto/input.pto#L10) | sequence | C | periodic-storage [small-count-expanded]; 1/0 | periodic-storage [small-count-expanded]; 1/0 |
| 1 / [L11](../../test/npu_validation/benchmarks/auto_sync/prepared/pypto/input.pto#L11) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 2 / [L29](../../test/npu_validation/benchmarks/auto_sync/prepared/pypto/input.pto#L29) | loop | PN | periodic-storage [small-count-expanded]; 1/0 | periodic-storage [small-count-expanded]; 1/0 |
| 3 / [L29](../../test/npu_validation/benchmarks/auto_sync/prepared/pypto/input.pto#L29) | sequence | C | regional-sequence [original]; 1/1 | unmet |
| 4 / [L30](../../test/npu_validation/benchmarks/auto_sync/prepared/pypto/input.pto#L30) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 5 / [L32](../../test/npu_validation/benchmarks/auto_sync/prepared/pypto/input.pto#L32) | loop | PP | regional-repetition [original]; 1/1 | unmet |
| 6 / [L71](../../test/npu_validation/benchmarks/auto_sync/prepared/pypto/input.pto#L71) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 7 / [L32](../../test/npu_validation/benchmarks/auto_sync/prepared/pypto/input.pto#L32) | sequence | C | finite-guarded-occurrences [small-count-expanded]; 0/0 | finite-guarded-occurrences [small-count-expanded]; 0/0 |
| 8 / [L33](../../test/npu_validation/benchmarks/auto_sync/prepared/pypto/input.pto#L33) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 9 / [L47](../../test/npu_validation/benchmarks/auto_sync/prepared/pypto/input.pto#L47) | loop | FN | finite-guarded-occurrences [small-count-expanded]; 0/0 | finite-guarded-occurrences [small-count-expanded]; 0/0 |
| 10 / [L47](../../test/npu_validation/benchmarks/auto_sync/prepared/pypto/input.pto#L47) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 11 / [L48](../../test/npu_validation/benchmarks/auto_sync/prepared/pypto/input.pto#L48) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 12 / [L62](../../test/npu_validation/benchmarks/auto_sync/prepared/pypto/input.pto#L62) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 13 / [L62](../../test/npu_validation/benchmarks/auto_sync/prepared/pypto/input.pto#L62) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 14 / [L62](../../test/npu_validation/benchmarks/auto_sync/prepared/pypto/input.pto#L62) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 15 / [L63](../../test/npu_validation/benchmarks/auto_sync/prepared/pypto/input.pto#L63) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 16 / [L65](../../test/npu_validation/benchmarks/auto_sync/prepared/pypto/input.pto#L65) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |

### persistent_gemm

| Node / source line | Kind | Expected | MayNotAlias | MayAlias |
| --- | --- | --- | --- | --- |
| 0 / [L16](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L16) | sequence | C | regional-sequence [original]; 1/1 | unmet |
| 1 / [L18](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L18) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 2 / [L43](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L43) | loop | PP | regional-repetition [original]; 1/1 | unmet |
| 3 / [L43](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L43) | sequence | C | regional-sequence [original]; 1/1 | unmet |
| 4 / [L44](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L44) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 5 / [L47](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L47) | conditional | C | regional-conditional [original]; 1/1 | unmet |
| 6 / [L47](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L47) | sequence | C | regional-sequence [original]; 1/1 | unmet |
| 7 / [L47](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L47) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 8 / [L48](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L48) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 9 / [L61](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L61) | loop | PP | regional-repetition [original]; 1/1 | unmet |
| 10 / [L118](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L118) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 11 / [L61](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L61) | sequence | C | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 12 / [L62](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L62) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 13 / [L65](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L65) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 14 / [L81](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L81) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 15 / [L89](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L89) | loop | FN | finite-guarded-occurrences [small-count-expanded]; 0/0 | finite-guarded-occurrences [small-count-expanded]; 0/0 |
| 16 / [L65](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L65) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 17 / [L65](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L65) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 18 / [L89](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L89) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 19 / [L66](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L66) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 20 / [L90](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L90) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 21 / [L92](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L92) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 22 / [L94](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L94) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 23 / [L106](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L106) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 24 / [L108](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L108) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 25 / [L111](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L111) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 26 / [L92](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L92) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 27 / [L92](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L92) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 28 / [L106](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L106) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 29 / [L106](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L106) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 30 / [L111](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L111) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 31 / [L111](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L111) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 32 / [L112](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L112) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 33 / [L114](../../test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto#L114) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |

### group_norm

| Node / source line | Kind | Expected | MayNotAlias | MayAlias |
| --- | --- | --- | --- | --- |
| 0 / [L10](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L10) | sequence | C | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 1 / [L11](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L11) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 2 / [L60](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L60) | conditional | C | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 3 / [L60](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L60) | sequence | C | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 4 / [L60](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L60) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 5 / [L61](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L61) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 6 / [L80](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L80) | loop | BR | regional-repetition [original]; 1/1 | finite-guarded-occurrences [small-count-expanded]; 0/0 |
| 7 / [L103](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L103) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 8 / [L155](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L155) | loop | BR | regional-repetition [original]; 1/1 | finite-guarded-occurrences [small-count-expanded]; 0/0 |
| 9 / [L80](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L80) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 10 / [L155](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L155) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 11 / [L81](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L81) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 12 / [L84](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L84) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 13 / [L94](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L94) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 14 / [L156](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L156) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 15 / [L159](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L159) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 16 / [L169](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L169) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 17 / [L84](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L84) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 18 / [L84](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L84) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 19 / [L159](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L159) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 20 / [L159](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L159) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 21 / [L85](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L85) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 22 / [L160](../../test/npu_validation/benchmarks/auto_sync/prepared/group_norm/input.pto#L160) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |

### causal_conv1d_prefill

| Node / source line | Kind | Expected | MayNotAlias | MayAlias |
| --- | --- | --- | --- | --- |
| 0 / [L10](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L10) | sequence | C | unmet | unmet |
| 1 / [L11](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L11) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 2 / [L107](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L107) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 3 / [L121](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L121) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 4 / [L123](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L123) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 5 / [L134](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L134) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 6 / [L146](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L146) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 7 / [L184](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L184) | loop | U | unmet | unmet |
| 8 / [L382](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L382) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 9 / [L386](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L386) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 10 / [L107](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L107) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 11 / [L107](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L107) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 12 / [L123](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L123) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 13 / [L123](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L123) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 14 / [L146](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L146) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 15 / [L146](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L146) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 16 / [L184](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L184) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 17 / [L386](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L386) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 18 / [L386](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L386) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 19 / [L108](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L108) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 20 / [L124](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L124) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 21 / [L147](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L147) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 22 / [L185](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L185) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 23 / [L191](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L191) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 24 / [L231](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L231) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 25 / [L334](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L334) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 26 / [L345](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L345) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 27 / [L346](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L346) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 28 / [L357](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L357) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 29 / [L358](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L358) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 30 / [L369](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L369) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 31 / [L370](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L370) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 32 / [L387](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L387) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 33 / [L191](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L191) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 34 / [L191](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L191) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 35 / [L334](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L334) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 36 / [L334](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L334) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 37 / [L346](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L346) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 38 / [L346](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L346) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 39 / [L358](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L358) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 40 / [L358](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L358) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 41 / [L370](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L370) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 42 / [L370](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L370) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 43 / [L192](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L192) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 44 / [L335](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L335) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 45 / [L347](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L347) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 46 / [L359](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L359) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 47 / [L371](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto#L371) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |

### causal_conv1d_decode

| Node / source line | Kind | Expected | MayNotAlias | MayAlias |
| --- | --- | --- | --- | --- |
| 0 / [L10](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_decode/input.pto#L10) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 1 / [L11](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_decode/input.pto#L11) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 2 / [L110](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_decode/input.pto#L110) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 3 / [L127](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_decode/input.pto#L127) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 4 / [L110](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_decode/input.pto#L110) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 5 / [L110](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_decode/input.pto#L110) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 6 / [L111](../../test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_decode/input.pto#L111) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |

### gdn_chunk_cumsum

| Node / source line | Kind | Expected | MayNotAlias | MayAlias |
| --- | --- | --- | --- | --- |
| 0 / [L10](../../test/npu_validation/benchmarks/auto_sync/prepared/gdn_chunk_cumsum/input.pto#L10) | sequence | C | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 1 / [L11](../../test/npu_validation/benchmarks/auto_sync/prepared/gdn_chunk_cumsum/input.pto#L11) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 2 / [L39](../../test/npu_validation/benchmarks/auto_sync/prepared/gdn_chunk_cumsum/input.pto#L39) | loop | A | regional-repetition [original]; 1/1 | regional-repetition [original]; 1/1 |
| 3 / [L52](../../test/npu_validation/benchmarks/auto_sync/prepared/gdn_chunk_cumsum/input.pto#L52) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 4 / [L39](../../test/npu_validation/benchmarks/auto_sync/prepared/gdn_chunk_cumsum/input.pto#L39) | sequence | C | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 5 / [L40](../../test/npu_validation/benchmarks/auto_sync/prepared/gdn_chunk_cumsum/input.pto#L40) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 6 / [L43](../../test/npu_validation/benchmarks/auto_sync/prepared/gdn_chunk_cumsum/input.pto#L43) | loop | A | regional-repetition [original]; 1/1 | regional-repetition [original]; 1/1 |
| 7 / [L43](../../test/npu_validation/benchmarks/auto_sync/prepared/gdn_chunk_cumsum/input.pto#L43) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 8 / [L44](../../test/npu_validation/benchmarks/auto_sync/prepared/gdn_chunk_cumsum/input.pto#L44) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |

### mhc_head_mix

| Node / source line | Kind | Expected | MayNotAlias | MayAlias |
| --- | --- | --- | --- | --- |
| 0 / [L10](../../test/npu_validation/benchmarks/auto_sync/prepared/mhc_head_mix/input.pto#L10) | sequence | C | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 1 / [L11](../../test/npu_validation/benchmarks/auto_sync/prepared/mhc_head_mix/input.pto#L11) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 2 / [L49](../../test/npu_validation/benchmarks/auto_sync/prepared/mhc_head_mix/input.pto#L49) | loop | FN | finite-guarded-occurrences [small-count-expanded]; 0/0 | finite-guarded-occurrences [small-count-expanded]; 0/0 |
| 3 / [L57](../../test/npu_validation/benchmarks/auto_sync/prepared/mhc_head_mix/input.pto#L57) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 4 / [L49](../../test/npu_validation/benchmarks/auto_sync/prepared/mhc_head_mix/input.pto#L49) | sequence | C | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 5 / [L50](../../test/npu_validation/benchmarks/auto_sync/prepared/mhc_head_mix/input.pto#L50) | loop | A | regional-repetition [original]; 1/1 | regional-repetition [original]; 1/1 |
| 6 / [L50](../../test/npu_validation/benchmarks/auto_sync/prepared/mhc_head_mix/input.pto#L50) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 7 / [L51](../../test/npu_validation/benchmarks/auto_sync/prepared/mhc_head_mix/input.pto#L51) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |

### elementwise_pipeline

| Node / source line | Kind | Expected | MayNotAlias | MayAlias |
| --- | --- | --- | --- | --- |
| 0 / [L10](../../test/npu_validation/benchmarks/auto_sync/prepared/elementwise_pipeline/input.pto#L10) | sequence | C | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 1 / [L11](../../test/npu_validation/benchmarks/auto_sync/prepared/elementwise_pipeline/input.pto#L11) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 2 / [L49](../../test/npu_validation/benchmarks/auto_sync/prepared/elementwise_pipeline/input.pto#L49) | loop | BR | regional-repetition [original]; 1/1 | finite-guarded-occurrences [small-count-expanded]; 0/0 |
| 3 / [L49](../../test/npu_validation/benchmarks/auto_sync/prepared/elementwise_pipeline/input.pto#L49) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 4 / [L50](../../test/npu_validation/benchmarks/auto_sync/prepared/elementwise_pipeline/input.pto#L50) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 5 / [L54](../../test/npu_validation/benchmarks/auto_sync/prepared/elementwise_pipeline/input.pto#L54) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 6 / [L70](../../test/npu_validation/benchmarks/auto_sync/prepared/elementwise_pipeline/input.pto#L70) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 7 / [L54](../../test/npu_validation/benchmarks/auto_sync/prepared/elementwise_pipeline/input.pto#L54) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 8 / [L54](../../test/npu_validation/benchmarks/auto_sync/prepared/elementwise_pipeline/input.pto#L54) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 9 / [L55](../../test/npu_validation/benchmarks/auto_sync/prepared/elementwise_pipeline/input.pto#L55) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |

### gated_delta_rule

| Node / source line | Kind | Expected | MayNotAlias | MayAlias |
| --- | --- | --- | --- | --- |
| 0 / [L10](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L10) | sequence | C | unmet | unmet |
| 1 / [L11](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L11) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 2 / [L96](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L96) | loop | U | unmet | unmet |
| 3 / [L96](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L96) | sequence | C | unmet | unmet |
| 4 / [L97](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L97) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 5 / [L98](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L98) | conditional | C | unmet | unmet |
| 6 / [L98](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L98) | sequence | C | unmet | unmet |
| 7 / [L98](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L98) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 8 / [L99](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L99) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 9 / [L110](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L110) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 10 / [L117](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L117) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 11 / [L144](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L144) | loop | U | bounded-lifetime [original]; 0/0 | unmet |
| 12 / [L261](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L261) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 13 / [L110](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L110) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 14 / [L110](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L110) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 15 / [L144](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L144) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 16 / [L111](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L111) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 17 / [L145](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L145) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 18 / [L157](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L157) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 19 / [L167](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L167) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 20 / [L196](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L196) | conditional | G | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 21 / [L218](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L218) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 22 / [L157](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L157) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 23 / [L157](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L157) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 24 / [L196](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L196) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 25 / [L196](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L196) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 26 / [L158](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L158) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 27 / [L160](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L160) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 28 / [L197](../../test/npu_validation/benchmarks/auto_sync/prepared/gated_delta_rule/input.pto#L197) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |

### lossless_block_cast

| Node / source line | Kind | Expected | MayNotAlias | MayAlias |
| --- | --- | --- | --- | --- |
| 0 / [L10](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L10) | sequence | C | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 1 / [L11](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L11) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 2 / [L46](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L46) | conditional | C | regional-conditional [original]; 1/1 | regional-conditional [original]; 1/1 |
| 3 / [L46](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L46) | sequence | C | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 4 / [L46](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L46) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 5 / [L47](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L47) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 6 / [L55](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L55) | loop | A | regional-repetition [original]; 1/1 | regional-repetition [original]; 1/1 |
| 7 / [L58](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L58) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 8 / [L70](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L70) | loop | A | regional-repetition [original]; 1/1 | regional-repetition [original]; 1/1 |
| 9 / [L81](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L81) | loop | A | regional-repetition [original]; 1/1 | regional-repetition [original]; 1/1 |
| 10 / [L87](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L87) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 11 / [L88](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L88) | loop | A | regional-repetition [original]; 1/1 | regional-repetition [original]; 1/1 |
| 12 / [L99](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L99) | loop | A | regional-repetition [original]; 1/1 | regional-repetition [original]; 1/1 |
| 13 / [L111](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L111) | loop | A | regional-repetition [original]; 1/1 | regional-repetition [original]; 1/1 |
| 14 / [L124](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L124) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 15 / [L55](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L55) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 16 / [L70](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L70) | sequence | C | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 17 / [L81](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L81) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 18 / [L88](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L88) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 19 / [L99](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L99) | sequence | C | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 20 / [L111](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L111) | sequence | C | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 21 / [L56](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L56) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 22 / [L71](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L71) | loop | A | regional-repetition [original]; 1/1 | regional-repetition [original]; 1/1 |
| 23 / [L82](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L82) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 24 / [L89](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L89) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 25 / [L100](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L100) | loop | A | regional-repetition [original]; 1/1 | regional-repetition [original]; 1/1 |
| 26 / [L112](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L112) | loop | A | regional-repetition [original]; 1/1 | regional-repetition [original]; 1/1 |
| 27 / [L71](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L71) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 28 / [L100](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L100) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 29 / [L112](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L112) | sequence | G | regional-sequence [original]; 1/1 | regional-sequence [original]; 1/1 |
| 30 / [L72](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L72) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 31 / [L101](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L101) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |
| 32 / [L113](../../test/npu_validation/benchmarks/auto_sync/prepared/lossless_block_cast/input.pto#L113) | explicit-run | F | finite-occurrences [original]; 1/1 | finite-occurrences [original]; 1/1 |

## Validation limits

Eleven focused groups pass, including 4,771 expression points, 3,296
integer projection/feasibility/witness points, physical-byte selectors,
parameter permutations, fixed-coordinate mappings and cache/ownership
checks. PyPTO GEMM retains successful logical preparation, six-ID
allocation and byte-identical C++ generation versus the optimized baseline.
These are compiler checks; no device execution was performed.

The broader phase-owned/rotating suites previously reached an external
180-second watchdog on both baseline and candidate. The broader expression
fixture has the same numerical child-reuse assertion failure on both.
Those checks are not reported as passing. This campaign covers the eleven
prepared kernels, not the separately pinned PTO/PyPTO/pypto-lib corpus.

Artifacts: `.local/section8-refactor/regional-gaps/m5/` (paired campaign),
`m4/` (focused validation, ownership, projection and C++ evidence).
