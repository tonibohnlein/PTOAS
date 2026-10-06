# Benchmark port audit

These are explicit translations of selected pinned source configurations.
The comparison holds the port's payloads, planned buffers and control fixed
between its three synchronization variants. It does not assume that manually
assigned port addresses reproduce TileLang's memory planner. Native upstream
performance must remain a separate baseline.

## Translation contract

- `port_*.py` construct benchmark inputs; no compiler recognizer matches their
  kernel names. No production analysis/emission/allocation code changes here.
- Preserve the selected source computation, loops, branches, banking and
  authored flag protocol. Compile-time specialization is documented below.
- Use aligned physical UB allocations with typed aliases for shape/bit
  reinterpretation and explicit bank formulas. Actual port layouts are in IR;
  layouts need device qualification, including column reductions and short
  scalar/scale-factor transfers.
- `Port` tags synchronization lines. The expert includes them; the input omits
  them. `export_ports.py` checks their exact equality after that removal.
- An additional terminal ALL barrier closes the expert invocation. All inserted
  local synchronization, including helper barriers and this final completion,
  is removed from the autosync input. Both automatic passes see the same input.
- Pointer arguments are distinct GM allocations. Repeated accesses through one
  state argument share storage. Cross-task state accesses must not introduce
  cross-core conflicts: decode's cache read/write row unions must be disjoint
  across tasks; prefill's cache indices must be distinct.

## Per-source correspondence

| Source/factory | Port | Decisions and checks |
|---|---|---|
| `example_gemm_intrinsic_persistent.py:matmul` | `port_persistent.py` | Default full matrix sizes. `task=20*wave+cid`, domain64×4, group4. Thirteen waves; guard task<256. Same L1/L0 bank pipeline as the existing audited intrinsic fixture, release at kk==3, FIX→M flag0. Flags remain live across task re-entry. NZ L1 layout is the prior device compatibility fix. |
| `example_group_norm.py:group_norm_kernel_serial` | `port_groupnorm.py` | N4,G4,cpg16,S256,blockS64; no padding, FP32. Four-step accumulation, two-stage spatial input, two-stage normalization/output. Reduction results have compact column/row aliases. Only vid0 executes. This is a factory specialization, not the wrapper's whole configuration matrix. |
| `causal_conv1d_pto.py:_build_kernel` | `port_conv.py` | Width4, dim_num1, two batches, dim128, state_len3. Thus one sequence chunk per batch, first/last=true; buffer-based checks remain. Two banks of four rows, runtime seqlens/cache/initial predicates and state writeback. Positive sequence lengths divisible by4, total≤32. Source tail stores are kept; partial-prefetch/empty behavior is excluded, not repaired. |
| `causal_conv1d_decode.py:_build_decode_kernel_jit` | `port_conv.py` | Width4, one dimension chunk, batch8, dim2048, BF16, SiLU. Dynamic initial/read/write indices, three history rows, bias, state updates. IDs0..6 preserved. |
| `opt_gdn_chunk_cumsum.py:cumsum_ker` | `port_vector.py` | Exact source test B2,H16,L16384,C128,CC8. Each task handles eight chunks, V→S handoffs per chunk, scalar recurrence, then one V→MTE3 handoff and full-buffer store after the chunk loop. Source indentation audited explicitly. |
| `head_compute_mix_kernel.py:_mhc_head_compute_mix_fwd` | `port_vector.py` | Exact test_fwd reshape:8192×4→2048×16, block128, two64-row lanes, epsilon0.01. Base replication, broadcast, AXPY, sigmoid. Backward is separate and pending. |
| `elementwise_add_pipeline.py:vec_add_pipeline` | `port_vector.py` | M=N1024, block128×128, subM32, two stages/two lanes. Prologue, next-bank prefetch, current add/store, release and final drains retained. |
| `fused_sigmoid_gating_delta_rule_varlen.py:kernel` | `port_gating.py` | 24 sequences/cache slots, positive runtime lengths≤8, nk=nv1, dk=dv=block_v32. Work assignment simplifies to one task per core but keeps the two-iteration guarded work loop. Initial-state branch, scalar softplus branch, normalized Q/K, QKV prefetch and recurrent H retained. Source IDs including6 preserved. |
| `per_block_cast_lossless_kernel.py:get_per_block_cast_lossless_kernel` | `port_lossless.py` | Input128×256, inSF1×32, outSF64×64, block64×128, ordinary FP32 scale layout. Source generic vid0 path: exponent extraction/reduction, scale store, relative-scale expansion, BF16 conversion, output. Not the optimized max4 or packed/TMA variants. |

## PTO helper and A3 adaptations

The pinned PTO `common.h` and code generator are archived with hashes. On A3:

- `TSIGMOID(dst,src)` scales/exp/adds **in src** and reciprocates into dst, with
  V barriers between those steps. The ports preserve this mutation.
- `TSILU` saves the original src in a temporary, invokes sigmoid, then multiplies
  by the saved input. `MulAddDst` uses a multiply temporary then addition.
  Those helpers and their barriers are explicit payloads in both variants.
- A3 PTOAS rejects in-place reciprocal. The gating port uses distinct temporary
  output, V barrier, then copy-back. This changes the lowered operation sequence
  from the raw source spelling and is recorded, not hidden as an identical
  native backend instruction.
- A3 PTOAS rejects i32 TANDS. The lossless port forms an i32 mask255, aliases
  both input and mask as i16 words, applies TAND to distinct output, then copies
  back. The high half of each mask word is zero, so this is bit-for-bit an
  i32 AND255, not an AND255 applied indiscriminately to both halves. Extra
  barriers protect the introduced steps. Positive finite power-of-two scales
  are the selected input contract.

Authored IDs6 and scalar/vector handoffs are preserved for device qualification.
The presence of a flag in upstream is not proof of its legality or of progress.
No local device test has established any new port's manual protocol correctness.

## Evidence and remaining gates

The compilation results below predate the subsequent analysis changes.

- All18 newly generated IR files parse and verify with the recorded PTOAS tool.
- All9 expert/input pairs pass synchronization-only comparison.
- Generator rerun matches registered files and metadata exactly.
- Expert and existing InsertSync generate C++ for all9. Frontier rejects their
  analysis; no allocator conclusion follows.
- The earlier2 GEMMs still generate all6 variants.
- Full CANN compilation, launch ABI, independent numerical/state validation and
  runtime measurement remain device tasks. In particular, C++ emission alone
  does not check PTO-ISA template instantiation or vector-lane indexing.

No native-source timing should be mixed with these ported baselines until the
translation and scheduling differences have been measured and reported.
