# Device adapters for the four previously blocked fixtures

These adapters supply the missing launch context for the extracted PTO kernels.
They have **not been compiled with the device toolkit or run on a device here**;
this workstation has neither CANN nor an NPU. The NumPy reference tests pass.
The pinned library's CPU FIFO implementation does not support the same public
GlobalTensor overloads, so CPU compilation cannot validate this mixed launcher.
The next device task must compile, validate and repair adapters before timing.

| Fixture | Previously missing | Supplied here |
| --- | --- | --- |
| Conv2D | Host compilation parsed device-only Tile prototypes; helper definitions were not linked | Guard around both helper header and generated source, shared translation unit, exact strided buffers and convolution golden |
| FlashAttention | Matching vector peer and complete FIFO launch | Fixed 8-row vector stripe peer using pinned softmax/GU macros; one cube plus two vectors, FFTS address, three 8-slot rings, online attention golden |
| GDN | Coordinated cube/two-vector launch and workspaces | Mixed entry with `get_subblockid()` stripes, disjoint workspaces, runtime FFTS address, final credit drain, WY golden |
| KDA | Coordinated cube/two-vector launch and workspaces | Same mixed entry, with the extraction's existing final credit drain and per-channel gated WY golden |

`launch.cpp` includes the compiler output unchanged. It does not repair local
InsertSync decisions or add local synchronization to generated functions.
FlashAttention's vector peer is benchmark support, fixed identically across
all arms; it is not a newly hand-tuned performance baseline. Its conservative
scratch-reuse barrier is part of whole-group wall time and must be counted
separately from the generated cube's synchronization.

Two concrete protocol details matter:

- The FlashAttention source pair and its generator now use
  `pto.tfree(%pentry, %ppipe : !pto.tensor_view<16x16xf16>, !pto.pipe)`.
  In pto-isa `1216c558`, `npu/a2a3/TFree.hpp` implements entry-less
  `TFREE_IMPL(Pipe&)` as a no-op; the GlobalTensor overload returns the free
  credit. Without this repair the vector P producer cannot receive all of
  its credits (including destructor drains). Both manual and automatic PTO
  were changed identically, and the source hashes were updated. Historical
  source and results at `9e061dcf` remain a separate population.
- GDN consumes each prior chunk's workspace-free credits in its loop but
  leaves the final pair to the caller. The adapter waits for flags 3 and 4
  on **both vector stripes** when `count > 0`. KDA already waits for its final
  flags 12 and 13, so the adapter does not duplicate them. These are fixed
  cross-core protocol waits, not local set/wait pairs.

`reference-manifest.json` pins the three recovered FA headers, copied verbatim
with their licenses. Existing `reference_sources.json` and
`kernel-pairs-manifest.json` pin the original reference sources and helpers.

Build one generated arm after sourcing the device toolkit:

```sh
cmake -S "$PERF/device" -B "$OUT/build/conv2d_interior/mmad_on" \
  -DCMAKE_CXX_COMPILER=bisheng \
  -DPTO_ISA_ROOT="$PTO_ISA_ROOT" \
  -DBENCH_CASE=conv2d_interior \
  -DKERNEL_SRC="$OUT/generated/conv2d_interior/mmad_on/output.cpp"
cmake --build "$OUT/build/conv2d_interior/mmad_on" --parallel 2
python "$PERF/device/run_device.py" \
  --library "$OUT/build/conv2d_interior/mmad_on/libcase_runner.so" \
  --case conv2d_interior --count 3 --device 0 --arm mmad_on \
  --correctness-launches 50 --samples 50 --batch 1 \
  --output "$OUT/results/conv2d_interior/mmad_on/block0.json"
```

Use the same commands with `flash_attention_cube`, `gdn_wy`, or `kda_wy`.
The build chooses a cube-only target for Conv2D and a mixed target for the
three cooperating kernels. `PTO_ISA_ROOT` must be the pinned pto-isa checkout.
The Python runner requires NumPy and the runtime libraries on `LD_LIBRARY_PATH`.
It uses ACL directly; `torch_npu` is not required.

Buffer order is the `LaunchCase` ABI (unused pointers are null):

| Fixture | Buffers 0 onward | Count domain |
| --- | --- | --- |
| Conv2D | fmap, weights, output | panels 1..32; zero stores an uninitialized accumulator and is outside this extraction's contract |
| FA | Q, K, V, QK ring, P ring, PV ring, output | tiles 0..16, two vector stripes of eight rows each |
| GDN/KDA | K, V, beta, gate, A, workspace2, workspace1, U, W | chunks 0..16; optional final 64-row half chunk |

`goldens.py` contains exact physical strides, extents, output footprints and
workspace rounding. The full allocated outputs are compared, including
untouched sentinel regions. Every allocation also has 256-byte guards on
both ends. Outputs are poisoned before **every** correctness launch; queues
and workspaces persist across launches to expose stale-credit/reuse bugs.
Mixed peers are launched together once; the host never fabricates queue
publications or FFTS credits.

Run the local reference tests serially:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s "$PERF/device" -p 'test_*.py'
```

`host.cpp` records `steady_clock` from immediately before submission until
the stream synchronization returns. `--batch 1` is synchronized wall time per
launch. `--batch N` with N > 1 is separately labelled throughput per launch;
it amortizes submission/synchronization costs and must not be merged with
single-launch samples. Python dispatch, copies, checks and reference computation
are outside both timed intervals. The runner emits raw samples, not speedup
claims; a campaign controller must balance arms and bootstrap paired blocks.
