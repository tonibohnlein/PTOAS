# Historical original versus 60db InsertSync counts

This report freezes the pre-follow-up `60db1026a` checkpoint. Follow-up v2
admits all eleven fixtures in default report mode; see the current
[hand-tuned/original/follow-up comparison](MANUAL_VS_FOLLOWUP_V2.md).

Measured on 2026-09-07 using the same automatic inputs on both native builds:

| Build | Source | Native SHA-256 |
| --- | --- | --- |
| Original fork main | `7e2ec3e29` | `0fa0a43290cb0a43f48755ded11fe9879323698b86ea8aa63cf127bb3844b7b3` |
| Revised pre-follow-up checkpoint | `60db1026a` | `879c0705280ced29a0dd43617d3fab6e7b5b0d0419051921ef7cabc8bf2c3ab3` |

A2 and A3 counts agree. Revised combined and staged modes also agree.
The original build emits all eleven cases. Each revised mode emits six and
rejects five with missing effect contracts. These are results of that
checkpoint's auxiliary contract gate, before follow-up v2 restored report mode.

All input local set/wait/barrier counts are zero. Entries below are static
**sets / waits / barriers = total**, counting each SetFlag and WaitFlag as
one operation. Fixed cross-core signals, FFTS setup and FIFO operations are
excluded. GDN/KDA static counts sum the vector and cube function bodies;
they do not multiply the vector body by its two runtime peers.

| Kernel pair | Original | Revised combined and staged |
| --- | ---: | ---: |
| Historical GEMM | 44 / 44 / 22 = **110** | 44 / 44 / 25 = **113** |
| TopK 128 | 15 / 15 / 17 = **47** | 15 / 15 / 17 = **47** |
| Conv2D interior | 0 / 0 / 9 = **9** | Rejected: helper contracts |
| FlashAttention cube | 27 / 27 / 1 = **55** | Rejected: queue effects |
| Triangular inverse 16 | 277 / 277 / 1 = **555** | Rejected: scalar-dependent TAXPY effects |
| GDN WY | 32 / 32 / 16 = **80** | Rejected: FFTS setup effects |
| KDA WY | 35 / 35 / 14 = **84** | Rejected: FFTS setup effects |

The revised gate stops before it produces an accepted output for those five
cases. Their inserted count is **unavailable, not zero**. The original build
emitting them is not evidence that it modeled their complete effects. In
particular, Conv2D's nine barriers do not certify the opaque convolution
helpers' synchronization.

The four small controls are unchanged:

| Control | Original and revised |
| --- | ---: |
| One buffer | 6 / 6 / 1 = **13** |
| Two buffers | 12 / 12 / 3 = **27** |
| Three buffers | 18 / 18 / 4 = **40** |
| Four-use producer | 24 / 24 / 3 = **51** |

## Executed action counts

These are bounded scalar replays of the emitted IR, not device timings.
Static operations inside loops can execute many times. All eligible scenarios
are retained in `counts.json`; representative instances are below. Counts
again exclude the fixed cross-core protocol.

| Scenario / peer | Original sets / waits / barriers = total | Revised combined and staged |
| --- | ---: | ---: |
| GEMM 4096³, 24 cores, core 0 | 3437 / 3437 / 1607 = **8481** | 3437 / 3437 / 1783 = **8657** |
| TopK, 16 groups | 180 / 180 / 257 = **617** | 180 / 180 / 257 = **617** |
| Triangular inverse, 16 matrices | 4402 / 4402 / 1 = **8805** | Rejected |
| GDN, 16 full chunks, cube | 180 / 180 / 17 = **377** | Rejected |
| GDN, 16 full chunks, vector stripe 0 | 148 / 148 / 177 = **473** | Rejected |
| KDA, 16 full chunks, cube | 214 / 214 / 17 = **445** | Rejected |
| KDA, 16 full chunks, vector stripe 0 | 179 / 179 / 129 = **487** | Rejected |

The revision adds three static GEMM barriers, which become 176 extra barrier
actions in the named large GEMM instance. It does not reduce counts on any
jointly admitted case in this population. Payload trace hashes match between
original and revised outputs for every jointly admitted replay scenario.
This does not prove numerical correctness, event lifetime or asynchronous
completion; no device run or performance claim is made.

## Reproduce and inspect

```bash
python3 test/experiments/insert_sync/performance/compare_revisions.py \
  --original-python-root /path/to/original/python \
  --revised-python-root /path/to/revised/python \
  --original-source 7e2ec3e29 \
  --revised-source 60db1026a \
  --allow-rejections \
  --output /path/to/disk-backed/results/original-vs-revised-sync-counts
```

The default runs A2 and A3, one compiler invocation at a time. The original
build lacks the new GM-contract/audit options, so those flags are passed only
to the revised build. Inputs promise disjoint GM arguments throughout.

Local results are in
`/home/toni/work/pypto3_sync_more/insertsync-builds/campaign/original-vs-revised-sync-counts/`:

- `counts.md` and `counts.json`: all static counts, replay scenarios, statuses
  and native/CLI fingerprints.
- `inputs/`, `inputs.tsv` and both JSON manifests: frozen population and hashes.
- `{a2,a3}/{main,combined,staged}/`: compiler commands, logs and emitted PTO/C++.
- `{a2,a3}/diffs/`: original-to-revised emitted-IR differences.

All native hashes remained unchanged during measurement. No measurement
failures occurred. Successful compilation here means PTO/C++ emission, not
compilation of that C++ with a device toolchain.
