# Eight Qwen A3 additions: local InsertSync comparison

All eight unchanged kernels compile with both passes. QK and SV matmul lose
their exit PIPE_ALL; Q projection grows from 19 to 45 set/wait pairs while
retaining its barriers. The five vector kernels retain their inventories.
Payloads, SSA wiring, control, views, allocation expressions and ABIs match
between the two arms. Device correctness and wall time have not been measured.

Measured 2026-09-08. Original is the frozen `7e2ec3e29` compiler. Revised is
`862ab8111` plus the existing worktree experiment, with that separate experiment
**disabled**; no new compiler change or rebuild was needed for this campaign.
Native hashes are `0fa0a43290cb0a43f48755ded11fe9879323698b86ea8aa63cf127bb3844b7b3`
and `bc28708f481463e0b9046a609d12525b973ffcbd6181f75a0b53bcc182664833` respectively.
This records the actual worktree binary, not a clean-build claim for `862ab8111`.

## Static synchronization

Each set and wait column is a site inventory. Named barriers and PIPE_ALL are
separate mechanisms; none are combined into a score. These additions have no
hand-tuned reference. SV is the attention probability/value (PV) matmul.

| Kernel | Original set / wait | Revised set / wait | Named barriers, both arms | PIPE_ALL original → revised |
| --- | ---: | ---: | --- | ---: |
| RMSNorm | 40 / 40 | 40 / 40 | V=29, MTE3=3 | 1 → 1 |
| Softmax | 15 / 15 | 15 / 15 | V=14 | 1 → 1 |
| Online softmax | 12 / 12 | 12 / 12 | V=20 | 1 → 1 |
| SiLU | 3 / 3 | 3 / 3 | V=6 | 1 → 1 |
| RoPE / KV-cache update | 15 / 15 | 15 / 15 | V=12, MTE3=2 | 1 → 1 |
| QK matmul | 21 / 21 | 21 / 21 | M=2 | 1 → 0 |
| SV / PV matmul | 25 / 25 | 25 / 25 | M=2 | 1 → 0 |
| Q projection | 19 / 19 | 45 / 45 | M=6 | 1 → 1 |

## Specialization and fallback

| Kernel | Committed lifecycle channels | Native outcome and first relevant gate |
| --- | ---: | --- |
| RMSNorm | 0 | Structural import declines; `pto.texpands` is outside the qualified opcode adapter |
| Softmax | 0 | Structural import declines; first excluded operation is `pto.tfillpad` |
| Online softmax | 0 | Structural import declines; first excluded operation is `pto.tmax` |
| SiLU | 0 | Structural import declines; first excluded operation is `pto.tneg` |
| RoPE / KV-cache | 0 | Structural import declines; first excluded operation is `pto.tcolexpandmul` |
| QK matmul | 4 | Two L0 bundles, one reused K panel, one ACC lifecycle; two one-shot Q panels remain with general insertion |
| SV / PV matmul | 5 | Two L0 bundles, two reused L1 panels, one ACC lifecycle |
| Q projection | 10 | Four L1 slots and six separate L0 slots; ACC candidate declined |

For the five vector kernels the emitted effect-coverage status is **complete**.
Their failure is the optional structural importer's narrower adapter, not a
production InsertSync coverage failure. The native diagnostic says
`physical phase has no qualified frontier adapter`; the specific first opcodes
above are identified by comparing the actual pass-entry IR with `NativeGraph`
in [StorageFrontierAnalysis.cpp](../../../../lib/PTO/Transforms/InsertSync/StorageFrontierAnalysis.cpp).
Removing one gate does not establish that the rest of a function is supported.

All three accepted cube plans commit on the first attempt, with combined event
checking proved. They supply 144, 173 and 156 access-pair visits for QK, SV and
Q projection, respectively. Native diagnostics on the captured pass-entry IR
reproduce the CLI's exact synchronization positions, keys and counts.

All three cube kernels report **zero eligible MMAD targets**. Their descriptors
retain `valid=?x?` in the tile type despite constant valid-row/column operands;
the current `staticDescriptor()` qualification requires the type's valid shape
to equal its physical shape. They also use BF16 operands, outside the current
F16-only rule. Distinct ACC allocation handles at the same address are an
additional identity limitation. These explicit checks are in
[MmadChainAnalysis.cpp](../../../../lib/PTO/Transforms/InsertSync/MmadChainAnalysis.cpp).
The retained M barriers therefore are not evidence that the qualified MMAD rule
was applied and proved ineffective. Extending a target rule needs separate
qualification; this benchmark does not change it.

Q projection's ACC candidate reports `read can reach an uninitialized buffer
generation`; its final cleanup reports `exit completion unproved: phase 22 at
node 453`. Those are the constructor/checker's conservative reasons. They do
not establish an execution defect in this kernel. No ACC lifecycle is selected,
and the exit drain remains.

## Placement differences

RMSNorm, softmax, online softmax and SiLU retain the same action inventory at
every non-sync boundary, including event IDs. The emitted order of adjacent
synchronization actions at some boundaries changes. No action crosses a
payload operation. RoPE/KV-cache has identical placements and byte-identical
generated C++; the other four have textual C++ differences. Device-binary
identity has not been checked.

QK and SV change priming/drain placement and event-key assignments, retain the
same directed set/wait inventory, and remove the exit PIPE_ALL. SV has no other
changed payload/control cut after ignoring scalar setup and key assignment.
QK has one additional substantive change: the initial MTE2→MTE1 publication
moves from after the first Q preload to after the second Q preload. The wait
still precedes the loop. The revised signal therefore includes both loads in
its source prefix. The first Q extract uses only the first panel; this broader
readiness boundary deserves an overlap check, even though a drain disappeared.
The static comparison does not establish net runtime benefit.

Q projection constructs separate readiness/release streams for four LEFT and
two RIGHT slots instead of bundled operands. That increases startup, body and
cleanup actions. Its directed **set-site** inventory explains all 26 additions:

| Direction | Original sets | Revised sets | Difference |
| --- | ---: | ---: | ---: |
| MTE1 → MTE2 | 6 | 10 | +4 |
| MTE2 → MTE1 | 3 | 4 | +1 |
| M → MTE1 | 3 | 18 | +15 |
| MTE1 → M | 6 | 12 | +6 |
| M → FIX | 1 | 1 | 0 |

The separate releases and readiness publications are visible after the relevant
extract/matrix operations; the scopes are primed and drained independently.
This is a committed increase in synchronization commands, not a rejected
candidate. It is a useful regression/control for future compatible-handoff
grouping, and should be timed rather than assumed faster because synthesis ran.

## Identity checks and reproducibility

The [manifest](qwen-additions-manifest.json) pins the eight existing inputs and
their numerical-golden/support files. The input bytes match committed
`862ab8111`, contain no local synchronization, and are passed unchanged to both
compilers. No annotations, addresses, shapes or payloads were rewritten.

Both arms use A3 and level3. Original uses `--enable-insert-sync` with its legacy
distinct-argument behavior. Revised additionally uses:

```text
--insert-sync-gm-alias=assume-disjoint-arguments
--insert-sync-effect-coverage=report
--insert-sync-defer-same-pipe
--insert-sync-mmad-chains
--insert-sync-buffer-generations
```

For all eight cases, comparison verifies identical pass-entry IR and identical
emitted non-sync operation order, attributes, types, SSA operands, region/block
structure, allocation-address expressions, view geometry and function ABI.
The projection removes only local synchronization, compiler status attributes,
and the explicitly recorded GM-contract attribute. It does not normalize away
payload differences or replace compilation failures with zero counts. Four
mutation checks confirm that synchronization/SSA renaming is ignored while
changed payload operands and allocation addresses are detected.

The campaign completed **32 PTO/C++ emissions and eight native diagnostic
runs**, with zero failures and 16.0 seconds summed process time. All compiler
fingerprints remained unchanged. This is local IR/C++ compilation and contract
comparison; numerical execution, runtime allocation validation and timing are
still device work. Static counts are not multiplied by loop trips here.

Run the complete comparison serially using the existing builds:

```bash
.venv/bin/python test/experiments/insert_sync/performance/run_qwen_additions.py \
  --original-python-root /path/to/frozen-original/python \
  --revised-python-root /path/to/revised/python \
  --pto-test-opt /path/to/revised/pto-test-opt \
  --output /path/to/new/disk-backed/campaign
```

`--analyze-existing` refreshes projections/tables from retained outputs without
recompilation. Full commands, before/after PTO, C++, diagnostics, exact
placement diffs, payload-cut changes and identity projections are retained at:

```text
/home/toni/work/pypto3_sync_more/insertsync-builds/campaign/qwen-additions/comparison-v2
```

The earlier `pilot/` and `comparison/` directories are setup evidence; the first
comparison stopped when its dump reader included diagnostic text after the
function. `comparison-v2` is the complete campaign. The corrected reader has a
regression check. [Compact machine-readable results](QWEN_ADDITIONS_RESULTS.json)
retain hashes, all commands, native decisions and comparison verdicts.
