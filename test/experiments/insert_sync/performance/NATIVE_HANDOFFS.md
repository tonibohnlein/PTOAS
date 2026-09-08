# Native guarded handoff regression

The integrated option is `--insert-sync-handoff-planning`, default off. Native
fact export landed in `75ed8c071`; native construction/realization landed in
`518572c3f`, each accepted by the architect, compiler and algorithms/performance
reviewers before the next stage began. This runner is the third stage.

The premise is to preserve the scheduled payload, discover its ordering
requirements independently of the selected synchronization, and use combined
completion plus first-demand boundaries to improve complete handoffs. Every
trial retains the original requirements and independently reconstructs actual
events, guarded participation and exit completion. The current native subset
constructs block-local cuts within supported loops/branches; it retains
unsupported plans conservatively. It is not arbitrary symbolic event lowering
or joint coloring of every stream.

## Run the nineteen frozen inputs

Use the Python interpreter matching the build's MLIR extension ABI. In this
checkout that is `.venv/bin/python`, with the runtime under
`../insertsync-builds/PTOAS-insertsync-r1/python`.

```sh
.venv/bin/python test/experiments/insert_sync/performance/run_native_handoffs.py \
  --python-root ../insertsync-builds/PTOAS-insertsync-r1/python \
  --output /disk/new-native-handoff-results
```

The runner validates all three original manifests and compiles their unchanged
automatic inputs: eleven earlier fixtures and eight Qwen additions. Both arms
use the same A3 compiler, GM argument-disjoint contract, staged repair, MMAD
and buffer-generation flags. Only the experimental option differs. Each arm
emits PTO and C++ separately, for 76 compiler invocations. Execution is serial
and disables MLIR threading. No build, original-runtime substitution, input
annotation or device invocation is performed.

`--case NAME` may be repeated for focused validation. Reports name the exact
selected population; a subset is never labeled a full corpus run. The existing
`run.py` also accepts `--handoff-planning` for its manual/combined/staged ladder,
and `../compare_native.py` accepts it for the unchanged TSV corpus workflow.

Each output directory retains original input hashes, compiler/runner/manifest
hashes, commands, logs, PTO/C++, static placements, keys by direction, native
fallback/status attributes and measured invocation durations. `results.json`
and `RESULTS.md` are checkpointed with an explicit running/completed status.
Failures stay in the population. Named barriers, `PIPE_ALL`, sets and waits are
never added into a synchronization score or assumed to form equal inventories.

The existing exact projection compares payload operations, SSA wiring, guards,
ABI, views and allocation expressions. Synchronization-only scalar control is
recorded separately and must remain unchanged for this experiment. Existing
finite scenarios additionally compare executed payload hashes and scalar
counts. The FlashAttention parser bridge retains the original and normalized
hashes and applies the same generic GM-pipe assembly normalization to both arms.

The independent prefix observer selects the specified function. It reports
unsupported helper/peer/resource or physical-operation semantics explicitly;
it never treats those observations as a proof. Both arms are queried, so an
unsupported seed does not hide a hard trial failure. In the supported domain,
token failures, changed payload and newly imposed later source prefixes fail
acceptance. Missing scenarios are recorded as not-run, including in the table.
This observer is finite execution evidence, not a universal asynchronous or
device correctness certificate.

The frozen online-softmax bound-16 gate requires the demonstrated improvement
to survive (at most 96 executed sets and waits, fewer than the seed's 188).
Q projection similarly retains at most 29 static sets/waits versus 39. These
are regression expectations in the runner, not compiler input or matching
criteria. Historical GEMM's existing 56-pair, zero-barrier plan remains a
preservation test: the stronger experimental seed proof currently declines it.

## Validation and remaining qualification

Before committing this third stage, selected production cases exercised both
output kinds, looping command reductions, unchanged GEMM, the FlashAttention
parser bridge and the GDN/KDA named peer entries. Their 60 peer-prefix scenarios
are explicitly unsupported; scalar payload replay and compilation pass
independently. Mutation tests cover changed contracts, missing scenarios,
ineffective fallback, asymmetric observer failures and additional blocking.

After the three reviewed commits, run the compiler regression selection,
accounting/reference tests, the full nineteen-input comparison and the existing
213-row native admission regression with the experimental flag. Preserve every
pre-existing failure in comparisons; a matched compilation failure is not
successful autosynchronization. The 9,754-row generated collection remains a
separate broader campaign, not the denominator of the 213-row regression.

Device correctness and wall time for these new outputs are still pending.
Command reductions and weaker observed prefixes do not establish a speedup.
Production leaves barriers fixed in this experiment, and a split can increase
commands to improve readiness. Event-scarcity fallback never broadens a cut or
adds a drain merely to allocate a new key.
