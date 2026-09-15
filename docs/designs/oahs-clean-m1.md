# OAHS clean port — milestone 1: baseline and evidence

## Decision and scope

Start from production `hw-native-sys/PTOAS` **master**, pinned at
`ad63e67a79d47e35a76692dd534df13bd4bd6b23` (15 September 2026).
The experimental branch at `c73c04fb3b7a76d10ab55b48d4dcbe0270e36da2` is
reference material, not the base and not a sequence to cherry-pick.
The proposed new branch name is `codex/oahs-clean-m1`.

This patch adds evidence infrastructure only. It does not change any compiler
source, pass registration, semantic admission, target assumption, emitted IR,
optimization setting, or default. In particular it does not add DemandSync,
composition, structured, or any other public planner mode. The final pass name
is deliberately not decided by this milestone.

The next milestones are: complete shared semantic import without changing
synchronization; one structurally compositional constructor; persistent
lifetimes in that constructor; independently qualified target optimizations.
No performance refinement starts before coverage/correctness on the pinned
supported population. Device latency is measured from the baseline onward,
not deferred until static synchronization counts look attractive.

## Architectural contract to preserve

Demands come from physical accesses and storage generations. Publications
transfer the actual preceding producer prefix; acquisitions make that knowledge
available at dependent consumers. Readiness, reuse release, visibility/resource
ordering and event consumption are distinct. Existing causal paths may establish
consume-before-rearm, so a reverse exchange is not mandatory for every forward
handoff. Region exit does not reset token state. Ordinary and persistent
handoffs must satisfy one population of obligations with one global allocation
and one final emitted verification path.

Operation/lowering declarations must explicitly cover complete phases, data
and descriptor/resource effects, result readiness and private event usage.
Having a pipe interface plus some memory effects, with no match in an exception
name list, is not a completeness certificate. The recent shared classification
helper is useful evidence, but the exception-list/default-classification policy
must not become the new semantic foundation. A new operation using an existing
complete declared contract must require no planner change.

## Foundation requirements clarified during native intake

The first constructor includes direct producer-prefix accounting and reuse of
already-established completion across storage cells. These are foundational
semantics, not optional performance work. Region summaries export outstanding
accesses and token state without forcing a hardware drain at each boundary.

Every acknowledgment requires a specific consumption-before-rearm obligation.
Use an existing causal path when it supplies that obligation; otherwise choose
another key or a conservative reusable realization. Allocation may request bounded
reconstruction of the complete affected protocol, never an unchecked repair after
numbering. Final emitted reconstruction is read-only and checks original effects.

Authored resource protocols must be understood in the foundation even when their
optional optimization credit is disabled. Automatic synthesis (including verified
no-op cases) and supported authored-protocol handling have separate report totals;
their sum is supported handling, not a claim that every protocol was synthesized.

The gates are cumulative: baseline inventory and measurements; complete shared
semantic import; one general constructor; full supported-population coverage with
correctness checks; then measured refinements. Each implementation increment is a
native vertical slice. Performance measurement starts at baseline, while optional
performance development waits for the coverage/correctness gate. Reconcile all
model kernels, including fallback and refusal cases, and attribute measured costs
across the model instead of selecting only the historical GEMM.

## Population and evidence policy

Coverage means every valid input supported by production InsertSync in a pinned,
explicitly enumerated population under identical contracts. It is not a claim
about all possible PTO programs. Five cohorts remain visible: PTOAS regressions,
PyPTO operation suite, pypto-lib, generated model kernels, and device regressions.
Do not silently drop A5, failed frontend exports, invalid tests, authored protocols,
helpers, or macros. Classify intentional invalids explicitly, with the expected
stage and diagnostic, rather than deriving invalidity from compiler failure.

Capture an upstream enumeration log as well as compiler invocations. A frontend
failure before PTOAS is invoked must still have an expected inventory row with a
missing input. Captured compiler calls alone cannot prove full frontend coverage.
Repeated identical bytes under different launches/contracts remain separate cases.
The old seven/eight-kernel gate and 363 snapshots may be retained as historical
cohorts, not substituted for the live suite and model population.

The supplied plan reports a 43% prefill regression at `f530ebbda`, six attention
refusals, and 44 kernels in the OAHS-executed part of the model. Those are
**reported observations**, not measurements made by this patch. Their raw PTO,
commands, binaries, numerical checks, per-kernel timings and full-model logs are
required intake. The inventory template deliberately marks them missing. The 44
is a lower bound for recovering those observations, not the total workload size;
all refused/fallback kernels must be included too. Historical GEMM is an additional
reference, not a replacement for small GEMM or attention failures.

The current upstream pin itself documents a remaining independent numerical
qproj failure. Production compilation success therefore does not certify numerical
correctness. Preserve the reproducer and report correctness separately; do not
convert a known failure into an exclusion merely to achieve a passing headline.

## What the tools establish

`capture_compiler.py` forwards the real compiler's argv and exit code without
adding flags, changing input, or falling back. It captures the entire input
module, not extracted kernel bodies, and retains failed invocations. It serializes
its own compiler calls; it does not control unrelated processes. Its lock wait
is excluded from the recorded compiler time. Use `--context` to capture exact
frontend revisions, target/alias/ABI/planner assumptions and input classification.
Unknown invocation forms create incomplete records, not omissions.

`collect_records.py` converts every capture into a candidate inventory. It leaves
cohort collection unqualified until an independent frontend enumeration and its
failures have been reconciled. IDs do not collapse duplicate source hashes.

`evidence.py capture` stores exact raw/prepared bytes and preparation provenance.
It never strips synchronization, changes architecture names, adds alias metadata,
or guesses missing ABI restrictions. Identity preparation must be byte-identical;
external preparation needs its command and evidence. Missing cases remain in the
manifest. Paths and object hashes are checked again before replay.

`baseline` executes only the explicitly selected pinned production compiler. It
runs synchronization to PTO and then feeds that actual PTO to final C++ lowering
without synchronization a second time. A zero exit code with absent output is a
failure. Timeouts, crashes, launch failures and invalid-test diagnostic mismatches
are not expected refusals. Each run records argv, explicit environment, artifact
hashes, dependency versions and per-stage times. All commands run serially.
The source revision in the toolchain file is a build-provenance declaration;
hashing binaries detects drift, not whether someone honestly compiled that source.
Retain build logs/CMake configuration as additional pinned artifacts.

Static inspection is deliberately a lexical diagnostic, not a PTO parser or a
semantic completeness proof. It separates SETs, WAITs, named barriers and ALL
sites. It does not label an adjacent reverse pair as a release or redundancy,
does not label every ALL as terminal retirement, and does not infer dynamic
counts. Opcode inventory cannot substitute for milestone 2's semantic ledger.

`compare` consumes explicit future-pass handling and verifier receipts. Generated,
authored-preserved, verified-noop, fallback, refused and missing remain separate.
Fallback does not count as OAHS coverage. Missing candidate cases do not reduce
the baseline denominator. Unresolved baseline positives block qualification.
Receipts are attributed assertions with retained artifacts, not independent proofs
of compiler identity. They need the actual independent verifier in later milestones.

`device` imports externally measured evidence bound to the same captured inputs,
emitted PTO, run ID, executed binary and recorded build command. It requires raw
logs, per-kernel correctness, samples, invocation counts, a complete measured-model
inventory, and separately measured model latency. Missing or mismatched data cannot
become success. Weighted per-kernel time ranks attribution candidates; it is not
summed as a model critical path and does not by itself explain a slowdown.

## M1 acceptance

1. All five cohorts have independently reconciled expected inventories and source
   revisions; no missing/pre-compiler-failure row is silently discarded.
2. Pinned production compilation and final lowering results exist for every case;
   negative cases match their intended diagnostics. Compilation and numerical
   correctness remain separate results.
3. Reported small GEMM, all six attention failures and the whole prefill population
   are captured with raw evidence. Baseline kernel/model timing and numerical
   checks are recorded under exact device/toolchain/input contracts.
4. The new branch still produces unchanged production InsertSync output; this
   patch has no compiler code to change it.

A successful test of this harness is not acceptance of items 1–3. The bundled
unit tests use an explicitly synthetic subprocess compiler and synthetic timing
logs. Their purpose is to demonstrate faithful capture and refusal of incomplete
or misleading evidence. No native or device baseline is bundled as a fabricated
substitute for the missing external campaign.

## Source references inspected for this patch

- Production default branch and pin:
  `https://api.github.com/repos/hw-native-sys/PTOAS/branches/master`
  at `ad63e67a79d47e35a76692dd534df13bd4bd6b23`.
- Production `tools/ptoas/ptoas.cpp`: explicit `--enable-insert-sync` entry.
- Experimental commit `c73c04fb3b7a76d10ab55b48d4dcbe0270e36da2`:
  one-phase macro handling, TReduce accumulator effects, chained internal-prefix
  transfer, and `SyncMacroModel.cpp::hasUnmodeledImplicitSyncResource` /
  `getSyncOperationSemantics`. These are not ported by M1.
- User-supplied clean-port plan, 15 September 2026: baseline/evidence first;
  one pass with shared semantics; 100% pinned production-supported coverage;
  no fallback credit; storage-lifetime and event-causality invariants.
