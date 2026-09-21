# Open OAHS device experiments: consolidated campaign

## Read this first

The user confirms that **the planned overnight sweep did not run**. Start or
resume the jobs below; do not report a prepared package as an executed campaign.
This master task supersedes scheduling, compiler pins and scope in the enclosed
older task documents. Their source contracts and correctness requirements still
apply. No access to the originating workstation is required.

Use the **current compiler snapshot in `sources/compiler.tar.gz`**, identified
by its tree manifest and archive hash. It is based on `6b1b32f46` and includes the
exact-proposal-order and replay-comparison fixes. HEAD alone is not
its identity. The local 88-module sweep passed construction/reconstruction with
all plans byte-identical; this is host evidence, not device qualification.

Work autonomously for up to **8 hours**, reserving the last 20 minutes for the
report and verified archive. Finish sooner if all eligible work is complete or
precisely blocked. Attempt every workstream within its cap; do not promise all
three substantial reference transcriptions will finish in one campaign.

## Package map and setup

1. Verify the outer checksum, extract, and run
   `python3 tools/verify_package.py .`.
2. Extract `sources/compiler.tar.gz` into a new compiler directory (archive has
   no enclosing root). Verify with
   `python3 tools/verify_package.py . --source-dir COMPILER`.
3. Build `pto-test-opt`, `pto-oahs-selected-test`, `pto-oahs-native-test` and
   `PTOASPythonPackage` using the established compatible LLVM/MLIR environment.
   Run portable suites and native diagnostics once. Share the immutable build.
   LLVM/MLIR, CANN, drivers and Python dependencies are environment prerequisites;
   record their versions. Do not silently change target profiles or the compiler.
4. Extract `bundles/corpus.tar.gz` and `bundles/retained.tar.gz` separately. Each
   has its own root, manifest and checksum verifier. Verify both. They provide
   prepared inputs, pinned plans, framework/runtime sources and device harnesses.
   **Do not build their archived candidate/compiler sources for current arms.**
   The retained experiment's historical baseline source is the one exception.
5. Extract `sources/catlass.tar.gz` and `sources/pto-isa.tar.gz` separately.
   The exact pins are in `manifest.json`; reference task details are in
   `references/`. Keep their original hardware contracts. The old `495fb9cbd`
   compiler pin in those documents is superseded by the root compiler snapshot.
6. The corpus leaf's `tools/prepare_plans.py` and `tools/build_device_arm.py`
   provide paired plan preparation and the corrected real `-v` cc1/`-O2` check.
   Supply the current compiler explicitly. Never synchronize selected PTO again
   when lowering. Runtime/pypto/pypto-lib archives and reusable MAT harnesses are
   in the corpus leaf. Record archive hashes and actual loaded-library hashes.

A source snapshot is frozen for this measurement task. Compiler fixes are out
of scope; harness/ABI adaptation and faithful source transcription are allowed.
Keep unsupported-contract reproducers and continue other work.

## Work queue and priorities

`campaign.json` records the workstreams, statuses and caps. Queue independent
ready cases immediately; host adoption jobs must not hold idle devices.

| ID | Priority | Experiment | Arms / useful result |
| --- | --- | --- | --- |
| D1 | 1 | Broad pypto-lib/PyPTO kernels and runnable model entries | Current default handoff vs current existing; breadth and new misses |
| D2 | 1 | Retained-input microkernels `retained_2_1`, `retained_4_2` | Frozen pre-cohort baseline vs current default; first device test of retained children |
| D3 | 2 | CATLASS example 25, retained A across output tiles | Original manual, matched manual PTO, current handoff, current existing |
| D4 | 2 | CATLASS example 06, cross-output-tile preload | Same four arms; preserve actual next-tile loads and scheduler |
| D5 | 2 | PTO-ISA manual delayed-QK/PV attention | Resume matched UF=0 transcription; keep original UF=1 reference separate |
| D6 | 2, gated | Captured Qwen partial/single-block coupled attention | Current default vs existing, only under authentic AIC/AIV runtime |
| D7 | 3 | One Shenggan no-motion event-overhead control | Current default vs `--no-frontier-motion`, same payload/ordering |

### D1: broad corpus, including post-RMSNorm

Follow the corpus leaf's `DEVICE_TASK.md` with the root compiler override.
Its inventory has 96 modules: 88 A3 candidates and eight A5 research inputs.
Attempt host construction for every captured pypto-lib A3 row; deduplicate aliases
for device timing and aim for at least 20 distinct qualified kernel families.
This is a coverage target, not permission to relabel duplicate rows as families.
A5 sources remain inventory-only under the current A3 contract.

Prioritize previously unmeasured supported families: partial merge seed/update,
attention finalization, post-RMSNorm rows22/23, final RMSNorm, reciprocal, SiLU,
RoPE/cache, embedding, last-token extraction, reductions and top-k. Then add
original runnable A3 model/layer entry points. Model timings require every
module mapped to its selected arm; no silent fallback or old cached binaries.
Report full-model invocation time separately from component/kernel latency.

Use down_proj and one Shenggan configuration as compact controls. Completed
projection and vector controls may populate the new unified two-arm table with
one representative case each; do not repeat the old four-arm/three-seed timing
matrices. Broad coverage takes precedence over repeated shape variants.

Cap first-time ABI/harness investigation at 20 minutes per family. The corpus
leaf supplies qualified MAT harness sources, not a universal ABI. Cache each
independent golden by shape/seed and share it across arms. Do not recompute slow
scalar FP64 LM-head references per arm. Deduplicate byte-identical binary/launch
cases and label them structural aliases rather than independently timed wins.

### D2: retained inputs, ready for qualification

Follow the retained leaf's task, except **candidate = root current compiler**.
Baseline remains that leaf's frozen pre-multi-input source. Its two candidate
PTO pins were regenerated with the root compiler and match exactly (evidence
included). Regenerate/lower and verify the pinned PTO/C++ before device builds.
Use the bundled wrappers and harness; do not infer ABI or shape from filenames.

Two seeds (7,23), four queued independent slots per correctness invocation, exact
FP32 integer-derived reference, full outputs/guards/input preservation. Both arms
must pass before timing. Expected local ordering: 12/48 relations removed,
zero added; dynamic pairs 10→14 and 20→26. These are synthetic mechanism tests,
not model performance. Cap: 45 minutes after shared environment setup.

The harness's two fresh timing processes each run 10 warmups and 10 samples:
20 samples and 20 warmups per arm total. This documented exception avoids editing
an unqualified harness merely to save ten warmups. Count all launches explicitly.

### D3 and D4: hand-tuned projection references

Use `references/retained_a.md`, `references/preload.md` and `references/COMMON.md`.
Start from the full source caller, not a block helper or the already-tested
Shenggan GEMM. Preserve full-load-A retention versus reload, B buffering, K
shuffling, output dtype/layout, padding, prefetch and original unit-flag modes.

Use at most two main timing shapes plus a reload/no-preload correctness control
per family initially. Produce the same prepared payload/control/allocation for
matched manual, handoff and existing. Archive the exact synchronization-removal
manifest; preserve all unsupported fixed protocols. Original manual vs matched
manual exposes transcription differences.

Allow up to 90 minutes of host adoption per reference. At that cap return the
smallest unsupported-contract/translation example, any qualified manual baseline,
and remaining work. Do not spend the whole campaign inventing a substitute
kernel. Source-supported explicit-event variants are separately labelled families;
never silently disable unit flags or change output precision to force admission.

### D5: resume manual attention from the reported milestone

Reported prior work (not independently reproduced by this bundle): UF=1 and UF=0
passed both shapes, three seeds and repeated launches. The gap was the missing
PTO representation/emission of `EN_UNIT_FLAG`, not a demonstrated kernel error.
The matched comparison therefore uses the source-supported **UF=0 explicit-event
family**, with UF=1 retained as a distinct performance reference.

Locate the earlier agent's source diffs, input manifests and job evidence on the
remote machine. Verify them before reuse. If unavailable, record that loss and
recreate one bounded qualification per shape before proceeding; do not claim the
old numerical results as new campaign runs. Earlier source adaptations were:

- Correct the kernel-launch closing `> > >` spelling in `Pto_prefetch.hpp`.
- Omit unsupported `--cce-pto-enable`; do not replace it with an AUTO-mode flag.
- Make `UF_ENABLE` overridable with its original default unchanged.
- Add explicit data seed selection and strict input size/hash/read-error gates.

Smoke: S0=128, S1=1024, TILE_S1=128. Pipeline: S0=2048, S1=2048,
TILE_S1=512. Preserve runTFA parameters, QK_PRELOAD=4, FIFO depths, consumption
cadence, prologue/steady-state/epilogue and coupled core launch. Never time AIC
and AIV independently as attention. Each built case must consume its own exact
data manifest; missing input files must fail, not compare zero buffers.

Next deliverable: matched-manual PTO, then current handoff/current existing from
that same payload with only admitted local synchronization removed. Keep semantic
row, physical bank use and queue generation distinct. Do not patch the compiler
or invent UnitFlag semantics. Cap further transcription investigation at 90
minutes, then package partial progress and an exact blocker. Use the source task's
independent numerical gate before timing; UF=1 timing is a separate reference,
not an unchanged-original comparison for the UF=0 automatic arms.

### D6: authentic Qwen attention, conditional on its runtime

This is a D1 subset, not another duplicate sweep. Rows44–49 need their original
coupled AIC/AIV queue/workspace/launch contract. Spend at most 30 minutes locating
and qualifying that runtime in the pinned model tree or verified prior artifacts.
If available, test one partial and one single-block case, including a meaningful
participation boundary. If unavailable, mark harness-blocked and retain host
construction evidence. Do not borrow the unrelated manual-attention harness.
No FIFO-only ablation or half-kernel latency claim is requested.

### D7: small no-motion control after breadth

Prior no-motion GEMM work was **host-only**, not a missing reported timing result.
This optional low-priority task measures the cost of extra commands with equal
checked payload ordering. Reuse the D1 Shenggan harness/input and current compiler;
compare default against `pto-oahs-selected-test --construct INPUT --no-frontier-motion`.
Use the committed ordinary-prefix trace checker and verify the expected default
200/394/782 versus no-motion 330/652/1296 pairs for one/two/four output tiles,
zero added/removed checked payload relations and unchanged genuine barriers.
If those premises fail, stop this comparison and report the discrepancy.

Time only one representative already-qualified configuration, 20 invocations per
arm. This is a separate two-arm control, not a mode added to all corpus cases.
Reuse the default timing only if device/binary/launch/input/timing scope match;
otherwise run the paired block anew. Cap: 30 minutes. No broad profile campaign.

## Correctness and timing rules for all lanes

- Correctness gates precede timing for every arm and the configuration actually
  timed. Existing is a comparator, not a golden oracle. Quarantine a failed arm;
  no ratio involving it. Do not patch a comparator during this campaign.
- For new corpus/microkernel cases use two deterministic nontrivial seeds and one
  queued four-slot repeat-entry check where meaningful. Reference-adoption cases
  retain their three-seed qualification. Reuse verified prior evidence only for
  exact source/binary/input/launch identities, with explicit provenance.
- Check full outputs, finite values, guards and input/state preservation as
  appropriate; enforce independent established tolerances. Reset mutable state
  outside timing. Verify file sizes/hashes and all read errors before running.
- Normally **10 warmups + 20 measured invocations per arm total**, two rotated
  rounds of ten. No hidden 100/180-sample sweeps or 1500-launch inner loops.
  Correctness launches are counted separately. D2's fresh-process exception is
  above. At most one identical-binary label control for timer scatter.
- Match effective compiler flags, actual cc1 expansion (`-v`, not `-###`),
  downstream autosync settings and actual loaded-library hash. Keep one fixed
  target contract per comparison family.
- Report samples and actual launch counts separately. Small or inconsistent
  differences are unresolved; overlapping IQRs do not prove parity. No automatic
  repetition escalation to obtain a win.
- Profiles are optional after breadth, at most three meaningful unexplained
  results, with the same binary/shape/state as timing. Do not infer instruction
  stalls from aggregate pipe counters or mix profiled and unprofiled latencies.

## Keep all available devices supplied with work

Use all **eight allocated devices that are actually free**. Inspect broker locks
and occupancy; do not pre-pin to an occupied card or assume card0 is unusable.
All arms of a matched block run on the **same physical device**. Serialize timing
and profiling within each card, while running independent blocks on other cards.
Do not interfere with unrelated jobs. Full-model multi-device runs need their own
nonoverlapping allocation.

Create one persistent coordinator and a durable queue immediately. States:
`discovered → preparing → built → correctness_passed → timed → archived`, plus
`blocked`, `failed`, `deduplicated`, `budget_deferred`. Use unique IDs from source,
input, flags and launch hashes; prevent duplicate submissions. Keep submitted
binaries immutable. Persist commands, job IDs, status, stage times and heartbeat
atomically. Resume finished work from evidence, not from a prose progress claim.

Do not leave ready work idle while waiting for an agent reply, a profile, LM-head
oracle, or a transcription. Host/reference workers and builds use explicit limits
suited to the remote machine's CPU/RAM; share immutable builds and golden caches.
Keep a steady ready queue. Report unexplained scheduler idle periods over five
minutes. Checkpoint broker jobs into chunks no longer than 40 minutes; do not
trust an unlimited-time flag. Bound hangs, save evidence and continue other jobs.

Start the coordinator in a durable broker/session context and verify the first
job actually starts. Report the first device assignment and job ID promptly,
then progress about hourly. Before ending any agent turn, confirm either the
coordinator is still advancing the durable queue or the campaign is explicitly
complete/blocked. Finish without waiting for another instruction.

## Closed experiments and remaining local work

Do not rerun the completed four-arm MAT campaign, placement microkernel suite,
conditional MTE2-barrier isolation, or broad helper/trial ablations. The existing
conditional-WAW compiler repair is local follow-up. The latest proposal/replay
fixes leave all corpus plans unchanged and need no dedicated device arm.

Last-reader placement, contextual frontier-motion certification, conditional
future-key-reuse admission and a positive equal-coverage provider example are
still local work; no candidate for those is being sent here.

## Deliverables

Write `oahs-open-experiments-REPORT.md`, `oahs-open-experiments.tar.gz` and its
matching detached SHA-256 under `/opt/pypto/` or the durable campaign directory.
Verify round-trip extraction and every internal checksum before declaring done.
Include:

1. One main kernel table: experiment/family, exact shape, covered source rows,
   device, arm identities, correctness, medians/IQRs, round medians, ratios,
   absolute delta, sample/launch counts and resolved/unresolved status.
2. Separate complete-model and manual-reference tables. Distinguish original
   native, matched manual, UF=0/UF=1 and reduced experiments. No component sums
   masquerading as model latency.
3. Status for **every D1–D7 workstream**, including precise blockers and remaining
   transcription files. Account for every captured row and all deduplications.
4. Machine-readable queue/results, raw samples, correctness outputs, source/
   input/golden/plan/C++/loaded-library hashes, actual build commands, compiler
   counters, wall time/RSS, harness/transcription diffs and job records.
5. Coverage denominators reconciled from current records. No recycled 264/88
   counts presented as newly executed device coverage. Historical evidence is
   labelled as such. Pack reproducible generators instead of huge oracle caches.
6. A short list of the most useful new scheduling gaps with exact source/deadline
   witnesses or unsupported contracts. Event count alone is not an explanation.
