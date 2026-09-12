# OAHS S7: typed accumulator order and a GEMM-oriented corpus

Base: `6e38952c22cf3bc6e9bb0a0bf68e4ff5579ac9c0`, after applied S6
`d83ddde736cdb4a70cb9906e431bc2e2dedbfbf6`. Remain on `codex/oahs-upstream`.
This changes the existing structured engine, not its organizing algorithm.
No Presburger query, search quota, payload rewrite, or new pass is introduced.

## 1. Scope and default

S7 adds an explicit **experimental lowering/hardware contract**:

```
-pto-insert-sync=planner=structured hardware-contract=a2a3-mmad-acc-v1
```

The default is `hardware-contract=conservative`. Ordinary InsertSync and the
previous structured premise set remain the defaults. Selecting the new
contract is an explicit assertion that the actual device and downstream
lowering obey this source-qualified rule. Source review and passing compiler
tests do not establish a production CANN release or device qualification.
Only the existing native pass option and C++ options are added; the Python
convenience CLI has no new forwarding spelling in S7. The native test driver
accepts the optional final argument `a2a3-mmad-acc-v1`.

The contract is deliberately narrower than the hardware documentation: plain
in-place half/bfloat16 -> float matrix accumulation, full constant valid
geometry, compatible exact L0C storage, unspecified AccPhase (no UnitFlag),
and `(m/16)*(n/16) > 10`. Equality with ten remains conservative because the
published prose and example do not state that boundary identically. Shapes
not divisible by 16, GEMV, bias, MX, mutable/partial descriptors, unknown
physical regions and reset/interleaved accumulator chains receive no credit.

The new optimizer does not synthesize UnitFlag, enlarge event pools, infer GM
visibility, or give generic scalar effects synchronous completion. Existing
scalar/core policies are not expanded by this patch.

## 2. Evidence and its limits

Primary source:
https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute/Mmad.html

The synchronization-optimization section describes consecutive K-axis updates
of the same L0C result. Its code requires an M barrier below ten 16x16 output
blocks and its prose describes hardware write/read handling above the
threshold. This is the ISASI API, not the unrelated Tensor API page. The site
states that it is generated from a development branch, possibly with
uncommitted modifications. S7 therefore does not silently promote this into
an unconditional A2/A3 production premise.

Reviewed PTO-ISA lowering:
`hw-native-sys/pto-isa/include/pto/npu/a2a3/TMatmul.hpp`, Git blob
`f501ef5b95252ec395036f3a56f80eedaeb93ba4`.

The plain overload derives m/k from LEFT valid shape and n from RIGHT valid
shape, then issues mad. The accumulator-input parameter does not supply a
separate physical mad address; S7 therefore requires exact in-place C input
and output. Its `m==1` special handling is outside S7's admitted rule.

Reviewed PTOAS lowering:
`lib/PTO/Transforms/PTOToEmitC/PTOToEmitCLoadStore.cpp`, Git blob
`a3d55a75c1a6ee9214624d2347b736f90694526e`.

It emits plain TMATMUL overloads with AccPhase template arguments when
requested. S7 admits only Unspecified. Before enabling this contract in a
new toolchain, compare these routines and qualify the actual emitted mad
instructions, geometry, accumulation flags, retirement and event semantics.
The source hashes are provenance, not an automatic version-negotiation API.

## 3. The essential semantic separation

Add `Property::AccumulatorUpdate` and per-atom `MmadInfo`. Immutable original
requirements remain in the ledger. Native discovery selects the narrow
property only when every retained overlapping conflict for the atom pair is
in ACC, both lanes are AIC/M, and no resource/visibility requirement is mixed
in. Unqualified matrix descriptors still retain that obligation, and it must
then be supplied by ordinary synchronization.

`intrinsicAccumulatorOrder(model, requirement)` checks the exact original M
stream. It follows every intervening M operation, including unknown ones.
Each continuation must accumulate into the same physical C with the same m/n,
input dtype, qualified full layout and effective dimensions. An unknown M
operation, another accumulator, a reset, small shape, or incompatible geometry
breaks the chain. The scan is finite in static M atoms, including at most one
period wrap. It does not unroll the loop's trip count.

A successful query means only that the particular ACC update is ordered.
It does NOT add an edge to the full completion table or event-causality graph.
It cannot satisfy a full Completion, AccResource, Visibility, M->MTE1 operand
release, M->FIX completion, or event-recycling obligation. This distinction is
why hardware order can be useful without making every earlier M instruction
appear completed.

Initial and steady phases can each use the rule. Cross-segment order and
cross-invocation carry remain conservatively synchronized in this increment.
This retains the startup transition barrier in the Q-projection model rather
than assuming its two independently represented phases are one full-completion
frontier.

## 4. Native qualification and fresh checking

Recover m/k from the LEFT operand and n from RIGHT, never from the maximum
accumulator allocation alone. Require direct immutable alloc_tile descriptors,
constant full valid shapes, LEFT/RIGHT/ACC scopes, supported layouts/fractal
sizes, F16/BF16 operands, F32 output, and one exact checked physical ACC range.
Reject any descriptor with a SetValidShape user. In-place accumulation requires
identical C input/output type, effective shape and physical interval.

Ordinary requirements and selected handoffs still drive the same constructor.
S6 full-region deletion and startup coalescing then operate on the complete
plan. Every accepted emitted program is reconstructed and checked. Following
fresh whole-function physical translation, matrix lowering descriptors are
rederived and compared before committing the clone.

Diagnostic JSON retains the S6 accounting schema with optional S7 fields:
- `hardware_contract` per unit;
- `matrix_lowering` per atom;
- `intrinsic_accumulator_order` per original requirement;
- `intrinsic_accumulator_requirements` at the root.

`new_hardware_elision` denotes nonzero qualified requirement credit, not a
measured number of eliminated commands. s6_report validates this accounting;
it is not an independent hardware proof.

## 5. Tests and the historical GEMM

The existing seven conservative native cases remain mandatory and unchanged.
The hardware gate additionally compares all seven under both contracts and
requires unchanged Q projection to exercise the rule and reduce named M
barriers, while QK retains its independent first-panel acquisition.
The gate additionally calls `check_hardware.py` on five synthetic geometries,
each on A2 and A3: eight, ten, eleven and sixteen output blocks plus a
128x256 GEMM-sized accumulator. They compare conservative versus selected
contract on identical input; require actual native rule hits; require named M
barrier reduction only for qualified shapes; and reject missing operand-release,
final-FIX, wrong-key and retirement actions. Zero/one/short loop probes check
participation, not numerical output of an uninitialized zero-trip accumulator.

These fixtures deliberately use two panel products per iteration. They compute
a documented synthetic expression; they are not relabeled historical GEMM or
performance data. The native pass-option entry is tested separately from the
direct C++ testing entry.

The portable suite retains all S1-S6 cases and adds independent finite
occurrence checks for the new narrow premise. The checker expands actual M
occurrences and tests their accumulator chain separately from its asynchronous
issue/completion/token graph. It never inserts the intrinsic premise into that
graph as a full operation-completion edge. Directed negative controls challenge
geometry thresholds, resets, unknown M instructions, changed C, generic
completion, operand release and FIX visibility.

`S7_QPROJ_MODEL` is a hand-transcribed access model, not native Q-projection
compilation. Its result remains distinctly labeled in qualification reports.

S7 does NOT make the full historical GEMM compile. Physical-section region
entry, runtime-strided wrappers and final-prefetch participation still need
structural transfers. The existing pinned `recover_historical_gemm.py` obtains
its exact source from Git history without changing branches. `s7_corpus.py`
includes the recovered archive, or reports it missing. `--require-historical`
prevents silently claiming completion of that requested corpus when absent.
Do not weaken or rewrite that input to turn the refusal into acceptance.

## 6. Expanded corpus workflow

`s7_corpus.py discover` includes the seven frozen sources, available
hc_head_linear/hc_pre_linear/weights_proj regressions, the new hardware fixtures,
and the recovered historical archive. `--all-a2a3` additionally enumerates
A2/A3 sample files. Missing requested regressions are explicit manifest entries
in the missing list; A5 is not silently counted as an A2/A3 failure.

Discovery labels inputs RAW compatibility. Production admission needs a
manifest of immutable unsynchronized snapshots captured at the actual pass
point, including required memory planning/temporary materialization. Such
entries require a preparation/provenance record. The runner never pretends that
an arbitrary no-PlanMemory invocation is the production pipeline.

All three arms receive exactly the same saved bytes. Raw a2a3 target metadata
may be bound explicitly to a3 once, with that adaptation recorded; prepared
snapshots must already name a concrete target. There are no payload, alias,
view, temporary or guard repairs. GM defaults to may-alias. Use
assume-disjoint-arguments only for a justified caller contract, never an
all-accesses-disjoint assumption.

Results separate preparation stages, production/focused/negative/historical
classes, missing inputs, parser/construction failures and measurement failures.
Mechanism counts remain sets, waits, named barriers by pipe and PIPE_ALL; key
use remains per directed domain. No combined score or aggregate runtime claim
is computed. The runner's single rotated round is diagnostic, not the <=2x
whole-compilation campaign. Its external process timeout is not a compiler
analysis quota. Device correctness/timing are not measured by this runner.

## 7. Acceptance and next boundary

Before deploying the selected contract: build native adapters, pass the entire
structured/default gates, qualify the exact CANN/PTO lowering on A2/A3, inspect
emitted operations and run numerical/progress/device timing tests. Corpus
admission and synchronization reductions are separate outcomes.

The next historical-GEMM increment should address its recorded first structural
blocker while retaining this typed premise interface. Do not replace the
occurrence algorithm, introduce a whole-GEMM recipe, or remove operand releases
because an ACC-specific requirement was discharged.

## Revised coverage candidate (packaging-corrected follow-up)

This follow-up keeps the MMAD accumulator-ordering contract above and adds a
coverage-oriented integration candidate. It is deliberately fail closed and
must not be confused with a complete historical-GEMM milestone.

### Physical-effect admission

The structured path now reuses the existing translated physical phase for an
ordinary single-phase operation instead of relying on a closed opcode list. An
operation is admitted only when its concrete A2/A3 lane is supported, exactly
one translated phase exists, the translated lane matches, every declared memory
effect is represented by the translated read/write sets, the summary is
nonempty, and every local footprint is physically qualified. Refusals identify
which of those predicates failed. Arbitrary PIPE_S payloads remain unsupported:
the target contract has no generic scalar-completes-at-issue premise and no
PIPE_S same-pipe barrier.

Visible pure helpers may be ignored. Effectful helpers require a complete
`pto.tileop.effects` contract, a qualified helper lane, and one matching
translated phase. Communication, queue/session, cache/visibility,
configuration-resource, debug/print, and unresolved multi-phase operations
remain unsupported.

### Direct sequential sibling loops

A restricted sequence of direct sibling `scf.for` operations in one block is
composed after each loop has independently passed local construction and fresh
checking. Cross-loop physical hazards are summarized at the scalar boundary.
A same-lane hazard uses a supported barrier before the later loop. A legal
cross-lane hazard uses Set after the latest conflicting earlier loop and Wait
before the later loop. Event intervals are colored from the qualified 0--5 pool
and checked for consume-before-rearm. No internal `PIPE_ALL` is introduced.
GM hazards remain unsupported until a qualified visibility recipe exists.
Nested children inside a sequential sibling remain outside this first fragment.

### Runtime wrappers and physical sections

Enclosing invocation wrappers accept constant positive coordinates and the
exact block-distributed form whose lower bound is `pto.get_block_idx` and whose
step is `pto.get_block_num`, through index casts. Generated predecessor and
successor predicates use the original lower/step SSA values. Arbitrary runtime
strides remain unsupported.

One qualified top-level `pto.section.cube` or `pto.section.vector` may define the
physical lifetime scope. Scalar setup outside the section is not imported as
physical payload, synchronization is emitted inside the selected context, and
the existing single retirement-drain policy is transferred to the section end.
Multiple or nested physical sections remain unsupported.

### Explicit remaining gap

Last-sensitive original payload such as the frozen historical GEMM's
`if (k + 1 < K) prefetch-next-panel` is **not** admitted by this follow-up. The
compact core does not yet have an occurrence domain for an operation that
executes on every body iteration except the last. Treating it as an ordinary
periodic atom would invent a final execution and could invalidate event
participation. The importer therefore continues to refuse that guard rather
than silently dropping it. The frozen historical GEMM is still an acceptance
target, not a claimed passing fixture.

UnitFlag and further hardware mechanisms remain separate future increments.
