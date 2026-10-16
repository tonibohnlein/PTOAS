# Broad validation and phase-call attribution

This is the historical M6 attribution. The
[final construction-first receipt](frontier-final-construction-validation.md)
records the later ordered production campaign, 234-test broad refresh,
reviewed repairs and current phase-stage attribution. The retained broad run
below completed 233 tests with 205 PASS,
27 FAIL and one TIMEOUT. Subsequent focused repairs and baseline comparisons
attribute parts of that result; they do not constitute a new all-green broad run.
Implementation acceptance, successful compilation, fixed-protocol capacity and
device validation retain their separate receipts.

## Broad failures and retained native baseline

The historical result is
`.local/milestone-loop/owned-recipe-broad/broad-lit.json`.
Four test expectation repairs subsequently passed in
`.local/milestone-loop/protocol-expectation-gate/focused-lit.json`:
`sync_native_class_certification`, `sync_descriptor_gm_provenance`,
`sync_storage_effects_views` and `sync_manual_skip_mixed`. That five-test gate
also passed the fixed-protocol capacity fixture, including the new prefill
certificate. It changed no production admission contract.

The other 23 failing tests were replayed serially against the three retained
`explicit-namespace-gate` native tools, with the current pinned test/helper
expectations. The local lit configuration loaded the current configuration and
replaced only `pto-test-opt`, `pto-sync-input-test` and
`pto-arithmetic-recognition-test`, using a separate execution directory.
Dependency inspection found no mutable repository/PTO shared library; the
selected helpers launched no other current native tool or compiler wrapper.
This is a native-binary baseline comparison, not a historical test checkout.

The replay used one worker on CPU1, a 4 GiB address-space limit and a 120-second
lit case watchdog. All 23 failed at the same executed RUN prefix as the broad
run. Twenty-one had identical full output after normalizing only native-tool
locations, output/scratch paths and hexadecimal addresses. Two require more
precise attribution:

- `sync_bounded_regional_allocation` had the same allocation-retry diagnostic
  with different raw numeric values at protocol field 167. Both versions first
  passed 11 independent reuse traces, then reported available 1/1, exports 2,
  endpoints 0/0 and protocol lengths 564/564. The numeric difference is retained;
  its cause is not inferred from its appearance.
- `sync_guarded_rotating_analysis` failed differently. The baseline hit the
  helper's 90-second timeout while tracing the parity fixture. The current run
  completed that call and failed the assertion that it must be rejected,
  returning an accepted identity-sequence result. This needs an independent
  semantic/capability review; it is neither an established regression nor an
  accepted expectation repair.

`sync_rotating_region` reproduced `std::bad_alloc` under the same memory ceiling.
This establishes a preexisting construction limit under that ceiling, not
mathematical impossibility. Baseline equality also does not make an existing
failure acceptable. The excluded `sync_composed_phases` timeout was investigated
separately below.

Raw outputs, exact selected tests, commands, tool hashes/dependencies, helper
pins, normalized differences and post-run pin checks are retained under
`.local/milestone-loop/namespace-baseline-broad/`. The replay completed in
165.22 seconds without a lit-level timeout; the guarded helper timeout above
is a separate failure.

## Independent phase RUN-entry attribution

Ten RUN entries were invoked independently using the pinned
`cofactor-repair-gate/pto-sync-input-test` and the original helper arguments and
child timeouts. The driver retained each child command, caller stack, generated
input and hash, output, and completion/exception event. Each checker ran serially
on CPU0 with 4 GiB address space and a 120-second whole-checker watchdog.
The pinned sources were unchanged after execution.

| Entry | Recorded result | Native child attribution |
|---|---|---|
| `check_composed_phases` | Success | 42/42 child calls completed |
| `check_carried_phase_closure` | Success | 15/15 child calls completed |
| `check_periodic_phase_lengths` | Whole-checker timeout | Call 1, structured trace, unfinished |
| `check_phase_owned_storage` | Whole-checker timeout | Call 21, structured trace, unfinished; first 20 completed |
| `check_repeated_region` | Helper assertion | 14/14 completed; final insertion succeeded, then version assertion failed |
| `check_repeated_phases` | Child timeout | Call 1, structured trace, raised the existing 90-second timeout |
| `check_nested_allocation` | Whole-checker timeout | Call 1, structured trace, unfinished |
| `check_guarded_rotating_analysis` | Helper rejection assertion | 113/113 completed; no failed native call |
| `rotating-region-checks` | Success | Direct native entry; no instrumented child calls |
| `check_rotating_region` | Native failure | Call 15, logical insertion, exited -6 with `std::bad_alloc` |

A started call without an end event identifies the interrupted invocation,
not the internal operation responsible for the timeout. In particular, a
structured-trace command can include production preparation/insertion before
interpretation; its mode alone does not identify the stalled internal stage.
These results do not establish that the complete original composed-phase lit
test passes. Instrumentation overhead also prevents paired timing claims.

The exact entries and caller/input evidence are in
`.local/milestone-loop/owned-recipe-phase-attribution/{results,call-attribution}.json`;
`pins.json`, `pin-verification.json` and `context.json` record source identity,
unchanged inputs and execution limits.

## Pinned receipt identities and remaining obligations

All paths in this table are relative to `.local/milestone-loop/`.

| Receipt | SHA256 |
|---|---|
| `owned-recipe-broad/broad-lit.json` | `21ce77ed1d8f4e8be73858ec0f422b3026bae5186d6ed0afe5489c0bcb6b37ee` |
| `protocol-expectation-gate/focused-lit.json` | `e389a5b7ec008fe988335ce2f3d71a9419c55f06f112759793b33a34f8351ef2` |
| `namespace-baseline-broad/comparison.json` | `d88d54d3772de3edff1af5235ad8ebf260268ffed4ba23920cb267f0d5e4aeaa` |
| `namespace-baseline-broad/source-manifest.json` | `029245ed5ed841fd4c0b249a2ecedb6d50dacf8d08d6be489f40aab76dd80367` |
| `namespace-baseline-broad/tools.json` | `83c84cc84c91dccf12b96e8a9a26ae140030457536c1ebc775a81e42e888ca8c` |
| `owned-recipe-phase-attribution/results.json` | `099ff181fffe25b237f3ade9118dbfcde47c1d49a0a07392c853ff985884f666` |
| `owned-recipe-phase-attribution/call-attribution.json` | `efb14ee28de97283c0b12d7ee1ad8e6bcae7fb9a44e451e623597f80a94a10ba` |
| `owned-recipe-phase-attribution/pins.json` | `592aebb9413a6920a3b988a7f1cda11fddaf86bbf645c21d41d80c1cebdb4c81` |

Remaining work includes the guarded-acceptance semantic review, attribution or
repair of unfinished phase calls beyond their CLI entry point, and explicit
disposition of the reproduced failures. A later broad gate must name its own
source/tool snapshot. The separate
[archived guard-definedness receipt](frontier-guard-arithmetic-definedness.md)
closes only its three exact artifact pairs. No device results, full M6 completion
or successful run of all 233 tests are claimed here.

## Internal phase-stage samples

The four unfinished phase calls were rerun individually under a debugger-owned
child using the retained 223-input tracer. Each was sampled after 30 seconds and
then deliberately terminated. The debugger completed successfully; **none of
these four original calls is recorded as completed**. One sample establishes
the stage executing at that moment, not its exclusive cost or every later stage.
Serial one-CPU/4-GiB runs and exact input/tool/command pins are retained under
`.local/milestone-loop/phase-native-stage-samples/`.

| Saved checker call | Observed internal stage |
|---|---|
| `check_periodic_phase_lengths` | Arithmetic single-ID allocation: handoff `Proof::compose`, integer-system intersection and normalization, reached through `attachAllocation`. |
| `check_phase_owned_storage` | Regional lane-certificate strengthening: Boolean evaluation in `constantUnder` via `regionalLaneEvidence`, reached through `attachAllocation`. |
| `check_repeated_phases` | Sequence mathematical composition: integer-relation translation and `RelationBuilder::conjunction` while normalizing guarded/modular predicates. |
| `check_nested_allocation` | Sequence mathematical composition: the same integer-relation translation/conjunction path. |

This narrows the earlier CLI-only attribution. In the first two cases, the
structured-trace caller currently requests allocation preparation even though
the saved input has no `test.eligible_ids` request. A request-based caller repair
is under review; these samples do not claim it is implemented or effective.
The latter two samples remain exact relational-construction work, not evidence
of an impossible protocol or permission for a conservative replacement.

| Sample receipt | SHA256 |
|---|---|
| `.local/milestone-loop/phase-native-stage-samples/results.json` | `4bde4ac9637340f97c04ab4824f3e3ebe8876568ded109096a7cd541e76d8e3f` |
| `.local/milestone-loop/phase-native-stage-samples/check_periodic_phase_lengths/sample.out` | `1bd3b5f3ad1d0067a7a8263cf01985cd37a9fd4ba8aacdc143125ebe1237b317` |
| `.local/milestone-loop/phase-native-stage-samples/check_phase_owned_storage/sample.out` | `e48caeed528a6767caa70186e36ce2b407c01c76dbd399b92ab740543f9c7483` |
| `.local/milestone-loop/phase-native-stage-samples/check_repeated_phases/sample.out` | `d5f9df80ab109dc7ffaff0b9162aa2ac69223feac4417bb7d4f2868d15973182` |
| `.local/milestone-loop/phase-native-stage-samples/check_nested_allocation/sample.out` | `d07fa97ce909ad916de86c14bf338cbb5d9604201855031db87a11264c58eeb7` |
