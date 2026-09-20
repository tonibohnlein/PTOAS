# MAT reader-region cycles

Implemented in `8afb90f17` from the working-tree snapshot based on `fe1fc454b`.
The [final family campaign and audit](oahs-mat-device-final-results.md) now
supersede the preliminary down-projection report below. Preserve the measured
source snapshot and its archive.

## Problem and construction

The preceding plan acquired a broad MTE1 return before refilling the first MAT
pair. Its publication included the second reader child's unrelated work. The
new constructor selects readiness and release together for an admitted physical
storage generation: publish after its producer, acquire at its first reader,
publish release at that reader region's exit, and acquire before its next writer.
The return is available before ordinary same-engine repair. Actual readiness
connects the old writer to the readers; actual release then covers both.

Admission is generic: exact nonexclusive cells, one writer and one reader engine,
cyclic writers, nonempty leaf reader regions with no internal regeneration or
writer-pipe activity, and a qualified entry or first-input acquisition. Every
read must be covered and both endpoint streams must pass the existing original
control participation check, including priming/draining and tail reuse. The
normal selected-plan checker still certifies the resulting commands.

The native first-consumer qualifier also admits a compact prefix when several
invariant inputs are rewritten by a counted parent. This preserves separate
first consumers for second-child and odd-tail inputs. It retains the existing
straight-prefix, original-bound, inactive-producer and storage qualifications.
No new runtime history counter, full nested expansion, fence-deletion search or
same-pipe completion assumption is introduced. Distinct release publications
remain distinct. Existing allocation admission and fallback are unchanged.

## Production versus controlled comparison

Complete cycles now discharge the MTE2 requirements before fence selection,
including down_proj, whose earlier diagnostic split-return protocol could not
prove them. The production candidate retains all M/FIX fences and terminal ALL.

For attribution, a separate `placement_control` restores each baseline MTE2
fence at its original static payload deadline in the candidate. All eleven
controls pass the unchanged full causal checker, and finite comparisons retain
the entire baseline fence population. This is a diagnostic arm, not a production
cleanup pass. Protocol comparisons also change dynamic event counts; do not
attribute their device effect solely to placement or solely to command counts.

## Validation

- Portable suites: 22/22; native selected suite passes.
- Supported corpus: 88/88 construct/reconstruct. Exactly eleven Qwen projection
  modules change. Other 77 outputs, including Shenggan, six attention modules,
  post-RMSNorm and vector controls, are byte-identical to fe1.
- Across 37 projection paths and 511,187 local conflict checks, payload identity
  is unchanged. Production removes 6,881 finish-to-issue relations, adds zero;
  retained-fence control removes 6,173, adds zero. These are finite ordering
  witnesses, not timing predictions or exhaustive native correctness proofs.
- New early-release assertion rejects fe1 on the second-child-reader to next
  first-pair refill relation. Positive repeated-child tests vary inner lengths;
  missing readiness/return channels fail the full checker. Independent readers
  and regeneration in a child are conservative admission negatives.
- Shenggan remains 200/394/782 pairs for one/two/four tiles, zero named barriers,
  one terminal ALL; its plan is byte-identical to fe1.
- Down/KV/LM candidate plans lower to C++; production emission regressions run
  separately. All eleven restored-fence PTO controls parse and receive a full
  original-program causal certificate in the diagnostic driver.

Representative dynamic counts (one tile): down/17 chunks 343 to 386 pairs;
LM/bounded one tile 363 to 382. Both controls use the candidate pair population.
More commands preserve narrower completion boundaries.

## Construction work

Deterministic work counts, baseline fe1 to candidate:

| Case | Sites | Replay-site evaluations | Selected updates | Recurring channels |
| --- | ---: | ---: | ---: | ---: |
| Down | 171 to 181 | 22,796 to 13,563 | 29 to 16 | 12 to 22 |
| Gate/up | 229 to 239 | 24,900 to 15,958 | 36 to 22 | 12 to 20 |
| KV | 223 to 233 | 49,686 to 30,253 | 36 to 22 | 12 to 20 |
| Q | 119 to 124 | 7,204 to 4,940 | 18 to 11 | 12 to 20 |
| LM | 127 to 132 | 7,307 to 5,014 | 18 to 11 | 12 to 20 |

Recurring omission trials remain zero for these cases. Qualification, replay and
final helper checks remain separately recorded in construction logs; zero
omission trials does not imply zero final checks. The new qualifier indexes
accesses once, then examines candidate cells and their reader loops. Native
first-consumer qualification scans enclosing writes per candidate child. These
are bounded by represented incidences/regions, not a global linear-time claim.
Single-run wall times are not a controlled compile-time benchmark.

## Next measurement and open work

Run down first, LM as transfer, then the projection family and unchanged GEMM.
Use four arms: existing, fe1 baseline, retained-fence placement control, production
candidate. Reconcile profiles using the same timing binary and launch arguments.
See [the device task](oahs-mat-release-device-task.md).

The optional-cohort starvation finding, remaining common-frontier certificate,
next-provider selection correction and FIFO contextual replay cost remain open.
This change does not claim to repair them. Existing's strict-oracle failures on
some diagnostic tail/re-entry paths remain documented in the prior diagnosis;
they are not native wrong-code findings or waived candidate checks.

Raw local evidence: workspace `mat-release-work/`; the self-contained device
package carries plans, source snapshot, controls, comparison tools and logs.


## Device feedback: down_proj (2026-09-20)

User-supplied preliminary report from the pinned MAT campaign, not a new local
measurement and not a device test of the later placement/admission experiments.
36/36 correctness runs pass across four arms: three seeds, blocks 0/19 and a
queued four-slot repeat, correctness-gated before timing. The timing campaign
contains 720 samples over six balanced rotated rounds.

| Arm | Median us | IQR us |
| --- | ---: | --- |
| placement_control | 25.704 | [25.686, 25.872] |
| candidate | 25.792 | [25.686, 26.030] |
| existing | 30.077 | [30.059, 30.230] |
| baseline (fe1) | 31.646 | [31.623, 31.684] |

Candidate/baseline is 0.8150 and candidate/existing is 0.8575, both IQR-disjoint:
18.5% faster than fe1 and 14.3% faster than existing on down_proj. The control
retaining both baseline MTE2 fences is also faster than baseline (0.8122).
Candidate/control is 1.0034 with overlapping IQRs: this campaign detects no
latency benefit from omitting the two MTE2 fences.

The useful evidence is that the shared readiness/return protocol and control
placement delivers the improvement even with those fences retained. It does
not isolate placement from synchronization overhead: baseline has 56/57 static
SET/WAIT instructions versus 66/69 in both new arms. The candidate/control
comparison does isolate the MTE2 omission, with equal SET/WAIT populations.
Do not transfer this result to other projections before their measurements.

Reported host gates verify 8,038 source files and modes, exact restored-fence
control reproduction at down's cuts 0 and 20, payload identity and all static
pins across eleven projection rows. Both new plans pass early-MAT and
early-release assertions; fe1 fails the new release negative control. Attention
44/45 and post-RMSNorm22/23 plans and C++ remain identical to fe1. The agent
reports all three OAHS host suites exit zero. LM head, gate/up, KV, Q/out,
GEMM, remaining host rows and matched four-arm profiling are still running.
The full archive and raw measurement report have not yet been supplied here.
