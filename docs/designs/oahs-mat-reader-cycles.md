# MAT reader-region cycles

Implemented as a working-tree change based on `fe1fc454b`, 2026-09-20.
Device performance is pending. This implements the follow-up to the remaining
projection-gap diagnosis; it does not change the already running device arm.

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
