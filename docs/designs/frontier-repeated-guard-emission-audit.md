# Repeated-guard endpoint audit: milestone 1

Starting revision: `4d1767ae22a97f739d329a1702dd91561a705a5f`.
The implementation plan is [repeated guard emission](frontier-repeated-guard-emission-plan.md).
This milestone adds diagnostics and an opt-in exhaustive audit; it does not change
mathematical analysis, guard recovery, command semantics or allocation.

## Gated Delta evidence

For `prepared/gated_delta_rule/input.pto`, MayNotAlias still certifies all 29
original regions. MayAlias still certifies 23 of 29. Certification output under
both policies and the retained-demand check are byte-identical to the pinned
starting executables. Failed endpoint attempts leave the original module intact.

The token loop at line 144 has 60 sites, span three and 1,172 retained symbolic
records. An isolated detached preparation of every record finds:

| Outcome | Records |
| --- | ---: |
| Endpoint recipe prepares | 101 |
| Publication contract unavailable | 4 |
| Consumer guard recovery unavailable | 1,067 |

The four publication failures are records 4, 11, 21 and 42: source site 2
(`tcvt`, line 151) to sites 7 (`tsetval`, line 160), 14 (`tsetval`, line 170),
16 (`tsetval`, line 173), and 21 (`tgetval`, line 178). All have displacement
zero. The first failure therefore is a SET/publication obligation, not a
consumer-recovery failure. Its current-visit branch condition is defined after
the source cut and depends on `tgetval`; it cannot be replayed there.

Consumer failures have displacement one (355), two (356), or three (356).
The dump records each original operation, phase pipe, failing cut, exact guard
DAG, integer recipe signature, requested evaluation coordinate and original
opaque SSA leaves. Definition dominance and visit displacement are separate
fields: a late current-visit value is not classified as a future visit.

A separate uniform proof using the existing monotone visit-presence certificate
proves 900 original guards identically false. Of these, 879 are currently
blocked: two publication failures and 877 consumer failures. This is evidence
for emission-only inactive recipes, not a change to the reducer. Even after
such a change, 192 blocked records would remain: two publication failures and
190 consumer failures (62/64/64 at displacements one/two/three).

## Verification and reproducibility

The source-pinned candidate build uses effective `-O2 -UNDEBUG -std=c++17`
flags across all 185 Frontier and test-tool object commands. Baseline executables
were preserved before rebuilding. Local receipts, exact source/binary hashes,
object commands, all obligation records and paper hashes are under
`.local/three-paths/repeated-guard/m1/`; the final manifest is `manifest.json`.
Large DAG dumps are local diagnostic artifacts, not committed test data.

Focused optimized lit checks pass (4/4): bounded endpoint audit, bounded boundary
presence, guarded publication, and bounded regional interfaces. The last covers
session/result ownership and retention after failed preparation. The audit
checks repeated failure counts, agreement with normal preparation and unchanged
original IR. Gated Delta exhaustive audit completes in 24.54 seconds with
82,248 KiB peak RSS; normal certification takes 0.80 seconds (MayNotAlias) and
0.10 seconds (MayAlias). Watchdog/memory limits are measurement safeguards.

The changed-code checker reports 10 files, zero errors and zero warnings;
`git diff --check` passes. No new compiler warnings were introduced. Full
EChecker/SecK analysis was not run. Applicable rules cover C++17, explicit error
paths, detached ownership, checked diagnostic inputs and scoped changes.

Section 8 was reread before freezing the milestone. Its current SHA-256 is
`c9a6a9ad72f94e78bf54740b3ee2c25dd577e269b5a10adb59f59dba8db2647d`,
which differs from the planning pin. The working-tree changes refine explicit
versus compact allocation schemas, direct-plan reuse queries and numerical
periodic allocation. They do not change this milestone's guard or detached
preparation contracts. The other three paper pins remain unchanged. Section 8
still requires available guards and retained exact mathematics after emission
failure. This milestone makes those obligations observable without implementing
stateful evaluation.
