# OAHS GM provenance and range coverage

Status: compiler increments implemented and locally verified. Ordinary
same-range MTE3-to-MTE2 publication remains unqualified.

## Frozen target

The before manifest is
`../oahs-coverage-work/gm-publication-targets-6d57749a6-v5/manifest.json`.
It freezes the 89 current first-publication-refusal rows from
`campaign-boundary-final-r1/summary.json`: 69 distinct prepared inputs, 56
`may-alias` rows, 33 `assume-disjoint-arguments` rows and 169
matching-consumer/cell writer candidates. Each row records the original input
and prepared hashes, compiler output hash, alias/hardware/ownership contracts,
writer and reader operations, roots, ranges, cells, control relation, bounded
reaching-writer relation, legacy overlap result and emitted InsertSync
synchronization. Other Cartesian candidates remain in a separate field.
Matching the failing reader and cell does not prove that a candidate writer
reaches that reader: 14 matching candidates cannot precede it within one
invocation, and cross-invocation generation/predicate facts remain conservative.

The source campaign was measured before commit `6d57749a6`, at revision
`0b54eff6f`; that commit preserved the measured compiler implementation. The
new manifest records this historical identity rather than relabeling it as a
fresh run of the current compiler.

## Compiler increments

Structured forwarding uses the existing bounded monotone transfer graph for
`scf.if`, `scf.for` initialization/backedges and `scf.while` before/after
condition/yield/result forwarding. Possible roots and unknown coverage are now
independent facts: analysis exhaustion or an unsupported path retains roots
already encountered on the source graph and adds unknown coverage instead of
replacing the finite set.
Translation clones the retained root facts and adds a wildcard memory fact for
the unresolved portion.

The regression carries a value selecting A or B through an `scf.if` and an
`scf.for`, writes explicitly disjoint C with MTE3 and reads the carried value
with MTE2. The witness preserves two read origins `{A,B}`, each disjoint from C.
Deleting the explicit noalias pairs restores the publication refusal.

GM cell construction now reserves one cell for each known root, possible-alias
group and unknown-only remainder before constructing optional geometry. An
unresolved access overlaps every known cell, while roots separated by the
caller contract retain independent cells. At the limit, the affected root or
alias group coarsens deterministically before it consumes a later group's
reserved cell. If the minimum population itself exceeds 256 cells, or the
shared local/GM partition later exceeds the global limit, construction still
falls back to a whole-GM cell.

Checked same-root intervals retain constant contiguous byte geometry through
supported ND pointer/view/partition composition. The positive regression
preserves `[0,64)` and `[64,128)` as separate cells. Moving the second access to
`[32,96)` restores the refusal. Invalid units, overflow, dynamic positions and
unsupported layouts retain conservative overlap.

## Frozen replay result

All commands are serial except the targeted compiler link, which used two
workers. Static synchronization is recorded by the campaign; no corpus device
execution is claimed.

| Increment | Focused native gate | GM refusals removed | Newly admitted inputs | New downstream refusals | Admission regressions |
| --- | --- | ---: | ---: | ---: | ---: |
| Structured origin propagation | 25 pass, 10 expected refusal | 0 | 0 | 0 | 0 |
| Unknown-access isolation | 25 pass, 11 expected refusal | 0 | 0 | 0 | 0 |
| Same-root interval precision | 26 pass, 12 expected refusal | 0 | 0 | 0 | 0 |

Each full replay remains 253/363 admissions: PTOAS 79, PyPTO 7 and pypto-lib
167. The first two comparisons are in
`gm-publication-comparison-origins-r1` and
`gm-publication-comparison-unknown-r1`; the final enriched comparison is
`gm-publication-comparison-ranges-v2`, all under the sibling
`oahs-coverage-work` directory.

The final hardened gate is
`composition-gm-final-precommit-r7`: 26 positive native cases, 14 expected
native refusals, 191 mutation checks and two frontend checks. Its two added
bounded tests retain a finite root when origin discovery exhausts and preserve
three disjoint known-root groups next to an unknown access at the 256-cell
limit. The final replay is `campaign-gm-final-precommit-r2`; the v5 comparison
is `gm-publication-comparison-final-precommit-v5`.

The four local CTest targets `oahs_composition_core`, `oahs_composition`,
`oahs_structured_core` and `oahs_structured` pass serially. The eight-case
conservative benchmark record is `benchmarks-gm-final-precommit-r3`. All 24
planner/case arms have the same static mechanisms, scalar sites and physical
keys as `benchmarks-boundary-final-r1`; device execution is `NOT_RUN` and the
single host compilation samples are telemetry. The qualified historical GEMM
gate retains 54 SET, 54 WAIT, no body barrier, one terminal `PIPE_ALL` drain
and no tested boundary difference.

Zero gains are explained by the current target candidates:

| Remaining category | Rows | Evidence |
| --- | ---: | --- |
| Missing external alias guarantee | 56 | 126 matching candidates have complete but different argument roots under `may-alias`. The current witnesses contain no established disjointness contract; distinct SSA values and legacy non-overlap decisions do not supply one. |
| Genuine or unresolved overlapping GM communication | 13 | Every matching candidate has exact overlapping same-root ranges: seven ordinary TSTORE/TLOAD rows and six communication-macro rows. A reaching dependency still requires publication qualification. |
| Unsupported range or layout precision | 20 | Nine rows have an exact 256-byte endpoint paired with a dynamic endpoint, six involve TGET, two use dynamic loop/backedge partitions, and three involve TPUT. |
| Unsupported provenance transformation | 0 | All 169 matching candidates retain complete finite roots. |

The structured and unknown regressions exercise information-loss modes absent
from these 89 rows. The constant interval regression exercises a supported
case, while the target's unresolved same-root positions are dynamic or macro
access paths. The current evidence establishes no noalias fact for the 56
cross-root rows; additional control or reaching-generation precision may still
remove candidate writers. Exact same-range reaching communication requires a
qualified publication mechanism.

The comparison tool records the exact disjointness proof for every gained
admission. This increment has none, so it emits no causal proof records. It
also verifies the same 363-row population, prepared hashes and contracts before
comparing outcomes.
