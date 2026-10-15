# Boolean cofactor construction checkpoint

A substitution with exactly one Boolean input bound to a literal uses an
arena-wide structural reverse index. Only affected nodes are rebuilt; immediate
unaffected operands retain their original identities. Numeric, mixed, nonliteral,
internal and duplicate bindings keep the original root-local traversal.

The index retains structural edges only. Rebuilt expressions and affected sets
are request-local. Transaction rollback removes indexed edges before deleting
nodes, without allocating cleanup storage; partial index insertion restores a
complete prefix on allocation failure. Opaque evaluation provenance is not an
expression operand. All bindings are validated and replacements remain atomic;
both Select arms retain the existing eager rebuilding semantics.

The native build and all 51 focused checks passed. Controls include independent
signed/modular results, unaffected branches, overlapping roots, original numeric
traversal, integer recipes, undefined eager arms, rollback/reused IDs and moves.
Correctness and architecture acceptance and exact source/tool pins are retained
in `.local/milestone-loop/cofactor-repair-gate/`.

## Paired production evidence

`.local/milestone-loop/construction-paired-final/` records serial, interleaved
runs against the retained namespace checkpoint, using identical inputs, policy,
flags, one CPU and a 4 GiB process bound. The candidate also includes the reviewed
allocation diagnostics and Balanced owned-recipe migration; these are combined
batch observations, not an isolated causal measurement of this change.

| Case | Baseline wall seconds | Candidate wall seconds | Baseline peak KiB | Candidate peak KiB |
|---|---|---|---|---|
| Delta, pair 1 | 17.554 | 14.696 | 395460 | 448244 |
| Delta, pair 2 | 19.762 | 17.654 | 395800 | 448424 |
| Delta, pair 3 | 19.458 | 17.704 | 395944 | 446452 |
| Prefill, one pair | 72.793 | 75.946 | 993612 | 993640 |

Delta history-substitution node visits fell from 39933154 to 9017177, including
39209210 to 7054596 at the largest incoming root. Tree comparisons and rollback
maintenance are not counted as node visits. Peak memory increased by roughly
51 MiB; prefill has no demonstrated improvement from its single pair.

Complete output bytes are identical across baseline and candidate for each
case. Delta still constructs 449430 prefix operations, including 207921 at its
largest incoming root. Thus the traversal improvement does not close the M1
requirement to reduce actual emitted construction. Prefill allocation, fixed
protocol certificates and device-kernel compilation remain separate outcomes.
