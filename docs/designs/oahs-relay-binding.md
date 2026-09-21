# Check relay key feasibility before placement ranking

2026-09-21, based on `91fc0e728`. This extends the bounded
[relay-selection correction](oahs-relay-selection.md) without changing its
ordering comparison or adding a mode.

## Reproduced admission failure

The linked ordinary constructor considers two directions from FIX to MTE2:
FIX→M→MTE2 and FIX→MTE1→MTE2. The existing ranking chooses the late M relay.
The fixture then occupies its only key until a fixed acquisition at exit.
The other route has usable keys at its actual source and forwarding gaps.

Previously, ranking selected M before inspecting physical state. Its binding
failed, so construction fell back to the same ordinary route and ultimately
refused with `no reusable key or nonrecursive consumption acknowledgment`.
Both first-leg and second-leg occupancy reproduce this through the production
constructor linked against the frozen pre-change core.

This is an admission limitation, not unsafe output. Nor is the alternate route
an ordering improvement over the unblocked preferred route: those placements
can have incomparable order. The practical improvement is that the admitted
key state no longer hides an available two-leg realization.

## Bounded change

Move the existing `keyFor` query before ranking each candidate, after its exact
forwarding gap has been chosen (including the earlier-receipt safeguard).
Keep its original checks:

- direction eligibility and exclusion of closed keys;
- actual source-time emptiness and consumption knowledge;
- no interfering selected use in the proposed interval;
- no same-key use in the existing source word when prepending at word start.

The first leg queries the saved publication's post-word state. An early second
leg queries the incoming word state; a late appended second leg queries the
post-word state. Store both physical key identities with the winning middle
engine/gap. Candidate queries do not change the ledger or replay state, so the
chosen identities refer to the same selected version when staged.

A failed query means no certificate from this bounded interface, not universal
impossibility. Only the winner is materialized and receives a full staged
protocol check. Failure of that check still falls back normally; the change
does not retry arbitrary candidates or search helper-augmented protocols.
The two legs have distinct directions and cannot reserve the same physical
directional key. The first receipt still supplies no prospective credit during
the second-leg probe: support available only after applying that receipt remains
a separate extension.

## Linked boundaries

The relay suite disables omission/helper trials. Each positive cold-checks the
selected plan and compares its full payload-order set with an explicit legal
alternative checked by the independent graph oracle.

- The unblocked control still chooses M, demonstrating the original preference.
- Occupying either preferred leg selects MTE1 with one staged solve, two added
  handoffs, and the fixed exchange intact (three total handoffs).
- Consuming the preferred key without returning consumption knowledge also
  selects the alternative, for either leg.
- A first-leg key that is empty at the source but used later in its proposed
  interval does not qualify.
- A second-leg key with an actual preceding return can be reused on the chosen
  alternate route. Deleting that return preserves memory coverage, balance and
  acyclicity but loses rearming.
- Removing a required alternate leg loses memory coverage. Removing its entire
  direction leaves no staged relay attempt; ordinary construction still refuses.

## Evidence and limits

Frozen baseline core/driver, pre-change reproductions, build/test logs and corpus
outputs are retained under `../relay-binding-work/`. The two baseline occupancy
reproductions exit 1; the new construction succeeds. These fixtures are portable
contracts, not native FIFO or device validation.

- All 24 portable suites, the final focused relay test and both native test
  executables pass. Six new binding cases pass, each with the missing-direction
  refusal control, plus missing-memory/rearming mutations.
- The row48/49 native attention regression passes all twelve finite cases and
  mutations. Both AIC plans retain two staged relay checks / 1,318 evaluations;
  their native-length order reductions remain 80/160 relations with none added.
- The paired 88-module / 97-function corpus constructs and reconstructs with
  all 88 selected plans byte-identical to `91fc0e728`. No changed native workload
  is available to justify another device task.
- Aggregate replay stays 3,262,799, staged relay solves stay four / 2,636 site
  evaluations, and relay-preparation visits stay 2,040. Key queries increase
  823→835: the two affected attention AIC functions each increase 34→40.

Commands use the existing core/native builds with `--parallel 2`, followed by
`ctest --test-dir ../oahs-m1-core-build --parallel 2 --output-on-failure`, the
two native executables and the committed attention runner. `evaluate.py` retains
the serial corpus commands and exact input/output hashes; `summary.json` records
the final counts and log identities. Baseline driver/core are frozen separately
from the updated shared build trees.

Key query work is charged to the existing `keyQueries` counter. More than one
candidate can now perform these queries; this is not a compilation speedup.
There is no new whole-graph analysis per candidate and no general relay-order
optimality claim.
