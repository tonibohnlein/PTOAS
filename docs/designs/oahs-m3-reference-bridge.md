# M3b: pinned causal-interface and paired-order verification bridge

## Source and scope

This is a TEST-ONLY bridge to the actual user-supplied v0.8 reference source, not a
second production planner and not a fresh imitation of its semantics.
`test/oahs/reference/vendor/v08/PROVENANCE.json` pins 29 unchanged reference files:
the causal interface, paired order interface, original full-history oracle,
program/certificate reader, and 12 emitted schemas with their certificates.
The source archive is `synchronization_theory_acm_v0_8.zip` (SHA-256
`7d25fe28f520dc3b541a64a72801cbbcdbbeb9418c8fc577653e6ac232c8e319`).
Its PDF is byte-identical to `main(20260915-194549).pdf` (SHA-256
`eeefd26f5c851af2e179cbe1ee2146c219b79669722d90931959e56dcbb0a258`).
Later Library drafts are not silently substituted. Vendored files retain their
original contents; the bridge driver and interchange adapter are new code.

The contract shared by the first adapter is exact finite ordinary cells,
issue-ordered asynchronous engines, directed consuming SET/WAIT, local prefix
fences, fresh entry, and empty event state at closed exit. The adapter refuses
synchronous lanes, actual ALL commands, DrainAllAtReturn, exclusive-resource,
visibility, authored internal phase, and compound operation contracts. Mere
availability of ALL is not a problem when the actual tested plan uses none.
Missing adapters and resource interruption are never reported as successful
verification. In particular, this does not qualify the native A3 profile.

## Actual compiler boundary

`oahs-reference-driver` links the same production `Plan.cpp` as all core tests.
A versioned numeric interchange supplies original operations/effects, the
Region tree, target keys/reservations, and either an actual command population
or a request to run `construct`. The driver exports the ACTUAL commands,
construction status, certified M1 snapshots and residual/protocol results.
`reference/bridge.py` creates the independent reference control graph directly
from the Region tree; it does not reuse C++ Control.h edges or the compact state
transfer. Every branch alternative, counted-loop bypass, and while before/exit
edge is retained. This adapter has the same conservative control choices as the
current core, not an independent proof of native control extraction.

The reference collector holds sets of WHOLE causal-interface states at original
static sites and iterates to successor closure. It uses the archived primitive
extension/projection and signature minimization code unchanged. There is no
maximum loop trip count. An optional `--max-states` is only an external test
resource guard; exhaustion is `inconclusive` with exit status 2. It is not enabled
by default and is not part of production analysis or acceptance.

## Proof-credit comparison

The compact implementation indexes memory completion by original physical phase,
whereas the reference safety domain indexes histories by (cell, engine, mode).
For this differential check ONLY, each phase additionally reads one private ghost
cell. No operation writes that cell, so this instrumentation creates no memory
conflict and adds no causal edge between original payloads. It allows reference
histories of successive visits of that phase to be inspected independently. The
paired-order check uses the original uninstrumented effects.

At incoming, before-issue, and outgoing physical cuts, each reached reference
state must satisfy these implications for compact facts:

* If phase a is absent from pending[q], every tagged historical completion of a
  reaches the current A_q port (an empty family is vacuous).
* Compact occupancy contains the reference empty/full value.
* A compact latest-consumption knowledge bit q requires D_e -> A_q.
* A must-valid compact receipt requires a live S_e.
* Absence of a from a receipt remainder requires all of a's tagged completion
  signatures to contain S_e.
* A carried latest-consumption bit f requires D_f -> S_e.

These checks include initial consumption anchors, publication freshness, and
new-consumption invalidation. They check safe reference prefixes even when a
later endpoint or payload fails. They do not continue an invalid candidate as an
executable program. At the plan level:

    compact verified => exact reference accepted

A counterexample is a test failure. Reference acceptance with compact refusal is
reported as a precision difference, not automatically a correctness defect. This
implication is tested, not proved universally by the harness. The complete
abstraction/simulation theorem for (N,U,O,K,J) remains a separate obligation.

## Paired order and certificates

Every reference-safe plan is also checked with the archived OrderedInterface.
This compares original issue AND completion vertices against the required
issue/conflict graph. `order_exact=false` with kind `extra_order` is a safe but
coarser plan, not a safety failure. The bridge does not conflate command counts,
completion at one consumer, or absence of a few forbidden edges with complete
payload-order equality. Whole-program fresh-entry order equality does not certify
arbitrary open-region replacement.

The self-test reruns the original certificate reader on all 12 stored artifacts:
544 stored state/site pairs and 558 successor-closure obligations. No synthesis
or SMT solving is rerun. These artifacts include richer guarded reference
schemas than the current C++ emitter supports; closure verification is not a
claim that C++ has generated those schemas. The adapter's own generated-program
comparisons separately exercise current C++ construction.

Eight negative tests check corrupted compact pending/receipt/consumption facts,
wrong occupancy, stale carried acknowledgments, reference-source integrity,
missing certificate entry, and an unobservable event-state guard. A separate
160-word finite test calls the archived full-history oracle: 6,400 command
verdicts and 3,000 residual sets are compared. Unsafe prefixes in that local
algebra test are reported before continuing, never treated as accepted code.

## Running and interpretation

Python 3.10 or newer is REQUIRED by the standalone reference-test target; missing
Python cannot silently remove this gate. Production compilation has no Python
or solver dependency. Assertion-disabled Python (-O/PYTHONOPTIMIZE) is rejected.

```
python3 test/oahs/reference/bridge.py \
  --driver /path/to/build/oahs-reference-driver --self-test --output report.json

python3 test/oahs/reference/bridge.py \
  --driver /path/to/build/oahs-reference-driver \
  --case test/oahs/reference/cases/payload_free_relay.json
```

Self-tests compare fixed candidates, real constructor outputs, and mutations.
Counts are finite test populations, not a proof-assistant result, coverage over
all source programs, native compilation, or device validation. The collecting
reference can have a large state space: it remains validation infrastructure,
not an unconditional production fallback. C++ and the reference still depend on
matching operation/primitive contracts; a shared wrong hardware premise cannot
be discovered by this comparison alone.
