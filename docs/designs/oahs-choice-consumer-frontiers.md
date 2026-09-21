# Original-choice consumer frontiers

## Problem and selected behavior

A source pipeline can produce A and then unrelated B before a conditional
consumer of A. When ordinary source selection cannot relate an early source
handle to the branch-local deadline, a common-cut transfer can publish the
broader A+B prefix. That unnecessarily gates A's consumer on B completion.

The constructor now uses an existing original-choice boundary when **both**
arms start with payloads on the same observer pipeline requiring the motivating
completion. It publishes from the earliest sufficient saved source in the
incoming straight corridor and acquires once at that boundary. Ordinary residual
construction still realizes B's readiness at its own deadline.

This is a general control/physical-access rule, not a matrix-opcode or KDA rule.
It does not change payloads, add guards, create observations or select fixed
coverage in the first pass. Original control provides candidate frontiers;
current source-time causal state and actual selected keys determine realization.

## Qualification and resource interface

`Control` indexes candidate frontiers by their first consumers. Each original
choice must have two straight arm prefixes, unique analytical command words,
and a first payload on the same observer. Empty arms, nested control, an earlier
payload on another pipeline, and ambiguous/shared words retain ordinary fallback.
Preparation scans those prefixes once per pipeline, not once per requirement.

At selection:

- Every first consumer physically conflicts with every motivating requirement.
- No crossed existing outward publication from the observer or ALL boundary may
  acquire an extra prerequisite. Deadline words are included in this check.
- The earlier source is reachable, current-version and in the straight incoming
  corridor. Claimed history must remain fresh up to the choice.
- A direct key must have actual source-time publication credit and a clear use
  interval. Closed/recurring reservations remain excluded.
- A distinct helper-free key must be available at the original broader boundary.
  This conservative check prevents the known one-key split from starving the
  subsequent ordinary receipt or forcing a broader return. It is not a complete
  allocation theorem; unsupported capacity retains fallback.
- One exact staged proposal checks all represented protocol occurrences and key
  neighbors. Pending payload requirements remain explicit; no desired return is
  granted as credit. Rejection changes no live ledger or reservations.

The already checked endpoints are committed through the existing frontier path.
There is no completed-plan deletion search, changed-plan omission trial, or
helper-augmented route search in this mechanism. Existing mandatory final checks
and native reconstruction remain authoritative. A staged protocol check proves
safety, not general ordering optimality; ordering inclusion is measured separately.

`choiceConsumerFrontiers` defaults to true. The native test driver exposes
`--no-choice-consumer-frontiers` as an ablation. The separate unfinished
`firstWriteConsumers` experiment remains disabled.

## Validation

- All 25 portable suites pass. The focused suite checks acyclic and repeated
  choices, zero/one/two/four trips and every branch sequence, complete order-set
  inclusion, forbidden later-bank completion, actual rearming and missing receipt.
- Negative tests cover regenerated sources, independent first consumers, empty
  arms, outward publication, earlier other-pipeline payloads, unavailable direct
  directions and tight one-key capacity.
- Both native test executables pass. A native MTE2→V fixture verifies that the
  mechanism is not confined to the KDA MTE1→M path, including emitted reconstruction.
- KDA selects two transfers. For the recorded 1,727-payload finite trace, complete
  explicit payload relations decrease 5,953,709→5,953,677: **32 removed, none added**.
  Event pairs increase 1,453→1,457; barrier counts stay fixed. AIV output is
  byte-identical. Payload/control identity passes reconstruction.
- On 88 archived corpus modules / 97 functions, both arms pass construction and
  reconstruction and every plan is identical. Total selected updates (1,784) and
  replay evaluations (3,262,799) are unchanged; no corpus case selects the rule.
  Choice preparation visits 36,106 sites in either arm: the switch disables
  selection, not construction of the control index.

KDA cost: original graph stays at 679 sites, choice preparation visits 140 sites,
two staged checks evaluate 5,144 sites, selected updates increase 183→187, and
replay evaluates 184,538→189,813 sites. These are work counters, not a matched
host-time result. The two full checks remain a measured cost to reduce if this
qualification is expanded.

The finite graph treats queue operations as opaque payloads and does not model
peer progress, hidden queue lowering or target ACC completion. It supplies
explicit-command ordering and event checks; native verification supplies admitted
physical requirement checking. No coupled device correctness or latency claim
follows from these local results. Existing InsertSync already separates the first
prologue readiness; this does not establish superiority over existing.

## Reproduction

See [KDA local evidence](../../test/benchmarks/kda_projection/LOCAL_OPPORTUNITIES.md)
for original input provenance and local artifact paths. With a current native
selected-test driver, construct the same normalized input with and without
`--no-choice-consumer-frontiers`. Keep `firstWriteConsumers` disabled.

The finite comparison uses:

```sh
PYTHONPATH=test/oahs python3 test/benchmarks/kda_projection/compare_order.py \
  CONTROL.pto CANDIDATE.pto --function kda_projection_aic \
  --bindings '{"%arg16":67,"%arg17":67,"%arg18":67,"%arg19":0,"%arg20":5}' \
  --pipes '{"tload":"MTE2","textract":"MTE1","tmov":"MTE1","tmatmul":"M","tmatmul.acc":"M","tpush":"FIX","tinsert":"FIX"}' \
  --output order.json --require-no-added
```

The corpus runner records source and driver hashes, applies archived scalar
load/store syntax migration identically, and retains both plans and logs. Use
`--jobs 2` only when no other resource-intensive local work is running.

## Remaining quality leads

The checked removal of all KDA MTE1 barriers removes no finite payload relations:
it is a possible instruction-overhead opportunity, not demonstrated extra
parallelism. MTE2/M barrier deletion fails physical completion checks. Useful next
work is constructive complete MAT-cycle support or first-conflicting-write
placement with occurrence-qualified rearming, not blanket fence deletion.
