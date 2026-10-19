# Configurable Frontier event-ID pools

The caller chooses physical capacity by supplying a list of eligible numeric
IDs. Capacity is independent per source/destination direction. Lists may use any
distinct subset of the eight encodable IDs, 0 through 7. The selected allocator
chooses labels within that pool. Duplicate, negative, and out-of-range values
are rejected.

Combined output without an explicit pool continues to use IDs 0 through 5.
IDs 6 and 7 remain experimental selections because the documented programming
contract reserves them for framework use. Accepting them in the compiler does
not establish that a particular device/runtime deployment leaves them unused.

## Commands

Six-ID combined output:

```bash
pto-test-opt input.pto \
  --pto-frontier-analysis="physical-output=true allocation-ids=0,1,2,3,4,5"
```

Eight-ID combined output:

```bash
pto-test-opt input.pto \
  --pto-frontier-analysis="physical-output=true allocation-ids=0,1,2,3,4,5,6,7"
```

Split output, including saved and reparsed recipes:

```bash
pto-test-opt input.pto \
  --pto-frontier-analysis="deferred-output=true allocation-ids=0,1,2,3,4,5,6,7" \
  -o logical.pto
pto-test-opt logical.pto \
  --pto-frontier-allocate="eligible-ids=0,1,2,3,4,5,6,7" \
  -o physical.pto
```

A two-ID experimental pool can instead be `6,7`. A noncontiguous pool can be
`0,2,5,7`. Supplying an empty pool to the library still means zero available IDs;
it is not promoted to a default. The separate allocation pass continues to
require its explicit pool for nonempty plans. Manual synchronization is preserved.

## Implementation contract

`Shared/EventIdPool.h` separates the eight-value encoding bound from the six-ID
default. Producer lane/search bounds and scalar modular labels admit up to the
encoding bound. Every physical assignment must still fit the actual supplied
pool. A mathematical certificate describing seven lanes therefore does not
allow a six-entry pool to allocate those lanes.

Finite, cyclic, regional, executed-counter, prefix, completion, and saved-family
paths accept the same native ID range. Matching, consumption-before-republication,
cut availability, payload order, and rollback checks continue to apply. Increasing
the pool does not introduce a new protocol or change a missing recognition/export
capability into a proven one. Automatic completion-child discovery still uses
its established six-ID within-visit qualification before the final pool is known.
An eight-ID parent can rename those qualified children into a larger or high-ID
pool, but does not requalify an otherwise unavailable seven/eight-lane child.
This is a remaining provider capability limit, separate from pool validation.

## Qualification

The incremental native compiler and Python binding build passes. All nine
focused tests pass, including actual seven/eight-wide arithmetic certificates,
high-ID cyclic reuse, storage-lane selectors, counter lowering, saved/combined
output and C++ generation. Seven/eight outstanding terminal publications still
refuse a six-ID pool and succeed with eight; the logical traces are unchanged.

The complete selected synchronization gate initially reports 341/347 passing:
three inherited legacy InsertSync expectation failures and three tests retaining
the old six-ID limit. After replacing those obsolete expectations with positive
seven/eight-width checks and negative ID-8/nine-width checks, all three affected
tests pass. The resulting aggregate is 344/347, with no timeout. This is a full
run plus targeted reruns, not a second fresh full run.

All 49 pinned primary PyPTO-lib modules compile with both six- and eight-ID
combined output. The six-ID outputs are byte-identical to the immediate
`c44c1df43` baseline, and every eight-ID output is identical to its six-ID output.
Nine selected kernels also have byte-identical six/eight physical output after
canonicalization/CSE. Three isolated interleaved timing pairs per kernel show no
material overall compile-time change; no speedup is claimed. The projection
fallbacks retain their previous plans: pool size alone does not supply a missing
recognition or export certificate.

The changed-code prefilter's one line-length finding was fixed. Its remaining
24 brace findings are manually verified false positives: every reported control
body has braces, including multiline conditions and conditions with nested calls.
The native build has no new warning in changed files; the three warnings in
untouched files are recorded in the local build log.

See the [machine-readable receipt](frontier-event-id-pools-results.json) for
input/compiler hashes, outcomes and paired measurements. Detailed commands and
raw receipts remain under `.local/event-id-pool/`. No device compiler or device
execution was run; compiler acceptance does not qualify runtime availability of
reserved IDs 6 and 7.
