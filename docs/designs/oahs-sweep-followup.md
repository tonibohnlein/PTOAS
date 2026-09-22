# Broad sweep follow-up: compatibility and independent workloads

The device sweep compares existing and handoff at `7c48f4ab3`. Its timing
results do not validate the uncommitted first-write/final-read experiments.
Local diagnostic artifacts: `../sweep-followup-work/`.

## Priority and evidence requirements

1. Reproduce one declared-effect/translation failure, one unrepresented-effect
   failure, and the exact `hc_head_reduce` resource failure. Fix effect contracts
   at their owning interface; do not bypass semantic checks.
2. Continue bounded KDA rearming diagnosis, keeping the early final-read source
   and acknowledgment. Require complete ordering comparison before timing.
3. Compare RMSNorm and hc_pre using the sweep's prepared inputs, not just kernel
   names. Use them to test generality independently of KDA.
4. Attribute decode-compressor ratio-4 serialization. Its golden validates only
   the finite 25% of `kv`; the other two outputs are checked. Preserve that
   limitation and do not treat timing as a complete correctness qualification.

## Initial reproductions (local working-tree compiler)

- A minimal `scalar-divisor.pto` initializes a vector tile and applies `tdivs`
  with an f32 constant. Handoff rejects `declared effect missing from physical
  translation`; existing InsertSync accepts it. `TDivSOp::getEffects` declares
  a default-resource memory read on the scalar operand. The translator has no
  physical buffer for that value. This reproduces the error category, not yet
  the exact sweep input. Qualify immediate-scalar versus memory-backed scalar
  effects at the operation interface; do not ignore arbitrary unmapped effects.
- Checked-in A3 `DeepseekV4DecodeA3/kernels/qk_pv.pto` rejects
  `unrepresented operation effects` at `pto.reserve_buffer`, line 29. This is
  an original sample, not a sweep-prepared module. The reserved-buffer/frontend
  preparation boundary must be checked on the exact sweep artifact before
  attributing its failure to this operation.
- Checked-in A3 `hc_head_reduce` constructs and reconstructs successfully.
  Therefore it does not reproduce the reported resource failure. A5 variants
  are not substitutes: the current native contract rejects their target.

## KDA diagnostic correction

The outer cut-330 error concealed a failed replay during return restoration.
A linked private-state probe reports:

```
current consumer: MAT -> LEFT textract, operation 90
outer publication word: 330, occurrences 330 and 669
underlying failure: latest consumption does not precede this publication
failed replay occurrence: 645
endpoint 160: publication cut 250, MTE1 -> M, physical key 0
```

The failed replay leaves later cached states unreachable. Those states must not
be interpreted as evidence that all candidate keys are physically unavailable.
The retained-binding path now preserves an already-recorded replay failure
instead of overwriting it with a generic resource error. This changes diagnosis,
not event selection or causal semantics. The missing consumption path still
requires attribution and repair; no new KDA success is claimed.

## Independent local samples (not measured sweep inputs)

Both arms construct the checked-in A3 RMSNorm, hc_pre_rms and hc_pre_seed samples.
Static synchronization command counts (including barriers):

| Sample | Existing | Handoff |
|---|---:|---:|
| rms_norm | 62 | 48 |
| hc_pre_rms | 34 | 25 |
| hc_pre_seed | 7 | 6 |

These counts do not establish better ordering or explain device latency.
In hc_pre_rms, existing publishes separate V->MTE2 releases within the reader
body; handoff uses a common V->MTE2 exchange and MTE2 barrier before the next
batch of four loads. This is a concrete placement lead: check which independent
bank can be refilled after its own final read instead of the later V prefix.
It is not a justification to delete the MTE2 barrier. First prove the actual
writer/read/reuse paths and compare complete payload relations on matched inputs.

## Exact sweep inputs: reproduced

The downloaded archive now passes gzip integrity. Its locally computed SHA-256
is `2846b12187be19e47d7e4388e547a707d9ae21d06241bf05b0322966cfad01ca`.
The detached checksum was not present when checked, so this is not an external
checksum comparison. The 447 extracted files covered by the internal hash
manifest all match, including the three prepared inputs below. Both arms have
identical prepared-input hashes for each reproduction.
The local existing-pass driver succeeds on all three same prepared inputs;
handoff fails as recorded below. This is pass-level reproduction, not a new
device or end-to-end code-generation run.

| dspark case/function | Local reproduction | Attribution |
|---|---|---|
| gate / route_sort | Same error, line 75 | `tmrgsort` declares a write to its `vector<4xi16>` executed operand; no physical buffer represents that operand. |
| decode_sparse_attn_csa / qk_pv | Same error, line 34 | `set_ffts` has unscoped read/write effects and no admitted configuration/protocol contract. This supersedes the reserve-buffer lead from the original sample. |
| hc_head / hc_head_reduce | Same resource error at cut 209; 43 updates, 37,881 replay evaluations | MTE2 readiness needed by V `trowexpandmul` at cut 228, cells 41 and 42. |

For hc_head_reduce the linked state probe finds:

| MTE2->V physical key | State at publication cut 209 | Constraint |
|---|---|---|
| 0, 1 | Empty, consumption not known to publisher | Recurring reservations; require actual return support. |
| 2 | Empty, consumption known | Owning publication at 227 overlaps the proposed interval to 228. |
| 3 | Empty, consumption known; interval clear | Reserved as a closed role from earlier uses (49/52 and 75/76); ordinary allocation excludes it. |
| 4, 5 | Empty, consumption not known to publisher | Previous consumers lie in alternative paths; current acknowledgment placement declines. |

Key 3 is a concrete ownership-lifetime opportunity, not proof that ignoring
`closedKeys` is safe. Check its complete owning interval and neighboring uses
before borrowing; retain actual rearming on every repeated/alternative path.
This is an independent workload for the reservation-lifetime mechanism explored
for KDA. Do not globally unreserve closed roles.

A private diagnostic appending the key-3 SET at 209 and WAIT at 228 analyzes
the complete original graph with zero event-protocol obligations, just like the
unchanged partial ledger. Uncovered payload requirements decrease from 130 to
122. This demonstrates feasible event use in that staged fragment, not a
completed construction or a certificate for arbitrary later insertions.

The diagnostic-only allocation change builds; the complete native selected
suite passes. KDA now reports the underlying failure directly: SelectedUpdate,
cut 645, `latest consumption does not precede this publication`. Its endpoints,
84 updates and 176,894 replay evaluations are unchanged.

## Exact workload inspection started

The archived dspark RMSNorm and hc_pre_rms retain a common V->MTE2 exchange
before their first load and an MTE2 barrier; hc_pre also has conditional shape
paths. These differ from the older straight sample, so the old sample's four-load
placement must not be used as an exact device attribution. Compare the branch
participation, per-bank last reads and next writes before changing either plan.

Decode compressor ratio 4 contains four changed components: kv_score_proj,
scatter_softmax_pool, rmsnorm_rope_cache_write, and compress_state_commit.
Its full-dispatch slowdown cannot yet be attributed to a single one. The saved
`exact-static-inventory.json` separates their command populations; it is not a
critical-path or device-time breakdown. The finite-golden limitation above
continues to apply. Obtain component timing/ordering evidence before selecting
a synchronization change.

## Implementation checkpoint (2026-09-22)

### Shared semantic admission

A3 `tmrgsort` with `exhausted=false` no longer declares a write to the
nonbuffer executed-result operand. The pinned PTO-ISA
`0c112d61f41342bd0867ce1080c29f1590d72484` `GetExhaustedData<false>` does
nothing. Exhaustion-enabled and A5 effects remain conservative. Exact sweep
`route_sort` now constructs and reconstructs successfully.

`set_ffts` now has a shared configuration classification. It requires an
unconditional entry setup before issued work; repeated original setups must
use the identical SSA address. Configuration remains in its original position
and supplies no local completion, token, peer or visibility credit. Late
initialization and changed addresses reject. Native semantic tests cover both
positive and negative cases. Exact `qk_pv` passes this admission and next rejects
`pto.sync.wait <PIPE_MTE2>` at line 56 (AIC), and another authored protocol at
line 527 (AIV). This is an additional protocol-contract boundary, not successful
end-to-end qk_pv compilation.

### Closed reservation reuse

The ordinary allocator can borrow an inactive closed role while retaining its
ownership. Source-time consumption credit, complete original-control interval
matching and consumption-before-next-publication are required. Dormant helpers
are ownership promises: when needed, their exact restoration and the new pair
are checked together in a private ledger before installation. Restoration work
is counted separately. A repeated-entry portable case covers ownership,
overlapping uses, missing return support and excluded entry reservations.

On exact `hc_head_reduce`, key 3 is successfully borrowed for 209→228. Two
dormant helpers at 117 and 161 are restored. Construction advances from cut
209 to a second shortage at 215→230; it is **not yet complete**. At 215 key 3
is now full until 228; key 2 has an owning use at 227; keys 0/1 and 4/5 lack
source consumption knowledge. The latter have alternative-path consumers.
Do not treat the first successful borrow as a general capacity proof.
Artifacts: `../sweep-followup-work/head-second.{log,pto}` and
`head-second-probe.log`.

### KDA rearming

Dormant acknowledgment keys remain owned until restored. Ordinary allocation
must not steal their temporarily inactive physical keys. In addition, selecting
a reverse key requires support for its next already-selected publication, not
just availability at the new helper's source. A linked negative demonstrates
that a locally publishable key can invalidate that neighboring publication.

The exact opt-in first-write + final-read KDA now constructs and reconstructs:
174 updates, 211,927 replay evaluations, one qualified final-read source and
two publications. This resolves the prior cut-645 and cut-428 event failures
without discarding the early source or its acknowledgment. However the finite
comparison against the committed choice-isolated plan **fails the quality
gate**: 1,784 relations removed and 706 added over 1,727 payload occurrences.
The experiment remains disabled; no new device comparison is justified yet.
Artifacts: `../sweep-followup-work/kda-third.{pto,log}` and
`kda-third-order.{json,log}`. Counts are explicit local-command ordering, not
a coupled queue or memory proof.

### Independent workloads

Initial explicit-command comparisons of exact archived plans, for a full-tile
block (128 token extent, block 0), find 1,792 added / zero removed relations
in dspark RMSNorm (422 payloads), and 620 added / zero removed in hc_pre_rms
(133 payloads). These are illustrative launch bindings, not a sweep-wide
dynamic count or a proof that existing has all necessary memory order.
The recorded script and bindings are `../sweep-followup-work/compare-independent.py`
and `independent-order.json`.

RMSNorm reuses input bank 14624 as reduction scratch, and input bank 4096 as
another reduction scratch. A release immediately after conversion is therefore
too early: final scratch use must be included. hc_pre's accumulator initialization
at 32832 and its first input load at 65696 are distinct physical ranges; the
common V→MTE2 entry exchange imports that initialization into the load. These
are concrete leads for physical last-use and conditional first-write analysis,
not permission to remove a barrier by opcode name.

Compressor dispatch includes four changed components. Its orchestration makes
both state commit and RMSNorm/rope/cache-write depend on the pool task; component
latency attribution is needed before optimizing that dispatch. Preserve the
reported finite-25% kv validation limitation (75% NaN golden). No new compressor
correctness or speedup claim is made here.

### Final host validation for this checkpoint

All 25 portable suites pass; the additional staged-restoration positive and
rollback negative run in `oahs_selected_update`. The native semantic and native
selected suites pass. The rebuilt default corpus constructs and reconstructs
88/88 inputs: 83 byte-identical plans, four attention plans differing only in
event-key numbers (same command and payload positions), and one RMSNorm plan
with extra helpers. Its tested finite trace retains exactly 1,968 relations
(32 payloads; zero additions or removals). This is neither a general ordering
equivalence proof for all launch values nor a performance improvement.

Evidence: `../sweep-followup-work/default-corpus/summary.json`,
`core-third.log`, `core-last.log`, `native-second.log`, `selected-third.log`,
and `corpus-rms-order.json`. The exact dspark RMSNorm sweep input is distinct
from that smaller corpus RMSNorm input. No device runs were performed locally.

### Configuration interface amendment

Removed the shared `dyn_cast<SetFFTsOp>` configuration classifier. The operation
now implements `SyncConfigurationOpInterface::getSyncConfigurationGap()`, and
shared translation queries that interface. The constructor has no configuration
opcode cases. The lifetime restrictions and side-effect preservation are
unchanged. Tests additionally cover the interface record's empty phase list,
repeated same-address setup inside a loop, and rejection when entry setup is
missing. See [the shared contract](oahs-shared-semantics.md#preserved-configuration).

Validation of the interface amendment: rebuilt PTOIR and dependent
InsertSync/OAHS native test executables; native semantic and selected suites
both exit 0. Exact route_sort constructs/reconstructs with a byte-identical
plan to the preceding semantic fix. Exact qk_pv still rejects the separate
authored protocols at lines 56 and 527, after passing configuration admission.
Logs: `../sweep-followup-work/config-interface-{build,native,selected,route,qkpv}.log`.
No constructor-policy change, broad corpus rerun or device run in this amendment.
