# Key ownership, split returns and continuation-query cost

Follow-up to the review of `6a45e53dd`, implemented after `5e0a72772`.

## One ownership condition

Ordinary binding now uses `availableKey()` to exclude closed roles, recurring
reservations and erased-but-restorable acknowledgment keys. The condition is
shared by ordinary transfers, common and split helpers, both F7 repair scans,
source gaps, final-read sources, alternative/choice/entry frontiers and FIFO
relay legs. Occupancy and source-time consumption credit remain separate checks.

Existing role reuse and explicitly certified reservation borrowing remain
separate entry points. They retain ownership and check restoration with the new
use; they do not globally free a reserved key. No target event pool is enlarged.

## Split returns protect their next use

The old split selector accepted the first reverse key with source-time credit.
A new consumption could invalidate an already-selected later publication of that
same key. The selector now tests candidates in the same deterministic order,
checking the continuation of the exact proposed four-endpoint packet:

```
source gap:         SET forward
consumer deadline: WAIT forward; SET reverse; WAIT reverse
```

The packet is materialized in a private ledger. Its forward receipt may carry
reverse-key consumption knowledge on the next visit; the query sees those actual
proposed commands, not an assumed future repair. Previous-generation credit must
still hold at every source occurrence. The live ledger is unchanged by a refused
candidate. Actual replay and final cold checking validate the committed packet.

This is a structural successor certificate, not a full-analysis solve for every
candidate. It retains the query's conservative direct-return vocabulary. Private
ledger copying and query work still cost construction time.

## Cheap proof that there is no next publication

The ledger maintains a per-directional-key multiset of publication component
indices. Each entry covers the latest reachable occurrence of that publication's
shared word. Insert, erase and restoration update it; private ledgers copy the
index along with their endpoints.

Components are topologically ordered strongly connected components of original
control. If the acquisition has one reachable acyclic component, and both the
latest selected publication and every occurrence of the proposed publication
are at or before it, no successor can encounter another publication. The query
can return without walking the remaining graph. Existing commands at the same
acyclic cut precede the newly appended consumption.

Repeated/shared acquisition contexts, later publication occurrences and cyclic
components retain the full continuation query. A missing active publication
alone is insufficient: the new publication may itself repeat.

The index uses space linear in active publications and logarithmic updates;
each absence probe looks up a key and reads precomputed occurrence spans.
It does not build graph-sized per-key reachability tables. The new
`split_rearming_no_next_use` counter distinguishes indexed answers from
`split_rearming_sites`, which counts the remaining continuation traversal.
This removes the repeated suffix-absence scans in the tested family, not all
possible quadratic construction work.

## Validation

- All 25 portable suites pass.
- Production source-gap and both relay-leg selectors exclude a genuinely
  discharged helper, select an available alternative, and permit its later
  restoration. Source-gap replay is compared against cold state evaluation.
- The production split selector skips a locally publishable lower key with a
  bad successor. The selected packet passes cold checking; the lower-key
  mutation fails consumption rearming. Independent finite graph checks for
  one through four visits isolate that failure while retaining balance,
  acyclicity and memory coverage.
- The existing shared-word constructor regression still exercises `edge()`'s
  split path, including missing support and repeated visits. It now accounts
  for the additional reverse-successor query.
- Index regressions cover insertion, deletion, restoration, private-copy
  isolation, a later shared-word occurrence and recurrence of a new publication.

The ordinary constructor, with default options, reports:

| Successive original choices | Helper queries | Indexed answers | Continuation-site visits |
| ---: | ---: | ---: | ---: |
| 8 | 7 | 7 | 0 |
| 16 | 15 | 15 | 0 |
| 32 | 31 | 31 | 0 |
| 64 | 63 | 63 | 0 |

These are linked constructor operation counts, not timing or a general complexity
bound.

Both native suites pass. The targeted native recheck passes construction and
emitted-plan reconstruction on all 19 previously failing prepared inputs and
four changed partial-attention modules: **23/23 pass, 22 byte-identical to
`5e0a72772`**. Qwen `topk_select` changes only event-ID assignments; normalizing
those numbers gives identical text. Payloads, endpoint positions and command
counts are unchanged. No complete payload-order comparison is claimed for that
renumbering. The full 88-module corpus was not repeated.

On the exact MTP decode-CSA `score_topk_aic`, continuation-query site visits
decrease from 2,520 to 928 (eight indexed answers), with an identical emitted
plan and unchanged 73,939 replay evaluations. This is measured query work,
not a native compile-time or device-latency claim. Split helpers now also pay
for their previously missing reverse-successor checks.

Logs and exact comparison manifests live under
`../sweep-followup-work/key-binding-*`; `check-key-bindings.py` records the exact
input paths, hashes, commands and paired emitted-plan hashes.
No device performance claim or KDA experiment promotion follows from this fix.
