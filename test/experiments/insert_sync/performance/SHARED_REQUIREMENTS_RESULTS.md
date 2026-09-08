# Shared requirements regression and GEMM boundaries

2026-09-08; commit `16d727c74` plus this worktree change.
Native SHA-256: `cd57d3aaac8af1382488dab5a60a7e05b3550a31f368b3bb11af44390c084e18`.

All 11 unchanged inputs passed in hand-tuned, combined and staged arms:
33 rows, 66 PTO/C++ invocations and 306 scalar replays. Both automatic arms
enable MMAD and buffer generations; staged also defers same-pipe repair.
No input annotations, payload rewrites or stronger alias assumptions were added.

## Static mechanisms

Pairs here mean the equal total set/wait site inventories. Individual directed
domains can have unequal static set/wait site counts because branch participation
differs. Dynamic token matching is checked separately. Barriers are not summed
with pairs or with each other. Combined and staged inventories match.

| Fixture | Hand pairs | Hand named | Hand ALL | New pairs | New named | New ALL |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| One buffer | 6 | 0 | 1 | 6 | 0 | 0 |
| Two buffers | 12 | 0 | 1 | 12 | MTE3=2 | 0 |
| Three buffers | 18 | 0 | 1 | 18 | MTE3=3 | 0 |
| Four-use producer | 24 | 0 | 1 | 24 | MTE3=2 | 0 |
| GEMM | 53 | 0 | 0 | 56 | 0 | 0 |
| TopK | 10 | V=22 | 0 | 15 | MTE3=2, V=14 | 1 |
| Conv2D | 24 | 0 | 0 | 0 | M=8 | 1 |
| FlashAttention | 18 | 0 | 1 | 15 | 0 | 1 |
| Triangular inverse | 278 | V=105 | 0 | 277 | 0 | 1 |
| GDN | 16 | V=6 | 2 | 32 | MTE1=1, V=13 | 2 |
| KDA | 17 | V=7 | 2 | 35 | MTE1=1, MTE2=2, MTE3=1, V=10 | 2 |

The static table matches the previous generation implementation. The change is
that the MTE3 sites in two/three-buffer, four-use and TopK are now conditional
fallbacks. They execute only when the invariant trip bound exceeds the proved
arithmetic range. The ordinary benchmark scenarios execute none of them.

## Executed MTE3 barriers

| Fixture / 16 trips or groups | Before (`16d727c74`) | New | Static fallback sites |
| --- | ---: | ---: | ---: |
| Two buffers | 16 | 0 | 2 |
| Three buffers | 16 | 0 | 3 |
| Four-use producer | 16 | 0 | 2 |
| TopK | 32 | 0 | 2 |

Native dependency traces retain the original repair group through allocation
and codegen. In two-buffer they identify GM WAW from output slot 1 to slot 0
on the carried path and slot 0 to slot 1 on the forward path. In four-use they
identify the last store of one slot episode against the first store of the
other. These are output partitions, not a requirement to drain unrelated local
slots. The existing guarded partition query establishes disjointness on the
bounded path; local ready/release protocols still protect actual slot reuse.

The two-buffer guard retains barriers above 4,194,303 trips. This is a conservative
arithmetic safety bound, not a new caller promise. Constant bounded inputs omit
the repairs without a runtime guard; deliberately overlapping partitions retain
them. TopK demonstrates useful refinement after complete protocol matching fails.

## GEMM readiness and release comparison

The observer compares required cross-lane completion prefixes at each physical
operation, with the exact same replayed payload and physical storage identities.
It detects no differing prefixes in any of the eight recorded scenarios:

| Scenario | Physical operations | Different prefixes | Hand executed pairs | New executed pairs |
| --- | ---: | ---: | ---: | ---: |
| empty_core | 0 | 0 | 6 | 7 |
| one_panel | 28 | 0 | 29 | 31 |
| two_panels | 55 | 0 | 51 | 53 |
| three_panels | 82 | 0 | 73 | 75 |
| outer_reuse | 218 | 0 | 185 | 187 |
| primary_core0 | 4774 | 0 | 3921 | 3923 |
| primary_core23 | 4557 | 0 | 3743 | 3745 |
| secondary_core0 | 2387 | 0 | 1963 | 1965 |

This finds no additional independent physical work blocked by the differing
handoffs on those concrete traces. It does not prove symbolic equivalence for
all inputs or predict runtime. Same-pipe instruction overlap, scalar issue cost,
physical key behavior and device latency are outside this observer.

The three additional static set sites are: a duplicated M→MTE1 release in the
first-matrix branch, an MTE1→MTE2 unused-preload return, and the FIX→M primer.
The duplicated branch does not execute both releases. In nonempty benchmark
launches the automatic accumulator protocol executes two more pairs overall
because it primes and drains its free stream; the manual protocol guards its
first acquisition and last publication. These differences preserve the compared
payload readiness/release boundaries. The empty-core case executes one extra pair.

The final generated C++ is byte-identical to `16d727c74` for GEMM, one-buffer,
Conv2D, FlashAttention, triangular inverse, GDN and KDA. Only two-buffer,
three-buffer, four-use and TopK change C++. A device comparison can reuse an
arm's measurements after confirming device binary identity; matching counts
alone are insufficient.

Device timing for this output is **not run**. The R8 archive measured older
44-pair GEMM output with named barriers; its 617.5 versus 519.2 µs wall times
cannot be assigned to this 56-pair output. Compare hand-tuned, `16d727c74` and
this follow-up on the device, reusing the existing launcher and excluding the
known defective hand-tuned TopK timing arm.

## Remaining fixture gates

The fresh native diagnosis reproduces the Conv2D helper/macro veto and the
FlashAttention, triangular-inverse, GDN and KDA optional effect gates. TopK now
passes the structural opcode gate; its overlapping views prevent exact-slot
lifecycle selection but no longer hide GM partition facts from refinement.
See the [integration and blocker notes](../../../../docs/designs/ptoas-insertsync-shared-requirements.md).

These optimizer gates are separate from the latest device campaign’s Conv2D
launch fault and FlashAttention layout assertion. GDN and KDA already have
coordinated launchers in that campaign. No production synchronization defect
is inferred from an optional importer declining an operation.

## Validation and retained evidence

- Incremental native compiler, optimizer and Python targets: passed.
- 35 focused native lit tests: passed, including guarded/bounded/overlap cases.
- 24 accounting, comparison and boundary mutation tests: passed.
- Frozen regression: 33/33 rows, 66 compiler invocations, 306 scalar replays.
- Eight GEMM boundary comparisons: identical physical payloads; zero differing prefixes.

Full command/output evidence is under:

```text
/home/toni/work/pypto3_sync_more/insertsync-builds/campaign/combined-structure-flow
```

`controls-final/` and `kernels-final/` contain the final campaign.
`native-diagnosis/` contains original repair witnesses and current fixture gates.
`gemm-boundaries-*.json` contains complete observed handoffs and physical payloads.
Earlier `controls/` and `kernels/` are intermediate validation, not the final fingerprint.

[Machine-readable results](SHARED_REQUIREMENTS_RESULTS.json) retain all three
arms, per-scenario counts, output hashes and the boundary summary.
