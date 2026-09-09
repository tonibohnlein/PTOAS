# Milestone one: independent logical construction

Milestone one passed native acceptance and was accepted by the software
architect, compiler, and algorithms/performance reviewers. The remaining three
milestones have not started. Device qualification is outstanding.

The new opt-in `--insert-sync-planner=logical` path constructs from unsynchronized
PTO without running existing insertion, lifecycle selection, or the handoff
refiner. The default remains `existing`. `logical-or-existing` discards an
unsupported candidate and runs existing construction on the untouched body.

## Unchanged native inputs

Each arm independently compiles the same frozen input with the existing
`assume-disjoint-arguments` benchmark contract. Payload operations, allocation,
views, ABI and executed payload hashes match. “Revised” uses the established
staged/MMAD/shared-generation flags; “refiner” additionally enables the previous
post-emission handoff experiment. Neither is the historical original InsertSync
arm from the older device campaigns.

Static sites are separate mechanisms:

| Fixture | Arm | Sets | Waits | Named barriers | PIPE_ALL |
| --- | --- | ---: | ---: | --- | ---: |
| One buffer | Revised | 6 | 6 | none | 0 |
| One buffer | Refiner | 6 | 6 | none | 0 |
| One buffer | Logical constructor | 4 | 5 | none | 0 |
| Online softmax | Revised | 20 | 20 | V: 20 | 1 |
| Online softmax | Refiner | 12 | 12 | V: 20 | 1 |
| Online softmax | Logical constructor | 13 | 13 | V: 20 | 0 |

One buffer has alternative guarded acquisition sites, so its unequal static
set/wait inventory is not an unmatched runtime event. Every observed execution
balances and drains its actual tokens. The independent symbolic reconstruction
also checks complete publication/acquisition matching.

Online softmax at bound 16:

| Arm | Executed sets | Executed waits | Executed V barriers | Executed PIPE_ALL | Scalar/control operations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Revised | 188 | 188 | 258 | 1 | 424 |
| Refiner | 96 | 96 | 258 | 1 | 424 |
| Logical constructor | 94 | 94 | 257 | 0 | 680 |

At bounds 0/1, refiner uses six pairs, three V barriers and one PIPE_ALL;
the constructor uses seven pairs, two V barriers and no PIPE_ALL. At bound 2,
pairs fall from 12 to 10. These are separate tradeoffs, not a combined score.
The scalar/control count includes guard arithmetic and structured control; it
is not a device instruction count or a timing model.

All four softmax boundary observations (0, 1, 2, 16) are supported and have
identical observed cross-lane completion prefixes at payload operations. These
finite observations challenge the claimed quality; they do not replace symbolic
requirement/event checking or establish equal device runtime.

The final single-process PTO emission took approximately 0.49 seconds for
revised, 0.56 seconds for refiner and 7.69 seconds for logical construction.
The constructor uses 97.1 million checked work units. Its default was explicitly
raised from 8 million to 128 million (16×), after reducing the initial working
run from approximately 457.5 million units and 30 seconds. This remains a
significant compile-cost limitation. Work accounting is not a hard time or
memory cap. The final run also emits C++ successfully; no device compiler or
runtime was executed.

## Validation and proof boundaries

- 45 native relation/reference checks passed, including integer parity,
  continued source-scoped completion, guarded invocations, plan replacement,
  selected-barrier feedback, and independent finite closure populations.
- 15 occurrence-import cases passed, including 1,936 source-derived softmax
  phase-pair relations before adding the constructor's explicit exit point.
- Five shared-requirement comparisons passed. Complete phase/read/write
  inventories feed an independent known-local-range enumeration. The unchanged
  softmax checks 439 local obligations and 2,025 occurrence relations including
  exit. Overlap and an added reader introduce additional obligations; an
  equivalent predicate preserves them.
- Six emitted-IR corruptions are rejected with the original input unchanged:
  removed wait, widened barrier guard, reordered loads, changed rounding,
  extra allocation, and swapped wait keys with unchanged set/wait counts.
- Sixteen additional executions cover zero/nonzero loops and skipped readers
  in two structured fixtures. Actual event tokens balance and drain; payload
  hashes match the existing arm.
- Explicit zero-budget strict mode fails with `analysis-limit`. Hybrid fallback
  preserves the existing arm's payload, allocations, views, ABI, placements,
  mechanisms and synchronization control.
- The native lit selector/strict/fallback test passes. All 22 mechanism/replay
  accounting tests pass, including the discovered MLIR i1 `true == -1` replay
  bug and signed Boolean comparisons, extensions and min/max.

Requirements are captured before selecting synchronization. New actions use
one logical stream allocator; allocation does not move their boundaries. Fresh
emitted reconstruction reimports effects, verifies original payload and allowed
new operations, recovers actual guards and key matching, checks selected cuts,
and proves the original requirements, exit retirement and consumption before
rearm. It does not accept the planner's proof receipts as a substitute.

The native footprint model remains conservative may-accesses. The reference
comparison excludes GM and unknown geometry from its local-conflict denominator
and records those exclusions. It challenges requirement enumeration and
occurrence calculations while sharing effect extraction; explicit f32/f16
conversion extent checks and future device tests challenge that shared boundary.
This is not a claim of precise native last-value analysis.

## Remaining scope

The two-buffer strict probe rejects a publication domain that needs a
slot-relative last-use guard beyond the current lowering vocabulary. Hybrid
compatibility is not counted as strict coverage. Three-buffer/four-use strict
coverage, Q projection/QK, historical GEMM, physical-section exits, qualified
MMAD discharge, wider operation summaries and full corpus qualification remain
later work. No whole-corpus coverage or new GEMM result is claimed here.

The existing production algorithm remains available. Unsupported semantics,
missing realization, analysis limits and allocation failure remain distinct;
internal construction errors do not authorize fallback.

## Reproduction

[README.md](README.md) lists serial build and acceptance commands. The exact
native/reference driver runs, emitted PTO/C++, commands and diagnostics are in
`/home/toni/work/pypto3_sync_more/insertsync-builds/campaign/logical-plan/`.
The final constructor campaign is `m1-native/final`; requirement comparison is
`m1-requirements/run2`; relation checks are `m1-query-parity/scoped4` and occurrence
checks are `m1-occurrences/run3`.

[M1_RESULTS.json](M1_RESULTS.json) retains compact results, source and binary
hashes, mutation verdicts and reference exclusions. This is a build of the
milestone worktree based on `96b5c282836bc06575f88cb79a0bd46d130bbf6d`, before its
commit; it is not described as an independently rebuilt release. The LLVM source
is pinned to `fa2fd1f75d742fec91ecf40d98c9c7961647ec42`. Builds used at most two
workers in aggregate; all native test contexts disable MLIR multithreading.
