# Frontier runtime architecture validation

This report evaluates the implementation changes through `6fa7701ee` against the
frozen optimized post-instrumentation baseline `63fa019e9`. It does not change
Section 5.4's algorithms, admitted classes, alias policies, hardware protection,
small-count normalization limits or six-ID allocation policy.

## What changed

| Milestone | Commit | Implementation boundary |
| --- | --- | --- |
| 1 | `2350d4be6` | Preserve simultaneous macro envelopes in bounded extrema; compare full unsigned arithmetic coefficient limits. |
| 2 | `63fa019e9` | Session-owned diagnostic ledger, request identities and optional exclusive stage profiling; avoid repeated audit/history snapshots. |
| 3 | `916153769` | Shared reaching descriptor state, target-aware contextual access contracts and reusable scalar facts. |
| 4 | `3bce714d8` | Borrow immutable atom memberships through lifetime scanning, separate occurrence metadata, omit unrequested witnesses and cache original effect relationships. |
| 5 | `f3e93f854` | Share original normalized control with regional views, retain bounded expansion alternatives, defer form recognition and separate root selection from exhaustive certification. |
| 6 | `844e99527` | Scope conversion reuse by arena and ordered coordinate schema, fold local threshold complements, deduplicate syntactic clauses and balance symbolic sequence merges. |
| 7 | `6fa7701ee` | Centralize regional context/frame checks and derived-certificate invalidation; reject malformed adapter exports before construction without losing retained results. |

Every implementation milestone received independent architecture and
correctness/ownership reviews after focused tests and the eleven-input replay.
The three historical arithmetic recognition oracle failures remain reported;
51 of the final 54 focused tests passed. The new scaling test adds independently
bounded source-visit and representation-size checks; it is not a universal
source-size complexity proof. The membership entry reaches
`checkNumericPreflight` in `NumericTemplateDump.cpp`, then
`numericAtomReferenceChecks` and `sharedScanFamilies` in
`NumericAtomReferenceChecks.h`: twelve atom-count/effect-count combinations
compare the borrowed scan with an independently materialized scan, require
`c = 4*U*E`, and check witness counts, retained sharing and protection. This
entry reports aggregate helper process time, not per-combination timings.

## Measurement contract

The paired campaign uses exactly eleven prepared TileLang inputs and 783 pinned
PTO/PyPTO/pypto-lib inputs. Existing success and manual-sync status are eligibility
filters; Frontier recognition is never a filter. Both compiler sets use `-O2`,
C++17 and retained assertions (`-UNDEBUG`). Baseline binaries are immutable
copies; their hashes, original build receipts and runtime dependencies are
pinned. Temporary baseline object files in the shared build directory were
subsequently replaced and are not mistaken for surviving baseline objects.

Each input is measured in fresh processes, with baseline/candidate order
alternating between inputs, one CPU and one resource-intensive worker. Process
watchdogs are 120 seconds and 4 GiB of virtual address space. They are measurement
limits, not mathematical class conditions. A timeout is censored and receives
no exact-duration speedup claim. Process timing has up to 100 ms polling overhead;
subsecond differences are not strong evidence of performance changes.

Production logical compilation, physical allocation and C++ source emission
are separate processes. Exhaustive TileLang regional certification is a separate
workload under both GM alias policies: 88 requests (44 baseline/candidate pairs),
covering 480 original
region/policy identities per compiler. Form recognition, exact mathematical
demands, requested queries/selectors, endpoint preparation, physical IDs and
C++ emission remain distinct achievements. No device compilation or execution
is established here.

Internal `frontier-profile` records report exclusive stage time, cache hits and
available construction counters. All failed attempts before the selected route
remain in the request. Parsing, input extraction outside profiled scopes, I/O,
process startup and reporting contribute to wall time; exclusive internal time
need not sum to it. Missing baseline counters remain unavailable, rather than
being replaced with zero. Cold paired processes and warm in-session cache
checks are reported separately.

## Complexity scope

The explicit scan still visits actual effect–cell incidences `c`. Sharing `U`
atom IDs among `E` effects can reduce backing storage without eliminating
`Theta(U*E)` incidence work. The new consumer avoids repeated full copies and
unrequested witness retention; it does not implement an `O(U+E)` compressed scan.
Uniform-alias pair products can remain dense despite original-pair cache reuse.

Sparse numerical quotient construction preserves encoded distances. Its
parameter is the supplied generator graph, not the size of source expressions
from which that graph must be recovered. The guarded construction retains its
stated quadratic or broader polynomial terms. Bounded lifetime's local demand
window does not imply an unrestricted global selector/query construction.

Local threshold simplification and syntactic clause deduplication remove
specific avoidable conversions. General Boolean-to-DNF conversion may still
expand exponentially. Balanced symbolic merging reduces repeated prefix copies
but cannot make an inherently large final relation small. No new construction
budget or refusal cutoff was added.

These implementation changes improve reuse and representation ownership.
They do not establish a general end-to-end polynomial bound from arbitrary PTO
IR to executable synchronization. Such a bound needs a source grammar, bounded
normalization and generator production, promised regional exports and supported
composition/physical-allocation conditions.

## Persistent GEMM stage diagnosis

For the MayNotAlias persistent-GEMM production root, the paired logical process
measured 30.042 seconds in the baseline and 31.439 seconds in the candidate.
The exclusive allocation-support scope accounts for 29.602 and 31.011 seconds,
respectively. This identifies support construction as the dominant scope in
this pair; it is distinct from the later physical-ID allocation process.
The largest individual producer-attempt scope is
approximately 0.092 and 0.094 seconds. This single pair does not establish a
systematic slowdown or speedup, and does not justify blaming relation
conversion or the periodic reducer for the whole-process duration.

## Additional MayAlias root requests

These are four additional serial profiling requests, separate from the full
MayNotAlias production corpus campaign. Both group-norm requests refuse logical
compilation: baseline 4.607 s, candidate 4.907 s. Their aggregate producer-attempt
scopes are 4.005 and 4.257 s, and selector scopes are 0.396 and 0.401 s.
Both persistent-GEMM MayAlias roots also refuse logical compilation, in 0.302
and 0.303 s. These measurements establish failure costs, not successful
end-to-end coverage under MayAlias.

The largest successful corpus logical-time increase is Qwen3DecodeA3
`down_proj_residual`: 8.513 to 9.819 s; arithmetic-form scopes account for 7.162
and 8.407 s. Qwen3DecodeA5 `out_proj_residual` increases from 8.813 to 9.818 s,
with arithmetic-form scopes of 7.514 and 8.377 s. This localizes measured
increases but does not identify their root cause or establish statistical
significance from one pair. No broad wall-clock speedup is claimed.

## Profiling availability and validation scope

All successful profiled requests emitted valid profiles. The twelve missing
profiles belong exactly to the six timed-out kernels under both binaries;
their processes were killed before a final profile was emitted. No missing
scope is replaced by a measured zero.

The first supplemental ControlFlow harness omitted the `test.eligible_ids`
attribute and therefore could not invoke the physical oracle. Its failure
receipt is retained. The corrected independent harness explicitly requests
IDs 0–5 and verifies both complete fixed-visit graphs and physical reuse.
This harness correction does not modify the measured corpus input or compiler.


## Changed output validation

Of 1,800 common successful stage outputs, 1,710 match under the stated
comparison and 90 differ: three stages for each of 30 kernels. C++ comparisons
are byte-for-byte. Logical and allocation comparisons remove only recognition
dictionaries and unused printer aliases; they retain commands and their
attributes. Differences are therefore reported rather than hidden by a broad
semantic normalizer.

A separate conservative guard checker verifies original operation signatures,
endpoint positions and kinds, pipe pairs, record-renaming consistency and
canonical ordinals. Its threshold proof covers 27 of the 30 kernels. Three
`decode_build_swa_metadata` variants require additional typed integer/select
reasoning and remain unproved by this independent checker. A supplemental
exact interval/select checker also reached its separate 120-second diagnostic
watchdog (approximately 34 MiB RSS) without completing a new proof. This is a
validation limitation, distinct from the six unchanged compiler timeouts in
the paired corpus; no equivalence claim is made for those three guards. A separate complete member-identity check establishes that the 78 syntactic
member differences are consistent constant-tuple renamings in five kernels,
independent of record IDs. The other 25 kernels retain identical member
expression DAGs. It checks every SET/WAIT action, injectivity, structural
endpoint position and canonical ordinal. Combined with record renaming and
guard equality, this validates those changed endpoint descriptions; it is not
a standalone proof of physical storage safety. The independent ControlFlow
graph and six-ID oracle additionally covers its complete fixed-visit graph.


## Paired outcomes

| Inputs | Compiler | Logical | Physical IDs | C++ source | Timeouts |
| --- | --- | --- | --- | --- | --- |
| tilelang | baseline | 9 | 7 | 7 | 0 |
| tilelang | candidate | 9 | 7 | 7 | 0 |
| corpus | baseline | 609 | 584 | 584 | 6 |
| corpus | candidate | 609 | 584 | 584 | 6 |

Production regressions: 0. Exact regional demand losses: 0.

| Alias policy | Exact original regions |
| --- | --- |
| may-not-alias | 238/240 |
| may-alias | 224/240 |

## All eleven TileLang kernels

| Kernel | Exact regions, MayNotAlias | Exact regions, MayAlias | Logical | Allocation | C++ source |
| --- | --- | --- | --- | --- | --- |
| tilelang_gemm | 22/22 | 20/22 | success | success | success |
| pypto_gemm | 17/17 | 15/17 | success | success | success |
| persistent_gemm | 34/34 | 28/34 | success | success | success |
| group_norm | 23/23 | 23/23 | success | compiler_failure | not reached |
| causal_conv1d_prefill | 48/48 | 48/48 | compiler_failure | not reached | not reached |
| causal_conv1d_decode | 7/7 | 7/7 | success | compiler_failure | not reached |
| gdn_chunk_cumsum | 9/9 | 9/9 | success | success | success |
| mhc_head_mix | 8/8 | 8/8 | success | success | success |
| elementwise_pipeline | 10/10 | 10/10 | success | success | success |
| gated_delta_rule | 27/29 | 23/29 | compiler_failure | not reached | not reached |
| lossless_block_cast | 33/33 | 33/33 | success | success | success |

## Process and internal stage costs

All attempted stages include explicit refusals and measurement failures. These totals are not success-only timings.

| Inputs | Compiler | Stage | Attempts | Total seconds | Median seconds |
| --- | --- | --- | --- | --- | --- |
| tilelang | baseline | logical | 11 | 115.694 | 0.206 |
| tilelang | baseline | allocation | 9 | 1.851 | 0.206 |
| tilelang | baseline | cpp | 7 | 1.635 | 0.206 |
| tilelang | candidate | logical | 11 | 117.794 | 0.206 |
| tilelang | candidate | allocation | 9 | 1.952 | 0.206 |
| tilelang | candidate | cpp | 7 | 1.641 | 0.206 |
| corpus | baseline | logical | 783 | 1116.956 | 0.205 |
| corpus | baseline | allocation | 609 | 129.251 | 0.206 |
| corpus | baseline | cpp | 584 | 125.684 | 0.206 |
| corpus | candidate | logical | 783 | 1122.165 | 0.205 |
| corpus | candidate | allocation | 609 | 128.950 | 0.206 |
| corpus | candidate | cpp | 584 | 124.191 | 0.206 |

The machine-readable report contains every case and original region, failure diagnostic, selected route, peak process RSS, raw argv, available internal scope and counter, and common-output comparison. Query/selector snapshots are not treated as absent capabilities unless requested.

## Scaling checks

See the checked-in scaling runner and its independently checked counter families. Descriptor nesting measurements cover stable metadata; branch joins and backedge updates are covered by separate descriptor-state correctness fixtures. Embedded oracle runs report aggregate process time, not invented per-parameter timings.

## Evidence

- Raw serial campaign: `.local/three-paths/review-plan-m8/paired`.
- Verification SHA256: `25e8075554127e229417c5b57d3338cb8af2eb8b135e350c345195478f9d7acd`.
- Compiler source and binary hashes, input manifests, loader configuration and review receipts are retained with the campaign artifacts.
