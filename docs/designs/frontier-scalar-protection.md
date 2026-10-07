# Same-scalar storage protection

Frontier follows the same hardware rule as existing InsertSync: storage RAW,
WAR and WAW interactions between two `PIPE_S` payloads need no inserted
synchronization. CANN 9.0 documents hardware-provided scalar synchronization
and rejects `PipeBarrier<PIPE_S>()`:

[PipeBarrier restrictions](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0271.html).

`ptoStorageProtection()` supplies the rule to storage generator construction.
Explicit, rotating, guarded, arithmetic and composed analyses apply it before
reduction, including residual overlap relationships with unresolved addresses.
Read/write accesses stay in lifetime and boundary summaries, so cross-pipe
readiness and release requirements survive. Supplied completion requirements
and native scalar value prerequisites are not storage hazards and remain intact.
The rule does not create arbitrary completion-before-start edges or assume
that scalar completion occurs at issue.

Standalone scan and rotating-extraction APIs require an explicit policy; their
numerical pipe labels have no implicit target meaning. Accumulator protection
continues to use the existing structured evidence and is unchanged.

Insertion preflight rejects a prepared scalar barrier before mutating IR. It
does not erase barriers after reduction or replace them with another mechanism.
Regression tests compare independent all-pair required graphs and emitted
command profiles, covering branches, zero trips, repeated regions, mixed scalar
and vector accesses, unresolved aliases, and separately supplied prerequisites.
