# Concrete blocked lifetime: down_proj A0 release

**Historical diagnosis at `2a130aefe`.** The subsequent
[joint-reader implementation](oahs-joint-reader-prefix.md) now makes the compiler
select this boundary. The supplied mutation below remains a separate witness.


2026-09-21, compiler `2a130aefe`. This diagnoses an existing native corpus input;
it does not change the constructor or implement a new observation certificate.

## The physical lifetime and unwanted relation

Input: `test/lit/pto/oahs_projection_down.pto`, the original Qwen down-projection
prepared payload. Its current emitted plan matches the corresponding corpus
plan. This is not a new synthetic retained-input kernel or a CATLASS transcription.

| Role | Concrete identity |
| --- | --- |
| A0 storage | MAT `[0,65536)`, imported physical cell 7 |
| A0 writer | MTE2 load into `%16`, original generation at outer chunk `%arg11` |
| Reader child | Owner 54, entry 54, exit 82; IV `%arg12`, lower 0, upper 256, step 128 |
| Last A0 reader | Third `textract` in the child, `%16` at `%39 = %arg12 + 64`; final visit reads the slice starting at 192 |
| Unrelated following reader | Fourth `textract`, B0 MAT `[65536,196608)` into RIGHT |
| Current return | `SET MTE1→MTE2 EVENT_ID0` at child exit |
| Actual deadline | Matching WAIT immediately before the next A0 load |

The A0 release includes the final B0 extraction although that extraction does
not read A0. In the four-chunk, one-output-tile trace, payload visit 13 is that
B0 extraction and visit 28 is the next A0 load. Current output orders the former's
finish before the latter's issue.

The desired release is immediately after the third extraction **on the final
visit only**, before the subsequent B0 slot-reuse acquisition and extraction.
Keep the current return key, its priming/draining, its acquisition deadline,
all readiness, all other endpoints, and all genuine fences.

## What the current analyses know—and what is missing

The diagnostic driver's new `--observations` mode reports existing native,
control and storage views. It does not derive another causal state or choose
endpoints. For this owner it reports:

```text
owner=54  lower=0  upper=256  step=128
first_visit_sites=5  final_visit_words=0  first_input_site=169
```

The first-input observation serves **B0 readiness**, while the requested
last-reader observation would release **A0**. This is a per-owner composition
problem spanning different physical lifetimes, not simply moving both ends of
one private channel.

Three exact restrictions explain the refusal:

1. `NativeFirstConsumer.h::importFirstConsumers` records the qualified owner.
   `Native.cpp` calls it before `importLastReaders`. The latter's
   `owners.count(owner)` check declines an already refined owner.
   `LastVisitFrontend.h::refineLastVisit` also explicitly rejects a nonempty
   `firstVisitPrefix` or existing body observations.
2. `NativeLastReader.h` requires unit step. The actual step is 128.
   The current final-visit atom uses parameter 1, and native emission compares
   `upper - iv <= parameter`. Simply deleting the unit-step restriction would
   emit a predicate that never fires for visits 0 and 128.
3. The last-reader importer and frontend require a wholly straight leaf body.
   This original child has two conditional matrix-initialization choices after
   its straight extraction prefix. Those branches do not read A0, but the
   current qualifier cannot retain that suffix while refining just the useful
   final-reader prefix.

`qualifyReaderRegionCycles` already looks for a qualified final-visit publication
before falling back to `loop.exit`. There are no such final words here, so
child exit is the available publication frontier. The current construction
reports no resource rejection. The diagnostic needs neither an extra key nor a
deleted fence: resource capacity is not the obstruction demonstrated here.

## Exact certificate needed next

Qualify an exactly-once final A0-reader boundary in a loop that already owns a
first-consumer prefix, preserving its original conditional suffix:

- Derive the final visit from original bounds and positive step, with checked
  arithmetic. For this input the last IV is 128. `upper - iv <= step` is an
  appropriate predicate under its established range premises; the general
  emitter and decoder must agree on the parameter's units.
- Compose first and final observations for the same owner, preserving existing
  shared words and payload identities. A one-visit generalization must admit
  both observations on that same visit.
- Prove that every participating child execution reaches the post-A0-read
  boundary exactly once on its final visit, and that no suffix path reads or
  regenerates A0. Keep unrelated conditional matrix work shared.
- Preserve the actual readiness and consumption paths for the existing return
  key across first entry, repeated chunks, odd tails and invocation exit.
- Validate the selected exact words and native predicate reconstruction. A
  finite trace result is not that whole-graph/emission certificate.

This is a specific target for joint observation qualification. It does not
justify relaxing owner, step or straight-body checks independently.

## Controlled supplied-plan experiment

`test/oahs/diagnose_projection_last_reader.py` moves only the body publication
of A0's return. It adds a final-visit predicate derived from the original loop
values and preserves every original payload line. The script refuses changed
anchor/key spellings rather than selecting a different target silently.

The existing independent projection interpreter checks local physical conflicts,
per-key matching, consumption-before-republication and empty event state at
exit. Complete strict payload issue/finish relation **sets** are compared.

| Chunks | Output tiles | Relations before → after | Removed | Added |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | 5,950 → 5,946 | 4 | 0 |
| 4 | 1 | 5,753 → 5,749 | 4 | 0 |
| 4 | 2 | 24,302 → 24,290 | 12 | 0 |
| 17 | 1 | 110,653 → 110,625 | 28 | 0 |
| 17 | 2 | 449,662 → 449,606 | 56 | 0 |

Across 18 paths (`chunks=0,1,2,3,4,17`; `tiles=0,1,2`), there are 73,449 local
conflict checks per arm, 104 removed relations and zero additions. The other
13 paths have identical order. Every path preserves the executed SET/WAIT and
barrier populations. The four removed relations in the four-chunk witness are
exactly B0 issue/finish → next A0 issue/finish; this is not inferred from counts.

Four negative mutations fail for the expected reason:

| Mutation | Independent check rejects |
| --- | --- |
| Publish on the first visit | Missing completion of a later A0 reader |
| Delete the body release | Unseeded return wait |
| Publish on every visit | Publication on an occupied key |
| Use `upper - iv <= 1` with step 128 | Unseeded return wait: final publication never executes |

The supplied early plan parses/verifies as native PTO. That is syntax/IR
verification, **not** production causal reconstruction of this new guard.
At the diagnosed baseline, the compiler still emitted the original child-exit release. There is no device
timing or numerical result; the added comparison/guard has an unmeasured cost.
The interpreter uses declared local footprints and separate GM arguments; it
does not establish a new GM-visibility or cross-core contract.

## Reproduction and next action

From the frozen baseline checkout (not the newer joint-reader compiler), with
the diagnostic target built:

```sh
DRIVER=../oahs-m1-native-build/tools/pto-test-opt/pto-oahs-selected-test
mkdir -p ../blocked-lifetime-work
$DRIVER --construct test/lit/pto/oahs_projection_down.pto > ../blocked-lifetime-work/current.pto
$DRIVER --observations test/lit/pto/oahs_projection_down.pto > ../blocked-lifetime-work/observations.tsv
python3 test/oahs/diagnose_projection_last_reader.py ../blocked-lifetime-work/current.pto --out ../blocked-lifetime-work/diagnosis
```

Raw plans, all 18 comparisons, the explicit removed-edge witness, negative
failures and native diagnostics are in `../blocked-lifetime-work/`.

Next implementation target: the jointly qualified final-reader prefix above,
with the existing B0 first-consumer observation retained. The subsequent explicit user instruction authorized that implementation; see
the linked report. This document retains the original supplied-plan evidence.
