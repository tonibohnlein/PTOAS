# Placement experiments: completed campaign intake

Reviewed locally on 2026-09-20. The campaign used the immutable placement
snapshot based on `8afb90f17`, not the later sibling-replay optimization or the
new common-frontier test-only work.

## Follow-up: barrier isolation and coverage correction

The [barrier-isolation follow-up](oahs-conditional-mte2-isolation.md) is complete:
existing fails all three taken-path seeds; adding only the conditional MTE2
barrier repairs them; OAHS passes. Its archive also recovers the dedicated probe
and row22 logs. The placement host matrix is 35 successes plus one expected
refusal across 36 attempts; the 264 successful rows belong to MAT. No-motion GEMM
was host-only as designed, not a missing device deliverable.

## Evidence available locally

The downloaded report is `/home/toni/Downloads/oahs-placement-experiments-REPORT.md`,
SHA-256 `667a2d80c2436ad67bdb9710dc9ee39764d3b486b2b15ac156532b806428560b`.
The device agent reports a 306-file, round-trip-verified archive named
`oahs-placement-experiments-device-campaign.tar.gz`, with SHA-256
`5b07c6bb795eacee98fee49520838961a609747da753e7015b9eb78fcd4269a9`.
The archive subsequently arrived in Downloads and its computed hash matches
that value exactly. The detached checksum file is still absent, but the archive
is matched against the hash supplied directly by the user. Extraction and audit
artifacts are in `/home/toni/work/pypto3_sync_more/placement-device-review/`.

Local verification now establishes:

- All **306 internal manifest entries** match; 307 regular files include the
  manifest itself. Gzip/tar extraction succeeds.
- All **11 libraries and 11 executables** match their build-record hashes.
  Each record's generated-C++ hash matches an archived source. This checks the
  saved artifacts, not independently which file was loaded remotely.
- Every archived compiler expansion contains cc1 commands with effective `-O2`.
- All **6,480 timing samples in 14 blocks** parse, with 180 per arm and one device
  per block. Stored medians and ratios reproduce from those samples, including
  the 0.9679 identical-binary source-gap ratio.
- All **eight regenerated microkernel PTO plans** match the shipped pins exactly.
- The archived report was byte-identical to the separately downloaded report at intake.
  That local download still has no appended erratum; the isolation report supplies
  the subsequent correction.

`audit.py` and `audit.json` retain the checks and per-block results. No archived
executables or scripts were executed as part of this read-only evidence audit.

Local source checks confirm that the shipped candidate source archive has the
report's SHA-256 `b43c2f085ab731905d248ac8f599e59c9774e95447f0e736089c572559231252`.
The retained deferred-ack plan pins also confirm the reported conditional MTE2
barrier difference. These source checks complement the returned-archive audit above; neither is a new device execution.

## Outcome by mechanism

| Mechanism | Reported host/device evidence | Status |
| --- | --- | --- |
| Optional-proposal rejection / exact-fit admission | Baseline failure reproduced; candidate falls back successfully; invalid proposal leaves no state; default plans match pins | Mechanism validated on the reported cases |
| Source-gap placement | Same seven pairs, two checked relations removed; 18/18 correctness passes | Ordering benefit reproduced; latency unresolved |
| Deferred acyclic acknowledgment | Five to four pairs; three/two relations removed on false/true arms; OAHS arms pass dedicated probes | Ordering/event benefit reproduced; latency unresolved |
| Class-invariant first consumer | Default OAHS refuses the fixture; opt-in and existing pass 12 initial correctness runs combined | Qualified support extension; no default-versus-candidate speedup comparison |
| Final helper trials | Row22 outputs bitwise equal across three arms; helper omission changes row22/44 plans | Device cost unresolved; trial work remains separately charged |
| Equal-coverage binding | Two encounters, four probes, zero choices; PTO and C++ identical | No useful changed provider yet; no timing needed |
| No-motion GEMM | 200/394/782 to 330/652/1296 pairs, identical checked payload order | Host reproduction complete; intentionally host-only, no device timing |
| Coupled partial attention, row44 | Host comparison complete; authentic coupled harness unavailable | Device measurement blocked, not a failed kernel |

The archive includes passing CTest/native logs and focused host results. Some
headline evidence was initially missing. The follow-up archive recovers it:
`host/all.json` contains 16 attempts (15 successes and one expected default
class-invariant refusal), and `host/task2/matrix.json` has 20 successes. The
264-row matrix is from the MAT campaign; it is not placement-campaign coverage.

## Timing interpretation

The measured microkernel and row22 comparisons are unresolved. The source-gap
identical-binary control gives a 3.2% apparent label difference; several option
ratios change sign across device replicas. A 4.9% row22 no-helper slowdown does
not reproduce. These observations justify withholding speedup/slowdown claims.
They do **not** prove all arm differences are noise or establish equivalence.
The archived harness resolves the timing scope: it uses host
`steady_clock` around a kernel launch plus `aclrtSynchronizeStream`. These are
**host-observed launch-and-synchronize durations**, not device-event kernel-only
latencies. They include host/runtime overhead. Do not compare their absolute
values directly to the earlier projection kernel-only campaigns; the exact
fraction attributable to launch overhead was not isolated.

No-motion GEMM was intentionally host-only. Its recovered result confirms the
pair-count and ordering comparison; no device binary, samples or job were created.
Do not infer that its extra commands are free from unrelated microkernel timing.
A device comparison is optional future work, not an overdue campaign deliverable.

The old 180-sample campaign is already complete. New work follows the revised
10-warmup/20-measured-launch budget, with more sampling only for a specific
unresolved decision. Correctness-only follow-ups need no performance sweep.

## New correctness finding: conditional same-pipe overwrite

The dedicated device probe reports, for each of default OAHS, deferred-ack OAHS
and existing InsertSync:

- `active=0`: 12/12 pass per arm.
- `active=1`: 12/12 pass for each OAHS arm; 12/12 fail for existing.

The archived initial correctness log records six existing/true failures across
three seeds and single/queued launches, plus 30 passing configurations. It also
contains source-gap 18/18 and class-invariant 12/12 passing summaries. However,
the separate 12-run-per-arm probe output was absent from the original archive (`tools/repro_da.sh` is present),
as was row22's nine-run correctness/output-hash log (`run_corr_r22.sh` is present).
Both omitted logs are now recovered in the isolation archive and support the
reported summaries: the dedicated failure probe and nine passing row22 runs
with identical cross-arm output hashes per seed.

The reported failures vary in first-bad row and retain b-derived values where
a-derived values were expected. This is strong evidence of a reproducible
correctness defect consistent with missing synchronization. Varying indices
alone do not prove its cause.

The shipped source loads b into the local x tile, then conditionally loads a
into the same x tile. Both OAHS plans contain `pto.barrier <PIPE_MTE2>` immediately
before the conditional overwrite; existing does not. A publication after both
loads can make their results complete for readers without establishing the
required old-writer-completion -> new-writer-issue order between those loads.
This makes the missing WAW barrier a specific hypothesis, not a blanket rule
about all MTE2 work or all existing plans.

The [quick correctness-only device task](../../test/benchmarks/placement/CONDITIONAL_MTE2_DEVICE_TASK.md)
is complete: 12 harness runs, 48 launches, no timing. Its one-line PTO/C++
mutation repairs all three failing seeds; default OAHS passes throughout.
See the [verified isolation result](oahs-conditional-mte2-isolation.md).
This supports the missing WAW-order explanation and barrier sufficiency here.
The next step is a local existing-pass regression and source diagnosis, not
another repetition campaign. A general compiler repair remains unimplemented.

## Packaging defects and accounting corrections

Both packaging defects are visible in retained local tooling:

- `verify_source.py` iterates a list of `path`/Git-`oid` rows, while these manifests
  are path-keyed SHA-256/mode/symlink dictionaries (8,038/8,046 entries). The
  first retained entries have modes 0664/0644. Patch content equality must not
  be described as mode equality; preserve or explicitly normalize modes in new
  bundles and verify the actual format. Do not rewrite dispatched archives.
- `build_device_arm.py` invokes `-###` and checks only its exit code; it does not
  enforce the existence of a device cc1 line. The reported compiler accepts the
  process with an unsupported-option diagnostic. Adopt the returned `-v`
  adaptation for future bundles, requiring a device cc1 line and its last
  optimization flag to be `-O2`. A zero exit status alone is insufficient.

There are two reporting distinctions to retain:

- The row44 table's `replay=102810`, 61 updates and 1,695 final-check sites are
  AIC counters. Its 78–85 second wall measurements cover the module. The returned raw log
  separately gives AIV **813,458 replay evaluations, 125 updates and one helper
  trial taking 515,328 us**. AIC's two trials take 127,688 us: combined helper
  time is about 0.643 seconds, not 0.128 seconds for the whole module. It remains
  small relative to module wall time. Retain per-function counters. Earlier local
  sibling-replay measurements are a later, separately pinned change.
- The row22 0.9834 replica is labelled device5 in the table and allocation list;
  prose calls the three replicas "three other devices." Treat it as another
  run, not evidence from a third distinct additional device, until raw job IDs
  resolve the wording.

## Scheduling and next work

Campaign complete; no campaign jobs remain. Seven devices were available because
card3 belonged to an unrelated job. The agent reports that card0 locking now
works and used it successfully. Discover current availability for the next task;
do not preserve a blanket card0 exclusion or interfere with unrelated jobs.

Priorities: preserve the isolated conditional WAW regression and trace the
existing-pass decision before a general repair; fix packaging verification
before another source bundle. Recovery and no-motion status questions are closed.
Continue OAHS retained-generation and frontier-placement work separately. Do not
reopen completed MAT timing or claim a latency improvement from this campaign.
