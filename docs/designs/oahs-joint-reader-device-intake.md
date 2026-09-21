# Joint-reader device campaign intake — 2026-09-21

The down_proj comparison of `2a130aefe613d93b9cdfaa47955c63510483b5c4`
against `a0c1d761e7624b42a0f494853cc70eecaabe9c44` passes the requested
correctness checks. **Performance remains unresolved.** This validates the
joint first/final reader candidate on the tested cases, not the later relay
selection correction or the coupled attention runtime.

## Locally checked evidence

Received `~/Downloads/oahs-joint-reader-device-20260921.md` and
`oahs-joint-reader-20260921.tar.gz`. Archive SHA-256:
`9efd678ca7b71e6616747422d5d37f03cd2e259509a7ee38c5c23fb2090a9d96`.
The detached checksum mentioned by the sender was not present with these two
local files; this is the locally computed archive identity.

- All 90 entries in the internal SHA256SUMS verify; standalone and archived
  reports are identical. Build-manifest artifact hashes verify.
- Candidate selected PTO is byte-identical to the retained local
  `../joint-reader-work/candidate-final.pto`.
- Rerunning the repository graph checker on the downloaded plans reproduces
  the archived JSON exactly: 18 paths, 73,449 conflicts per arm, 214 full
  relations removed, zero added, equal executed event/fence populations.
- Baseline has a disclosed driver-only `--explain` patch; inspection shows
  the construction command path unchanged. Candidate source is recorded clean.
  Both actual AIC compiler expansions end with effective `-O2`.
- Raw correctness logs show all 12 cases / 24 launches passing: both arms,
  seeds 7/23, block 0, block 19 and four slots (0/7/13/19). Paired input and
  output hashes agree. Error-bound, finite-output, guard, untouched-column and
  input-preservation checks pass.
- Raw timing samples reproduce every reported median and inclusive quartile.
  Each label/device has 20 measured launches and 20 warmups. The control invokes
  the exact baseline executable/library. Two devices, three labels, two rounds
  total 240 timing/warmup launches; no hidden kernel batch appears in the harness.

These are local artifact checks and a graph-checker rerun, not a local device
execution or a fresh compiler rebuild. Extracted evidence and local graph output
are in `../joint-reader-device-work/`.

## Timing and interpretation

Medians in microseconds; ratios are within-device only.

| Device | Baseline | Candidate | Identical baseline control | Candidate/baseline | Control/baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 50.260 | 45.660 | 45.850 | 0.9085 | 0.9123 |
| 1 | 49.490 | 45.130 | 42.650 | 0.9119 | 0.8618 |

The identical binary shifts by 8.77% and 13.82%, comparable to or larger than
the apparent candidate changes. On device 1, the candidate/baseline round ratios
change from about 1.008 to 0.815. The controls establish instability, not its
cause, and do not establish that the true candidate effect is zero.

The harness resets the atomic-add residual before each launch, outside the
ACL event interval, and checks final output. It records a start event, submits
one kernel, records an end event and synchronizes. Host submission gaps can
contaminate that interval; this experiment has not qualified it as pure kernel
execution time. Each label also runs in a separate process. The report correctly
declines attribution rather than claiming a 9% compiler speedup.

The experiment used pinned PTO-ISA `5a4f74cbf627d4aac2e0ce10d5e0d8b118343265`
and CANN 9.0.0. It is separate from the attention task's FIFO contract pin.
The earlier MAT-cycle speedups are also a different comparison.

## Disposition

Close the requested down_proj device correctness task. Keep performance marked
unresolved; do not launch more repetitions of this method. If latency attribution
is pursued, first qualify a device execution measurement or a state-correct
short batch against the identical-binary control, then rerun a small matched
comparison. Static ordering improvement remains established; whether it pays
for the added guards on hardware is still open.
