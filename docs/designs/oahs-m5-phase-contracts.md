# M5: supplied final-block phases on the existing OAHS path

Base: exact `oahs_m4_50e95419` source postimage. This is an incremental M5
implementation, not a replacement for M4 and not a new public planner mode.
Specification: `main(20260915-212314).pdf`, sections 16–17. The PDF in the
retrieved `synchronization_theory_acm_v0_10.zip` has the same SHA-256:
`09d0d2b014ebfd6d53942b06966538f21ee54972cf7f970467ca70fc6cb4d210`.

## What is implemented

A supplied `final-block-pair-v1` model now participates in the existing
`analyze`, `verify`, `PrefixQuery`, `BundleQuery`, `construct`, and M4 observed
control interfaces. Original operations retain their complete coarse access
signatures; the validator checks them against the supplied role signature.
The phase interpreter then checks the finer **fixed input contract**, not a
post-construction deletion of ordinary hazards.

The new `Phases.h` exposes complete-profile validation, original analytical
fragments, endpoint-aware quality checking, and the native-qualification gate.
`AnalysisResult` additionally exposes unresolved resource obligations, endpoint
witnesses, and permission-anchor facts. `BundleQuery` reports resolved/introduced
resource obligations alongside memory, software-event, and retirement changes.

No native importer currently creates this profile. `phaseNativeQualification()`
returns disabled with an explicit missing-evidence list. A caller-written profile
name or a boolean is not a hardware qualification. Ordinary A3 admission,
UnitFlag modes, and native lowering are unchanged.

## Admission and evidence levels

The profile is explicitly `reference-contract-only`. Its model premises all
default false: writable entry, ordered service of each role's per-block requests,
enabled-phase progress, and common forward control. Supplying these premises
establishes which mathematical model is requested; it supplies no device evidence.

The complete program is checked before any fragment is exposed:

* Cells have one exact nonempty domain/range, no unknown overlap or overflow, and
  no unpartitioned overlap within a domain. Equal offsets in distinct declared
  domains do not alias under this *supplied* physical mapping.
* Every group has fixed distinct supported engines and a complete ordered map of
  unique, disjoint, 512-byte-aligned, 512-byte ACC blocks.
* A final producer reads exactly its declared LEFT and RIGHT operand cells and
  writes all group blocks. It has no accumulating ACC read.
* A final consumer reads the same complete block map and identity-copies to one
  unique aligned 512-byte GM cell per block. The actual access signature must
  match; capability filtering never drops a contradictory effect.
* Ordinary operations cannot bypass the protected-block protocol. KEEP, partial
  or permuted maps, aliases, unexplained resources/private events, visibility,
  and layout/mode inference are refused.

The initial adapter is for issue-ordered asynchronous engines, directed consuming
software events, and local prefix fences. Synchronous lanes, ALL-capable phase
targets, and invocation retirement are explicitly outside this adapter. They
require matching semantics and reference adapters, not changing real hardware
assumptions to make the test profile pass. Generic visibility and authored/private
communication integration remain separate coverage work.

## Original endpoints and resource causality

Producer block access: `I -> beginW -> endW -> C`. Operand reads conservatively
span `I` to `C`. Consumer block read and output write have separate begin/end
pairs. The native identity-copy edge is `endRead -> endOutput`, **not**
`endRead -> beginOutput`. Different blocks are unordered unless another actual
path orders them. These internal endpoints are analytical observables; they do
not create command insertion cuts or permit payload splitting/reordering.

Each protected block retains a current permission anchor U and its next-role
state. The next legal access receives an actual `U -> beginAccess` edge, then U
is replaced by that access's end. Logical writable/readable state does not itself
mean the access has completed in real time. Ordered same-role block service is
an additional model premise; ordered issue and permission bits do not prove it.

The state retains every engine launch/prefix port, used software-event publication
and latest-consumption ports, and the protected-block anchors. History signatures
are attached to original access ends; separate whole-operation completion
signatures support the existing public snapshots. Resource credit is never
converted into blanket engine completion.

`PhaseResourceFacts::completedOperations` means that **all represented prior
occurrences** of that original operation complete before the current anchor.
Like the ordinary pending abstraction, this universal statement is vacuously true
when no such occurrence is represented. It is not evidence that the operation
ran. `carriedConsumptions` records causal reachability from the current latest
software consumption, not an event generation number or a permission-bit claim.

## Same construction path, explicit richer semantic domain

Ordinary programs retain the compact M4 backend. The explicitly supplied phase
contract uses its live-port collecting extension through the same Transfer
service. This is not an automatic exponential fallback when the compact ordinary
checker refuses, and it is not a separate synchronization constructor.

The phase domain joins sets of whole interface states, rather than combining
causal edges from incompatible paths. Sequences, choices, while-before exits,
and M4 observed-control backedges use the same original control graph. A worklist
runs to closure without a dynamic trip limit or numerical analysis-work budget.
The domain is finite but can have an exponential population. Both total and
maximum per-site state counts are reported; no production-scale cost bound is
claimed. The generic `work` counter is not a count of matrix-closure bit operations.

M2 still discovers prospective **whole source prefixes**. It does not assume a
protected-block access end equals whole-operation completion. Actual M3 bundle
replay sees all scoped native edges and can establish additional joint credit.
The existing constructor inserts software handoffs only for remaining obligations,
including ordinary operand readiness/release and output reuse. A wrong original
resource-role word is not repaired by pretending a software fence resets it.

Partially constructed event protocols can still use the private provisional query
to propose repairs. The public report grants no completion from an unmatched,
occupied, or unproved-rearm endpoint. Such a command is retained as an obligation,
but its transfer edge is withheld, its receipt is invalidated, and subsequent
commands cannot rehabilitate it by numeric key identity. Resource-role failures
similarly withhold the permission edge. These conservative continuations collect
additional errors; they never establish a verified execution.

Successful resource/event paths can legitimately carry software-consumption
knowledge. The recurring test `Q->M; produce; consume; F->Q` establishes a real
return for the Q->M key. It still needs independent ordinary output-WAW ordering.
The same path does not release operand reads lasting until producer completion,
or a later unrelated producer write. Stale notifications do not acknowledge a
newer consumption merely because their key names agree.

Closed phase exits require writable resources and empty software keys. This is
NOT a whole-payload retirement or general fresh-caller ABI certificate. The
adapter intentionally refuses the native terminal-retirement contract for now.

## Endpoint-complete quality checking

`checkPhaseOrder` uses the same actual fragments and a separate ideal graph with
original native fragments and every original conflict at its true access begin.
Software synchronization never modifies the ideal graph. It compares all old-to-
new observable pairs and all new-to-new pairs before projecting. Original issue,
completion, and access endpoints are included; the core issue-only shortcut is
not reused.

Dangerous actual/ideal signature pairs use the paired-order dominance relation,
which differs from inclusion-minimal safety histories. Isolated completion-only,
access-endpoint, and within-fragment ordering must not be hidden by testing only
issue vertices. A safe whole-operation producer-to-consumer handshake is therefore
reported coarser than the supplied block protocol. `complete`, `safe`, and `exact`
are distinct; no unsafe candidate receives an exactness certificate.

## Validation and local proof obligations

The unchanged reference from archive v0.10 is pinned under `vendor/v010`; its phase
module identifies its retained implementation as revision 0.9. The old v0.8 pin is
not edited. Tests check all seven stored phase certificates and can rerun the
original mixed-protocol, operational service-order, and population checks in a
scratch copy. Generated reports never overwrite the pin.

The new C++ driver links the actual production core. The bridge independently
reconstructs the actual command words on the original control graph and collects
the pinned phase reference to closure. It compares admission/safety, exactness,
original fragments, and exported completion/event/resource facts. Read-only ghost
I/C tags observe whole-operation completion without adding payload conflicts or
actual edges. Constructed cases remove the supplied software commands first;
acceptance does not claim the constructor regenerates the stored word verbatim.

Local justification follows the draft's finite port factorization: every edge
from the old prefix attaches through A/T/S/D/U; whole-fragment closure precedes
projection; signature image is monotone; retaining inclusion-minimal access-end
signatures preserves universal queries. Actual command/resource legality is checked
before adding its credit. Whole-state successor closure covers arbitrary finite
represented control paths. The paired monitor uses a separate ideal attachment
and checks every fresh observable, including within-fragment pairs. These are
written implementation invariants and differential evidence, **not** a completed
machine-checked proof of the C++ code or of the old compact backend.

## Native qualification still to do

No public planner switch or attribute enables native phase credit in this patch.
Before a native importer can construct this profile, it needs release-pinned
instruction/SDK/PTO lowering evidence, exact physical/block maps and layouts,
mode/coverage agreement, ordered same-role per-block service, initial permissions,
private resource/event accounting, and progress/queue and output-visibility
contracts. It also needs actual native import/emission/reconstruction regressions
and the focused device campaign. Current reference success cannot discharge those
premises. General KEEP/accumulation support and automatic mode selection remain
out of scope even after this narrow profile is qualified.

A phase residual's source context is reported only when the static operation has
one unambiguous original context; otherwise it is `NoAnalysisId`. Consumer cuts
and access endpoint indices remain explicit. This conservative provenance is not
an inferred dynamic producer/consumer pairing or an executable history predicate.
