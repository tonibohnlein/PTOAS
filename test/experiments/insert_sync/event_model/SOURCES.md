# Sources reread and what is actually adapted

Accessed 2026-09-08. Literature establishes the listed ideas under its own
assumptions. `ALGORITHM.md` is a new adaptation for the stated Ascend-style model;
none of the sources proves this prototype correct or its schedules optimal.

## Cypress — full article HTML, compilation sections reread

Rohan Yadav, Michael Garland, Alex Aiken, Michael Bauer. **Task-Based Tensor
Computations on Modern GPUs.** PACMPL / PLDI 2025, DOI 10.1145/3729262.

https://arxiv.org/html/2504.07004v1

Relevant sections: 4.1; 4.2.1; 4.2.4–4.2.6.

Source-derived: asynchronous operations produce logical completion events and
accept event preconditions; loops/branches retain event structure; these are
compile-time objects, not dynamic runtime dependency tracking. The compiler
preserves dependencies through later transformations. Allocation introduces
anti-dependencies for reused storage.

Adaptation: keep physical schedule and allocation already present in PTO. Derive
logical event relationships from that schedule; do not copy Cypress's mapping,
copy-in/copy-out or warp-specialization passes. A logical event is not a reusable
Ascend flag, and the source's event arrays do not themselves solve that lowering.

## Tawa — full preprint HTML, operational and lowering sections reread

Hongzheng Chen et al. **Tawa: Automatic Warp Specialization for Modern GPUs with
Asynchronous References.** arXiv 2510.14719v1; CGO 2026 listing.

https://arxiv.org/html/2510.14719v1
https://2026.cgo.org/details/cgo-2026-papers/35/Tawa-Automatic-Warp-Specialization-for-Modern-GPUs-with-Asynchronous-References

Relevant sections: III-B (put/get/consumed), III-C, III-D, III-E.

Source-derived: put publishes a produced value; get acquires/borrows it;
consumed returns storage permission. Completion of consumption is distinct from
acquisition. Ring slots and target lowering provide pipelining.

Adaptation: preserve this semantic distinction in RAW and reclamation relations.
Do not force every source program into a closed aref cycle before analyzing it;
empty paths and one-shot uses need not manufacture unused tokens. Do not import
NVIDIA mbarrier phase, arrival or warp rules into Ascend's consumable flag model.
The prototype derives per-occurrence handoffs; it does not implement Tawa's IR.

## Fixed-schedule resynchronization — both reports' parsed text reread

Shuvra S. Bhattacharyya, Sundararajan Sriram, Edward A. Lee.
**Resynchronization of Multiprocessor Schedules: Part 1 / Part 2—Latency-Constrained
Resynchronization.** UCB/ERL M96/55 and M96/56, 1996.

https://www2.eecs.berkeley.edu/Pubs/TechRpts/1996/ERL-96-55.pdf
https://digicoll.lib.berkeley.edu/record/133701/files/ERL-96-56.pdf

Relevant portions: Part 1 synchronization/IPC graphs and redundancy; Part 2
sections 3–5, especially the example where several new edges jointly eliminate
one edge although no new edge alone does so.

Source-derived: occurrence distances matter in iterative schedules; validate
redundancy against the combined synchronization graph. Reducing synchronization
commands by adding constraints can increase latency, so count minimization and
latency preservation are distinct. Part 2's latency definition concerns the first
influence of an input, not a general worst-case latency bound over all invocations.

Adaptation: preserve iteration vectors, not merely static edges or carried bits.
Use full-plan completion rather than cached singleton coverage. The reports'
serial processor model MUST NOT be applied directly to an asynchronous Ascend
pipeline. Our model separates payload issue/completion and flag submit/fire.
We do not implement their unbounded-latency resynchronization objective.

The PDF text was available. Screenshot calls on the cited pages failed; no claims
here depend on visual inspection of the diagrams or on reconstructed chart values.

## Feautrier — abstract/metadata accessible, full 1991 article not accessible

Paul Feautrier. **Dataflow Analysis of Array and Scalar References.** International
Journal of Parallel Programming 20, 23–53 (1991), DOI 10.1007/BF01407931.

https://link.springer.com/article/10.1007/BF01407931

The accessible abstract describes recovering the source statement and iteration
instance for each read in its supported affine setting. I did not obtain or reread
the full paywalled article. Implementation details in this prototype are therefore
based on the public isl manual and tested C API, not attributed to unseen text.

## isl — public dependence-analysis manual reread; actual C API exercised

https://libisl.sourceforge.io/user.html

Relevant section: Dependence Analysis / High-level Interface.

The public page currently describes isl 0.28; the executed local library identifies
itself as **isl-0.27-GMP**. Used C functions include
`isl_union_access_info_from_sink`, `set_must_source`, `set_schedule_map`,
`compute_flow` and dependence/no-source getters. Native C API results, not a mock
flow implementation, drive the symbolic prototype.

Source-derived: flow combines schedules, source/sink access relations, and
must/may/kill semantics. Must sources act as kills; possible writes cannot be
silently treated as definite production. The prototype chooses a conservative
all-pairs fallback for may-writes rather than a more elaborate mixed flow model.

Adaptation: forward flow for RAW and WAW, reversed-schedule flow for WAR, full byte
alias relationships, and symbolic prefix maxima to synthesize handoffs. isl does
not establish native memory effects, bitvector no-overflow, or event semantics.
Its approximate transitive closure is not used as a positive correctness proof.

## Lazy code motion — limited supporting analogy, not a reused theorem

Jens Knoop, Oliver Rüthing, Bernhard Steffen. **Lazy Code Motion.** PLDI 1992.
https://doi.org/10.1145/143095.143136

The abstract/metadata were accessible, not a full reread of the paper. It motivates
separating availability from needs in a placement problem. The prototype's prefix
staircase is our target-specific construction, NOT an implementation of the paper
or a transfer of its optimality result from pure computations to consumable flags.

## Ascend event contract — official project API page reread

https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/SetFlag_WaitFlag_ISASI.html

Contract used: SetFlag waits for completion of prior source reads/writes before
setting its flag, but does not block following source instructions. WaitFlag waits
for a set flag, consumes it, and gates following destination instructions. Pair
arguments must match; consecutive sets using the same pipe/event before proper
consumption are not a safe reusable-event construction.

The retrieved site is marked as a development preview. The model is a reference
for these specific semantics, NOT blanket hardware certification. Reconcile it
with the pinned production target definitions and focused silicon tests before
native code generation. This package does not infer GM publication/visibility
from event-direction legality and does not claim every event direction is legal.
The six-key default is an experiment policy, not a substitute for native reserved
keys. The caller can lower the pool; no fallback serialization is added silently.

## Project evidence used, not reclassified as new experiments

Reviewed source baseline: `tonibohnlein/PTOAS`, branch
`codex/insertsync-revision-r1`, commit
`862ab811124d2e67a7f2cde992461983163f6a7a`.

Read this turn: current commit metadata, `SyncRequirements.h`, the scalar replay
observer interface in `performance/measure.py`, and original `two_buffer.auto.pto`.
No full clone was obtained (container network resolution failed), no production
files were changed, and no native compiler was available in this runtime.

The supplied `R8_DIAGNOSIS.md` is historical evidence: it motivates occurrence
precision and warns about shape-specific composition and proof-coverage failures.
Its R8 benchmark counts must not be presented as current-head results. Its lines
281–311 describe the fixed-schedule per-slot semantic target and robustness tests.
