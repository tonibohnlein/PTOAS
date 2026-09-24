# FrontierSynch foundation

Branch: `codex/handoff-foundation`.
Base: upstream PTOAS `master`, `f5eff3ee249697f6157088f649c6434fcc9d7c5b`.
The upstream repository has no `main` branch.

## Scope of this increment

`pto-insert-sync` accepts `algorithm=existing` (default) and
`algorithm=handoff`. Existing construction, motion, redundancy removal,
allocation and emission remain in place. Authored synchronization and function
declarations keep their existing exclusion before either constructor runs.

Both modes consume `SyncInput`, which owns the original InsertSync translator's
control/instruction nodes and memory records. Instruction views retain every
translated phase, its original operation, pipeline, reads/writes, macro phase
and existing UnitFlag metadata. No second opcode/effect registry is introduced.
This shares the translator's contract; it does not establish completeness of
all hardware effects or enable any new native-order exemption.

The handoff entry point currently reports that construction is not implemented.
It does not emit synchronization or invoke another constructor. No code from the
old OAHS constructor, its retries, campaign tools or reference data was imported.

Translation failures now propagate through nested structured regions as
`LogicalResult`; partial nodes and memory records are discarded instead of
calling `llvm_unreachable`. All current translator callers handle failure.

## Validation and next step

The source diff passes `git diff --check`. No compile or runtime validation has
been completed. An isolated CMake configuration with Python bindings disabled
hit an upstream tools/ptoas dependency on the Python extension targets; build
setup was stopped to keep this increment focused. The worktree has its own venv.

Next: establish focused compilation/import checks, then implement immutable
original physical-use/control queries and the first draft constructor engine
from paper revision 0.40. Treat that paper as a working specification. Keep the
existing mode intact throughout; do not import the old OAHS constructor to
preserve historical example outputs.
