# Ordinary GM DMA completion reuse

OAHS native composition now applies a production-compatibility contract to
ordinary same-core TSTORE/TLOAD communication: acquired completion of the
MTE3 write is sufficient for the subsequent MTE2 read. This is an explicit
compiler assumption, not a new device-qualified hardware guarantee. The
documentary qualification investigation remains separate.

Both `may-alias` and `assume-disjoint-arguments` remain supported. The latter
does not erase overlapping accesses through the same root. Actual RAW, WAR,
and WAW demands remain in the ordinary pending-access state.

The existing source-prefix transfer supplies completion across cells and
transitively through other handoffs. An already satisfied GM demand adds no
exchange. A newer write remains pending and requires fresh completion. This
change introduces no endpoint motion, coalescing, or alternate planner.

Native import derives eligibility per physical GM cell. Any MTE2/MTE3 access
outside ordinary TLOAD/TSTORE disables this contract for that cell. Atomic,
special-form and explicit cache-policy operations, and opaque macro phases,
do not inherit it. Scalar cache visibility and remote-protocol obligations
retain their separate checks. Portable core callers default to strict
publication qualification unless they explicitly provide the contract.

Regression coverage includes strict refusal without the contract, missing
acquisitions, fresh generations, direct cross-cell and transitive completion
reuse, overlapping versus disjoint ranges, and atomic/cache-policy exclusion.
Construction and reconstruction use the same imported contract and retain
global event participation, consumption-before-rearm, and retirement checks.

## Local validation, 15 September 2026

Built on `899a15aa2e4b46d5a5315b61abe8929c8263a95d` with the uncommitted
implementation. The stable-source campaign records source-diff hash
`a2c3c07a21f8aa31ed5e708f893fb47a9d378b17c51ae795a1d09837d9b3135e`
and native binary hash
`88da99e4dc73b37d3a3b5696a6af71cf3230fbc28c29fe85898262881e11a55d`.
This results section was added after measurement.

| Frozen cohort | Before | After |
| --- | ---: | ---: |
| PTOAS | 79/150 | 135/150 |
| PyPTO | 7/35 | 20/35 |
| pypto-lib | 167/178 | 172/178 |
| Total | 253/363 | 327/363 |

The unchanged frozen alias contracts yield 56 gains under `may-alias` and
18 under `assume-disjoint-arguments`, with no losses. Remaining first
refusals: 15 GM publication, seven signal/payload alias, eight reserved-buffer,
two helper-contract, and four invalid physical-context fixtures. The ordinary
contract does not automatically extend to the remaining macro communication.

Native validation passed 32 positive, 10 expected-refusal, 191 mutation, and
two frontend checks. Composition-core, structured-core, and structured CTests
passed. Qualified GEMM retains 54 SET / 54 WAIT / zero body barriers / one
terminal drain; all five tested prototype boundary comparisons match.

All eight benchmarks compiled in all three arms (`existing`, `composition`,
`demands`), including C++ generation. All 24 arms have identical mechanism
counts, scalar sites, and physical-key populations to the prior
`benchmarks-gm-final-precommit-r3` campaign. Timing is only a single-sample
smoke measurement, not a performance qualification. No device run was made.

Evidence relative to the workspace parent of this repository:

- `oahs-coverage-work/campaign-dma-completion-r2/summary.json`
- `oahs-coverage-work/composition-dma-completion-r1/summary.json`
- `oahs-coverage-work/benchmarks-dma-completion-r1/summary.json`

The earlier `campaign-dma-completion-r1` is exploratory only: its source
identity changed during replay and it did not produce a valid final report.
