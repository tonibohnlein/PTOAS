# Stage 3: ready parameters, signed phases and the remaining interfaces

No caller bound or phase rewrite was installed in production. The fresh
[receipt](frontier-parameter-phase-contracts-20261020.json) records eight original
regional requests, one stronger export request, independent boundary arithmetic,
and the source/capture/orchestration hashes establishing each premise. These
are focused requests, not another kernel survey. The 35-second watchdog remains
unchanged.

## Causal-convolution prefill

The four metadata loads (`seqlens[block]`, `seqlens[block+1]`, `cache[block]`,
`initial[block]`) and their index casts occur before the group loop. Their SSA
values are invariant snapshots for that loop, even if later memory changes.
`SyncInput.cpp::resultAvailability` explicitly classifies their ordinary PIPE_S
scalar results as synchronous. This is the supplied compiler execution contract,
not new device qualification. Other target-dependent asynchronous results must
still obtain their own readiness proof.

The port's `port.json` requires positive lengths divisible by four, total tokens
at most 32, distinct valid cache indices, and disjoint GM argument allocations
under may-not-alias. With its 6x128 state, cache index is 0 or 1; a cache visit
owns three rows. Sequence starts/ends must describe valid ordered partitions of
that input. These are caller/port requirements, **not range metadata enforced by
the synchronization input**. Positive length is retained; zero-length tail
writeback would need another padding premise and is not added to the contract.

For admitted lengths, `(length+3)/4=length/4` without signed overflow. Four-token
startup prefetch, `next_group<groups`, output tails, and final three-row cache
writeback therefore have ready control parameters. This does not prove a
periodic certificate for every view/prerequisite inside the loop.

Fresh results:

* The existing explicit numerical regional request refuses `unsupported-control`
  before supplying a template, both with and without query/selector requirements.
* An explicit arithmetic demand-only request **succeeds**: bounded coefficients,
  observed D=6, P=2, C=1024, 60 sites, 12083 primitive relations and 2586 retained
  demand pieces. It takes 30.638 s. This is a qualified modeled-effect result,
  not merely a statement that Presburger arithmetic could express the program.
* Ordinary exact regional selection reaches the 35-second watchdog. The stronger
  arithmetic query+selector request also times out at 35 seconds. A timeout does
  not classify a missing mathematical premise or prove an export impossible.
  Endpoints and physical allocation are not established by these calls.

The independent integer/view interpreter reads the original prepared operations
with the existing checked metadata/flag trace bindings. For lengths 4, 8, 12, 32
and initial-state false/true, it verifies in-bounds GM transfers, every output row
exactly once, startup state load selection, prefetch/final-group behavior, and
three final state writebacks. This is a GM boundary/control check, **not a complete
native-footprint, demand-cover, or numerical-convolution oracle**. No unsupported
whole-loop interface is relabeled as independently validated.

## Qwen attention finalize

The original frontend capture names argument 5 `__pypto_spmd_block_idx`, and
argument 6 `__pypto_spmd_block_num`. The pinned source uses `pl.spmd(48)` and the
orchestrator sets `block_num(48)` on this task. Under that runtime ABI, start is
0..47. The lowered physical-storage input has an ordinary signed i32 argument
with no range annotation; the synchronization analysis cannot obtain the ABI
restriction from its name or one test valuation. A checked propagation of this
caller fact would be a separate prerequisite for a uniform rewrite.

For `w=start+48*t>=0`, write `a=w/64`, `b=(w-64*a)/8`, `c=w-64*a-8*b`:

* `c=start mod 8`;
* `dt=8*a+b=floor(start/8)+6*t`, so `dt<active` is an active prefix;
* four visits later `(a,b,c,dt)` becomes `(a+3,b,c,dt+24)`;
* the input row is `1024*a+128*c+16*b`, not an invariant GM address;
* the output row is `token_offset+dt`, with column `640*c`.

Thus only the residue components repeat. Absolute GM movement and the clipped
prefix need compatible storage/occurrence exports. Making the entire guard
periodic would be incorrect. For negative start `w=-1`, signed truncating
division gives `c=-1`; at `w+192=191`, `c=7`. This is a concrete counterexample to
applying the positive-domain four-phase identity without a range proof. Active
bounds also require valid output rows (`0<=offset`, `offset+active<=128`) and
input view ranges. Nothing here installs these external conditions implicitly.

1024 finite original work positions check the identities and GM row/column
bounds. Six original five-payload children at work positions 0,16,32,48,64,1008
pass independent presence, closure, actual covers, and local **and GM** selectors
when fully specialized. They establish bounded child interfaces, not a uniform
root export.

Explicit finite expansions of zero/one/two/four loop visits preserve retained
mathematics. Zero visits pass the full check. Nonzero complete-selector requests
refuse **`finite storage exports for discharged effects not implemented yet`**.
This is a capability failure, not an empty GM answer or failed demand reduction.
The original symbolic loop's numerical route refuses `loop-domain`; its
arithmetic importer reports `index-arithmetic` at the loop. Adding signed div to
one scalar folder would not supply the required active-prefix and evolving-GM
certificate through composition, so no speculative normalization was added.

## Coverage table

| Region | Candidate class | Recognition result | Demand guarantee | Executable export | Allocation | First remaining obligation |
|---|---|---|---|---|---|---|
| Prefill group loop | Parameterized numerical periodic | Refuses unsupported-control | Not established | Unavailable | Not requested | Complete repeated control/access certificate |
| Prefill group loop | Restricted arithmetic | D=6/P=2/C=1024 admitted; 30.638 s | Minimum for selected modeled effects | Stronger query/selector request times out | Not requested | Attribute large relation/export construction before changing it |
| Finalize symbolic work loop | Four-phase/active-prefix candidate | Caller ABI evidence exists; not supplied as input bounds | Not established uniformly | Unavailable | Existing root allocation is separate prior evidence | Checked caller-range transport plus nonstationary GM/prefix interface |
| Finalize specialized child | Finite | Six cases pass | Modeled closure and actual covers checked | Bounded local/GM selectors pass | Not requested | Does not establish repeated root export |
| Finalize finite repeated derivative | Expanded finite | Mathematics retained | Selected modeled result retained | Complete selectors refuse discharged-effect adapter | Not requested | Preserve exact absence/unavailability and add a qualified adapter separately |

Reproduce bounded checks:

```sh
.venv/bin/python test/lit/pto/Inputs/check_parameter_phase_contracts.py \
  --trace-tool .local/section8-refactor/source-pinned/candidate-build/tools/pto-test-opt/pto-sync-input-test \
  --output .local/compact-class-checkpoints/stage3/bounded
```

Original diagnostic inputs and exact commands remain in the receipt. Timings
are single descriptive observations, not speedups. No allocation or expert
protocol changed; no device was run.

| Established result | Remaining premise | Appropriate manuscript claim |
|---|---|---|
| Entry-loaded metadata can be ready invariant region parameters | Range, alias and initialization remain caller contracts | Readiness and arithmetic/periodic admission are distinct |
| Four-phase residues and active prefix have explicit positive-domain identities | ABI ranges and evolving GM interfaces must be supplied | A supported normalized subclass needs named premises |
| Concrete children have checked interfaces | Repeated selectors for discharged effects remain unavailable | Exact internal reduction does not imply complete boundary export |
