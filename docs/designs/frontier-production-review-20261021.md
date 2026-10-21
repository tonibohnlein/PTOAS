# Review and commit acceptance: compact production coverage

Review scope is the complete uncommitted checkpoint on
`147cb72edd014a0cc7950ae50e1d870a370a0253`, including the nine preserved files from
the earlier production-coverage task. The October 20 source/tool pins are
historical evidence. This review records the small additional corrections and
new tool hashes separately; it does not overwrite those receipts.

The production algorithms are accepted for their documented contracts after the
two corrections below and the focused gate. This is a scoped source review and
host-model qualification, not an independent external proof audit or device
qualification.

## Findings and disposition

| Finding | Evidence and impact | Disposition |
|---|---|---|
| Extracted expanded prepare callbacks did not retain the supplied input owner | The full view's query closures held `inputOwner`, but its new prepare/filtered-prepare closures held only state/program, whose physical-phase/access pointers borrow the input. An extracted callback therefore had a weaker lifetime than the owning export. No native crash is claimed. | Both callbacks now share a preparation closure retaining and checking the supplied binding. A control-block lifetime regression extracts each callback, drops the view/other owners, then checks retention and release independently. |
| The test consumer could report an undecodable occurrence as absent | `Probe::resolve` continued when either candidate construction or original-coordinate decoding failed. That could weaken finite validation outside the admitted decoder frames. It is not a production selector bug. | Return an explicit unavailable diagnostic. Four arithmetic fixtures include a deliberately unsupported quotient-frame view and require unavailability. Ordinary false-presence and untouched-byte absence checks remain. |
| Nested scalar map admission | Physical coordinate zero is the source ordinal; record labels are separate. A scalar rule is valid with tuple arity one even in a nested inventory. | Accepted. Nonconstant wider tuples still require a checked tuple rule; nondefault-ID and malformed-arity tests pass. No allocator/coloring algorithm changed. |
| Atomic destination access | Full f32 ACC→ND AtomicAdd uses the declared rectangle and both read/write modes; conversion/unknown-valid-extent cases remain conservative. | Accepted under the documented operation/layout model. Independent offset/stride byte rectangles and negative extent/conversion tests pass. No device qualification added. |
| Fixed original-point emission | Finite and arithmetic crossings retain their own source and consumer point masks, matching labels and caller frames. | Accepted for qualified stable activity. Unsupported history/frames still refuse and preserve mathematics. No expansion limit, protocol or barrier policy changed. |
| Research/dataflow and range claims | May-facts do not select dynamic sources or publication cuts; ABI ranges are not inferred from benchmark values. | Kept explicitly test-only/conditional. No new uniform token-to-command theorem or production recognizer added. |

## Generality and ownership

No new production path recognizes a kernel name or an ad-hoc tile/trip tuple.
The access repair is selected by instruction mode, type, layout, extents and
strides. The mapping repair is selected by the existing certificate's physical
arity. Endpoint conversion uses producer-owned original coordinates, guards and
legal cuts. The generic initialized nested pipeline fixture exercises this
adapter without depending on Elementwise or GroupNorm recognition.

Existing mathematical strategies, SCC transfer and allocation implementations
are reused. Logical matching, closed notifications, pending payload effects,
realization order and physical resource evidence remain separate facts. The
three specialized derivative allocation refusals remain explicit; they are not
capacity-impossibility findings. Compact-K/replay and zero-inner-body semantics
are preserved by the focused checks.

The field-level code review covered overflow/arity bounds, callback ownership,
source/consumer coordinate separation, availability, actual command activity,
and read/write footprint modes. The recipe owner retains the input only when
an owning input was supplied; legacy borrowed calls retain their existing
caller-lifetime obligation. Retaining the input introduces no session cycle.
The callback release regression checks the observed ownership independently.

## Qualification and packages

The [receipt](frontier-production-review-20261021.json) records final tool/source
hashes, exact paths/messages, complete commands, build status, focused test
results and the two corrected findings. The accepted gate uses one worker and
contains 26 tests, including the access and coordinate-mapping tests in addition
to the prior 24-test checkpoint. No build runs during this campaign. Earlier
failed negative-test construction/build attempts are retained on disk and are
not acceptance receipts.

Ruff F checks and `git diff --check` pass. The compliance prefilter emits 14
G.FMT.11 matches; direct inspection confirms a braced body at every one. Its
single-line greedy regex misreads nested parentheses/multiline conditions.
No suppression was introduced and no unreviewed error remains. The build adds
no warnings. Full native/device analyzers are unavailable, so this does not
claim complete static-analyzer compliance.

The reviewed changes are packaged in dependency order:

1. Exact f32 atomic-store rectangle and read/write declaration.
2. Existing scalar cyclic certificate in a nested physical inventory, plus
   production-coverage regressions and receipts.
3. Test-only reaching-access dataflow experiment and its explicit realization gap.
4. Original-coordinate finite/arithmetic endpoint adapter, owned callbacks and
   independent command/cover/consumer checks.
5. Indexed-arithmetic and parameter/phase qualification, with unsupported
   contracts retained explicitly.
6. Aggregate checkpoint and this acceptance review.

Only the exact listed paths enter each commit. No push or PR publication is
part of this request. After committing, the next work is a bounded original
whole-kernel probe of lossless block cast and Gated Delta, using the same
35-second watchdog, six-ID pool and ordinary/explicit-completion policies.
It will diagnose stage failures and control growth without changing algorithms
or repeating the seven-kernel survey.
