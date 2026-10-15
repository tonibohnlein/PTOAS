# Conditional captured-rank evaluation

The invocation-prefix adapter evaluates a consumer's exact incoming activity at
its original cut. When that activity is dynamic, captured-rank selector code now
executes inside a result-bearing branch: the active arm returns the selected
executed publication rank, and the inactive arm returns zero. Constant-false
activity retains its existing shortcut; constant-true activity retains its
original scalar select without introducing a branch or changing structural
full-acquisition qualification.

This changes evaluation placement, not source matching or the publication
protocol. Source publications, counter increments, snapshot updates and source
match predicates remain unchanged. The acquired identity is the captured
executed-source rank, never a substituted reference ordinal. The existing
acquisition watermark and terminal cleanup remain responsible for consumption.

The internal `sourceRank` callback owns pure scalar selection and reads captured
slots. Its current producers are sequence occurrence selection and relational
slot lookup. A dynamic guard still invokes and qualifies that callback exactly
once while constructing the active arm. Unavailable callbacks and wrong result
types still fail preparation; runtime inactivity does not authorize a missing
compile-time proof. No new solver, callback admission flag or private-state
serialization format is introduced.

Prefix expression sharing is scoped by original cut, insertion block, original
mapping and history environment. Entering an active arm cannot borrow outer
emission memos, and the next outer or sibling callback cannot reuse arm-local
SSA. The existing staged-function owner and expression-session scope discard
failed attempts; insertion-point guards restore the caller on failure.

The focused `sync_invocation_prefix.pto` test uses the real collection and
materialization services with nested zero/partial visits and a nonunit source
loop. An independent interpreter checks all publication and acquisition ranks,
payload order and command causality. Inactive division-by-zero selectors must
execute no division; active selectors return exact captured ranks. Two same-cut
callbacks exercise branch memo isolation, and unavailable or wrong-type attempts
must leave the original unchanged and permit an identical successful retry.
Existing sequence and relational production oracles remain enabled.

Source review, compilation and runtime acceptance are separate. This change has
not yet established a Delta timing improvement, physical allocation result,
device compilation or milestone completion. Serial paired runs must measure
both constructed and executed work: separate branch environments may construct
less-shared code even while avoiding evaluation on inactive executions.
