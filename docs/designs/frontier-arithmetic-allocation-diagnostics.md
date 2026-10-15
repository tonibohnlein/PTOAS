# Arithmetic allocation diagnostics

Arithmetic allocation still uses the selected immutable policy. The default
`SingleId` policy proves one-ID reuse; `experimental-coordinate-allocation=true`
selects the existing bounded coordinate candidate. A failed sufficient placement
proof is unresolved allocation, not a hardware capacity lower bound.

With `profile-stages=true`, both policies use the same borrowed allocation
profile. The existing projection, composition, difference, extrema, order and
containment scopes report exclusive work. Profiling changes neither proof call
order nor successful allocation contents, and no recipe retains the profile.

The profile adds three kinds of attribution:

- `arithmetic-family-binding:` rows associate original record and local
  arithmetic source/target site IDs with the retained original operations and
  locations. These are not parent-translated site IDs or indices inferred from
  a flat payload walk. At most 16 bindings are sampled per exporter request;
  the inventory reports total, sampled and omitted counts. Operation names and
  file names are bounded in diagnostic text; non-file locations are explicitly
  labelled rather than recursively printing arbitrary location graphs. Missing
  original operations are explicitly labelled. These limits affect diagnostics
  only: every original family still participates in the proof.
- `arithmetic-handoff-refusal:` rows identify the family, supplied width bound,
  policy, last attempted proof stage and outcome. `construction-unavailable`
  means an exact operation did not produce its requested result;
  `reuse-uncovered-within-bound` means the constructed sufficient reuse test did
  not cover the obligation. Missing coordinate or consumer-order premises are
  separate outcomes. None is reported as a capacity theorem.
- `arithmetic-export-refusal:` and
  `sequence-allocation-child-export-unavailable` rows distinguish protocol
  alternatives, matching tuple shapes, lifetime selectors and child-summary
  failure from the mathematical handoff proof. The sequence row records the
  child index, parent type/record offsets, strength and policy after the same
  existing finite fallback has been attempted.

A cached arithmetic failure preserves its detailed message. Failed exporter
attempts still restore their prior expression/error state. Diagnostic labels
are copied strings; they do not retain new operation or analysis owners.
These profiles are emitted when the enclosing request finishes. A watchdog
that interrupts an unfinished request can still require a separately bounded
stack observation; absent final profile data is not proof of unavailability.

## Prefill attribution boundary

The committed construction checkpoint produced prefill logical output in
74.410 seconds with the default one-ID policy. Its missing supported export
does not exercise the opt-in coordinate candidate. Historical diagnostics in
`.local/production-closure-m2/` found the one-ID sufficient proof uncovered for
local family `0 -> 4`. A separate coordinate probe completed four width-two
families and stopped during family `11 -> 56` composition. Those local IDs must
be bound to their retained original operations before making source-location
claims. Current coordinate construction uses lifted containment, so the older
projected-composition stack does not identify the current candidate's stall.

The diagnostic regression checks enabled/disabled result, error and structural
work equivalence for the same one-ID failure, distinguishes malformed required
relation construction from uncovered reuse, and retains bounded aggregate
profiling checks for both policies. The frozen native build and all 51 focused
checks passed, including the general arithmetic allocation controls. Source,
retained tools and runtime results are pinned in
`.local/milestone-loop/cofactor-repair-gate/`. Fresh production attribution remains
pending. This change supplies attribution, not a new allocation export or a
measured prefill speedup.
