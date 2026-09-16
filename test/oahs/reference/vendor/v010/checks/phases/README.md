# Supplied final-block protocol: reference implementation 0.9

Run from this directory:

```sh
python3 run_checks.py
python3 operational_checks.py
python3 population_checks.py
python3 phase_interface.py certificates/resource_consumption_relay_loop.json
```

Only Python 3.10+ standard-library modules are needed. No solver or hardware is
used. The module imports bit-set and antichain utilities from `../automatic/`;
the dedicated deliverable contains those utilities as well as the complete
phase implementation.

Read `QUALIFICATION.md` before interpreting any result as a hardware claim.

* `phase_interface.py`: typed qualification, complete analytical fragments,
  finite safety interface, generalized paired endpoint monitor, collecting CFG
  fixed points, and read-back checking of inductive certificates.
* `full_history.py`: separately written adjacency-set graph oracle. It shares
  the declared typed operation fragments, but not the finite transfer rules,
  projection, closure implementation, history antichains, or paired monitor.
* `examples.py`: supplied unsynchronized payload/resource profiles and authored
  ordinary handoffs. This is schema checking, not automatic schema synthesis.
* `run_checks.py`: positive/negative interactions, import refusals, three endpoint
  monitor adversaries, seven certificates, mutations, randomized comparison.
* `operational_checks.py`: a busy-state resource sequencer and the necessary
  per-role service-order assumption, independently checked against graph prefixes.

The reference bits `live` and `readable` are bookkeeping for the common control
traversal. Causal anchors, not those bits, justify asynchronous progress. A state
is never reset at a loop boundary. The generalized monitor compares all original
I/C and access begin/end vertices, including pairs within the same operation.

`adversary_edges` and `adversary_internal` are explicit test-only hooks in the
Python API. They are not fields in emitted/serialized commands and are not part
of the admitted target vocabulary.

Certificates are finite invariant sets, not iteration bounds. Rechecking checks
entry inclusion and every successor; it does not rerun synthesis or rediscover
an invariant. Two one-shot certificates include exact-order checks; the five
cyclic/branching mixed-protocol certificates are safety checks. This distinction
is intentional: necessary whole-operation handoffs/fences can strengthen the
ideal fine-grained payload order.
