# OAHS placement experiments

These synthetic A3 vector inputs isolate source-word placement, deferred
acknowledgments and class-invariant readiness. They complement the existing
Shenggan, projection, RMSNorm and attention corpus.

- [Mechanisms and local results](../../../docs/designs/oahs-placement-experiments.md)
- [Independent device tasks 2–6](DEVICE_TASKS.md)
- `generate.py --out DIR`: deterministic prepared PTO and exact numerical contract.
- `check.py DIR`: independent local memory/event and ordering checks for a
  `<kernel>/<arm>/plan.pto` directory (arms `default`, `candidate`, `existing`).
- `check_no_motion_gemm.py --repo SOURCE --default PTO --candidate PTO`: compare
  Shenggan's conservative no-motion control without weakening the default oracle.

The microkernel device launch wrapper remains a remote task. Eight runnable
arms were lowered locally; device compilation, numerical correctness and timing
are not claimed. Default OAHS refuses the class-invariant fixture; preserve that
result rather than treating it as a working baseline. The equal-coverage
experiment has no changed native output and no eligible timing arm yet.
