# Experimental handoff integration

The first integration stage provides the diagnostic flag
`--insert-sync-handoff-facts-dir=DIR` (empty/off by default). It captures
qualified pass-entry physical/control facts on a clone before synchronization
selection. Export does not change the generated synchronization. Unsupported
projection or bypass produces an explicit record; an output I/O failure is a
compilation error. The equivalent function-pass option is `handoff-facts-dir`.

The reference uses a conservative local-memory projection. Global-memory and
ACC resource obligations remain native; a passing reference comparison is not
a native transformation or device correctness proof. No new input annotations,
dialect operations, alias promises, or Python IR builders are introduced. The
existing Python CLI forwards the same native flag.

See [native bridge contracts, results and reproduction](../../test/experiments/insert_sync/event_model/NATIVE_BRIDGE.md).
