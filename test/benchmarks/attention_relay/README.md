# Native single-block attention relay regression

This regression imports the production native model and calls the ordinary
selected constructor. It supplies no candidate effects or candidate endpoints.
The current default compiler refines the FIFO slots and constructs the two
receipts. A captured pre-change command population is the ordering control.
See the [diagnosis and contract boundaries](../../../docs/designs/oahs-attention-bank-prefix-diagnosis.md).

## Run

Use an already-built native Makefiles build containing `pto-oahs-selected-test`:

```sh
python3 test/benchmarks/attention_relay/run.py \
  --build ../oahs-m1-native-build \
  --out ../attention-relay-work/repro \
  PATH_TO_ROW48/prepared.pto PATH_TO_ROW49/prepared.pto
```

The runner checks exact input hashes before compiling, generates a diagnostic
driver outside the repository, and reuses the native build's compile/link
recipes. Compilation and cases run serially. The probe is optimized because
the independent graph oracle compares all payload pairs. No device, network,
sanitizer or new production option is involved.

Prepared-input SHA-256 pins:

| Input | SHA-256 |
| --- | --- |
| `pypto_lib__prefill_fwd__48` | `2127cef1e8080fa8972ba4b952167d7c4e164265da592b66cdaaac144485faac` |
| `pypto_lib__prefill_fwd__49` | `7ca19d0b9d6d506c9bbe32c177390fe9afd00ec7ec3c50039354af4aa83cd7c2` |

The exact input pins and the graph signature in `baseline.json` bind the
captured baseline words to these fixtures. The runner generates the reference
words only; candidate endpoints are found through the production decision
records. Fixture operation IDs appear only in test assertions, not production
qualification. Both directions retain the same underlying root and slot map. The lowering premise comes from PTO-ISA
`0c112d61f41342bd0867ce1080c29f1590d72484` A3 `TPush.hpp`.

## Checks and output

- Production native import/construction, followed by cold causal acceptance.
- Six independent finite graph cases per input: empty, one/two native-length
  outer entries, and shorter/varying lengths as checker-CFG stress cases.
- Full payload-order subset, explicit forbidden QK3-to-first-receive/preparation
  relations, and required same-slot FIX-to-receive completion.
- Memory, balance, rearming and acyclicity; independently counted cursor slots
  across prologue/body/epilogue and repeat entry.
- Missing first/second new relay leg and missing retained broad receipt.
- A safe but broader mutation postponing the middle forwarding endpoints to
  the later receive. Its additional ordering must be detected separately from
  missing-completion failures.

`1111` in the finite logs means memory, rearming, balance and acyclicity all
pass; missing-support mutations separately check memory and rearming with
balanced, acyclic events. Logs, model JSON, input pins and build commands are written under
`--out`. Candidate model JSON contains the actual constructor records. The runner
also executes the native wrapper with emission and reconstruction; the
`*-native.pto` files are those emitted candidate modules.

The probe establishes local command-graph evidence. It does not establish peer
progress, GM visibility, numerical correctness or device performance. The production implementation is limited to the qualified static two-slot
interface; it makes no general least-order claim.
