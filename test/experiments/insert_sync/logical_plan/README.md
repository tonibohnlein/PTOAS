# Native logical-relation checks

This is the first query-parity component of milestone one, not an implemented
native synchronization constructor. The driver links the same C++ query source
used by the compiler; the Python test process compares it with libisl and
explicit expected integer relations. No Python or libisl enters compilation.

Build only this small target using the established LLVM/MLIR build:

```sh
cmake -S test/experiments/insert_sync/logical_plan -B "$RELATION_BUILD" \
  -DLLVM_DIR="$LLVM_BUILD/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_BUILD/lib/cmake/mlir"
cmake --build "$RELATION_BUILD" --parallel 2
python test/experiments/insert_sync/logical_plan/check_relations.py \
  --driver "$RELATION_BUILD/pto-logical-relations-test" --output "$NEW_RESULTS"
```

Use disk-backed paths and a fresh results directory. If libisl is outside the
dynamic loader's search paths, set `PTOAS_EVENT_MODEL_ISL_LIBRARY` explicitly.
The runner preserves every request, answer, stderr and driver hash. Record the
LLVM source revision and linked library hashes alongside the results.

The cases cover carried slots one through four, partial replacement, may-write
retention, conditional first acquisitions, nested reader boundaries, exact
integer existential constraints, wrong invocations, absent maxima, continued
completion queries, selected-barrier feedback and exhausted/incompatible queries.
They do not establish native footprint correctness or emitted event safety.
