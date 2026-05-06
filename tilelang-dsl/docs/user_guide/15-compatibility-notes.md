## Compatibility Notes

The current experimental implementation in `python/pto/dialects/pto.py` differs from this specification in several ways:

1. **Mask types**: The experimental version uses untyped `mask` instead of `mask_b8`/`mask_b16`/`mask_b32`
2. **Barrier operation**: Uses `pto.barrier()` instead of `pto.pipe_barrier()`
3. **Operation coverage**: Implements only a subset of operations

When implementing new code, follow this specification. The experimental implementation will be updated to match over time.
