## Common Errors

### Typed Mask Mismatch

```
Error: f32 vector operation cannot consume mask_b16
```

**Solution:** Ensure mask granularity matches vector element size:
- `f32` vectors use `mask_b32`
- `f16` vectors use `mask_b16`
- `i8` vectors use `mask_b8`

### Strict Scope Implicit Capture

```
Error: strict_vecscope body cannot capture outer value 'ub_in' implicitly
```

**Solution:** Pass all required values in the capture list:

```python
# Wrong:
with pto.strict_vecscope() as ():
    vec = pto.vlds(ub_in, offset)  # ub_in from outer scope

# Correct:
with pto.strict_vecscope(ub_in) as (ub):
    vec = pto.vlds(ub, offset)
```

### Untyped Loop Carried State

```
Error: loop-carried value must have explicit machine type
```

**Solution:** Add type annotations to loop-carried variables:

```python
# Wrong:
remaining = 1024  # Plain Python int
for i in range(0, N, step):
    mask, remaining = pto.make_mask(pto.f32, remaining)

# Correct:
remaining: pto.i32 = 1024
# or
remaining = pto.i32(1024)
```

