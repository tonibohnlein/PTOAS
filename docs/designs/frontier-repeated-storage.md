# Repeated symbolic storage

The constant-stride adapter extends repeated regions with visit-owned storage
and invariant shared read-only storage. It uses shared modeled accesses, including
the complete union of byte ranges for each effect. It does not infer narrower
instruction footprints.

## Constructed subclass

A family has an invariant address space, canonical base, origin `a`, and a
positive constant byte stride `S`. Every owned access must reconstruct exactly as

```
a + ((iv - lower) / step) * S + localByte
```

The complete local footprint must materialize as finite byte intervals contained
in `[0, S)`. Element widths are included. Local footprints cannot depend on inner
occurrences: the existing first/last occurrence selectors for each static effect
therefore apply to every byte of its footprint. Overlapping views within a family
are combined before selecting boundary readers and writers.

Shared read-only families use stride zero and invariant finite local byte
intervals. Their origins and the number of visits can remain symbolic. Each
origin symbol must have a nonnegative range proved by the generic integer range
interface, and the origin expression must fit unsigned 64-bit arithmetic under
those bounds. Unconstrained signed origins, including negative offsets from an
interior canonical pointer, are unavailable. Writes in
a proposed read-only family are rejected. Distinct owned families and persistent
cells require physical separation, using canonical bases, the invocation's GM
alias policy, or checked interval comparisons. Unknown separation is unavailable.

These are the shared mathematical address maps. Scalar address arithmetic is
expanded by `SyncScalarEvolution` only under its range or no-signed-wrap proof;
unproved visit-dependent arithmetic remains opaque and cannot reconstruct the
required translation. As with the shared memory model, physical pointer accesses
must be valid in their address domain. This physical-domain premise does not
imply a nonnegative relative origin; the separate range proof above establishes
that requirement before unsigned owner inversion. The adapter does not interpret invalid
wrapped pointer accesses as disjoint mathematical addresses.

## Owner and selectors

For a queried byte offset `x` in the family's address domain:

```
d = x - a
owner = d / S
localByte = d % S
present = x >= a && owner < trips
```

The arithmetic is total even for absent queries. Presence guards the unsigned
subtraction's natural-number interpretation. Reservation membership may hold for
a hole: actual selectors additionally test membership in each modeled interval.
Zero trips suppress every selector. Read-only first readers use visit zero and
last readers use visit `trips - 1`, guarded by positive trips.

`storageSelectors` returns `nullopt` when a query cannot be constructed, and a
selector set whose guards are false for proved absence. First readers precede
the first writer strictly; last readers follow the last writer strictly. A read
and write at the same payload do not create either boundary reader.

## Integration and limits

The certificate captures its exact body interface. Repetition constructs its
inter-visit bridges from a temporary persistent-only storage view. The body's
full all-event queries and internal insertion recipes remain intact. Every
classified effect is restored in the output alongside its owner-based selectors.

Discharged effect occurrences are retained separately for this adapter; ordinary
finite composition does not reinterpret them as uniform conflicts. Production
recognition supports an outer-owned output after a compact inner loop, including
the case where the inner loop has zero trips.

The current finite port backend rejects composition of a symbolic-storage child
with additional children. It does not replace a family of owner-specific edges
with one static first/last effect edge. Nested symbolic ownership, phase-prefix
ownership, parameter-valued strides, and byte-dependent inner occurrence
selectors require further adapters and are explicitly unavailable. Version-four
logical identity allocation remains a separate milestone.
