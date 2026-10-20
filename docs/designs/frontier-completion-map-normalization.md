# Constant physical maps at the completion-child boundary

A decoded map assigns `ids[offset(x)]` to an applicable coordinate tuple `x`.
Its logical recipe owns activity, record selection, matching and availability.
The conversion must retain that domain and every logical record.

After validating the original tuple rule and expected coordinate arity, an
all-equal nonempty table `[c, ..., c]` can become `[c]` with zero offset. Every
valid original lookup returns `c`, so this equality holds over the full admitted
coordinate domain, independently of guards, trip counts or lane activity. A
legacy cyclic offset is already total modulo the table size. A tuple rule must
pass `validPhysicalTupleRule` before normalization; a constant table cannot
rescue an invalid offset or foreign frame.

`completionVisitMapping`, private to the completion adapter, owns this
conversion. It consumes already-decoded assignments. It retains labels and pipe
directions, preserves original tuple arity, and then appends the checked enclosing
arity. The existing contiguous-table conversion represents the same assigned ID
as a palette offset. Normalization leaves guards, endpoint cuts, dynamic record
selection and namespace translation to the unchanged recipe/adapter interfaces.

Empty tables, invalid IDs, invalid original tuple rules, overflowing arities and
noncontiguous nonconstant tables remain unavailable. Nonconstant contiguous
tables keep their existing function. No record is removed based on inactivity,
and no coloring, synchronization command or allocation algorithm is introduced.

The proof applies to any already-decoded mapping satisfying these conditions.
There are no kernel names, shape tests, caller bounds or special lane numbers in
the production conversion. `CompletionVisitMappingChecks.cpp` checks active
constant/cyclic/tuple maps, singleton and repeated tables, different arities and
independently varied coordinates, metadata preservation, nonconstant maps and
malformed/refused inputs. It is called by the existing region-expression test.

Normalization establishes assignment equivalence only. The provider must still
prove logical matching and availability; the parent must establish resource
separation across invocations. It does not establish payload completion or make
an otherwise unavailable grouped or contextual child export available.
