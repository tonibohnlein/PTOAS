These files retain pass-entry inputs and emitted baselines from the
`3400426f3` representative native corpus campaign. `manifest.json` records their
case identities and hashes, including original hashes before removing extra
blank lines at baseline EOFs. Inputs are byte-identical. The baseline files have the previously improved
small-GEMM completion boundaries, before initial-pair cleanup.

`check_local_refinements.py` compares actual completion cuts and scalar-command
counts, checks token participation in unchanged and varied loops, and verifies
payload, allocation, view and ABI identity. `exp_gate_mm` is the unchanged nested
loop that previously exceeded the concrete reconstruction node limit. These are
compiler/replay tests, not executable device harnesses or numerical references.
