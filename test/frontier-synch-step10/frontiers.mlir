// RUN: python3 %S/run_native.py
// The native executable asserts exact answers through ProgramAnalysis, compares
// interval endpoints with concrete occurrences and checks unchanged obligations
// and original IR. This is not an "analysis ran" acceptance test.
module {}
