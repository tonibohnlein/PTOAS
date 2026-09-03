# Repository Guidelines

## Project Structure

PTOAS is an LLVM/MLIR-based compiler for PTO programs. Public C++ interfaces and
MLIR pass declarations are under `include/PTO/`; implementations are under
`lib/PTO/`. The `tools/ptoas/` directory contains the command-line frontend and
pipeline setup. Lit tests and PTO examples live in `test/lit/` (notably
`test/lit/pto/`), while user and ISA documentation is under `docs/`.

## Build, Test, and Development Commands

Use the PTOAS virtual environment and a compatible VPTO LLVM/MLIR build:

```bash
source .venv/bin/activate
ninja -C build ptoas       # build the compiler
ninja -C build              # build all PTOAS targets
ninja -C build check-pto    # run the lit/FileCheck suite
ptoas test/lit/pto/empty_func.pto -o /tmp/empty.cpp
```

For a fresh checkout, configure with `quick_install.sh` after setting
`LLVM_BUILD_DIR`, `PTO_BUILD_DIR`, and `PYTHON_BIN`; see `README.md` and
`docs/build_with_installed_llvm.md` for the complete LLVM setup.

## Coding Style and Naming

C++ follows LLVM style: two-space indentation, descriptive PascalCase types,
camelCase methods, and `UPPER_SNAKE_CASE` constants. Format changed C++ with
the repository-compatible `clang-format` (typically version 14), and keep
headers self-contained with explicit includes. Python follows PEP 8 and uses
snake_case names. Prefer small, focused changes and preserve existing license
headers.

## Testing Guidelines

Add regression cases as `.pto` files under the relevant `test/lit/` directory;
use `RUN:` and `FileCheck` assertions, and give files descriptive
lowercase names. Run `ninja -C build check-pto` plus focused tests while
iterating. Backend or NPU-specific tests should document required architecture
and runtime prerequisites.

## Commits and Pull Requests

Write concise imperative commit subjects (for example, `Add ProtocolSync ...`)
and keep unrelated fixes separate. Pull requests should explain the behavior
change, identify tests/build commands run, link an issue or design document
when applicable, and call out required LLVM/CANN versions or known limitations.
