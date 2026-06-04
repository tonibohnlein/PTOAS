# Module Backend Driver and Object Emission

## Purpose

This design adds a module-level backend selector and reorganizes `ptoas` around
an explicit driver layer. The driver owns command-line parsing, input loading,
target-arch resolution, textual PTO parsing, PTOBC decoding, backend job
dispatch, object emission, and final output handling.

The PTO compiler pipelines remain in `tools/ptoas/ptoas.cpp`. The driver calls
those pipelines through `compilePTOASModule`; it does not duplicate or own the
lowering pipeline itself.

The design also replaces backend-specific object/fatobj emitters with one
`ObjectEmission` module. `ObjectEmission` provides shared CCE/Bisheng-facing
helpers for compiling C++ or VPTO LLVM artifacts and packaging or linking
fatobj objects.

## User Contract

### `pto.backend` Module Attribute

Use the `pto.backend` attribute on `module`:

```mlir
module attributes {pto.backend = "emitc"} {
  ...
}

module attributes {pto.backend = "vpto"} {
  ...
}
```

Valid values are `emitc` and `vpto`. Unknown values are invalid.

The attribute is intentionally module-level. Function-level backend selection is
not part of this contract; a function uses the backend of its nearest enclosing
backend module.

### Backend Selection Priority

`--pto-backend` remains the strongest selector.

1. If the user passes `--pto-backend=emitc` or `--pto-backend=vpto`, the command
   line forces a single backend for the input.
2. If the user does not pass `--pto-backend`, PTOAS reads `pto.backend`
   attributes.
3. If neither the command line nor the input specifies a backend, PTOAS keeps
   the existing default: `emitc`.

For a container with child modules, each child has an effective backend. A
missing child `pto.backend` uses the default backend. If all effective child
backends are the same, PTOAS uses a single backend job. If child modules have
more than one effective backend and `--pto-backend` is absent, PTOAS enters
mixed fatobj mode.

The driver must distinguish "the user did not pass `--pto-backend`" from "the
user passed `--pto-backend=emitc`"; the current option default alone is not
enough.

### Single-Backend Input

For a single VPTO module:

```mlir
module attributes {pto.target_arch = "a5", pto.backend = "vpto"} {
  func.func @kernel(...) attributes {pto.kernel} {
    pto.section.vector {
      ...
    }
    return
  }
}
```

If `--pto-backend` is absent, this is equivalent to:

```bash
ptoas --pto-backend=vpto input.pto -o kernel.o
```

`pto.kernel` is the preferred spelling for launched device kernels. The legacy
`pto.aicore` spelling is still accepted for compatibility.

For `pto.backend = "emitc"`, PTOAS uses the existing EmitC lowering to produce
CCE C++ source. In the current single-backend EmitC path, the driver writes that
text output to `-o`; it does not internally compile the single EmitC result into
a fatobj.

For `pto.backend = "vpto"`, normal object mode produces a host-linkable fatobj
at `-o`. VPTO object output requires an explicit file path.

### Mixed-Backend Container

A mixed-backend input is an outer module containing backend-selected child
modules:

```mlir
module attributes {pto.target_arch = "a5"} {
  module attributes {pto.backend = "emitc"} {
    func.func private @vpto_post(%arg0: i64)

    func.func @emitc_entry() attributes {pto.kernel} {
      %c0 = arith.constant 0 : i64
      func.call @vpto_post(%c0) : (i64) -> ()
      return
    }
  }

  module attributes {
    pto.backend = "vpto",
    pto.kernel_kind = #pto.kernel_kind<vector>
  } {
    func.func public @vpto_post(%arg0: i64) {
      ...
      return
    }
  }
}
```

The outer module carries shared attributes such as `pto.target_arch`. Each child
module carries its own backend selection. Child modules may also carry
backend-specific attributes such as `pto.kernel_kind`. When the driver detaches
a child module into a backend job, it propagates shared outer attributes to that
child unless the child already defines them.

Mixed-backend mode produces a final fatobj. It requires an explicit `-o` file
path and rejects debug IR output modes because those modes do not produce the
child fatobjs needed by the mixed linker.

If child modules use more than one backend and `--pto-backend` is present, the
command line wins and the input is treated as a forced single-backend
compilation.

## Export and Symbol Contract

For non-kernel functions, PTOAS follows the normal `func.func` shape:

| `func.func` form | Meaning in a backend child module |
|------------------|-----------------------------------|
| public function with a body | exported definition |
| private function with a body | module-local helper definition |
| private function without a body | external import declaration |

A function without a body does not export a symbol from the current child
module. It declares a symbol that must be resolved outside the current child. A
function with a body defines the symbol in the current child module; it is
exported only when it is public.

For `pto.kernel` functions, symbol visibility is not the import/export
contract. A `pto.kernel` function is a launched device entry and follows the
backend's kernel ABI. Users write kernel function names without `_mix_aiv` or
`_mix_aic`; VPTO lowering derives those suffixes from the normalized
`pto.kernel_kind`.

Users write source-level non-kernel function names without backend ABI
suffixes. The object emission path derives public ABI names for VPTO LLVM
modules from the target unit and the detected CANN package version:

| CANN package version | `pto.kernel_kind` | Source symbol | Generated ABI export symbol |
|----------------------|-------------------|---------------|-----------------------------|
| `9.0.0-beta.1` | `#pto.kernel_kind<vector>` | `@foo` | `@foo_mix_aiv` |
| `9.0.0-beta.1` | `#pto.kernel_kind<cube>` | `@foo` | `@foo_mix_aic` |
| Other versions | `#pto.kernel_kind<vector>` | `@foo` | `@foo.vector` |
| Other versions | `#pto.kernel_kind<cube>` | `@foo` | `@foo.cube` |

If the CANN version files cannot be read, the toolchain defaults to the
`9.0.0-beta.1` ABI.

This ABI naming is applied to compiled artifacts. The driver does not build a
separate export/import table and does not pre-resolve every cross-backend call
before compilation.

### Mixed-Backend External Call Case

Cross-backend calls are represented as normal external symbol references inside
the caller module. The callee is written as a source-level symbol and provided
by another backend child module or by a later link input.

```mlir
module attributes {pto.target_arch = "a5"} {
  module attributes {pto.backend = "emitc"} {
    func.func private @vpto_post(
      %src: !pto.ptr<f32, gm>,
      %dst: !pto.ptr<f32, gm>,
      %n: index)

    func.func @emitc_entry(
      %src: !pto.ptr<f32, gm>,
      %dst: !pto.ptr<f32, gm>,
      %n: index) attributes {pto.kernel} {
      func.call @vpto_post(%src, %dst, %n)
        : (!pto.ptr<f32, gm>, !pto.ptr<f32, gm>, index) -> ()
      return
    }
  }

  module attributes {
    pto.backend = "vpto",
    pto.kernel_kind = #pto.kernel_kind<vector>
  } {
    func.func public @vpto_post(
      %src: !pto.ptr<f32, gm>,
      %dst: !pto.ptr<f32, gm>,
      %n: index) {
      ...
      return
    }
  }
}
```

In this case, the EmitC child module owns an external import declaration and the
call site for source symbol `@vpto_post`. The VPTO child module owns a public
definition for the same source symbol. The input does not spell a backend ABI
symbol such as `@vpto_post_mix_aiv` or `@vpto_post.vector`; the backend object
path derives the required ABI symbol.

The backend pipelines do not inline or lower across the child-module boundary.
Each child is compiled into a fatobj, and the final mixed link resolves
cross-child device references with the CCE fatobj linker.

The declaration is private because current `func.func` usage represents
body-less external declarations as private symbols.

## Validation

PTOAS should reject:

1. Any `pto.backend` value other than `emitc` or `vpto`.
2. A normalized VPTO child module missing `pto.kernel_kind`.
3. Mixed fatobj mode without an explicit file path passed with `-o`.
4. Mixed fatobj mode with debug IR output flags.
5. Invalid VPTO section sugar, including duplicate vector/cube sections in one
   function and nested vector/cube sections.
6. VPTO host stub generation where vector/cube variants of the same logical
   kernel have incompatible signatures inside the module being compiled.

External symbol problems that survive backend compilation are diagnosed by the
object-emission or fatobj-link stages.

## Architecture

```text
ptoas main (tools/ptoas/driver.cpp)
  ├─ register PTOAS dialects, passes, and command-line options
  ├─ parse command line
  ├─ load input buffer
  ├─ parse .pto or decode .ptobc
  ├─ resolve target arch and backend mode
  ├─ runPTOASJobs()
  │    ├─ EmitCBackendJob ──────────────── compilePTOASModule -> C++ source
  │    ├─ VPTOBackendJob ──────────────── compilePTOASModule -> fatobj
  │    └─ mixed backend
  │         ├─ EmitCBackendChildJob ───── compilePTOASModule -> child fatobj
  │         ├─ VPTOBackendChildJob ───── compilePTOASModule -> child fatobj
  │         └─ FatobjLinkJob ─────────── link child fatobjs
  └─ write text output or keep final fatobj at -o
```

Pass pipelines stop at compiler artifacts:

| Backend path | Pipeline output |
|--------------|-----------------|
| EmitC | CCE C++ source |
| VPTO | VPTO LLVM modules plus optional host stub source |

`ObjectEmission` is not a pass and is not appended to any pass pipeline. It is a
driver-called service layer. The driver chooses which `ObjectEmission` API to
call after the relevant pipeline has returned its artifact.

## PTOAS Driver

`ptoas` enters the driver layer immediately. Command-line parsing, input
loading, MLIR context setup, textual `.pto` parsing, and `.ptobc` decoding are
driver responsibilities. Single-backend EmitC, single-backend VPTO, and
mixed-backend fatobj all go through this driver layer; backend-specific
shortcuts should not bypass it.

`tools/ptoas/driver.cpp` provides `main`. `tools/ptoas/ptoas.cpp` keeps the PTO
compiler options, dialect/pass registration helpers, and `compilePTOASModule`.

### Driver Responsibilities

The driver owns the control flow:

```text
load input
  -> parse/decode module
  -> resolve backend mode
  -> run backend job(s)
  -> write text output or final fatobj
```

1. Input setup.
   - Parse PTOAS command-line options.
   - Track whether `--pto-backend` and `--pto-arch` appeared on the command
     line.
   - Load `.pto` text or `.ptobc` bytes.
   - Decode `.ptobc` inputs.
   - Parse textual `.pto` inputs with the effective parser target arch.
   - Set or preserve `pto.target_arch`.

2. Backend resolution.
   - Parse and validate `pto.backend`.
   - Decide single-backend EmitC, single-backend VPTO, or mixed-backend fatobj
     mode.
   - Detach backend child modules for mixed mode.
   - Preserve shared outer attributes such as `pto.target_arch`.

3. Backend jobs.
   - `EmitCBackendJob` compiles a module to CCE C++ source.
   - `VPTOBackendJob` compiles a module to VPTO LLVM artifacts and emits a
     fatobj.
   - `EmitCBackendChildJob` compiles an EmitC child to a temporary fatobj.
   - `VPTOBackendChildJob` compiles a VPTO child to a temporary fatobj.
   - `FatobjLinkJob` links child fatobjs into the final mixed fatobj.

4. Output handling.
   - In text-output mode, write the compiler text result to `-o`.
   - In object-output mode, require an explicit file path passed with `-o`.
   - In mixed mode, reject debug IR output flags before object emission starts.

Current implementation status:

- The driver parses `pto.backend`, detaches backend child modules, propagates
  shared outer attributes, and dispatches single or mixed backend jobs.
- `ObjectEmission` owns the CCE/Bisheng object and fatobj stage APIs.
- Mixed fatobj mode requires an explicit `-o` file path and rejects debug IR
  output flags.
- Cross-backend external imports are represented by private body-less
  `func.func` declarations and are resolved by the object/fatobj link flow, not
  by a separate driver symbol-resolution table.
- Each VPTO child job emits its own host stub source when needed; the current
  mixed path does not merge multiple VPTO children into one shared stub context.

### Driver Data Model

The driver keeps only the state needed to run backend jobs:

```text
PTOASContext
  MLIRContext
  output path
  target arch
  original argc / argv
  CANNToolchain
  TempFileRegistry

PTOASCompileResult
  kind: text | VPTO object | mixed object
  text output
  VPTO cube LLVM module
  VPTO vector LLVM module
  VPTO host stub source
```

Mixed mode does not require a separate persistent plan object. Child modules
are detached into backend jobs, and each job owns the actions required to
produce its artifact.

## `ObjectEmission`

`ObjectEmission` owns all Bisheng-facing object and fatobj operations. It
replaces the separate C++ and VPTO fatobj emitter concepts with one module that
supports both composed helpers and stage-level operations.

`ObjectEmission` does not decide backend selection and does not run PTO or VPTO
MLIR lowering pipelines. The driver produces C++ source or VPTO LLVM modules,
then requests object/fatobj operations from this component.

### High-Level Emit Interfaces

```text
emitFatobjCCE(cppSource) -> fatobj
emitFatobjLLVM(cubeLLVM, vectorLLVM, stubSource, moduleId) -> fatobj
emitFatobjLLVMWithRuntime(cubeLLVM, vectorLLVM, stubSource) -> fatobj
```

The C++ source path compiles EmitC-generated CCE source into a fatobj:

```text
CCE C++ source
  -> Bisheng CCE compilation
  -> fatobj
```

The VPTO path compiles VPTO LLVM modules for the matching device target:

```text
VPTO vector LLVM -> Bisheng IR compilation -> vector device object
VPTO cube LLVM   -> Bisheng IR compilation -> cube device object
device object(s) + stub.cpp -> fatobj
```

### Fine-Grained Stage API

`ObjectEmission` exposes stage-level APIs so the driver and tests can run
individual pieces without going through a monolithic helper:

```text
writeCppSource(cppSource, path)
writeLLVMModule(llvmModule, path)
writeHostStubSource(stubSource, path)

compileCppToDeviceObject(cppPath, outObjPath, target)
compileLLVMToDeviceObject(llPath, outObjPath, target)

mergeDeviceObjects(objectPaths, outObjPath)
compileStubToFatobj(stubPath, deviceObjPath, outputPath, moduleId)
linkFatobjs(fatobjPaths, outputPath)
```

`compileStubToFatobj` is the stage that consumes `stub.cpp` and the compiled
device object and produces a fatobj. The current flow does not expose host stub
compilation as a separate output object.

### ObjectEmission Responsibilities

1. Discover and validate the Bisheng/cce-ld/ld.lld toolchain.
2. Own temporary-file creation and cleanup for source, LLVM IR, device objects,
   command stderr, merged object, host stub source, and fatobj output.
3. Compile CCE C++ source into fatobj artifacts.
4. Compile VPTO LLVM modules into vector or cube device objects.
5. Merge VPTO device objects when both cube and vector modules are present.
6. Compile the host stub together with the device object into a fatobj.
7. Link multiple child fatobjs into the final mixed-backend fatobj.
8. Keep diagnostics separated by stage and artifact kind.

## Implementation Status

Implemented behavior covered by tests:

1. `pto.backend` attr fallback when `--pto-backend` is absent.
2. CLI backend override when `--pto-backend` is present.
3. Mixed backend mode selection from child module backends.
4. Missing child `pto.backend` defaulting to the default backend.
5. VPTO child `pto.kernel_kind` validation.
6. Mixed mode requiring an explicit output file.
7. Mixed mode rejecting debug IR output flags.
8. VPTO public non-kernel ABI suffix generation from suffix-free source
   symbols in object emission.
9. VPTO kernel `_mix_aiv` / `_mix_aic` suffix generation from suffix-free
   `pto.kernel` source symbols.
10. Cross-backend external calls represented as normal `func.func` external
    declarations and resolved by the fatobj link flow.
