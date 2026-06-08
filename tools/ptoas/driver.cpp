// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "ptoas.h"

#include "ObjectEmission.h"
#include "PTO/IR/PTO.h"
#include "VPTOHostStubEmission.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "ptobc/ptobc_decode.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace mlir;
using mlir::pto::PTOASContext;

#ifndef PTOAS_RELEASE_VERSION
#define PTOAS_RELEASE_VERSION "unknown"
#endif

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static void printPTOASVersion(llvm::raw_ostream &os) {
  os << "ptoas " << PTOAS_RELEASE_VERSION << "\n";
}

static bool hasCLIOption(int argc, char **argv, llvm::StringRef option) {
  const std::string optionWithValue = (option + "=").str();
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg == option || arg.starts_with(optionWithValue))
      return true;
  }
  return false;
}

static std::string normalizePTOASArch(llvm::StringRef archValue) {
  std::string normalized = archValue.str();
  for (char &c : normalized)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return normalized;
}

static bool isSupportedPTOASArch(llvm::StringRef archValue) {
  return archValue == "a3" || archValue == "a5";
}

static std::optional<std::string>
detectPTOASTextualModuleArch(llvm::StringRef text) {
  llvm::SmallVector<llvm::StringRef, 4> matches;
  llvm::Regex archRegex(
      R"ptoarch("?(pto\.target_arch)"?[[:space:]]*=[[:space:]]*"([[:alpha:][:digit:]_]+)")ptoarch");
  if (!archRegex.match(text, &matches) || matches.size() < 3)
    return std::nullopt;
  return normalizePTOASArch(matches[2]);
}

static bool isPTOBCBuffer(llvm::StringRef buffer) {
  return buffer.size() >= 6 && std::memcmp(buffer.data(), "PTOBC\0", 6) == 0;
}

static std::unique_ptr<llvm::MemoryBuffer> readInputBuffer() {
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!fileOrErr) {
    llvm::errs() << "Error: Could not open input file: "
                 << fileOrErr.getError().message() << "\n";
    return nullptr;
  }
  return std::move(*fileOrErr);
}

static bool resolveTextInputArch(llvm::StringRef buffer, bool cliArchSpecified,
                                 std::string &arch) {
  arch = normalizePTOASArch(mlir::pto::ptoTargetArch);
  if (cliArchSpecified) {
    if (!isSupportedPTOASArch(arch)) {
      llvm::errs() << "Error: invalid --pto-arch='" << mlir::pto::ptoTargetArch
                   << "'. Expected 'a3' or 'a5'.\n";
      return false;
    }
    return true;
  }

  if (auto detectedArch = detectPTOASTextualModuleArch(buffer))
    arch = *detectedArch;
  if (!isSupportedPTOASArch(arch))
    arch = "a3";
  return true;
}

static OwningOpRef<ModuleOp> decodePTOBCModule(llvm::StringRef buffer,
                                               MLIRContext &context) {
  llvm::ArrayRef<uint8_t> bytes(reinterpret_cast<const uint8_t *>(buffer.data()),
                                buffer.size());
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
  try {
    return ptobc::decodePTOBCToModule(bytes, context);
  } catch (...) {
    llvm::errs() << "Error: Failed to decode PTOBC.\n";
    return {};
  }
#else
  OwningOpRef<ModuleOp> module = ptobc::decodePTOBCToModule(bytes, context);
  if (!module)
    llvm::errs() << "Error: Failed to decode PTOBC.\n";
  return module;
#endif
}

static OwningOpRef<ModuleOp>
parseTextualModule(std::unique_ptr<llvm::MemoryBuffer> inputBuffer,
                   MLIRContext &context, llvm::StringRef arch) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(inputBuffer), llvm::SMLoc());
  mlir::pto::ScopedPTOParserTargetArch scopedParserArch(
      &context, arch == "a5" ? mlir::pto::PTOParserTargetArch::A5
                             : mlir::pto::PTOParserTargetArch::A3);
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module)
    llvm::errs() << "Error: Failed to parse MLIR.\n";
  return module;
}

static OwningOpRef<ModuleOp>
loadInputModule(std::unique_ptr<llvm::MemoryBuffer> inputBuffer,
                MLIRContext &context, bool cliArchSpecified,
                std::string &arch) {
  llvm::StringRef buffer = inputBuffer->getBuffer();

  OwningOpRef<ModuleOp> module;
  if (isPTOBCBuffer(buffer)) {
    arch = normalizePTOASArch(mlir::pto::ptoTargetArch);
    if (cliArchSpecified && !isSupportedPTOASArch(arch)) {
      llvm::errs() << "Error: invalid --pto-arch='" << mlir::pto::ptoTargetArch
                   << "'. Expected 'a3' or 'a5'.\n";
      return {};
    }
    module = decodePTOBCModule(buffer, context);
  } else {
    if (!resolveTextInputArch(buffer, cliArchSpecified, arch))
      return {};
    module = parseTextualModule(std::move(inputBuffer), context, arch);
  }
  if (!module)
    return {};

  Operation *moduleOp = module.get().getOperation();
  if (cliArchSpecified) {
    moduleOp->setAttr("pto.target_arch",
                      mlir::StringAttr::get(moduleOp->getContext(), arch));
    return module;
  }

  if (auto archAttr = moduleOp->getAttrOfType<StringAttr>("pto.target_arch")) {
    std::string moduleArch = normalizePTOASArch(archAttr.getValue());
    if (isSupportedPTOASArch(moduleArch)) {
      arch = std::move(moduleArch);
      return module;
    }
  }

  if (!isSupportedPTOASArch(arch))
    arch = "a3";
  moduleOp->setAttr("pto.target_arch",
                    mlir::StringAttr::get(moduleOp->getContext(), arch));
  return module;
}

static bool parseDriverBackend(llvm::StringRef backendStr,
                               mlir::pto::PTOBackend &out) {
  std::string s = backendStr.str();
  for (char &c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "emitc") {
    out = mlir::pto::PTOBackend::EmitC;
    return true;
  }
  if (s == "vpto") {
    out = mlir::pto::PTOBackend::VPTO;
    return true;
  }
  return false;
}

static LogicalResult
parseDriverBackendAttr(Operation *op,
                       std::optional<mlir::pto::PTOBackend> &backend) {
  backend = std::nullopt;
  Attribute rawBackendAttr = op->getAttr("pto.backend");
  if (!rawBackendAttr)
    return success();

  auto backendAttr = dyn_cast<StringAttr>(rawBackendAttr);
  if (!backendAttr) {
    return op->emitError("invalid pto.backend attribute. Expected string "
                         "value 'emitc' or 'vpto'.");
  }

  mlir::pto::PTOBackend attrBackend = mlir::pto::PTOBackend::EmitC;
  if (!parseDriverBackend(backendAttr.getValue(), attrBackend)) {
    return op->emitError("invalid pto.backend '")
           << backendAttr.getValue() << "'. Expected 'emitc' or 'vpto'.";
  }

  backend = attrBackend;
  return success();
}

static OwningOpRef<ModuleOp> detachBackendChildModule(ModuleOp outer,
                                                      ModuleOp child) {
  for (NamedAttribute attr : outer->getAttrs()) {
    StringRef attrName = attr.getName().getValue();
    if (attrName == SymbolTable::getSymbolAttrName() ||
        attrName == "pto.backend")
      continue;
    if (!child->hasAttr(attr.getName()))
      child->setAttr(attr.getName(), attr.getValue());
  }

  child.getOperation()->remove();
  return OwningOpRef<ModuleOp>(child);
}

static constexpr llvm::StringLiteral kEmptyHostStubSource =
    "#ifndef __global__\n#define __global__\n#endif\n\n"
    "#ifndef __gm__\n#define __gm__\n#endif\n\n";

static LogicalResult emitVPTOLLVMFatobj(
    const mlir::pto::PTOASCompileResult &jobResult, PTOASContext &context,
    llvm::StringRef moduleId, llvm::StringRef outputPath);

mlir::pto::PTOASContext::PTOASContext(DialectRegistry &registry,
                                      llvm::StringRef outputPath, int argc,
                                      char **argv)
    : mlirContext(registry), outputPath(outputPath.str()), argc(argc),
      argv(argv) {}

mlir::pto::PTOASContext::~PTOASContext() = default;

LogicalResult
mlir::pto::PTOASContext::initializeEnvironment(bool requiresToolchain,
                                               llvm::raw_ostream &diagOS) {
  if (requiresToolchain)
    return initializeToolchain(diagOS);
  return success();
}

void mlir::pto::PTOASContext::initializeMLIRContext() {
  // Be tolerant: ptobc decode may materialize ops from dialects that aren't
  // explicitly registered/loaded in this tool yet.
  mlirContext.allowUnregisteredDialects(true);
  mlir::pto::loadPTOASDialects(mlirContext);
}

MLIRContext &mlir::pto::PTOASContext::getMLIRContext() { return mlirContext; }

void mlir::pto::PTOASContext::setArch(std::string value) {
  arch = std::move(value);
}

llvm::StringRef mlir::pto::PTOASContext::getArch() const { return arch; }

void mlir::pto::PTOASContext::setBackendInfo(BackendInfo value) {
  backendInfo = std::move(value);
}

const mlir::pto::BackendInfo &mlir::pto::PTOASContext::getBackendInfo() const {
  return backendInfo;
}

int mlir::pto::PTOASContext::getArgc() const { return argc; }

char **mlir::pto::PTOASContext::getArgv() const { return argv; }

llvm::StringRef mlir::pto::PTOASContext::getOutputPath() const {
  return outputPath;
}

std::string mlir::pto::PTOASContext::allocModuleId() {
  static size_t nextModuleId = 0;
  return "ptoas_module_" + std::to_string(nextModuleId++);
}

LogicalResult
mlir::pto::PTOASContext::initializeToolchain(llvm::raw_ostream &diagOS) {
  if (toolchain)
    return success();
  std::optional<mlir::pto::CANNToolchain> discovered =
      mlir::pto::CANNToolchain::create(diagOS);
  if (!discovered)
    return failure();
  std::optional<CANNVersion> parsedVersion =
      parseCANNVersion(discovered->cannVersionString);
  if (!parsedVersion) {
    diagOS << "Warning: unable to parse CANN version: "
           << discovered->cannVersionString
           << "; using 9.0.0-beta.1 compatibility behavior.\n";
    parsedVersion = CANNVersion{9, 0, 0, 1};
  }
  discovered->cannVersion = *parsedVersion;
  cannVersion = discovered->cannVersion;
  toolchain = std::move(*discovered);
  return success();
}

const mlir::pto::CANNToolchain *
mlir::pto::PTOASContext::getToolchain(llvm::raw_ostream &diagOS) const {
  if (!toolchain) {
    diagOS << "Error: CANN toolchain is required but was not initialized.\n";
    return nullptr;
  }
  return &*toolchain;
}

mlir::pto::CANNVersion
mlir::pto::PTOASContext::getCANNVersionOrDefault() const {
  return cannVersion;
}

mlir::pto::TempFileRegistry &mlir::pto::PTOASContext::getTempFiles() {
  return tempFiles;
}

LogicalResult
mlir::pto::PTOASContext::createTempPath(llvm::StringRef prefix,
                                        llvm::StringRef suffix,
                                        std::string &path) {
  return tempFiles.create(prefix, suffix, path, llvm::errs());
}

static bool hasPTOKernel(ModuleOp module) {
  bool found = false;
  module.walk([&](func::FuncOp func) {
    if (mlir::pto::isPTOKernelFunction(func)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

class EmitCBackendJob {
public:
  EmitCBackendJob(OwningOpRef<ModuleOp> &module,
                  mlir::pto::PTOASCompileResult &result)
      : module(module), result(result) {}

  LogicalResult run(PTOASContext &context);

private:
  OwningOpRef<ModuleOp> &module;
  mlir::pto::PTOASCompileResult &result;
};

class VPTOBackendJob {
public:
  VPTOBackendJob(OwningOpRef<ModuleOp> &module,
                 mlir::pto::PTOASCompileResult &result)
      : module(module), result(result) {}

  LogicalResult run(PTOASContext &context);

private:
  OwningOpRef<ModuleOp> &module;
  mlir::pto::PTOASCompileResult &result;
};

class BackendChildJob {
public:
  virtual ~BackendChildJob() = default;
  virtual LogicalResult run(PTOASContext &context) = 0;
};

class EmitCBackendChildJob final : public BackendChildJob {
public:
  EmitCBackendChildJob(OwningOpRef<ModuleOp> &&module,
                       SmallVectorImpl<std::string> &fatobjPaths)
      : module(std::move(module)), fatobjPaths(fatobjPaths) {}

  LogicalResult run(PTOASContext &context) override {
    ModuleOp op = module.get();
    op->setAttr("pto.backend", StringAttr::get(op.getContext(), "emitc"));

    mlir::pto::PTOASCompileResult jobResult;
    if (mlir::pto::compilePTOASModule(module, context,
                                      mlir::pto::PTOBackend::EmitC, jobResult,
                                      /*emitVPTOHostStub=*/false) != 0)
      return failure();
    if (jobResult.kind != mlir::pto::PTOASCompileResultKind::Text) {
      llvm::errs() << "Error: EmitC backend child job produced non-text "
                      "output.\n";
      return failure();
    }

    std::string fatobjPath;
    if (failed(context.createTempPath("ptoas-emitc-fatobj", ".o", fatobjPath)))
      return failure();
    const mlir::pto::CANNToolchain *toolchain =
        context.getToolchain(llvm::errs());
    if (!toolchain)
      return failure();
    if (failed(mlir::pto::emitFatobjCCE(
            jobResult.textOutput, fatobjPath, *toolchain,
            context.getTempFiles(), llvm::errs())))
      return failure();

    fatobjPaths.push_back(std::move(fatobjPath));
    return success();
  }

private:
  OwningOpRef<ModuleOp> module;
  SmallVectorImpl<std::string> &fatobjPaths;
};

class VPTOBackendChildJob final : public BackendChildJob {
public:
  VPTOBackendChildJob(OwningOpRef<ModuleOp> &&module, std::string moduleId,
                      SmallVectorImpl<std::string> &fatobjPaths)
      : module(std::move(module)), moduleId(std::move(moduleId)),
        fatobjPaths(fatobjPaths) {}

  LogicalResult run(PTOASContext &context) override {
    ModuleOp op = module.get();
    op->setAttr("pto.backend", StringAttr::get(op.getContext(), "vpto"));

    bool emitHostStub = hasPTOKernel(op);
    mlir::pto::PTOASCompileResult jobResult;
    if (mlir::pto::compilePTOASModule(
            module, context, mlir::pto::PTOBackend::VPTO, jobResult,
            emitHostStub) != 0)
      return failure();
    if (jobResult.kind != mlir::pto::PTOASCompileResultKind::VPTOObject) {
      llvm::errs() << "Error: VPTO backend child job produced non-object "
                      "output.\n";
      return failure();
    }

    std::string fatobjPath;
    if (failed(context.createTempPath("ptoas-vpto-fatobj", ".o", fatobjPath)))
      return failure();

    if (failed(emitVPTOLLVMFatobj(jobResult, context, moduleId, fatobjPath)))
      return failure();

    fatobjPaths.push_back(std::move(fatobjPath));
    return success();
  }

private:
  OwningOpRef<ModuleOp> module;
  std::string moduleId;
  SmallVectorImpl<std::string> &fatobjPaths;
};

class FatobjLinkJob {
public:
  explicit FatobjLinkJob(ArrayRef<std::string> fatobjPaths)
      : fatobjPaths(fatobjPaths) {}

  LogicalResult run(PTOASContext &context) {
    if (fatobjPaths.size() < 2) {
      llvm::errs()
          << "Error: mixed backend link requires at least two fatobjs.\n";
      return failure();
    }

    std::string stderrPath;
    if (failed(context.createTempPath("ptoas-fatobj", ".log", stderrPath)))
      return failure();
    const mlir::pto::CANNToolchain *toolchain =
        context.getToolchain(llvm::errs());
    if (!toolchain)
      return failure();
    return mlir::pto::linkFatobjs(fatobjPaths, context.getOutputPath(),
                                  *toolchain, stderrPath, llvm::errs());
  }

private:
  ArrayRef<std::string> fatobjPaths;
};

LogicalResult EmitCBackendJob::run(PTOASContext &context) {
  ModuleOp op = module.get();
  op->setAttr("pto.backend", StringAttr::get(op.getContext(), "emitc"));

  if (mlir::pto::compilePTOASModule(module, context,
                                    mlir::pto::PTOBackend::EmitC, result,
                                    /*emitVPTOHostStub=*/false) != 0)
    return failure();
  if (result.kind != mlir::pto::PTOASCompileResultKind::Text) {
    llvm::errs() << "Error: EmitC backend job produced non-text output.\n";
    return failure();
  }
  return success();
}

LogicalResult VPTOBackendJob::run(PTOASContext &context) {
  ModuleOp op = module.get();
  op->setAttr("pto.backend", StringAttr::get(op.getContext(), "vpto"));

  bool emitHostStub = hasPTOKernel(op);
  if (mlir::pto::compilePTOASModule(
          module, context, mlir::pto::PTOBackend::VPTO, result,
          emitHostStub) != 0)
    return failure();
  if (result.kind == mlir::pto::PTOASCompileResultKind::Text)
    return success();
  if (result.kind != mlir::pto::PTOASCompileResultKind::VPTOObject) {
    llvm::errs() << "Error: VPTO backend job produced non-VPTO output.\n";
    return failure();
  }

  if (context.getOutputPath().empty() || context.getOutputPath() == "-") {
    llvm::errs() << "Error: object output requires an explicit file path "
                    "passed with -o.\n";
    return failure();
  }

  std::string moduleId = context.allocModuleId();
  if (failed(emitVPTOLLVMFatobj(result, context, moduleId,
                                context.getOutputPath())))
    return failure();

  result.reset();
  result.kind = mlir::pto::PTOASCompileResultKind::MixedObject;
  return success();
}

static LogicalResult emitVPTOLLVMFatobj(
    const mlir::pto::PTOASCompileResult &jobResult, PTOASContext &context,
    llvm::StringRef moduleId, llvm::StringRef outputPath) {
  llvm::StringRef stubSource = kEmptyHostStubSource;
  if (!jobResult.vptoStubSource.empty())
    stubSource = jobResult.vptoStubSource;

  const mlir::pto::CANNToolchain *toolchain =
      context.getToolchain(llvm::errs());
  if (!toolchain)
    return failure();
  if (failed(mlir::pto::emitFatobjLLVM(
          jobResult.vptoCubeModule.module.get(),
          jobResult.vptoVectorModule.module.get(), stubSource,
          outputPath, moduleId, *toolchain, context.getTempFiles(),
          llvm::errs())))
    return failure();
  return success();
}

static LogicalResult collectChildJobs(
    ModuleOp module, mlir::pto::PTOBackend defaultBackend,
    PTOASContext &context, SmallVectorImpl<std::string> &fatobjPaths,
    SmallVectorImpl<std::unique_ptr<BackendChildJob>> &backendJobs) {
  SmallVector<ModuleOp, 4> children(module.getOps<ModuleOp>());
  for (ModuleOp child : children) {
    std::optional<mlir::pto::PTOBackend> childBackend;
    if (failed(parseDriverBackendAttr(child.getOperation(), childBackend)))
      return failure();

    OwningOpRef<ModuleOp> jobModule = detachBackendChildModule(module, child);
    if (childBackend.value_or(defaultBackend) == mlir::pto::PTOBackend::VPTO)
      backendJobs.push_back(std::make_unique<VPTOBackendChildJob>(
          std::move(jobModule), context.allocModuleId(), fatobjPaths));
    else
      backendJobs.push_back(std::make_unique<EmitCBackendChildJob>(
          std::move(jobModule), fatobjPaths));
  }
  return success();
}

static LogicalResult resolveSingleBackend(
    bool cliBackendSpecified,
    std::optional<mlir::pto::PTOBackend> moduleBackend,
    mlir::pto::PTOBackend defaultBackend, ModuleOp module,
    std::optional<mlir::pto::PTOBackend> &singleBackend) {
  singleBackend = std::nullopt;
  if (cliBackendSpecified) {
    singleBackend = defaultBackend;
    return success();
  }
  if (moduleBackend) {
    singleBackend = *moduleBackend;
    return success();
  }

  std::optional<mlir::pto::PTOBackend> firstChildBackend;
  for (ModuleOp child : module.getOps<ModuleOp>()) {
    std::optional<mlir::pto::PTOBackend> childBackend;
    if (failed(parseDriverBackendAttr(child.getOperation(), childBackend)))
      return failure();

    mlir::pto::PTOBackend effectiveChildBackend =
        childBackend.value_or(defaultBackend);
    if (!firstChildBackend) {
      firstChildBackend = effectiveChildBackend;
      continue;
    }
    if (*firstChildBackend != effectiveChildBackend)
      return success();
  }

  if (firstChildBackend)
    singleBackend = *firstChildBackend;
  else
    singleBackend = defaultBackend;
  return success();
}

static LogicalResult buildBackendInfo(ModuleOp module, bool cliBackendSpecified,
                                      mlir::pto::BackendInfo &backendInfo) {
  backendInfo = mlir::pto::BackendInfo();
  if (!parseDriverBackend(mlir::pto::ptoBackend,
                          backendInfo.defaultBackend)) {
    llvm::errs() << "Error: invalid --pto-backend='" << mlir::pto::ptoBackend
                 << "'. Expected 'emitc' or 'vpto'.\n";
    return failure();
  }

  std::optional<mlir::pto::PTOBackend> moduleBackend;
  if (!cliBackendSpecified) {
    if (failed(parseDriverBackendAttr(module.getOperation(), moduleBackend)))
      return failure();
  }

  if (failed(resolveSingleBackend(cliBackendSpecified, moduleBackend,
                                  backendInfo.defaultBackend, module,
                                  backendInfo.singleBackend)))
    return failure();

  if (backendInfo.singleBackend) {
    backendInfo.requiresToolchain =
        *backendInfo.singleBackend == mlir::pto::PTOBackend::VPTO &&
        !mlir::pto::emitMlirIR && !mlir::pto::emitVPTO;
    return success();
  }

  if (mlir::pto::emitMlirIR || mlir::pto::emitVPTO ||
      mlir::pto::ptoPrintSeamIR || !mlir::pto::ptoSeamIRFile.empty()) {
    llvm::errs() << "Error: mixed pto.backend fatobj mode does not support "
                    "debug IR output flags.\n";
    return failure();
  }
  if (outputFilename.empty() || outputFilename == "-") {
    llvm::errs() << "Error: mixed pto.backend fatobj mode requires an "
                    "explicit file path passed with -o.\n";
    return failure();
  }

  backendInfo.requiresToolchain = true;
  return success();
}

static LogicalResult runPTOASJobs(OwningOpRef<ModuleOp> &module,
                                  PTOASContext &context,
                                  mlir::pto::PTOASCompileResult &result) {
  const mlir::pto::BackendInfo &backendInfo = context.getBackendInfo();
  if (backendInfo.singleBackend) {
    if (*backendInfo.singleBackend == mlir::pto::PTOBackend::EmitC) {
      EmitCBackendJob singleJob(module, result);
      return singleJob.run(context);
    }
    VPTOBackendJob singleJob(module, result);
    return singleJob.run(context);
  }

  SmallVector<std::unique_ptr<BackendChildJob>, 4> backendJobs;
  SmallVector<std::string, 4> fatobjPaths;
  if (failed(collectChildJobs(module.get(), backendInfo.defaultBackend,
                              context, fatobjPaths, backendJobs)))
    return failure();

  result.reset();
  result.kind = mlir::pto::PTOASCompileResultKind::MixedObject;

  for (size_t i = 0, e = backendJobs.size(); i < e; ++i) {
    if (failed(backendJobs[i]->run(context)))
      return failure();
  }

  FatobjLinkJob linkJob(fatobjPaths);
  if (failed(linkJob.run(context)))
    return failure();

  return success();
}

static LogicalResult writeTextOutput(llvm::StringRef output,
                                     llvm::StringRef outputPath) {
  std::error_code ec;
  llvm::ToolOutputFile outputFile(outputPath, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << ec.message() << "\n";
    return failure();
  }
  outputFile.os() << output;
  outputFile.os().flush();
  outputFile.keep();
  return success();
}

// PTOAS driver jobs:
// +----------------------------------------------------------+
// |                        .pto                              |
// +----------------------------------------------------------+
// +-------------+ +------------+ +------------+ +------------+
// | EmitC job   | | VPTO job   | | EmitC      | | VPTO       |
// |             | |            | | child job  | | child job  |
// |             | |            | +------------+ +------------+
// |             | |            | +---------------------------+
// |             | |            | | Fatobj link job           |
// +-------------+ +------------+ +---------------------------+
// +-------------+ +------------------------------------------+
// | C++ source  | |                fatobj                    |
// +-------------+ +------------------------------------------+
int main(int argc, char **argv) {
  DialectRegistry registry;
  mlir::pto::registerPTOASDialects(registry);
  mlir::pto::registerPTOASPassesAndCLOptions();
  llvm::cl::SetVersionPrinter(printPTOASVersion);

  const bool cliArchSpecified = hasCLIOption(argc, argv, "--pto-arch");
  const bool cliBackendSpecified = hasCLIOption(argc, argv, "--pto-backend");

  llvm::cl::ParseCommandLineOptions(argc, argv, "PTO Assembler (ptoas)\n");

  PTOASContext context(registry, outputFilename, argc, argv);
  context.initializeMLIRContext();

  std::unique_ptr<llvm::MemoryBuffer> inputBuffer = readInputBuffer();
  if (!inputBuffer)
    return 1;

  std::string arch;
  OwningOpRef<ModuleOp> module = loadInputModule(
      std::move(inputBuffer), context.getMLIRContext(), cliArchSpecified, arch);
  if (!module)
    return 1;
  context.setArch(std::move(arch));

  mlir::pto::BackendInfo backendInfo;
  if (failed(buildBackendInfo(module.get(), cliBackendSpecified, backendInfo)))
    return 1;
  context.setBackendInfo(std::move(backendInfo));
  (void)context.initializeEnvironment(context.getBackendInfo().requiresToolchain,
                                      llvm::errs());

  mlir::pto::PTOASCompileResult result;
  if (failed(runPTOASJobs(module, context, result)))
    return 1;

  if (result.kind == mlir::pto::PTOASCompileResultKind::Text)
    return failed(writeTextOutput(result.textOutput, context.getOutputPath()));
  if (result.kind == mlir::pto::PTOASCompileResultKind::MixedObject)
    return 0;

  llvm::errs() << "Error: unsupported ptoas compile result.\n";
  return 1;
}
