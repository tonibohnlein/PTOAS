// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/SlotAffineAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"

#include <iostream>
#include <string_view>

namespace {

using namespace mlir;
using namespace mlir::pto;

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "PTOIRTranslatorTest failure: " << message << '\n';
  }
  return condition;
}

bool testStrictUnknownMemoryHelperFailsClosed() {
  MLIRContext context;
  context.loadDialect<PTODialect, arith::ArithDialect, func::FuncDialect,
                      scf::SCFDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @entry(%arg0: !pto.ptr<f32, gm>) attributes {pto.entry} {
        func.call @unknown_helper(%arg0) : (!pto.ptr<f32, gm>) -> ()
        return
      }
      func.func private @unknown_helper(!pto.ptr<f32, gm>)
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse strict helper fixture")) {
    return false;
  }

  func::FuncOp entry = module->lookupSymbol<func::FuncOp>("entry");
  SyncIRs syncIR;
  Buffer2MemInfoMap bufferMap;
  OperationMemInfoStorage operationStorage;
  PTOIRTranslatorOptions options;
  options.includeExtendedEffects = true;
  options.failOnUnmodeledEffects = true;

  bool sawExpectedDiagnostic = false;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
    sawExpectedDiagnostic |=
        diagnostic.str().find("unrecognized helper") != std::string::npos;
    return success();
  });
  PTOIRTranslator translator(syncIR, bufferMap, operationStorage, entry,
                             options);
  return check(failed(translator.Build()),
               "strict extraction rejects unknown memory helper") &&
         check(sawExpectedDiagnostic, "strict rejection is diagnostic");
}

bool testStrictUnknownNoArgumentHelperFailsClosed() {
  MLIRContext context;
  context.loadDialect<PTODialect, arith::ArithDialect, func::FuncDialect,
                      scf::SCFDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @entry() attributes {pto.entry} {
        func.call @unknown_helper() : () -> ()
        return
      }
      func.func private @unknown_helper()
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse no-argument helper fixture")) {
    return false;
  }

  func::FuncOp entry = module->lookupSymbol<func::FuncOp>("entry");
  SyncIRs syncIR;
  Buffer2MemInfoMap bufferMap;
  OperationMemInfoStorage operationStorage;
  PTOIRTranslatorOptions options;
  options.includeExtendedEffects = true;
  options.failOnUnmodeledEffects = true;
  PTOIRTranslator translator(syncIR, bufferMap, operationStorage, entry,
                             options);
  return check(failed(translator.Build()),
               "strict extraction rejects unknown no-argument helper");
}

bool testStrictMultiTileHelperRequiresEffects() {
  MLIRContext context;
  context.loadDialect<PTODialect, arith::ArithDialect, func::FuncDialect,
                      scf::SCFDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @entry(%tiles: !pto.multi_tile_buf<vec, 16x16xf16, count=2>)
          attributes {pto.entry} {
        func.call @bad_helper(%tiles)
            : (!pto.multi_tile_buf<vec, 16x16xf16, count=2>) -> ()
        return
      }
      func.func private @bad_helper(
          !pto.multi_tile_buf<vec, 16x16xf16, count=2>)
          attributes {pto.tileop.helper, pto.tileop.kind = "vector"}
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse multi-tile helper fixture")) {
    return false;
  }

  func::FuncOp entry = module->lookupSymbol<func::FuncOp>("entry");
  SyncIRs syncIR;
  Buffer2MemInfoMap bufferMap;
  OperationMemInfoStorage operationStorage;
  PTOIRTranslatorOptions options;
  options.includeExtendedEffects = true;
  options.failOnUnmodeledEffects = true;
  PTOIRTranslator translator(syncIR, bufferMap, operationStorage, entry,
                             options);
  return check(failed(translator.Build()),
               "strict extraction recognizes multi-tile memory operands");
}

bool testShiftedSlotComparisonFailsClosed() {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, func::FuncDialect>();
  const Location location = UnknownLoc::get(&context);
  OpBuilder builder(&context);
  auto module = ModuleOp::create(location);
  auto function = func::FuncOp::create(
      location, "slot_forms",
      builder.getFunctionType({builder.getIndexType(), builder.getIndexType()},
                              {}));
  module.push_back(function);
  Block *body = function.addEntryBlock();
  builder.setInsertionPointToStart(body);
  Value two = builder.create<arith::ConstantIndexOp>(location, 2);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  Value lhs =
      builder.create<arith::RemUIOp>(location, body->getArgument(0), two);
  Value rhs =
      builder.create<arith::RemUIOp>(location, body->getArgument(0), two);
  Value raw =
      builder.create<arith::AddIOp>(location, body->getArgument(0), one);
  builder.create<func::ReturnOp>(location);

  return check(compareSlotSSAWithOffset(lhs, rhs, 2, {}, 1) ==
                   SlotRelation::kUnknown,
               "missing shifted symbol is unknown") &&
         check(compareSlotSSAWithOffset(lhs, rhs, 2, body->getArgument(1), 1) ==
                   SlotRelation::kUnknown,
               "wrong shifted symbol is unknown") &&
         check(compareSlotSSAWithOffset(lhs, rhs, 2, body->getArgument(0), 1) ==
                   SlotRelation::kDisjoint,
               "normalized shifted forms prove disjointness") &&
         check(compareSlotSSAWithOffset(raw, raw, 2, body->getArgument(0), 1) ==
                   SlotRelation::kUnknown,
               "unnormalized shifted forms are unknown") &&
         check(!enumerateSlotSSAOrdinalPairs(lhs, rhs, 2, {}, 1),
               "ordinal enumeration rejects a missing shifted symbol");
}

bool testStrictMalformedHelperEffectsFailClosed() {
  MLIRContext context;
  context.loadDialect<PTODialect, arith::ArithDialect, func::FuncDialect,
                      scf::SCFDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @entry(%arg0: !pto.ptr<f32, gm>) attributes {pto.entry} {
        func.call @bad_helper(%arg0) : (!pto.ptr<f32, gm>) -> ()
        return
      }
      func.func private @bad_helper(!pto.ptr<f32, gm>)
          attributes {pto.tileop.helper, pto.tileop.kind = "vector"}
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse malformed helper fixture")) {
    return false;
  }

  func::FuncOp entry = module->lookupSymbol<func::FuncOp>("entry");
  SyncIRs syncIR;
  Buffer2MemInfoMap bufferMap;
  OperationMemInfoStorage operationStorage;
  PTOIRTranslatorOptions options;
  options.includeExtendedEffects = true;
  options.failOnUnmodeledEffects = true;
  PTOIRTranslator translator(syncIR, bufferMap, operationStorage, entry,
                             options);
  return check(failed(translator.Build()),
               "strict extraction rejects malformed helper effects");
}

bool testStrictRecognizedExtendedEffectSucceeds() {
  MLIRContext context;
  context.loadDialect<PTODialect, arith::ArithDialect, func::FuncDialect,
                      scf::SCFDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @entry(%scale: !pto.tile_buf<loc=scaling, dtype=f16,
          rows=1, cols=32, v_row=1, v_col=32, blayout=row_major,
          slayout=none_box, fractal=512, pad=0>) attributes {pto.entry} {
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse extended-effect fixture")) {
    return false;
  }

  func::FuncOp entry = module->lookupSymbol<func::FuncOp>("entry");
  Block &body = entry.getBody().front();
  OpBuilder builder(body.getTerminator());
  OperationState state(entry.getLoc(), SetQuantVectorOp::getOperationName());
  state.addOperands(body.getArgument(0));
  state.addAttribute("id", builder.getI64IntegerAttr(0));
  builder.create(state);
  SyncIRs syncIR;
  Buffer2MemInfoMap bufferMap;
  OperationMemInfoStorage operationStorage;
  PTOIRTranslatorOptions options;
  options.includeExtendedEffects = true;
  options.failOnUnmodeledEffects = true;
  PTOIRTranslator translator(syncIR, bufferMap, operationStorage, entry,
                             options);
  return check(succeeded(translator.Build()),
               "strict extraction accepts modeled extended effects") &&
         check(syncIR.size() == 1,
               "modeled extended effect produces one synchronization node");
}

} // namespace

int main() {
  const bool passed = testStrictUnknownMemoryHelperFailsClosed() &&
                      testStrictUnknownNoArgumentHelperFailsClosed() &&
                      testStrictMultiTileHelperRequiresEffects() &&
                      testShiftedSlotComparisonFailsClosed() &&
                      testStrictMalformedHelperEffectsFailClosed() &&
                      testStrictRecognizedExtendedEffectSucceeds();
  return passed ? 0 : 1;
}
