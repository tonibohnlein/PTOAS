// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// mlir/include/mlir/Dialect/PTO/IR/PTOInterfaces.td
 
#ifndef MLIR_DIALECT_PTO_IR_PTOINTERFACES
#define MLIR_DIALECT_PTO_IR_PTOINTERFACES
 
include "mlir/IR/OpBase.td"
 
// 定义 OpPipeInterface
def OpPipeInterface : OpInterface<"OpPipeInterface"> {
  let description = [{
    Interface for operations to expose their execution pipeline (Pipe) on Ascend architecture.
    Together with MemoryEffectOpInterface this is the production ordinary
    synchronization contract: one physical pipeline and the declared byte
    reads/writes. Shared translation validates that every effect is retained.
    Macro models and explicit protocol/resource semantics take precedence.
    This compatibility contract does not qualify new hardware behavior.
  }];
  
  let cppNamespace = "::mlir::pto";
 
  let methods = [
    InterfaceMethod<
      "Get the hardware pipe where this operation executes.",
      "::mlir::pto::PIPE",  // 返回类型对应 PTOAttrs.td 中的定义
      "getPipe"
    >
  ];
}

// An explicit lowering-owned declaration, independent of autosync algorithms.
// False means that a richer phase/resource contract is still required.
def SyncStorageOpInterface : OpInterface<"SyncStorageOpInterface"> {
  let cppNamespace = "::mlir::pto";
  let description = [{
    Declares physical storage identity/geometry without issuing asynchronous
    payload work or accessing a private synchronization resource.
  }];
}

def SinglePhaseSyncOpInterface : OpInterface<"SinglePhaseSyncOpInterface"> {
  let cppNamespace = "::mlir::pto";
  let description = [{
    Optional refinement of the ordinary OpPipeInterface/MemoryEffect contract
    for operations with special lowering variants. False requires additional
    semantics; ordinary operations need not implement this interface. Shared
    translation, rather than a constructor opcode list, applies the restriction.
    Algorithms must also validate translated effects and target support.
  }];
  let methods = [InterfaceMethod<
    "Whether this variant supplies complete ordinary single-phase effects.",
    "bool", "hasCompleteSinglePhaseSyncEffects"
  >];
}

def SyncConfigurationOpInterface : OpInterface<"SyncConfigurationOpInterface"> {
  let cppNamespace = "::mlir::pto";
  let description = [{
    An operation-owned contract for preserved configuration, independent of
    the automatic synchronization algorithm. The operation qualifies its
    configuration lifetime against the original IR. Successful qualification
    permits keeping it in place without a local payload phase; it supplies no
    completion, event consumption, peer progress or memory visibility credit.
    Generic side-effect declarations remain authoritative for other passes.
  }];
  let methods = [InterfaceMethod<
    "Empty when the preserved configuration is qualified; otherwise the missing contract.",
    "std::string", "getSyncConfigurationGap"
  >];
}

def MacroSyncOpInterface : OpInterface<"MacroSyncOpInterface"> {
  let cppNamespace = "::mlir::pto";
  let description = [{
    Declares that the lowering-owned SyncMacroModel is the complete physical
    phase, storage-effect, and private-event description for this exact
    operation variant. Presence of a model without this interface is not a
    completeness claim.
  }];
}

def AuthoredSyncOpInterface : OpInterface<"AuthoredSyncOpInterface"> {
  let cppNamespace = "::mlir::pto";
  let description = [{
    Declares an authored synchronization or event-resource boundary. Automatic
    construction must preserve the operation and reserve any event identity it
    names; it may only take ordering credit justified by the operation's exact
    lowering contract.
  }];
}

def VisibilitySyncOpInterface : OpInterface<"VisibilitySyncOpInterface"> {
  let cppNamespace = "::mlir::pto";
  let description = [{
    Declares an authored cache-maintenance or fence boundary whose visibility
    effect is distinct from ordinary byte completion and resource exclusion.
  }];
}

// Instruction-class marker interfaces. These categories are independent from
// OpPipeInterface: a pipe describes where an operation executes, while these
// markers describe the operation's abstraction level and instruction family.
def TileOpInterface : OpInterface<"TileOpInterface"> {
  let description = [{
    Marks a high-level PTO Tile operation with tile-granular semantics.
  }];
  let cppNamespace = "::mlir::pto";
}

def MicroOpInterface : OpInterface<"MicroOpInterface"> {
  let description = [{
    Marks a low-level PTO micro-instruction that directly models a hardware
    instruction rather than a structural, view, or high-level Tile operation.
  }];
  let cppNamespace = "::mlir::pto";
}

def VectorMicroOpInterface : OpInterface<"VectorMicroOpInterface"> {
  let description = [{
    Marks a vector micro-instruction operating on vector registers, masks, or
    vector memory state.
  }];
  let cppNamespace = "::mlir::pto";
}

def CubeMicroOpInterface : OpInterface<"CubeMicroOpInterface"> {
  let description = [{
    Marks a Cube micro-instruction operating on the matrix execution unit.
  }];
  let cppNamespace = "::mlir::pto";
}

def MteOpInterface : OpInterface<"MteOpInterface"> {
  let description = [{
    Marks an operation that performs MTE/DMA data movement. High-level Tile
    operations and raw micro-instructions may both carry this marker.
  }];
  let cppNamespace = "::mlir::pto";
  let methods = [
    InterfaceMethod<"Return the source buffer.",
      "::mlir::Value", "getSource", (ins), [{}],
      /*defaultImplementation=*/[{
        auto operation = $_op.getOperation();
        if (auto dps = ::llvm::dyn_cast<::mlir::DestinationStyleOpInterface>(
                operation)) {
          auto inputs = dps.getDpsInputOperands();
          if (!inputs.empty())
            return inputs.front()->get();
        }
        return operation->getNumOperands() ? operation->getOperand(0)
                                           : ::mlir::Value();
      }]>,
    InterfaceMethod<"Return the destination buffer.",
      "::mlir::Value", "getDestination", (ins), [{}],
      /*defaultImplementation=*/[{
        auto operation = $_op.getOperation();
        if (auto dps = ::llvm::dyn_cast<::mlir::DestinationStyleOpInterface>(
                operation)) {
          auto inits = dps.getDpsInitOperands();
          if (!inits.empty())
            return inits.front()->get();
        }
        return operation->getNumOperands() > 1 ? operation->getOperand(1)
                                              : ::mlir::Value();
      }]>
  ];
}

def SimtOpInterface : OpInterface<"SimtOpInterface"> {
  let description = [{
    Marks an instruction executed directly by a SIMT entry. This excludes the
    pto.simt_launch boundary operation.
  }];
  let cppNamespace = "::mlir::pto";
}

// PTO destination-style ops: have "init" operands (e.g. $dst) for plan-memory.
// No tensor/memref type requirement (unlike MLIR DestinationStyleOpInterface).
// Each op implements getDpsInitsMutable(); this interface provides getDpsInits().
def PTO_DpsInitOpInterface : OpInterface<"PTO_DpsInitOpInterface"> {
  let description = [{
    PTO ops that have destination (init) operands for memory planning. Init
    operands may be tile_buf, partition_tensor_view, tensor, or memref. Used by
    PTOPlanMemory to get written-to buffers without MLIR's tensor/memref-only
    verification.
  }];
  let cppNamespace = "::mlir::pto";
  let methods = [
    InterfaceMethod<
      "Return the mutable range of destination (init) operands.",
      "::mlir::MutableOperandRange",
      "getDpsInitsMutable",
      (ins)
    >
  ];
  let extraSharedClassDeclaration = [{
    ::mlir::OperandRange getDpsInits() {
      return $_op.getDpsInitsMutable();
    }
  }];
}

#endif // MLIR_DIALECT_PTO_IR_PTOINTERFACES
