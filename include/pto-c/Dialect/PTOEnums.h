// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOEnums.h ---------------------------------------------------------===//
//
// C mirror of the PTO dialect enums generated in PTO/IR/PTOEnums.h.inc.
//
// The Python bindings translation unit (PTOModule.cpp) must be compilable
// online with only the shipped C-API headers (no LLVM/MLIR C++ tree). These
// mirror enums replace the C++ `enum class mlir::pto::*` types that pybind
// otherwise binds. Integer values MUST stay in lock-step with the generated
// C++ enums; lib/CAPI/Dialect/PTO.cpp holds static_asserts that break the
// build on any drift.
//
// CmpMode, Coalesce and MaskPattern are already mirrored in pto-c/Dialect/PTO.h
// and are intentionally NOT redefined here.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_PTO_ENUMS_H
#define MLIR_C_DIALECT_PTO_ENUMS_H

typedef enum MlirPTOAddressSpace {
  MlirPTOAddressSpace_Zero = 0,
  MlirPTOAddressSpace_GM = 1,
  MlirPTOAddressSpace_MAT = 2,
  MlirPTOAddressSpace_LEFT = 3,
  MlirPTOAddressSpace_RIGHT = 4,
  MlirPTOAddressSpace_ACC = 5,
  MlirPTOAddressSpace_VEC = 6,
  MlirPTOAddressSpace_BIAS = 7,
  MlirPTOAddressSpace_SCALING = 8,
} MlirPTOAddressSpace;

typedef enum MlirPTOFenceScope {
  MlirPTOFenceScope_LocalMemory = 0,
  MlirPTOFenceScope_GM = 1,
  MlirPTOFenceScope_All = 2,
} MlirPTOFenceScope;

typedef enum MlirPTOBLayout {
  MlirPTOBLayout_RowMajor = 0,
  MlirPTOBLayout_ColMajor = 1,
} MlirPTOBLayout;

typedef enum MlirPTOSLayout {
  MlirPTOSLayout_NoneBox = 0,
  MlirPTOSLayout_RowMajor = 1,
  MlirPTOSLayout_ColMajor = 2,
} MlirPTOSLayout;

typedef enum MlirPTOPadValue {
  MlirPTOPadValue_Null = 0,
  MlirPTOPadValue_Zero = 1,
  MlirPTOPadValue_Max = 2,
  MlirPTOPadValue_Min = 3,
} MlirPTOPadValue;

typedef enum MlirPTOCompactMode {
  MlirPTOCompactMode_Null = 0,
  MlirPTOCompactMode_Normal = 1,
  MlirPTOCompactMode_RowPlusOne = 2,
} MlirPTOCompactMode;

typedef enum MlirPTORoundMode {
  MlirPTORoundMode_NONE = 0,
  MlirPTORoundMode_RINT = 1,
  MlirPTORoundMode_ROUND = 2,
  MlirPTORoundMode_FLOOR = 3,
  MlirPTORoundMode_CEIL = 4,
  MlirPTORoundMode_TRUNC = 5,
  MlirPTORoundMode_ODD = 6,
  MlirPTORoundMode_CAST_RINT = 7,
} MlirPTORoundMode;

typedef enum MlirPTODivPrecision {
  MlirPTODivPrecision_Default = 0,
  MlirPTODivPrecision_HighPrecision = 1,
} MlirPTODivPrecision;

typedef enum MlirPTOExpPrecision {
  MlirPTOExpPrecision_Default = 0,
  MlirPTOExpPrecision_HighPrecision = 1,
} MlirPTOExpPrecision;

typedef enum MlirPTOLogPrecision {
  MlirPTOLogPrecision_Default = 0,
  MlirPTOLogPrecision_HighPrecision = 1,
} MlirPTOLogPrecision;

typedef enum MlirPTORecipPrecision {
  MlirPTORecipPrecision_Default = 0,
  MlirPTORecipPrecision_HighPrecision = 1,
} MlirPTORecipPrecision;

typedef enum MlirPTORemPrecision {
  MlirPTORemPrecision_Default = 0,
  MlirPTORemPrecision_HighPrecision = 1,
} MlirPTORemPrecision;

typedef enum MlirPTORsqrtPrecision {
  MlirPTORsqrtPrecision_Default = 0,
  MlirPTORsqrtPrecision_HighPrecision = 1,
} MlirPTORsqrtPrecision;

typedef enum MlirPTOSqrtPrecision {
  MlirPTOSqrtPrecision_Default = 0,
  MlirPTOSqrtPrecision_HighPrecision = 1,
} MlirPTOSqrtPrecision;

typedef enum MlirPTOFmodPrecision {
  MlirPTOFmodPrecision_Default = 0,
  MlirPTOFmodPrecision_HighPrecision = 1,
} MlirPTOFmodPrecision;

typedef enum MlirPTOSaturationMode {
  MlirPTOSaturationMode_ON = 0,
  MlirPTOSaturationMode_OFF = 1,
} MlirPTOSaturationMode;

typedef enum MlirPTOPIPE {
  MlirPTOPIPE_PIPE_S = 0,
  MlirPTOPIPE_PIPE_V = 1,
  MlirPTOPIPE_PIPE_M = 2,
  MlirPTOPIPE_PIPE_MTE1 = 3,
  MlirPTOPIPE_PIPE_MTE2 = 4,
  MlirPTOPIPE_PIPE_MTE3 = 5,
  MlirPTOPIPE_PIPE_ALL = 6,
  MlirPTOPIPE_PIPE_MTE4 = 7,
  MlirPTOPIPE_PIPE_MTE5 = 8,
  MlirPTOPIPE_PIPE_V2 = 9,
  MlirPTOPIPE_PIPE_FIX = 10,
  MlirPTOPIPE_VIRTUAL_PIPE_MTE2_L1A = 11,
  MlirPTOPIPE_VIRTUAL_PIPE_MTE2_L1B = 12,
  MlirPTOPIPE_PIPE_NUM = 13,
  MlirPTOPIPE_PIPE_UNASSIGNED = 99,
} MlirPTOPIPE;

typedef enum MlirPTOLayout {
  MlirPTOLayout_ND = 0,
  MlirPTOLayout_DN = 1,
  MlirPTOLayout_NZ = 2,
  MlirPTOLayout_MX_A_ZZ = 3,
  MlirPTOLayout_MX_B_NN = 4,
} MlirPTOLayout;

typedef enum MlirPTOAccToVecMode {
  MlirPTOAccToVecMode_SingleModeVec0 = 0,
  MlirPTOAccToVecMode_SingleModeVec1 = 1,
  MlirPTOAccToVecMode_DualModeSplitM = 2,
  MlirPTOAccToVecMode_DualModeSplitN = 3,
} MlirPTOAccToVecMode;

typedef enum MlirPTOTInsertMode {
  MlirPTOTInsertMode_SPLIT2 = 2,
  MlirPTOTInsertMode_SPLIT4 = 3,
} MlirPTOTInsertMode;

typedef enum MlirPTOReluPreMode {
  MlirPTOReluPreMode_NoRelu = 0,
  MlirPTOReluPreMode_NormalRelu = 1,
  MlirPTOReluPreMode_ScalarRelu = 2,
  MlirPTOReluPreMode_VectorRelu = 3,
  MlirPTOReluPreMode_Pwl = 4,
} MlirPTOReluPreMode;

typedef enum MlirPTOAtomicType {
  MlirPTOAtomicType_AtomicNone = 0,
  MlirPTOAtomicType_AtomicAdd = 1,
} MlirPTOAtomicType;

typedef enum MlirPTONotifyOp {
  MlirPTONotifyOp_AtomicAdd = 0,
  MlirPTONotifyOp_Set = 1,
} MlirPTONotifyOp;

typedef enum MlirPTOWaitCmp {
  MlirPTOWaitCmp_EQ = 0,
  MlirPTOWaitCmp_NE = 1,
  MlirPTOWaitCmp_GT = 2,
  MlirPTOWaitCmp_GE = 3,
  MlirPTOWaitCmp_LT = 4,
  MlirPTOWaitCmp_LE = 5,
} MlirPTOWaitCmp;

typedef enum MlirPTOReduceOp {
  MlirPTOReduceOp_Sum = 0,
  MlirPTOReduceOp_Max = 1,
  MlirPTOReduceOp_Min = 2,
} MlirPTOReduceOp;

typedef enum MlirPTOSyncOpType {
  MlirPTOSyncOpType_TLOAD = 0,
  MlirPTOSyncOpType_TSTORE_ACC = 1,
  MlirPTOSyncOpType_TSTORE_VEC = 2,
  MlirPTOSyncOpType_TMOV_M2L = 3,
  MlirPTOSyncOpType_TMOV_M2S = 4,
  MlirPTOSyncOpType_TMOV_M2B = 5,
  MlirPTOSyncOpType_TMOV_M2V = 6,
  MlirPTOSyncOpType_TMOV_V2M = 7,
  MlirPTOSyncOpType_TMATMUL = 8,
  MlirPTOSyncOpType_TVEC = 9,
  MlirPTOSyncOpType_TVECWAIT_EVENT = 10,
} MlirPTOSyncOpType;

typedef enum MlirPTOEVENT {
  MlirPTOEVENT_EVENT_ID0 = 0,
  MlirPTOEVENT_EVENT_ID1 = 1,
  MlirPTOEVENT_EVENT_ID2 = 2,
  MlirPTOEVENT_EVENT_ID3 = 3,
  MlirPTOEVENT_EVENT_ID4 = 4,
  MlirPTOEVENT_EVENT_ID5 = 5,
  MlirPTOEVENT_EVENT_ID6 = 6,
  MlirPTOEVENT_EVENT_ID7 = 7,
} MlirPTOEVENT;

typedef enum MlirPTOQuantType {
  MlirPTOQuantType_INT8_SYM = 0,
  MlirPTOQuantType_INT8_ASYM = 1,
  MlirPTOQuantType_MXFP8 = 2,
  MlirPTOQuantType_MXFP4_E2M1 = 3,
} MlirPTOQuantType;

typedef enum MlirPTOQuantScaleAlg {
  MlirPTOQuantScaleAlg_OCP = 0,
  MlirPTOQuantScaleAlg_NV = 1,
} MlirPTOQuantScaleAlg;

typedef enum MlirPTOMxGroupAxis {
  MlirPTOMxGroupAxis_Axis0 = 0,
  MlirPTOMxGroupAxis_Axis1 = 1,
} MlirPTOMxGroupAxis;

typedef enum MlirPTOVecStoreMode {
  MlirPTOVecStoreMode_ND = 0,
  MlirPTOVecStoreMode_NZ = 1,
} MlirPTOVecStoreMode;

#endif // MLIR_C_DIALECT_PTO_ENUMS_H
