# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(_ptoas_transforms_private_link_libs)
if(TARGET vfsim::native_core)
  list(APPEND _ptoas_transforms_private_link_libs
    $<BUILD_INTERFACE:vfsim::native_core>)
endif()
if(TARGET vfsim::ir_planner)
  list(APPEND _ptoas_transforms_private_link_libs
    $<BUILD_INTERFACE:vfsim::ir_planner>)
endif()

# VMIToVPTO lowering internals are textual include units of VMIToVPTO.cpp.
# They are listed here so LLVM per-directory source check is satisfied and
# marked HEADER_FILE_ONLY so they are not compiled as separate translation
# units (VMIToVPTO.cpp includes them inside its own anonymous namespace).
set(_ptoas_vmi_split_units
  VMIToVPTO/VMIToVPTOConversionInternals.cpp
  VMIToVPTO/VMIToVPTOMemoryInternals.cpp
  VMIToVPTO/VMIToVPTOMaskInternals.cpp
  VMIToVPTO/VMIToVPTODataLayoutInternals.cpp
  VMIToVPTO/VMIToVPTOUnifiedMaskInternals.cpp
  VMIToVPTO/VMIToVPTOPatternInternals0.cpp
  VMIToVPTO/VMIToVPTOPatternInternals1.cpp
  VMIToVPTO/VMIToVPTOPatternInternals2.cpp
  VMIToVPTO/VMIToVPTOPatternInternals3.cpp
  VMIToVPTO/VMIToVPTOPatternInternals4.cpp
  VMIToVPTO/VMIToVPTOPatternInternals5.cpp
  VMIToVPTO/VMIToVPTOPatternInternals10.cpp
  VMIToVPTO/VMIToVPTOPatternInternals6.cpp
  VMIToVPTO/VMIToVPTOUnifiedPatternInternals.cpp
  VMIToVPTO/VMIToVPTOPatternInternals7.cpp
  VMIToVPTO/VMIToVPTOPatternInternals8.cpp
  VMIToVPTO/VMIToVPTOPatternInternals9.cpp
)
set_source_files_properties(${_ptoas_vmi_split_units} PROPERTIES HEADER_FILE_ONLY TRUE)

add_mlir_dialect_library(PTOTransforms
  ENABLE_AGGREGATION

  TileFusion/FusionAnalysis.cpp
  TileFusion/FusionOpSemantics.cpp
  TileFusion/PTOPreFusionAnalysis.cpp
  TileFusion/PTOPrintPreFusionAnalysis.cpp
  TileFusion/PTOFusionPlan.cpp
  TileFusion/PTOOpScheduling.cpp
  TileFusion/PTOMarkLastUse.cpp
  TileFusion/PTOFusionRegionGen.cpp
  TileFusion/PTOLowLevelLoopFusion.cpp
  TileFusion/PTOFusionPredicateElision.cpp
  TileFusion/PTOFusionLoadStoreElision.cpp
  TileFusion/PTOVexpdifFusion.cpp
  TileFusion/PTOUnrollAfterLoopFusion.cpp
  TileFusion/PTOFlattenFusionRegion.cpp
  VecScopeMemBar/VecScopeMemoryFootprint.cpp
  VecScopeMemBar/VecScopeMemBarIR.cpp
  VecScopeMemBar/VecScopeMemBarAnalysis.cpp
  VecScopeMemBar/VecScopeMemBarPlacement.cpp
  VecScopeMemBar/VecScopeMemBarRedundancy.cpp
  VecScopeMemBar/VecScopeMemBarCodegen.cpp
  VecScopeMemBar/PTOInsertVecScopeMemBarPass.cpp
  VecScopeMemBar/PTOInsertVecScopeMemBarAllPass.cpp
  VPTO/VPTOCANN900LLVMEmitterCommon.cpp
  VPTO/VPTOCANN900LLVMEmitterUbuf.cpp
  VPTO/VPTOCANN900LLVMEmitter.cpp
  VPTO/VPTOCANN900LLVMEmitterArithmeticPatterns.cpp
  VPTO/VPTOCANN900LLVMEmitterCalleeCore.cpp
  VPTO/VPTOCANN900LLVMEmitterCalleeMemory.cpp
  VPTO/VPTOCANN900LLVMEmitterMemoryPatterns.cpp
  VPTO/VPTOCANN900LLVMEmitterPacking.cpp
  VPTO/VPTOCANN900LLVMEmitterPipeline.cpp
  VPTO/VPTOCANN900LLVMEmitterScalarPatterns.cpp
  VPTO/VPTOCANN900LLVMEmitterTypeHelpers.cpp
  VPTO/VPTOCANN900LLVMEmitterTypePatterns.cpp
  VPTO/VPTOLLVMEmitterDispatcher.cpp
  VPTO/VPTOLLVMEmitterHelper.cpp
  VPTO/VPTOPtrNormalize.cpp
  VPTO/VPTOPtrCastCleanup.cpp
  VPTO/VPTOCombineReductions.cpp
  VPTO/VPTOOptimizeVcvt.cpp
  VPTO/VPTOMaskSimplify.cpp
  VPTOExpandMadOps.cpp
  VPTO/VPTOExpandAsyncComm.cpp
  VPTO/VPTOExpandWrapperOps.cpp
  VPTO/VPTOExpandL1L0a3d.cpp
  VPTOOptimizeCtrlState.cpp
  VPTO/VPTOSoftPostUpdate.cpp
  VPTO/VPTOGuardedLICM.cpp
  Passes/PTOPrintAddressAnalysis.cpp
  VPTOScheduler/VPTORegPressureTracker.cpp
  VPTOScheduler/VPTOSchedBoundary.cpp
  VPTOScheduler/VPTOSchedDAG.cpp
  VPTOScheduler/VPTOSchedDAGBuilder.cpp
  VPTOScheduler/VPTOSchedModel.cpp
  VPTOScheduler/VPTOSchedRegion.cpp
  VPTOScheduler/VPTOSchedResourceTracker.cpp
  VPTOScheduler/VPTOSchedStrategy.cpp
  VPTOScheduler/VPTOScheduler.cpp
  VPTOScheduler/VPTOSchedulerRematCandidates.cpp
  VPTOScheduler/VPTOSchedulerRematPlanner.cpp
  VPTOScheduler/VPTOSchedulerRematPressure.cpp
  VPTOScheduler/VPTOSchedulerRematerialization.cpp
  VPTOScheduler/VPTOSchedulerPass.cpp
  Passes/PTOVPTOPtrBoundary.cpp
  VPTO/VPTOBufferMaterialization.cpp
  VPTOBridgeWhitelist.cpp
  VPTOResolveBridgeInstances.cpp
  VPTOBridgeCallLowering.cpp
  VPTOPrintDTDataABI.cpp
  VPTOBridgeWrapperGen.cpp
  PTOLowerBridgeOps.cpp
  PTOLowerPipeFamilyOps.cpp
  VPTOBridgeTokens.cpp
  Passes/PTOValidateVPTOIR.cpp
  Passes/PTONarrowVPTOLoopCounters.cpp
  Passes/PTOConvertSCFToCFWithLoopHintsPass.cpp
  Passes/PTOPromotePersistentFragmentLoopsPass.cpp
  Passes/PTOUnrollLoopsPass.cpp
  Passes/PTOValidateVMIIR.cpp
  VMI/VMIExpandImplicitEnsureLayouts.cpp
  VMI/VMIPreAssignmentCombine.cpp
  VMI/VMIControlFlowSupport.cpp
  VMI/VMILegalizeArithSelect.cpp
  VMI/VMIMaskGranularityAssignment.cpp
  VMI/VMILowerUnifiedToLegacy.cpp
  VMI/VMINormalizeSignlessIntToUnsigned.cpp
  VMI/VMILayoutRematerializeWeakProducers.cpp
  VMI/VMILayoutAssignment.cpp
  VMI/VMILayoutSpineAnalysis.cpp
  VMI/VMILayoutFold.cpp
  VMI/VMILayoutPropagation.cpp
  VMI/VMILayoutSupport.cpp
  VMI/VMILayoutSupportAlignment.cpp
  VMI/VMILayoutSupportBF16.cpp
  VMI/VMILayoutSupportGroupSlots.cpp
  VMI/VMILayoutRematerialize.cpp
  VMI/VMILayoutSinkMaterialization.cpp
  VMIToVPTO/VMIToVPTO.cpp
  ${_ptoas_vmi_split_units}
  VPTO/VPTOStatefulStreamFusion.cpp
  Lowering/PTOExpandSoftLib.cpp
  Passes/SIMTPersistentFragmentAnalysis.cpp
  Passes/PTOAnalyzeSIMTPersistentFragment.cpp
  Passes/PTOMaterializeSIMTPersistentFragment.cpp
  Passes/PTOOutlineSIMTSections.cpp
  VPTO/PTOInferVPTOVecScope.cpp

  InsertSync/PTOInsertSync.cpp
  InsertSync/SyncScalarFacts.cpp
  InsertSync/SyncDescriptorState.cpp
  InsertSync/SyncInput.cpp
  InsertSync/SyncStorageEffects.cpp
  InsertSync/SyncAccessRegion.cpp
  InsertSync/SyncScalarConstraints.cpp
  InsertSync/SyncRegionOverlap.cpp
  InsertSync/SyncRegionDigits.cpp
  InsertSync/SyncEffectRanges.cpp
  # Session and dispatch
  FrontierSynch/AnalysisRequest.cpp
  FrontierSynch/FrontierAnalysis.cpp
  FrontierSynch/ProgramRecognition.cpp
  FrontierSynch/SessionCosts.cpp
  FrontierSynch/SessionIdentity.cpp
  FrontierSynch/SessionPreparation.cpp
  FrontierSynch/SessionProducers.cpp

  # Finite occurrences and finite guards
  FrontierSynch/Finite/ExplicitAnalysis.cpp
  FrontierSynch/Finite/ExplicitReduction.cpp
  FrontierSynch/Finite/FiniteAllocation.cpp
  FrontierSynch/Finite/FiniteAllocationDecode.cpp
  FrontierSynch/Finite/FiniteGuardedAnalysis.cpp
  FrontierSynch/Finite/FiniteGuardedInsertion.cpp
  FrontierSynch/Finite/FiniteGuardedQueries.cpp
  FrontierSynch/Finite/LifetimeScan.cpp

  # Bounded lifetime and periodic storage
  FrontierSynch/Structured/BoundedLifetime.cpp
  FrontierSynch/Structured/BoundedLifetimeAllocation.cpp
  FrontierSynch/Structured/BoundedLifetimeAllocationContext.cpp
  FrontierSynch/Structured/BoundedLifetimeRegionalAllocation.cpp
  FrontierSynch/Structured/BoundedLifetimeInsertion.cpp
  FrontierSynch/Structured/BoundedLifetimeRegional.cpp
  FrontierSynch/Structured/BoundedLifetimeSelectors.cpp
  FrontierSynch/Structured/GuardedPeriodicInsertion.cpp
  FrontierSynch/Structured/GuardedPeriodicQuotient.cpp
  FrontierSynch/Structured/GuardedRotatingAnalysis.cpp
  FrontierSynch/Structured/GuardedRotatingInsertion.cpp
  FrontierSynch/Structured/GuardedRotatingRegional.cpp
  FrontierSynch/Structured/LifetimeStream.cpp
  FrontierSynch/Structured/MixedStrideAnalysis.cpp
  FrontierSynch/Structured/MixedStrideInsertion.cpp
  FrontierSynch/Structured/NumericTemplate.cpp
  FrontierSynch/Structured/NumericTemplateAnalysis.cpp
  FrontierSynch/Structured/NumericTemplateControl.cpp
  FrontierSynch/Structured/NumericTemplateEffects.cpp
  FrontierSynch/Structured/NumericTemplateEndpoints.cpp
  FrontierSynch/Structured/NumericTemplateInsertion.cpp
  FrontierSynch/Structured/NumericTemplateRegional.cpp
  FrontierSynch/Structured/NumericTemplateStorage.cpp
  FrontierSynch/Structured/NumericTemplateRecurringGlobal.cpp
  FrontierSynch/Structured/PeriodicAllocation.cpp
  FrontierSynch/Structured/PeriodicAllocationQueries.cpp
  FrontierSynch/Structured/PeriodicDemandGraph.cpp
  FrontierSynch/Structured/PeriodicExcessCertificate.cpp
  FrontierSynch/Structured/PeriodicFrontier.cpp
  FrontierSynch/Structured/PeriodicQueries.cpp
  FrontierSynch/Structured/PeriodicSharedAllocation.cpp
  FrontierSynch/Structured/PeriodicSharedCertificate.cpp
  FrontierSynch/Structured/PhysicalRefresh.cpp
  FrontierSynch/Structured/ResolvedLifetimeStream.cpp
  FrontierSynch/Structured/OrderCoveredPeriodic.cpp
  FrontierSynch/Structured/RotatingAnalysis.cpp
  FrontierSynch/Structured/RotatingBoundary.cpp
  FrontierSynch/Structured/RotatingExtraction.cpp
  FrontierSynch/Structured/RotatingRegion.cpp
  FrontierSynch/Structured/RotatingRegionRecognition.cpp
  FrontierSynch/Structured/RotationFootprint.cpp
  FrontierSynch/Structured/RotationFragments.cpp
  FrontierSynch/Structured/RotationPattern.cpp
  FrontierSynch/Structured/RotationSelectors.cpp
  FrontierSynch/Structured/StorageLaneAllocation.cpp
  FrontierSynch/Structured/VaryingBoundaryQuery.cpp
  FrontierSynch/Structured/VaryingRotatingAllocation.cpp
  FrontierSynch/Structured/VaryingRotatingRecognition.cpp
  FrontierSynch/Structured/VaryingRotatingRegional.cpp

  # Restricted arithmetic
  FrontierSynch/Arithmetic/ArithmeticAccesses.cpp
  FrontierSynch/Arithmetic/ArithmeticDemandAnalysis.cpp
  FrontierSynch/Arithmetic/ArithmeticGuards.cpp
  FrontierSynch/Arithmetic/ArithmeticHandoffAllocation.cpp
  FrontierSynch/Arithmetic/ArithmeticInsertion.cpp
  FrontierSynch/Arithmetic/ArithmeticMachineScalars.cpp
  FrontierSynch/Arithmetic/ArithmeticPeriodicConversion.cpp
  FrontierSynch/Arithmetic/ArithmeticPeriodicImport.cpp
  FrontierSynch/Arithmetic/ArithmeticProgram.cpp
  FrontierSynch/Arithmetic/ArithmeticRecognition.cpp
  FrontierSynch/Arithmetic/ArithmeticRegional.cpp
  FrontierSynch/Arithmetic/ArithmeticRegionalComposition.cpp
  FrontierSynch/Arithmetic/ArithmeticResidues.cpp
  FrontierSynch/Arithmetic/ArithmeticRows.cpp
  FrontierSynch/Arithmetic/ArithmeticSelectors.cpp
  FrontierSynch/Arithmetic/ArithmeticStorageSelectors.cpp
  FrontierSynch/Arithmetic/DifferenceBoundRelations.cpp
  FrontierSynch/Arithmetic/GeneralArithmeticAllocation.cpp
  FrontierSynch/Arithmetic/GeneralArithmeticSelectors.cpp
  FrontierSynch/Arithmetic/IntegerImplicationCache.cpp
  FrontierSynch/Arithmetic/IntegerRelations.cpp

  # Shared composition, normalization, emission and allocation
  FrontierSynch/Shared/AllocationCertificate.cpp
  FrontierSynch/Shared/BalancedCompactBody.cpp
  FrontierSynch/Shared/BalancedCompactInsertion.cpp
  FrontierSynch/Shared/BoundaryExcess.cpp
  FrontierSynch/Shared/BoundingRegionalAnalysis.cpp
  FrontierSynch/Shared/BoundingRepeatedQuery.cpp
  FrontierSynch/Shared/BoundingRepetition.cpp
  FrontierSynch/Shared/BoundingSequence.cpp
  FrontierSynch/Shared/CertifiedPartialReduction.cpp
  FrontierSynch/Shared/ChainComposition.cpp
  FrontierSynch/Shared/CompletionTransfer.cpp
  FrontierSynch/Shared/CompletionRegion.cpp
  FrontierSynch/Shared/ChainInterface.cpp
  FrontierSynch/Shared/ChainMergeQueries.cpp
  FrontierSynch/Shared/ClosedCallees.cpp
  FrontierSynch/Shared/CompactAllocation.cpp
  FrontierSynch/Shared/CompactBoundaryRanks.cpp
  FrontierSynch/Shared/CompactBoundingAllocation.cpp
  FrontierSynch/Shared/CompactBoundingInsertion.cpp
  FrontierSynch/Shared/CompactClassRepetition.cpp
  FrontierSynch/Shared/CompactLowerFacts.cpp
  FrontierSynch/Shared/CompactOrderBounds.cpp
  FrontierSynch/Shared/CompactStorageBoundary.cpp
  FrontierSynch/Shared/CompactWriterReader.cpp
  FrontierSynch/Shared/CompactWriterReaderInput.cpp
  FrontierSynch/Shared/ConditionalCompactInput.cpp
  FrontierSynch/Shared/ControlOriginDistances.cpp
  FrontierSynch/Shared/EndpointCutChoices.cpp
  FrontierSynch/Shared/GuardedPublication.cpp
  FrontierSynch/Shared/EndpointFamilies.cpp
  FrontierSynch/Shared/EndpointPieces.cpp
  FrontierSynch/Shared/ExecutionContexts.cpp
  FrontierSynch/Shared/FamilyExpressions.cpp
  FrontierSynch/Shared/FamilyInsertion.cpp
  FrontierSynch/Shared/RegionalAllocationCandidates.cpp
  FrontierSynch/Shared/AllocationPreparation.cpp
  FrontierSynch/Shared/FamilyMaterialization.cpp
  FrontierSynch/Shared/FiniteExpansion.cpp
  FrontierSynch/Shared/FiniteOverlay.cpp
  FrontierSynch/Shared/FiniteOverlayAllocation.cpp
  FrontierSynch/Shared/FiniteOverlayInsertion.cpp
  FrontierSynch/Shared/FiniteRequirementReplacement.cpp
  FrontierSynch/Shared/FiniteVisitDemands.cpp
  FrontierSynch/Shared/FiniteVisitRecognition.cpp
  FrontierSynch/Shared/GuardedCompactMatching.cpp
  FrontierSynch/Shared/GuardedRanks.cpp
  FrontierSynch/Shared/GuardedRecognition.cpp
  FrontierSynch/Shared/HardwareProtection.cpp
  FrontierSynch/Shared/InvariantExpressions.cpp
  FrontierSynch/Shared/InvocationProtection.cpp
  FrontierSynch/Shared/LogicalEndpoints.cpp
  FrontierSynch/Shared/LogicalInsertion.cpp
  FrontierSynch/Shared/LogicalControlProfile.cpp
  FrontierSynch/Shared/PreparedControlLiveness.cpp
  FrontierSynch/Shared/ModularScalar.cpp
  FrontierSynch/Shared/NormalizedControl.cpp
  FrontierSynch/Shared/NormalizedScalar.cpp
  FrontierSynch/Shared/NumericalRepeatedBinding.cpp
  FrontierSynch/Shared/NumericalRepeatedExports.cpp
  FrontierSynch/Shared/NumericalRepeatedSquaring.cpp
  FrontierSynch/Shared/NumericalWeightedRepetition.cpp
  FrontierSynch/Shared/PhaseIndex.cpp
  FrontierSynch/Shared/PhaseNormalization.cpp
  FrontierSynch/Shared/PhasedRecognition.cpp
  FrontierSynch/Shared/SingletonBoundary.cpp
  FrontierSynch/Shared/PhysicalAllocation.cpp
  FrontierSynch/Shared/PhysicalExecutedCounters.cpp
  FrontierSynch/Shared/Recognition.cpp
  FrontierSynch/Shared/RegionExpressionContext.cpp
  FrontierSynch/Shared/RegionExpressionRelations.cpp
  FrontierSynch/Shared/RegionThresholdProof.cpp
  FrontierSynch/Shared/RegionExpressions.cpp
  FrontierSynch/Shared/RegionIntegerExpressions.cpp
  FrontierSynch/Shared/RegionSubstitution.cpp
  FrontierSynch/Shared/RegionalAllocation.cpp
  FrontierSynch/Shared/RegionalConstantPalette.cpp
  FrontierSynch/Shared/RegionalEvents.cpp
  FrontierSynch/Shared/RegionalLaneCertificate.cpp
  FrontierSynch/Shared/RegionalLaneExports.cpp
  FrontierSynch/Shared/RegionalPalette.cpp
  FrontierSynch/Shared/RegionalRelationCallbacks.cpp
  FrontierSynch/Shared/RegionalRelationNormalization.cpp
  FrontierSynch/Shared/RegionalRelations.cpp
  FrontierSynch/Shared/RepeatedAllocation.cpp
  FrontierSynch/Shared/RepeatedExcess.cpp
  FrontierSynch/Shared/RepeatedFiniteStorage.cpp
  FrontierSynch/Shared/RepeatedLaneAllocation.cpp
  FrontierSynch/Shared/RepeatedPhases.cpp
  FrontierSynch/Shared/RepeatedReadOnlyStorage.cpp
  FrontierSynch/Shared/RepeatedRecognition.cpp
  FrontierSynch/Shared/RepeatedRegion.cpp
  FrontierSynch/Shared/RepeatedRegionExports.cpp
  FrontierSynch/Shared/RepeatedRegionInsertion.cpp
  FrontierSynch/Shared/RepeatedCrossingFamilies.cpp
  FrontierSynch/Shared/RepeatedStorage.cpp
  FrontierSynch/Shared/RepeatedStorageGeometry.cpp
  FrontierSynch/Shared/RepeatedStorageIntegration.cpp
  FrontierSynch/Shared/RepeatedStorageRecognition.cpp
  FrontierSynch/Shared/RequirementProvenance.cpp
  FrontierSynch/Shared/SequenceAdapters.cpp
  FrontierSynch/Shared/SequenceAnalysis.cpp
  FrontierSynch/Shared/SequenceBoundary.cpp
  FrontierSynch/Shared/SequenceConditional.cpp
  FrontierSynch/Shared/SequenceFiniteFamilies.cpp
  FrontierSynch/Shared/SequenceInsertion.cpp
  FrontierSynch/Shared/BoundedPredicateHistory.cpp
  FrontierSynch/Shared/InvocationSourceBounds.cpp
  FrontierSynch/Shared/InvocationPrefixCollection.cpp
  FrontierSynch/Shared/PrefixExpression.cpp
  FrontierSynch/Shared/PrefixOccurrence.cpp
  FrontierSynch/Shared/PrefixPredicateSnapshots.cpp
  FrontierSynch/Shared/SequencePrefix.cpp
  FrontierSynch/Structured/BoundedPrefixRealization.cpp
  FrontierSynch/Shared/PrivateStatePlan.cpp
  FrontierSynch/Shared/InvocationPrefix.cpp
  FrontierSynch/Shared/StructuredPrivateState.cpp
  FrontierSynch/Shared/SequenceNumerical.cpp
  FrontierSynch/Shared/SequenceQueries.cpp
  FrontierSynch/Shared/SequenceStorage.cpp
  FrontierSynch/Shared/SharedHandoffAllocation.cpp
  Passes/PTOInjectBarrierAllSync.cpp
  InsertSync/InsertSyncDebug.cpp
  Passes/PTOMaterializeImplicitTmp.cpp
  Lowering/PTORematerializeFixpipeVectorQuant.cpp
  TileOp/TileLibService.cpp
  Support/SoftLibService.cpp
  TileOp/InsertTemplateAttributes.cpp
  TileOp/ExpandTileOp.cpp
  TileOp/FoldTileBufIntrinsics.cpp
  Lowering/LowerPTOTensorDump.cpp
  Lowering/LowerPTOToUBufOps.cpp
  TileOp/PTOLowerToOpLibCalls.cpp
  TileOp/PTOInstantiateAndInlineOpLib.cpp
  PTOToEmitC/PTOToEmitCCommon.cpp
  PTOToEmitC/PTOToEmitCTypeConverter.cpp
  PTOToEmitC/PTOToEmitCPass.cpp
  PTOToEmitC/Arith/ArithBinary.cpp
  PTOToEmitC/Arith/ArithCast.cpp
  PTOToEmitC/Arith/ArithCmp.cpp
  PTOToEmitC/Arith/ArithFloatMinMax.cpp
  PTOToEmitC/Arith/ArithIntBinary.cpp
  PTOToEmitC/Arith/ArithIntDiv.cpp
  PTOToEmitC/Arith/ArithIntMinMax.cpp
  PTOToEmitC/Arith/ArithMisc.cpp
  PTOToEmitC/Arith/ArithShift.cpp
  PTOToEmitC/Arith/ArithSupport.cpp
  PTOToEmitC/Arith/ArithPatterns.cpp
  PTOToEmitC/Tensor/TActivation.cpp
  PTOToEmitC/Tensor/TAdd.cpp
  PTOToEmitC/Tensor/TAnd.cpp
  PTOToEmitC/Tensor/TCI.cpp
  PTOToEmitC/Tensor/TCmp.cpp
  PTOToEmitC/Tensor/TColExpand.cpp
  PTOToEmitC/Tensor/TColReduce.cpp
  PTOToEmitC/Tensor/TConcat.cpp
  PTOToEmitC/Tensor/TCvt.cpp
  PTOToEmitC/Tensor/TDiv.cpp
  PTOToEmitC/Tensor/TElemwise.cpp
  PTOToEmitC/Tensor/TExp.cpp
  PTOToEmitC/Tensor/TExtractInsert.cpp
  PTOToEmitC/Tensor/TFillPad.cpp
  PTOToEmitC/Tensor/TGather.cpp
  PTOToEmitC/Tensor/TMinMax.cpp
  PTOToEmitC/Tensor/TMov.cpp
  PTOToEmitC/Tensor/TMul.cpp
  PTOToEmitC/Tensor/TQuant.cpp
  PTOToEmitC/Tensor/TRandom.cpp
  PTOToEmitC/Tensor/TSort.cpp
  PTOToEmitC/Tensor/TSub.cpp
  PTOToEmitC/Tensor/TTrans.cpp
  PTOToEmitC/Tensor/TTri.cpp
  PTOToEmitC/Tensor/TensorSupport.cpp
  PTOToEmitC/Tensor/TensorPatterns.cpp
  PTOToEmitC/Reduce/ArithCmpI.cpp
  PTOToEmitC/Reduce/ReduceMisc.cpp
  PTOToEmitC/Reduce/TInterleave.cpp
  PTOToEmitC/Reduce/TMath.cpp
  PTOToEmitC/Reduce/TPart.cpp
  PTOToEmitC/Reduce/TPrintTrap.cpp
  PTOToEmitC/Reduce/TRowExpand.cpp
  PTOToEmitC/Reduce/TRowReduce.cpp
  PTOToEmitC/Reduce/TScatter.cpp
  PTOToEmitC/Reduce/TSel.cpp
  PTOToEmitC/Reduce/TShiftScalar.cpp
  PTOToEmitC/Reduce/ReduceSupport.cpp
  PTOToEmitC/Reduce/ReducePatterns.cpp
  PTOToEmitC/ScalarMisc/AsyncSession.cpp
  PTOToEmitC/ScalarMisc/Comm.cpp
  PTOToEmitC/ScalarMisc/GlobalEventArray.cpp
  PTOToEmitC/ScalarMisc/LocalArray.cpp
  PTOToEmitC/ScalarMisc/RuntimeQuery.cpp
  PTOToEmitC/ScalarMisc/ScalarMiscMisc.cpp
  PTOToEmitC/ScalarMisc/ScalarMiscOps.cpp
  PTOToEmitC/ScalarMisc/ScalarPtr.cpp
  PTOToEmitC/ScalarMisc/Struct.cpp
  PTOToEmitC/ScalarMisc/ScalarMiscSupport.cpp
  PTOToEmitC/ScalarMisc/ScalarMiscPatterns.cpp
  PTOToEmitC/Tile/PartitionView.cpp
  PTOToEmitC/Tile/Section.cpp
  PTOToEmitC/Tile/TensorView.cpp
  PTOToEmitC/Tile/TileAlloc.cpp
  PTOToEmitC/Tile/TileMisc.cpp
  PTOToEmitC/Tile/TileImg2col.cpp
  PTOToEmitC/Tile/TileSupport.cpp
  PTOToEmitC/Tile/TilePatterns.cpp
  PTOToEmitC/SyncComm/BlockIdx.cpp
  PTOToEmitC/SyncComm/BufPool.cpp
  PTOToEmitC/SyncComm/FuncCall.cpp
  PTOToEmitC/SyncComm/Sync.cpp
  PTOToEmitC/SyncComm/SyncCommSupport.cpp
  PTOToEmitC/SyncComm/SyncCommPatterns.cpp
  PTOToEmitC/LoadStore/TLoad.cpp
  PTOToEmitC/LoadStore/TMatmul.cpp
  PTOToEmitC/LoadStore/TStore.cpp
  PTOToEmitC/LoadStore/LoadStoreSupport.cpp
  PTOToEmitC/LoadStore/LoadStorePatterns.cpp
  PTOToEmitC/Memref/ReinterpretCast.cpp
  PTOToEmitC/Memref/Subview.cpp
  PTOToEmitC/Memref/MemrefSupport.cpp
  PTOToEmitC/Memref/MemrefPatterns.cpp
  Lowering/CppPostprocess.cpp
  Support/Utils.cpp
  Passes/OptMemPlanForPipeline.cpp
  Passes/InferPTOMemScope.cpp
  Passes/PTOPlanMemoryModern.cpp
  Passes/PTORemoveRedundantBarrier.cpp
  Passes/InferPTOLayout.cpp
  Lowering/PTOA5NormalizeTMovPass.cpp
  Lowering/PTORemoveIdentityTMov.cpp
  Lowering/PTOArithRangeOptimize.cpp
  Lowering/PTOCanonicalizeIR.cpp
  Lowering/BufferizableOpInterfaceImpl.cpp
  Lowering/PTOLowerGenericOps.cpp
  Passes/PTOAssignDefaultFrontendPipeIdPass.cpp
  Lowering/PTOLowerFrontendPipeOpsPass.cpp
  Passes/PTOInferValidatePipeInitPass.cpp
  Passes/PTOResolveBufferSelect.cpp
  Passes/PTOResolveReservedBuffersPass.cpp
  Passes/SlotAffineAnalysis.cpp
  Passes/SubviewSlotAnalysis.cpp
  Passes/PTOWrapFunctionsInSectionsPass.cpp
  Passes/PTONormalizeUncoveredTileSections.cpp
  Passes/PTOValidatePhysicalSectionBoundaries.cpp
  Passes/PTOMaterializeTileOpSections.cpp
  VPTO/VPTONormalizeContainer.cpp
  VPTO/VPTOSplitCVModule.cpp
  InsertSync/PTOIRTranslator.cpp
  InsertSync/SyncMacroModel.cpp
  InsertSync/SyncCommon.cpp
  InsertSync/InsertSyncAnalysis.cpp
  InsertSync/MemoryDependentAnalyzer.cpp
  InsertSync/MoveSyncState.cpp
  InsertSync/RemoveRedundantSync.cpp
  InsertSync/RemoveSlotRedundantSync.cpp
  InsertSync/SyncEventIdAllocation.cpp
  InsertSync/SyncCodegen.cpp
  InsertSync/UnifiedSyncModel.cpp
  InsertSync/PTOUnifiedSync.cpp
  InsertSync/SyncOracleExtract.cpp
  InsertSync/SyncOracleGates.cpp
  BufidSync/BufidSyncPass.cpp
  BufidSync/BufidSyncAnalysis.cpp
  BufidSync/BufidSyncIdAlloc.cpp
  BufidSync/BufidSyncCodegen.cpp
  Passes/LoweringSyncToPipe.cpp
  Passes/PTOVerifyTFreePass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/PTO

  DEPENDS
  PTOPassesIncGen
  PTOOpsIncGen

  LINK_COMPONENTS
  Analysis

  LINK_LIBS
  PUBLIC
  PTOAnalysis
  PTOIR
  MLIRAnalysis
  MLIRPresburger
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRMathDialect
  MLIRArithDialect
  MLIRAffineDialect
  MLIREmitCDialect
  MLIRTargetCpp
  MLIRTransformUtils
  MLIRTransforms
  MLIRTensorDialect
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRVectorDialect
  MLIRParser
  MLIRArithTransforms
  MLIRSCFToEmitC
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRConvertToLLVMPass
  MLIRMathToLLVM
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
  PRIVATE
  ${_ptoas_transforms_private_link_libs}
)

install(TARGETS PTOTransforms
  EXPORT PTOASTargets      # 将此 Target 加入导出集
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

# 设置头文件包含路径，确保自动获得头文件路径
target_include_directories(PTOTransforms INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Sources grouped in subdirectories keep including headers that live directly in
# this directory by bare name (e.g. "Utils.h"), so keep it on the private
# include path.
target_include_directories(PTOTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(PTOTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Support)
# PTOValidateVPTOIR.cpp shares the SIMT keep/resume verify helpers with the
# PTO IR implementation (lib/PTO/IR/PTOPipeline); expose that internal dir.
target_include_directories(PTOTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../IR)
if(TARGET obj.PTOTransforms)
  target_include_directories(obj.PTOTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_include_directories(obj.PTOTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Support)
  target_include_directories(obj.PTOTransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../IR)
endif()

# MLIR conversion patterns legitimately hide base-class virtual
# matchAndRewrite overloads; upstream advice is to silence the warning.
# https://discourse.llvm.org/t/matchandrewrite-hiding-virtual-functions/84933/6
# (Replaces the per-file #pragma GCC diagnostic ignored "-Woverloaded-virtual"
# that PTOToEmitC.cpp carried before the split.)
if(TARGET obj.PTOTransforms)
  target_compile_options(obj.PTOTransforms PRIVATE -Wno-overloaded-virtual)
endif()
target_compile_options(PTOTransforms PRIVATE -Wno-overloaded-virtual)

# GCC 12/13 emit a false-positive -Warray-bounds when SmallVector's inline-buffer
# move assignment is inlined: moving the SyncModel temporary returned by
# unified::buildSyncModel() into the member at PTOUnifiedSync.cpp triggers a bogus
# "__builtin_memmove ... out of bounds" on the SmallVector<HiddenReservation>
# inline storage. The access is in-bounds; GCC models the small buffer wrongly. Keep
# the warning visible but non-fatal for just this one unit, mirroring the project's
# -Wno-error=deprecated-declarations policy for upstream/compiler churn. Clang does
# not emit this, so scope it to GNU.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set_source_files_properties(InsertSync/PTOUnifiedSync.cpp
    PROPERTIES COMPILE_OPTIONS "-Wno-error=array-bounds")
endif()

if(TARGET vfsim::native_core AND TARGET obj.PTOTransforms)
  # Preserve the simulator target's compile usage requirements on the object
  # library; its link dependency is already captured in aggregation metadata.
  target_link_libraries(obj.PTOTransforms PRIVATE
    $<BUILD_INTERFACE:vfsim::native_core>)
endif()

if(TARGET vfsim::ir_planner)
  target_compile_definitions(PTOTransforms PRIVATE PTO_ENABLE_VFSIM_IR_PLANNER=1)
  if(TARGET obj.PTOTransforms)
    target_link_libraries(obj.PTOTransforms PRIVATE
      $<BUILD_INTERFACE:vfsim::ir_planner>)
    target_compile_definitions(obj.PTOTransforms PRIVATE PTO_ENABLE_VFSIM_IR_PLANNER=1)
  endif()
endif()
