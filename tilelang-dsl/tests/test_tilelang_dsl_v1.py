# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import tempfile
import unittest
from unittest import mock
from importlib import util
from pathlib import Path

import tilelang_dsl as pto
import tilelang_dsl.expand_helper as expand_helper
import tilelang_dsl.kernel as kernel_impl
from tilelang_dsl.support_matrix import (
    ADVANCED_EXPLICIT_VECSCOPE_SURFACES,
    ADVANCED_LOW_LEVEL_DMA_SURFACES,
    ADVANCED_RAW_POINTER_SURFACES,
    ADVANCED_TILE_HELPER_SURFACES,
    ADVANCED_TIER,
    AUTHORING_TIER_SURFACE_GROUPS,
    BASIC_TIER,
    BASIC_TILE_INDEXING_SURFACES,
    ADVANCED_VECSCOPE_PTO_CALLS,
    INFERRED_VECSCOPE_ACTIVITY_PTO_CALLS,
    INFERRED_VECSCOPE_NEUTRAL_PTO_CALLS,
    SUPPORTED_VECSCOPE_PTO_CALLS,
    get_feature_tier,
    get_surface_group_tier,
)
from tilelang_dsl.frontend_ast import (
    FrontendAssignStmt,
    FrontendCallExpr,
    FrontendExprStmt,
    FrontendForStmt,
    FrontendIfStmt,
    FrontendStrictVecscopeStmt,
    FrontendVecscopeStmt,
    FrontendNoOpStmt,
    build_frontend_kernel_node,
)
from tilelang_dsl.lowering import AuthoringModule, lower_semantic_kernel
from tilelang_dsl.semantic import (
    SemanticAlignStoreStmt,
    SemanticAlignType,
    SemanticAssignStmt,
    SemanticBindingRef,
    SemanticBinaryExpr,
    SemanticCallExpr,
    SemanticDmaConfigStmt,
    SemanticDmaUnaryConfigStmt,
    SemanticExprStmt,
    SemanticForStmt,
    SemanticGetBufStmt,
    SemanticIfStmt,
    SemanticIndexType,
    SemanticLiteralExpr,
    SemanticMemBarStmt,
    SemanticLowLevelCopyStmt,
    SemanticMaskType,
    SemanticPadValueType,
    SemanticPartitionTensorViewType,
    SemanticPipeBarrierStmt,
    SemanticPtrType,
    SemanticPredicateStoreStmt,
    SemanticReturnStmt,
    SemanticRlsBufStmt,
    SemanticScalarStoreStmt,
    SemanticScalarType,
    SemanticSetCrossCoreStmt,
    SemanticSetFlagStmt,
    SemanticSetIntraBlockStmt,
    SemanticSetIntraCoreStmt,
    SemanticStrictVecscopeStmt,
    SemanticSymbolExpr,
    SemanticTensorViewType,
    SemanticTileConfigType,
    SemanticTileType,
    SemanticVecscopeStmt,
    SemanticVScatterStmt,
    SemanticVectorPairStoreStmt,
    SemanticVectorStoreStmt,
    SemanticVRegType,
    SemanticWaitFlagDevStmt,
    SemanticWaitFlagStmt,
    SemanticWaitIntraCoreStmt,
    analyze_frontend_kernel,
)

GLOBAL_TILELANG_LITERAL_BLOCK_SIZE = 32
INLINE_PROC_GLOBAL_LANE = 0


def _walk_semantic_stmts(statements):
    for stmt in statements:
        yield stmt
        if isinstance(stmt, SemanticVecscopeStmt):
            yield from _walk_semantic_stmts(stmt.body)
        elif isinstance(stmt, SemanticForStmt):
            yield from _walk_semantic_stmts(stmt.body)
        elif isinstance(stmt, SemanticIfStmt):
            yield from _walk_semantic_stmts(stmt.then_body)
            yield from _walk_semantic_stmts(stmt.else_body)


def _find_inline_helper(semantic_kernel, symbol_prefix):
    return next(
        helper for helper in semantic_kernel.inline_helpers if helper.symbol_name.startswith(symbol_prefix)
    )


def _find_helper_assign_by_ssa(helper, ssa_name):
    return next(
        stmt
        for stmt in helper.body
        if isinstance(stmt, SemanticAssignStmt)
        and any(target.ssa_name == ssa_name for target in stmt.targets)
    )


def _find_last_helper_assign_by_name(helper, name):
    return next(
        stmt
        for stmt in reversed(helper.body)
        if isinstance(stmt, SemanticAssignStmt)
        and any(target.name == name for target in stmt.targets)
    )


def _find_helper_return_stmt(helper):
    return next(stmt for stmt in helper.body if isinstance(stmt, SemanticReturnStmt))


def _resolve_helper_expr(helper, expr):
    if isinstance(expr, SemanticBindingRef):
        assign = _find_helper_assign_by_ssa(helper, expr.binding.ssa_name)
        return _resolve_helper_expr(helper, assign.value)
    return expr


def _resolve_helper_broadcast_scalar_literal(helper, expr):
    resolved = _resolve_helper_expr(helper, expr)
    if isinstance(resolved, SemanticLiteralExpr):
        return resolved.value
    if isinstance(resolved, SemanticCallExpr) and resolved.namespace == "pto" and resolved.name == "vbr":
        return _resolve_helper_broadcast_scalar_literal(helper, resolved.args[0])
    raise AssertionError(f"expected helper scalar literal or broadcast, got {resolved!r}")


class TileLangDSLPackageTests(unittest.TestCase):
    def test_package_exports_surface(self) -> None:
        self.assertIsNotNone(pto.__file__)
        self.assertTrue(hasattr(pto, "vkernel"))
        self.assertTrue(hasattr(pto, "KernelRegistry"))
        self.assertTrue(hasattr(pto, "select_kernel"))
        self.assertTrue(hasattr(pto, "TensorView"))
        self.assertTrue(hasattr(pto, "Tile"))
        self.assertTrue(hasattr(pto, "TileSpecialization"))
        self.assertTrue(hasattr(pto, "PointerType"))
        self.assertTrue(hasattr(pto, "VRegType"))
        self.assertTrue(hasattr(pto, "MaskType"))
        self.assertTrue(hasattr(pto, "AlignType"))
        self.assertTrue(hasattr(pto, "ptr"))
        self.assertTrue(hasattr(pto, "vreg"))
        self.assertTrue(hasattr(pto, "align"))
        self.assertTrue(hasattr(pto, "mask_b8"))
        self.assertTrue(hasattr(pto, "mask_b16"))
        self.assertTrue(hasattr(pto, "mask_b32"))
        self.assertTrue(hasattr(pto, "constexpr"))
        self.assertTrue(hasattr(pto, "bytewidth"))
        self.assertTrue(hasattr(pto, "get_lanes"))
        self.assertTrue(hasattr(pto, "elements_per_vreg"))
        self.assertTrue(hasattr(pto, "PAT"))
        self.assertTrue(hasattr(pto, "PredicateDist"))
        self.assertTrue(hasattr(pto, "PadMode"))
        self.assertTrue(hasattr(pto, "BarrierType"))
        self.assertTrue(hasattr(pto, "BLayout"))
        self.assertTrue(hasattr(pto, "DeinterleaveDist"))
        self.assertTrue(hasattr(pto, "InterleaveDist"))
        self.assertTrue(hasattr(pto, "PositionMode"))
        self.assertTrue(hasattr(pto, "OrderMode"))
        self.assertTrue(hasattr(pto, "PadValue"))
        self.assertTrue(hasattr(pto, "VcvtRoundMode"))
        self.assertTrue(hasattr(pto, "VcvtSatMode"))
        self.assertTrue(hasattr(pto, "VcvtPartMode"))
        self.assertTrue(hasattr(pto, "PostUpdateMode"))
        self.assertTrue(hasattr(pto, "SLayout"))
        self.assertTrue(hasattr(pto, "PIPE"))
        self.assertTrue(hasattr(pto, "EVENT"))
        self.assertTrue(hasattr(pto, "si8"))
        self.assertTrue(hasattr(pto, "ui8"))
        self.assertTrue(hasattr(pto, "si16"))
        self.assertTrue(hasattr(pto, "ui16"))
        self.assertTrue(hasattr(pto, "si32"))
        self.assertTrue(hasattr(pto, "ui32"))
        self.assertTrue(hasattr(pto, "si64"))
        self.assertTrue(hasattr(pto, "ui64"))
        self.assertEqual(pto.BarrierType.VST_VLD.value, "VST_VLD")
        self.assertEqual(pto.BarrierType.VST_VST.value, "VST_VST")
        self.assertEqual(pto.BarrierType.VS_ALL.value, "VS_ALL")
        self.assertEqual(pto.BarrierType.VST_LD.value, "VST_LD")
        self.assertEqual(pto.BarrierType.VLD_ST.value, "VLD_ST")
        self.assertEqual(pto.BarrierType.VST_ST.value, "VST_ST")
        self.assertEqual(pto.BarrierType.SV_ALL.value, "SV_ALL")
        self.assertEqual(pto.BarrierType.ST_VLD.value, "ST_VLD")
        self.assertEqual(pto.BarrierType.LD_VST.value, "LD_VST")
        self.assertEqual(pto.BarrierType.ST_VST.value, "ST_VST")
        self.assertEqual(pto.PadMode.PadNull.value, "PadNull")
        self.assertEqual(pto.PadMode.PadFirstElem.value, "PadFirstElem")
        self.assertEqual(pto.PadMode.PadValue.value, "PadValue")
        self.assertEqual(pto.BLayout.ROW_MAJOR.value, "row_major")
        self.assertEqual(pto.SLayout.NONE_BOX.value, "none_box")
        self.assertEqual(pto.PadValue.NULL.encoded, 0)
        self.assertEqual(pto.PadValue.ZERO.encoded, 1)
        self.assertEqual(pto.PadValue.MAX.encoded, 2)
        self.assertEqual(pto.PadValue.MIN.encoded, 3)
        self.assertEqual(pto.PadValue.NULL.text, "null")
        self.assertEqual(pto.DeinterleaveDist.DINTLV.value, "DINTLV")
        self.assertEqual(pto.DeinterleaveDist.BDINTLV.value, "BDINTLV")
        self.assertEqual(pto.InterleaveDist.INTLV.value, "INTLV")
        self.assertEqual(pto.PositionMode.LOWEST.value, "LOWEST")
        self.assertEqual(pto.PositionMode.HIGHEST.value, "HIGHEST")
        self.assertEqual(pto.OrderMode.ASC.value, "ASC")
        self.assertEqual(pto.OrderMode.DESC.value, "DESC")
        self.assertEqual(pto.PredicateDist.NORM.value, "NORM")
        self.assertEqual(pto.PredicateDist.US.value, "US")
        self.assertEqual(pto.PredicateDist.DS.value, "DS")
        self.assertEqual(pto.PredicateDist.PK.value, "PK")
        self.assertTrue(hasattr(pto, "PredicatePart"))
        self.assertEqual(pto.PredicatePart.LOWER.value, "LOWER")
        self.assertEqual(pto.PredicatePart.HIGHER.value, "HIGHER")
        self.assertTrue(hasattr(pto, "CmpMode"))
        self.assertEqual(pto.CmpMode.EQ.value, "eq")
        self.assertEqual(pto.CmpMode.NE.value, "ne")
        self.assertEqual(pto.CmpMode.LT.value, "lt")
        self.assertEqual(pto.CmpMode.LE.value, "le")
        self.assertEqual(pto.CmpMode.GT.value, "gt")
        self.assertEqual(pto.CmpMode.GE.value, "ge")
        self.assertEqual(pto.VcvtRoundMode.R.value, "R")
        self.assertEqual(pto.VcvtSatMode.SAT.value, "SAT")
        self.assertEqual(pto.VcvtPartMode.EVEN.value, "EVEN")
        self.assertEqual(pto.VcvtPartMode.ODD.value, "ODD")
        self.assertEqual(pto.VcvtPartMode.P0.value, "P0")
        self.assertEqual(pto.VcvtPartMode.P1.value, "P1")
        self.assertEqual(pto.VcvtPartMode.P2.value, "P2")
        self.assertEqual(pto.VcvtPartMode.P3.value, "P3")
        self.assertEqual(pto.PostUpdateMode.POST_UPDATE.value, "POST_UPDATE")
        self.assertEqual(pto.PostUpdateMode.NO_POST_UPDATE.value, "NO_POST_UPDATE")
        self.assertEqual(pto.Event.ID31.value, "EVENT_ID31")
        self.assertIs(pto.DeinterleaveDist.B32, pto.DeinterleaveDist.DINTLV)
        self.assertIs(pto.InterleaveDist.B32, pto.InterleaveDist.INTLV)
        self.assertEqual(pto.si8.name, "si8")
        self.assertEqual(pto.ui16.name, "ui16")
        self.assertEqual(pto.si32.name, "si32")
        self.assertEqual(pto.ui64.name, "ui64")
        self.assertIsNot(pto.si8, pto.i8)
        self.assertIsNot(pto.ui32, pto.i32)
        self.assertEqual(pto.bytewidth(pto.si16), 2)
        self.assertEqual(pto.bytewidth(pto.ui64), 8)
        self.assertEqual(pto.get_lanes(pto.ui32), 64)
        self.assertEqual(pto.get_lanes(pto.i64), 32)
        self.assertEqual(pto.elements_per_vreg(pto.si8), 256)
        self.assertEqual(repr(pto.align), "align")

    def test_tile_config_exposes_normalized_query_properties(self) -> None:
        default_config = pto.TileConfig()
        self.assertEqual(default_config.b_layout, pto.BLayout.ROW_MAJOR)
        self.assertEqual(default_config.s_layout, pto.SLayout.NONE_BOX)
        self.assertEqual(default_config.s_fractal_size, 512)
        self.assertEqual(default_config.pad_value, pto.PadValue.NULL)

        config = pto.TileConfig.from_mapping(
            {
                "layout": "col_major",
                "s_layout": "row_major",
                "fractal": 16,
                "pad": "max",
            }
        )
        self.assertEqual(config.b_layout, pto.BLayout.COL_MAJOR)
        self.assertEqual(config.s_layout, pto.SLayout.ROW_MAJOR)
        self.assertEqual(config.s_fractal_size, 16)
        self.assertEqual(config.pad_value, pto.PadValue.MAX)

    def test_pad_value_supports_standard_and_custom_payloads(self) -> None:
        custom = pto.PadValue.custom_f32(-1.0)
        self.assertTrue(custom.is_custom)
        self.assertEqual(custom.float32_bits, 0xBF800000)
        self.assertEqual(custom.encoded, pto.PadValue.CustomBase | (0xBF800000 << 32))
        self.assertAlmostEqual(custom.as_float32(), -1.0)
        self.assertAlmostEqual(custom.eval(pto.f32), -1.0)
        self.assertEqual(pto.PadValue.MAX.eval(pto.ui16), 0xFFFF)
        self.assertEqual(pto.PadValue.MIN.eval(pto.ui16), 0)
        self.assertEqual(pto.PadValue.MAX.eval(pto.i16), 0x7FFF)
        self.assertEqual(pto.PadValue.MIN.eval(pto.i16), -0x8000)
        self.assertIsNone(pto.PadValue.NULL.eval(pto.f16))
        with self.assertRaises(AttributeError):
            _ = pto.PadValue.ZERO.value


class TileLangDSLExpandHelperTests(unittest.TestCase):
    def test_operand_specs_preserve_tile_valid_shape_and_pad_value(self) -> None:
        source = """
import tilelang_dsl as pto

@pto.vkernel(op="pto.expand_helper_tile_config_unique", dtypes=[(pto.f32, pto.f32)])
def kernel(src: pto.Tile, dst: pto.Tile):
    rows, cols = src.valid_shape
    pad = dst.pad_value
    if pto.constexpr(pad != pto.PadValue.NULL):
        scalar = pad.eval()
    return None
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = Path(tmpdir) / "expand_helper_tile_config_unique.py"
            module_path.write_text(source, encoding="utf-8")

            mod = expand_helper._import_py_file(module_path)
            self.assertIsNotNone(mod)
            descriptors = expand_helper._find_descriptors(mod)
            self.assertTrue(descriptors)

            operand_specs = expand_helper._parse_operand_specs(
                """
[
  {
    "kind": "tile",
    "dtype": "f32",
    "shape": [16, 64],
    "valid_shape": [8, 48],
    "memory_space": "ub",
    "config": {
      "b_layout": "row_major",
      "s_layout": "none_box",
      "s_fractal_size": 512,
      "pad_value": "0x0"
    }
  },
  {
    "kind": "tile",
    "dtype": "f32",
    "shape": [16, 64],
    "valid_shape": [8, 48],
    "memory_space": "ub",
    "config": {
      "b_layout": "row_major",
      "s_layout": "none_box",
      "s_fractal_size": 512,
      "pad_value": "0x1"
    }
  }
]
"""
            )
            desc = expand_helper._select_descriptor(
                descriptors,
                target="a5",
                op_name="pto.expand_helper_tile_config_unique",
                operand_specs=operand_specs,
            )
            self.assertIsNotNone(desc)

            tile_specs = {}
            for param, operand_spec in zip(desc.parameters, operand_specs):
                self.assertEqual(param.kind, "tile")
                tile_specs[param.name] = pto.TileSpecialization(
                    shape=operand_spec["shape"],
                    memory_space=operand_spec["memory_space"],
                    config=operand_spec["config"],
                    valid_shape=operand_spec["valid_shape"],
                )

            mlir_text = desc.specialize(**tile_specs).mlir_text()

        self.assertIn("valid_shape=(8, 48)", mlir_text)
        self.assertIn(
            "!pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=8, v_col=48, "
            "blayout=row_major, slayout=none_box, fractal=512, pad=0>",
            mlir_text,
        )
        self.assertIn(
            "!pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=8, v_col=48, "
            "blayout=row_major, slayout=none_box, fractal=512, pad=1>",
            mlir_text,
        )

    def test_select_descriptor_uses_positional_context_for_named_constraints(self) -> None:
        source = """
import tilelang_dsl as pto

@pto.vkernel(
    target="a5",
    op="pto.expand_helper_positional_constraints_unique",
    dtypes=[(pto.f32, pto.f32)],
    constraints=[
        lambda src: src.rank == 5,
        lambda src: src.strides[4] == 1,
        lambda dst: dst.config.b_layout == pto.BLayout.ROW_MAJOR,
    ],
)
def template_nd(src: pto.TensorView, dst: pto.Tile):
    return None

@pto.vkernel(
    target="a5",
    op="pto.expand_helper_positional_constraints_unique",
    dtypes=[(pto.f32, pto.f32)],
    constraints=[
        lambda inp: inp.rank == 5,
        lambda out: out.config.b_layout == pto.BLayout.COL_MAJOR,
    ],
    priority=9,
)
def template_dn(inp: pto.TensorView, out: pto.Tile):
    return None
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = Path(tmpdir) / "expand_helper_positional_constraints_unique.py"
            module_path.write_text(source, encoding="utf-8")

            mod = expand_helper._import_py_file(module_path)
            self.assertIsNotNone(mod)
            descriptors = expand_helper._find_descriptors(mod)
            self.assertTrue(descriptors)

            operand_specs = expand_helper._parse_operand_specs(
                """
[
  {
    "kind": "view",
    "dtype": "f32",
    "shape": [1, 1, 1, 16, 64],
    "strides": [1024, 1024, 1024, 64, 1],
    "memory_space": "gm"
  },
  {
    "kind": "tile",
    "dtype": "f32",
    "shape": [16, 64],
    "valid_shape": [16, 64],
    "memory_space": "ub",
    "config": {
      "b_layout": "row_major",
      "s_layout": "none_box",
      "s_fractal_size": 512,
      "pad_value": "0x0"
    }
  }
]
"""
            )

            selected = expand_helper._select_descriptor(
                descriptors,
                target="a5",
                op_name="pto.expand_helper_positional_constraints_unique",
                operand_specs=operand_specs,
            )

        self.assertEqual(selected.name, "template_nd")


class TileLangDSLSupportMatrixTests(unittest.TestCase):
    def test_stable_starter_surface_groups_map_to_stable_tier(self) -> None:
        self.assertEqual(get_surface_group_tier("TensorView"), BASIC_TIER)
        self.assertEqual(get_surface_group_tier("Tile"), BASIC_TIER)
        self.assertEqual(get_surface_group_tier("base_vector_ops"), BASIC_TIER)
        self.assertEqual(get_surface_group_tier("tile_indexing_sugar"), BASIC_TIER)

        self.assertIn("TensorView", AUTHORING_TIER_SURFACE_GROUPS["TensorView"])
        self.assertIn("Tile", AUTHORING_TIER_SURFACE_GROUPS["Tile"])
        self.assertNotIn("dma_load/store", AUTHORING_TIER_SURFACE_GROUPS)
        self.assertIn("pto.vlds", AUTHORING_TIER_SURFACE_GROUPS["base_vector_ops"])
        self.assertIn("pto.vsts", AUTHORING_TIER_SURFACE_GROUPS["base_vector_ops"])
        self.assertIn("pto.vadd", AUTHORING_TIER_SURFACE_GROUPS["base_vector_ops"])
        self.assertIn("pto.vmuls", AUTHORING_TIER_SURFACE_GROUPS["base_vector_ops"])
        self.assertIn("pto.vmod", AUTHORING_TIER_SURFACE_GROUPS["base_vector_ops"])
        self.assertIn("tile[start:]", BASIC_TILE_INDEXING_SURFACES)
        self.assertIn("tile[row, col:]", BASIC_TILE_INDEXING_SURFACES)

        self.assertEqual(get_feature_tier("TensorView"), BASIC_TIER)
        self.assertEqual(get_feature_tier("Tile"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.vlds"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.vsts"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.vadd"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.vmuls"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.vmod"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.get_buf"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.rls_buf"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.get_block_idx"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.get_subblock_num"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.mem_bar"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.set_cross_core"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.set_intra_block"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.set_intra_core"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.wait_flag_dev"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.wait_intra_core"), BASIC_TIER)
        self.assertEqual(get_feature_tier("BarrierType"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.vaddrelu"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.vaxpy"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.vmull"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.vands"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.vbr"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.vdup"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.vci"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.vpack"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.vsort32"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.vldsx2"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.vstsx2"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.vscatter"), ADVANCED_TIER)
        self.assertEqual(get_feature_tier("pto.vbitsort"), ADVANCED_TIER)
        self.assertEqual(get_feature_tier("pto.vmrgsort4"), ADVANCED_TIER)
        self.assertEqual(get_feature_tier("PadMode"), BASIC_TIER)
        self.assertEqual(get_feature_tier("VRegType"), BASIC_TIER)
        self.assertEqual(get_feature_tier("MaskType"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.vreg"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.mask_b8"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.mask_b16"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.mask_b32"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.bytewidth"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.get_lanes"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.elements_per_vreg"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.constexpr"), BASIC_TIER)
        self.assertEqual(get_feature_tier("constexpr"), BASIC_TIER)
        self.assertEqual(get_feature_tier("tile[start:]"), BASIC_TIER)
        self.assertEqual(get_feature_tier("tile[row, col:]"), BASIC_TIER)

    def test_non_stable_surface_groups_keep_advanced_boundaries(self) -> None:
        self.assertEqual(get_surface_group_tier("strict_vecscope"), ADVANCED_TIER)
        self.assertEqual(get_surface_group_tier("raw_pointer_family"), ADVANCED_TIER)
        self.assertEqual(get_surface_group_tier("low_level_dma_family"), ADVANCED_TIER)
        self.assertEqual(get_surface_group_tier("tile_helper_family"), ADVANCED_TIER)

        self.assertIn("pto.strict_vecscope", ADVANCED_EXPLICIT_VECSCOPE_SURFACES)
        self.assertIn("pto.ptr", ADVANCED_RAW_POINTER_SURFACES)
        self.assertIn("pto.castptr", ADVANCED_RAW_POINTER_SURFACES)
        self.assertIn("pto.set_mov_pad_val", ADVANCED_LOW_LEVEL_DMA_SURFACES)
        self.assertIn("pto.copy_ubuf_to_ubuf", ADVANCED_LOW_LEVEL_DMA_SURFACES)
        self.assertIn("pto.tile_with_strides", ADVANCED_TILE_HELPER_SURFACES)

        self.assertEqual(get_feature_tier("strict_vecscope"), ADVANCED_TIER)
        self.assertEqual(get_feature_tier("pto.strict_vecscope"), ADVANCED_TIER)
        self.assertEqual(get_feature_tier("pto.ptr"), ADVANCED_TIER)
        self.assertEqual(get_feature_tier("pto.castptr"), ADVANCED_TIER)
        self.assertEqual(get_feature_tier("pto.load_scalar"), ADVANCED_TIER)
        self.assertEqual(get_feature_tier("pto.store_scalar"), ADVANCED_TIER)
        self.assertEqual(get_feature_tier("pto.set_mov_pad_val"), ADVANCED_TIER)
        self.assertEqual(get_feature_tier("pto.copy_ubuf_to_ubuf"), ADVANCED_TIER)
        self.assertEqual(get_feature_tier("pto.tile_with_strides"), ADVANCED_TIER)

    def test_unsupported_features_do_not_report_legacy_tiers(self) -> None:
        with self.assertRaises(KeyError):
            get_surface_group_tier("dma_load/store")
        with self.assertRaises(KeyError):
            get_feature_tier("pto.dma_load")
        with self.assertRaises(KeyError):
            get_feature_tier("pto.dma_store")
        with self.assertRaises(KeyError):
            get_feature_tier("pto.dma_copy")
        with self.assertRaises(KeyError):
            get_feature_tier("pto.vreduce")

    def test_inferred_vecscope_tables_follow_supported_vecscope_surfaces(self) -> None:
        self.assertTrue(
            SUPPORTED_VECSCOPE_PTO_CALLS.issubset(
                INFERRED_VECSCOPE_ACTIVITY_PTO_CALLS
            )
        )
        self.assertTrue(
            ADVANCED_VECSCOPE_PTO_CALLS.issubset(
                INFERRED_VECSCOPE_ACTIVITY_PTO_CALLS
            )
        )
        self.assertIn("vbitcast", INFERRED_VECSCOPE_ACTIVITY_PTO_CALLS)
        self.assertIn("vselr", INFERRED_VECSCOPE_ACTIVITY_PTO_CALLS)
        self.assertIn("mem_bar", INFERRED_VECSCOPE_NEUTRAL_PTO_CALLS)


class TileLangDSLMatcherEntryTests(unittest.TestCase):
    def test_select_kernel_returns_descriptor_from_default_registry(self) -> None:
        @pto.vkernel(op="matcher_entry_default_registry_unique", dtypes=[(pto.f32, pto.i32)])
        def kernel(inp: pto.TensorView, scale: pto.i32):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_entry_default_registry_unique",
            (pto.f32, pto.i32),
        )

        self.assertIs(selected, kernel)

    def test_select_kernel_uses_explicit_registry_without_falling_back(self) -> None:
        @pto.vkernel(op="matcher_entry_registry_isolation_unique", dtypes=[(pto.f32,)])
        def default_kernel(inp: pto.TensorView):
            return None

        empty_registry = pto.KernelRegistry()
        with self.assertRaises(LookupError) as ctx:
            pto.select_kernel(
                "a5",
                "matcher_entry_registry_isolation_unique",
                (pto.f32,),
                registry=empty_registry,
            )
        self.assertIn("found no registered kernel", str(ctx.exception))

        isolated_registry = pto.KernelRegistry()
        isolated_registry.register(default_kernel)
        selected = pto.select_kernel(
            "a5",
            "matcher_entry_registry_isolation_unique",
            (pto.f32,),
            registry=isolated_registry,
        )

        self.assertIs(selected, default_kernel)
        self.assertEqual(len(isolated_registry.descriptors), 1)

    def test_select_kernel_binds_concrete_signature_from_multi_signature_descriptor(self) -> None:
        @pto.vkernel(
            op="matcher_multi_signature_unique",
            dtypes=[
                (pto.f16, pto.f16),
                (pto.f32, pto.f32),
            ],
        )
        def kernel(inp: pto.TensorView, tile: pto.Tile):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_multi_signature_unique",
            (pto.f32, pto.f32),
        )

        self.assertEqual(selected.dtype_signature, (pto.f32, pto.f32))
        self.assertEqual(
            [(param.name, param.kind, param.dtype) for param in selected.parameters],
            [("inp", "tensorview", pto.f32), ("tile", "tile", pto.f32)],
        )
        specialized = selected.specialize(
            tile=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB)
        )
        self.assertIn(
            "!pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16",
            specialized.mlir_text(),
        )

    def test_select_kernel_binds_omitted_dtypes_via_anytype_defaults(self) -> None:
        @pto.vkernel(op="matcher_default_dtypes_unique")
        def kernel(inp: pto.Tile, out: pto.Tile):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_default_dtypes_unique",
            (pto.f16, pto.f16),
        )

        self.assertEqual(selected.dtype_signature, (pto.f16, pto.f16))
        self.assertEqual(
            [(param.name, param.kind, param.dtype) for param in selected.parameters],
            [("inp", "tile", pto.f16), ("out", "tile", pto.f16)],
        )
        specialized = selected.specialize(
            inp=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            out=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )
        self.assertIn(
            "!pto.tile_buf<loc=vec, dtype=f16, rows=8, cols=16, v_row=8, v_col=16",
            specialized.mlir_text(),
        )

    def test_select_kernel_default_dtypes_preserve_scalar_annotations(self) -> None:
        @pto.vkernel(op="matcher_default_dtypes_scalar_guard_unique")
        def kernel(inp: pto.TensorView, scale: pto.i32):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_default_dtypes_scalar_guard_unique",
            (pto.f32, pto.i32),
        )
        self.assertEqual(selected.dtype_signature, (pto.f32, pto.i32))

        with self.assertRaises(LookupError) as ctx:
            pto.select_kernel(
                "a5",
                "matcher_default_dtypes_scalar_guard_unique",
                (pto.f32, pto.f16),
            )
        self.assertIn("found no registered kernel", str(ctx.exception))

    def test_select_kernel_matches_wildcards_deterministically(self) -> None:
        @pto.vkernel(
            op="matcher_wildcard_unique",
            dtypes=[
                (pto.AnyInt, pto.AnyType),
                (pto.AnyFloat, pto.AnyType),
            ],
        )
        def kernel(lhs: pto.TensorView, rhs: pto.Tile):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_wildcard_unique",
            (pto.f32, pto.i32),
        )

        self.assertEqual(selected.dtype_signature, (pto.f32, pto.i32))
        self.assertEqual(selected.parameters[0].dtype, pto.f32)
        self.assertEqual(selected.parameters[1].dtype, pto.i32)

        selected_int = pto.select_kernel(
            "a5",
            "matcher_wildcard_unique",
            (pto.ui16, pto.si16),
        )
        self.assertEqual(selected_int.dtype_signature, (pto.ui16, pto.si16))
        self.assertEqual(selected_int.parameters[0].dtype, pto.ui16)
        self.assertEqual(selected_int.parameters[1].dtype, pto.si16)

    def test_select_kernel_enforces_typevar_consistency_per_signature(self) -> None:
        @pto.vkernel(
            op="matcher_typevar_unique",
            dtypes=[(pto.TypeVar("T"), pto.TypeVar("T"))],
        )
        def kernel(lhs: pto.TensorView, rhs: pto.Tile):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_typevar_unique",
            (pto.f32, pto.f32),
        )
        self.assertEqual(selected.dtype_signature, (pto.f32, pto.f32))

        with self.assertRaises(LookupError) as ctx:
            pto.select_kernel(
                "a5",
                "matcher_typevar_unique",
                (pto.f32, pto.i32),
            )
        self.assertIn("found no registered kernel", str(ctx.exception))

    def test_scalar_typevar_annotation_tracks_selected_dtype(self) -> None:
        elem = pto.TypeVar("Elem")

        @pto.vkernel(
            op="scalar_typevar_binding_unique",
            dtypes=[(elem, elem, elem)],
        )
        def kernel(inp: pto.Tile, scale: elem, out: pto.Tile):
            return None

        selected = pto.select_kernel(
            "a5",
            "scalar_typevar_binding_unique",
            (pto.bf16, pto.bf16, pto.bf16),
        )

        self.assertEqual(selected.dtype_signature, (pto.bf16, pto.bf16, pto.bf16))
        self.assertEqual(
            [(param.name, param.kind, param.dtype) for param in selected.parameters],
            [("inp", "tile", pto.bf16), ("scale", "scalar", pto.bf16), ("out", "tile", pto.bf16)],
        )

    def test_scalar_wildcard_annotation_accepts_selected_dtype(self) -> None:
        @pto.vkernel(
            op="scalar_wildcard_binding_unique",
            dtypes=[(pto.AnyType, pto.AnyType, pto.AnyType)],
        )
        def kernel(inp: pto.Tile, scale: pto.AnyType, out: pto.Tile):
            return None

        selected = pto.select_kernel(
            "a5",
            "scalar_wildcard_binding_unique",
            (pto.i16, pto.i16, pto.i16),
        )

        self.assertEqual(selected.dtype_signature, (pto.i16, pto.i16, pto.i16))
        self.assertEqual(
            [(param.name, param.kind, param.dtype) for param in selected.parameters],
            [("inp", "tile", pto.i16), ("scale", "scalar", pto.i16), ("out", "tile", pto.i16)],
        )

    def test_polymorphic_descriptor_requires_select_kernel_before_materialization(self) -> None:
        @pto.vkernel(
            op="matcher_materialization_gate_unique",
            dtypes=[(pto.AnyFloat, pto.AnyFloat)],
        )
        def kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        with self.assertRaises(ValueError) as ctx:
            kernel.mlir_text()
        self.assertIn("requires pto.select_kernel(...)", str(ctx.exception))

    def test_select_kernel_evaluates_constraints_before_priority(self) -> None:
        def requires_large_batch(batch=0):
            return batch >= 1024

        @pto.vkernel(
            op="matcher_constraint_priority_unique",
            dtypes=[(pto.AnyFloat, pto.AnyFloat)],
            constraints=[requires_large_batch],
            priority=100,
        )
        def high_priority_kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        @pto.vkernel(
            op="matcher_constraint_priority_unique",
            dtypes=[(pto.AnyFloat, pto.AnyFloat)],
            constraints=[],
            priority=10,
        )
        def fallback_kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_constraint_priority_unique",
            (pto.f32, pto.f32),
            context_attrs={"batch": 128},
        )
        self.assertIs(selected.py_fn, fallback_kernel.py_fn)
        self.assertEqual(selected.priority, 10)

        selected = pto.select_kernel(
            "a5",
            "matcher_constraint_priority_unique",
            (pto.f32, pto.f32),
            context_attrs={"batch": 4096},
        )
        self.assertIs(selected.py_fn, high_priority_kernel.py_fn)
        self.assertEqual(selected.priority, 100)

    def test_select_kernel_raises_tie_error_for_equal_highest_priority(self) -> None:
        @pto.vkernel(
            op="matcher_priority_tie_unique",
            dtypes=[(pto.AnyFloat, pto.AnyFloat)],
            priority=50,
        )
        def lhs(inp: pto.TensorView, out: pto.TensorView):
            return None

        @pto.vkernel(
            op="matcher_priority_tie_unique",
            dtypes=[(pto.AnyFloat, pto.AnyFloat)],
            priority=50,
        )
        def rhs(inp: pto.TensorView, out: pto.TensorView):
            return None

        with self.assertRaises(LookupError) as ctx:
            pto.select_kernel(
                "a5",
                "matcher_priority_tie_unique",
                (pto.f32, pto.f32),
            )
        self.assertIn("multiple highest-priority kernels", str(ctx.exception))
        self.assertIn("lhs(priority=50", str(ctx.exception))
        self.assertIn("rhs(priority=50", str(ctx.exception))

    def test_select_kernel_reports_no_candidate_after_constraint_evaluation(self) -> None:
        @pto.vkernel(
            op="matcher_constraint_empty_unique",
            dtypes=[(pto.AnyFloat, pto.AnyFloat)],
            constraints=[lambda enabled=False: enabled],
            priority=1,
        )
        def kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        with self.assertRaises(LookupError) as ctx:
            pto.select_kernel(
                "a5",
                "matcher_constraint_empty_unique",
                (pto.f32, pto.f32),
                context_attrs={"enabled": False},
            )
        self.assertIn("after constraint evaluation", str(ctx.exception))

    def test_select_kernel_report_mode_keeps_default_descriptor_path_compatible(self) -> None:
        @pto.vkernel(op="matcher_report_default_compat_unique", dtypes=[(pto.f32, pto.f32)])
        def kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_report_default_compat_unique",
            (pto.f32, pto.f32),
            return_metadata=False,
            include_mlir=False,
        )

        self.assertIsInstance(selected, pto.VKernelDescriptor)
        self.assertIs(selected.py_fn, kernel.py_fn)
        self.assertEqual(selected.dtype_signature, (pto.f32, pto.f32))

    def test_select_kernel_report_mode_records_dtype_mismatch_candidates(self) -> None:
        @pto.vkernel(
            op="matcher_report_dtype_mismatch_unique",
            dtypes=[(pto.f32, pto.f32)],
            priority=5,
        )
        def mismatch(inp: pto.TensorView, out: pto.TensorView):
            return None

        @pto.vkernel(
            op="matcher_report_dtype_mismatch_unique",
            dtypes=[(pto.AnyFloat, pto.AnyFloat)],
            priority=10,
        )
        def fallback(inp: pto.TensorView, out: pto.TensorView):
            return None

        report = pto.select_kernel(
            "a5",
            "matcher_report_dtype_mismatch_unique",
            (pto.bf16, pto.bf16),
            return_metadata=True,
            include_mlir=False,
        )

        self.assertIsInstance(report, pto.KernelSelectionReport)
        self.assertEqual(report.final_status, "selected")
        self.assertIsNotNone(report.selected)
        assert report.selected is not None
        self.assertEqual(report.selected.py_fn, fallback.py_fn)
        self.assertEqual(
            [(candidate.name, candidate.status) for candidate in report.candidates],
            [("mismatch", "dtype_mismatch"), ("fallback", "selected")],
        )

    def test_select_kernel_report_mode_records_constraint_failure_candidates(self) -> None:
        constrained_check = lambda enabled=False: enabled

        @pto.vkernel(
            op="matcher_report_constraint_failure_unique",
            dtypes=[(pto.AnyFloat, pto.AnyFloat)],
            constraints=[constrained_check],
            priority=20,
        )
        def constrained(inp: pto.TensorView, out: pto.TensorView):
            return None

        @pto.vkernel(
            op="matcher_report_constraint_failure_unique",
            dtypes=[(pto.AnyFloat, pto.AnyFloat)],
            priority=5,
        )
        def fallback(inp: pto.TensorView, out: pto.TensorView):
            return None

        report = pto.select_kernel(
            "a5",
            "matcher_report_constraint_failure_unique",
            (pto.f32, pto.f32),
            context_attrs={"enabled": False},
            return_metadata=True,
            include_mlir=False,
        )

        self.assertEqual(report.final_status, "selected")
        self.assertIsNotNone(report.selected)
        assert report.selected is not None
        self.assertEqual(report.selected.py_fn, fallback.py_fn)
        expected_location = (
            f"{constrained_check.__code__.co_filename}:{constrained_check.__code__.co_firstlineno}"
        )
        self.assertEqual(
            [
                (
                    candidate.name,
                    candidate.status,
                    candidate.failed_constraint_index,
                    candidate.failed_constraint_location,
                )
                for candidate in report.candidates
            ],
            [
                ("constrained", "constraint_failed", 0, expected_location),
                ("fallback", "selected", None, None),
            ],
        )
        self.assertIn(expected_location, report.candidates[0].reason)

    def test_select_kernel_report_mode_records_constraint_exceptions(self) -> None:
        bad_constraint = lambda missing: missing

        @pto.vkernel(
            op="matcher_report_constraint_exception_unique",
            dtypes=[(pto.f32, pto.f32)],
            constraints=[bad_constraint],
        )
        def bad(inp: pto.TensorView, out: pto.TensorView):
            return None

        report = pto.select_kernel(
            "a5",
            "matcher_report_constraint_exception_unique",
            (pto.f32, pto.f32),
            return_metadata=True,
            include_mlir=False,
        )

        self.assertEqual(report.final_status, "no_candidate")
        self.assertIsNone(report.selected)
        self.assertIn("requires unsupported parameter", report.final_error)
        self.assertEqual(len(report.candidates), 1)
        expected_location = (
            f"{bad_constraint.__code__.co_filename}:{bad_constraint.__code__.co_firstlineno}"
        )
        candidate = report.candidates[0]
        self.assertEqual(candidate.name, "bad")
        self.assertEqual(candidate.status, "constraint_error")
        self.assertEqual(candidate.failed_constraint_index, 0)
        self.assertEqual(candidate.failed_constraint_location, expected_location)
        self.assertEqual(candidate.error_type, "TypeError")
        self.assertIn("requires unsupported parameter", candidate.error_message)
        self.assertIn(expected_location, candidate.error_message)

    def test_select_kernel_report_mode_reports_priority_ties(self) -> None:
        @pto.vkernel(
            op="matcher_report_priority_tie_unique",
            dtypes=[(pto.f32, pto.f32)],
            priority=33,
        )
        def lhs(inp: pto.TensorView, out: pto.TensorView):
            return None

        @pto.vkernel(
            op="matcher_report_priority_tie_unique",
            dtypes=[(pto.f32, pto.f32)],
            priority=33,
        )
        def rhs(inp: pto.TensorView, out: pto.TensorView):
            return None

        report = pto.select_kernel(
            "a5",
            "matcher_report_priority_tie_unique",
            (pto.f32, pto.f32),
            return_metadata=True,
            include_mlir=False,
        )

        self.assertEqual(report.final_status, "priority_tie")
        self.assertIsNone(report.selected)
        self.assertIn("multiple highest-priority kernels", report.final_error)
        self.assertEqual(
            [(candidate.name, candidate.status) for candidate in report.candidates],
            [("lhs", "priority_tie"), ("rhs", "priority_tie")],
        )

    def test_select_kernel_report_mode_reports_no_candidate_without_candidates(self) -> None:
        empty_registry = pto.KernelRegistry()

        report = pto.select_kernel(
            "a5",
            "matcher_report_empty_registry_unique",
            (pto.f32,),
            registry=empty_registry,
            return_metadata=True,
            include_mlir=False,
        )

        self.assertEqual(report.final_status, "no_candidate")
        self.assertIsNone(report.selected)
        self.assertEqual(report.candidates, ())
        self.assertIn("found no registered kernel", report.final_error)

    def test_select_kernel_report_mode_includes_mlir_text_for_materializable_candidate(self) -> None:
        @pto.vkernel(
            op="matcher_report_mlir_text_unique",
            dtypes=[(pto.f32, pto.f32)],
        )
        def kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        report = pto.select_kernel(
            "a5",
            "matcher_report_mlir_text_unique",
            (pto.f32, pto.f32),
            return_metadata=True,
            include_mlir=True,
        )

        self.assertEqual(report.final_status, "selected")
        self.assertEqual(len(report.candidates), 1)
        candidate = report.candidates[0]
        self.assertEqual(candidate.status, "selected")
        self.assertIsNotNone(candidate.mlir_text)
        self.assertIsNone(candidate.mlir_error)
        self.assertIn("module attributes", candidate.mlir_text)
        self.assertIn("@kernel", candidate.mlir_text)
        self.assertIn("!pto.tensor_view", candidate.mlir_text)

    def test_select_kernel_report_mode_includes_mlir_error_for_unspecialized_tile_candidate(self) -> None:
        @pto.vkernel(
            op="matcher_report_mlir_error_unique",
            dtypes=[(pto.f32, pto.f32)],
        )
        def kernel(inp: pto.TensorView, out: pto.Tile):
            return None

        report = pto.select_kernel(
            "a5",
            "matcher_report_mlir_error_unique",
            (pto.f32, pto.f32),
            return_metadata=True,
            include_mlir=True,
        )

        self.assertEqual(report.final_status, "selected")
        self.assertEqual(len(report.candidates), 1)
        candidate = report.candidates[0]
        self.assertEqual(candidate.status, "selected")
        self.assertIsNone(candidate.mlir_text)
        self.assertIsNotNone(candidate.mlir_error)
        self.assertIn("requires specialize() bindings for bare Tile parameters", candidate.mlir_error)

    def test_materialization_constraints_can_see_specializations_and_selected_context_attrs(self) -> None:
        @pto.vkernel(
            op="matcher_materialization_constraint_unique",
            dtypes=[(pto.f32, pto.f32)],
            constraints=[
                lambda src: src.rank == 5,
                lambda dst, expected_rows=None: dst.shape[0] == expected_rows,
                lambda src, dst: dst.valid_shape[1] <= src.shape[4],
            ],
        )
        def kernel(src: pto.TensorView, dst: pto.Tile):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_materialization_constraint_unique",
            (pto.f32, pto.f32),
            context_attrs={"expected_rows": 8, "src_shape": (2, 2, 1, 1, 16), "src_strides": (32, 16, 16, 16, 1)},
        ).specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB, valid_shape=(4, 16)),
        )
        text = selected.mlir_text()
        self.assertIn("!pto.tensor_view<?x?x?x?x?xf32>", text)
        self.assertIn("!pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16", text)

        rejected = pto.select_kernel(
            "a5",
            "matcher_materialization_constraint_unique",
            (pto.f32, pto.f32),
            context_attrs={"expected_rows": 8, "src_shape": (2, 2, 1, 1, 8), "src_strides": (16, 8, 8, 8, 1)},
        ).specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB, valid_shape=(4, 16)),
        )
        with self.assertRaises(LookupError) as ctx:
            rejected.mlir_text()
        self.assertIn("constraint evaluation rejected", str(ctx.exception))

    def test_constraints_support_parameter_style_shape_and_stride_access(self) -> None:
        @pto.vkernel(
            op="matcher_parameter_style_constraints_unique",
            dtypes=[(pto.f32, pto.f32)],
            constraints=[
                lambda src, dst: src.rank == 5,
                lambda src: src.strides[4] == 1,
                lambda src, dst: src.shape[0] <= dst.shape[0],
            ],
        )
        def kernel(src: pto.TensorView, dst: pto.Tile):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_parameter_style_constraints_unique",
            (pto.f32, pto.f32),
            context_attrs={"src_shape": (4, 1, 1, 1, 16), "src_strides": (16, 16, 16, 16, 1)},
        ).specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )
        self.assertIn("!pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16", selected.mlir_text())

        rejected = pto.select_kernel(
            "a5",
            "matcher_parameter_style_constraints_unique",
            (pto.f32, pto.f32),
            context_attrs={"src_shape": (16, 1, 1, 1, 16), "src_strides": (16, 16, 16, 16, 1)},
        ).specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )
        with self.assertRaises(LookupError):
            rejected.mlir_text()

    def test_select_kernel_supports_positional_context_attrs(self) -> None:
        @pto.vkernel(
            op="matcher_positional_context_unique",
            dtypes=[(pto.f32, pto.f32)],
            constraints=[
                lambda src: src.rank == 5,
                lambda src: src.strides[4] == 1,
                lambda dst: dst.config.b_layout == pto.BLayout.ROW_MAJOR,
            ],
        )
        def template_nd(src: pto.TensorView, dst: pto.Tile):
            return None

        @pto.vkernel(
            op="matcher_positional_context_unique",
            dtypes=[(pto.f32, pto.f32)],
            constraints=[
                lambda inp: inp.rank == 5,
                lambda out: out.config.b_layout == pto.BLayout.COL_MAJOR,
            ],
            priority=9,
        )
        def template_dn(inp: pto.TensorView, out: pto.Tile):
            return None

        operand_specs = expand_helper._parse_operand_specs(
            """
[
  {
    "kind": "view",
    "dtype": "f32",
    "shape": [1, 1, 1, 16, 64],
    "strides": [1024, 1024, 1024, 64, 1],
    "memory_space": "gm"
  },
  {
    "kind": "tile",
    "dtype": "f32",
    "shape": [16, 64],
    "valid_shape": [16, 64],
    "memory_space": "ub",
    "config": {
      "b_layout": "row_major",
      "s_layout": "none_box",
      "s_fractal_size": 512,
      "pad_value": "0x0"
    }
  }
]
"""
        )

        registry = pto.KernelRegistry((template_nd, template_dn))
        selected = pto.select_kernel(
            "a5",
            "matcher_positional_context_unique",
            (pto.f32, pto.f32),
            context_attrs=expand_helper._build_positional_context_attrs(operand_specs),
            registry=registry,
        )

        self.assertEqual(selected.name, "template_nd")

    def test_select_kernel_binds_selected_op_for_multi_op_descriptor(self) -> None:
        @pto.vkernel(
            ops=["matcher_multi_op_bind_add_unique", "matcher_multi_op_bind_sub_unique"],
            dtypes=[(pto.f32, pto.f32)],
        )
        def kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_multi_op_bind_sub_unique",
            (pto.f32, pto.f32),
        )

        self.assertIs(selected.py_fn, kernel.py_fn)
        self.assertEqual(selected.match_ops, ("matcher_multi_op_bind_add_unique", "matcher_multi_op_bind_sub_unique"))
        self.assertEqual(selected.selected_op, "matcher_multi_op_bind_sub_unique")
        self.assertEqual(selected.op, "matcher_multi_op_bind_sub_unique")
        self.assertEqual(selected.dtype_signature, (pto.f32, pto.f32))

    def test_select_kernel_hits_same_multi_op_descriptor_for_multiple_query_ops(self) -> None:
        @pto.vkernel(
            ops=[
                "matcher_multi_hit_add_unique",
                "matcher_multi_hit_mul_unique",
                "matcher_multi_hit_div_unique",
            ],
            dtypes=[(pto.f32, pto.f32)],
        )
        def kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        add_selected = pto.select_kernel(
            "a5",
            "matcher_multi_hit_add_unique",
            (pto.f32, pto.f32),
        )
        mul_selected = pto.select_kernel(
            "a5",
            "matcher_multi_hit_mul_unique",
            (pto.f32, pto.f32),
        )

        self.assertIs(add_selected.py_fn, kernel.py_fn)
        self.assertIs(mul_selected.py_fn, kernel.py_fn)
        self.assertEqual(add_selected.match_ops, kernel.match_ops)
        self.assertEqual(mul_selected.match_ops, kernel.match_ops)
        self.assertEqual(add_selected.selected_op, "matcher_multi_hit_add_unique")
        self.assertEqual(mul_selected.selected_op, "matcher_multi_hit_mul_unique")
        self.assertEqual(add_selected.op, "matcher_multi_hit_add_unique")
        self.assertEqual(mul_selected.op, "matcher_multi_hit_mul_unique")

    def test_select_kernel_prefers_higher_priority_single_op_over_multi_op(self) -> None:
        @pto.vkernel(
            op="matcher_single_beats_multi_priority_unique",
            dtypes=[(pto.f32, pto.f32)],
            priority=12,
        )
        def single(inp: pto.TensorView, out: pto.TensorView):
            return None

        @pto.vkernel(
            ops=[
                "matcher_single_beats_multi_priority_unique",
                "matcher_single_beats_multi_priority_alt_unique",
            ],
            dtypes=[(pto.f32, pto.f32)],
            priority=4,
        )
        def multi(inp: pto.TensorView, out: pto.TensorView):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_single_beats_multi_priority_unique",
            (pto.f32, pto.f32),
        )

        self.assertIs(selected.py_fn, single.py_fn)
        self.assertEqual(selected.selected_op, "matcher_single_beats_multi_priority_unique")
        self.assertEqual(selected.priority, 12)

    def test_select_kernel_prefers_priority_over_single_op_specificity(self) -> None:
        @pto.vkernel(
            op="matcher_single_vs_multi_priority_unique",
            dtypes=[(pto.f32, pto.f32)],
            priority=5,
        )
        def single(inp: pto.TensorView, out: pto.TensorView):
            return None

        @pto.vkernel(
            ops=["matcher_single_vs_multi_priority_unique", "matcher_single_vs_multi_priority_alt_unique"],
            dtypes=[(pto.f32, pto.f32)],
            priority=9,
        )
        def multi(inp: pto.TensorView, out: pto.TensorView):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_single_vs_multi_priority_unique",
            (pto.f32, pto.f32),
        )

        self.assertIs(selected.py_fn, multi.py_fn)
        self.assertEqual(selected.selected_op, "matcher_single_vs_multi_priority_unique")
        self.assertEqual(selected.priority, 9)

    def test_select_kernel_raises_tie_error_when_single_and_multi_op_candidates_tie(self) -> None:
        @pto.vkernel(
            op="matcher_single_multi_tie_unique",
            dtypes=[(pto.f32, pto.f32)],
            priority=17,
        )
        def single(inp: pto.TensorView, out: pto.TensorView):
            return None

        @pto.vkernel(
            ops=["matcher_single_multi_tie_unique", "matcher_single_multi_tie_alt_unique"],
            dtypes=[(pto.f32, pto.f32)],
            priority=17,
        )
        def multi(inp: pto.TensorView, out: pto.TensorView):
            return None

        with self.assertRaises(LookupError) as ctx:
            pto.select_kernel(
                "a5",
                "matcher_single_multi_tie_unique",
                (pto.f32, pto.f32),
            )

        self.assertIn("multiple highest-priority kernels", str(ctx.exception))
        self.assertIn("single(priority=17", str(ctx.exception))
        self.assertIn("multi(priority=17", str(ctx.exception))


class TileLangDSLDescriptorTests(unittest.TestCase):
    def test_descriptor_metadata_and_parameter_binding(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16, pto.i32)], verify=False)
        def kernel(inp: pto.TensorView, tile: pto.Tile, scale: pto.i32):
            return None

        self.assertEqual(kernel.target, "a5")
        self.assertEqual(kernel.op, "eltwise")
        self.assertEqual(kernel.name, "kernel")
        self.assertFalse(kernel.verify_enabled)
        self.assertFalse(kernel.advanced_enabled)
        self.assertEqual(kernel.metadata["verify"], False)
        self.assertEqual(kernel.metadata["advanced"], False)
        self.assertEqual(kernel.dtype_signature, (pto.f32, pto.f16, pto.i32))
        self.assertEqual(
            [(param.name, param.kind, param.dtype) for param in kernel.parameters],
            [("inp", "tensorview", pto.f32), ("tile", "tile", pto.f16), ("scale", "scalar", pto.i32)],
        )
        self.assertEqual(kernel.parameters[0].element_dtype, pto.f32)
        self.assertEqual(kernel.parameters[1].element_dtype, pto.f16)
        self.assertIsNone(kernel.parameters[2].element_dtype)

    def test_descriptor_accepts_multi_op_matcher_metadata(self) -> None:
        @pto.vkernel(ops=["tadd", "tsub"], dtypes=[(pto.f32, pto.f32)])
        def kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        self.assertEqual(kernel.match_ops, ("tadd", "tsub"))
        self.assertIsNone(kernel.selected_op)
        self.assertIsNone(kernel.metadata["op"])
        self.assertEqual(kernel.metadata["match_ops"], ("tadd", "tsub"))
        self.assertIsNone(kernel.metadata["selected_op"])
        self.assertEqual(kernel.dtype_signature, (pto.f32, pto.f32))
        self.assertEqual(
            [(param.name, param.kind, param.dtype) for param in kernel.parameters],
            [("inp", "tensorview", pto.f32), ("out", "tensorview", pto.f32)],
        )
        with self.assertRaises(ValueError) as ctx:
            _ = kernel.op
        self.assertIn("bind a concrete op", str(ctx.exception))

    def test_descriptor_defaults_dtypes_for_beginner_tile_kernels(self) -> None:
        @pto.vkernel(op="default_dtypes_unique")
        def kernel(inp: pto.Tile, out: pto.Tile):
            return None

        self.assertEqual(kernel.match_ops, ("default_dtypes_unique",))
        self.assertEqual(kernel.dtypes, ((pto.AnyType, pto.AnyType),))
        self.assertEqual(kernel.metadata["dtypes"], ((pto.AnyType, pto.AnyType),))
        with self.assertRaises(ValueError) as ctx:
            _ = kernel.dtype_signature
        self.assertIn("choose a concrete dtype signature", str(ctx.exception))

    def test_descriptor_defaults_scalar_typevar_to_anytype(self) -> None:
        elem = pto.TypeVar("Elem")

        @pto.vkernel(op="default_scalar_typevar_unique")
        def kernel(inp: pto.Tile, scale: elem, out: pto.Tile):
            return None

        self.assertEqual(kernel.match_ops, ("default_scalar_typevar_unique",))
        self.assertEqual(kernel.dtypes, ((pto.AnyType, pto.AnyType, pto.AnyType),))
        self.assertEqual(kernel.metadata["dtypes"], ((pto.AnyType, pto.AnyType, pto.AnyType),))
        with self.assertRaises(ValueError) as ctx:
            _ = kernel.dtype_signature
        self.assertIn("choose a concrete dtype signature", str(ctx.exception))

    def test_descriptor_defaults_scalar_wildcard_to_anytype(self) -> None:
        @pto.vkernel(op="default_scalar_wildcard_unique")
        def kernel(inp: pto.Tile, scale: pto.AnyType, out: pto.Tile):
            return None

        self.assertEqual(kernel.match_ops, ("default_scalar_wildcard_unique",))
        self.assertEqual(kernel.dtypes, ((pto.AnyType, pto.AnyType, pto.AnyType),))
        self.assertEqual(kernel.metadata["dtypes"], ((pto.AnyType, pto.AnyType, pto.AnyType),))
        with self.assertRaises(ValueError) as ctx:
            _ = kernel.dtype_signature
        self.assertIn("choose a concrete dtype signature", str(ctx.exception))

    def test_descriptor_accepts_templates_metadata(self) -> None:
        @pto.vkernel(
            ops=["tadd", "tsub", "tmul"],
            dtypes=[(pto.f32, pto.f32)],
            templates={
                "core": {
                    "tadd": "vadd",
                    "tsub": "vsub",
                },
                "post": {
                    "tmul": "vrelu",
                },
            },
        )
        def kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        self.assertEqual(
            kernel.templates,
            {
                "core": {
                    "tadd": "vadd",
                    "tsub": "vsub",
                },
                "post": {
                    "tmul": "vrelu",
                },
            },
        )
        self.assertEqual(kernel.metadata["templates"], kernel.templates)

    def test_descriptor_rejects_op_and_ops_together(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            @pto.vkernel(op="tadd", ops=["tsub"], dtypes=[(pto.f32,)])
            def kernel(inp: pto.TensorView):
                return None

        self.assertIn("either op= or ops=", str(ctx.exception))

    def test_descriptor_requires_one_of_op_or_ops(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            @pto.vkernel(dtypes=[(pto.f32,)])
            def kernel(inp: pto.TensorView):
                return None

        self.assertIn("exactly one of op= or ops=", str(ctx.exception))

    def test_descriptor_rejects_template_slot_with_non_string_name(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            @pto.vkernel(
                ops=["tadd"],
                dtypes=[(pto.f32,)],
                templates={1: {"tadd": "vadd"}},
            )
            def kernel(inp: pto.TensorView):
                return None

        self.assertIn("template slot names must be non-empty strings", str(ctx.exception))

    def test_descriptor_rejects_template_op_outside_matcher_set(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            @pto.vkernel(
                ops=["tadd", "tsub"],
                dtypes=[(pto.f32, pto.f32)],
                templates={"core": {"tmul": "vmul"}},
            )
            def kernel(inp: pto.TensorView, out: pto.TensorView):
                return None

        self.assertIn("outside descriptor matcher set", str(ctx.exception))

    def test_descriptor_rejects_template_mapping_to_unknown_pto_op(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            @pto.vkernel(
                ops=["tadd"],
                dtypes=[(pto.f32,)],
                templates={"core": {"tadd": "vunknown"}},
            )
            def kernel(inp: pto.TensorView):
                return None

        self.assertIn("maps to unsupported pto op", str(ctx.exception))

    def test_pointer_parameter_annotation_binds_as_ptr_kind(self) -> None:
        @pto.vkernel(op="ptr_surface", dtypes=[(pto.f32, pto.i64)], advanced=True)
        def kernel(src: pto.ptr(pto.f32, pto.MemorySpace.UB), addr: pto.i64):
            return None

        self.assertEqual(kernel.parameters[0].kind, "ptr")
        self.assertEqual(kernel.parameters[0].dtype, pto.f32)
        self.assertEqual(kernel.parameters[0].annotation, pto.ptr(pto.f32, pto.MemorySpace.UB))
        self.assertEqual(kernel.parameters[0].element_dtype, pto.f32)

    def test_vreg_type_constructor_exposes_inferred_lane_count(self) -> None:
        vec_type = pto.vreg(pto.f32)
        self.assertIsInstance(vec_type, pto.VRegType)
        self.assertEqual(vec_type.element_dtype, pto.f32)
        self.assertEqual(vec_type.lanes, 64)
        self.assertEqual(repr(vec_type), "vreg(f32)")

    def test_mask_type_constants_expose_granularity(self) -> None:
        self.assertIsInstance(pto.mask_b8, pto.MaskType)
        self.assertIsInstance(pto.mask_b16, pto.MaskType)
        self.assertIsInstance(pto.mask_b32, pto.MaskType)
        self.assertEqual(pto.mask_b8.granularity, "b8")
        self.assertEqual(pto.mask_b16.granularity, "b16")
        self.assertEqual(pto.mask_b32.granularity, "b32")
        self.assertEqual(repr(pto.mask_b32), "mask_b32")

    def test_mask_parameter_annotation_binds_as_mask_kind(self) -> None:
        @pto.vkernel(op="mask_surface", dtypes=[(pto.mask_b32, pto.f32)], advanced=True)
        def kernel(mask: pto.mask_b32, dst: pto.Tile):
            return None

        self.assertEqual(kernel.parameters[0].kind, "mask")
        self.assertEqual(kernel.parameters[0].dtype, pto.mask_b32)
        self.assertEqual(kernel.parameters[0].annotation, pto.mask_b32)
        self.assertIsNone(kernel.parameters[0].element_dtype)

    def test_specialization_enables_materialization_apis(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16)])
        def kernel(inp: pto.TensorView, tile: pto.Tile):
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 32),
                memory_space=pto.MemorySpace.UB,
                config=pto.TileConfig.from_mapping(
                    {
                        "layout": "row_major",
                        "pad_value": pto.PadValue.ZERO,
                    }
                ),
            )
        )

        self.assertIn("tile", specialized.specializations_by_name)
        text = specialized.mlir_text()
        self.assertIn("// tilelang.target = a5", text)
        self.assertIn("// tilelang.specialize tile shape=(16, 32) memory_space=ub", text)
        self.assertIn('module attributes {pto.target_arch = "a5"} {', text)
        self.assertIn(
            "func.func @kernel(%arg0: !pto.tensor_view<?x?x?x?x?xf32>, %arg1: !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32, v_row=16, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=1>) attributes { pto.tilelang.instance } {",
            text,
        )
        module = specialized.mlir_module()
        self.assertEqual(type(module).__name__, "MaterializedMLIRModule")
        mocked_result = kernel_impl.VerificationResult(
            status="passed",
            available=True,
            passed=True,
            message="ok",
            command=("ptoas",),
            returncode=0,
        )
        with mock.patch("tilelang_dsl.kernel._run_ptoas_verifier", return_value=mocked_result):
            self.assertTrue(module.verify())
            self.assertTrue(specialized.verify())
            self.assertEqual(module.verify().status, "passed")
            self.assertEqual(specialized.verify().status, "passed")

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "kernel.mlir"
            specialized.emit(out)
            self.assertEqual(out.read_text(encoding="utf-8"), text)

    def test_multi_op_descriptor_requires_select_kernel_before_materialization_apis(self) -> None:
        @pto.vkernel(
            ops=["multi_op_gate_add_unique", "multi_op_gate_sub_unique"],
            dtypes=[(pto.f32, pto.f32)],
        )
        def kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        with self.assertRaises(ValueError) as text_ctx:
            kernel.mlir_text()
        self.assertIn("mlir_text() requires pto.select_kernel(...) to bind a concrete op", str(text_ctx.exception))

        with self.assertRaises(ValueError) as module_ctx:
            kernel.mlir_module()
        self.assertIn(
            "mlir_module() requires pto.select_kernel(...) to bind a concrete op",
            str(module_ctx.exception),
        )

        with self.assertRaises(ValueError) as verify_ctx:
            kernel.verify()
        self.assertIn("verify() requires pto.select_kernel(...) to bind a concrete op", str(verify_ctx.exception))

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "kernel.mlir"
            with self.assertRaises(ValueError) as emit_ctx:
                kernel.emit(out)
        self.assertIn("emit() requires pto.select_kernel(...) to bind a concrete op", str(emit_ctx.exception))

    def test_selected_multi_op_descriptor_can_materialize_normally(self) -> None:
        @pto.vkernel(
            ops=["multi_op_materialize_add_unique", "multi_op_materialize_sub_unique"],
            dtypes=[(pto.f32, pto.f32)],
        )
        def kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        selected = pto.select_kernel(
            "a5",
            "multi_op_materialize_sub_unique",
            (pto.f32, pto.f32),
        )

        text = selected.mlir_text()
        self.assertIn("// tilelang.target = a5", text)
        self.assertIn("// tilelang.op = multi_op_materialize_sub_unique", text)
        self.assertIn(
            'func.func @kernel(%arg0: !pto.tensor_view<?x?x?x?x?xf32>, %arg1: !pto.tensor_view<?x?x?x?x?xf32>) attributes { pto.tilelang.instance } {',
            text,
        )

    def test_verify_reports_structured_unavailable_when_ptoas_is_missing(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16)])
        def kernel(inp: pto.TensorView, tile: pto.Tile):
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 32),
                memory_space=pto.MemorySpace.UB,
            )
        )

        result = specialized.verify(ptoas_bin="/definitely-missing/ptoas")
        self.assertFalse(result)
        self.assertEqual(result.status, "unavailable")
        self.assertFalse(result.available)
        self.assertFalse(result.passed)
        self.assertIn("verifier unavailable", result.message)

    def test_descriptor_materialization_flows_through_pipeline(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16, pto.i32)])
        def kernel(inp: pto.TensorView, tile: pto.Tile, scale: pto.i32):
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        frontend_kernel = build_frontend_kernel_node(specialized)
        self.assertEqual(frontend_kernel.name, "kernel")
        self.assertEqual(
            [(param.name, param.kind) for param in frontend_kernel.parameters],
            [("inp", "tensorview"), ("tile", "tile"), ("scale", "scalar")],
        )
        self.assertEqual(frontend_kernel.tile_specializations[0].shape, (8, 16))

        semantic_kernel = analyze_frontend_kernel(frontend_kernel)
        self.assertEqual(semantic_kernel.symbol_name, "kernel")
        self.assertEqual(semantic_kernel.tile_bindings[0].memory_space, "ub")

        authoring_module = lower_semantic_kernel(semantic_kernel)
        self.assertIsInstance(authoring_module, AuthoringModule)
        self.assertEqual(authoring_module.render(), specialized.mlir_text())
        self.assertIn("return", authoring_module.render())

    def test_descriptor_pipeline_ignores_kernel_docstring_expression(self) -> None:
        @pto.vkernel(op="docstring_passthrough_unique", dtypes=[(pto.f32,)])
        def kernel(inp: pto.TensorView):
            """This docstring should be ignored as a no-op expression statement."""
            return None

        frontend_kernel = build_frontend_kernel_node(kernel)
        self.assertEqual(len(frontend_kernel.body), 2)
        self.assertIsInstance(frontend_kernel.body[0], FrontendExprStmt)

        semantic_kernel = analyze_frontend_kernel(frontend_kernel)
        self.assertEqual(len(semantic_kernel.body), 1)

        text = lower_semantic_kernel(semantic_kernel).render()
        self.assertIn("// tilelang.op = docstring_passthrough_unique", text)
        self.assertIn("func.func @kernel", text)
        self.assertIn("return", text)

    def test_frontend_rejects_hidden_dma_load_surface(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="dma_load_hidden", dtypes=[(pto.f32, pto.f32)])
            def kernel(inp: pto.TensorView, tile: pto.Tile):
                pto.dma_load(inp[0:16, 0:16], tile)
                return None

        self.assertIn("unsupported op surface `pto.dma_load`", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_frontend_rejects_hidden_dma_store_surface(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="dma_store_hidden", dtypes=[(pto.f32, pto.f32)])
            def kernel(out: pto.TensorView, tile: pto.Tile):
                pto.dma_store(tile, out[0:16, 0:16])
                return None

        self.assertIn("unsupported op surface `pto.dma_store`", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_frontend_rejects_hidden_dma_copy_surface(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="dma_copy_hidden", dtypes=[(pto.f32, pto.f32)])
            def kernel(src: pto.Tile, dst: pto.Tile):
                pto.dma_copy(src, dst)
                return None

        self.assertIn("unsupported op surface `pto.dma_copy`", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_frontend_rejects_keyword_arguments_on_public_surfaces(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="dma_kw_wrong_surface", dtypes=[(pto.f32, pto.f32)])
            def kernel(inp: pto.TensorView, tile: pto.Tile):
                pto.vlds(tile, offset=0)
                return None

        self.assertIn(
            "unsupported keyword `offset` for `pto.vlds` in TileLang DSL v1",
            str(ctx.exception),
        )
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_frontend_rewrites_template_slot_to_selected_real_op(self) -> None:
        @pto.vkernel(
            ops=["template_slot_add_unique", "template_slot_sub_unique"],
            dtypes=[(pto.f32, pto.f32, pto.f32)],
            advanced=True,
            templates={
                "core": {
                    "template_slot_add_unique": "vadd",
                    "template_slot_sub_unique": "vsub",
                }
            },
        )
        def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
            with pto.strict_vecscope(dst, src0, src1, 0, 64, 64) as (
                out_tile,
                lhs_tile,
                rhs_tile,
                lb,
                ub,
                step,
            ):
                for lane in range(lb, ub, step):
                    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                    lhs = pto.vlds(lhs_tile, lane)
                    rhs = pto.vlds(rhs_tile, lane)
                    out = pto.tpl("core", lhs, rhs, mask)
                    pto.vsts(out, out_tile, lane, mask)
            return None

        add_selected = pto.select_kernel(
            "a5",
            "template_slot_add_unique",
            (pto.f32, pto.f32, pto.f32),
        ).specialize(
            dst=pto.TileSpecialization(shape=(16, 16), memory_space=pto.MemorySpace.UB),
            src0=pto.TileSpecialization(shape=(16, 16), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(16, 16), memory_space=pto.MemorySpace.UB),
        )
        sub_selected = pto.select_kernel(
            "a5",
            "template_slot_sub_unique",
            (pto.f32, pto.f32, pto.f32),
        ).specialize(
            dst=pto.TileSpecialization(shape=(16, 16), memory_space=pto.MemorySpace.UB),
            src0=pto.TileSpecialization(shape=(16, 16), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(16, 16), memory_space=pto.MemorySpace.UB),
        )

        add_frontend = build_frontend_kernel_node(add_selected)
        sub_frontend = build_frontend_kernel_node(sub_selected)

        add_vecscope = add_frontend.body[0]
        sub_vecscope = sub_frontend.body[0]
        self.assertIsInstance(add_vecscope, FrontendStrictVecscopeStmt)
        self.assertIsInstance(sub_vecscope, FrontendStrictVecscopeStmt)

        add_loop = add_vecscope.body[0]
        sub_loop = sub_vecscope.body[0]
        self.assertIsInstance(add_loop, FrontendForStmt)
        self.assertIsInstance(sub_loop, FrontendForStmt)

        add_out_assign = add_loop.body[3]
        sub_out_assign = sub_loop.body[3]
        self.assertIsInstance(add_out_assign, FrontendAssignStmt)
        self.assertIsInstance(sub_out_assign, FrontendAssignStmt)
        self.assertIsInstance(add_out_assign.value, FrontendCallExpr)
        self.assertIsInstance(sub_out_assign.value, FrontendCallExpr)
        self.assertEqual(add_out_assign.value.namespace, "pto")
        self.assertEqual(sub_out_assign.value.namespace, "pto")
        self.assertEqual(add_out_assign.value.name, "vadd")
        self.assertEqual(sub_out_assign.value.name, "vsub")

        add_text = add_selected.mlir_text()
        sub_text = sub_selected.mlir_text()
        self.assertIn("pto.vadd", add_text)
        self.assertNotIn("pto.vsub", add_text)
        self.assertIn("pto.vsub", sub_text)
        self.assertNotIn("pto.vadd", sub_text)

    def test_template_slot_shared_kernel_body_expands_for_four_ops(self) -> None:
        @pto.vkernel(
            ops=[
                "template_slot_tadd_unique",
                "template_slot_tsub_unique",
                "template_slot_tmul_unique",
                "template_slot_tdiv_unique",
            ],
            dtypes=[(pto.f32, pto.f32, pto.f32)],
            advanced=True,
            templates={
                "core": {
                    "template_slot_tadd_unique": "vadd",
                    "template_slot_tsub_unique": "vsub",
                    "template_slot_tmul_unique": "vmul",
                    "template_slot_tdiv_unique": "vdiv",
                }
            },
        )
        def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
            with pto.strict_vecscope(dst, src0, src1, 0, 64, 64) as (
                out_tile,
                lhs_tile,
                rhs_tile,
                lb,
                ub,
                step,
            ):
                for lane in range(lb, ub, step):
                    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                    lhs = pto.vlds(lhs_tile, lane)
                    rhs = pto.vlds(rhs_tile, lane)
                    out = pto.tpl("core", lhs, rhs, mask)
                    pto.vsts(out, out_tile, lane, mask)
            return None

        isolated_registry = pto.KernelRegistry((kernel,))
        expected_ops = {
            "template_slot_tadd_unique": "vadd",
            "template_slot_tsub_unique": "vsub",
            "template_slot_tmul_unique": "vmul",
            "template_slot_tdiv_unique": "vdiv",
        }

        for query_op, real_op in expected_ops.items():
            selected = pto.select_kernel(
                "a5",
                query_op,
                (pto.f32, pto.f32, pto.f32),
                registry=isolated_registry,
            ).specialize(
                dst=pto.TileSpecialization(shape=(16, 16), memory_space=pto.MemorySpace.UB),
                src0=pto.TileSpecialization(shape=(16, 16), memory_space=pto.MemorySpace.UB),
                src1=pto.TileSpecialization(shape=(16, 16), memory_space=pto.MemorySpace.UB),
            )

            frontend_kernel = build_frontend_kernel_node(selected)
            vecscope = frontend_kernel.body[0]
            self.assertIsInstance(vecscope, FrontendStrictVecscopeStmt)
            loop_stmt = vecscope.body[0]
            self.assertIsInstance(loop_stmt, FrontendForStmt)
            out_assign = loop_stmt.body[3]
            self.assertIsInstance(out_assign, FrontendAssignStmt)
            self.assertIsInstance(out_assign.value, FrontendCallExpr)
            self.assertEqual(out_assign.value.name, real_op)

            text = selected.mlir_text()
            self.assertIn(f"pto.{real_op}", text)
            self.assertNotIn("pto.tpl(", text)

    def test_template_slot_rejects_non_literal_slot_name(self) -> None:
        slot_name = "core"

        @pto.vkernel(
            op="template_slot_non_literal_unique",
            dtypes=[(pto.f32, pto.f32, pto.f32)],
            advanced=True,
            templates={"core": {"template_slot_non_literal_unique": "vadd"}},
        )
        def kernel(dst: pto.TensorView, src0: pto.TensorView, src1: pto.TensorView):
            with pto.strict_vecscope(dst, src0, src1, 0, 64, 64) as (out_tile, lhs_tile, rhs_tile, lb, ub, step):
                for lane in range(lb, ub, step):
                    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                    out = pto.tpl(slot_name, lhs_tile, rhs_tile, mask)
            return None

        with self.assertRaises(pto.TileLangFrontendError) as ctx:
            build_frontend_kernel_node(kernel)

        self.assertIn("pto.tpl() requires a non-empty string literal slot name", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_template_slot_rejects_unknown_slot_before_ir_generation(self) -> None:
        @pto.vkernel(
            op="template_slot_unknown_slot_unique",
            dtypes=[(pto.f32, pto.f32, pto.f32)],
            advanced=True,
            templates={"core": {"template_slot_unknown_slot_unique": "vadd"}},
        )
        def kernel(dst: pto.TensorView, src0: pto.TensorView, src1: pto.TensorView):
            with pto.strict_vecscope(dst, src0, src1, 0, 64, 64) as (out_tile, lhs_tile, rhs_tile, lb, ub, step):
                for lane in range(lb, ub, step):
                    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                    out = pto.tpl("missing", lhs_tile, rhs_tile, mask)
            return None

        with self.assertRaises(pto.TileLangFrontendError) as ctx:
            build_frontend_kernel_node(kernel)

        self.assertIn("unknown template slot 'missing'", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_template_slot_rejects_missing_selected_op_mapping(self) -> None:
        @pto.vkernel(
            ops=["template_slot_missing_map_add_unique", "template_slot_missing_map_sub_unique"],
            dtypes=[(pto.f32, pto.f32, pto.f32)],
            advanced=True,
            templates={"core": {"template_slot_missing_map_add_unique": "vadd"}},
        )
        def kernel(dst: pto.TensorView, src0: pto.TensorView, src1: pto.TensorView):
            with pto.strict_vecscope(dst, src0, src1, 0, 64, 64) as (out_tile, lhs_tile, rhs_tile, lb, ub, step):
                for lane in range(lb, ub, step):
                    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                    out = pto.tpl("core", lhs_tile, rhs_tile, mask)
            return None

        selected = pto.select_kernel(
            "a5",
            "template_slot_missing_map_sub_unique",
            (pto.f32, pto.f32, pto.f32),
        )

        with self.assertRaises(pto.TileLangFrontendError) as ctx:
            build_frontend_kernel_node(selected)

        self.assertIn("template slot 'core' does not define an implementation for selected op", str(ctx.exception))
        self.assertIn("template_slot_missing_map_sub_unique", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_template_slot_requires_selected_op_before_expansion(self) -> None:
        @pto.vkernel(
            ops=["template_slot_unbound_add_unique", "template_slot_unbound_sub_unique"],
            dtypes=[(pto.f32, pto.f32, pto.f32)],
            advanced=True,
            templates={
                "core": {
                    "template_slot_unbound_add_unique": "vadd",
                    "template_slot_unbound_sub_unique": "vsub",
                }
            },
        )
        def kernel(dst: pto.TensorView, src0: pto.TensorView, src1: pto.TensorView):
            with pto.strict_vecscope(dst, src0, src1, 0, 64, 64) as (out_tile, lhs_tile, rhs_tile, lb, ub, step):
                for lane in range(lb, ub, step):
                    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                    out = pto.tpl("core", lhs_tile, rhs_tile, mask)
            return None

        with self.assertRaises(pto.TileLangFrontendError) as ctx:
            build_frontend_kernel_node(kernel)

        self.assertIn("pto.tpl() requires pto.select_kernel(...) to bind a concrete op before expansion", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_template_slot_respects_resolved_op_surface_rules(self) -> None:
        @pto.vkernel(
            op="template_slot_advanced_surface_unique",
            dtypes=[(pto.i32, pto.i32, pto.i32)],
            templates={"cmp": {"template_slot_advanced_surface_unique": "vcmp"}},
        )
        def kernel(dst: pto.TensorView, src0: pto.TensorView, src1: pto.TensorView):
            mask = pto.make_mask(pto.i32, pto.PAT.ALL)
            out = pto.tpl("cmp", dst, src0, mask, "lt")
            return None

        with self.assertRaises(pto.TileLangFrontendError) as ctx:
            build_frontend_kernel_node(kernel)

        self.assertIn("surface `pto.vcmp` requires advanced=True", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_callable_based_runtime_template_dispatch_remains_rejected(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(
                op="template_slot_callable_dispatch_unique",
                dtypes=[(pto.f32, pto.f32, pto.f32)],
                advanced=True,
            )
            def kernel(dst: pto.TensorView, src0: pto.TensorView, src1: pto.TensorView):
                table = {"core": pto.vadd}
                with pto.strict_vecscope(dst, src0, src1, 0, 64, 64) as (
                    out_tile,
                    lhs_tile,
                    rhs_tile,
                    lb,
                    ub,
                    step,
                ):
                    for lane in range(lb, ub, step):
                        mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                        out = table["core"](lhs_tile, rhs_tile, mask)
                return None

        self.assertIn("unsupported call surface in TileLang DSL v1", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_semantic_pipeline_binds_parameter_loop_and_strict_vecscope_types(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16, pto.i32)], advanced=True)
        def kernel(inp: pto.TensorView, tile: pto.Tile, scale: pto.i32):
            rows = tile.shape[0]
            step = rows
            with pto.strict_vecscope(inp, tile, scale, 0, rows, step) as (
                vin,
                vtmp,
                factor,
                lb,
                ub,
                vec_step,
            ):
                for lane in range(lb, ub, vec_step):
                    current = factor
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        frontend_kernel = build_frontend_kernel_node(specialized)
        self.assertEqual(len(frontend_kernel.body), 4)

        semantic_kernel = analyze_frontend_kernel(frontend_kernel)
        self.assertIsInstance(semantic_kernel.parameters[0].type, SemanticTensorViewType)
        self.assertEqual(semantic_kernel.parameters[0].type.rank, 5)
        self.assertIsInstance(semantic_kernel.parameters[1].type, SemanticTileType)
        self.assertEqual(semantic_kernel.parameters[1].type.shape, (8, 16))
        self.assertIsInstance(semantic_kernel.parameters[2].type, SemanticScalarType)

        rows_assign = semantic_kernel.body[0]
        self.assertIsInstance(rows_assign, SemanticAssignStmt)
        self.assertIsInstance(rows_assign.targets[0].type, SemanticIndexType)
        self.assertTrue(rows_assign.targets[0].ssa_name.startswith("%rows_"))

        vecscope_stmt = semantic_kernel.body[2]
        self.assertIsInstance(vecscope_stmt, SemanticStrictVecscopeStmt)
        self.assertEqual(
            [binding.name for binding in vecscope_stmt.block_arguments],
            ["vin", "vtmp", "factor", "lb", "ub", "vec_step"],
        )
        self.assertIsInstance(vecscope_stmt.block_arguments[0].type, SemanticTensorViewType)
        self.assertIsInstance(vecscope_stmt.block_arguments[1].type, SemanticTileType)
        self.assertIsInstance(vecscope_stmt.block_arguments[2].type, SemanticScalarType)
        self.assertIsInstance(vecscope_stmt.block_arguments[3].type, SemanticIndexType)
        self.assertIsInstance(vecscope_stmt.block_arguments[4].type, SemanticIndexType)
        self.assertIsInstance(vecscope_stmt.block_arguments[5].type, SemanticIndexType)
        self.assertTrue(vecscope_stmt.block_arguments[0].ssa_name.startswith("%vin_"))

        loop_stmt = vecscope_stmt.body[0]
        self.assertIsInstance(loop_stmt, SemanticForStmt)
        self.assertEqual(loop_stmt.induction_variable.name, "lane")
        self.assertIsInstance(loop_stmt.induction_variable.type, SemanticIndexType)
        self.assertTrue(loop_stmt.induction_variable.ssa_name.startswith("%lane_"))
        self.assertEqual(loop_stmt.loop_carried, ())

        text = specialized.mlir_text()
        self.assertIn("%rows_", text)
        self.assertIn("= arith.constant 8 : index", text)
        self.assertRegex(
            text,
            r"pto\.strict_vecscope\(%tmp_\d+, %tmp_\d+, %arg2, %c0, %rows_\d+, %rows_\d+\)",
        )
        self.assertIn("^bb0(", text)
        self.assertIn("scf.for %lane_", text)
        self.assertIn("to %ub_6 step %vec_step_7 {", text)

    def test_tensorview_defaults_to_5d_shape_profile(self) -> None:
        @pto.vkernel(op="tensorview_5d_shape_profile_unique", dtypes=[(pto.f32,)])
        def kernel(inp: pto.TensorView):
            d0, d1, d2, d3, d4 = inp.valid_shape
            return None

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(kernel))
        self.assertIsInstance(semantic_kernel.parameters[0].type, SemanticTensorViewType)
        self.assertEqual(semantic_kernel.parameters[0].type.rank, 5)
        self.assertEqual(
            [(param.name, param.kind) for param in semantic_kernel.parameters],
            [("inp", "tensorview")],
        )

        text = kernel.mlir_text()
        self.assertIn(
            "func.func @kernel(%arg0: !pto.tensor_view<?x?x?x?x?xf32>) "
            "attributes { pto.tilelang.instance } {",
            text,
        )
        self.assertEqual(text.count("pto.get_tensor_view_dim"), 5)

    def test_tensorview_strides_profile_lowers_through_explicit_stride_queries(self) -> None:
        @pto.vkernel(op="tensorview_5d_stride_profile_unique", dtypes=[(pto.f32,)])
        def kernel(inp: pto.TensorView):
            s0, s1, s2, s3, s4 = inp.strides
            for lane in range(0, s4, 1):
                current = lane
            return None

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(kernel))
        self.assertEqual(
            [(param.name, param.kind) for param in semantic_kernel.parameters],
            [("inp", "tensorview")],
        )

        text = kernel.mlir_text()
        self.assertIn(
            "func.func @kernel(%arg0: !pto.tensor_view<?x?x?x?x?xf32>) "
            "attributes { pto.tilelang.instance } {",
            text,
        )
        self.assertEqual(text.count("pto.get_tensor_view_stride"), 5)
        self.assertRegex(text, r"scf\.for %lane_\d+ = %c0 to %s4_\d+ step %c1 \{")

    def test_tensorview_accepts_full_5d_slice_profile(self) -> None:
        @pto.vkernel(op="tensorview_5d_slice_profile_unique", dtypes=[(pto.f32,)])
        def kernel(inp: pto.TensorView):
            view = inp[0:1, 0:2, 0:3, 0:4, 0:5]
            return None

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(kernel))
        slice_assign = semantic_kernel.body[0]
        self.assertIsInstance(slice_assign, SemanticAssignStmt)
        self.assertEqual(slice_assign.value.type.rank, 5)
        self.assertEqual(slice_assign.value.type.extents, (1, 2, 3, 4, 5))
        self.assertEqual(slice_assign.value.type.physical_axes, (0, 1, 2, 3, 4))

    def test_tensorview_3d_slice_profile_right_aligns_into_5d_descriptor(self) -> None:
        @pto.vkernel(op="tensorview_3d_slice_profile_unique", dtypes=[(pto.f32,)])
        def kernel(inp: pto.TensorView):
            view = inp[0:8, 0:16, 0:32]
            return None

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(kernel))
        slice_assign = semantic_kernel.body[0]
        self.assertIsInstance(slice_assign, SemanticAssignStmt)
        self.assertEqual(slice_assign.value.type.rank, 3)
        self.assertEqual(slice_assign.value.type.extents, (8, 16, 32))
        self.assertEqual(slice_assign.value.type.physical_axes, (2, 3, 4))

    def test_tensorview_2d_slice_profile_right_aligns_into_5d_descriptor(self) -> None:
        @pto.vkernel(op="tensorview_2d_slice_profile_unique", dtypes=[(pto.f32,)])
        def kernel(inp: pto.TensorView):
            view = inp[0:16, 0:32]
            return None

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(kernel))
        slice_assign = semantic_kernel.body[0]
        self.assertIsInstance(slice_assign, SemanticAssignStmt)
        self.assertEqual(slice_assign.value.type.rank, 2)
        self.assertEqual(slice_assign.value.type.extents, (16, 32))
        self.assertEqual(slice_assign.value.type.physical_axes, (3, 4))

    def test_tensorview_slice_binding_lowers_to_partition_tensor_view_descriptor(self) -> None:
        @pto.vkernel(op="tensorview_slice_partition_binding_unique", dtypes=[(pto.f32,)])
        def kernel(inp: pto.TensorView):
            part = inp[0:16, 0:32]
            rows, cols = part.shape
            s0, s1 = part.strides
            if rows != 0 and cols != 0:
                rows = s0 + s1
            return None

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(kernel))
        slice_assign = semantic_kernel.body[0]
        self.assertIsInstance(slice_assign, SemanticAssignStmt)
        self.assertIsInstance(slice_assign.targets[0].type, SemanticPartitionTensorViewType)
        self.assertEqual(slice_assign.targets[0].type.rank, 2)

        text = kernel.mlir_text()
        self.assertIn(" = pto.partition_view %arg0, offsets = [%c0, %c0], sizes = [%c16, %c32] : ", text)
        self.assertIn("-> !pto.partition_tensor_view<16x32xf32>", text)
        self.assertEqual(text.count("pto.get_tensor_view_dim"), 2)
        self.assertEqual(text.count("pto.get_tensor_view_stride"), 2)

    def test_partition_tensor_view_annotation_accepts_tensorview_slice_binding(self) -> None:
        @pto.vkernel(op="partition_tensor_view_annotation_unique", dtypes=[(pto.f32,)])
        def kernel(inp: pto.TensorView):
            part: pto.PartitionTensorView = inp[0:8, 0:8]
            r0, r1 = part.shape
            return None

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(kernel))
        slice_assign = semantic_kernel.body[0]
        self.assertIsInstance(slice_assign, SemanticAssignStmt)
        self.assertIsInstance(slice_assign.targets[0].type, SemanticPartitionTensorViewType)
        self.assertEqual(slice_assign.targets[0].type.rank, 2)

        text = kernel.mlir_text()
        self.assertIn(" = pto.partition_view %arg0, offsets = [%c0, %c0], sizes = [%c8, %c8] : ", text)
        self.assertIn("-> !pto.partition_tensor_view<8x8xf32>", text)
        self.assertEqual(text.count("pto.get_tensor_view_dim"), 2)

    def test_dynamic_tensorview_shape_profile_supports_runtime_bound_without_high_level_dma(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32)])
        def kernel(inp: pto.TensorView, tile: pto.Tile):
            rows = inp.shape[0]
            for lane in range(0, rows, 1):
                current = lane
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertEqual(
            [(param.name, param.kind) for param in semantic_kernel.parameters],
            [("inp", "tensorview"), ("tile", "tile")],
        )

        rows_assign = semantic_kernel.body[0]
        self.assertIsInstance(rows_assign, SemanticAssignStmt)
        self.assertIsInstance(rows_assign.targets[0].type, SemanticIndexType)

        loop_stmt = semantic_kernel.body[1]
        self.assertIsInstance(loop_stmt, SemanticForStmt)

        text = specialized.mlir_text()
        self.assertIn(
            "func.func @kernel(%arg0: !pto.tensor_view<?x?x?x?x?xf32>, %arg1: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>) attributes { pto.tilelang.instance } {",
            text,
        )
        self.assertIn("scf.for %lane_", text)
        self.assertIn("pto.get_tensor_view_dim", text)

    def test_semantic_recognizes_padmode_symbol(self) -> None:
        @pto.vkernel(op="pad_mode_symbol", dtypes=[(pto.f32, pto.f32)])
        def kernel(inp: pto.TensorView, tile: pto.Tile):
            mode = pto.PadMode.PadFirstElem
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        assign_stmt = semantic_kernel.body[0]
        self.assertIsInstance(assign_stmt, SemanticAssignStmt)
        self.assertIsInstance(assign_stmt.value, SemanticSymbolExpr)
        self.assertEqual(assign_stmt.value.value, pto.PadMode.PadFirstElem)
        self.assertEqual(assign_stmt.value.type.kind, "pad_mode")

    def test_tile_config_attributes_bind_as_static_metadata(self) -> None:
        @pto.vkernel(op="tile_config_attrs_unique", dtypes=[(pto.f16,)])
        def kernel(tile: pto.Tile):
            config = tile.config
            layout = config.b_layout
            secondary = config.s_layout
            fractal = config.s_fractal_size
            pad = config.pad_value
            pad_direct = tile.pad_value
            pad_scalar = pad.eval()
            pad_direct_scalar = pad_direct.eval()
            rank = tile.rank
            space = tile.memory_space
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 16),
                memory_space=pto.MemorySpace.UB,
                config=pto.TileConfig.from_mapping(
                    {
                        "b_layout": pto.BLayout.COL_MAJOR,
                        "s_layout": pto.SLayout.ROW_MAJOR,
                        "s_fractal_size": 16,
                        "pad_value": pto.PadValue.ZERO,
                    }
                ),
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        (
            config_assign,
            layout_assign,
            secondary_assign,
            fractal_assign,
            pad_assign,
            pad_direct_assign,
            pad_scalar_assign,
            pad_direct_scalar_assign,
            rank_assign,
            space_assign,
        ) = (
            semantic_kernel.body[:10]
        )

        self.assertIsInstance(config_assign, SemanticAssignStmt)
        self.assertIsInstance(config_assign.targets[0].type, SemanticTileConfigType)
        self.assertIsInstance(config_assign.value, SemanticLiteralExpr)
        self.assertEqual(config_assign.targets[0].value, config_assign.value.value)
        self.assertIsInstance(config_assign.value.type, SemanticTileConfigType)

        self.assertIsInstance(layout_assign.value, SemanticSymbolExpr)
        self.assertEqual(layout_assign.value.value, pto.BLayout.COL_MAJOR)
        self.assertEqual(layout_assign.value.type.kind, "b_layout")

        self.assertIsInstance(secondary_assign.value, SemanticSymbolExpr)
        self.assertEqual(secondary_assign.value.value, pto.SLayout.ROW_MAJOR)
        self.assertEqual(secondary_assign.value.type.kind, "s_layout")

        self.assertIsInstance(fractal_assign.value, SemanticLiteralExpr)
        self.assertEqual(fractal_assign.value.value, 16)
        self.assertIsInstance(fractal_assign.targets[0].type, SemanticScalarType)
        self.assertEqual(fractal_assign.targets[0].type.dtype, pto.i32)

        self.assertIsInstance(pad_assign.value, SemanticSymbolExpr)
        self.assertEqual(pad_assign.value.value, pto.PadValue.ZERO)
        self.assertIsInstance(pad_assign.targets[0].type, SemanticPadValueType)
        self.assertEqual(pad_assign.targets[0].type.element_dtype, pto.f16)

        self.assertIsInstance(pad_direct_assign.value, SemanticSymbolExpr)
        self.assertEqual(pad_direct_assign.value.value, pto.PadValue.ZERO)
        self.assertIsInstance(pad_direct_assign.targets[0].type, SemanticPadValueType)
        self.assertEqual(pad_direct_assign.targets[0].type.element_dtype, pto.f16)

        self.assertIsInstance(pad_scalar_assign.value, SemanticLiteralExpr)
        self.assertEqual(pad_scalar_assign.value.value, 0.0)
        self.assertIsInstance(pad_scalar_assign.targets[0].type, SemanticScalarType)
        self.assertEqual(pad_scalar_assign.targets[0].type.dtype, pto.f16)

        self.assertIsInstance(pad_direct_scalar_assign.value, SemanticLiteralExpr)
        self.assertEqual(pad_direct_scalar_assign.value.value, 0.0)
        self.assertIsInstance(pad_direct_scalar_assign.targets[0].type, SemanticScalarType)
        self.assertEqual(pad_direct_scalar_assign.targets[0].type.dtype, pto.f16)

        self.assertEqual(rank_assign.value.value, 2)
        self.assertIsInstance(rank_assign.targets[0].type, SemanticIndexType)

        self.assertIsInstance(space_assign.value, SemanticSymbolExpr)
        self.assertEqual(space_assign.value.value, pto.MemorySpace.UB)
        self.assertEqual(space_assign.value.type.kind, "memory_space")

    def test_pad_value_eval_requires_non_null_enum(self) -> None:
        @pto.vkernel(op="tile_pad_value_null_eval", dtypes=[(pto.f16,)])
        def kernel(tile: pto.Tile):
            scalar = tile.pad_value.eval()
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        with self.assertRaises(TypeError) as ctx:
            analyze_frontend_kernel(build_frontend_kernel_node(specialized))

        self.assertIn("PadValue.NULL.eval() is invalid", str(ctx.exception))

    def test_standalone_pad_value_eval_accepts_explicit_dtype(self) -> None:
        @pto.vkernel(op="standalone_pad_value_eval_dtype", dtypes=[(pto.f32,)])
        def kernel(tile: pto.Tile):
            scalar = pto.PadValue.MAX.eval(pto.f32)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        scalar_assign = semantic_kernel.body[0]

        self.assertIsInstance(scalar_assign, SemanticAssignStmt)
        self.assertIsInstance(scalar_assign.value, SemanticLiteralExpr)
        self.assertAlmostEqual(scalar_assign.value.value, pto.PadValue.MAX.eval(pto.f32))
        self.assertIsInstance(scalar_assign.targets[0].type, SemanticScalarType)
        self.assertEqual(scalar_assign.targets[0].type.dtype, pto.f32)

    def test_standalone_pad_value_eval_accepts_static_dtype_binding(self) -> None:
        @pto.vkernel(op="standalone_pad_value_eval_dtype_binding", dtypes=[(pto.f32,)])
        def kernel(tile: pto.Tile):
            dtype = tile.element_type
            scalar = pto.PadValue.MAX.eval(dtype)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        dtype_assign, scalar_assign = semantic_kernel.body[:2]

        self.assertIsInstance(dtype_assign, SemanticAssignStmt)
        self.assertIsInstance(dtype_assign.value, SemanticSymbolExpr)
        self.assertEqual(dtype_assign.value.value, pto.f32)

        self.assertIsInstance(scalar_assign, SemanticAssignStmt)
        self.assertIsInstance(scalar_assign.value, SemanticLiteralExpr)
        self.assertAlmostEqual(scalar_assign.value.value, pto.PadValue.MAX.eval(pto.f32))
        self.assertIsInstance(scalar_assign.targets[0].type, SemanticScalarType)
        self.assertEqual(scalar_assign.targets[0].type.dtype, pto.f32)

    def test_static_dtype_binding_supports_constructor_call_surface(self) -> None:
        @pto.vkernel(op="static_dtype_binding_constructor_unique", dtypes=[(pto.i32,)])
        def kernel(tile: pto.Tile):
            idx_dtype = tile.element_type
            cols = tile.shape[1]
            zero_idx = idx_dtype(0)
            v_col = idx_dtype(cols)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        dtype_assign, cols_assign, zero_assign, cast_assign = semantic_kernel.body[:4]

        self.assertIsInstance(dtype_assign, SemanticAssignStmt)
        self.assertIsInstance(dtype_assign.value, SemanticSymbolExpr)
        self.assertEqual(dtype_assign.value.value, pto.i32)

        self.assertIsInstance(cols_assign, SemanticAssignStmt)
        self.assertIsInstance(cols_assign.targets[0].type, SemanticIndexType)

        self.assertIsInstance(zero_assign, SemanticAssignStmt)
        self.assertIsInstance(zero_assign.value, SemanticLiteralExpr)
        self.assertEqual(zero_assign.value.value, 0)
        self.assertIsInstance(zero_assign.targets[0].type, SemanticScalarType)
        self.assertEqual(zero_assign.targets[0].type.dtype, pto.i32)

        self.assertIsInstance(cast_assign, SemanticAssignStmt)
        self.assertIsInstance(cast_assign.value, SemanticCallExpr)
        self.assertEqual(cast_assign.value.namespace, "pto")
        self.assertEqual(cast_assign.value.name, "i32")
        self.assertIsInstance(cast_assign.targets[0].type, SemanticScalarType)
        self.assertEqual(cast_assign.targets[0].type.dtype, pto.i32)

    def test_unsigned_integer_constants_lower_with_signless_arith_types(self) -> None:
        @pto.vkernel(op="tile_pad_value_ui32_max_eval_unique", dtypes=[(pto.ui32,)])
        def kernel(tile: pto.Tile):
            scalar = tile.pad_value.eval()
            explicit = pto.ui32(4294967295)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 16),
                memory_space=pto.MemorySpace.UB,
                config=pto.TileConfig.from_mapping(
                    {
                        "pad_value": pto.PadValue.MAX,
                    }
                ),
            )
        )

        text = specialized.mlir_text()
        self.assertIn("dtype=ui32", text)
        self.assertIn("arith.constant 4294967295 : i32", text)
        self.assertNotIn("arith.constant 4294967295 : ui32", text)

    def test_cached_unsigned_integer_constructor_constant_preserves_typed_bridge(self) -> None:
        @pto.vkernel(
            op="cached_ui16_constructor_constant_bridge_unique",
            dtypes=[(pto.ui16, pto.ui16)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            all_mask = pto.make_mask(pto.ui16, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            biased = pto.vadds(vec, pto.ui16(1), all_mask)
            out = pto.vadds(biased, pto.ui16(1), all_mask)
            pto.vsts(out, dst, 0, all_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertEqual(text.count("arith.constant 1 : i16"), 1)
        self.assertEqual(text.count(": i16 to ui16"), 1)
        self.assertNotIn("arith.constant 1 : ui16", text)

    def test_narrow_typed_integer_zero_constructors_lower_with_signless_bridge(self) -> None:
        @pto.vkernel(op="si16_zero_constructor_bridge_unique", dtypes=[(pto.si16, pto.si16)], advanced=True)
        def si16_kernel(dst: pto.Tile, src: pto.Tile):
            all_mask = pto.make_mask(pto.si16, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vadds(vec, pto.si16(0), all_mask)
            pto.vsts(out, dst, 0, all_mask)
            return None

        @pto.vkernel(op="ui16_zero_constructor_bridge_unique", dtypes=[(pto.ui16, pto.ui16)], advanced=True)
        def ui16_kernel(dst: pto.Tile, src: pto.Tile):
            all_mask = pto.make_mask(pto.ui16, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vadds(vec, pto.ui16(0), all_mask)
            pto.vsts(out, dst, 0, all_mask)
            return None

        @pto.vkernel(op="si8_zero_constructor_bridge_unique", dtypes=[(pto.si8, pto.si8)], advanced=True)
        def si8_kernel(dst: pto.Tile, src: pto.Tile):
            all_mask = pto.make_mask(pto.si8, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vadds(vec, pto.si8(0), all_mask)
            pto.vsts(out, dst, 0, all_mask)
            return None

        @pto.vkernel(op="ui8_zero_constructor_bridge_unique", dtypes=[(pto.ui8, pto.ui8)], advanced=True)
        def ui8_kernel(dst: pto.Tile, src: pto.Tile):
            all_mask = pto.make_mask(pto.ui8, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vadds(vec, pto.ui8(0), all_mask)
            pto.vsts(out, dst, 0, all_mask)
            return None

        tile_specs = dict(
            dst=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )
        for dtype_name, raw_type, kernel in (
            ("si16", "i16", si16_kernel),
            ("ui16", "i16", ui16_kernel),
            ("si8", "i8", si8_kernel),
            ("ui8", "i8", ui8_kernel),
        ):
            with self.subTest(dtype=dtype_name):
                text = kernel.specialize(**tile_specs).mlir_text()
                self.assertEqual(text.count(f"arith.constant 0 : {raw_type}"), 1)
                self.assertEqual(text.count(f": {raw_type} to {dtype_name}"), 1)
                self.assertNotIn(f"arith.constant 0 : {dtype_name}", text)

    def test_unsigned_pad_value_eval_broadcast_bitcasts_signless_literal(self) -> None:
        @pto.vkernel(op="tile_pad_value_ui16_vbr_unique", dtypes=[(pto.ui16,)], advanced=True)
        def kernel(tile: pto.Tile):
            scalar = tile.pad_value.eval()
            vec = pto.vbr(scalar)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 16),
                memory_space=pto.MemorySpace.UB,
                config=pto.TileConfig.from_mapping(
                    {
                        "pad_value": pto.PadValue.MAX,
                    }
                ),
            )
        )

        text = specialized.mlir_text()
        self.assertIn("dtype=ui16", text)
        self.assertIn("arith.constant 65535 : i16", text)
        self.assertIn("builtin.unrealized_conversion_cast", text)
        self.assertIn(": i16 to ui16", text)
        self.assertIn("pto.vbr", text)

    def test_index_to_unsigned_scalar_constructor_bridges_via_signless_integer(self) -> None:
        @pto.vkernel(op="index_to_ui32_constructor_unique", dtypes=[(pto.ui32,)], advanced=True)
        def kernel(tile: pto.Tile):
            cols = tile.valid_shape[1]
            for col in range(0, cols, 1):
                offset = pto.ui32(col)
                vec = pto.vbr(offset)
                mask, _ = pto.make_mask(pto.ui32, 1)
                pto.vsts(vec, tile[col, 0:], mask)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 1),
                valid_shape=(8, 1),
                memory_space=pto.MemorySpace.UB,
                config=pto.TileConfig.from_mapping(
                    {
                        "b_layout": pto.BLayout.COL_MAJOR,
                        "s_layout": pto.SLayout.NONE_BOX,
                    }
                ),
            )
        )

        text = specialized.mlir_text()
        self.assertIn("arith.index_castui", text)
        self.assertIn(": index to i32", text)
        self.assertIn("builtin.unrealized_conversion_cast", text)
        self.assertIn(": i32 to ui32", text)
        self.assertNotIn(": index to ui32", text)

    def test_index_to_ui16_scalar_constructor_bridges_via_signless_integer(self) -> None:
        @pto.vkernel(op="index_to_ui16_constructor_unique", dtypes=[(pto.ui16,)], advanced=True)
        def kernel(tile: pto.Tile):
            cols = tile.valid_shape[1]
            for col in range(0, cols, 1):
                offset = pto.ui16(col)
                vec = pto.vbr(offset)
                mask, _ = pto.make_mask(pto.ui16, 1)
                pto.vsts(vec, tile[col, 0:], mask)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 1),
                valid_shape=(8, 1),
                memory_space=pto.MemorySpace.UB,
                config=pto.TileConfig.from_mapping(
                    {
                        "b_layout": pto.BLayout.COL_MAJOR,
                        "s_layout": pto.SLayout.NONE_BOX,
                    }
                ),
            )
        )

        text = specialized.mlir_text()
        self.assertIn("arith.index_castui", text)
        self.assertIn(": index to i32", text)
        self.assertIn("arith.trunci", text)
        self.assertIn(": i32 to i16", text)
        self.assertIn("builtin.unrealized_conversion_cast", text)
        self.assertIn(": i16 to ui16", text)
        self.assertNotIn(": index to ui16", text)

    def test_index_to_ui8_scalar_constructor_bridges_via_signless_integer(self) -> None:
        @pto.vkernel(op="index_to_ui8_constructor_unique", dtypes=[(pto.ui8,)], advanced=True)
        def kernel(tile: pto.Tile):
            cols = tile.valid_shape[1]
            for col in range(0, cols, 1):
                offset = pto.ui8(col)
                vec = pto.vbr(offset)
                mask, _ = pto.make_mask(pto.ui8, 1)
                pto.vsts(vec, tile[col, 0:], mask)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 1),
                valid_shape=(8, 1),
                memory_space=pto.MemorySpace.UB,
                config=pto.TileConfig.from_mapping(
                    {
                        "b_layout": pto.BLayout.COL_MAJOR,
                        "s_layout": pto.SLayout.NONE_BOX,
                    }
                ),
            )
        )

        text = specialized.mlir_text()
        self.assertIn("arith.index_castui", text)
        self.assertIn(": index to i32", text)
        self.assertIn("arith.trunci", text)
        self.assertIn(": i32 to i8", text)
        self.assertIn("builtin.unrealized_conversion_cast", text)
        self.assertIn(": i8 to ui8", text)
        self.assertNotIn(": index to ui8", text)

    def test_index_to_si8_scalar_constructor_bridges_via_signless_integer(self) -> None:
        @pto.vkernel(op="index_to_si8_constructor_unique", dtypes=[(pto.si8,)], advanced=True)
        def kernel(tile: pto.Tile):
            cols = tile.valid_shape[1]
            for col in range(0, cols, 1):
                offset = pto.si8(col)
                vec = pto.vbr(offset)
                mask, _ = pto.make_mask(pto.si8, 1)
                pto.vsts(vec, tile[col, 0:], mask)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 1),
                valid_shape=(8, 1),
                memory_space=pto.MemorySpace.UB,
                config=pto.TileConfig.from_mapping(
                    {
                        "b_layout": pto.BLayout.COL_MAJOR,
                        "s_layout": pto.SLayout.NONE_BOX,
                    }
                ),
            )
        )

        text = specialized.mlir_text()
        self.assertIn("arith.index_cast", text)
        self.assertIn(": index to i32", text)
        self.assertIn("arith.trunci", text)
        self.assertIn(": i32 to i8", text)
        self.assertIn("builtin.unrealized_conversion_cast", text)
        self.assertIn(": i8 to si8", text)
        self.assertNotIn(": index to si8", text)

    def test_index_to_i16_scalar_constructor_lowers_via_index_cast_then_trunci(self) -> None:
        @pto.vkernel(op="index_to_i16_constructor_unique", dtypes=[(pto.i16,)], advanced=True)
        def kernel(tile: pto.Tile):
            cols = tile.valid_shape[1]
            for col in range(0, cols, 1):
                offset = pto.i16(col)
                vec = pto.vbr(offset)
                mask, _ = pto.make_mask(pto.i16, 1)
                pto.vsts(vec, tile[col, 0:], mask)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 1),
                valid_shape=(8, 1),
                memory_space=pto.MemorySpace.UB,
                config=pto.TileConfig.from_mapping(
                    {
                        "b_layout": pto.BLayout.COL_MAJOR,
                        "s_layout": pto.SLayout.NONE_BOX,
                    }
                ),
            )
        )

        text = specialized.mlir_text()
        self.assertIn("arith.index_cast", text)
        self.assertIn(": index to i32", text)
        self.assertIn("arith.trunci", text)
        self.assertIn(": i32 to i16", text)
        self.assertNotIn("builtin.unrealized_conversion_cast", text)
        self.assertNotIn(": index to i16", text)

    def test_index_to_si16_scalar_constructor_bridges_via_signless_integer(self) -> None:
        @pto.vkernel(op="index_to_si16_constructor_unique", dtypes=[(pto.si16,)], advanced=True)
        def kernel(tile: pto.Tile):
            cols = tile.valid_shape[1]
            for col in range(0, cols, 1):
                offset = pto.si16(col)
                vec = pto.vbr(offset)
                mask, _ = pto.make_mask(pto.si16, 1)
                pto.vsts(vec, tile[col, 0:], mask)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 1),
                valid_shape=(8, 1),
                memory_space=pto.MemorySpace.UB,
                config=pto.TileConfig.from_mapping(
                    {
                        "b_layout": pto.BLayout.COL_MAJOR,
                        "s_layout": pto.SLayout.NONE_BOX,
                    }
                ),
            )
        )

        text = specialized.mlir_text()
        self.assertIn("arith.index_cast", text)
        self.assertIn(": index to i32", text)
        self.assertIn("arith.trunci", text)
        self.assertIn(": i32 to i16", text)
        self.assertIn("builtin.unrealized_conversion_cast", text)
        self.assertIn(": i16 to si16", text)
        self.assertNotIn(": index to si16", text)

    def test_index_to_32bit_integer_scalar_constructors_bridge_via_signless_integer(self) -> None:
        @pto.vkernel(op="index_to_i32_constructor_unique", dtypes=[(pto.i32,)], advanced=True)
        def i32_kernel(tile: pto.Tile):
            cols = tile.valid_shape[1]
            for col in range(0, cols, 1):
                offset = pto.i32(col)
                vec = pto.vbr(offset)
                mask, _ = pto.make_mask(pto.i32, 1)
                pto.vsts(vec, tile[col, 0:], mask)
            return None

        @pto.vkernel(op="index_to_si32_constructor_unique", dtypes=[(pto.si32,)], advanced=True)
        def si32_kernel(tile: pto.Tile):
            cols = tile.valid_shape[1]
            for col in range(0, cols, 1):
                offset = pto.si32(col)
                vec = pto.vbr(offset)
                mask, _ = pto.make_mask(pto.si32, 1)
                pto.vsts(vec, tile[col, 0:], mask)
            return None

        @pto.vkernel(op="index_to_ui32_constructor_bridge_unique", dtypes=[(pto.ui32,)], advanced=True)
        def ui32_kernel(tile: pto.Tile):
            cols = tile.valid_shape[1]
            for col in range(0, cols, 1):
                offset = pto.ui32(col)
                vec = pto.vbr(offset)
                mask, _ = pto.make_mask(pto.ui32, 1)
                pto.vsts(vec, tile[col, 0:], mask)
            return None

        tile_spec = pto.TileSpecialization(
            shape=(8, 1),
            valid_shape=(8, 1),
            memory_space=pto.MemorySpace.UB,
            config=pto.TileConfig.from_mapping(
                {
                    "b_layout": pto.BLayout.COL_MAJOR,
                    "s_layout": pto.SLayout.NONE_BOX,
                }
            ),
        )

        for dtype_name, op_name, kernel in (
            ("i32", "arith.index_cast", i32_kernel),
            ("si32", "arith.index_cast", si32_kernel),
            ("ui32", "arith.index_castui", ui32_kernel),
        ):
            with self.subTest(dtype=dtype_name):
                text = kernel.specialize(tile=tile_spec).mlir_text()
                self.assertIn(op_name, text)
                self.assertIn(": index to i32", text)
                if dtype_name == "i32":
                    self.assertNotIn("builtin.unrealized_conversion_cast", text)
                else:
                    self.assertIn("builtin.unrealized_conversion_cast", text)
                    self.assertIn(f": i32 to {dtype_name}", text)
                    self.assertNotIn(f": index to {dtype_name}", text)

    def test_index_to_64bit_integer_scalar_constructors_bridge_via_signless_integer(self) -> None:
        @pto.vkernel(op="index_to_i64_constructor_unique", dtypes=[(pto.i64,)], advanced=True)
        def i64_kernel(tile: pto.Tile):
            cols = tile.valid_shape[1]
            for col in range(0, cols, 1):
                offset = pto.i64(col)
                vec = pto.vbr(offset)
                mask, _ = pto.make_mask(pto.i64, 1)
                pto.vsts(vec, tile[col, 0:], mask)
            return None

        @pto.vkernel(op="index_to_si64_constructor_unique", dtypes=[(pto.si64,)], advanced=True)
        def si64_kernel(tile: pto.Tile):
            cols = tile.valid_shape[1]
            for col in range(0, cols, 1):
                offset = pto.si64(col)
                vec = pto.vbr(offset)
                mask, _ = pto.make_mask(pto.si64, 1)
                pto.vsts(vec, tile[col, 0:], mask)
            return None

        @pto.vkernel(op="index_to_ui64_constructor_unique", dtypes=[(pto.ui64,)], advanced=True)
        def ui64_kernel(tile: pto.Tile):
            cols = tile.valid_shape[1]
            for col in range(0, cols, 1):
                offset = pto.ui64(col)
                vec = pto.vbr(offset)
                mask, _ = pto.make_mask(pto.ui64, 1)
                pto.vsts(vec, tile[col, 0:], mask)
            return None

        tile_spec = pto.TileSpecialization(
            shape=(8, 1),
            valid_shape=(8, 1),
            memory_space=pto.MemorySpace.UB,
            config=pto.TileConfig.from_mapping(
                {
                    "b_layout": pto.BLayout.COL_MAJOR,
                    "s_layout": pto.SLayout.NONE_BOX,
                }
            ),
        )

        for dtype_name, op_name, kernel in (
            ("i64", "arith.index_cast", i64_kernel),
            ("si64", "arith.index_cast", si64_kernel),
            ("ui64", "arith.index_castui", ui64_kernel),
        ):
            with self.subTest(dtype=dtype_name):
                text = kernel.specialize(tile=tile_spec).mlir_text()
                self.assertIn(op_name, text)
                self.assertIn(": index to i64", text)
                if dtype_name == "i64":
                    self.assertNotIn("builtin.unrealized_conversion_cast", text)
                else:
                    self.assertIn("builtin.unrealized_conversion_cast", text)
                    self.assertIn(f": i64 to {dtype_name}", text)
                    self.assertNotIn(f": index to {dtype_name}", text)


    def test_make_mask_vlds_vsts_and_vector_families_lower_inside_strict_vecscope(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32)], advanced=True)
        def kernel(tile: pto.Tile, scale: pto.f32):
            with pto.strict_vecscope(tile, tile, scale, 0, 256, 64) as (
                src,
                dst,
                factor,
                lb,
                ub,
                step,
            ):
                for lane in range(lb, ub, step):
                    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                    vec = pto.vlds(src, lane)
                    biased = pto.vadds(vec, factor, mask)
                    summed = pto.vadd(biased, vec, mask)
                    activated = pto.vrelu(summed, mask)
                    pto.vsts(activated, dst, lane, mask)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope = semantic_kernel.body[0]
        self.assertIsInstance(vecscope, SemanticStrictVecscopeStmt)
        loop_stmt = vecscope.body[0]
        self.assertIsInstance(loop_stmt, SemanticForStmt)
        mask_assign = loop_stmt.body[0]
        self.assertIsInstance(mask_assign, SemanticAssignStmt)
        self.assertIsInstance(mask_assign.value, SemanticCallExpr)
        self.assertEqual(mask_assign.value.name, "make_mask")
        self.assertIsInstance(mask_assign.targets[0].type, SemanticMaskType)
        self.assertIsInstance(loop_stmt.body[-1], SemanticVectorStoreStmt)

        text = specialized.mlir_text()
        self.assertRegex(text, r'%mask_\d+ = pto\.pset_b32 "PAT_ALL" : !pto\.mask<b32>')
        self.assertRegex(text, r"%vec_\d+ = pto\.vlds %src_\d+\[%lane_\d+\] : !pto\.ptr<f32, ub> -> !pto\.vreg<64xf32>")
        self.assertRegex(text, r"%biased_\d+ = pto\.vadds %vec_\d+, %factor_\d+, %mask_\d+ : !pto\.vreg<64xf32>, f32, !pto\.mask<b32> -> !pto\.vreg<64xf32>")
        self.assertRegex(text, r"%summed_\d+ = pto\.vadd %biased_\d+, %vec_\d+, %mask_\d+ : !pto\.vreg<64xf32>, !pto\.vreg<64xf32>, !pto\.mask<b32> -> !pto\.vreg<64xf32>")
        self.assertRegex(text, r"%activated_\d+ = pto\.vrelu %summed_\d+, %mask_\d+ : !pto\.vreg<64xf32>, !pto\.mask<b32> -> !pto\.vreg<64xf32>")
        self.assertRegex(text, r"pto\.vsts %activated_\d+, %dst_\d+\[%lane_\d+\], %mask_\d+ : !pto\.vreg<64xf32>, !pto\.ptr<f32, ub>, !pto\.mask<b32>")

    def test_vrelu_accepts_i32_inside_strict_vecscope(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.i32, pto.i32)], advanced=True)
        def kernel(tile: pto.Tile, bias: pto.i32):
            with pto.strict_vecscope(tile, tile, bias, 0, 256, 64) as (
                src,
                dst,
                offset,
                lb,
                ub,
                step,
            ):
                for lane in range(lb, ub, step):
                    mask = pto.make_mask(pto.i32, pto.PAT.ALL)
                    vec = pto.vlds(src, lane)
                    shifted = pto.vadds(vec, offset, mask)
                    activated = pto.vrelu(shifted, mask)
                    pto.vsts(activated, dst, lane, mask)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 16),
                memory_space=pto.MemorySpace.UB,
            ),
        )

        text = specialized.mlir_text()
        self.assertRegex(text, r'%mask_\d+ = pto\.pset_b32 "PAT_ALL" : !pto\.mask<b32>')
        self.assertRegex(text, r"%vec_\d+ = pto\.vlds %src_\d+\[%lane_\d+\] : !pto\.ptr<i32, ub> -> !pto\.vreg<64xi32>")
        self.assertRegex(text, r"%shifted_\d+ = pto\.vadds %vec_\d+, %offset_\d+, %mask_\d+ : !pto\.vreg<64xi32>, i32, !pto\.mask<b32> -> !pto\.vreg<64xi32>")
        self.assertRegex(text, r"%activated_\d+ = pto\.vrelu %shifted_\d+, %mask_\d+ : !pto\.vreg<64xi32>, !pto\.mask<b32> -> !pto\.vreg<64xi32>")
        self.assertRegex(text, r"pto\.vsts %activated_\d+, %dst_\d+\[%lane_\d+\], %mask_\d+ : !pto\.vreg<64xi32>, !pto\.ptr<i32, ub>, !pto\.mask<b32>")

    def test_tail_make_mask_lowers_to_typed_plt_and_updates_remaining(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.i32)], advanced=True)
        def kernel(tile: pto.Tile, remaining: pto.i32):
            with pto.strict_vecscope(tile, tile, remaining, 0, 64, 64) as (src, dst, rem_in, lb, ub, step):
                mask, next_remaining = pto.make_mask(pto.f32, rem_in)
                vec = pto.vlds(src, lb)
                pto.vsts(vec, dst, lb, mask)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope = semantic_kernel.body[0]
        self.assertIsInstance(vecscope, SemanticStrictVecscopeStmt)
        mask_assign = vecscope.body[0]
        self.assertIsInstance(mask_assign, SemanticAssignStmt)
        self.assertEqual(mask_assign.value.name, "make_mask")
        self.assertEqual(len(mask_assign.targets), 2)
        self.assertIsInstance(mask_assign.targets[0].type, SemanticMaskType)
        self.assertIsInstance(mask_assign.targets[1].type, SemanticScalarType)
        self.assertEqual(mask_assign.targets[1].type.dtype, pto.i32)

        text = specialized.mlir_text()
        self.assertRegex(
            text,
            r"%mask_\d+, %next_remaining_\d+ = pto\.plt_b32 %rem_in_\d+ : i32 -> !pto\.mask<b32>, i32",
        )
        self.assertIn(
            "pto.vsts %vec_",
            text,
        )

    def test_nested_index_arithmetic_lowers_before_vector_accesses(self) -> None:
        @pto.vkernel(
            op="eltwise",
            dtypes=[(pto.f32, pto.f32, pto.f32)],
            advanced=True,
        )
        def kernel(
            lhs_tile: pto.Tile,
            rhs_tile: pto.Tile,
            dst_tile: pto.Tile,
        ):
            rows = lhs_tile.shape[0]
            cols = lhs_tile.shape[1]
            row_stride = lhs_tile.shape[1]

            with pto.strict_vecscope(
                lhs_tile,
                rhs_tile,
                dst_tile,
                rows,
                cols,
                row_stride,
                0,
                rows,
                1,
            ) as (lhs, rhs, dst, valid_rows, valid_cols, stride, row_lb, row_ub, row_step):
                for row in range(row_lb, row_ub, row_step):
                    for lane in range(0, valid_cols, 64):
                        offset = row * stride + lane
                        mask, next_remaining = pto.make_mask(pto.f32, valid_cols - lane)
                        summed = pto.vadd(pto.vlds(lhs, offset), pto.vlds(rhs, offset), mask)
                        pto.vsts(summed, dst, offset, mask)
            return None

        specialized = kernel.specialize(
            lhs_tile=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            rhs_tile=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            dst_tile=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertRegex(text, r"%tmp_\d+ = arith\.muli %row_\d+, %stride_\d+ : index")
        self.assertRegex(text, r"%offset_\d+ = arith\.addi %tmp_\d+, %lane_\d+ : index")
        self.assertRegex(text, r"%tmp_\d+ = arith\.subi %valid_cols_\d+, %lane_\d+ : index")
        self.assertRegex(text, r"%tmp_\d+ = arith\.index_cast %tmp_\d+ : index to i32")
        self.assertIn("pto.plt_b32", text)
        self.assertIn("pto.vadd", text)

    def test_scalar_binary_arithmetic_supports_float_and_integer_paths(self) -> None:
        @pto.vkernel(
            op="scalar_binary_arithmetic_unique",
            dtypes=[(pto.f32, pto.f32, pto.i32)],
            advanced=True,
        )
        def kernel(dst_tile: pto.Tile, src_tile: pto.Tile, gate: pto.i32):
            rows = src_tile.shape[0]
            cols = src_tile.shape[1]
            with pto.strict_vecscope(
                src_tile,
                dst_tile,
                gate,
                rows,
                cols,
                0,
                rows,
                1,
            ) as (src, dst, in_gate, valid_rows, valid_cols, row_lb, row_ub, row_step):
                for row in range(row_lb, row_ub, row_step):
                    for lane in range(0, valid_cols, 64):
                        half = in_gate // pto.i32(2)
                        remain = in_gate % pto.i32(7)
                        factor = pto.f32(half) + pto.f32(remain) * pto.f32(0.5)
                        mask, _ = pto.make_mask(pto.f32, valid_cols - lane)
                        vec = pto.vlds(src, lane)
                        vec = pto.vmuls(vec, factor, mask)
                        pto.vsts(vec, dst, lane, mask)
            return None

        specialized = kernel.specialize(
            dst_tile=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src_tile=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertRegex(text, r"= arith\.floordivsi %in_gate_\d+, %c2_i32 : i32")
        self.assertRegex(text, r"= arith\.remsi %in_gate_\d+, %c7_i32 : i32")
        self.assertRegex(text, r"%c0_5_f32 = arith\.constant 0\.5 : f32")
        self.assertRegex(text, r"= arith\.mulf %tmp_\d+, %c0_5_f32 : f32")
        self.assertRegex(text, r"= arith\.addf %tmp_\d+, %tmp_\d+ : f32")

    def test_index_floordiv_lowers_to_divui_instead_of_floordivsi(self) -> None:
        @pto.vkernel(
            op="index_floordiv_lowering_unique",
            dtypes=[(pto.f32, pto.f32, pto.f32)],
            advanced=True,
        )
        def kernel(
            lhs_tile: pto.Tile,
            rhs_tile: pto.Tile,
            dst_tile: pto.Tile,
        ):
            rows = lhs_tile.shape[0]
            cols = lhs_tile.shape[1]
            with pto.strict_vecscope(
                lhs_tile,
                rhs_tile,
                dst_tile,
                rows,
                cols,
                0,
                rows,
                1,
            ) as (lhs, rhs, dst, valid_rows, valid_cols, row_lb, row_ub, row_step):
                for row in range(row_lb, row_ub, row_step):
                    for lane in range(0, valid_cols, 64):
                        row_bucket = row // valid_cols
                        offset = row_bucket * valid_cols + lane
                        mask, _ = pto.make_mask(pto.f32, valid_cols - lane)
                        summed = pto.vadd(pto.vlds(lhs, offset), pto.vlds(rhs, offset), mask)
                        pto.vsts(summed, dst, offset, mask)
            return None

        specialized = kernel.specialize(
            lhs_tile=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            rhs_tile=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            dst_tile=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertRegex(text, r"= arith\.divui %row_\d+, %valid_cols_\d+ : index")
        self.assertNotRegex(text, r"arith\.floordivsi .*: index")

    def test_scalar_bitwise_and_shift_ops_lower_for_signed_and_unsigned(self) -> None:
        @pto.vkernel(
            op="scalar_bitwise_shift_unique",
            dtypes=[(pto.i32, pto.ui32)],
            advanced=True,
        )
        def kernel(signed_val: pto.i32, unsigned_val: pto.ui32):
            signed_mix = (signed_val & pto.i32(15)) | pto.i32(1)
            signed_mix = signed_mix ^ pto.i32(2)
            signed_mix = signed_mix >> pto.i32(1)
            signed_mix = signed_mix << pto.i32(3)

            unsigned_mix = unsigned_val & pto.ui32(31)
            unsigned_mix = unsigned_mix >> pto.ui32(2)
            unsigned_mix = unsigned_mix << pto.ui32(1)
            unsigned_mix = unsigned_mix ^ pto.ui32(7)
            return None

        specialized = kernel.specialize()
        text = specialized.mlir_text()

        self.assertIn("arith.andi", text)
        self.assertIn("arith.ori", text)
        self.assertIn("arith.xori", text)
        self.assertRegex(text, r"= arith\.shrsi %\w+_\d+, %c1_i32 : i32")
        self.assertRegex(text, r"= arith\.shli %\w+_\d+, %c3_i32 : i32")
        self.assertRegex(text, r"= arith\.shrui %\w+_\d+, %c2_ui32 : ui32")
        self.assertRegex(text, r"= arith\.shli %\w+_\d+, %c1_ui32 : ui32")

    def test_scalar_bitwise_rejects_float_operands(self) -> None:
        @pto.vkernel(op="scalar_bitwise_float_reject_unique", dtypes=[(pto.f32,)])
        def kernel(value: pto.f32):
            _ = value & pto.f32(1.0)
            return None

        specialized = kernel.specialize()
        with self.assertRaises(TypeError) as ctx:
            specialized.mlir_text()
        self.assertIn("mod/floordiv/bitwise/shift for integer", str(ctx.exception))

    def test_stable_mode_infers_vecscope_and_lowers_tile_vector_sugar(self) -> None:
        @pto.vkernel(op="tadd_stable", dtypes=[(pto.f32, pto.f32, pto.f32)])
        def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
            dtype = dst.element_type
            rows, cols = dst.valid_shape
            all_mask = pto.make_mask(dtype, pto.PAT.ALL)
            for row in range(0, rows, 1):
                for col in range(0, cols, pto.get_lanes(dtype)):
                    lhs = pto.vlds(src0[row, col:])
                    rhs = pto.vlds(src1[row, col:])
                    summed = pto.vadd(lhs, rhs, all_mask)
                    pto.vsts(summed, dst[row, col:], all_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src0=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope_stmts = [stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt)]
        self.assertEqual(len(vecscope_stmts), 1)

        text = specialized.mlir_text()
        self.assertIn("pto.vecscope {", text)
        self.assertNotIn("pto.strict_vecscope(", text)
        self.assertRegex(text, r"memref\.subview %tmp_\d+\[%row_\d+, %col_\d+\] \[%c1, %tmp_\d+\] \[%c1, %c1\]")
        self.assertRegex(text, r"pto\.vlds %tmp_\d+\[%c0\]")
        self.assertRegex(text, r"pto\.vsts %summed_\d+, %tmp_\d+\[%c0\], %(?:all_mask|mask)_\d+")

    def test_advanced_mode_infers_vecscope_and_lowers_tile_vector_sugar(self) -> None:
        @pto.vkernel(op="tadd", dtypes=[(pto.f32, pto.f32, pto.f32)], advanced=True)
        def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
            dtype = dst.element_type
            rows, cols = dst.valid_shape
            all_mask = pto.make_mask(dtype, pto.PAT.ALL)
            for row in range(0, rows, 1):
                for col in range(0, cols, pto.get_lanes(dtype)):
                    lhs = pto.vlds(src0[row, col:])
                    rhs = pto.vlds(src1[row, col:])
                    summed = pto.vadd(lhs, rhs, all_mask)
                    pto.vsts(summed, dst[row, col:], all_mask)
            return None

        self.assertTrue(kernel.advanced_enabled)

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src0=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope_stmts = [stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt)]
        self.assertEqual(len(vecscope_stmts), 1)
        vecscope = vecscope_stmts[0]
        self.assertIsInstance(vecscope, SemanticVecscopeStmt)
        outer_loop = next(stmt for stmt in vecscope.body if isinstance(stmt, SemanticForStmt))
        self.assertIsInstance(outer_loop, SemanticForStmt)
        inner_loop = outer_loop.body[0]
        self.assertIsInstance(inner_loop, SemanticForStmt)
        self.assertTrue(inner_loop.body)

        text = specialized.mlir_text()
        self.assertIn("// tilelang.advanced = True", text)
        self.assertIn("pto.vecscope {", text)
        self.assertNotIn("pto.strict_vecscope(", text)
        self.assertRegex(text, r"pto\.vecscope \{\n(?:.|\n)*scf\.for %row_")
        self.assertEqual(text.count("pto.vecscope {"), 1)
        self.assertIn("!pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64", text)
        self.assertIn("pto.tile_valid_rows %arg0", text)
        self.assertIn("pto.tile_valid_cols %arg0", text)
        self.assertNotIn("pto.tile_valid_rows %arg1", text)
        self.assertNotIn("pto.tile_valid_cols %arg1", text)
        self.assertNotIn("pto.tile_valid_rows %arg2", text)
        self.assertNotIn("pto.tile_valid_cols %arg2", text)
        self.assertRegex(text, r"pto\.tile_buf_addr %arg1 : !pto\.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64")
        self.assertRegex(text, r"memref\.subview %tmp_\d+\[%row_\d+, %col_\d+\] \[%c1, %tmp_\d+\] \[%c1, %c1\] : memref<8x64xf32, #pto\.address_space<vec>> to memref<\?x\?xf32, strided<\[\?, \?\], offset: \?>, #pto\.address_space<vec>>")
        self.assertRegex(text, r"pto\.vlds %tmp_\d+\[%c0\] : memref<\?x\?xf32, strided<\[\?, \?\], offset: \?>, #pto\.address_space<vec>> -> !pto\.vreg<64xf32>")
        self.assertRegex(text, r"pto\.vsts %summed_\d+, %tmp_\d+\[%c0\], %(?:all_mask|mask)_\d+ : !pto\.vreg<64xf32>, memref<\?x\?xf32, strided<\[\?, \?\], offset: \?>, #pto\.address_space<vec>>, !pto\.mask<b32>")
        self.assertNotRegex(text, r"arith\.muli %row_\d+, %c64 : index")
        self.assertNotRegex(text, r"arith\.addi %tmp_\d+, %col_\d+ : index")
        self.assertLess(text.index("pto.tile_buf_addr %arg1"), text.index("pto.vecscope {"))
        self.assertLess(text.index("pto.tile_buf_addr %arg2"), text.index("pto.vecscope {"))
        self.assertLess(text.index("pto.tile_buf_addr %arg0"), text.index("pto.vecscope {"))
        self.assertLess(text.index("pto.tile_valid_rows %arg0"), text.index("pto.vecscope {"))
        self.assertLess(text.index("pto.tile_valid_cols %arg0"), text.index("pto.vecscope {"))
        self.assertLess(text.index("pto.vecscope {"), text.index("scf.for %row_"))
        self.assertLess(text.rindex("pto.vecscope {"), text.index("return"))

    def test_element_type_valid_shape_and_get_lanes_surface_lower_in_advanced_mode(self) -> None:
        @pto.vkernel(op="tadd", dtypes=[(pto.f32, pto.f32, pto.f32)], advanced=True)
        def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
            dtype = dst.element_type
            valid_rows, valid_cols = dst.valid_shape
            remained = valid_cols
            for row in range(0, valid_rows, 1):
                for col in range(0, valid_cols, pto.get_lanes(dtype)):
                    mask, remained = pto.make_mask(dtype, remained)
                    summed = pto.vadd(pto.vlds(src0[row, col:]), pto.vlds(src1[row, col:]), mask)
                    pto.vsts(summed, dst[row, col:], mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src0=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("step %c64", text)
        self.assertRegex(text, r"%mask_\d+, %remained_\d+ = pto\.plt_b32 %remained_iter_\d+ : i32 -> !pto\.mask<b32>, i32")
        self.assertIn("pto.vadd", text)
        self.assertIn("pto.vsts", text)
        self.assertIn("pto.tile_valid_rows %arg0", text)
        self.assertIn("pto.tile_valid_cols %arg0", text)
        self.assertRegex(text, r"memref\.subview %tmp_\d+\[%row_\d+, %col_\d+\] \[%c1, %tmp_\d+\] \[%c1, %c1\]")
        self.assertRegex(text, r"pto\.vlds %tmp_\d+\[%c0\]")
        self.assertRegex(text, r"pto\.vsts %summed_\d+, %tmp_\d+\[%c0\], %mask_\d+")

    def test_bytewidth_surface_lowers_to_constant_index(self) -> None:
        @pto.vkernel(op="bytewidth_query_unique", dtypes=[(pto.f32,)], advanced=True)
        def kernel(dst: pto.Tile):
            elem_bytes = pto.bytewidth(dst.element_type)
            rows, cols = dst.valid_shape
            for col in range(0, cols, elem_bytes):
                current = col
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("= arith.constant 4 : index", text)
        self.assertRegex(text, r"scf\.for %col_\d+ = %c0 to %cols_\d+ step %elem_bytes_\d+")
        self.assertIn("pto.tile_valid_cols %arg0", text)

    def test_elements_per_vreg_alias_surface_lowers_to_constant_index(self) -> None:
        @pto.vkernel(op="elements_per_vreg_query_unique", dtypes=[(pto.f32,)], advanced=True)
        def kernel(dst: pto.Tile):
            lanes = pto.elements_per_vreg(dst.element_type)
            rows, cols = dst.valid_shape
            for col in range(0, cols, lanes):
                current = col
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("= arith.constant 64 : index", text)
        self.assertRegex(text, r"scf\.for %col_\d+ = %c0 to %cols_\d+ step %lanes_\d+")
        self.assertIn("pto.tile_valid_cols %arg0", text)

    def test_vreg_type_constructor_and_annotation_match_vector_value(self) -> None:
        @pto.vkernel(op="vreg_type_annotation_unique", dtypes=[(pto.f32,)], advanced=True)
        def kernel(dst: pto.Tile):
            dtype = dst.element_type
            vec_ty = pto.vreg(dtype)
            mask = pto.make_mask(dtype, pto.PAT.ALL)
            vec: pto.vreg(dtype) = pto.vlds(dst, 0)
            pto.vsts(vec, dst, 0, mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope = next(
            stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt)
        )
        self.assertIsInstance(vecscope, SemanticVecscopeStmt)
        vec_assign = next(
            stmt
            for stmt in vecscope.body
            if isinstance(stmt, SemanticAssignStmt)
            and stmt.targets[0].name == "vec"
        )
        self.assertIsInstance(vec_assign.targets[0].type, SemanticVRegType)
        self.assertEqual(vec_assign.targets[0].type.element_dtype, pto.f32)
        self.assertEqual(vec_assign.targets[0].type.lanes, 64)
        self.assertTrue(
            any(
                isinstance(stmt, SemanticAssignStmt)
                and stmt.targets[0].name == "vec_ty"
                for stmt in semantic_kernel.body
            )
        )

        text = specialized.mlir_text()
        self.assertRegex(text, r"%vec_\d+ = pto\.vlds %tmp_\d+\[%c0\] : memref<8x64xf32, #pto\.address_space<vec>> -> !pto\.vreg<64xf32>")
        self.assertRegex(text, r"pto\.vsts %vec_\d+, %tmp_\d+\[%c0\], %mask_\d+ : !pto\.vreg<64xf32>, memref<8x64xf32, #pto\.address_space<vec>>, !pto\.mask<b32>")

    def test_mask_type_annotation_matches_make_mask_result(self) -> None:
        @pto.vkernel(op="mask_type_annotation_unique", dtypes=[(pto.f32,)], advanced=True)
        def kernel(dst: pto.Tile):
            mask_ty = pto.mask_b32
            mask: pto.mask_b32 = pto.make_mask(pto.f32, pto.PAT.ALL)
            alias_mask: mask_ty = mask
            vec: pto.vreg(pto.f32) = pto.vlds(dst, 0)
            pto.vsts(vec, dst, 0, alias_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertRegex(text, r'%mask_\d+ = pto\.pset_b32 "PAT_ALL" : !pto\.mask<b32>')
        self.assertRegex(text, r"pto\.vsts %vec_\d+, %tmp_\d+\[%c0\], %\w+ : !pto\.vreg<64xf32>, memref<8x64xf32, #pto\.address_space<vec>>, !pto\.mask<b32>")

    def test_extended_float_vector_ops_surface_lowers(self) -> None:
        @pto.vkernel(
            op="extended_float_vector_ops_unique",
            dtypes=[(pto.f32, pto.f32, pto.f32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile, alpha: pto.f32):
            all_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            vec0 = pto.vlds(src, 0)
            vec1 = pto.vlds(src, 64)
            vec2 = pto.vlds(src, 128)
            vec3 = pto.vlds(src, 192)

            out = pto.vln(vec0, all_mask)
            out = pto.vsqrt(out, all_mask)
            out = pto.vrec(out, all_mask)
            out = pto.vrsqrt(out, all_mask)
            out = pto.vexpdif(out, vec1, all_mask, pto.VcvtPartMode.ODD)
            out = pto.vcadd(out, all_mask)
            out = pto.vcmax(out, all_mask)
            out = pto.vcmin(out, all_mask)
            out = pto.vmov(out, all_mask)
            out = pto.vtrc(out, all_mask)
            out = pto.vprelu(out, vec1, all_mask)
            out = pto.vlrelu(out, alpha, all_mask)
            out = pto.vcvt(out, pto.f32, all_mask)
            pto.vsts(out, dst, 0, all_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 256), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 256), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.vln", text)
        self.assertIn("pto.vsqrt", text)
        self.assertIn("pto.vrec", text)
        self.assertIn("pto.vrsqrt", text)
        self.assertIn("pto.vexpdif", text)
        self.assertIn("pto.vcadd", text)
        self.assertIn("pto.vcmax", text)
        self.assertIn("pto.vcmin", text)
        self.assertIn("pto.vmov", text)
        self.assertIn("pto.vtrc", text)
        self.assertIn("pto.vprelu", text)
        self.assertIn("pto.vlrelu", text)
        self.assertIn("pto.vcvt", text)

    def test_vexpdif_f16_surface_lowers_to_f32_half_lanes(self) -> None:
        @pto.vkernel(
            op="vexpdif_f16_surface_unique",
            dtypes=[(pto.f32, pto.f16, pto.f16)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile, max_src: pto.Tile):
            vec = pto.vlds(src, 0)
            max_vec = pto.vlds(max_src, 0)
            mask = pto.make_mask(pto.f16, pto.PAT.ALL)
            out = pto.vexpdif(vec, max_vec, mask, pto.VcvtPartMode.ODD)
            mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            pto.vsts(out, dst, 0, mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            max_src=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertRegex(
            text,
            r'pto\.vexpdif %\w+_\d+, %\w+_\d+, %\w+_\d+, "ODD" : !pto\.vreg<128xf16>, !pto\.vreg<128xf16>, !pto\.mask<b16> -> !pto\.vreg<64xf32>',
        )

    def test_vcvt_supports_keyword_attrs_with_enums(self) -> None:
        @pto.vkernel(
            op="vcvt_keyword_attrs_unique",
            dtypes=[(pto.f16, pto.f32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            src_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            dst_mask = pto.make_mask(pto.f16, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vcvt(
                vec,
                pto.f16,
                src_mask,
                rnd=pto.VcvtRoundMode.R,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.ODD,
            )
            pto.vsts(out, dst, 0, dst_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn('pto.vcvt', text)
        self.assertIn('rnd = "R"', text)
        self.assertIn('sat = "SAT"', text)
        self.assertIn('part = "ODD"', text)
        self.assertRegex(
            text,
            r"= pto\.vcvt %[^,\s]+, %[^,\s]+(?: \{[^}]+\})? : !pto\.vreg<[^>]+>, !pto\.mask<b32> -> !pto\.vreg<[^>]+>",
        )

    def test_vcvt_supports_part_t_modes_with_enum(self) -> None:
        @pto.vkernel(
            op="vcvt_part_t_enum_unique",
            dtypes=[(pto.i8, pto.f16)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            src_mask = pto.make_mask(pto.f16, pto.PAT.ALL)
            dst_mask = pto.make_mask(pto.i8, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vcvt(
                vec,
                pto.i8,
                src_mask,
                rnd=pto.VcvtRoundMode.R,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.P0,
            )
            pto.vsts(out, dst, 0, dst_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 256), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.vcvt", text)
        self.assertIn('rnd = "R"', text)
        self.assertIn('sat = "SAT"', text)
        self.assertIn('part = "P0"', text)

    def test_vcvt_supports_part_t_modes_with_canonical_string(self) -> None:
        @pto.vkernel(
            op="vcvt_part_t_string_unique",
            dtypes=[(pto.i8, pto.f16)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            src_mask = pto.make_mask(pto.f16, pto.PAT.ALL)
            dst_mask = pto.make_mask(pto.i8, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vcvt(
                vec,
                pto.i8,
                src_mask,
                rnd="R",
                sat="SAT",
                part="P3",
            )
            pto.vsts(out, dst, 0, dst_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 256), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.vcvt", text)
        self.assertIn('part = "P3"', text)

    def test_vcvt_i32_to_i64_reuses_b32_mask_and_emits_i64_vreg(self) -> None:
        @pto.vkernel(
            op="vcvt_i32_to_i64_unique",
            dtypes=[(pto.i64, pto.i32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            src_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
            dst_mask = pto.make_mask(pto.i64, pto.PAT.ALL)
            vec = pto.vlds(src, 0, dist=pto.VLoadDist.UNPK_B32)
            out = pto.vcvt(
                vec,
                pto.i64,
                src_mask,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(out, dst, 0, dst_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 32), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope = next(stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt))
        store_stmt = next(stmt for stmt in vecscope.body if isinstance(stmt, SemanticVectorStoreStmt))
        self.assertIsInstance(store_stmt.mask.type, SemanticMaskType)
        self.assertEqual(store_stmt.mask.type.granularity, "b32")

        text = specialized.mlir_text()
        self.assertIn("!pto.mask<b32>", text)
        self.assertIn('dist = "UNPK_B32"', text)
        self.assertRegex(text, r"!pto\.vreg<32xi64>")
        self.assertIn('part = "EVEN"', text)
        self.assertIn("pto.vsts", text)

    def test_vlds_dist_requires_vload_dist_enum(self) -> None:
        with self.assertRaises(TypeError) as ctx:

            @pto.vkernel(
                op="vlds_dist_requires_enum_unique",
                dtypes=[(pto.i32, pto.i32)],
                advanced=True,
            )
            def kernel(dst: pto.Tile, src: pto.Tile):
                mask = pto.make_mask(pto.i32, pto.PAT.ALL)
                vec = pto.vlds(src, 0, dist="UNPK_B32")
                pto.vsts(vec, dst, 0, mask)
                return None

            specialized = kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
                src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            )
            specialized.mlir_text()

        self.assertIn("VLoadDist enum", str(ctx.exception))

    def test_vsts_dist_requires_vstore_dist_enum(self) -> None:
        with self.assertRaises(TypeError) as ctx:

            @pto.vkernel(
                op="vsts_dist_requires_enum_unique",
                dtypes=[(pto.ui8, pto.ui8)],
                advanced=True,
            )
            def kernel(dst: pto.Tile, src: pto.Tile):
                mask = pto.make_mask(pto.ui8, pto.PAT.ALL)
                vec = pto.vlds(src, 0)
                pto.vsts(vec, dst, 0, mask, dist="NORM_B8")
                return None

            specialized = kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 256), memory_space=pto.MemorySpace.UB),
                src=pto.TileSpecialization(shape=(8, 256), memory_space=pto.MemorySpace.UB),
            )
            specialized.mlir_text()

        self.assertIn("VStoreDist enum", str(ctx.exception))

    def test_vtrc_defaults_to_round_nearest(self) -> None:
        @pto.vkernel(
            op="vtrc_default_rnd_unique",
            dtypes=[(pto.f32, pto.f32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            all_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vtrc(vec, all_mask)
            pto.vsts(out, dst, 0, all_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.vtrc", text)
        self.assertIn(', "R" :', text)
        self.assertRegex(
            text,
            r"= pto\.vtrc %[^,\s]+, %[^,\s]+, \"R\" : !pto\.vreg<[^>]+>, !pto\.mask<[^>]+> -> !pto\.vreg<[^>]+>",
        )

    def test_vtrc_supports_keyword_rnd_with_enums(self) -> None:
        @pto.vkernel(
            op="vtrc_keyword_rnd_unique",
            dtypes=[(pto.f32, pto.f32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            all_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vtrc(vec, all_mask, rnd=pto.VcvtRoundMode.F)
            pto.vsts(out, dst, 0, all_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.vtrc", text)
        self.assertIn(', "F" :', text)

    def test_vtrc_rejects_round_mode_o(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            @pto.vkernel(
                op="vtrc_round_mode_o_unique",
                dtypes=[(pto.f32, pto.f32)],
                advanced=True,
            )
            def kernel(dst: pto.Tile, src: pto.Tile):
                all_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                vec = pto.vlds(src, 0)
                out = pto.vtrc(vec, all_mask, rnd=pto.VcvtRoundMode.O)
                pto.vsts(out, dst, 0, all_mask)
                return None

            kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
                src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            ).mlir_text()

        self.assertIn("pto.vtrc rnd must be one of", str(ctx.exception))

    def test_advanced_sort_memory_ops_surface_lower(self) -> None:
        @pto.vkernel(
            op="advanced_sort_memory_ops_unique",
            dtypes=[(pto.f32, pto.f32, pto.i32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile, idx: pto.Tile):
            dst_ptr = dst.as_ptr()
            src_ptr = src.as_ptr()
            idx_ptr = idx.as_ptr()

            pto.vbitsort(dst_ptr, src_ptr, idx_ptr, 1)
            pto.vmrgsort4(
                dst_ptr,
                src_ptr,
                pto.addptr(src_ptr, 64),
                pto.addptr(src_ptr, 128),
                pto.addptr(src_ptr, 192),
                pto.i64(64),
                pto.i64(0),
            )
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 256), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 256), memory_space=pto.MemorySpace.UB),
            idx=pto.TileSpecialization(shape=(8, 256), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertRegex(
            text,
            r"pto\.vbitsort %dst_ptr_\d+, %src_ptr_\d+, %idx_ptr_\d+, %c1 : !pto\.ptr<f32, ub>, !pto\.ptr<f32, ub>, !pto\.ptr<i32, ub>, index",
        )
        self.assertRegex(
            text,
            r"pto\.vmrgsort4 %dst_ptr_\d+, %src_ptr_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %c\d+_i64, %c\d+_i64 : "
            r"!pto\.ptr<f32, ub>, !pto\.ptr<f32, ub>, !pto\.ptr<f32, ub>, !pto\.ptr<f32, ub>, !pto\.ptr<f32, ub>, i64, i64",
        )

    def test_vbitsort_helper_stays_outside_inferred_vecscope(self) -> None:
        @pto.vkernel(
            op="vbitsort_vecscope_boundary_unique",
            dtypes=[(pto.f32, pto.f32, pto.i32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile, idx: pto.Tile):
            dst_ptr = dst.as_ptr()
            src_ptr = src.as_ptr()
            idx_ptr = idx.as_ptr()

            pto.vbitsort(dst_ptr, src_ptr, idx_ptr, 1)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 256), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 256), memory_space=pto.MemorySpace.UB),
            idx=pto.TileSpecialization(shape=(8, 256), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope_stmts = [stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt)]
        self.assertEqual(vecscope_stmts, [])

        text = specialized.mlir_text()
        self.assertRegex(
            text,
            r"pto\.vbitsort %dst_ptr_\d+, %src_ptr_\d+, %idx_ptr_\d+, %c1 : !pto\.ptr<f32, ub>, !pto\.ptr<f32, ub>, !pto\.ptr<i32, ub>, index",
        )
        self.assertNotIn("pto.vecscope {", text)

    def test_vcvt_rejects_legacy_string_spellings(self) -> None:
        with self.assertRaises(TypeError) as ctx:

            @pto.vkernel(
                op="vcvt_keyword_attrs_legacy_unique",
                dtypes=[(pto.f16, pto.f32)],
                advanced=True,
            )
            def kernel(dst: pto.Tile, src: pto.Tile):
                src_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                dst_mask = pto.make_mask(pto.f16, pto.PAT.ALL)
                vec = pto.vlds(src, 0)
                out = pto.vcvt(
                    vec,
                    pto.f16,
                    src_mask,
                    rnd="ROUND_R",
                    sat="RS_ENABLE",
                    part="PART_EVEN",
                )
                pto.vsts(out, dst, 0, dst_mask)
                return None

            specialized = kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
                src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            )
            specialized.mlir_text()

        self.assertIn("pto.vcvt rnd must be a VcvtRoundMode enum", str(ctx.exception))

    def test_vcvt_requires_explicit_required_attrs_for_type_pair(self) -> None:
        with self.assertRaises(TypeError) as ctx:

            @pto.vkernel(
                op="vcvt_missing_required_attrs_unique",
                dtypes=[(pto.i32, pto.f32)],
                advanced=True,
            )
            def kernel(dst: pto.Tile, src: pto.Tile):
                src_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                dst_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
                vec = pto.vlds(src, 0)
                out = pto.vcvt(vec, pto.i32, src_mask, rnd=pto.VcvtRoundMode.R)
                pto.vsts(out, dst, 0, dst_mask)
                return None

            specialized = kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
                src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            )
            specialized.mlir_text()

        self.assertIn("requires explicit `sat=`", str(ctx.exception))

    def test_vcvt_rejects_disallowed_attrs_for_type_pair(self) -> None:
        with self.assertRaises(TypeError) as ctx:

            @pto.vkernel(
                op="vcvt_disallowed_attr_unique",
                dtypes=[(pto.i32, pto.f32)],
                advanced=True,
            )
            def kernel(dst: pto.Tile, src: pto.Tile):
                src_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                dst_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
                vec = pto.vlds(src, 0)
                out = pto.vcvt(
                    vec,
                    pto.i32,
                    src_mask,
                    rnd=pto.VcvtRoundMode.R,
                    sat=pto.VcvtSatMode.SAT,
                    part=pto.VcvtPartMode.ODD,
                )
                pto.vsts(out, dst, 0, dst_mask)
                return None

            specialized = kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
                src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            )
            specialized.mlir_text()

        self.assertIn("does not accept `part=`", str(ctx.exception))

    def test_vcvt_f16_to_i32_requires_rnd_and_part(self) -> None:
        with self.assertRaises(TypeError) as ctx:

            @pto.vkernel(
                op="vcvt_f16_to_i32_missing_rnd_unique",
                dtypes=[(pto.i32, pto.f16)],
                advanced=True,
            )
            def kernel(dst: pto.Tile, src: pto.Tile):
                src_mask = pto.make_mask(pto.f16, pto.PAT.ALL)
                dst_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
                vec = pto.vlds(src, 0, dist=pto.VLoadDist.UNPK_B16)
                out = pto.vcvt(
                    vec,
                    pto.i32,
                    src_mask,
                    part=pto.VcvtPartMode.EVEN,
                )
                pto.vsts(out, dst, 0, dst_mask)
                return None

            specialized = kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
                src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            )
            specialized.mlir_text()

        self.assertIn("requires explicit `rnd=`", str(ctx.exception))

        with self.assertRaises(TypeError) as ctx:

            @pto.vkernel(
                op="vcvt_f16_to_i32_missing_part_unique",
                dtypes=[(pto.i32, pto.f16)],
                advanced=True,
            )
            def kernel(dst: pto.Tile, src: pto.Tile):
                src_mask = pto.make_mask(pto.f16, pto.PAT.ALL)
                dst_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
                vec = pto.vlds(src, 0, dist=pto.VLoadDist.UNPK_B16)
                out = pto.vcvt(
                    vec,
                    pto.i32,
                    src_mask,
                    rnd=pto.VcvtRoundMode.R,
                )
                pto.vsts(out, dst, 0, dst_mask)
                return None

            specialized = kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
                src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            )
            specialized.mlir_text()

        self.assertIn("requires explicit `part=`", str(ctx.exception))

    def test_vcvt_f16_to_i32_rejects_sat(self) -> None:
        with self.assertRaises(TypeError) as ctx:

            @pto.vkernel(
                op="vcvt_f16_to_i32_sat_unique",
                dtypes=[(pto.i32, pto.f16)],
                advanced=True,
            )
            def kernel(dst: pto.Tile, src: pto.Tile):
                src_mask = pto.make_mask(pto.f16, pto.PAT.ALL)
                dst_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
                vec = pto.vlds(src, 0, dist=pto.VLoadDist.UNPK_B16)
                out = pto.vcvt(
                    vec,
                    pto.i32,
                    src_mask,
                    rnd=pto.VcvtRoundMode.R,
                    sat=pto.VcvtSatMode.SAT,
                    part=pto.VcvtPartMode.EVEN,
                )
                pto.vsts(out, dst, 0, dst_mask)
                return None

            specialized = kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
                src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            )
            specialized.mlir_text()

        self.assertIn("does not accept `sat=`", str(ctx.exception))

    def test_vcvt_f16_to_i32_accepts_rnd_and_part(self) -> None:
        @pto.vkernel(
            op="vcvt_f16_to_i32_attrs_unique",
            dtypes=[(pto.i32, pto.f16)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            src_mask = pto.make_mask(pto.f16, pto.PAT.ALL)
            dst_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
            vec = pto.vlds(src, 0, dist=pto.VLoadDist.UNPK_B16)
            out = pto.vcvt(
                vec,
                pto.i32,
                src_mask,
                rnd=pto.VcvtRoundMode.R,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(out, dst, 0, dst_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.vcvt", text)
        self.assertIn('rnd = "R"', text)
        self.assertIn('part = "EVEN"', text)
        self.assertNotIn('sat = "SAT"', text)

    def test_vcvt_bf16_to_f16_requires_rnd_and_sat(self) -> None:
        with self.assertRaises(TypeError) as ctx:

            @pto.vkernel(
                op="vcvt_bf16_to_f16_missing_rnd_unique",
                dtypes=[(pto.f16, pto.bf16)],
                advanced=True,
            )
            def kernel(dst: pto.Tile, src: pto.Tile):
                src_mask = pto.make_mask(pto.bf16, pto.PAT.ALL)
                dst_mask = pto.make_mask(pto.f16, pto.PAT.ALL)
                vec = pto.vlds(src, 0)
                out = pto.vcvt(
                    vec,
                    pto.f16,
                    src_mask,
                    sat=pto.VcvtSatMode.SAT,
                )
                pto.vsts(out, dst, 0, dst_mask)
                return None

            specialized = kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
                src=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            )
            specialized.mlir_text()

        self.assertIn("requires explicit `rnd=`", str(ctx.exception))

        with self.assertRaises(TypeError) as ctx:

            @pto.vkernel(
                op="vcvt_bf16_to_f16_missing_sat_unique",
                dtypes=[(pto.f16, pto.bf16)],
                advanced=True,
            )
            def kernel(dst: pto.Tile, src: pto.Tile):
                src_mask = pto.make_mask(pto.bf16, pto.PAT.ALL)
                dst_mask = pto.make_mask(pto.f16, pto.PAT.ALL)
                vec = pto.vlds(src, 0)
                out = pto.vcvt(
                    vec,
                    pto.f16,
                    src_mask,
                    rnd=pto.VcvtRoundMode.R,
                )
                pto.vsts(out, dst, 0, dst_mask)
                return None

            specialized = kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
                src=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            )
            specialized.mlir_text()

        self.assertIn("requires explicit `sat=`", str(ctx.exception))

    def test_vcvt_bf16_to_f16_rejects_part(self) -> None:
        with self.assertRaises(TypeError) as ctx:

            @pto.vkernel(
                op="vcvt_bf16_to_f16_part_unique",
                dtypes=[(pto.f16, pto.bf16)],
                advanced=True,
            )
            def kernel(dst: pto.Tile, src: pto.Tile):
                src_mask = pto.make_mask(pto.bf16, pto.PAT.ALL)
                dst_mask = pto.make_mask(pto.f16, pto.PAT.ALL)
                vec = pto.vlds(src, 0)
                out = pto.vcvt(
                    vec,
                    pto.f16,
                    src_mask,
                    rnd=pto.VcvtRoundMode.R,
                    sat=pto.VcvtSatMode.SAT,
                    part=pto.VcvtPartMode.EVEN,
                )
                pto.vsts(out, dst, 0, dst_mask)
                return None

            specialized = kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
                src=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            )
            specialized.mlir_text()

        self.assertIn("does not accept `part=`", str(ctx.exception))

    def test_vcvt_bf16_to_f16_accepts_rnd_and_sat(self) -> None:
        @pto.vkernel(
            op="vcvt_bf16_to_f16_attrs_unique",
            dtypes=[(pto.f16, pto.bf16)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            src_mask = pto.make_mask(pto.bf16, pto.PAT.ALL)
            dst_mask = pto.make_mask(pto.f16, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vcvt(
                vec,
                pto.f16,
                src_mask,
                rnd=pto.VcvtRoundMode.R,
                sat=pto.VcvtSatMode.SAT,
            )
            pto.vsts(out, dst, 0, dst_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.vcvt", text)
        self.assertIn('rnd = "R"', text)
        self.assertIn('sat = "SAT"', text)
        self.assertNotIn('part = "EVEN"', text)

    def test_vbitcast_supports_direct_interface(self) -> None:
        @pto.vkernel(
            op="vbitcast_direct_unique",
            dtypes=[(pto.f32, pto.f32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            all_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            # Load float vector
            fvec = pto.vlds(src, 0)  # !pto.vreg<64xf32>
            # Convert to integer via vbitcast
            ivec = pto.vbitcast(fvec, pto.i32)  # !pto.vreg<64xi32>
            # Convert back to float
            fvec2 = pto.vbitcast(ivec, pto.f32)  # !pto.vreg<64xf32>
            # Store result
            pto.vsts(fvec2, dst, 0, all_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.vbitcast", text)
        self.assertRegex(text, r"= pto\.vbitcast %[^:]+ : !pto\.vreg<64xf32> -> !pto\.vreg<64xi32>")
        self.assertRegex(text, r"= pto\.vbitcast %[^:]+ : !pto\.vreg<64xi32> -> !pto\.vreg<64xf32>")

    def test_vbitcast_supports_astype_syntax_sugar(self) -> None:
        @pto.vkernel(
            op="vbitcast_astype_unique",
            dtypes=[(pto.f32, pto.f32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            all_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            # Load float vector
            fvec = pto.vlds(src, 0)  # !pto.vreg<64xf32>
            # Convert to integer via astype syntax sugar
            ivec = fvec.astype(pto.i32)  # !pto.vreg<64xi32>
            # Convert back to float
            fvec2 = ivec.astype(pto.f32)  # !pto.vreg<64xf32>
            # Store result
            pto.vsts(fvec2, dst, 0, all_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.vbitcast", text)
        # astype calls should be lowered to vbitcast
        count = text.count("pto.vbitcast")
        self.assertGreaterEqual(count, 2)

    def test_vbitcast_rejects_non_vreg_input(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            @pto.vkernel(
                op="vbitcast_non_vreg_input_unique",
                dtypes=[(pto.f32, pto.f32)],
                advanced=True,
            )
            def kernel(dst: pto.Tile, src: pto.Tile):
                all_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                # Try to vbitcast a non-vector value
                scalar = pto.f32(1.0)
                ivec = pto.vbitcast(scalar, pto.i32)
                pto.vsts(ivec, dst, 0, all_mask)
                return None

            specialized = kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
                src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            )
            specialized.mlir_text()

        self.assertIn("vector register value", str(ctx.exception))

    def test_astype_requires_vector_register(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            @pto.vkernel(
                op="astype_non_vreg_input_unique",
                dtypes=[(pto.f32, pto.f32)],
                advanced=True,
            )
            def kernel(dst: pto.Tile, src: pto.Tile):
                # Try to call astype on a non-vector value
                scalar = pto.f32(1.0)
                ivec = scalar.astype(pto.i32)
                all_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                pto.vsts(ivec, dst, 0, all_mask)
                return None

            specialized = kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
                src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            )
            specialized.mlir_text()

        self.assertIn("vector register or mask value", str(ctx.exception))

    def test_vbitcast_supports_element_size_change(self) -> None:
        @pto.vkernel(
            op="vbitcast_element_size_change_unique",
            dtypes=[(pto.f32, pto.f32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            f32_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            f16_mask = pto.make_mask(pto.f16, pto.PAT.ALL)
            # Load f32 vector (64 elements)
            f32_vec = pto.vlds(src, 0)  # !pto.vreg<64xf32>
            # Convert to f16 (128 elements)
            f16_vec = pto.vbitcast(f32_vec, pto.f16)  # !pto.vreg<128xf16>
            # Convert back to f32
            f32_vec2 = pto.vbitcast(f16_vec, pto.f32)  # !pto.vreg<64xf32>
            # Store result
            pto.vsts(f32_vec2, dst, 0, f32_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.vbitcast", text)
        self.assertRegex(text, r"= pto\.vbitcast %[^:]+ : !pto\.vreg<64xf32> -> !pto\.vreg<128xf16>")
        self.assertRegex(text, r"= pto\.vbitcast %[^:]+ : !pto\.vreg<128xf16> -> !pto\.vreg<64xf32>")

    def test_pbitcast_supports_direct_interface(self) -> None:
        @pto.vkernel(
            op="pbitcast_direct_unique",
            dtypes=[(pto.f32, pto.f32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            src_mask = pto.make_mask(pto.f16, pto.PAT.ALL)
            dst_mask = pto.pbitcast(src_mask, pto.mask_b32)
            vec = pto.vlds(src, 0)
            pto.vsts(vec, dst, 0, dst_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.pbitcast", text)
        self.assertRegex(text, r'%src_mask_\d+ = pto\.pset_b16 "PAT_ALL" : !pto\.mask<b16>')
        self.assertRegex(text, r'= pto\.pbitcast %[^:]+ : !pto\.mask<b16> -> !pto\.mask<b32>')

    def test_pbitcast_rejects_non_mask_input(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            @pto.vkernel(
                op="pbitcast_non_mask_input_unique",
                dtypes=[(pto.f32, pto.f32)],
                advanced=True,
            )
            def kernel(dst: pto.Tile, src: pto.Tile):
                vec = pto.vlds(src, 0)
                mask = pto.pbitcast(vec, pto.mask_b32)
                pto.vsts(vec, dst, 0, mask)
                return None

            specialized = kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
                src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            )
            specialized.mlir_text()

        self.assertIn("mask value", str(ctx.exception))

    def test_mask_astype_lowers_to_pbitcast(self) -> None:
        @pto.vkernel(
            op="mask_astype_unique",
            dtypes=[(pto.f32, pto.f32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            src_mask = pto.make_mask(pto.f16, pto.PAT.ALL)
            dst_mask = src_mask.astype(pto.mask_b32)
            vec = pto.vlds(src, 0)
            pto.vsts(vec, dst, 0, dst_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.pbitcast", text)
        self.assertRegex(text, r'%src_mask_\d+ = pto\.pset_b16 "PAT_ALL" : !pto\.mask<b16>')
        self.assertRegex(text, r'= pto\.pbitcast %[^:]+ : !pto\.mask<b16> -> !pto\.mask<b32>')

    def test_astype_rejects_non_vreg_or_mask_receiver(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            @pto.vkernel(
                op="astype_invalid_receiver_unique",
                dtypes=[(pto.f32, pto.f32)],
                advanced=True,
            )
            def kernel(dst: pto.Tile, src: pto.Tile):
                scalar = pto.f32(1.0)
                mask = scalar.astype(pto.mask_b32)
                vec = pto.vlds(src, 0)
                pto.vsts(vec, dst, 0, mask)
                return None

            specialized = kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
                src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            )
            specialized.mlir_text()

        self.assertIn("vector register or mask value", str(ctx.exception))

    def test_index_to_float_scalar_cast_lowers_via_integer_bridge(self) -> None:
        @pto.vkernel(
            op="index_to_float_scalar_cast_unique",
            dtypes=[(pto.f32, pto.f32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            mask, _ = pto.make_mask(pto.f32, 1)
            vec = pto.vlds(src, 0)
            for col in range(0, 1, 1):
                scalar = pto.f32(col)
                out = pto.vadds(vec, scalar, mask)
                pto.vsts(out, dst, 0, mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("arith.index_castui", text)
        self.assertRegex(text, r"arith\.uitofp %\w+ : i64 to f32")
        self.assertNotRegex(text, r"arith\.uitofp %\w+ : index to f32")

    def test_extended_integer_vector_ops_surface_lowers(self) -> None:
        @pto.vkernel(
            op="extended_integer_vector_ops_unique",
            dtypes=[(pto.i32, pto.i32, pto.i16)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile, shift: pto.i16):
            all_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
            vec0 = pto.vlds(src, 0)
            vec1 = pto.vlds(src, 64)

            out = pto.vbcnt(vec0, all_mask)
            out = pto.vneg(out, all_mask)
            out = pto.vcls(out, all_mask)
            pto.vsunpack(vec0, 0)
            pto.vzunpack(vec0.astype(pto.ui32), 0)
            pto.vusqz(vec0.astype(pto.ui32), pto.make_mask(pto.ui32, pto.PAT.ALL))
            pto.vsqz(vec0, all_mask)
            out = pto.vshl(out, vec1, all_mask)
            out = pto.vshr(out, vec1, all_mask)
            out = pto.vshls(out, shift, all_mask)
            out = pto.vshrs(out, shift, all_mask)
            pto.vsts(out, dst, 0, all_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.vbcnt", text)
        self.assertIn("pto.vneg", text)
        self.assertIn("pto.vcls", text)
        self.assertIn("pto.vsunpack", text)
        self.assertIn("pto.vzunpack", text)
        self.assertIn("pto.vusqz", text)
        self.assertIn("pto.vsqz", text)
        self.assertIn("pto.vshl", text)
        self.assertIn("pto.vshr", text)
        self.assertIn("pto.vshls", text)
        self.assertIn("pto.vshrs", text)

    def test_fused_vector_ops_surface_lowers(self) -> None:
        @pto.vkernel(
            op="fused_vector_ops_unique",
            dtypes=[(pto.f32, pto.f32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            all_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            vec0 = pto.vlds(src, 0)
            vec1 = pto.vlds(src, 64)
            vec2 = pto.vlds(src, 128)
            vec3 = pto.vlds(src, 192)

            out = pto.vaddrelu(vec0, vec1, all_mask)
            out = pto.vaddreluconv(out, vec2, all_mask)
            out = pto.vsubrelu(out, vec3, all_mask)
            out = pto.vmulconv(out, vec1, all_mask)
            out = pto.vaxpy(vec1, out, vec2, all_mask)
            out = pto.vmula(vec1, vec2, out, all_mask)
            pto.vsts(out, dst, 0, all_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 256), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 256), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.vaddrelu", text)
        self.assertIn("pto.vaddreluconv", text)
        self.assertIn("pto.vsubrelu", text)
        self.assertIn("pto.vmulconv", text)
        self.assertIn("pto.vaxpy", text)
        self.assertIn("pto.vmula", text)

    def test_vmull_and_vector_scalar_bitwise_surface_lowers(self) -> None:
        @pto.vkernel(
            op="vmull_and_scalar_bitwise_unique",
            dtypes=[(pto.i32, pto.i32, pto.i32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile, scalar: pto.i32):
            all_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
            vec0 = pto.vlds(src, 0)
            vec1 = pto.vlds(src, 64)

            low, high = pto.vmull(vec0, vec1, all_mask)
            out = pto.vadd(low, high, all_mask)
            out = pto.vands(out, scalar, all_mask)
            out = pto.vors(out, scalar, all_mask)
            out = pto.vxors(out, scalar, all_mask)
            pto.vsts(out, dst, 0, all_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.vmull", text)
        self.assertIn("pto.vands", text)
        self.assertIn("pto.vors", text)
        self.assertIn("pto.vxors", text)

    def test_vci_typed_integer_inputs_lower_without_typed_arith(self) -> None:
        @pto.vkernel(
            op="vci_typed_integer_inputs_unique",
            dtypes=[(pto.ui16, pto.si16, pto.i32)],
            advanced=True,
        )
        def kernel(dst_u: pto.Tile, dst_s: pto.Tile, seed: pto.i32):
            unsigned_mask = pto.make_mask(pto.ui16, pto.PAT.ALL)
            signed_mask = pto.make_mask(pto.si16, pto.PAT.ALL)

            unsigned_idx = pto.vci(pto.ui16(0))
            signed_idx = pto.vci(pto.si16(seed), pto.OrderMode.ASC)

            pto.vsts(unsigned_idx, dst_u, 0, unsigned_mask)
            pto.vsts(signed_idx, dst_s, 0, signed_mask)
            return None

        specialized = kernel.specialize(
            dst_u=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            dst_s=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.vci", text)
        self.assertIn(": i16 to ui16", text)
        self.assertIn(": i16 to si16", text)
        self.assertNotIn("arith.constant 0 : ui16", text)
        self.assertNotRegex(text, r"arith\.(extsi|extui|trunci|bitcast) %\w+ : .* to (ui16|si16)")

    def test_vector_scalar_bitwise_typed_scalar_inputs_lower_without_typed_arith(self) -> None:
        @pto.vkernel(
            op="vector_scalar_bitwise_typed_scalar_inputs_unique",
            dtypes=[(pto.ui16, pto.ui16, pto.i32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile, seed: pto.i32):
            mask = pto.make_mask(pto.ui16, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            scalar = pto.ui16(seed)
            out = pto.vands(vec, scalar, mask)
            out = pto.vors(out, scalar, mask)
            out = pto.vxors(out, scalar, mask)
            pto.vsts(out, dst, 0, mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.vands", text)
        self.assertIn("pto.vors", text)
        self.assertIn("pto.vxors", text)
        self.assertIn("arith.trunci", text)
        self.assertIn(": i32 to i16", text)
        self.assertIn(": i16 to ui16", text)
        self.assertNotRegex(text, r"arith\.trunci %\w+ : i32 to ui16")

    def test_broadcast_and_index_vector_ops_surface_lowers(self) -> None:
        @pto.vkernel(
            op="broadcast_and_index_vector_ops_unique",
            dtypes=[(pto.i32, pto.i32, pto.i32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile, seed: pto.i32):
            all_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
            vec0 = pto.vlds(src, 0)

            broadcast = pto.vbr(seed)
            dup_from_vec = pto.vdup(vec0, all_mask, pto.PositionMode.HIGHEST)
            dup_from_scalar = pto.vdup(seed, all_mask)
            idx0 = pto.vci(seed)
            idx1 = pto.vci(seed, pto.OrderMode.ASC)

            out = pto.vadd(broadcast, dup_from_vec, all_mask)
            out = pto.vadd(out, dup_from_scalar, all_mask)
            out = pto.vadd(out, idx0, all_mask)
            out = pto.vadd(out, idx1, all_mask)
            pto.vsts(out, dst, 0, all_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.vbr", text)
        self.assertIn("pto.vdup", text)
        self.assertIn("pto.vci", text)
        self.assertRegex(
            text,
            r'pto\.vdup\s+%[^\s]+,\s+%[^\s]+\s+\{position = "HIGHEST"\}\s+:',
        )
        self.assertRegex(
            text,
            r'pto\.vdup\s+%[^\s]+,\s+%[^\s]+\s+:',
        )
        self.assertNotIn('position = "LOWEST"', text)
        self.assertNotIn('position = "POS_LOWEST"', text)
        self.assertRegex(
            text,
            r'pto\.vci\s+%[^\s]+\s+\{order = "ASC"\}\s+:',
        )
        self.assertNotRegex(
            text,
            r'pto\.vci\s+%[^\s]+,\s*"ASC"\s+:',
        )

    def test_vci_desc_lowers_to_desc_order_attr(self) -> None:
        @pto.vkernel(
            op="vci_desc_order_unique",
            dtypes=[(pto.i32, pto.i32, pto.i32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile, seed: pto.i32):
            mask = pto.make_mask(pto.i32, pto.PAT.ALL)
            indices = pto.vci(seed, pto.OrderMode.DESC)
            vec = pto.vlds(src, 0)
            out = pto.vadd(vec, indices, mask)
            pto.vsts(out, dst, 0, mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertRegex(
            text,
            r'pto\.vci\s+%[^\s]+\s+\{order = "DESC"\}\s+:',
        )

    def test_vdup_scalar_input_rejects_position_argument(self) -> None:
        with self.assertRaises(TypeError) as ctx:

            @pto.vkernel(
                op="vdup_scalar_reject_position_unique",
                dtypes=[(pto.i32, pto.i32)],
                advanced=True,
            )
            def kernel(dst: pto.Tile, seed: pto.i32):
                all_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
                out = pto.vdup(seed, all_mask, pto.PositionMode.HIGHEST)
                pto.vsts(out, dst, 0, all_mask)
                return None

            specialized = kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            )
            specialized.mlir_text()

        self.assertIn("pto.vdup scalar input does not accept `position`", str(ctx.exception))

    def test_vbr_and_vdup_accept_narrow_typed_scalar_constructors_with_explicit_bridges(self) -> None:
        @pto.vkernel(
            op="narrow_typed_vbr_vdup_scalar_constructors_unique",
            dtypes=[(pto.si16, pto.ui16)],
            advanced=True,
        )
        def kernel(dst_s: pto.Tile, dst_u: pto.Tile):
            signed_mask = pto.make_mask(pto.si16, pto.PAT.ALL)
            unsigned_mask = pto.make_mask(pto.ui16, pto.PAT.ALL)

            signed = pto.vadd(
                pto.vbr(pto.si16(0)),
                pto.vdup(pto.si16(0), signed_mask),
                signed_mask,
            )
            unsigned = pto.vadd(
                pto.vbr(pto.ui16(0)),
                pto.vdup(pto.ui16(0), unsigned_mask),
                unsigned_mask,
            )

            pto.vsts(signed, dst_s, 0, signed_mask)
            pto.vsts(unsigned, dst_u, 0, unsigned_mask)
            return None

        specialized = kernel.specialize(
            dst_s=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            dst_u=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.vbr", text)
        self.assertIn("pto.vdup", text)
        self.assertIn("arith.constant 0 : i16", text)
        self.assertIn(": i16 to si16", text)
        self.assertIn(": i16 to ui16", text)
        self.assertNotIn("arith.constant 0 : si16", text)
        self.assertNotIn("arith.constant 0 : ui16", text)

    def test_signed_and_unsigned_integer_dtypes_lower_distinctly(self) -> None:
        @pto.vkernel(
            op="signed_unsigned_integer_types_unique",
            dtypes=[(pto.si16, pto.si16, pto.ui16, pto.ui16)],
            advanced=True,
        )
        def kernel(dst_s: pto.Tile, src_s: pto.Tile, dst_u: pto.Tile, src_u: pto.Tile):
            signed_mask = pto.make_mask(pto.si16, pto.PAT.ALL)
            unsigned_mask = pto.make_mask(pto.ui16, pto.PAT.ALL)
            signed_vec = pto.vlds(src_s, 0)
            unsigned_vec = pto.vlds(src_u, 0)
            signed_out = pto.vadds(signed_vec, pto.si16(-1), signed_mask)
            unsigned_out = pto.vadds(unsigned_vec, pto.ui16(1), unsigned_mask)
            pto.vsts(signed_out, dst_s, 0, signed_mask)
            pto.vsts(unsigned_out, dst_u, 0, unsigned_mask)
            return None

        specialized = kernel.specialize(
            dst_s=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            src_s=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            dst_u=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            src_u=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("dtype=si16", text)
        self.assertIn("dtype=ui16", text)
        self.assertIn("!pto.vreg<128xsi16>", text)
        self.assertIn("!pto.vreg<128xui16>", text)

    def test_vcmps_literal_scalar_uses_signless_integer_bridge(self) -> None:
        @pto.vkernel(
            op="vcmps_literal_scalar_bridge_unique",
            dtypes=[(pto.si16, pto.si16)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            all_mask = pto.make_mask(pto.si16, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            cmp_mask = pto.vcmps(vec, pto.si16(-1), all_mask, pto.CmpMode.GT)
            selected = pto.vsel(vec, vec, cmp_mask)
            pto.vsts(selected, dst, 0, all_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.vcmps", text)
        self.assertIn("arith.constant -1 : i16", text)
        self.assertIn(": i16 to si16", text)
        self.assertNotIn("arith.constant -1 : si16", text)

    def test_vadds_index_constructor_scalar_uses_signless_integer_bridge(self) -> None:
        @pto.vkernel(
            op="vadds_index_constructor_bridge_unique",
            dtypes=[(pto.ui16, pto.ui16)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            cols = dst.valid_shape[1]
            vec = pto.vlds(src, 0)
            mask, _ = pto.make_mask(pto.ui16, 1)
            for col in range(0, cols, 1):
                scalar = pto.ui16(col)
                out = pto.vadds(vec, scalar, mask)
                pto.vsts(out, dst, 0, mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(
                shape=(8, 128),
                valid_shape=(8, 1),
                memory_space=pto.MemorySpace.UB,
            ),
            src=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.vadds", text)
        self.assertIn("arith.index_castui", text)
        self.assertIn(": index to i32", text)
        self.assertIn("arith.trunci", text)
        self.assertIn(": i32 to i16", text)
        self.assertIn(": i16 to ui16", text)
        self.assertNotIn(": index to ui16", text)

    def test_vshrs_cast_result_scalar_uses_signless_integer_bridge(self) -> None:
        @pto.vkernel(
            op="vshrs_cast_result_scalar_bridge_unique",
            dtypes=[(pto.i32, pto.i32, pto.ui16)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile, shift_seed: pto.ui16):
            all_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            shift = pto.i16(shift_seed)
            out = pto.vshrs(vec, shift, all_mask)
            pto.vsts(out, dst, 0, all_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.vshrs", text)
        self.assertIn(": ui16 to i16", text)
        self.assertNotRegex(text, r"arith\.bitcast %\w+ : ui16 to i16")
        self.assertNotRegex(text, r"arith\.trunci %\w+ : ui16 to i16")

    def test_vbr_accepts_float_literal_constant(self) -> None:
        @pto.vkernel(
            op="broadcast_float_literal_constant_unique",
            dtypes=[(pto.f32, pto.f32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            all_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            vec0 = pto.vlds(src, 0)
            bias = pto.vbr(0.0)
            out = pto.vadd(vec0, bias, all_mask)
            pto.vsts(out, dst, 0, all_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("= arith.constant 0.0 : f32", text)
        self.assertIn("pto.vbr", text)

    def test_kernel_accepts_module_level_literal_constant_reference(self) -> None:
        @pto.vkernel(
            op="module_level_literal_constant_reference_unique",
            dtypes=[(pto.f32, pto.f32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            all_mask, _ = pto.make_mask(pto.f32, GLOBAL_TILELANG_LITERAL_BLOCK_SIZE)
            vec = pto.vlds(src, 0)
            out = pto.vadds(vec, pto.f32(0.0), all_mask)
            pto.vsts(out, dst, 0, all_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertRegex(text, r"arith\.constant 32 : (index|i64)")
        self.assertIn("pto.plt_b32", text)

    def test_scalar_constructor_call_surfaces_lower(self) -> None:
        @pto.vkernel(
            op="scalar_constructor_call_surfaces_unique",
            dtypes=[(pto.i32, pto.i32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            mask = pto.make_mask(pto.i32, pto.PAT.ALL)
            base = pto.i32(1)
            idx = pto.i16(base)
            idx = pto.i8(idx)
            idx = pto.i64(idx)
            flt = pto.f16(idx)
            flt = pto.bf16(flt)
            flt = pto.f32(flt)
            gate = pto.i1(flt)
            scalar = pto.i32(gate)
            vec = pto.vlds(src, 0)
            out = pto.vadds(vec, scalar, mask)
            pto.vsts(out, dst, 0, mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("arith.trunci", text)
        self.assertIn("arith.extsi", text)
        self.assertIn("arith.sitofp", text)
        self.assertIn("arith.fptosi", text)
        self.assertIn("arith.extf", text)
        self.assertIn("arith.truncf", text)

    def test_typed_integer_scalar_coercion_uses_signless_integer_carriers(self) -> None:
        @pto.vkernel(
            op="typed_integer_scalar_coercion_unique",
            dtypes=[(pto.si16, pto.si16, pto.ui16, pto.ui16, pto.i32, pto.i32, pto.i32)],
            advanced=True,
        )
        def kernel(
            dst_s: pto.Tile,
            src_s: pto.Tile,
            dst_u: pto.Tile,
            src_u: pto.Tile,
            dst_i: pto.Tile,
            src_i: pto.Tile,
            seed: pto.i32,
        ):
            signed_mask = pto.make_mask(pto.si16, pto.PAT.ALL)
            unsigned_mask = pto.make_mask(pto.ui16, pto.PAT.ALL)
            scalar_mask = pto.make_mask(pto.i32, pto.PAT.ALL)

            signed_scalar = pto.si16(seed)
            unsigned_scalar = pto.ui16(seed)

            signed_vec = pto.vlds(src_s, 0)
            unsigned_vec = pto.vlds(src_u, 0)
            scalar_vec = pto.vlds(src_i, 0)

            signed_out = pto.vadds(signed_vec, signed_scalar, signed_mask)
            unsigned_out = pto.vadds(unsigned_vec, unsigned_scalar, unsigned_mask)
            scalar_out = pto.vadds(scalar_vec, pto.i32(signed_scalar), scalar_mask)
            scalar_out = pto.vadds(scalar_out, pto.i32(unsigned_scalar), scalar_mask)

            pto.vsts(signed_out, dst_s, 0, signed_mask)
            pto.vsts(unsigned_out, dst_u, 0, unsigned_mask)
            pto.vsts(scalar_out, dst_i, 0, scalar_mask)
            return None

        specialized = kernel.specialize(
            dst_s=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            src_s=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            dst_u=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            src_u=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            dst_i=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src_i=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("arith.trunci", text)
        self.assertIn(": i32 to i16", text)
        self.assertIn(": i16 to si16", text)
        self.assertIn(": i16 to ui16", text)
        self.assertIn(": si16 to i16", text)
        self.assertIn(": ui16 to i16", text)
        self.assertIn("arith.extsi", text)
        self.assertIn("arith.extui", text)
        self.assertNotRegex(text, r"arith\.trunci %\w+ : i32 to (si16|ui16)")
        self.assertNotRegex(text, r"arith\.extsi %\w+ : si16 to i32")
        self.assertNotRegex(text, r"arith\.extui %\w+ : ui16 to i32")

    def test_typed_integer_float_scalar_coercion_uses_signless_integer_carriers(self) -> None:
        @pto.vkernel(
            op="typed_integer_float_scalar_coercion_unique",
            dtypes=[(pto.ui16, pto.ui16)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            mask = pto.make_mask(pto.ui16, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            scalar = pto.ui16(1)
            flt = pto.f32(scalar)
            back = pto.ui16(flt)
            out = pto.vadds(vec, back, mask)
            pto.vsts(out, dst, 0, mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn(": ui16 to i16", text)
        self.assertIn("arith.uitofp", text)
        self.assertIn(": i16 to f32", text)
        self.assertIn("arith.fptoui", text)
        self.assertIn(": f32 to i16", text)
        self.assertIn(": i16 to ui16", text)
        self.assertNotRegex(text, r"arith\.uitofp %\w+ : ui16 to f32")
        self.assertNotRegex(text, r"arith\.fptoui %\w+ : f32 to ui16")

    def test_scalar_constructor_accepts_signed_float_literals(self) -> None:
        @pto.vkernel(op="scalar_constructor_signed_float_literals_unique", dtypes=[(pto.f32,)])
        def kernel(inp: pto.TensorView):
            a = pto.f16(-1.5)
            b = pto.bf16(+2.5)
            c = pto.f32(-3.5)
            return None

        text = kernel.mlir_text()
        self.assertIn("= arith.constant -1.5 : f16", text)
        self.assertIn("= arith.constant 2.5 : bf16", text)
        self.assertIn("= arith.constant -3.5 : f32", text)

    def test_scalar_constructor_accepts_special_float_string_literals(self) -> None:
        @pto.vkernel(op="scalar_constructor_special_float_literals_unique", dtypes=[(pto.f32,)])
        def kernel(inp: pto.TensorView):
            a = pto.f16("-inf")
            b = pto.bf16("inf")
            c = pto.f32("nan")
            d = pto.f16("0xFC00")
            e = pto.bf16("0xFF80")
            f = pto.f32("0xFF800000")
            return None

        text = kernel.mlir_text()
        self.assertIn("= arith.constant 0xFC00 : f16", text)
        self.assertIn("= arith.constant 0x7F80 : bf16", text)
        self.assertIn("= arith.constant 0x7FC00000 : f32", text)
        self.assertIn("= arith.constant 0xFF80 : bf16", text)
        self.assertIn("= arith.constant 0xFF800000 : f32", text)

    def test_scalar_constructor_emits_negative_zero_as_stable_bit_pattern(self) -> None:
        @pto.vkernel(op="scalar_constructor_negative_zero_unique", dtypes=[(pto.f32,)])
        def kernel(inp: pto.TensorView):
            a = pto.f16(-0.0)
            b = pto.bf16(-0.0)
            c = pto.f32(-0.0)
            return None

        text = kernel.mlir_text()
        self.assertIn("= arith.constant 0x8000 : f16", text)
        self.assertIn("= arith.constant 0x8000 : bf16", text)
        self.assertIn("= arith.constant 0x80000000 : f32", text)

    def test_scalar_constructor_rejects_bad_arity(self) -> None:
        @pto.vkernel(op="scalar_constructor_bad_arity_no_arg_unique", dtypes=[(pto.f32,)])
        def kernel_no_arg(inp: pto.TensorView):
            x = pto.i32()
            return None

        with self.assertRaises(TypeError) as no_arg_ctx:
            kernel_no_arg.mlir_text()

        self.assertIn("pto.i32 expects exactly 1 positional argument", str(no_arg_ctx.exception))

        @pto.vkernel(op="scalar_constructor_bad_arity_two_arg_unique", dtypes=[(pto.f32,)])
        def kernel_two_arg(inp: pto.TensorView):
            x = pto.f32(1.0, 2.0)
            return None

        with self.assertRaises(TypeError) as two_arg_ctx:
            kernel_two_arg.mlir_text()

        self.assertIn("pto.f32 expects exactly 1 positional argument", str(two_arg_ctx.exception))

    def test_scalar_constructor_rejects_non_scalar_operand(self) -> None:
        @pto.vkernel(op="scalar_constructor_bad_operand_unique", dtypes=[(pto.f32,)])
        def kernel(inp: pto.TensorView):
            x = pto.i32(inp)
            return None

        with self.assertRaises(TypeError) as ctx:
            kernel.mlir_text()

        self.assertIn("pto.i32 value must be a scalar or index value", str(ctx.exception))

    def test_scalar_constructor_accepts_integer_hex_bit_pattern_strings(self) -> None:
        @pto.vkernel(op="scalar_constructor_integer_hex_bit_patterns_unique", dtypes=[(pto.f32,)])
        def kernel(inp: pto.TensorView):
            x = pto.i16("0x7FFF")
            y = pto.i32("0x7FFFFFFF")
            z = pto.i16("0x8000")
            a = pto.i32("0x80000000")
            b = pto.ui16("0x8000")
            return None

        text = kernel.mlir_text()
        self.assertIn("= arith.constant 32767 : i16", text)
        self.assertIn("= arith.constant 2147483647 : i32", text)
        self.assertIn("= arith.constant -32768 : i16", text)
        self.assertIn("= arith.constant -2147483648 : i32", text)
        self.assertIn("= arith.constant 32768 : i16", text)

    def test_scalar_constructor_rejects_non_hex_integer_string_literals(self) -> None:
        @pto.vkernel(op="scalar_constructor_non_hex_integer_strings_unique", dtypes=[(pto.f32,)])
        def kernel(inp: pto.TensorView):
            x = pto.i32("1024")
            return None

        with self.assertRaises(TypeError) as ctx:
            kernel.mlir_text()

        self.assertIn("string literals must use hex bit-pattern form", str(ctx.exception))

    def test_scalar_constructor_rejects_out_of_range_integer_literal(self) -> None:
        @pto.vkernel(op="scalar_constructor_oob_int_unique", dtypes=[(pto.f32,)])
        def kernel(inp: pto.TensorView):
            x = pto.i8(1024)
            return None

        with self.assertRaises(TypeError) as ctx:
            kernel.mlir_text()

        self.assertIn("out of range for i8", str(ctx.exception))

    def test_scalar_constructor_rejects_out_of_range_integer_string_literal(self) -> None:
        @pto.vkernel(op="scalar_constructor_oob_integer_string_unique", dtypes=[(pto.f32,)])
        def kernel(inp: pto.TensorView):
            x = pto.i16("0x10000")
            return None

        with self.assertRaises(TypeError) as ctx:
            kernel.mlir_text()

        self.assertIn("exceeds 16-bit width for i16", str(ctx.exception))

    def test_inferred_vecscope_propagates_bindings_to_constexpr_if(self) -> None:
        @pto.vkernel(
            op="inferred_vecscope_binding_propagation_unique",
            dtypes=[(pto.f32, pto.f32)],
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            acc = pto.vbr(0.0)
            vec = pto.vlds(src, 0)
            acc = pto.vadd(acc, vec, mask)
            if pto.constexpr(True):
                pto.vsts(acc, dst, 0, mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.vadd", text)
        self.assertIn("pto.vsts", text)
        self.assertIn("= arith.constant 0.0 : f32", text)

    def test_loop_lowering_supports_multiple_loop_carried_bindings(self) -> None:
        @pto.vkernel(
            op="loop_multi_carried_bindings_unique",
            dtypes=[(pto.f32, pto.f32)],
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            remained = 64
            acc = pto.vbr(0.0)
            all_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            for col in range(0, 64, 64):
                mask, remained = pto.make_mask(pto.f32, remained)
                vec = pto.vlds(src, col)
                acc = pto.vadd(acc, vec, mask)
            pto.vsts(acc, dst, 0, all_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertRegex(text, r"%remained_\d+, %acc_\d+ = scf\.for")
        self.assertRegex(text, r"iter_args\(%remained_iter_\d+_0 = [^,]+, %acc_iter_\d+_1 = [^)]+\)")
        self.assertRegex(text, r"scf\.yield %remained_\d+, %acc_\d+ : i32, !pto\.vreg<64xf32>")

    def test_reduction_and_rearrangement_vector_ops_surface_lowers(self) -> None:
        @pto.vkernel(
            op="reduction_and_rearrangement_vector_ops_unique",
            dtypes=[(pto.i32, pto.i32, pto.i32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile, shift: pto.i32):
            all_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
            vec0 = pto.vlds(src, 0)
            vec1 = pto.vlds(src, 64)
            packed_mask = pto.make_mask(pto.ui16, pto.PAT.ALL)

            out = pto.vcgadd(vec0, all_mask)
            out = pto.vcgmax(out, all_mask)
            out = pto.vcgmin(out, all_mask)
            out = pto.vcpadd(out, all_mask)
            packed0 = pto.vpack(vec0, pto.PredicatePart.LOWER)
            packed1 = pto.vpack(vec1, pto.PredicatePart.HIGHER)
            indices = pto.vci(pto.i16(shift), pto.OrderMode.ASC)
            packed0 = pto.vperm(packed0, indices, packed_mask)
            packed0 = pto.vshift(packed0, pto.i16(shift), packed_mask)
            packed0 = pto.vslide(packed0, pto.i16(shift), packed_mask)
            packed0 = pto.vmrgsort(packed0, packed1, packed_mask)
            out = pto.vsort32(out, all_mask)
            pto.vsts(out, dst, 0, all_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.vcgadd", text)
        self.assertIn("pto.vcgmax", text)
        self.assertIn("pto.vcgmin", text)
        self.assertIn("pto.vcpadd", text)
        self.assertIn("pto.vpack", text)
        self.assertIn("pto.vperm", text)
        self.assertIn("pto.vshift", text)
        self.assertIn("pto.vslide", text)
        self.assertIn("pto.vsort32", text)
        self.assertIn("pto.vmrgsort", text)

    def test_scalar_loop_prologue_does_not_force_vecscope_into_inner_loop(self) -> None:
        @pto.vkernel(op="tadd_outer_scope_unique", dtypes=[(pto.f32, pto.f32, pto.f32)])
        def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
            dtype = dst.element_type
            valid_rows, valid_cols = dst.valid_shape
            for row in range(0, valid_rows, 1):
                remained = valid_cols
                for col in range(0, valid_cols, pto.get_lanes(dtype)):
                    mask, remained = pto.make_mask(dtype, remained)
                    lhs = pto.vlds(src0[row, col:])
                    rhs = pto.vlds(src1[row, col:])
                    summed = pto.vadd(lhs, rhs, mask)
                    pto.vsts(summed, dst[row, col:], mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src0=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope_stmts = [stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt)]
        self.assertEqual(len(vecscope_stmts), 1)
        outer_loop = vecscope_stmts[0].body[0]
        self.assertIsInstance(outer_loop, SemanticForStmt)
        self.assertIsInstance(outer_loop.body[0], SemanticAssignStmt)
        self.assertIsInstance(outer_loop.body[1], SemanticForStmt)

        text = specialized.mlir_text()
        self.assertEqual(text.count("pto.vecscope {"), 1)
        self.assertRegex(text, r"pto\.vecscope \{\n\s+scf\.for %row_\d+ = %c0 to %valid_rows_\d+ step %c1")
        self.assertNotRegex(text, r"scf\.for %row_\d+ = [^\n]+\{\n\s+pto\.vecscope \{")

    def test_unused_tile_does_not_hoist_tile_buf_addr_or_valid_shape_intrinsics(self) -> None:
        @pto.vkernel(op="tile_usage_scan_unique", dtypes=[(pto.f32, pto.f32, pto.f32)], advanced=True)
        def kernel(dst: pto.Tile, src: pto.Tile, scratch: pto.Tile):
            rows, cols = dst.valid_shape
            mask = pto.make_mask(dst.element_type, pto.PAT.ALL)
            for row in range(0, rows, 1):
                for col in range(0, cols, pto.get_lanes(dst.element_type)):
                    value = pto.vlds(src[row, col:])
                    pto.vsts(value, dst[row, col:], mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            scratch=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.tile_buf_addr %arg0", text)
        self.assertIn("pto.tile_buf_addr %arg1", text)
        self.assertNotIn("pto.tile_buf_addr %arg2", text)
        self.assertIn("pto.tile_valid_rows %arg0", text)
        self.assertIn("pto.tile_valid_cols %arg0", text)
        self.assertNotIn("pto.tile_valid_rows %arg1", text)
        self.assertNotIn("pto.tile_valid_cols %arg1", text)
        self.assertNotIn("pto.tile_valid_rows %arg2", text)
        self.assertNotIn("pto.tile_valid_cols %arg2", text)

    def test_tile_dynamic_valid_shape_profile_lowers_to_runtime_bounds_in_advanced_mode(self) -> None:
        elem = pto.TypeVar("Elem")

        @pto.vkernel(op="tadd_dynamic_valid_shape_unique", dtypes=[(elem, elem, elem)], advanced=True)
        def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
            dtype = dst.element_type
            valid_rows, valid_cols = dst.valid_shape
            remained = valid_cols
            for row in range(0, valid_rows, 1):
                for col in range(0, valid_cols, pto.get_lanes(dtype)):
                    mask, remained = pto.make_mask(dtype, remained)
                    summed = pto.vadd(pto.vlds(src0[row, col:]), pto.vlds(src1[row, col:]), mask)
                    pto.vsts(summed, dst[row, col:], mask)
            return None

        selected = pto.select_kernel(
            "a5",
            "tadd_dynamic_valid_shape_unique",
            (pto.f16, pto.f16, pto.f16),
        )
        specialized = selected.specialize(
            dst=pto.TileSpecialization(
                shape=(8, 128),
                memory_space=pto.MemorySpace.UB,
                valid_shape=("valid_rows", "valid_cols"),
            ),
            src0=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertEqual(
            [(param.name, param.kind) for param in semantic_kernel.parameters],
            [
                ("dst", "tile"),
                ("src0", "tile"),
                ("src1", "tile"),
                ("__valid_shape_dst_0", "tile_valid_shape"),
                ("__valid_shape_dst_1", "tile_valid_shape"),
            ],
        )
        self.assertEqual(semantic_kernel.tile_bindings[0].valid_shape, (None, None))

        text = specialized.mlir_text()
        self.assertIn(
            "func.func @kernel(%arg0: !pto.tile_buf<loc=vec, dtype=f16, rows=8, cols=128, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>, %arg1: !pto.tile_buf<loc=vec, dtype=f16, rows=8, cols=128, v_row=8, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>, %arg2: !pto.tile_buf<loc=vec, dtype=f16, rows=8, cols=128, v_row=8, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>) attributes { pto.tilelang.instance } {",
            text,
        )
        self.assertIn("valid_shape=(?, ?)", text)
        self.assertIn("pto.vecscope {", text)
        self.assertIn("step %c128", text)
        self.assertIn("pto.tile_valid_rows %arg0", text)
        self.assertIn("pto.tile_valid_cols %arg0", text)
        self.assertNotIn("pto.tile_valid_rows %arg1", text)
        self.assertNotIn("pto.tile_valid_cols %arg1", text)
        self.assertNotIn("pto.tile_valid_rows %arg2", text)
        self.assertNotIn("pto.tile_valid_cols %arg2", text)
        self.assertLess(text.index("pto.tile_valid_rows %arg0"), text.index("pto.vecscope {"))
        self.assertLess(text.index("pto.tile_valid_cols %arg0"), text.index("pto.vecscope {"))
        self.assertRegex(text, r"scf\.for %row_\d+ = %c0 to %valid_rows_\d+ step %c1")
        self.assertRegex(text, r"scf\.for %col_\d+ = %c0 to %valid_cols_\d+ step %c128")
        self.assertRegex(text, r"%tmp_\d+ = arith\.index_cast %valid_cols_\d+ : index to i32")
        self.assertRegex(
            text,
            r"pto\.tile_buf_addr %arg1 : !pto\.tile_buf<loc=vec, dtype=f16, rows=8, cols=128, v_row=8, v_col=128",
        )
        self.assertRegex(
            text,
            r"memref\.subview %tmp_\d+\[%row_\d+, %col_\d+\] \[%c1, %tmp_\d+\] \[%c1, %c1\] : memref<8x128xf16, #pto\.address_space<vec>> to memref<\?x\?xf16, strided<\[\?, \?\], offset: \?>, #pto\.address_space<vec>>",
        )
        self.assertRegex(
            text,
            r"pto\.vlds %tmp_\d+\[%c0\] : memref<\?x\?xf16, strided<\[\?, \?\], offset: \?>, #pto\.address_space<vec>> -> !pto\.vreg<128xf16>",
        )
        self.assertRegex(
            text,
            r"pto\.vsts %summed_\d+, %tmp_\d+\[%c0\], %mask_\d+ : !pto\.vreg<128xf16>, memref<\?x\?xf16, strided<\[\?, \?\], offset: \?>, #pto\.address_space<vec>>, !pto\.mask<b16>",
        )

    def test_tile_valid_shape_subscript_profile_lowers_to_runtime_bounds_in_advanced_mode(self) -> None:
        @pto.vkernel(op="tile_valid_shape_subscript_unique", dtypes=[(pto.f16,)], advanced=True)
        def kernel(dst: pto.Tile):
            valid_rows = dst.valid_shape[0]
            valid_cols = dst.valid_shape[1]
            area = valid_rows * valid_cols
            if area == 0:
                area = 1
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(
                shape=(8, 128),
                memory_space=pto.MemorySpace.UB,
                valid_shape=("valid_rows", "valid_cols"),
            ),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertEqual(
            [(param.name, param.kind) for param in semantic_kernel.parameters],
            [
                ("dst", "tile"),
                ("__valid_shape_dst_0", "tile_valid_shape"),
                ("__valid_shape_dst_1", "tile_valid_shape"),
            ],
        )
        valid_rows_assign = semantic_kernel.body[0]
        valid_cols_assign = semantic_kernel.body[1]
        self.assertIsInstance(valid_rows_assign, SemanticAssignStmt)
        self.assertIsInstance(valid_cols_assign, SemanticAssignStmt)
        self.assertIsInstance(valid_rows_assign.targets[0].type, SemanticIndexType)
        self.assertIsInstance(valid_cols_assign.targets[0].type, SemanticIndexType)

        text = specialized.mlir_text()
        self.assertIn("valid_shape=(?, ?)", text)
        self.assertRegex(text, r"%valid_rows_\d+ = pto\.tile_valid_rows %arg0")
        self.assertRegex(text, r"%valid_cols_\d+ = pto\.tile_valid_cols %arg0")

    def test_tile_partial_dynamic_valid_shape_profile_tracks_dynamic_axes_only(self) -> None:
        elem = pto.TypeVar("Elem")

        @pto.vkernel(op="tadd_partial_dynamic_valid_shape_unique", dtypes=[(elem, elem, elem)], advanced=True)
        def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
            dtype = dst.element_type
            valid_rows, valid_cols = dst.valid_shape
            remained = valid_cols
            for row in range(0, valid_rows, 1):
                for col in range(0, valid_cols, pto.get_lanes(dtype)):
                    mask, remained = pto.make_mask(dtype, remained)
                    summed = pto.vadd(pto.vlds(src0[row, col:]), pto.vlds(src1[row, col:]), mask)
                    pto.vsts(summed, dst[row, col:], mask)
            return None

        selected = pto.select_kernel(
            "a5",
            "tadd_partial_dynamic_valid_shape_unique",
            (pto.f16, pto.f16, pto.f16),
        )

        rows_dynamic = selected.specialize(
            dst=pto.TileSpecialization(
                shape=(8, 128),
                memory_space=pto.MemorySpace.UB,
                valid_shape=("valid_rows", 128),
            ),
            src0=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )
        rows_dynamic_semantic = analyze_frontend_kernel(build_frontend_kernel_node(rows_dynamic))
        self.assertEqual(
            [(param.name, param.kind) for param in rows_dynamic_semantic.parameters],
            [
                ("dst", "tile"),
                ("src0", "tile"),
                ("src1", "tile"),
                ("__valid_shape_dst_0", "tile_valid_shape"),
            ],
        )
        rows_dynamic_text = rows_dynamic.mlir_text()
        self.assertIn("valid_shape=(?, 128)", rows_dynamic_text)
        self.assertIn("pto.tile_valid_rows %arg0", rows_dynamic_text)
        self.assertIn("pto.tile_valid_cols %arg0", rows_dynamic_text)
        self.assertRegex(rows_dynamic_text, r"scf\.for %row_\d+ = %c0 to %valid_rows_\d+ step %c1")
        self.assertRegex(rows_dynamic_text, r"scf\.for %col_\d+ = %c0 to %valid_cols_\d+ step %c128")

        cols_dynamic = selected.specialize(
            dst=pto.TileSpecialization(
                shape=(8, 128),
                memory_space=pto.MemorySpace.UB,
                valid_shape=(8, "valid_cols"),
            ),
            src0=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )
        cols_dynamic_semantic = analyze_frontend_kernel(build_frontend_kernel_node(cols_dynamic))
        self.assertEqual(
            [(param.name, param.kind) for param in cols_dynamic_semantic.parameters],
            [
                ("dst", "tile"),
                ("src0", "tile"),
                ("src1", "tile"),
                ("__valid_shape_dst_1", "tile_valid_shape"),
            ],
        )
        cols_dynamic_text = cols_dynamic.mlir_text()
        self.assertIn("valid_shape=(8, ?)", cols_dynamic_text)
        self.assertIn("pto.tile_valid_rows %arg0", cols_dynamic_text)
        self.assertIn("pto.tile_valid_cols %arg0", cols_dynamic_text)
        self.assertRegex(cols_dynamic_text, r"scf\.for %row_\d+ = %c0 to %valid_rows_\d+ step %c1")
        self.assertRegex(cols_dynamic_text, r"scf\.for %col_\d+ = %c0 to %valid_cols_\d+ step %c128")

    def test_advanced_mode_scalar_assignments_stay_inside_inferred_vecscope_runs(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32)], advanced=True)
        def kernel(src: pto.Tile, dst: pto.Tile):
            dtype = src.element_type
            first_mask = pto.make_mask(dtype, pto.PAT.ALL)
            first = pto.vlds(src[0, 0:])
            pto.vsts(first, dst[0, 0:], first_mask)
            boundary = 1
            second_mask = pto.make_mask(dtype, pto.PAT.ALL)
            second = pto.vlds(src[1, 0:])
            pto.vsts(second, dst[1, 0:], second_mask)
            return None

        specialized = kernel.specialize(
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope_stmts = [stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt)]
        self.assertEqual(len(vecscope_stmts), 1)

        text = specialized.mlir_text()
        self.assertEqual(text.count("pto.vecscope {"), 1)
        boundary_index = text.index("%boundary_")
        first_vsts = text.index("pto.vsts")
        second_vsts = text.rindex("pto.vsts")
        self.assertLess(text.index("pto.vecscope {"), boundary_index)
        self.assertLess(first_vsts, boundary_index)
        self.assertLess(boundary_index, second_vsts)
        self.assertLess(boundary_index, text.index("return"))

    def test_explicit_vecscope_is_supported_in_stable_mode(self) -> None:
        @pto.vkernel(op="explicit_vecscope_stable_unique", dtypes=[(pto.f32, pto.f32)])
        def kernel(src: pto.Tile, dst: pto.Tile):
            mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            with pto.vecscope():
                vec = pto.vlds(src, 0)
                pto.vsts(vec, dst, 0, mask)
            return None

        specialized = kernel.specialize(
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        frontend_kernel = build_frontend_kernel_node(specialized)
        self.assertIsInstance(frontend_kernel.body[1], FrontendVecscopeStmt)

        semantic_kernel = analyze_frontend_kernel(frontend_kernel)
        vecscope_stmts = [stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt)]
        self.assertEqual(len(vecscope_stmts), 1)

        text = specialized.mlir_text()
        self.assertEqual(text.count("pto.vecscope {"), 1)
        self.assertIn("pto.vlds", text)
        self.assertIn("pto.vsts", text)

    def test_explicit_vecscope_disables_automatic_inference(self) -> None:
        @pto.vkernel(op="explicit_vecscope_disables_infer_unique", dtypes=[(pto.f32, pto.f32)], advanced=True)
        def kernel(src: pto.Tile, dst: pto.Tile):
            mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            with pto.vecscope():
                first = pto.vlds(src, 0)
                pto.vsts(first, dst, 0, mask)
            second = pto.vlds(src, 64)
            pto.vsts(second, dst, 64, mask)
            return None

        specialized = kernel.specialize(
            src=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            dst=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope_stmts = [stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt)]
        self.assertEqual(len(vecscope_stmts), 1)

        text = specialized.mlir_text()
        self.assertEqual(text.count("pto.vecscope {"), 1)
        self.assertIn("pto.vlds", text)
        self.assertIn("pto.vsts", text)

    def test_constexpr_if_tail_store_does_not_split_inferred_vecscope(self) -> None:
        @pto.vkernel(op="trowsum_like_vecscope_unique", dtypes=[(pto.f32, pto.f32, pto.f32)], advanced=True)
        def kernel(dst: pto.Tile, src: pto.Tile, tmp: pto.Tile):
            src_dtype = src.element_type
            valid_rows, valid_cols = src.valid_shape

            for row in range(0, valid_rows, 1):
                remained = valid_cols
                acc = pto.vbr(0.0)
                for col in range(0, valid_cols, pto.get_lanes(src_dtype)):
                    mask, remained = pto.make_mask(src_dtype, remained)
                    vec = pto.vlds(src[row, col:])
                    reduced = pto.vcadd(vec, mask)
                    one_mask, _ = pto.make_mask(src_dtype, 1)
                    acc = pto.vadd(acc, reduced, one_mask)
                out_mask, _ = pto.make_mask(src_dtype, 1)
                if pto.constexpr(src_dtype != dst.element_type):
                    casted = pto.vcvt(acc, dst.element_type, out_mask)
                    pto.vsts(casted, dst[row, 0:], out_mask)
                else:
                    pto.vsts(acc, dst[row, 0:], out_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            tmp=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope_stmts = [stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt)]
        self.assertEqual(len(vecscope_stmts), 1)

        text = specialized.mlir_text()
        self.assertEqual(text.count("pto.vecscope {"), 1)
        self.assertRegex(text, r"pto\.vecscope \{\n(?:.|\n)*scf\.for %row_\d+")
        self.assertIn("pto.vsts", text)

    def test_advanced_mode_control_flow_infers_vecscope_per_branch(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32, pto.i32)], advanced=True)
        def kernel(src: pto.Tile, dst: pto.Tile, flag: pto.i32):
            dtype = src.element_type
            all_mask = pto.make_mask(dtype, pto.PAT.ALL)
            if flag:
                first = pto.vlds(src[0, 0:])
                pto.vsts(first, dst[0, 0:], all_mask)
            else:
                second = pto.vlds(src[1, 0:])
                pto.vsts(second, dst[1, 0:], all_mask)
            return None

        specialized = kernel.specialize(
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertEqual([type(stmt).__name__ for stmt in semantic_kernel.body[:-1]], [
            "SemanticAssignStmt",
            "SemanticAssignStmt",
            "SemanticIfStmt",
        ])
        if_stmt = semantic_kernel.body[2]
        self.assertIsInstance(if_stmt, SemanticIfStmt)
        self.assertEqual(len(if_stmt.then_body), 1)
        self.assertEqual(len(if_stmt.else_body), 1)
        self.assertIsInstance(if_stmt.then_body[0], SemanticVecscopeStmt)
        self.assertIsInstance(if_stmt.else_body[0], SemanticVecscopeStmt)

        text = specialized.mlir_text()
        self.assertIn("scf.if", text)
        self.assertEqual(text.count("pto.vecscope {"), 2)
        self.assertLess(text.index("scf.if"), text.index("pto.vecscope {"))
        self.assertLess(text.index("scf.if"), text.index("return"))

    def test_advanced_mode_keeps_strict_vecscope_as_hard_boundary(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32)], advanced=True)
        def kernel(src: pto.Tile, dst: pto.Tile):
            all_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            rows = src.shape[0]
            for row in range(0, rows, 1):
                vec = pto.vlds(src[row, 0:])
                pto.vsts(vec, dst[row, 0:], all_mask)
            with pto.strict_vecscope(src, dst, all_mask, 0, 64, 64) as (vin, vout, mask, lb, ub, step):
                for lane in range(lb, ub, step):
                    scoped = pto.vlds(vin, lane)
                    pto.vsts(scoped, vout, lane, mask)
            return None

        specialized = kernel.specialize(
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertEqual(text.count("pto.vecscope {"), 1)
        self.assertEqual(text.count("pto.strict_vecscope("), 1)

    def test_advanced_mode_lowers_raw_pointer_and_low_level_dma_surface(self) -> None:
        @pto.vkernel(op="ptr_dma", dtypes=[(pto.f32, pto.f32, pto.i64)], advanced=True)
        def kernel(
            src_gm: pto.ptr(pto.f32, pto.MemorySpace.GM),
            dst_gm: pto.ptr(pto.f32, pto.MemorySpace.GM),
            addr: pto.i64,
        ):
            ub_src = pto.castptr(addr, pto.ptr(pto.f32, pto.MemorySpace.UB))
            ub_dst = pto.addptr(ub_src, 64)
            mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            vec = pto.vlds(ub_src, 0)
            pto.vsts(vec, ub_dst, 0, mask)

            src_bytes = pto.castptr(src_gm, pto.ptr(pto.i8, pto.MemorySpace.GM))
            dst_bytes = pto.castptr(dst_gm, pto.ptr(pto.i8, pto.MemorySpace.GM))
            src_offset = pto.addptr(src_bytes, 0)
            dst_offset = pto.addptr(dst_bytes, 0)
            typed_src = pto.castptr(src_offset, pto.ptr(pto.f32, pto.MemorySpace.GM))
            typed_dst = pto.castptr(dst_offset, pto.ptr(pto.f32, pto.MemorySpace.GM))

            pto.set_loop2_stride_outtoub(4096, 4096)
            pto.set_loop1_stride_outtoub(4096, 4096)
            pto.set_loop_size_outtoub(1, 1)
            pto.copy_gm_to_ubuf(typed_src, ub_src, 0, 32, 128, 0, 0, False, 0, 128, 128)

            pto.set_loop2_stride_ubtoout(4096, 4096)
            pto.set_loop1_stride_ubtoout(4096, 4096)
            pto.set_loop_size_ubtoout(1, 1)
            pto.copy_ubuf_to_ubuf(ub_src, ub_dst, 0, 32, 128, 128, 128)
            pto.copy_ubuf_to_gm(ub_dst, typed_dst, 0, 32, 128, 0, 128, 128)
            return None

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(kernel))
        self.assertIsInstance(semantic_kernel.parameters[0].type, SemanticPtrType)
        self.assertEqual(semantic_kernel.parameters[0].type.memory_space, "gm")
        self.assertIsInstance(semantic_kernel.parameters[1].type, SemanticPtrType)
        self.assertEqual(semantic_kernel.parameters[1].type.memory_space, "gm")
        self.assertTrue(any(isinstance(stmt, SemanticDmaConfigStmt) for stmt in semantic_kernel.body))
        self.assertTrue(any(isinstance(stmt, SemanticLowLevelCopyStmt) for stmt in semantic_kernel.body))
        vecscope_stmts = [stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt)]
        self.assertEqual(len(vecscope_stmts), 1)

        text = kernel.mlir_text()
        self.assertIn(
            "func.func @kernel(%arg0: !pto.ptr<f32, gm>, %arg1: !pto.ptr<f32, gm>, %arg2: i64) attributes { pto.tilelang.instance } {",
            text,
        )
        self.assertRegex(
            text,
            r"%ub_src_\d+ = pto\.castptr %arg2 : i64 -> !pto\.ptr<f32, ub>",
        )
        self.assertRegex(
            text,
            r"%ub_dst_\d+ = pto\.addptr %ub_src_\d+, %c64 : !pto\.ptr<f32, ub> -> !pto\.ptr<f32, ub>",
        )
        self.assertIn("pto.vecscope {", text)
        self.assertRegex(
            text,
            r"%vec_\d+ = pto\.vlds %ub_src_\d+\[%c0\] : !pto\.ptr<f32, ub> -> !pto\.vreg<64xf32>",
        )
        self.assertRegex(
            text,
            r"pto\.vsts %vec_\d+, %ub_dst_\d+\[%c0\], %mask_\d+ : !pto\.vreg<64xf32>, !pto\.ptr<f32, ub>, !pto\.mask<b32>",
        )
        self.assertRegex(
            text,
            r"%src_bytes_\d+ = pto\.castptr %arg0 : !pto\.ptr<f32, gm> -> !pto\.ptr<i8, gm>",
        )
        self.assertRegex(
            text,
            r"%dst_bytes_\d+ = pto\.castptr %arg1 : !pto\.ptr<f32, gm> -> !pto\.ptr<i8, gm>",
        )
        self.assertRegex(
            text,
            r"%src_offset_\d+ = pto\.addptr %src_bytes_\d+, %c0 : !pto\.ptr<i8, gm> -> !pto\.ptr<i8, gm>",
        )
        self.assertRegex(
            text,
            r"%dst_offset_\d+ = pto\.addptr %dst_bytes_\d+, %c0 : !pto\.ptr<i8, gm> -> !pto\.ptr<i8, gm>",
        )
        self.assertRegex(
            text,
            r"pto\.set_loop2_stride_outtoub %tmp_\d+, %tmp_\d+ : i64, i64",
        )
        self.assertRegex(
            text,
            r"pto\.set_loop1_stride_outtoub %tmp_\d+, %tmp_\d+ : i64, i64",
        )
        self.assertRegex(
            text,
            r"pto\.set_loop_size_outtoub %tmp_\d+, %tmp_\d+ : i64, i64",
        )
        self.assertRegex(
            text,
            r"pto\.copy_gm_to_ubuf %typed_src_\d+, %ub_src_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %false, %tmp_\d+, %tmp_\d+, %tmp_\d+",
        )
        self.assertIn(
            ": !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i1, i64, i64, i64",
            text,
        )
        self.assertRegex(
            text,
            r"pto\.set_loop2_stride_ubtoout %tmp_\d+, %tmp_\d+ : i64, i64",
        )
        self.assertRegex(
            text,
            r"pto\.set_loop1_stride_ubtoout %tmp_\d+, %tmp_\d+ : i64, i64",
        )
        self.assertRegex(
            text,
            r"pto\.set_loop_size_ubtoout %tmp_\d+, %tmp_\d+ : i64, i64",
        )
        self.assertRegex(
            text,
            r"pto\.copy_ubuf_to_ubuf %ub_src_\d+, %ub_dst_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+",
        )
        self.assertIn(
            ": !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64",
            text,
        )
        self.assertRegex(
            text,
            r"pto\.copy_ubuf_to_gm %ub_dst_\d+, %typed_dst_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+",
        )

    def test_as_ptr_method_and_keyword_low_level_dma_surface_lower_in_advanced_mode(self) -> None:
        @pto.vkernel(op="tensorview_tile_as_ptr_dma_unique", dtypes=[(pto.f32, pto.f32)], advanced=True)
        def kernel(inp: pto.TensorView, dst: pto.Tile):
            gm_ptr = inp.as_ptr()
            ub_ptr = dst.as_ptr()

            pto.set_loop2_stride_outtoub(src_stride=4096, dst_stride=2048)
            pto.set_loop1_stride_outtoub(src_stride=1024, dst_stride=512)
            pto.set_loop_size_outtoub(loop1=1, loop2=1)
            pto.copy_gm_to_ubuf(
                src=gm_ptr,
                dst=ub_ptr,
                n_burst=1,
                len_burst=64,
                gm_stride=128,
                ub_stride=128,
                enable_ub_pad=False,
            )
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertTrue(any(isinstance(stmt, SemanticDmaConfigStmt) for stmt in semantic_kernel.body))
        self.assertTrue(any(isinstance(stmt, SemanticLowLevelCopyStmt) for stmt in semantic_kernel.body))

        text = specialized.mlir_text()
        self.assertRegex(
            text,
            r"%gm_ptr_\d+ = pto\.tensor_view_addr %arg0 : !pto\.tensor_view<\?x\?x\?x\?x\?xf32> -> !pto\.ptr<f32, gm>",
        )
        self.assertRegex(
            text,
            r"%ub_ptr_\d+ = pto\.tile_buf_addr %arg1 : !pto\.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0> -> !pto\.ptr<f32, ub>",
        )
        self.assertRegex(text, r"pto\.set_loop2_stride_outtoub %tmp_\d+, %tmp_\d+ : i64, i64")
        self.assertRegex(text, r"pto\.set_loop1_stride_outtoub %tmp_\d+, %tmp_\d+ : i64, i64")
        self.assertRegex(text, r"pto\.set_loop_size_outtoub %tmp_\d+, %tmp_\d+ : i64, i64")
        self.assertRegex(
            text,
            r"pto\.copy_gm_to_ubuf %gm_ptr_\d+, %ub_ptr_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %false, %tmp_\d+, %tmp_\d+, %tmp_\d+",
        )

    def test_set_mov_pad_val_lowers_in_advanced_mode(self) -> None:
        @pto.vkernel(op="set_mov_pad_val_dma_unique", dtypes=[(pto.f32, pto.f32)], advanced=True)
        def kernel(inp: pto.TensorView, dst: pto.Tile):
            gm_ptr = inp.as_ptr()
            ub_ptr = dst.as_ptr()

            pto.set_mov_pad_val(pad_value=pto.f32(0.0))
            pto.set_loop2_stride_outtoub(src_stride=4096, dst_stride=2048)
            pto.set_loop1_stride_outtoub(src_stride=1024, dst_stride=512)
            pto.set_loop_size_outtoub(loop1=1, loop2=1)
            pto.copy_gm_to_ubuf(
                src=gm_ptr,
                dst=ub_ptr,
                n_burst=1,
                len_burst=64,
                gm_stride=128,
                ub_stride=128,
                enable_ub_pad=True,
            )
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertTrue(any(isinstance(stmt, SemanticDmaUnaryConfigStmt) for stmt in semantic_kernel.body))

        text = specialized.mlir_text()
        self.assertRegex(text, r"pto\.set_mov_pad_val %[^ ]+ : f32")
        self.assertRegex(
            text,
            r"pto\.copy_gm_to_ubuf %gm_ptr_\d+, %ub_ptr_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %true, %tmp_\d+, %tmp_\d+, %tmp_\d+",
        )

    def test_set_mov_pad_val_automatically_bitcasts_unsigned_tile_pad_value_to_signless_scalar(self) -> None:
        @pto.vkernel(op="set_mov_pad_val_tile_pad_bitcast_unique", dtypes=[(pto.ui16,)], advanced=True)
        def kernel(dst: pto.Tile):
            pto.set_mov_pad_val(dst.pad_value.eval())
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(
                shape=(260, 32),
                memory_space=pto.MemorySpace.UB,
                config=pto.TileConfig.from_mapping(
                    {
                        "b_layout": "row_major",
                        "s_layout": "none_box",
                        "s_fractal_size": 512,
                        "pad_value": "0x2",
                    }
                ),
                valid_shape=(260, 7),
            )
        )

        text = specialized.mlir_text()
        self.assertIn("builtin.unrealized_conversion_cast", text)
        self.assertRegex(text, r"pto\.set_mov_pad_val %[^ ]+ : i16")

    def test_copy_ubuf_to_gm_keyword_surface_lowers_in_advanced_mode(self) -> None:
        @pto.vkernel(op="tile_to_tensorview_dma_unique", dtypes=[(pto.f32, pto.f32)], advanced=True)
        def kernel(src: pto.Tile, dst: pto.TensorView):
            ub_ptr = src.as_ptr()
            gm_ptr = dst.as_ptr()

            pto.set_loop2_stride_ubtoout(src_stride=4096, dst_stride=2048)
            pto.set_loop1_stride_ubtoout(src_stride=1024, dst_stride=512)
            pto.set_loop_size_ubtoout(loop1=1, loop2=1)
            pto.copy_ubuf_to_gm(
                src=ub_ptr,
                dst=gm_ptr,
                n_burst=1,
                len_burst=64,
                gm_stride=128,
                ub_stride=128,
            )
            return None

        specialized = kernel.specialize(
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertTrue(any(isinstance(stmt, SemanticDmaConfigStmt) for stmt in semantic_kernel.body))
        self.assertTrue(any(isinstance(stmt, SemanticLowLevelCopyStmt) for stmt in semantic_kernel.body))

        text = specialized.mlir_text()
        self.assertRegex(
            text,
            r"%ub_ptr_\d+ = pto\.tile_buf_addr %arg0 : !pto\.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0> -> !pto\.ptr<f32, ub>",
        )
        self.assertRegex(
            text,
            r"%gm_ptr_\d+ = pto\.tensor_view_addr %arg1 : !pto\.tensor_view<\?x\?x\?x\?x\?xf32> -> !pto\.ptr<f32, gm>",
        )
        self.assertRegex(text, r"pto\.set_loop2_stride_ubtoout %tmp_\d+, %tmp_\d+ : i64, i64")
        self.assertRegex(text, r"pto\.set_loop1_stride_ubtoout %tmp_\d+, %tmp_\d+ : i64, i64")
        self.assertRegex(text, r"pto\.set_loop_size_ubtoout %tmp_\d+, %tmp_\d+ : i64, i64")
        self.assertRegex(
            text,
            r"pto\.copy_ubuf_to_gm %ub_ptr_\d+, %gm_ptr_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+",
        )

    def test_castptr_rejects_tensorview_or_tile_inputs_in_advanced_mode(self) -> None:
        @pto.vkernel(op="castptr_tensorview_reject_unique", dtypes=[(pto.f32,)], advanced=True)
        def tensorview_kernel(inp: pto.TensorView):
            tmp = pto.castptr(inp, pto.ptr(pto.f32, pto.MemorySpace.GM))
            return None

        with self.assertRaises(TypeError) as tensorview_ctx:
            analyze_frontend_kernel(build_frontend_kernel_node(tensorview_kernel))
        self.assertIn("pto.castptr input must be an index/i64, pointer, or memref-backed address value", str(tensorview_ctx.exception))

        @pto.vkernel(op="castptr_tile_reject_unique", dtypes=[(pto.f32,)], advanced=True)
        def tile_kernel(inp: pto.Tile):
            tmp = pto.castptr(inp, pto.ptr(pto.f32, pto.MemorySpace.UB))
            return None

        specialized = tile_kernel.specialize(
            inp=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )
        with self.assertRaises(TypeError) as tile_ctx:
            analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertIn("pto.castptr input must be an index/i64, pointer, or memref-backed address value", str(tile_ctx.exception))

    def test_constexpr_if_folds_static_dtype_condition_without_scf_if(self) -> None:
        @pto.vkernel(op="constexpr_if_dtype_fold", dtypes=[(pto.f16, pto.f32)])
        def kernel(dst: pto.Tile, src: pto.Tile):
            step = 64
            if pto.constexpr(dst.element_type != src.element_type):
                step = 128
            else:
                step = 64
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertFalse(any(isinstance(stmt, SemanticIfStmt) for stmt in semantic_kernel.body))

        text = specialized.mlir_text()
        self.assertNotIn("scf.if", text)
        self.assertNotIn("arith.cmpi ne", text)
        self.assertRegex(text, r"%step_\d+ = arith\.constant 128 : index")

    def test_constexpr_if_rejects_non_static_condition(self) -> None:
        @pto.vkernel(op="constexpr_if_dynamic_reject", dtypes=[(pto.f32,)])
        def kernel(src: pto.TensorView):
            step = 64
            if pto.constexpr(src.shape[0] != 1):
                step = 128
            return None

        with self.assertRaises(TypeError) as ctx:
            analyze_frontend_kernel(build_frontend_kernel_node(kernel))
        self.assertIn(
            "if pto.constexpr(...) condition must be a compile-time bool",
            str(ctx.exception),
        )

    def test_if_compare_or_condition_lowers_to_cmp_and_bool_ops(self) -> None:
        @pto.vkernel(op="if_compare_or", dtypes=[(pto.f32,)])
        def kernel(src: pto.TensorView):
            loop1 = src.shape[3]
            loop2 = src.shape[4]
            step = 64
            if loop1 != 1 or loop2 != 1:
                step = 128
            else:
                step = 64
            return None

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(kernel))
        self.assertEqual(
            [(param.name, param.kind) for param in semantic_kernel.parameters],
            [("src", "tensorview")],
        )
        self.assertIsInstance(semantic_kernel.body[3], SemanticIfStmt)
        condition = semantic_kernel.body[3].condition
        self.assertIsInstance(condition, SemanticBinaryExpr)
        self.assertEqual(condition.op, "or")
        self.assertIsInstance(condition.lhs, SemanticBinaryExpr)
        self.assertEqual(condition.lhs.op, "ne")
        self.assertIsInstance(condition.rhs, SemanticBinaryExpr)
        self.assertEqual(condition.rhs.op, "ne")

        text = kernel.mlir_text()
        self.assertEqual(text.count("arith.cmpi ne"), 2)
        self.assertRegex(text, r"%loop1_\d+ = pto\.get_tensor_view_dim %arg0, %c3 : !pto\.tensor_view<\?x\?x\?x\?x\?xf32> -> index")
        self.assertRegex(text, r"%loop2_\d+ = pto\.get_tensor_view_dim %arg0, %c4 : !pto\.tensor_view<\?x\?x\?x\?x\?xf32> -> index")
        self.assertRegex(text, r"arith\.cmpi ne, %loop1_\d+, %c1 : index")
        self.assertRegex(text, r"arith\.cmpi ne, %loop2_\d+, %c1 : index")
        self.assertRegex(text, r"arith\.ori %tmp_\d+, %tmp_\d+ : i1")
        self.assertRegex(text, r"%step_\d+ = scf\.if %tmp_\d+ -> \(index\) \{")

    def test_if_ordered_index_comparisons_lower_to_signed_cmp_predicates(self) -> None:
        @pto.vkernel(op="if_compare_ordered_index", dtypes=[(pto.f32,)])
        def kernel(src: pto.TensorView):
            dim0 = src.shape[0]
            dim1 = src.shape[1]
            step = 64
            if dim0 > 1 and dim1 <= 8:
                step = 128
            else:
                step = 32
            return None

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(kernel))
        self.assertIsInstance(semantic_kernel.body[3], SemanticIfStmt)
        condition = semantic_kernel.body[3].condition
        self.assertIsInstance(condition, SemanticBinaryExpr)
        self.assertEqual(condition.op, "and")

        text = kernel.mlir_text()
        self.assertRegex(text, r"arith\.cmpi sgt, %dim0_\d+, %c1 : index")
        self.assertRegex(text, r"arith\.cmpi sle, %dim1_\d+, %c8 : index")
        self.assertRegex(text, r"arith\.andi %tmp_\d+, %tmp_\d+ : i1")
        self.assertRegex(text, r"%step_\d+ = scf\.if %tmp_\d+ -> \(index\) \{")

    def test_if_ordered_float_comparison_lowers_to_cmpf_predicate(self) -> None:
        @pto.vkernel(op="if_compare_ordered_float", dtypes=[(pto.f32, pto.f32, pto.f32)])
        def kernel(src: pto.TensorView, lhs: pto.f32, rhs: pto.f32):
            step = 64
            if lhs > rhs:
                step = 128
            else:
                step = 64
            return None

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(kernel))
        self.assertIsInstance(semantic_kernel.body[1], SemanticIfStmt)

        text = kernel.mlir_text()
        self.assertRegex(text, r"arith\.cmpf ogt, %arg1, %arg2 : f32")
        self.assertRegex(text, r"%step_\d+ = scf\.if %tmp_\d+ -> \(index\) \{")

    def test_shape_and_stride_tuple_unpacking_lower_cleanly(self) -> None:
        @pto.vkernel(op="shape_stride_unpack", dtypes=[(pto.f32, pto.f32)], advanced=True)
        def kernel(src: pto.TensorView, dst: pto.Tile):
            g0, g1, g2, g3, g4 = src.shape
            s0, s1, s2, s3, s4 = src.strides
            ub_rows, ub_cols = dst.shape
            total = g0 + g1 + g2 + g3 + g4
            stride_total = s0 + s1 + s2 + s3 + s4
            area = ub_rows * ub_cols
            if total != 0 or stride_total != area:
                total = area
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertEqual(
            [(param.name, param.kind) for param in semantic_kernel.parameters],
            [("src", "tensorview"), ("dst", "tile")],
        )

        text = specialized.mlir_text()
        self.assertEqual(text.count("pto.get_tensor_view_dim"), 5)
        self.assertEqual(text.count("pto.get_tensor_view_stride"), 5)
        self.assertRegex(text, r"%ub_rows_\d+ = arith\.constant 8 : index")
        self.assertRegex(text, r"%ub_cols_\d+ = arith\.constant 64 : index")

    def test_shape_subscript_rejects_non_literal_index_in_semantic(self) -> None:
        @pto.vkernel(op="shape_dynamic_subscript_reject_unique", dtypes=[(pto.f32,)])
        def kernel(src: pto.TensorView):
            axis = src.shape[0]
            value = src.shape[axis]
            return None

        with self.assertRaises(TypeError) as ctx:
            analyze_frontend_kernel(build_frontend_kernel_node(kernel))
        self.assertIn(
            "shape/stride/valid_shape subscript index must be an integer literal in TileLang DSL v1",
            str(ctx.exception),
        )

    def test_valid_shape_subscript_rejects_non_literal_index_in_semantic(self) -> None:
        @pto.vkernel(op="valid_shape_dynamic_subscript_reject_unique", dtypes=[(pto.f16,)], advanced=True)
        def kernel(dst: pto.Tile):
            axis = 0
            value = dst.valid_shape[axis]
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(
                shape=(8, 128),
                memory_space=pto.MemorySpace.UB,
                valid_shape=("valid_rows", "valid_cols"),
            )
        )

        with self.assertRaises(TypeError) as ctx:
            analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertIn(
            "tuple subscript index must be an integer literal in TileLang DSL v1",
            str(ctx.exception),
        )

    def test_tuple_call_result_subscript_rejects_in_semantic(self) -> None:
        @pto.vkernel(op="tuple_call_result_subscript_reject_unique", dtypes=[(pto.f16,)], advanced=True)
        def kernel(dst: pto.Tile):
            mask = pto.make_mask(dst.element_type, pto.i32(64))[0]
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )

        with self.assertRaises(TypeError) as ctx:
            analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertIn(
            "tuple subscripting currently requires a shape-like tuple expression in TileLang DSL v1",
            str(ctx.exception),
        )

    def test_advanced_mode_lowers_compare_predicate_carry_and_rearrangement_families(self) -> None:
        @pto.vkernel(op="advanced_family", dtypes=[(pto.i32, pto.i32, pto.i32, pto.i32)], advanced=True)
        def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile, scalar: pto.i32):
            all_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
            lhs = pto.vlds(src0[0, 0:])
            rhs = pto.vlds(src1[0, 0:])
            cmp_mask = pto.vcmp(lhs, rhs, all_mask, pto.CmpMode.LT)
            cmp_scalar_mask = pto.vcmps(lhs, scalar, all_mask, pto.CmpMode.GT)
            negated = pto.pnot(cmp_mask, all_mask)
            picked = pto.psel(cmp_mask, negated, cmp_scalar_mask)
            packed = pto.ppack(picked, pto.PredicatePart.LOWER)
            unpacked = pto.punpack(packed, pto.PredicatePart.HIGHER)
            sum_vec, carry_mask = pto.vaddc(lhs, rhs, all_mask)
            diff_vec, borrow_mask = pto.vsubc(lhs, rhs, all_mask)
            sum_with_carry, carry_mask2 = pto.vaddcs(sum_vec, diff_vec, carry_mask, all_mask)
            diff_with_borrow, borrow_mask2 = pto.vsubcs(sum_with_carry, diff_vec, borrow_mask, all_mask)
            low, high = pto.vintlv(sum_with_carry, diff_with_borrow)
            dlow, dhigh = pto.vdintlv(low, high)
            even = pto.vintlvv2(dlow, dhigh, "PART_EVEN")
            odd = pto.vdintlvv2(dlow, dhigh, "PART_ODD")
            selected = pto.vsel(even, odd, unpacked)
            selected_r = pto.vselr(selected, sum_with_carry)
            final = pto.vselrv2(selected_r, diff_with_borrow)
            pto.vsts(final, dst[0, 0:], all_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src0=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope_stmts = [stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt)]
        self.assertEqual(len(vecscope_stmts), 1)

        text = specialized.mlir_text()
        self.assertIn("pto.vecscope {", text)
        self.assertIn('pto.vcmp ', text)
        self.assertIn(', "lt" : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.mask<b32>', text)
        self.assertIn('pto.vcmps ', text)
        self.assertIn(', "gt" : !pto.vreg<64xi32>, i32, !pto.mask<b32> -> !pto.mask<b32>', text)
        self.assertIn(" = pto.pnot ", text)
        self.assertIn(" = pto.psel ", text)
        self.assertIn(' = pto.ppack ', text)
        self.assertIn('"LOWER"', text)
        self.assertIn(' = pto.punpack ', text)
        self.assertIn('"HIGHER"', text)
        self.assertRegex(
            text,
            r"%sum_vec_\d+, %carry_mask_\d+ = pto\.vaddc %lhs_\d+, %rhs_\d+, %all_mask_\d+ : !pto\.vreg<64xi32>, !pto\.vreg<64xi32>, !pto\.mask<b32> -> !pto\.vreg<64xi32>, !pto\.mask<b32>",
        )
        self.assertRegex(
            text,
            r"%diff_vec_\d+, %borrow_mask_\d+ = pto\.vsubc %lhs_\d+, %rhs_\d+, %all_mask_\d+ : !pto\.vreg<64xi32>, !pto\.vreg<64xi32>, !pto\.mask<b32> -> !pto\.vreg<64xi32>, !pto\.mask<b32>",
        )
        self.assertRegex(
            text,
            r"%sum_with_carry_\d+, %carry_mask2_\d+ = pto\.vaddcs %sum_vec_\d+, %diff_vec_\d+, %carry_mask_\d+, %all_mask_\d+ : !pto\.vreg<64xi32>, !pto\.vreg<64xi32>, !pto\.mask<b32>, !pto\.mask<b32> -> !pto\.vreg<64xi32>, !pto\.mask<b32>",
        )
        self.assertRegex(
            text,
            r"%diff_with_borrow_\d+, %borrow_mask2_\d+ = pto\.vsubcs %sum_with_carry_\d+, %diff_vec_\d+, %borrow_mask_\d+, %all_mask_\d+ : !pto\.vreg<64xi32>, !pto\.vreg<64xi32>, !pto\.mask<b32>, !pto\.mask<b32> -> !pto\.vreg<64xi32>, !pto\.mask<b32>",
        )
        self.assertRegex(
            text,
            r"%low_\d+, %high_\d+ = pto\.vintlv %sum_with_carry_\d+, %diff_with_borrow_\d+ : !pto\.vreg<64xi32>, !pto\.vreg<64xi32> -> !pto\.vreg<64xi32>, !pto\.vreg<64xi32>",
        )
        self.assertRegex(
            text,
            r"%dlow_\d+, %dhigh_\d+ = pto\.vdintlv %low_\d+, %high_\d+ : !pto\.vreg<64xi32>, !pto\.vreg<64xi32> -> !pto\.vreg<64xi32>, !pto\.vreg<64xi32>",
        )
        self.assertIn(" = pto.vintlvv2 ", text)
        self.assertIn(" = pto.vdintlvv2 ", text)
        self.assertIn(" = pto.vsel ", text)
        self.assertIn(" = pto.vselr ", text)
        self.assertIn(" = pto.vselrv2 ", text)
        self.assertIn("pto.vsts ", text)

    def test_inferred_vecscope_keeps_vbitcast_and_mem_bar_with_vector_users(self) -> None:
        @pto.vkernel(op="issue_217_vecscope", dtypes=[(pto.i32, pto.ui8)], advanced=True)
        def kernel(src: pto.Tile, dst: pto.Tile):
            valid_rows, valid_cols = dst.valid_shape
            full_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
            idx_mask = pto.make_mask(pto.i16, pto.PAT.ALL)
            v_idx = pto.vci(pto.i8(0), pto.OrderMode.ASC)
            v_idx_i16 = pto.vbitcast(v_idx, pto.i16)
            v_idx_i16 = pto.vmuls(v_idx_i16, pto.i16(4), idx_mask)
            v_idx_ui8 = pto.vbitcast(v_idx_i16, pto.ui8)
            for row in range(0, valid_rows, 1):
                remained = valid_cols
                for col in range(0, valid_cols, pto.get_lanes(pto.i32)):
                    store_mask, remained = pto.make_mask(pto.ui8, remained)
                    vec = pto.vlds(src[row, col:])
                    converted = pto.vcvt(
                        vec,
                        pto.ui8,
                        full_mask,
                        sat=pto.VcvtSatMode.NOSAT,
                        part=pto.VcvtPartMode.P0,
                    )
                    result = pto.vselr(converted, v_idx_ui8)
                    pto.mem_bar(pto.BarrierType.VST_VST)
                    pto.vsts(result, dst[row, col:], store_mask, dist=pto.VStoreDist.NORM_B8)

            return None

        specialized = kernel.specialize(
            src=pto.TileSpecialization(shape=(16, 64), memory_space=pto.MemorySpace.UB),
            dst=pto.TileSpecialization(shape=(16, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope_stmts = [stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt)]
        self.assertEqual(len(vecscope_stmts), 1)

        text = specialized.mlir_text()
        self.assertEqual(text.count("pto.vecscope {"), 1)
        self.assertIn("pto.vbitcast", text)
        self.assertIn('pto.mem_bar "VST_VST"', text)
        self.assertIn("pto.vselr", text)
        self.assertIn("pto.vsts", text)

    def test_inferred_vecscope_keeps_scalar_get_lanes_between_vector_def_and_use(self) -> None:
        @pto.vkernel(op="issue_240_vecscope", dtypes=[(pto.si8, pto.i32)], advanced=True)
        def kernel(src: pto.Tile, dst: pto.Tile):
            valid_rows, valid_cols = dst.valid_shape
            b8_mask = pto.make_mask(pto.ui8, pto.PAT.ALL)
            v_zero = pto.vdup(pto.ui8(0), b8_mask)
            lanes_i32 = pto.get_lanes(pto.i32)
            lanes_i16 = pto.get_lanes(pto.i16)
            for row in range(0, valid_rows, 1):
                remained = valid_cols
                for col in range(0, valid_cols, lanes_i16):
                    mask_b16_cur, remained = pto.make_mask(pto.i16, remained)
                    mask_b16_next, remained2 = pto.make_mask(pto.i16, remained)
                    mask_b32_cur = pto.punpack(mask_b16_cur, pto.PredicatePart.LOWER)
                    mask_b32_next = pto.punpack(mask_b16_next, pto.PredicatePart.LOWER)
                    vec_si8 = pto.vlds(src[row, col:], dist=pto.VLoadDist.UNPK_B8)
                    vec_ui8 = pto.vbitcast(vec_si8, pto.ui8)
                    vec_ui8_lo, vec_ui8_hi = pto.vintlv(vec_ui8, v_zero)
                    vec_si8_lo = pto.vbitcast(vec_ui8_lo, pto.si8)
                    vec_si8_hi = pto.vbitcast(vec_ui8_hi, pto.si8)
                    out_lo = pto.vcvt(vec_si8_lo, pto.i32, b8_mask, part=pto.VcvtPartMode.P0)
                    out_hi = pto.vcvt(vec_si8_hi, pto.i32, b8_mask, part=pto.VcvtPartMode.P0)
                    pto.vsts(out_lo, dst[row, col:], mask_b32_cur, dist=pto.VStoreDist.NORM_B32)
                    pto.vsts(
                        out_hi,
                        dst[row, col + lanes_i32:],
                        mask_b32_next,
                        dist=pto.VStoreDist.NORM_B32,
                    )
            return None

        specialized = kernel.specialize(
            src=pto.TileSpecialization(shape=(16, 64), memory_space=pto.MemorySpace.UB),
            dst=pto.TileSpecialization(shape=(16, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope_stmts = [stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt)]
        self.assertEqual(len(vecscope_stmts), 1)

        text = specialized.mlir_text()
        self.assertEqual(text.count("pto.vecscope {"), 1)
        self.assertIn(" = arith.constant 64 : index", text)
        self.assertIn(" = arith.constant 128 : index", text)
        self.assertIn(" = pto.vdup ", text)
        self.assertIn(" = pto.vintlv ", text)

    def test_punpack_widens_b16_mask_for_norm_b32_store_in_advanced_mode(self) -> None:
        @pto.vkernel(op="punpack_widen_b16_to_b32_unique", dtypes=[(pto.si8, pto.i32)], advanced=True)
        def kernel(src: pto.Tile, dst: pto.Tile):
            valid_rows, valid_cols = dst.valid_shape
            lanes_i32 = pto.get_lanes(pto.i32)
            for row in range(0, valid_rows, 1):
                b8_mask = pto.make_mask(pto.i8, pto.PAT.ALL)
                mask_b16, _ = pto.make_mask(pto.i16, valid_cols)
                mask_b32 = pto.punpack(mask_b16, pto.PredicatePart.LOWER)
                vec_si8 = pto.vlds(src[row, 0:], dist=pto.VLoadDist.UNPK_B8)
                vec_ui8 = pto.vbitcast(vec_si8, pto.ui8)
                v_zero_i8 = pto.vdup(pto.i8(0), b8_mask)
                v_zero = pto.vbitcast(v_zero_i8, pto.ui8)
                wide_lo, _ = pto.vintlv(vec_ui8, v_zero)
                narrowed = pto.vbitcast(wide_lo, pto.si8)
                converted = pto.vcvt(narrowed, pto.i32, b8_mask, part=pto.VcvtPartMode.P0)
                pto.vsts(converted, dst[row, 0:], mask_b32, dist=pto.VStoreDist.NORM_B32)
                pto.vsts(converted, dst[row, lanes_i32:], mask_b32, dist=pto.VStoreDist.NORM_B32)
            return None

        specialized = kernel.specialize(
            src=pto.TileSpecialization(shape=(16, 64), memory_space=pto.MemorySpace.UB),
            dst=pto.TileSpecialization(shape=(16, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn(' = pto.punpack ', text)
        self.assertRegex(
            text,
            r"pto\.punpack %mask_b16_\d+, \"LOWER\" : !pto\.mask<b16> -> !pto\.mask<b32>",
        )
        self.assertRegex(
            text,
            r"pto\.vsts %converted_\d+, %tmp_\d+\[%c0\], %mask_b32_\d+ \{dist = \"NORM_B32\"\} : !pto\.vreg<64xi32>, memref<\?x\?xi32, strided<\[\?, \?\], offset: \?>, #pto\.address_space<vec>>, !pto\.mask<b32>",
        )

    def test_elementwise_kernel_positive_regression_covers_vecscope_tail_mask_and_dynamic_loop_bound(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32, pto.i32)], advanced=True)
        def kernel(inp: pto.TensorView, tile: pto.Tile, remaining: pto.i32):
            rows = inp.shape[0]
            with pto.strict_vecscope(tile, tile, remaining, 0, rows, 64) as (
                src,
                dst,
                rem,
                lb,
                ub,
                step,
            ):
                for lane in range(lb, ub, step):
                    mask, rem = pto.make_mask(pto.f32, rem)
                    vec = pto.vlds(src, lane)
                    pto.vsts(vec, dst, lane, mask)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertEqual(len(semantic_kernel.body), 3)
        self.assertIsInstance(semantic_kernel.body[1], SemanticStrictVecscopeStmt)

        vecscope = semantic_kernel.body[1]
        self.assertIsInstance(vecscope, SemanticStrictVecscopeStmt)
        loop_stmt = vecscope.body[0]
        self.assertIsInstance(loop_stmt, SemanticForStmt)
        self.assertEqual(len(loop_stmt.loop_carried), 1)
        self.assertEqual(loop_stmt.loop_carried[0].name, "rem")

        text = specialized.mlir_text()
        self.assertIn(
            "func.func @kernel(%arg0: !pto.tensor_view<?x?x?x?x?xf32>, %arg1: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>, %arg2: i32) attributes { pto.tilelang.instance } {",
            text,
        )
        self.assertRegex(
            text,
            r"%rows_\d+ = pto\.get_tensor_view_dim %arg0, %c0 : !pto\.tensor_view<\?x\?x\?x\?x\?xf32> -> index",
        )
        self.assertRegex(
            text,
            r"pto\.strict_vecscope\(%tmp_\d+, %tmp_\d+, %arg2, %c0, %rows_\d+, %c64\)",
        )
        self.assertRegex(
            text,
            r"scf\.for %lane_\d+ = %lb_\d+ to %ub_\d+ step %step_\d+ iter_args\(%rem_iter_\d+ = %rem_\d+\) -> \(i32\) \{",
        )
        self.assertRegex(
            text,
            r"%mask_\d+, %rem_\d+ = pto\.plt_b32 %rem_iter_\d+ : i32 -> !pto\.mask<b32>, i32",
        )

    def test_if_else_and_sync_ops_lower_to_scf_if_and_authoring_sync_ops(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32, pto.i32)], advanced=True)
        def kernel(inp: pto.TensorView, tile: pto.Tile, flag: pto.i32):
            pto.set_flag(pto.PIPE.MTE2, pto.PIPE.V, pto.EVENT.ID0)
            pto.wait_flag(pto.PIPE.MTE2, pto.PIPE.V, pto.EVENT.ID0)
            step = 64
            if flag:
                step = 64
                pto.set_flag(pto.PIPE.V, pto.PIPE.MTE3, pto.EVENT.ID0)
            else:
                step = 128
                pto.wait_flag(pto.PIPE.V, pto.PIPE.MTE3, pto.EVENT.ID0)
            with pto.strict_vecscope(tile, tile, 0, 256, step) as (src, dst, lb, ub, vec_step):
                for lane in range(lb, ub, vec_step):
                    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                    vec = pto.vlds(src, lane)
                    pto.vsts(vec, dst, lane, mask)
            pto.pipe_barrier(pto.PIPE.ALL)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertIsInstance(semantic_kernel.body[0], SemanticSetFlagStmt)
        self.assertIsInstance(semantic_kernel.body[1], SemanticWaitFlagStmt)
        self.assertIsInstance(semantic_kernel.body[3], SemanticIfStmt)
        self.assertIsInstance(semantic_kernel.body[5], SemanticPipeBarrierStmt)

        text = specialized.mlir_text()
        self.assertIn('pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]', text)
        self.assertIn('pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]', text)
        self.assertIn("= arith.cmpi ne, %arg2, %c0_i32 : i32", text)
        self.assertRegex(text, r"%step_\d+ = scf\.if %tmp_\d+ -> \(index\) \{")
        self.assertIn('pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]', text)
        self.assertIn('pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]', text)
        self.assertRegex(text, r"scf\.yield %step_\d+ : index")
        self.assertIn("%step_2 = arith.constant 128 : index", text)
        self.assertRegex(
            text,
            r"pto\.strict_vecscope\(%tmp_\d+, %tmp_\d+, %c0, %c256, %step_\d+\)",
        )
        self.assertIn("scf.for %lane_", text)
        self.assertIn("pto.barrier #pto.pipe<PIPE_ALL>", text)

    def test_extended_sync_buffer_ops_lower_to_authoring_surface(self) -> None:
        Pipe = pto.Pipe
        Event = pto.Event
        BarrierType = pto.BarrierType

        @pto.vkernel(
            op="extended_sync_surface",
            dtypes=[(pto.f32, pto.i64, pto.i64, pto.i64, pto.i64, pto.i32)],
            advanced=True,
        )
        def kernel(
            tile: pto.Tile,
            buf_id: pto.i64,
            mode: pto.i64,
            core_id: pto.i64,
            block_id: pto.i64,
            config: pto.i32,
        ):
            pto.get_buf(Pipe.MTE2, buf_id, mode)
            pto.rls_buf(Pipe.V, buf_id)
            pto.mem_bar(BarrierType.VST_VLD)
            pto.set_cross_core(core_id, Event.ID7)
            pto.set_intra_block(block_id, Event.ID16)
            pto.set_intra_core(config)
            pto.wait_flag_dev(core_id, Event.ID8)
            pto.wait_intra_core(block_id, Event.ID31)
            with pto.strict_vecscope(tile, tile, 0, 128, 64) as (src, dst, lb, ub, step):
                for lane in range(lb, ub, step):
                    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                    vec = pto.vlds(src, lane)
                    pto.vsts(vec, dst, lane, mask)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertIsInstance(semantic_kernel.body[0], SemanticGetBufStmt)
        self.assertIsInstance(semantic_kernel.body[1], SemanticRlsBufStmt)
        self.assertIsInstance(semantic_kernel.body[2], SemanticMemBarStmt)
        self.assertIsInstance(semantic_kernel.body[3], SemanticSetCrossCoreStmt)
        self.assertIsInstance(semantic_kernel.body[4], SemanticSetIntraBlockStmt)
        self.assertIsInstance(semantic_kernel.body[5], SemanticSetIntraCoreStmt)
        self.assertIsInstance(semantic_kernel.body[6], SemanticWaitFlagDevStmt)
        self.assertIsInstance(semantic_kernel.body[7], SemanticWaitIntraCoreStmt)

        text = specialized.mlir_text()
        self.assertIn('pto.get_buf "PIPE_MTE2", %arg1, %arg2 : i64, i64', text)
        self.assertIn('pto.rls_buf "PIPE_V", %arg1, %c0_i64 : i64, i64', text)
        self.assertIn('pto.mem_bar "VST_VLD"', text)
        self.assertIn("pto.set_cross_core %arg3, %c7_i64 : i64, i64", text)
        self.assertIn("pto.set_intra_block %arg4, %c16_i64 : i64, i64", text)
        self.assertIn("pto.set_intra_core %arg5 : i32", text)
        self.assertIn("pto.wait_flag_dev %arg3, %c8_i64 : i64, i64", text)
        self.assertIn("pto.wait_intra_core %arg4, %c31_i64 : i64, i64", text)

    def test_mem_bar_accepts_extended_barrier_type_enum(self) -> None:
        BarrierType = pto.BarrierType

        @pto.vkernel(
            op="mem_bar_extended_enum_unique",
            dtypes=[(pto.f32, pto.f32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            pto.mem_bar(BarrierType.ST_VST)
            mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            pto.vsts(vec, dst, 0, mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertIsInstance(semantic_kernel.body[0], SemanticMemBarStmt)

        text = specialized.mlir_text()
        self.assertIn('pto.mem_bar "ST_VST"', text)

    def test_mem_bar_accepts_extended_barrier_type_enum_vst_st(self) -> None:
        BarrierType = pto.BarrierType

        @pto.vkernel(
            op="mem_bar_extended_enum_vst_st_unique",
            dtypes=[(pto.f32, pto.f32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            pto.mem_bar(BarrierType.VST_ST)
            mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            pto.vsts(vec, dst, 0, mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn('pto.mem_bar "VST_ST"', text)

    def test_mem_bar_rejects_unknown_barrier_string(self) -> None:
        with self.assertRaises(TypeError) as ctx:

            @pto.vkernel(
                op="mem_bar_invalid_string_unique",
                dtypes=[(pto.f32, pto.f32)],
                advanced=True,
            )
            def kernel(dst: pto.Tile, src: pto.Tile):
                pto.mem_bar("NOT_A_BARRIER")
                mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                vec = pto.vlds(src, 0)
                pto.vsts(vec, dst, 0, mask)
                return None

            specialized = kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
                src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            )
            specialized.mlir_text()

        self.assertIn("canonical barrier string", str(ctx.exception))

    def test_runtime_block_queries_and_scalar_pointer_helpers_lower_to_v0_3_surface(self) -> None:
        @pto.vkernel(
            op="runtime_block_queries_and_scalar_helpers",
            dtypes=[(pto.f32, pto.f32)],
            advanced=True,
        )
        def kernel(
            src: pto.ptr(pto.f32, pto.MemorySpace.UB),
            dst: pto.ptr(pto.f32, pto.MemorySpace.UB),
        ):
            block = pto.get_block_idx()
            block_num = pto.get_block_num()
            subblock = pto.get_subblock_idx()
            subblock_num = pto.get_subblock_num()
            value = pto.load_scalar(src, 0)
            pto.store_scalar(dst, 0, value)
            return None

        specialized = kernel.specialize()
        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))

        store_stmt = next(stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticScalarStoreStmt))
        self.assertIsInstance(store_stmt, SemanticScalarStoreStmt)
        self.assertEqual(store_stmt.destination.type.element_dtype, pto.f32)

        text = specialized.mlir_text()
        self.assertIn("= pto.get_block_idx", text)
        self.assertIn("= pto.get_block_num", text)
        self.assertIn("= pto.get_subblock_idx", text)
        self.assertIn("= pto.get_subblock_num", text)
        self.assertIn("= pto.load_scalar %arg0[%c0] : !pto.ptr<f32, ub> -> f32", text)
        self.assertIn("pto.store_scalar", text)

    def test_vldsx2_and_vstsx2_tile_sugar_lower_with_normalized_dist_tokens(self) -> None:
        @pto.vkernel(op="vldsx2_vstsx2_tile_sugar", dtypes=[(pto.f32, pto.f32)])
        def kernel(src: pto.Tile, dst: pto.Tile):
            mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            low, high = pto.vldsx2(src[0, 0:], pto.DeinterleaveDist.B32)
            pto.vstsx2(low, high, dst[0, 0:], pto.InterleaveDist.B32, mask)
            return None

        specialized = kernel.specialize(
            src=pto.TileSpecialization(shape=(1, 128), memory_space=pto.MemorySpace.UB),
            dst=pto.TileSpecialization(shape=(1, 128), memory_space=pto.MemorySpace.UB),
        )
        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope = next(stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt))
        pair_store = next(stmt for stmt in vecscope.body if isinstance(stmt, SemanticVectorPairStoreStmt))
        self.assertIsInstance(pair_store, SemanticVectorPairStoreStmt)

        text = specialized.mlir_text()
        self.assertIn("pto.vldsx2", text)
        self.assertIn("pto.vstsx2", text)
        self.assertIn('"DINTLV"', text)
        self.assertIn('"INTLV"', text)
        self.assertNotIn("DINTLV_B32", text)
        self.assertNotIn("INTLV_B32", text)

    def test_vldsx2_and_vstsx2_still_accept_legacy_string_tokens_for_compatibility(self) -> None:
        @pto.vkernel(op="vldsx2_vstsx2_legacy_tokens", dtypes=[(pto.f32, pto.f32)])
        def kernel(src: pto.Tile, dst: pto.Tile):
            mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            low, high = pto.vldsx2(src[0, 0:], "DINTLV_B32")
            pto.vstsx2(low, high, dst[0, 0:], "INTLV_B32", mask)
            return None

        specialized = kernel.specialize(
            src=pto.TileSpecialization(shape=(1, 128), memory_space=pto.MemorySpace.UB),
            dst=pto.TileSpecialization(shape=(1, 128), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn('"DINTLV"', text)
        self.assertIn('"INTLV"', text)

    def test_vscatter_lowers_from_advanced_pointer_surface(self) -> None:
        @pto.vkernel(
            op="vscatter_pointer_surface",
            dtypes=[(pto.i32, pto.f32)],
            advanced=True,
        )
        def kernel(
            offsets_src: pto.ptr(pto.i32, pto.MemorySpace.UB),
            dst: pto.ptr(pto.f32, pto.MemorySpace.UB),
        ):
            vec = pto.vbr(1.0)
            offsets = pto.vlds(offsets_src, 0)
            mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            pto.vscatter(vec, dst, offsets, mask)
            return None

        specialized = kernel.specialize()
        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope = next(stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt))
        scatter_stmt = next(stmt for stmt in vecscope.body if isinstance(stmt, SemanticVScatterStmt))

        self.assertIsInstance(scatter_stmt, SemanticVScatterStmt)
        self.assertEqual(scatter_stmt.destination.type.memory_space, "ub")
        self.assertEqual(scatter_stmt.value.type.element_dtype, pto.f32)
        self.assertEqual(scatter_stmt.offsets.type.element_dtype, pto.i32)
        self.assertEqual(scatter_stmt.mask.type.granularity, "b32")

        text = specialized.mlir_text()
        self.assertIn("pto.vscatter", text)
        self.assertIn("!pto.vreg<64xf32>", text)
        self.assertIn("!pto.vreg<64xi32>", text)
        self.assertIn("!pto.mask<b32>", text)

    def test_align_load_and_stateful_store_ops_lower_to_current_vpto_surface(self) -> None:
        @pto.vkernel(
            op="align_load_and_stateful_store_ops",
            dtypes=[(pto.f32, pto.f32)],
            advanced=True,
        )
        def kernel(
            src: pto.ptr(pto.f32, pto.MemorySpace.UB),
            dst: pto.ptr(pto.f32, pto.MemorySpace.UB),
        ):
            load_align = pto.vldas(src)
            vec, load_align = pto.vldus(src, load_align)
            store_align = pto.init_align()
            store_align = pto.vstus(store_align, 0, vec, dst)
            store_align = pto.vstur(store_align, vec, dst)
            pto.vstas(store_align, dst, 0)
            post_align = pto.vstur(pto.init_align(), vec, dst, pto.PostUpdateMode.POST_UPDATE)
            pto.vstar(post_align, dst)
            return None

        specialized = kernel.specialize()
        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope = next(stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt))
        align_store_stmts = [stmt for stmt in vecscope.body if isinstance(stmt, SemanticAlignStoreStmt)]

        self.assertTrue(any(isinstance(stmt, SemanticAssignStmt) and isinstance(stmt.value.type, SemanticAlignType) for stmt in vecscope.body))
        self.assertEqual(len(align_store_stmts), 2)
        self.assertEqual([stmt.op_name for stmt in align_store_stmts], ["vstas", "vstar"])

        text = specialized.mlir_text()
        self.assertIn("pto.vldas", text)
        self.assertIn("pto.vldus", text)
        self.assertIn("pto.init_align", text)
        self.assertIn("pto.vstus", text)
        self.assertIn("pto.vstur", text)
        self.assertIn("pto.vstas", text)
        self.assertIn("pto.vstar", text)
        self.assertIn('"POST_UPDATE"', text)
        self.assertIn('"NO_POST_UPDATE"', text)
        self.assertIn("!pto.align", text)

    def test_predicate_store_and_compatibility_store_sugar_lower_to_supported_ops(self) -> None:
        @pto.vkernel(
            op="predicate_store_and_store_sugar",
            dtypes=[(pto.f32, pto.ui32)],
            advanced=True,
        )
        def kernel(
            dst: pto.ptr(pto.f32, pto.MemorySpace.UB),
            mask_dst: pto.ptr(pto.ui32, pto.MemorySpace.UB),
        ):
            mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            pto.psts(mask, mask_dst, 0)
            align = pto.init_align()
            align, mask_base = pto.pstu(align, mask, mask_dst)
            pto.vsta(align, mask_base, 0)
            pto.vsst(1.0, dst, 0, mask)
            return None

        specialized = kernel.specialize()
        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope = next(stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt))

        self.assertTrue(any(isinstance(stmt, SemanticPredicateStoreStmt) for stmt in vecscope.body))
        self.assertTrue(any(isinstance(stmt, SemanticAlignStoreStmt) and stmt.op_name == "vstas" for stmt in vecscope.body))

        text = specialized.mlir_text()
        self.assertIn("pto.psts", text)
        self.assertIn('"NORM"', text)
        self.assertIn("pto.pstu", text)
        self.assertIn("pto.vbr", text)
        self.assertIn("pto.vsts", text)
        self.assertIn("pto.vstas", text)
        self.assertNotIn("pto.vsst", text)
        self.assertNotIn("pto.vsta ", text)

    def test_psts_rejects_tile_indexing_surface(self) -> None:
        @pto.vkernel(
            op="predicate_store_tile_indexing_reject",
            dtypes=[(pto.ui32,)],
            advanced=True,
        )
        def kernel(mask_dst: pto.Tile):
            mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            pto.psts(mask, mask_dst[0, 0:])
            return None

        specialized = kernel.specialize(
            mask_dst=pto.TileSpecialization(shape=(16, 64), memory_space=pto.MemorySpace.UB),
        )
        with self.assertRaises(TypeError) as ctx:
            analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertIn("does not support Tile element-indexing syntax", str(ctx.exception))
        self.assertIn("pto.psts(mask, buf, offset", str(ctx.exception))

    def test_plds_load_lower_to_supported_op(self) -> None:
        @pto.vkernel(
            op="predicate_load_from_ub_buffer",
            dtypes=[(pto.ui32, pto.ui32)],
            advanced=True,
        )
        def kernel(
            mask_src: pto.ptr(pto.ui32, pto.MemorySpace.UB),
            mask_dst: pto.ptr(pto.ui32, pto.MemorySpace.UB),
        ):
            mask = pto.plds(mask_src, 0)
            pto.psts(mask, mask_dst, 0)
            return None

        specialized = kernel.specialize()
        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope = next(stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt))
        load_assign = next(
            stmt
            for stmt in vecscope.body
            if isinstance(stmt, SemanticAssignStmt)
            and isinstance(stmt.value, SemanticCallExpr)
            and stmt.value.name == "plds"
        )
        self.assertIsInstance(load_assign.value.type, SemanticMaskType)
        self.assertEqual(load_assign.value.type.granularity, "b32")

        text = specialized.mlir_text()
        self.assertIn("pto.plds", text)
        self.assertIn('"NORM"', text)
        self.assertIn("pto.psts", text)

    def test_plds_rejects_unsupported_dist_token(self) -> None:
        with self.assertRaises(TypeError) as ctx:

            @pto.vkernel(
                op="predicate_load_invalid_dist",
                dtypes=[(pto.ui32,)],
                advanced=True,
            )
            def kernel(mask_src: pto.ptr(pto.ui32, pto.MemorySpace.UB)):
                _mask = pto.plds(mask_src, 0, pto.PredicateDist.PK)
                return None

            kernel.specialize().mlir_text()

        self.assertIn("predicate load dist must be one of", str(ctx.exception))
        self.assertIn("pto.PredicateDist.DS", str(ctx.exception))

    def test_predicate_generation_and_logic_families_lower_to_supported_ops(self) -> None:
        @pto.vkernel(
            op="predicate_generation_and_logic_families",
            dtypes=[(pto.ui32,)],
            advanced=True,
        )
        def kernel(mask_dst: pto.ptr(pto.ui32, pto.MemorySpace.UB)):
            mask8 = pto.pset_b8(pto.PAT.ALL)
            mask16 = pto.pge_b16(pto.PAT.VL16)
            mask32, _next = pto.plt_b32(64)
            and_mask = pto.pand(mask32, mask32, mask32)
            or_mask = pto.por(and_mask, mask32, mask32)
            xor_mask = pto.pxor(or_mask, mask32, mask32)
            pto.psts(xor_mask, mask_dst, 0)
            _ = mask8
            _ = mask16
            return None

        text = kernel.specialize().mlir_text()
        self.assertIn("pto.pset_b8", text)
        self.assertIn("pto.pge_b16", text)
        self.assertIn("pto.plt_b32", text)
        self.assertIn("pto.pand", text)
        self.assertIn("pto.por", text)
        self.assertIn("pto.pxor", text)

    def test_predicate_load_store_alias_and_immediate_forms_lower_to_supported_ops(self) -> None:
        @pto.vkernel(
            op="predicate_load_store_alias_and_immediate_forms",
            dtypes=[(pto.ui32, pto.ui32, pto.ui32, pto.si32)],
            advanced=True,
        )
        def kernel(
            mask_src: pto.ptr(pto.ui32, pto.MemorySpace.UB),
            mask_dst: pto.ptr(pto.ui32, pto.MemorySpace.UB),
            off_u: pto.ui32,
            off_s: pto.si32,
        ):
            mask0 = pto.pld(mask_src, 0, pto.PredicateDist.NORM)
            mask1 = pto.pldi(mask_src, pto.i32(off_u), pto.PredicateDist.US)
            pto.pst(mask0, mask_dst, 0)
            pto.psti(mask1, mask_dst, pto.i32(off_s), pto.PredicateDist.PK)
            return None

        text = kernel.specialize().mlir_text()
        self.assertIn("pto.plds", text)
        self.assertIn("pto.pldi", text)
        self.assertIn("pto.psts", text)
        self.assertIn("pto.psti", text)
        self.assertIn("builtin.unrealized_conversion_cast", text)
        self.assertIn("arith.index_cast", text)
        self.assertNotRegex(text, r"arith\.extsi %\w+ : si32 to i32")
        self.assertNotRegex(text, r"arith\.extui %\w+ : ui32 to i32")

    def test_predicate_reorder_families_lower_to_supported_ops(self) -> None:
        @pto.vkernel(
            op="predicate_reorder_families",
            dtypes=[(pto.ui32,)],
            advanced=True,
        )
        def kernel(mask_dst: pto.ptr(pto.ui32, pto.MemorySpace.UB)):
            mask8 = pto.pset_b8(pto.PAT.ALL)
            mask16 = pto.pset_b16(pto.PAT.ALL)
            mask32 = pto.pset_b32(pto.PAT.ALL)
            low8, high8 = pto.pdintlv_b8(mask8, mask8)
            low8i, high8i = pto.pintlv_b8(mask8, mask8)
            low16d, high16d = pto.pdintlv_b16(mask16, mask16)
            low16, high16 = pto.pintlv_b16(mask16, mask16)
            low32, high32 = pto.pdintlv_b32(mask32, mask32)
            low32i, high32i = pto.pintlv_b32(mask32, mask32)
            all32 = pto.make_mask(pto.ui32, pto.PAT.ALL)
            pto.psts(all32, mask_dst, 0)
            return None

        text = kernel.specialize().mlir_text()
        self.assertIn("pto.pdintlv_b8", text)
        self.assertIn("pto.pintlv_b8", text)
        self.assertIn("pto.pdintlv_b16", text)
        self.assertIn("pto.pintlv_b16", text)
        self.assertIn("pto.pdintlv_b32", text)
        self.assertIn("pto.pintlv_b32", text)

    def test_pdintlv_b8_rejects_wrong_mask_granularity(self) -> None:
        with self.assertRaises(TypeError) as ctx:

            @pto.vkernel(
                op="predicate_reorder_wrong_mask_granularity",
                dtypes=[(pto.ui32,)],
                advanced=True,
            )
            def kernel(mask_src: pto.ptr(pto.ui32, pto.MemorySpace.UB)):
                mask32 = pto.plds(mask_src, 0)
                _low, _high = pto.pdintlv_b8(mask32, mask32)
                return None

            kernel.specialize().mlir_text()

        self.assertIn("expects !pto.mask<b8> operands", str(ctx.exception))

    def test_strict_vecscope_rejects_implicit_capture_during_semantic_analysis(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16, pto.i32)], advanced=True)
        def kernel(inp: pto.TensorView, tile: pto.Tile, scale: pto.i32):
            with pto.strict_vecscope(inp, tile) as (vin, vtmp):
                leaked = scale
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        with self.assertRaises(ValueError) as ctx:
            analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertIn("implicit capture of 'scale' is not allowed", str(ctx.exception))


class TileLangDSLInlineProcTests(unittest.TestCase):
    @pto.inline_proc
    def _inline_copy_row(dst: pto.Tile, src: pto.Tile, lane: pto.i32):
        mask = pto.make_mask(pto.f32, pto.PAT.ALL)
        vec = pto.vlds(src, lane)
        pto.vsts(vec, dst, lane, mask)
        return None

    @pto.inline_proc
    def _inline_recur(dst: pto.Tile):
        _inline_recur(dst)
        return None

    @pto.inline_proc
    def _inline_capture(dst: pto.Tile):
        pto.vlds(dst, lane)
        return None

    @pto.inline_proc
    def _inline_capture_global_literal(dst: pto.Tile):
        mask = pto.make_mask(pto.f32, pto.PAT.ALL)
        vec = pto.vlds(dst, INLINE_PROC_GLOBAL_LANE)
        pto.vsts(vec, dst, INLINE_PROC_GLOBAL_LANE, mask)
        return None

    def test_inline_proc_exports_from_package_surface(self) -> None:
        self.assertTrue(hasattr(pto, "inline_proc"))
        self.assertTrue(hasattr(pto, "InlineProcDescriptor"))

    def test_inline_proc_call_keeps_call_in_frontend_and_mlir_text(self) -> None:
        @pto.vkernel(op="inline_proc_backend_call_unique", dtypes=[(pto.f32, pto.f32)])
        def kernel(dst: pto.Tile, src: pto.Tile):
            _inline_copy_row(dst, src, 0)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )

        frontend_kernel = build_frontend_kernel_node(specialized)
        self.assertEqual(len(frontend_kernel.body), 2)
        self.assertIsInstance(frontend_kernel.body[0], FrontendExprStmt)
        self.assertIsInstance(frontend_kernel.body[0].expr, FrontendCallExpr)
        self.assertEqual(frontend_kernel.body[0].expr.name, "_inline_copy_row")
        self.assertGreaterEqual(len(frontend_kernel.inline_procs), 1)
        self.assertIn("_inline_copy_row", {proc.name for proc in frontend_kernel.inline_procs})

        text = specialized.mlir_text()
        self.assertIn("func.call", text)
        self.assertIn("pto.tilelang.inline_proc", text)
        self.assertRegex(text, r"func\.call @__tl_inline_")

    def test_inline_proc_supports_default_parameters_and_keyword_call(self) -> None:
        @pto.inline_proc
        def inline_store(dst: pto.Tile, src: pto.Tile, lane: pto.i32 = 0):
            mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            vec = pto.vlds(src, lane)
            pto.vsts(vec, dst, lane, mask)
            return None

        @pto.vkernel(op="inline_proc_keyword_default_unique", dtypes=[(pto.f32, pto.f32)])
        def kernel(dst: pto.Tile, src: pto.Tile):
            inline_store(dst=dst, src=src)
            return None

        text = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        ).mlir_text()
        self.assertIn("func.call", text)
        self.assertIn("pto.tilelang.inline_proc", text)

    def test_inline_proc_supports_return_expression_in_expression_position(self) -> None:
        @pto.inline_proc
        def inline_load(src: pto.Tile, lane: pto.i32 = 0):
            return pto.vlds(src, lane)

        @pto.vkernel(op="inline_proc_expr_return_unique", dtypes=[(pto.f32, pto.f32)])
        def kernel(dst: pto.Tile, src: pto.Tile):
            mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            vec = inline_load(src)
            pto.vsts(vec, dst, 0, mask)
            return None

        text = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        ).mlir_text()
        self.assertIn("func.call", text)
        self.assertRegex(text, r"= func\.call @__tl_inline_")
        self.assertIn("pto.vsts", text)

    def test_vdiv_integer_vector_types_rewrite_to_internal_helper(self) -> None:
        @pto.vkernel(op="vdiv_i16_dtype_support_unique", dtypes=[(pto.i16, pto.i16)])
        def kernel_i16(dst: pto.Tile, src: pto.Tile):
            dtype = dst.element_type
            mask = pto.make_mask(dtype, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vdiv(vec, vec, mask)
            pto.vsts(out, dst, 0, mask)
            return None

        @pto.vkernel(op="vdiv_i32_dtype_support_unique", dtypes=[(pto.i32, pto.i32)])
        def kernel_i32(dst: pto.Tile, src: pto.Tile):
            dtype = dst.element_type
            mask = pto.make_mask(dtype, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vdiv(vec, vec, mask)
            pto.vsts(out, dst, 0, mask)
            return None

        specialized_i16 = kernel_i16.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )
        semantic_i16 = analyze_frontend_kernel(build_frontend_kernel_node(specialized_i16))
        assign_i16 = next(
            stmt
            for stmt in _walk_semantic_stmts(semantic_i16.body)
            if isinstance(stmt, SemanticAssignStmt)
            and len(stmt.targets) == 1
            and stmt.targets[0].name == "out"
            and isinstance(stmt.value, SemanticCallExpr)
        )
        self.assertIsNone(assign_i16.value.namespace)
        self.assertRegex(assign_i16.value.name, r"^__tl_inline__tl_soft_vdiv_")
        self.assertEqual(assign_i16.value.type, SemanticVRegType(element_dtype=pto.i16, lanes=128))
        self.assertGreaterEqual(len(semantic_i16.inline_helpers), 1)
        self.assertRegex(specialized_i16.mlir_text(), r"func\.call @__tl_inline__tl_soft_vdiv_")

        text_i32 = kernel_i32.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )
        semantic_i32 = analyze_frontend_kernel(build_frontend_kernel_node(text_i32))
        assign_i32 = next(
            stmt
            for stmt in _walk_semantic_stmts(semantic_i32.body)
            if isinstance(stmt, SemanticAssignStmt)
            and len(stmt.targets) == 1
            and stmt.targets[0].name == "out"
            and isinstance(stmt.value, SemanticCallExpr)
        )
        self.assertIsNone(assign_i32.value.namespace)
        self.assertRegex(assign_i32.value.name, r"^__tl_inline__tl_soft_vdiv_")
        self.assertEqual(assign_i32.value.type, SemanticVRegType(element_dtype=pto.i32, lanes=64))
        self.assertGreaterEqual(len(semantic_i32.inline_helpers), 1)
        self.assertRegex(text_i32.mlir_text(), r"func\.call @__tl_inline__tl_soft_vdiv_")

    def test_vdiv_f16_and_f32_vector_types_keep_authoring_form_vpto_path(self) -> None:
        @pto.vkernel(
            op="vdiv_float_dtype_support_unique",
            dtypes=[(pto.f16, pto.f16), (pto.f32, pto.f32)],
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            dtype = dst.element_type
            mask = pto.make_mask(dtype, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vdiv(vec, vec, mask)
            pto.vsts(out, dst, 0, mask)
            return None

        cases = [
            (pto.f16, 128),
            (pto.f32, 64),
        ]

        for dtype, lanes in cases:
            with self.subTest(dtype=dtype):
                selected = pto.select_kernel("a5", "vdiv_float_dtype_support_unique", (dtype, dtype))
                specialized = selected.specialize(
                    dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
                    src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
                )
                semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))

                assign_stmt = next(
                    stmt
                    for stmt in _walk_semantic_stmts(semantic_kernel.body)
                    if isinstance(stmt, SemanticAssignStmt)
                    and len(stmt.targets) == 1
                    and stmt.targets[0].name == "out"
                    and isinstance(stmt.value, SemanticCallExpr)
                )
                self.assertEqual(assign_stmt.value.namespace, "pto")
                self.assertEqual(assign_stmt.value.name, "vdiv")
                self.assertEqual(
                    assign_stmt.value.type,
                    SemanticVRegType(element_dtype=dtype, lanes=lanes),
                )
                self.assertEqual(len(semantic_kernel.inline_helpers), 0)

                text = lower_semantic_kernel(semantic_kernel).render()
                self.assertEqual(text, specialized.mlir_text())
                self.assertIn("= pto.vdiv ", text)
                self.assertNotIn("__tl_inline__tl_soft_vdiv_", text)

    def test_vdiv_i8_and_ui8_vector_types_rewrite_to_internal_helper(self) -> None:
        @pto.vkernel(op="vdiv_i8_dtype_support_unique", dtypes=[(pto.i8, pto.i8)])
        def kernel_i8(dst: pto.Tile, src: pto.Tile):
            mask = pto.make_mask(pto.i8, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vdiv(vec, vec, mask)
            pto.vsts(out, dst, 0, mask)
            return None

        @pto.vkernel(op="vdiv_ui8_dtype_support_unique", dtypes=[(pto.ui8, pto.ui8)])
        def kernel_ui8(dst: pto.Tile, src: pto.Tile):
            mask = pto.make_mask(pto.ui8, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vdiv(vec, vec, mask)
            pto.vsts(out, dst, 0, mask)
            return None

        specialized_i8 = kernel_i8.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )
        semantic_i8 = analyze_frontend_kernel(build_frontend_kernel_node(specialized_i8))
        assign_i8 = next(
            stmt
            for stmt in _walk_semantic_stmts(semantic_i8.body)
            if isinstance(stmt, SemanticAssignStmt)
            and len(stmt.targets) == 1
            and stmt.targets[0].name == "out"
            and isinstance(stmt.value, SemanticCallExpr)
        )
        self.assertIsNone(assign_i8.value.namespace)
        self.assertRegex(assign_i8.value.name, r"^__tl_inline__tl_soft_vdiv_")
        self.assertEqual(assign_i8.value.type, SemanticVRegType(element_dtype=pto.i8, lanes=256))
        self.assertGreaterEqual(len(semantic_i8.inline_helpers), 1)
        self.assertRegex(specialized_i8.mlir_text(), r"func\.call @__tl_inline__tl_soft_vdiv_")

        specialized_ui8 = kernel_ui8.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )
        semantic_ui8 = analyze_frontend_kernel(build_frontend_kernel_node(specialized_ui8))
        assign_ui8 = next(
            stmt
            for stmt in _walk_semantic_stmts(semantic_ui8.body)
            if isinstance(stmt, SemanticAssignStmt)
            and len(stmt.targets) == 1
            and stmt.targets[0].name == "out"
            and isinstance(stmt.value, SemanticCallExpr)
        )
        self.assertIsNone(assign_ui8.value.namespace)
        self.assertRegex(assign_ui8.value.name, r"^__tl_inline__tl_soft_vdiv_")
        self.assertEqual(assign_ui8.value.type, SemanticVRegType(element_dtype=pto.ui8, lanes=256))
        self.assertGreaterEqual(len(semantic_ui8.inline_helpers), 1)
        self.assertRegex(specialized_ui8.mlir_text(), r"func\.call @__tl_inline__tl_soft_vdiv_")

    def test_vdiv_rejects_bf16_vector_type(self) -> None:
        @pto.vkernel(op="vdiv_bf16_reject_unique", dtypes=[(pto.bf16, pto.bf16)])
        def kernel(dst: pto.Tile, src: pto.Tile):
            mask = pto.make_mask(pto.bf16, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vdiv(vec, vec, mask)
            pto.vsts(out, dst, 0, mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )

        with self.assertRaises(TypeError) as ctx:
            specialized.mlir_text()

        self.assertIn(
            "pto.vdiv only supports 8/16/32-bit integer families and f16/f32 in TileLang DSL v1",
            str(ctx.exception),
        )

    def test_vmod_integer_vector_types_rewrite_to_internal_helper(self) -> None:
        @pto.vkernel(op="vmod_i16_dtype_support_unique", dtypes=[(pto.i16, pto.i16)])
        def kernel_i16(dst: pto.Tile, src: pto.Tile):
            dtype = dst.element_type
            mask = pto.make_mask(dtype, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vmod(vec, vec, mask)
            pto.vsts(out, dst, 0, mask)
            return None

        @pto.vkernel(op="vmod_i32_dtype_support_unique", dtypes=[(pto.i32, pto.i32)])
        def kernel_i32(dst: pto.Tile, src: pto.Tile):
            dtype = dst.element_type
            mask = pto.make_mask(dtype, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vmod(vec, vec, mask)
            pto.vsts(out, dst, 0, mask)
            return None

        specialized_i16 = kernel_i16.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )
        semantic_i16 = analyze_frontend_kernel(build_frontend_kernel_node(specialized_i16))
        assign_i16 = next(
            stmt
            for stmt in _walk_semantic_stmts(semantic_i16.body)
            if isinstance(stmt, SemanticAssignStmt)
            and len(stmt.targets) == 1
            and stmt.targets[0].name == "out"
            and isinstance(stmt.value, SemanticCallExpr)
        )
        self.assertIsNone(assign_i16.value.namespace)
        self.assertRegex(assign_i16.value.name, r"^__tl_inline__tl_soft_vmod_")
        self.assertEqual(assign_i16.value.type, SemanticVRegType(element_dtype=pto.i16, lanes=128))
        self.assertGreaterEqual(len(semantic_i16.inline_helpers), 1)
        self.assertRegex(specialized_i16.mlir_text(), r"func\.call @__tl_inline__tl_soft_vmod_")

        specialized_i32 = kernel_i32.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )
        semantic_i32 = analyze_frontend_kernel(build_frontend_kernel_node(specialized_i32))
        assign_i32 = next(
            stmt
            for stmt in _walk_semantic_stmts(semantic_i32.body)
            if isinstance(stmt, SemanticAssignStmt)
            and len(stmt.targets) == 1
            and stmt.targets[0].name == "out"
            and isinstance(stmt.value, SemanticCallExpr)
        )
        self.assertIsNone(assign_i32.value.namespace)
        self.assertRegex(assign_i32.value.name, r"^__tl_inline__tl_soft_vmod_")
        self.assertEqual(assign_i32.value.type, SemanticVRegType(element_dtype=pto.i32, lanes=64))
        self.assertGreaterEqual(len(semantic_i32.inline_helpers), 1)
        self.assertRegex(specialized_i32.mlir_text(), r"func\.call @__tl_inline__tl_soft_vmod_")

    def test_vmod_i8_and_ui8_vector_types_rewrite_to_internal_helper(self) -> None:
        @pto.vkernel(op="vmod_i8_dtype_support_unique", dtypes=[(pto.i8, pto.i8)])
        def kernel_i8(dst: pto.Tile, src: pto.Tile):
            mask = pto.make_mask(pto.i8, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vmod(vec, vec, mask)
            pto.vsts(out, dst, 0, mask)
            return None

        @pto.vkernel(op="vmod_ui8_dtype_support_unique", dtypes=[(pto.ui8, pto.ui8)])
        def kernel_ui8(dst: pto.Tile, src: pto.Tile):
            mask = pto.make_mask(pto.ui8, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vmod(vec, vec, mask)
            pto.vsts(out, dst, 0, mask)
            return None

        specialized_i8 = kernel_i8.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )
        semantic_i8 = analyze_frontend_kernel(build_frontend_kernel_node(specialized_i8))
        assign_i8 = next(
            stmt
            for stmt in _walk_semantic_stmts(semantic_i8.body)
            if isinstance(stmt, SemanticAssignStmt)
            and len(stmt.targets) == 1
            and stmt.targets[0].name == "out"
            and isinstance(stmt.value, SemanticCallExpr)
        )
        self.assertIsNone(assign_i8.value.namespace)
        self.assertRegex(assign_i8.value.name, r"^__tl_inline__tl_soft_vmod_")
        self.assertEqual(assign_i8.value.type, SemanticVRegType(element_dtype=pto.i8, lanes=256))
        self.assertGreaterEqual(len(semantic_i8.inline_helpers), 1)
        self.assertRegex(specialized_i8.mlir_text(), r"func\.call @__tl_inline__tl_soft_vmod_")

        specialized_ui8 = kernel_ui8.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )
        semantic_ui8 = analyze_frontend_kernel(build_frontend_kernel_node(specialized_ui8))
        assign_ui8 = next(
            stmt
            for stmt in _walk_semantic_stmts(semantic_ui8.body)
            if isinstance(stmt, SemanticAssignStmt)
            and len(stmt.targets) == 1
            and stmt.targets[0].name == "out"
            and isinstance(stmt.value, SemanticCallExpr)
        )
        self.assertIsNone(assign_ui8.value.namespace)
        self.assertRegex(assign_ui8.value.name, r"^__tl_inline__tl_soft_vmod_")
        self.assertEqual(assign_ui8.value.type, SemanticVRegType(element_dtype=pto.ui8, lanes=256))
        self.assertGreaterEqual(len(semantic_ui8.inline_helpers), 1)
        self.assertRegex(specialized_ui8.mlir_text(), r"func\.call @__tl_inline__tl_soft_vmod_")

    def test_vmod_rejects_f32_vector_type(self) -> None:
        @pto.vkernel(op="vmod_f32_reject_unique", dtypes=[(pto.f32, pto.f32)])
        def kernel(dst: pto.Tile, src: pto.Tile):
            mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vmod(vec, vec, mask)
            pto.vsts(out, dst, 0, mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )

        with self.assertRaises(TypeError) as ctx:
            specialized.mlir_text()

        self.assertIn(
            "pto.vmod only supports 8/16/32-bit integer families in TileLang DSL v1",
            str(ctx.exception),
        )

    def test_integer_divmod_helpers_lock_zero_divisor_sentinel_convention(self) -> None:
        @pto.vkernel(
            op="integer_divmod_zero_divisor_contract_unique",
            dtypes=[
                (pto.i8, pto.i8),
                (pto.ui8, pto.ui8),
                (pto.i16, pto.i16),
                (pto.ui16, pto.ui16),
                (pto.i32, pto.i32),
                (pto.ui32, pto.ui32),
            ],
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            dtype = dst.element_type
            mask = pto.make_mask(dtype, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            quot = pto.vdiv(vec, vec, mask)
            rem = pto.vmod(vec, vec, mask)
            pto.vsts(quot, dst, 0, mask)
            pto.vsts(rem, dst, 0, mask)
            return None

        cases = [
            ("vdiv", pto.i8, "__tl_inline__tl_soft_vdiv_i8_", -1),
            ("vdiv", pto.ui8, "__tl_inline__tl_soft_vdiv_u8_", 0xFF),
            ("vdiv", pto.i16, "__tl_inline__tl_soft_vdiv_i16_", -1),
            ("vdiv", pto.ui16, "__tl_inline__tl_soft_vdiv_u16_", 0xFFFF),
            ("vdiv", pto.i32, "__tl_inline__tl_soft_vdiv_i32_", -1),
            ("vdiv", pto.ui32, "__tl_inline__tl_soft_vdiv_u32_", 0xFFFFFFFF),
            ("vmod", pto.i8, "__tl_inline__tl_soft_vmod_i8_", -1),
            ("vmod", pto.ui8, "__tl_inline__tl_soft_vmod_u8_", 0xFF),
            ("vmod", pto.i16, "__tl_inline__tl_soft_vmod_i16_", -1),
            ("vmod", pto.ui16, "__tl_inline__tl_soft_vmod_u16_", 0xFFFF),
            ("vmod", pto.i32, "__tl_inline__tl_soft_vmod_i32_", -1),
            ("vmod", pto.ui32, "__tl_inline__tl_soft_vmod_u32_", 0xFFFFFFFF),
        ]

        for op_name, dtype, helper_prefix, expected_sentinel in cases:
            with self.subTest(op=op_name, dtype=dtype):
                selected = pto.select_kernel(
                    "a5",
                    "integer_divmod_zero_divisor_contract_unique",
                    (dtype, dtype),
                )
                specialized = selected.specialize(
                    dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
                    src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
                )
                semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
                helper = _find_inline_helper(semantic_kernel, helper_prefix)

                zero_mask_assign = _find_last_helper_assign_by_name(helper, "zero_mask")
                self.assertIsInstance(zero_mask_assign.value, SemanticCallExpr)
                self.assertEqual(zero_mask_assign.value.namespace, "pto")
                self.assertEqual(zero_mask_assign.value.name, "vcmps")
                self.assertEqual(zero_mask_assign.value.args[3].value, "eq")
                self.assertEqual(
                    _resolve_helper_broadcast_scalar_literal(helper, zero_mask_assign.value.args[1]),
                    0,
                )

                return_stmt = _find_helper_return_stmt(helper)
                self.assertIsInstance(return_stmt.value, SemanticCallExpr)
                self.assertEqual(return_stmt.value.namespace, "pto")
                self.assertEqual(return_stmt.value.name, "vsel")
                self.assertIsInstance(return_stmt.value.args[2], SemanticBindingRef)
                self.assertEqual(return_stmt.value.args[2].binding.name, "zero_mask")
                self.assertEqual(
                    _resolve_helper_broadcast_scalar_literal(helper, return_stmt.value.args[0]),
                    expected_sentinel,
                )

    def test_signed_vdiv_helpers_derive_result_sign_from_operand_signs(self) -> None:
        @pto.vkernel(
            op="signed_vdiv_sign_contract_unique",
            dtypes=[(pto.i16, pto.i16), (pto.i32, pto.i32)],
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            dtype = dst.element_type
            mask = pto.make_mask(dtype, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vdiv(vec, vec, mask)
            pto.vsts(out, dst, 0, mask)
            return None

        cases = [
            (pto.i16, "__tl_inline__tl_soft_vdiv_i16_", "i16"),
            (pto.i32, "__tl_inline__tl_soft_vdiv_i32_", "i32"),
        ]

        for dtype, helper_prefix, dtype_name in cases:
            with self.subTest(dtype=dtype):
                selected = pto.select_kernel("a5", "signed_vdiv_sign_contract_unique", (dtype, dtype))
                specialized = selected.specialize(
                    dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
                    src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
                )
                semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
                helper = _find_inline_helper(semantic_kernel, helper_prefix)

                xor_assign = _find_last_helper_assign_by_name(helper, "x_xor_y")
                self.assertIsInstance(xor_assign.value, SemanticCallExpr)
                self.assertEqual(xor_assign.value.namespace, "pto")
                self.assertEqual(xor_assign.value.name, "vxor")
                self.assertEqual(xor_assign.value.args[0].binding.name, "vec")
                self.assertEqual(xor_assign.value.args[1].binding.name, "scalar_vec")
                self.assertEqual(xor_assign.value.args[2].binding.name, "active_mask")

                p_pos_assign = _find_last_helper_assign_by_name(helper, "p_pos")
                self.assertIsInstance(p_pos_assign.value, SemanticCallExpr)
                self.assertEqual(p_pos_assign.value.namespace, "pto")
                self.assertEqual(p_pos_assign.value.name, "vcmps")
                self.assertEqual(p_pos_assign.value.args[0].binding.name, "x_xor_y")
                self.assertEqual(p_pos_assign.value.args[1].binding.name, "zero")
                self.assertEqual(p_pos_assign.value.args[2].binding.name, "active_mask")
                self.assertEqual(p_pos_assign.value.args[3].value, "ge")

                q_assign = _find_last_helper_assign_by_name(helper, "q")
                self.assertIsInstance(q_assign.value, SemanticCallExpr)
                self.assertEqual(q_assign.value.namespace, "pto")
                self.assertEqual(q_assign.value.name, "vsel")
                self.assertEqual(q_assign.value.args[1].binding.name, "neg_q")
                self.assertEqual(q_assign.value.args[2].binding.name, "p_pos")
                self.assertIsInstance(q_assign.value.args[0], SemanticCallExpr)
                self.assertEqual(q_assign.value.args[0].namespace, "pto")
                self.assertEqual(q_assign.value.args[0].name, "vbitcast")
                self.assertIsInstance(q_assign.value.args[0].args[0], SemanticBindingRef)
                self.assertEqual(q_assign.value.args[0].args[1].name, dtype_name)

    def test_signed_vmod_helpers_apply_floor_fix_when_signs_differ(self) -> None:
        @pto.vkernel(
            op="signed_vmod_sign_contract_unique",
            dtypes=[(pto.i16, pto.i16), (pto.i32, pto.i32)],
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            dtype = dst.element_type
            mask = pto.make_mask(dtype, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vmod(vec, vec, mask)
            pto.vsts(out, dst, 0, mask)
            return None

        cases = [
            (pto.i16, "__tl_inline__tl_soft_vmod_i16_"),
            (pto.i32, "__tl_inline__tl_soft_vmod_i32_"),
        ]

        for dtype, helper_prefix in cases:
            with self.subTest(dtype=dtype):
                selected = pto.select_kernel("a5", "signed_vmod_sign_contract_unique", (dtype, dtype))
                specialized = selected.specialize(
                    dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
                    src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
                )
                semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
                helper = _find_inline_helper(semantic_kernel, helper_prefix)

                nonzero_assign = _find_last_helper_assign_by_name(helper, "nonzero_remainder")
                self.assertIsInstance(nonzero_assign.value, SemanticCallExpr)
                self.assertEqual(nonzero_assign.value.namespace, "pto")
                self.assertEqual(nonzero_assign.value.name, "vcmps")
                self.assertEqual(nonzero_assign.value.args[1].binding.name, "zero")
                self.assertEqual(nonzero_assign.value.args[2].binding.name, "active_mask")
                self.assertEqual(nonzero_assign.value.args[3].value, "ne")

                sign_diff_assign = _find_last_helper_assign_by_name(helper, "sign_diff")
                self.assertIsInstance(sign_diff_assign.value, SemanticCallExpr)
                self.assertEqual(sign_diff_assign.value.namespace, "pto")
                self.assertEqual(sign_diff_assign.value.name, "pxor")
                self.assertEqual(sign_diff_assign.value.args[0].binding.name, "sign_x")
                self.assertEqual(sign_diff_assign.value.args[1].binding.name, "sign_y")
                self.assertEqual(sign_diff_assign.value.args[2].binding.name, "active_mask")

                need_fix_assign = _find_last_helper_assign_by_name(helper, "need_floor_fix")
                self.assertIsInstance(need_fix_assign.value, SemanticCallExpr)
                self.assertEqual(need_fix_assign.value.namespace, "pto")
                self.assertEqual(need_fix_assign.value.name, "pand")
                self.assertEqual(need_fix_assign.value.args[0].binding.name, "sign_diff")
                self.assertEqual(need_fix_assign.value.args[1].binding.name, "nonzero_remainder")
                self.assertEqual(need_fix_assign.value.args[2].binding.name, "active_mask")

                amended_assign = _find_last_helper_assign_by_name(helper, "amended_remainder")
                self.assertIsInstance(amended_assign.value, SemanticCallExpr)
                self.assertEqual(amended_assign.value.namespace, "pto")
                self.assertEqual(amended_assign.value.name, "vadd")
                self.assertEqual(amended_assign.value.args[0].binding.name, "scalar_vec")
                self.assertEqual(amended_assign.value.args[1].binding.name, "remainder")
                self.assertEqual(amended_assign.value.args[2].binding.name, "active_mask")

                remainder_assign = _find_last_helper_assign_by_name(helper, "remainder")
                self.assertIsInstance(remainder_assign.value, SemanticCallExpr)
                self.assertEqual(remainder_assign.value.namespace, "pto")
                self.assertEqual(remainder_assign.value.name, "vsel")
                self.assertEqual(remainder_assign.value.args[0].binding.name, "amended_remainder")
                self.assertEqual(remainder_assign.value.args[2].binding.name, "need_floor_fix")

    def test_i8_divmod_helpers_use_explicit_widen_narrow_profile(self) -> None:
        @pto.vkernel(
            op="i8_divmod_widen_narrow_contract_unique",
            dtypes=[
                (pto.i8, pto.i8),
                (pto.ui8, pto.ui8),
            ],
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            dtype = dst.element_type
            mask = pto.make_mask(dtype, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            quot = pto.vdiv(vec, vec, mask)
            rem = pto.vmod(vec, vec, mask)
            pto.vsts(quot, dst, 0, mask)
            pto.vsts(rem, dst, 1, mask)
            return None

        cases = [
            (
                pto.i8,
                "__tl_inline__tl_soft_vdiv_i8_",
                "__tl_inline__tl_soft_vdiv_i16_",
                "vsunpack",
                "q",
                "q_low",
                "q_high",
                "vbitcast",
            ),
            (
                pto.ui8,
                "__tl_inline__tl_soft_vdiv_u8_",
                "__tl_inline__tl_soft_vdiv_u16_",
                "vzunpack",
                "q",
                "q_low",
                "q_high",
                "vor",
            ),
            (
                pto.i8,
                "__tl_inline__tl_soft_vmod_i8_",
                "__tl_inline__tl_soft_vmod_i16_",
                "vsunpack",
                "r",
                "r_low",
                "r_high",
                "vbitcast",
            ),
            (
                pto.ui8,
                "__tl_inline__tl_soft_vmod_u8_",
                "__tl_inline__tl_soft_vmod_u16_",
                "vzunpack",
                "r",
                "r_low",
                "r_high",
                "vor",
            ),
        ]

        for (
            dtype,
            helper_prefix,
            widened_helper_prefix,
            unpack_name,
            packed_result_name,
            lower_result_name,
            higher_result_name,
            packed_result_op,
        ) in cases:
            with self.subTest(dtype=dtype, helper=helper_prefix):
                selected = pto.select_kernel(
                    "a5",
                    "i8_divmod_widen_narrow_contract_unique",
                    (dtype, dtype),
                )
                specialized = selected.specialize(
                    dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
                    src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
                )
                semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
                helper = _find_inline_helper(semantic_kernel, helper_prefix)

                active_low_assign = _find_last_helper_assign_by_name(helper, "active_low")
                self.assertIsInstance(active_low_assign.value, SemanticCallExpr)
                self.assertEqual(active_low_assign.value.namespace, "pto")
                self.assertEqual(active_low_assign.value.name, "punpack")
                self.assertEqual(active_low_assign.value.args[0].binding.name, "active_mask")

                active_high_assign = _find_last_helper_assign_by_name(helper, "active_high")
                self.assertIsInstance(active_high_assign.value, SemanticCallExpr)
                self.assertEqual(active_high_assign.value.namespace, "pto")
                self.assertEqual(active_high_assign.value.name, "punpack")
                self.assertEqual(active_high_assign.value.args[0].binding.name, "active_mask")

                for name, expected_half in (
                    ("vec_low", 0),
                    ("vec_high", 1),
                    ("scalar_low", 0),
                    ("scalar_high", 1),
                ):
                    assign = _find_last_helper_assign_by_name(helper, name)
                    self.assertIsInstance(assign.value, SemanticCallExpr)
                    self.assertEqual(assign.value.namespace, "pto")
                    self.assertEqual(assign.value.name, unpack_name)
                    self.assertEqual(assign.value.args[1].value, expected_half)

                lower_assign = _find_last_helper_assign_by_name(helper, lower_result_name)
                self.assertIsInstance(lower_assign.value, SemanticCallExpr)
                self.assertIsNone(lower_assign.value.namespace)
                self.assertRegex(lower_assign.value.name, rf"^{widened_helper_prefix}")
                self.assertEqual(lower_assign.value.args[2].binding.name, "active_low")

                higher_assign = _find_last_helper_assign_by_name(helper, higher_result_name)
                self.assertIsInstance(higher_assign.value, SemanticCallExpr)
                self.assertIsNone(higher_assign.value.namespace)
                self.assertRegex(higher_assign.value.name, rf"^{widened_helper_prefix}")
                self.assertEqual(higher_assign.value.args[2].binding.name, "active_high")

                packed_low_assign = _find_last_helper_assign_by_name(helper, "packed_low")
                self.assertIsInstance(packed_low_assign.value, SemanticCallExpr)
                self.assertEqual(packed_low_assign.value.namespace, "pto")
                self.assertEqual(packed_low_assign.value.name, "vpack")
                self.assertEqual(packed_low_assign.value.args[0].binding.name, lower_result_name)

                packed_high_assign = _find_last_helper_assign_by_name(helper, "packed_high")
                self.assertIsInstance(packed_high_assign.value, SemanticCallExpr)
                self.assertEqual(packed_high_assign.value.namespace, "pto")
                self.assertEqual(packed_high_assign.value.name, "vpack")
                self.assertEqual(packed_high_assign.value.args[0].binding.name, higher_result_name)

                packed_result_assign = _find_last_helper_assign_by_name(helper, packed_result_name)
                self.assertIsInstance(packed_result_assign.value, SemanticCallExpr)
                self.assertEqual(packed_result_assign.value.namespace, "pto")
                self.assertEqual(packed_result_assign.value.name, packed_result_op)
                if packed_result_op == "vor":
                    combined_expr = packed_result_assign.value
                else:
                    self.assertIsInstance(packed_result_assign.value.args[0], SemanticCallExpr)
                    combined_expr = packed_result_assign.value.args[0]
                    self.assertEqual(combined_expr.namespace, "pto")
                    self.assertEqual(combined_expr.name, "vor")
                self.assertEqual(combined_expr.args[0].binding.name, "packed_low")
                self.assertEqual(combined_expr.args[1].binding.name, "packed_high")
                self.assertEqual(combined_expr.args[2].binding.name, "full_mask_b8")

    def test_integer_divmod_rewrite_uses_injected_internal_helpers(self) -> None:
        @pto.vkernel(op="divmod_internal_helper_injection_unique", dtypes=[(pto.i32, pto.i32)])
        def kernel(dst: pto.Tile, src: pto.Tile):
            mask = pto.make_mask(pto.i32, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            quot = pto.vdiv(vec, vec, mask)
            rem = pto.vmod(vec, vec, mask)
            pto.vsts(quot, dst, 0, mask)
            pto.vsts(rem, dst, 1, mask)
            return None

        self.assertNotIn("vdiv", kernel.py_fn.__globals__)
        self.assertNotIn("vmod", kernel.py_fn.__globals__)
        self.assertNotIn("_tl_soft_vdiv_i32", kernel.inline_procs)
        self.assertNotIn("_tl_soft_vmod_i32", kernel.inline_procs)

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )
        frontend_kernel = build_frontend_kernel_node(specialized)

        self.assertEqual({proc.name for proc in frontend_kernel.inline_procs}, set())
        internal_names = {proc.name for proc in frontend_kernel.internal_inline_procs}
        self.assertIn("_tl_soft_vdiv", internal_names)
        self.assertIn("_tl_soft_vmod", internal_names)
        self.assertIn("_tl_soft_vdiv_i32", internal_names)
        self.assertIn("_tl_soft_vmod_i32", internal_names)

        semantic_kernel = analyze_frontend_kernel(frontend_kernel)
        helper_symbols = {helper.symbol_name for helper in semantic_kernel.inline_helpers}
        self.assertTrue(any(name.startswith("__tl_inline__tl_soft_vdiv_") for name in helper_symbols))
        self.assertTrue(any(name.startswith("__tl_inline__tl_soft_vmod_") for name in helper_symbols))

    def test_internal_vdiv_helper_name_is_not_public_surface(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="internal_vdiv_helper_reject_unique", dtypes=[(pto.i32, pto.i32)])
            def kernel(dst: pto.Tile, src: pto.Tile):
                mask = pto.make_mask(pto.i32, pto.PAT.ALL)
                vec = pto.vlds(src, 0)
                out = _tl_soft_vdiv_i32(vec, vec, mask)
                pto.vsts(out, dst, 0, mask)
                return None

        self.assertIn(
            "arbitrary external call `_tl_soft_vdiv_i32` is not supported in TileLang DSL v1",
            str(ctx.exception),
        )

    def test_internal_vmod_helper_name_is_not_public_surface(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="internal_vmod_helper_reject_unique", dtypes=[(pto.i32, pto.i32)])
            def kernel(dst: pto.Tile, src: pto.Tile):
                mask = pto.make_mask(pto.i32, pto.PAT.ALL)
                vec = pto.vlds(src, 0)
                out = _tl_soft_vmod_i32(vec, vec, mask)
                pto.vsts(out, dst, 0, mask)
                return None

        self.assertIn(
            "arbitrary external call `_tl_soft_vmod_i32` is not supported in TileLang DSL v1",
            str(ctx.exception),
        )

    def test_inline_proc_and_pto_surface_can_share_basename(self) -> None:
        @pto.inline_proc
        def vdiv(src: pto.Tile, lane: pto.i32 = 0):
            return pto.vlds(src, lane)

        @pto.vkernel(op="inline_proc_same_basename_as_pto_surface_unique",
                     dtypes=[(pto.f32, pto.f32)])
        def kernel(dst: pto.Tile, src: pto.Tile):
            mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            helper_vec = vdiv(src, 0)
            raw_vec = pto.vdiv(helper_vec, helper_vec, mask)
            pto.vsts(raw_vec, dst, 0, mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )

        frontend_kernel = build_frontend_kernel_node(specialized)
        self.assertGreaterEqual(len(frontend_kernel.inline_procs), 1)
        self.assertIn("vdiv", {proc.name for proc in frontend_kernel.inline_procs})
        call_values = [
            stmt.value
            for stmt in frontend_kernel.body
            if isinstance(stmt, FrontendAssignStmt)
            and isinstance(stmt.value, FrontendCallExpr)
        ]
        helper_call = next(
            value for value in call_values if value.namespace is None and value.name == "vdiv"
        )
        raw_call = next(
            value for value in call_values if value.namespace == "pto" and value.name == "vdiv"
        )
        self.assertEqual(len(helper_call.args), 2)
        self.assertEqual(len(raw_call.args), 3)
        self.assertIsInstance(raw_call, FrontendCallExpr)
        self.assertIsNone(helper_call.namespace)
        self.assertEqual(helper_call.name, "vdiv")
        self.assertEqual(raw_call.namespace, "pto")
        self.assertEqual(raw_call.name, "vdiv")

        text = specialized.mlir_text()
        self.assertIn("pto.tilelang.inline_proc", text)
        self.assertRegex(text, r"func\.call @__tl_inline_vdiv_")
        self.assertIn("= pto.vdiv ", text)

    def test_inline_proc_rejects_non_trailing_return(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.inline_proc
            def bad_inline(flag: pto.i32):
                if flag:
                    return flag
                return flag

        self.assertIn("optional trailing `return`", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_inline_proc_rejects_recursive_calls(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="inline_proc_recursive_unique", dtypes=[(pto.f32,)])
            def kernel(dst: pto.Tile):
                _inline_recur(dst)
                return None

            kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB)
            ).mlir_text()

        self.assertIn("recursive inline_proc call `_inline_recur` is not supported", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_inline_proc_rejects_implicit_capture(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="inline_proc_capture_unique", dtypes=[(pto.f32,)])
            def kernel(dst: pto.Tile):
                lane = pto.i32(0)
                _inline_capture(dst)
                return None

            kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB)
            ).mlir_text()

        self.assertIn("implicit capture of 'lane' is not allowed in inline_proc", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_inline_proc_allows_module_level_literal_capture(self) -> None:
        @pto.vkernel(op="inline_proc_global_literal_capture_unique", dtypes=[(pto.f32,)])
        def kernel(dst: pto.Tile):
            _inline_capture_global_literal(dst)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB)
        )

        frontend_kernel = build_frontend_kernel_node(specialized)
        self.assertIn(
            "_inline_capture_global_literal",
            {proc.name for proc in frontend_kernel.inline_procs},
        )

        text = specialized.mlir_text()
        self.assertIn("func.call", text)
        self.assertIn("arith.constant 0 : index", text)

    def test_inline_proc_rejects_kw_only_vararg_and_kwargs(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as kw_only_ctx:

            @pto.inline_proc
            def bad_kw_only(dst: pto.Tile, *, lane: pto.i32):
                return None

        self.assertIn("keyword-only parameters", str(kw_only_ctx.exception))

        with self.assertRaises(pto.TileLangFrontendError) as vararg_ctx:

            @pto.inline_proc
            def bad_vararg(dst: pto.Tile, *lanes: pto.i32):
                return None

        self.assertIn("does not support *args", str(vararg_ctx.exception))

        with self.assertRaises(pto.TileLangFrontendError) as kwargs_ctx:

            @pto.inline_proc
            def bad_kwargs(dst: pto.Tile, **opts: pto.i32):
                return None

        self.assertIn("does not support **kwargs", str(kwargs_ctx.exception))

    def test_inline_proc_rejects_invalid_keyword_binding(self) -> None:
        @pto.inline_proc
        def inline_store(dst: pto.Tile, src: pto.Tile):
            return None

        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="inline_proc_invalid_keyword_unique", dtypes=[(pto.f32, pto.f32)])
            def kernel(dst: pto.Tile, src: pto.Tile):
                inline_store(dst=dst, src=src, lane=0)
                return None

        self.assertIn("unexpected keyword argument 'lane'", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_inline_proc_rejects_missing_required_argument(self) -> None:
        @pto.inline_proc
        def inline_store(dst: pto.Tile, src: pto.Tile):
            return None

        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="inline_proc_missing_required_unique", dtypes=[(pto.f32, pto.f32)])
            def kernel(dst: pto.Tile, src: pto.Tile):
                inline_store(dst=dst)
                return None

        self.assertIn("missing a required argument: 'src'", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_inline_proc_rejects_multiple_values_for_single_parameter(self) -> None:
        @pto.inline_proc
        def inline_store(dst: pto.Tile, src: pto.Tile):
            return None

        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="inline_proc_multiple_values_unique", dtypes=[(pto.f32, pto.f32)])
            def kernel(dst: pto.Tile, src: pto.Tile):
                inline_store(dst, src, src=src)
                return None

        self.assertIn("multiple values for argument 'src'", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_inline_proc_semantic_emits_controlled_namespace_none_call(self) -> None:
        @pto.vkernel(op="inline_proc_semantic_call_unique", dtypes=[(pto.f32, pto.f32)])
        def kernel(dst: pto.Tile, src: pto.Tile):
            _inline_copy_row(dst, src, 0)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )
        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))

        call_stmts = [
            stmt
            for stmt in semantic_kernel.body
            if isinstance(stmt, SemanticExprStmt) and isinstance(stmt.expr, SemanticCallExpr)
        ]
        self.assertGreaterEqual(len(call_stmts), 1)
        inline_call = call_stmts[0].expr
        self.assertIsNone(inline_call.namespace)
        self.assertRegex(inline_call.name, r"^__tl_inline_")
        self.assertGreaterEqual(len(semantic_kernel.inline_helpers), 1)
        self.assertRegex(semantic_kernel.inline_helpers[0].symbol_name, r"^__tl_inline_")

    def test_inline_proc_semantic_keeps_expression_call_return_type(self) -> None:
        @pto.inline_proc
        def inline_const_i32():
            return pto.i32(1)

        @pto.vkernel(op="inline_proc_semantic_expr_unique", dtypes=[(pto.f32,)])
        def kernel(dst: pto.Tile):
            lane = inline_const_i32()
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )
        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))

        assign_stmt = next(
            stmt
            for stmt in semantic_kernel.body
            if isinstance(stmt, SemanticAssignStmt) and isinstance(stmt.value, SemanticCallExpr)
        )
        self.assertIsNone(assign_stmt.value.namespace)
        self.assertRegex(assign_stmt.value.name, r"^__tl_inline_")
        self.assertIsInstance(assign_stmt.value.type, SemanticScalarType)

    def test_inline_proc_lowering_renders_private_helpers_and_call_bindings(self) -> None:
        @pto.inline_proc
        def inline_const_i32():
            return 1

        @pto.inline_proc
        def inline_store(dst: pto.Tile, src: pto.Tile):
            lane = inline_const_i32()
            _inline_copy_row(dst, src, lane)
            return None

        @pto.vkernel(op="inline_proc_lowering_helpers_unique", dtypes=[(pto.f32, pto.f32)])
        def kernel(dst: pto.Tile, src: pto.Tile):
            lane = inline_const_i32()
            inline_store(dst, src)
            _inline_copy_row(dst, src, lane)
            return None

        text = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        ).mlir_text()
        self.assertIn("func.func private @__tl_inline_", text)
        self.assertIn("attributes { pto.tilelang.inline_proc }", text)
        self.assertGreaterEqual(text.count("func.func"), 3)
        self.assertGreaterEqual(text.count("pto.tilelang.inline_proc"), 2)
        self.assertRegex(text, r"= func\.call @__tl_inline_[A-Za-z0-9_]+\(.*\) : \([^\)]*\) -> index")
        self.assertRegex(text, r"func\.call @__tl_inline_[A-Za-z0-9_]+\(.*\) : \([^\)]*\) -> \(\)")

    def test_inline_proc_supports_constexpr_dtype_dispatch(self) -> None:
        @pto.inline_proc
        def inline_pick_lane(dtype):
            if pto.constexpr(dtype == pto.ui16):
                lane = 1
            elif pto.constexpr(dtype == pto.i16):
                lane = 2
            elif pto.constexpr(dtype == pto.ui32):
                lane = 3
            else:
                lane = 4
            return lane

        @pto.vkernel(op="inline_proc_constexpr_dtype_dispatch_unique", dtypes=[(pto.f32,)])
        def kernel(dst: pto.Tile):
            lane = inline_pick_lane(dst.element_type)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )
        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))

        assign_stmt = next(
            stmt
            for stmt in semantic_kernel.body
            if isinstance(stmt, SemanticAssignStmt) and isinstance(stmt.value, SemanticCallExpr)
        )
        self.assertRegex(assign_stmt.value.name, r"^__tl_inline_")
        self.assertEqual(len(semantic_kernel.inline_helpers), 1)
        helper_assign = next(
            stmt
            for stmt in semantic_kernel.inline_helpers[0].body
            if isinstance(stmt, SemanticAssignStmt)
            and len(stmt.targets) == 1
            and stmt.targets[0].name == "lane"
        )
        self.assertIsInstance(helper_assign.value, SemanticLiteralExpr)
        self.assertEqual(helper_assign.value.value, 4)

    def test_inline_proc_specializes_same_type_with_different_static_values(self) -> None:
        @pto.inline_proc
        def inline_scale(lane: pto.i32):
            if pto.constexpr(lane == 1):
                value = 2
            else:
                value = 4
            return value

        @pto.vkernel(op="inline_proc_static_value_specialization_unique", dtypes=[(pto.f32,)])
        def kernel(dst: pto.Tile):
            lane0 = inline_scale(1)
            lane1 = inline_scale(2)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )
        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))

        self.assertEqual(len(semantic_kernel.inline_helpers), 2)
        literal_values = []
        for helper in semantic_kernel.inline_helpers:
            helper_assign = next(
                stmt
                for stmt in helper.body
                if isinstance(stmt, SemanticAssignStmt)
                and len(stmt.targets) == 1
                and stmt.targets[0].name == "value"
            )
            self.assertIsInstance(helper_assign.value, SemanticLiteralExpr)
            literal_values.append(helper_assign.value.value)
        self.assertEqual(sorted(literal_values), [2, 4])

    def test_inline_proc_rejects_mutual_recursion(self) -> None:
        @pto.inline_proc
        def inline_a(dst: pto.Tile):
            inline_b(dst)
            return None

        @pto.inline_proc
        def inline_b(dst: pto.Tile):
            inline_a(dst)
            return None

        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="inline_proc_mutual_recursion_unique", dtypes=[(pto.f32,)])
            def kernel(dst: pto.Tile):
                inline_a(dst)
                return None

            kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB)
            ).mlir_text()

        self.assertIn("recursive inline_proc call", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))


class TileLangDSLDiagnosticsTests(unittest.TestCase):
    def test_matcher_feature_validation_rejects_invalid_constraints_and_priority(self) -> None:
        def kernel(x: pto.TensorView):
            return None

        with self.assertRaises(TypeError) as constraints_ctx:
            pto.vkernel(op="x", dtypes=[(pto.f32,)], constraints=[123])(kernel)
        self.assertIn("constraints[0] must be callable", str(constraints_ctx.exception))

        with self.assertRaises(TypeError) as priority_ctx:
            pto.vkernel(op="x", dtypes=[(pto.f32,)], priority=True)(kernel)
        self.assertIn("priority must be an int", str(priority_ctx.exception))

    def test_advanced_mode_keeps_vreduce_rejected_until_authoring_op_exists(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.i32,)], advanced=True)
            def kernel(x: pto.Tile):
                pto.vreduce(x)
                return None

        self.assertIn("advanced family surface `pto.vreduce`", str(ctx.exception))

    def test_set_mov_pad_val_rejects_unsupported_scalar_dtype(self) -> None:
        with self.assertRaises(TypeError) as ctx:

            @pto.vkernel(op="set_mov_pad_val_bad_dtype_unique", dtypes=[(pto.f32,)], advanced=True)
            def kernel(dst: pto.Tile):
                pto.set_mov_pad_val(pto.i64(0))
                return None

            kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB)
            ).mlir_text()

        self.assertIn(
            "pto.set_mov_pad_val pad_value must be an 8/16/32-bit integer or f16/bf16/f32",
            str(ctx.exception),
        )

    def test_unsupported_python_syntax_reports_source_location(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.f32,)])
            def kernel(x: pto.TensorView):
                while True:
                    return None

        self.assertIn("unsupported Python syntax `while`", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_pass_statement_builds_frontend_noop_and_compiles(self) -> None:
        @pto.vkernel(op="pass_statement_frontend_noop_unique", dtypes=[(pto.f32,)])
        def kernel(dst: pto.Tile):
            pass
            if pto.constexpr(True):
                pass
            else:
                pass
            return None

        selected = pto.select_kernel(
            "a5",
            "pass_statement_frontend_noop_unique",
            (pto.f32,),
        )
        frontend_kernel = build_frontend_kernel_node(selected)
        self.assertIsInstance(frontend_kernel.body[0], FrontendNoOpStmt)
        self.assertIsInstance(frontend_kernel.body[1], FrontendIfStmt)
        if_stmt = frontend_kernel.body[1]
        self.assertTrue(if_stmt.is_constexpr)
        self.assertIsInstance(if_stmt.then_body[0], FrontendNoOpStmt)
        self.assertIsInstance(if_stmt.else_body[0], FrontendNoOpStmt)

        text = selected.specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        ).mlir_text()
        self.assertIn("return", text)
        self.assertNotIn("scf.if", text)

    def test_vreg_annotated_assignment_rejects_mismatched_dtype(self) -> None:
        with self.assertRaises(TypeError) as ctx:

            @pto.vkernel(op="vreg_annotation_mismatch_unique", dtypes=[(pto.f32,)], advanced=True)
            def kernel(dst: pto.Tile):
                vec: pto.vreg(pto.f16) = pto.vlds(dst, 0)
                return None

            kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            ).mlir_text()

        self.assertIn("annotated vector type `vreg(f16)` does not match inferred !pto.vreg<64xf32>", str(ctx.exception))

    def test_mask_annotated_assignment_rejects_mismatched_granularity(self) -> None:
        with self.assertRaises(TypeError) as ctx:

            @pto.vkernel(op="mask_annotation_mismatch_unique", dtypes=[(pto.f32,)], advanced=True)
            def kernel(dst: pto.Tile):
                mask: pto.mask_b16 = pto.make_mask(pto.f32, pto.PAT.ALL)
                return None

            kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            ).mlir_text()

        self.assertIn("annotated mask type `mask_b16` does not match inferred !pto.mask<b32>", str(ctx.exception))

    def test_arbitrary_external_call_reports_source_location(self) -> None:
        def helper():
            return None

        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.f32,)])
            def kernel(x: pto.TensorView):
                helper()
                return None

        self.assertIn("arbitrary external call `helper`", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_vstur_rejects_raw_string_mode_and_requires_enum(self) -> None:
        with self.assertRaises(TypeError) as ctx:

            @pto.vkernel(op="vstur_raw_string_mode_unique", dtypes=[(pto.f32,)], advanced=True)
            def kernel(dst: pto.ptr(pto.f32, pto.MemorySpace.UB)):
                align = pto.init_align()
                vec = pto.vbr(1.0)
                pto.vstur(align, vec, dst, "POST_UPDATE")
                return None

            kernel.specialize().mlir_text()

        self.assertIn("pto.vstur mode must be a PostUpdateMode enum", str(ctx.exception))

    def test_unsupported_pto_surface_reports_source_location(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.f32,)])
            def kernel(x: pto.TensorView):
                pto.not_a_real_surface(x)
                return None

        self.assertIn("unsupported op surface `pto.not_a_real_surface`", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_strict_vecscope_requires_advanced_mode(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.f32, pto.f32)])
            def kernel(x: pto.TensorView, tile: pto.Tile):
                with pto.strict_vecscope(tile, tile, 0, 256, 64) as (lhs, rhs, lb, ub, step):
                    pass
                return None

        self.assertIn("surface `pto.strict_vecscope` requires advanced=True", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_advanced_family_requires_advanced_mode(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.f32, pto.f32)])
            def kernel(x: pto.TensorView, tile: pto.Tile):
                mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                pto.vcmp(tile, tile, mask, "lt")
                return None

        self.assertIn("surface `pto.vcmp` requires advanced=True", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_missing_specialization_reports_source_location(self) -> None:
        @pto.vkernel(op="x", dtypes=[(pto.f32, pto.f16)])
        def kernel(x: pto.TensorView, tile: pto.Tile):
            return None

        with self.assertRaises(pto.TileLangFrontendError) as ctx:
            kernel.mlir_text()

        self.assertIn("requires specialize() bindings for bare Tile parameters", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_dynamic_shape_and_illegal_profile_report_source_location(self) -> None:
        @pto.vkernel(op="x", dtypes=[(pto.f32, pto.f16)])
        def kernel(x: pto.TensorView, tile: pto.Tile):
            return None

        with self.assertRaises(pto.TileLangFrontendError) as dynamic_ctx:
            kernel.specialize(tile={"shape": (16, "n"), "memory_space": "ub"})
        self.assertIn("dynamic physical Tile shape is not supported", str(dynamic_ctx.exception))
        self.assertIn(f"{__file__}:", str(dynamic_ctx.exception))

        with self.assertRaises(pto.TileLangFrontendError) as rank_ctx:
            kernel.specialize(tile={"shape": (4, 4, 4), "memory_space": "ub"})
        self.assertIn("v1 only supports rank-1 or rank-2 Tile shapes", str(rank_ctx.exception))
        self.assertIn(f"{__file__}:", str(rank_ctx.exception))

        with self.assertRaises(pto.TileLangFrontendError) as space_ctx:
            kernel.specialize(tile={"shape": (4, 4), "memory_space": "gm"})
        self.assertIn("v1 only supports MemorySpace.UB", str(space_ctx.exception))
        self.assertIn(f"{__file__}:", str(space_ctx.exception))

        with self.assertRaises(pto.TileLangFrontendError) as valid_shape_ctx:
            kernel.specialize(tile={"shape": (4, 4), "memory_space": "ub", "valid_shape": (5, 4)})
        self.assertIn("valid_shape axis 0=5 must be <= shape axis 0=4", str(valid_shape_ctx.exception))
        self.assertIn(f"{__file__}:", str(valid_shape_ctx.exception))

    def test_slice_index_type_error_reports_template_source_location(self) -> None:
        source = """
import tilelang_dsl as pto

@pto.inline_proc
def store_row(dst: pto.Tile, src: pto.Tile, row: pto.i32):
    vec = pto.vlds(src[row, 0:])
    mask = pto.make_mask(dst.element_type, pto.PAT.ALL)
    pto.vsts(vec, dst[row, 0:], mask)
    return None

@pto.vkernel(op="diag_index_type_unique", dtypes=[(pto.f32, pto.f32, pto.i32)])
def kernel(dst: pto.Tile, src: pto.Tile, row: pto.i32):
    store_row(dst, src, row)
    return None
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = Path(tmpdir) / "diag_index_type_kernel.py"
            module_path.write_text(source, encoding="utf-8")
            spec = util.spec_from_file_location("diag_index_type_kernel", module_path)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)

            specialized = module.kernel.specialize(
                dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
                src=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            )

            with self.assertRaises(TypeError) as ctx:
                specialized.mlir_text()

        message = str(ctx.exception)
        self.assertIn(str(module_path), message)
        self.assertIn(":6:", message)
        self.assertIn("slice bounds and vector offsets must be index-typed", message)


if __name__ == "__main__":
    unittest.main()
