# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Support-matrix definitions and diagnostics for TileLang DSL v1."""

from __future__ import annotations

FOLLOW_UP_CHANGE = "extend-tilelang-dsl-matcher-and-advanced-surface"

# Tier definitions for TileLang DSL surface classification
# These tiers represent the user-facing support level of language features:
# - BASIC: Core surface that is fully supported and recommended for general use
# - ADVANCED: Features requiring advanced=True, suitable for expert users

BASIC_TIER = "basic"
ADVANCED_TIER = "advanced"

# Tier metadata for PTO calls and language constructs
# This provides a unified source of truth for documentation and testing

SUPPORTED_TOPLEVEL_PTO_CALLS = frozenset(
    {
        "constexpr",
        "bytewidth",
        "get_lanes",
        "elements_per_vreg",
        "get_op_attr",
        "vreg",
        "i1",
        "i8",
        "si8",
        "ui8",
        "i16",
        "si16",
        "ui16",
        "i32",
        "si32",
        "ui32",
        "i64",
        "si64",
        "ui64",
        "f16",
        "bf16",
        "f32",
        "get_block_idx",
        "get_subblock_idx",
        "get_block_num",
        "get_subblock_num",
        "set_flag",
        "wait_flag",
        "pipe_barrier",
        "barrier",
        "get_buf",
        "rls_buf",
        "mem_bar",
        "set_cross_core",
        "set_intra_block",
        "set_intra_core",
        "wait_flag_dev",
        "wait_intra_core",
    }
)

SUPPORTED_VECSCOPE_PTO_CALLS = frozenset(
    {
        "make_mask",
        "init_align",
        "vlds",
        "vldas",
        "vldus",
        "vldsx2",
        "plds",
        "psts",
        "pstu",
        "vsst",
        "vsta",
        "vstas",
        "vstar",
        "vsts",
        "vstsx2",
        "vstus",
        "vstur",
        "vabs",
        "vrelu",
        "vexp",
        "vln",
        "vsqrt",
        "vrec",
        "vnot",
        "vcadd",
        "vcmax",
        "vbcnt",
        "vneg",
        "vcls",
        "vcmin",
        "vrsqrt",
        "vmov",
        "vsunpack",
        "vzunpack",
        "vusqz",
        "vsqz",
        "vexpdif",
        "vexpdiff",
        "vtrc",
        "vbr",
        "vdup",
        "vadd",
        "vsub",
        "vmul",
        "vdiv",
        "vmod",
        "vmax",
        "vmin",
        "vand",
        "vor",
        "vxor",
        "vaddrelu",
        "vaddreluconv",
        "vsubrelu",
        "vaxpy",
        "vmulconv",
        "vmull",
        "vmula",
        "vshl",
        "vshr",
        "vprelu",
        "vadds",
        "vsubs",
        "vmuls",
        "vdivs",
        "vmaxs",
        "vmins",
        "vlrelu",
        "vshls",
        "vshrs",
        "vands",
        "vors",
        "vxors",
        "vcgadd",
        "vcgmax",
        "vcgmin",
        "vcpadd",
        "vpack",
        "vperm",
        "vshift",
        "vslide",
        "vsort32",
        "vmrgsort",
        "vcvt",
        "vbitcast",
        "pbitcast",
        "vci",
    }
)

ADVANCED_VECSCOPE_PTO_CALLS = frozenset(
    {
        "vscatter",
        "vcmp",
        "vcmps",
        "vsel",
        "vselr",
        "vselrv2",
        "pset_b8",
        "pset_b16",
        "pset_b32",
        "pge_b8",
        "pge_b16",
        "pge_b32",
        "plt_b8",
        "plt_b16",
        "plt_b32",
        "pnot",
        "psel",
        "pand",
        "por",
        "pxor",
        "ppack",
        "punpack",
        "pld",
        "pldi",
        "pst",
        "psti",
        "pdintlv_b8",
        "pdintlv_b16",
        "pdintlv_b32",
        "pintlv_b8",
        "pintlv_b16",
        "pintlv_b32",
        "vaddc",
        "vsubc",
        "vaddcs",
        "vsubcs",
        "vintlv",
        "vdintlv",
        "vintlvv2",
        "vdintlvv2",
        "vbitsort",
        "vmrgsort4",
    }
)

ADVANCED_EXPR_PTO_CALLS = frozenset(
    {
        "ptr",
        "castptr",
        "addptr",
        "load_scalar",
    }
)

ADVANCED_TOPLEVEL_PTO_CALLS = frozenset(
    {
        "strict_vecscope",
        "store_scalar",
        "set_mov_pad_val",
        "copy_gm_to_ubuf",
        "copy_ubuf_to_gm",
        "copy_ubuf_to_ubuf",
        "set_loop2_stride_outtoub",
        "set_loop1_stride_outtoub",
        "set_loop_size_outtoub",
        "set_loop2_stride_ubtoout",
        "set_loop1_stride_ubtoout",
        "set_loop_size_ubtoout",
    }
)

CUBE_ONLY_PTO_CALLS = frozenset(
    {
        "cube_load",
        "cube_store",
        "cube_load_frac",
        "bias_load",
        "left_load",
        "right_load",
        "left_load_mx",
        "right_load_mx",
        "mad",
        "mad_acc",
        "mad_bias",
        "mad_mx",
        "mad_mx_acc",
        "mad_mx_bias",
        "acc_store",
        "acc_store_gm",
        "acc_store_ub",
    }
)

DEFERRED_PTO_SURFACES = frozenset(
    {
        "vreduce",
    }
)

# Public surface groupings used by the guide, migration notes, and tests.
# These groupings intentionally mirror the user-facing authoring tiers rather
# than the internal lowering organization.

BASIC_TENSORVIEW_SURFACES = frozenset({"TensorView"})
BASIC_TILE_SURFACES = frozenset({"Tile"})
BASIC_HIGH_LEVEL_DMA_SURFACES = frozenset()
BASIC_BASE_VECTOR_SURFACES = frozenset(
    f"pto.{name}" for name in sorted(SUPPORTED_VECSCOPE_PTO_CALLS)
)

ADVANCED_RAW_POINTER_SURFACES = frozenset(
    {
        "ptr",
        "pto.ptr",
        "PointerType",
        "pto.castptr",
        "pto.addptr",
    }
)
ADVANCED_LOW_LEVEL_DMA_SURFACES = frozenset(
    {
        "pto.set_mov_pad_val",
        "pto.copy_gm_to_ubuf",
        "pto.copy_ubuf_to_gm",
        "pto.copy_ubuf_to_ubuf",
        "pto.set_loop2_stride_outtoub",
        "pto.set_loop1_stride_outtoub",
        "pto.set_loop_size_outtoub",
        "pto.set_loop2_stride_ubtoout",
        "pto.set_loop1_stride_ubtoout",
        "pto.set_loop_size_ubtoout",
    }
)
ADVANCED_EXPLICIT_VECSCOPE_SURFACES = frozenset({"pto.strict_vecscope"})
ADVANCED_TILE_HELPER_SURFACES = frozenset(
    {
        "tile.slice",
        "tile.reshape",
        "tile.as_ptr",
        "tensorview.as_ptr",
        "pto.tile_from_ptr",
        "pto.tile_with_strides",
        "pto.tile_config",
    }
)
BASIC_TILE_INDEXING_SURFACES = frozenset(
    {
        "tile[start:]",
        "tile[row, col:]",
    }
)

AUTHORING_TIER_SURFACE_GROUPS = {
    "TensorView": BASIC_TENSORVIEW_SURFACES,
    "Tile": BASIC_TILE_SURFACES,
    "base_vector_ops": BASIC_BASE_VECTOR_SURFACES,
    "tile_indexing_sugar": BASIC_TILE_INDEXING_SURFACES,
    "strict_vecscope": ADVANCED_EXPLICIT_VECSCOPE_SURFACES,
    "raw_pointer_family": ADVANCED_RAW_POINTER_SURFACES,
    "low_level_dma_family": ADVANCED_LOW_LEVEL_DMA_SURFACES,
    "tile_helper_family": ADVANCED_TILE_HELPER_SURFACES,
}

AUTHORING_TIER_GROUP_TIERS = {
    "TensorView": BASIC_TIER,
    "Tile": BASIC_TIER,
    "base_vector_ops": BASIC_TIER,
    "tile_indexing_sugar": BASIC_TIER,
    "strict_vecscope": ADVANCED_TIER,
    "raw_pointer_family": ADVANCED_TIER,
    "low_level_dma_family": ADVANCED_TIER,
    "tile_helper_family": ADVANCED_TIER,
}


def unsupported_feature_message(feature: str) -> str:
    return (
        f"{feature} is not supported in TileLang DSL v1; "
        f"see follow-up change `{FOLLOW_UP_CHANGE}`"
    )


def deferred_surface_message(name: str) -> str:
    return unsupported_feature_message(f"advanced family surface `pto.{name}`")


def advanced_mode_message(name: str) -> str:
    return f"surface `pto.{name}` requires advanced=True in TileLang DSL"


# Tier mapping for PTO calls
def get_pto_call_tier(call_name: str) -> str:
    """Return the tier of a PTO call.

    Args:
        call_name: Name of the PTO call (without 'pto.' prefix)

    Returns:
        One of BASIC_TIER or ADVANCED_TIER

    Raises:
        KeyError: If the PTO call is not part of the supported DSL surface
    """
    if call_name in SUPPORTED_TOPLEVEL_PTO_CALLS:
        return BASIC_TIER
    if call_name in SUPPORTED_VECSCOPE_PTO_CALLS:
        return BASIC_TIER
    if call_name in ADVANCED_VECSCOPE_PTO_CALLS:
        return ADVANCED_TIER
    if call_name in ADVANCED_EXPR_PTO_CALLS:
        return ADVANCED_TIER
    if call_name in ADVANCED_TOPLEVEL_PTO_CALLS:
        return ADVANCED_TIER
    raise KeyError(unsupported_feature_message(f"pto.{call_name}"))


UNSUPPORTED_LANGUAGE_CONSTRUCTS = frozenset(
    {
        "dma_load",
        "dma_store",
        "pto.dma_load",
        "pto.dma_store",
        "pto.dma_copy",
        "pto.vreduce",
        "pto.tile",
        "SyncOpType",
    }
)


# Tier mapping for language constructs (non-PTO-call features)
# These are higher-level abstractions in the TileLang DSL
LANGUAGE_CONSTRUCT_TIERS = {
    # Basic tier constructs
    "TensorView": BASIC_TIER,
    "Tile": BASIC_TIER,
    "VRegType": BASIC_TIER,
    "MaskType": BASIC_TIER,
    "pto.vreg": BASIC_TIER,
    "pto.mask_b8": BASIC_TIER,
    "pto.mask_b16": BASIC_TIER,
    "pto.mask_b32": BASIC_TIER,
    "BarrierType": BASIC_TIER,
    "PadMode": BASIC_TIER,
    "BLayout": BASIC_TIER,
    "SLayout": BASIC_TIER,
    "PadValue": BASIC_TIER,
    "constexpr": BASIC_TIER,
    "pto.constexpr": BASIC_TIER,
    "tile[start:]": BASIC_TIER,
    "tile[row, col:]": BASIC_TIER,
    # Advanced tier constructs
    "ptr": ADVANCED_TIER,  # raw pointer constructor
    "strict_vecscope": ADVANCED_TIER,  # explicit vecscope management
    "pto.strict_vecscope": ADVANCED_TIER,
    "tile.slice": ADVANCED_TIER,
    "tile.reshape": ADVANCED_TIER,
    "tile.as_ptr": ADVANCED_TIER,
    "tensorview.as_ptr": ADVANCED_TIER,
    "pto.tile_from_ptr": ADVANCED_TIER,
    "pto.tile_with_strides": ADVANCED_TIER,
    "pto.tile_config": ADVANCED_TIER,
}


def get_feature_tier(feature_name: str) -> str:
    """Return the tier of a TileLang DSL feature.

    Args:
        feature_name: Name of the feature, which can be:
            - A PTO call name (e.g., 'vadd', 'ptr')
            - A language construct (e.g., 'TensorView', 'dma_load')
            - A qualified construct (e.g., 'tile.slice', 'pto.tile_from_ptr')

    Returns:
        One of BASIC_TIER or ADVANCED_TIER

    Raises:
        KeyError: If the feature is documented but not part of the supported DSL surface
    """
    # First check if it's a known language construct
    if feature_name in LANGUAGE_CONSTRUCT_TIERS:
        return LANGUAGE_CONSTRUCT_TIERS[feature_name]
    if feature_name in UNSUPPORTED_LANGUAGE_CONSTRUCTS:
        raise KeyError(unsupported_feature_message(feature_name))

    # Check if it's a PTO call (might be qualified with 'pto.' prefix)
    call_name = feature_name
    if feature_name.startswith("pto."):
        call_name = feature_name[4:]

    # Check PTO call tier
    return get_pto_call_tier(call_name)


def get_surface_group_tier(group_name: str) -> str:
    """Return the authoring tier for a documented public-surface group."""

    return AUTHORING_TIER_GROUP_TIERS[group_name]


__all__ = [
    "CUBE_ONLY_PTO_CALLS",
    "DEFERRED_PTO_SURFACES",
    "FOLLOW_UP_CHANGE",
    "ADVANCED_EXPR_PTO_CALLS",
    "ADVANCED_TOPLEVEL_PTO_CALLS",
    "ADVANCED_VECSCOPE_PTO_CALLS",
    "SUPPORTED_TOPLEVEL_PTO_CALLS",
    "SUPPORTED_VECSCOPE_PTO_CALLS",
    "BASIC_TIER",
    "ADVANCED_TIER",
    "BASIC_TENSORVIEW_SURFACES",
    "BASIC_TILE_SURFACES",
    "BASIC_HIGH_LEVEL_DMA_SURFACES",
    "BASIC_BASE_VECTOR_SURFACES",
    "BASIC_TILE_INDEXING_SURFACES",
    "ADVANCED_EXPLICIT_VECSCOPE_SURFACES",
    "ADVANCED_RAW_POINTER_SURFACES",
    "ADVANCED_LOW_LEVEL_DMA_SURFACES",
    "ADVANCED_TILE_HELPER_SURFACES",
    "AUTHORING_TIER_SURFACE_GROUPS",
    "AUTHORING_TIER_GROUP_TIERS",
    "UNSUPPORTED_LANGUAGE_CONSTRUCTS",
    "LANGUAGE_CONSTRUCT_TIERS",
    "advanced_mode_message",
    "deferred_surface_message",
    "unsupported_feature_message",
    "get_pto_call_tier",
    "get_feature_tier",
    "get_surface_group_tier",
]
