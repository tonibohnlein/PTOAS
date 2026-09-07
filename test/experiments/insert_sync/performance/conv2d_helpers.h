// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//
#pragma once
#include <cstdint>
#include <pto/pto-inst.hpp>

// Source-derived, fixed interior tile: mIter=3, nIter=0, hinStart=3,
// hinCount=4, woutStart=0; baseM/K/N=128/48/256, stepKa=stepKb=3.
// Include before generated Conv2D C++. These calls deliberately have no
// invented InsertSync effects/pipe contract. They require future dialect support.
using namespace pto;
using BenchmarkFmapBacking = Tile<TileType::Mat, half, 128, 48, BLayout::ColMajor,
    128, 48, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>;
using BenchmarkWeightBacking = Tile<TileType::Mat, half, 144, 256, BLayout::ColMajor,
    144, 256, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>;
using BenchmarkLeft = Tile<TileType::Left, half, 128, 48, BLayout::RowMajor,
    128, 48, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>;
using BenchmarkRight = Tile<TileType::Right, half, 48, 256, BLayout::RowMajor,
    48, 256, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>;
using BenchmarkAcc = Tile<TileType::Acc, float, 128, 256, BLayout::ColMajor,
    128, 256, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal>;
using BenchmarkFmap = ConvTile<TileType::Mat, half, 12288, Layout::NC1HWC0,
    ConvTileShape<1, 1, -1, 96, 16>>;
using BenchmarkWeight = ConvTile<TileType::Mat, half, 73728, Layout::FRACTAL_Z,
    ConvTileShape<9, 16, 16, 16>>;

AICORE inline BenchmarkFmap benchmark_fmap(uint64_t address)
{
    BenchmarkFmap tile(int64_t{4});
    tile.SetFmapH(4);
    tile.SetFmapW(96);
    tile.SetChannelSize(16);
    tile.SetFilterH(3);
    tile.SetFilterW(3);
    tile.SetPadList(0, 1);
    tile.SetPadList(1, 1);
    tile.SetPadList(2, 0);
    tile.SetPadList(3, 0);
    TASSIGN(tile, address);
    return tile;
}

extern "C" AICORE inline void benchmark_conv_setfmatrix()
{
    auto tile = benchmark_fmap(0);
    SETFMATRIX(tile);
}

extern "C" AICORE inline void benchmark_conv_load_fmap(
    __gm__ half* source, BenchmarkFmapBacking backing, int64_t panel)
{
    auto tile = benchmark_fmap((uint64_t)backing.data());
    using ShapeT = Shape<1, 1, -1, 96, 16>;
    using GlobalT = GlobalTensor<half, ShapeT,
        Stride<786432, 24576, 1536, 16, 1>, Layout::NC1HWC0>;
    GlobalT input(source + panel * 24576 + 3 * 1536, ShapeT(int64_t{4}));
    TLOAD(tile, input);
}

extern "C" AICORE inline void benchmark_conv_load_weight(
    __gm__ half* source, BenchmarkWeightBacking backing, int64_t panel)
{
    BenchmarkWeight tile;
    TASSIGN(tile, (uint64_t)backing.data());
    using GlobalT = GlobalTensor<half, Shape<1, 9, 16, 16, 16>,
        Stride<28311552, 98304, 256, 16, 1>, Layout::FRACTAL_Z>;
    GlobalT input(source + panel * 9 * 98304);
    TLOAD(tile, input);
}

extern "C" AICORE inline void benchmark_conv_img2col(
    BenchmarkFmapBacking backing, BenchmarkLeft left, int64_t offset)
{
    auto tile = benchmark_fmap((uint64_t)backing.data());
    TIMG2COL(left, tile, 0, offset);
}

extern "C" AICORE inline void benchmark_conv_extract_weight(
    BenchmarkWeightBacking backing, BenchmarkRight right, int64_t offset)
{
    BenchmarkWeight tile;
    TASSIGN(tile, (uint64_t)backing.data());
    TEXTRACT(right, tile, offset, 0);
}

extern "C" AICORE inline void benchmark_conv_store(__gm__ half* target, BenchmarkAcc acc)
{
    using GlobalT = GlobalTensor<half, Shape<1, 16, 2, 96, 16>,
        Stride<9437184, 24576, 1536, 16, 1>, Layout::NC1HWC0>;
    GlobalT output(target + 384 * 16);
    TSTORE(output, acc);
}
