# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent NumPy goldens and exact physical ABI for four extracted fixtures.

These are the extracted operations, not the full original models. Workspace
rounding to fp16 is part of the WY contract. No device result is an oracle.
"""
from dataclasses import dataclass
import numpy as np

CASES = ('conv2d_interior', 'flash_attention_cube', 'gdn_wy', 'kda_wy')
SENTINEL = -123.0

@dataclass
class Fixture:
    buffers: list
    outputs: dict
    atol: float
    rtol: float


def wy_reference(k, v, beta, gate, a, chunks, tail, kda):
    u = np.full((2048, 128), SENTINEL, dtype=np.float16)
    w = u.copy()
    for chunk in range(chunks):
        start = chunk * 128
        live = 64 if tail and chunk + 1 == chunks else 128
        sl = slice(start, start + live)
        b = np.zeros(128, np.float32)
        b[:live] = beta[start:start + live]
        ac = a[sl].astype(np.float32)
        a2 = (ac * b[None, :]).astype(np.float16).astype(np.float32)
        kc = np.zeros((128, 128), np.float32)
        vc = kc.copy()
        kc[:live] = k[sl]
        vc[:live] = v[sl]
        u[sl] = a2 @ vc
        if kda:
            effective = np.zeros((128, 128), np.float32)
            effective[:live] = (kc[:live] * np.exp(gate[sl])).astype(np.float16)
            w[sl] = a2 @ effective
        else:
            gb = np.zeros(128, np.float32)
            gb[:live] = np.exp(gate[start:start + live]) * b[:live]
            a1 = (ac * gb[None, :]).astype(np.float16).astype(np.float32)
            w[sl] = a1 @ kc
    return u, w


def fa_reference(q, k, v, tiles):
    # Online FP32 softmax with the same explicit FP16 probability interface.
    # The cube extraction computes Q @ K^T and P @ V, one 16-key tile at a time.
    if not tiles:
        return np.full((16, 16), SENTINEL, np.float32)
    maximum = np.full((16, 1), -np.inf, np.float32)
    denominator = np.zeros((16, 1), np.float32)
    numerator = np.zeros((16, 16), np.float32)
    for i in range(tiles):
        scores = q.astype(np.float32) @ k[16*i:16*(i+1)].astype(np.float32).T
        new_max = np.maximum(maximum, scores.max(axis=1, keepdims=True))
        scale = np.exp((maximum - new_max) * np.float32(0.25))
        probability = np.exp((scores - new_max) * np.float32(0.25))
        denominator = denominator * scale + probability.sum(axis=1, keepdims=True)
        product = probability.astype(np.float16).astype(np.float32) @ v[16*i:16*(i+1)].astype(np.float32)
        numerator = numerator * scale + product
        maximum = new_max
    return numerator / denominator


def conv_reference(fmap, weights, panels):
    # Fixed mIter=3, nIter=0: flattened spatial rows 384..511, output C=256.
    # fmap panel: [16,96,16], weights z stride=98304 elements (full N=6144).
    acc = np.zeros((128, 256), np.float32)
    row = np.arange(128)
    yy = (384 + row) // 96
    xx = (384 + row) % 96
    for panel in range(panels):
        for kh in range(3):
            for kw in range(3):
                x = xx + kw - 1
                valid = (x >= 0) & (x < 96)
                patch = np.zeros((128, 16), np.float32)
                offsets = panel*24576 + (yy[valid] + kh - 1)*1536 + x[valid]*16
                patch[valid] = fmap[offsets[:, None] + np.arange(16)]
                start = (panel*9 + kh*3 + kw)*98304
                # FRACTAL_Z stores [n1,n0,c0] in each z plane.
                weight = weights[start:start+4096].reshape(16,16,16).reshape(256,16)
                acc += patch @ weight.astype(np.float32).T
    output = np.full(16*24576, SENTINEL, np.float16)
    locations = (np.arange(256)[None, :]//16)*24576 + (384+row[:, None])*16 + np.arange(256)[None, :]%16
    output[locations] = acc.astype(np.float16)
    return output


def make_fixture(case, count, tail=False, seed=2026):
    if case not in CASES:
        raise ValueError(case)
    maximum = 32 if case == 'conv2d_interior' else 16
    minimum = 1 if case == 'conv2d_interior' else 0  # Conv stores an uninitialized ACC at zero.
    if not minimum <= count <= maximum or (tail and case not in ('gdn_wy', 'kda_wy')):
        raise ValueError('Unsupported count/tail for frozen extraction')
    rng = np.random.default_rng(seed)
    def random(shape, dtype=np.float16):
        return rng.uniform(-0.125, 0.125, shape).astype(dtype)
    if case in ('gdn_wy', 'kda_wy'):
        k, v, a = (random((2048,128)) for _ in range(3))
        beta = rng.uniform(0.25, 0.75, 2048).astype(np.float16)
        gate = random((2048,128) if case == 'kda_wy' else 2048, np.float32)
        u, w = wy_reference(k, v, beta, gate, a, count, tail, case == 'kda_wy')
        buffers = [k, v, beta, gate, a, np.zeros((128,128),np.float16),
                   np.zeros((128,128),np.float16), np.full_like(u,SENTINEL), np.full_like(w,SENTINEL)]
        return Fixture(buffers, {7:u,8:w}, 0.002, 0.005)
    if case == 'flash_attention_cube':
        q, k, v = random((16,16)), random((256,16)), random((256,16))
        expected = fa_reference(q,k,v,count)
        buffers = [q,k,v,np.zeros(8*256,np.float32),np.zeros(8*256,np.float16),
                   np.zeros(8*256,np.float32),np.full((16,16),SENTINEL,np.float32)]
        return Fixture(buffers,{6:expected}, 0.0002, 0.005)
    fmap = random(count*24576)
    weights = random(count*9*98304)
    expected = conv_reference(fmap,weights,count)
    return Fixture([fmap,weights,np.full_like(expected,SENTINEL)],{2:expected}, 0.005, 0.005)
