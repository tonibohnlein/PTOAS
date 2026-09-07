# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import unittest
import numpy as np
from goldens import conv_reference, fa_reference, wy_reference, make_fixture, SENTINEL


class GoldenTests(unittest.TestCase):
    def test_fa_uniform_scores_single_and_multiple_tiles(self):
        q = np.zeros((16,16),np.float16)
        k = np.zeros((256,16),np.float16)
        v = np.arange(4096,dtype=np.float32).reshape(256,16).astype(np.float16)
        for tiles in (1,2,3,16):
            expected = np.broadcast_to(v[:tiles*16].astype(np.float32).mean(axis=0),(16,16))
            np.testing.assert_allclose(fa_reference(q,k,v,tiles), expected,rtol=1e-6)
        self.assertTrue(np.all(fa_reference(q,k,v,0)==SENTINEL))

    def test_fa_matches_dense_attention_for_nonuniform_scores(self):
        rng = np.random.default_rng(4)
        q = rng.normal(size=(16,16)).astype(np.float16)
        k = rng.normal(size=(256,16)).astype(np.float16)
        v = rng.normal(size=(256,16)).astype(np.float16)
        scores = q.astype(np.float64) @ k.astype(np.float64).T / 4
        e = np.exp(scores - scores.max(axis=1,keepdims=True))
        dense = (e / e.sum(axis=1,keepdims=True)) @ v.astype(np.float64)
        np.testing.assert_allclose(fa_reference(q,k,v,16),dense,atol=0.0003,rtol=0.002)

    def test_wy_identity_and_half_tail(self):
        a = np.tile(np.eye(128,dtype=np.float16),(16,1))
        k = np.full((2048,128),2,np.float16)
        v = np.full_like(k,3)
        beta = np.full(2048,0.5,np.float16)
        for kda in (False,True):
            gate = np.zeros((2048,128) if kda else 2048,np.float32)
            for chunks in (0,1,3,16):
                for tail in (False,True):
                    u,w=wy_reference(k,v,beta,gate,a,chunks,tail,kda)
                    live=chunks*128-(64 if chunks and tail else 0)
                    np.testing.assert_array_equal(u[:live],np.full((live,128),1.5,np.float16))
                    np.testing.assert_array_equal(w[:live],np.ones((live,128),np.float16))
                    self.assertTrue(np.all(u[live:]==SENTINEL))
                    self.assertTrue(np.all(w[live:]==SENTINEL))

    def test_conv_spatial_padding_and_physical_store_footprint(self):
        f=np.ones(24576,np.float16)
        weights=np.ones(9*98304,np.float16)
        out=conv_reference(f,weights,1)
        # 3*3*16 terms except at x=0/95, where one filter column is padding.
        written=[]
        for row in range(128):
            x=(384+row)%96
            expected=96 if x in (0,95) else 144
            for c in range(256):
                offset=(c//16)*24576+(384+row)*16+c%16
                self.assertEqual(out[offset],expected)
                written.append(offset)
        mask=np.ones(out.size,bool);mask[written]=False
        self.assertTrue(np.all(out[mask]==SENTINEL))

    def test_abi_extents_and_invalid_shapes(self):
        for name in ('gdn_wy','kda_wy','flash_attention_cube'):
            f=make_fixture(name,0)
            for index,expected in f.outputs.items():
                np.testing.assert_array_equal(expected,f.buffers[index])
        with self.assertRaises(ValueError):make_fixture('conv2d_interior',0)
        with self.assertRaises(ValueError):make_fixture('gdn_wy',17)
        with self.assertRaises(ValueError):make_fixture('flash_attention_cube',1,True)

if __name__=='__main__':unittest.main()
