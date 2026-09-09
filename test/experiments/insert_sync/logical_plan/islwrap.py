# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# Original Event Lab reference code retains its MIT notice below and in LICENSE.
# SPDX-License-Identifier: MIT
"""Small ownership-safe binding to the installed isl C API (no Python package needed).

Only a reference implementation. It does not parse or change PTO IR. Every operation
copies its inputs; errors become exceptions, never optimistic analysis results.
"""
from __future__ import annotations
import ctypes as C
import ctypes.util
import os
from typing import Callable

P = C.c_void_p
I = C.c_int
S = C.c_char_p

class ISLError(RuntimeError):
    pass

class ISL:
    def __init__(self, max_operations: int = 10_000_000):
        if not isinstance(max_operations, int) or max_operations <= 0:
            raise ValueError('max_operations must be a positive integer')
        path = os.environ.get('PTOAS_EVENT_MODEL_ISL_LIBRARY') or ctypes.util.find_library('isl')
        if not path:
            raise ISLError('libisl not found; install the system isl runtime library')
        self.lib = C.CDLL(path)
        self.libc = C.CDLL(None)
        self.libc.free.argtypes = [P]
        self.libc.free.restype = None
        self._bind('isl_ctx_alloc', [], P)
        self._bind('isl_ctx_free', [P], None)
        self._bind('isl_ctx_reset_operations', [P], None)
        self._bind('isl_ctx_set_max_operations', [P, C.c_ulong], None)
        self._bind('isl_ctx_last_error_msg', [P], S)
        self._bind('isl_ctx_reset_error', [P], None)
        self._bind('isl_options_set_on_error', [P, I], I)
        self._bind('isl_version', [], S)
        self.ctx = self.lib.isl_ctx_alloc()
        self.lib.isl_options_set_on_error(self.ctx, 1)  # continue; wrapper raises
        self.lib.isl_ctx_set_max_operations(self.ctx, max_operations)
        for kind in ('union_map', 'union_set', 'set', 'map'):
            for op, args, ret in [('read_from_str', [P,S], P), ('copy',[P],P),
                                  ('free',[P],P), ('to_str',[P],P)]:
                self._bind(f'isl_{kind}_{op}',args,ret)
        for op in ('union','subtract','intersect','apply_range','apply_domain',
                   'lex_lt_union_map','lex_le_union_map'):
            self._bind('isl_union_map_'+op,[P,P],P)
        for op in ('reverse','coalesce','lexmax','lexmin','wrap'):
            self._bind('isl_union_map_'+op,[P],P)
        for op in ('domain','range'):
            self._bind('isl_union_map_'+op,[P],P)
        for op in ('is_subset','is_equal'):
            self._bind('isl_union_map_'+op,[P,P],I)
        for op in ('is_empty','is_single_valued','is_injective'):
            self._bind('isl_union_map_'+op,[P],I)
        self._bind('isl_union_map_intersect_params',[P,P],P)
        self._bind('isl_union_map_intersect_domain',[P,P],P)
        self._bind('isl_union_map_intersect_range',[P,P],P)
        self._bind('isl_union_map_from_domain_and_range',[P,P],P)
        self._bind('isl_union_set_identity',[P],P)
        self._bind('isl_union_set_project_out_all_params',[P],P)
        self._bind('isl_union_map_transitive_closure',[P,C.POINTER(I)],P)
        self._bind('isl_union_map_foreach_map',[P,P,P],I)
        self._bind('isl_map_wrap',[P],P)
        self._bind('isl_map_get_tuple_name',[P,I],S)
        self._bind('isl_map_dim',[P,I],C.c_uint)
        self._bind('isl_set_dim',[P,I],C.c_uint)
        self._bind('isl_set_is_bounded',[P],I)
        self._bind('isl_set_foreach_point',[P,P,P],I)
        self._bind('isl_point_get_coordinate_val',[P,I,I],P)
        self._bind('isl_point_free',[P],P)
        self._bind('isl_val_to_str',[P],P)
        self._bind('isl_val_free',[P],P)
        self._bind('isl_union_access_info_from_sink',[P],P)
        for op in ('set_must_source','set_may_source','set_kill','set_schedule_map'):
            self._bind('isl_union_access_info_'+op,[P,P],P)
        self._bind('isl_union_access_info_compute_flow',[P],P)
        self._bind('isl_union_flow_free',[P],P)
        for op in ('get_must_dependence','get_may_dependence','get_must_no_source','get_may_no_source'):
            self._bind('isl_union_flow_'+op,[P],P)

    def __del__(self):
        # Rel objects retain this owner, so the context outlives all their maps.
        if getattr(self, 'ctx', None):
            self.lib.isl_ctx_free(self.ctx)
            self.ctx = None

    def _bind(self, name, args, result):
        fn = getattr(self.lib, name)
        fn.argtypes, fn.restype = args, result

    @property
    def version(self):
        return self.lib.isl_version().decode().strip()

    def checked(self, ptr):
        if not ptr:
            message = self.lib.isl_ctx_last_error_msg(self.ctx)
            self.lib.isl_ctx_reset_error(self.ctx)
            raise ISLError(message.decode() if message else 'isl operation failed')
        return ptr

    def string(self, ptr):
        self.checked(ptr)
        try:
            return C.string_at(ptr).decode()
        finally:
            self.libc.free(ptr)

    def map(self, text: str):
        self.lib.isl_ctx_reset_operations(self.ctx)
        return Rel(self, self.checked(self.lib.isl_union_map_read_from_str(self.ctx,text.encode())))

    def flow(self, sinks: 'Rel', writes: 'Rel', schedule: 'Rel', must=True):
        self.lib.isl_ctx_reset_operations(self.ctx)
        access = self.checked(self.lib.isl_union_access_info_from_sink(sinks.copy()))
        source_fn = (self.lib.isl_union_access_info_set_must_source if must else
                     self.lib.isl_union_access_info_set_may_source)
        access = self.checked(source_fn(access,writes.copy()))
        access = self.checked(self.lib.isl_union_access_info_set_schedule_map(access,schedule.copy()))
        result = self.checked(self.lib.isl_union_access_info_compute_flow(access))
        try:
            return {key: Rel(self,self.checked(getattr(self.lib,'isl_union_flow_get_'+key)(result)))
                    for key in ('must_dependence','may_dependence','must_no_source','may_no_source')}
        finally:
            self.lib.isl_union_flow_free(result)

class Rel:
    def __init__(self, owner: ISL, ptr):
        self.isl, self.ptr = owner, ptr
    def __del__(self):
        if getattr(self,'ptr',None):
            self.isl.lib.isl_union_map_free(self.ptr)
            self.ptr = None
    def copy(self):
        return self.isl.checked(self.isl.lib.isl_union_map_copy(self.ptr))
    def __str__(self):
        return self.isl.string(self.isl.lib.isl_union_map_to_str(self.ptr))
    def _unary(self,op):
        return Rel(self.isl,self.isl.checked(getattr(self.isl.lib,'isl_union_map_'+op)(self.copy())))
    def _binary(self,op,other):
        if self.isl is not other.isl: raise ValueError('different isl contexts')
        return Rel(self.isl,self.isl.checked(getattr(self.isl.lib,'isl_union_map_'+op)(self.copy(),other.copy())))
    def __or__(self,b): return self._binary('union',b).coalesce()
    def __and__(self,b): return self._binary('intersect',b).coalesce()
    def __sub__(self,b): return self._binary('subtract',b).coalesce()
    def then(self,b,*,simplify=True):
        result = self._binary('apply_range', b)
        return result.coalesce() if simplify else result
    def intersect(self,b,*,simplify=True):
        result = self._binary('intersect', b)
        return result.coalesce() if simplify else result
    def reverse(self): return self._unary('reverse')
    def coalesce(self): return self._unary('coalesce')
    def lexmax(self): return self._unary('lexmax')
    def lexmin(self): return self._unary('lexmin')
    def order(self,other,strict=True):
        return self._binary('lex_lt_union_map' if strict else 'lex_le_union_map',other)
    def _test(self,op,other=None):
        fn = getattr(self.isl.lib,'isl_union_map_'+op)
        r = fn(self.ptr) if other is None else fn(self.ptr,other.ptr)
        if r < 0: self.isl.checked(None)
        return bool(r)
    def subset(self,b): return self._test('is_subset',b)
    def equal(self,b): return self._test('is_equal',b)
    def empty(self): return self._test('is_empty')
    def single(self): return self._test('is_single_valued')
    def injective(self): return self._test('is_injective')
    def filter(self,domain: 'Rel'|None=None,range_: 'Rel'|None=None):
        p = self.copy()
        for which,rel in [('domain',domain),('range',range_)]:
            if rel is not None:
                s = self.isl.checked(self.isl.lib.isl_union_map_domain(rel.copy()))
                p = self.isl.checked(getattr(self.isl.lib,'isl_union_map_intersect_'+which)(p,s))
        return Rel(self.isl,p)
    def domain_identity(self):
        source = self.isl.checked(self.isl.lib.isl_union_map_domain(self.copy()))
        return Rel(self.isl, self.isl.checked(self.isl.lib.isl_union_set_identity(source)))
    def range_identity(self, project_parameters=False):
        target = self.isl.checked(self.isl.lib.isl_union_map_range(self.copy()))
        if project_parameters:
            target = self.isl.checked(self.isl.lib.isl_union_set_project_out_all_params(target))
        return Rel(self.isl, self.isl.checked(self.isl.lib.isl_union_set_identity(target)))
    def specialize(self,params: dict[str,int]):
        constraints = ' and '.join(f'{k}={int(v)}' for k,v in sorted(params.items())) or 'true'
        text = '['+','.join(sorted(params))+'] -> { : '+constraints+' }'
        s = self.isl.checked(self.isl.lib.isl_set_read_from_str(self.isl.ctx,text.encode()))
        return Rel(self.isl,self.isl.checked(self.isl.lib.isl_union_map_intersect_params(self.copy(),s)))
    def closure(self,rounds=5,*,required=None):
        """Only exact finite path compositions. Never treat an overapproximation as proof.

        This is intentionally bounded and may fail to establish a true relationship.
        Each round doubles the maximum covered path length.
        """
        r = self
        if required is not None and required.subset(r):
            return r, False  # goal proved; a global fixed point was not tested
        for _ in range(rounds):
            nxt = r | r.then(r)
            if nxt.equal(r): return nxt, True
            r = nxt
            if required is not None and required.subset(r):
                return r, False
        return r, False
    def points(self,params: dict[str,int],limit=100000):
        """Enumerate only a fully specialized, finite relation for reference testing."""
        rel = self.specialize(params)
        rows=[]
        errors=[]
        lib=self.isl.lib
        MAPCB=C.CFUNCTYPE(I,P,P)
        POINTCB=C.CFUNCTYPE(I,P,P)
        @MAPCB
        def visit_map(m,_):
            try:
                a=lib.isl_map_get_tuple_name(m,2)
                b=lib.isl_map_get_tuple_name(m,3)
                na,nb=lib.isl_map_dim(m,2),lib.isl_map_dim(m,3)
                an=a.decode() if a else ''; bn=b.decode() if b else ''
                wrapped=self.isl.checked(lib.isl_map_wrap(m)); m=None
                bounded = lib.isl_set_is_bounded(wrapped)
                if bounded <= 0:
                    lib.isl_set_free(wrapped)
                    if bounded < 0: self.isl.checked(None)
                    raise ValueError('point enumeration requires a bounded specialization')
                @POINTCB
                def visit_point(p,__):
                    try:
                        vals=[]
                        for d in range(na+nb):
                            v=self.isl.checked(lib.isl_point_get_coordinate_val(p,3,d))
                            try: vals.append(int(self.isl.string(lib.isl_val_to_str(v))))
                            finally: lib.isl_val_free(v)
                        rows.append(((an,tuple(vals[:na])),(bn,tuple(vals[na:]))))
                        if len(rows)>limit: raise ValueError('enumeration budget exceeded')
                        return 0
                    except Exception as exc:
                        errors.append(exc); return -1
                    finally: lib.isl_point_free(p)
                result=lib.isl_set_foreach_point(wrapped,visit_point,None)
                lib.isl_set_free(wrapped)
                return result
            except Exception as exc:
                errors.append(exc); return -1
            finally:
                if m: lib.isl_map_free(m)
        status=lib.isl_union_map_foreach_map(rel.ptr,visit_map,None)
        if errors:
            self.isl.lib.isl_ctx_reset_error(self.isl.ctx)
            raise errors[0]
        if status<0: self.isl.checked(None)
        return sorted(rows)
