// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_PHASE_FIXTURES_H
#define PTO_OAHS_PHASE_FIXTURES_H
#include "PTO/Transforms/OAHS/Phases.h"
namespace phase_fixture {
  namespace o=mlir::pto::oahs;
  constexpr unsigned L=0,M=1,F=2,Q=3;
  inline o::Program program(unsigned blocks=2) {
    o::Program p;
    p.target.contract="supplied-reference-issue-only";
    for(unsigned a=0;a<4;++a) {
      p.target.supported[a]=p.target.barriers[a]=true;
      for(unsigned b=0;b<4;++b)if(a!=b)p.target.keys[a][b]= {
        0,1
      };
    }
    auto cell=[&](const char *domain,uint64_t offset) {
      o::Cell c;
      c.addressSpace=domain;
      c.ranges= {
        {
          offset,512
        }
      };
      p.cells.push_back(c);
    };
    cell("LEFT",0);
    cell("RIGHT",0);
    cell("LOCAL",0);
    for(unsigned j=0;j<blocks;++j)cell("ACC",512*j);
    for(unsigned j=0;j<blocks;++j)cell("GM",512*j);
    o::FinalBlockProfile profile;
    profile.entryWritable=profile.orderedPerBlockService=profile.enabledPhaseProgress=profile.commonForwardControl=true;
    o::FinalBlockGroup g;
    g.name="acc";
    g.producer=o::Pipe(M);
    g.consumer=o::Pipe(F);
    for(unsigned j=0;j<blocks;++j)g.blocks.push_back(3+j);
    profile.groups.push_back(g);
    p.finalBlocks=profile;
    return p;
  }
  inline o::Operation ordinary(unsigned pipe,std::vector<unsigned> reads= {
  },std::vector<unsigned> writes= {
  }) {
    o::Operation op;
    op.pipe=o::Pipe(pipe);
    op.complete=true;
    for(auto c:reads)op.accesses.push_back( {
      c,true,false,false
    });
    for(auto c:writes)op.accesses.push_back( {
      c,false,true,true
    });
    return op;
  }
  inline o::Operation producer(const o::Program &p) {
    auto op=ordinary(M, {
      0,1
    });
    o::FinalBlockOperation f;
    f.group=0;
    f.blocks=p.finalBlocks->groups[0].blocks;
    f.operands= {
      0,1
    };
    for(auto b:f.blocks)op.accesses.push_back( {
      b,false,true,true
    });
    op.finalBlock=f;
    return op;
  }
  inline o::Operation consumer(const o::Program &p) {
    auto op=ordinary(F);
    o::FinalBlockOperation f;
    f.role=o::FinalBlockOperation::Role::Consumer;
    f.group=0;
    f.blocks=p.finalBlocks->groups[0].blocks;
    for(auto b:f.blocks)op.accesses.push_back( {
      b,true,false,false
    });
    for(unsigned j=0;j<f.blocks.size();++j) {
      unsigned out=3+f.blocks.size()+j;
      f.outputs.push_back(out);
      op.accesses.push_back( {
        out,false,true,true
      });
    }
    op.finalBlock=f;
    return op;
  }
  inline std::vector<o::Command> handoff(unsigned a,unsigned b,unsigned key=0) {
    return {
      {
        o::Command::Publish,o::Pipe(a),o::Pipe(b),key
      }, {
        o::Command::Acquire,o::Pipe(a),o::Pipe(b),key
      }
    };
  }
  inline o::Region leaf(unsigned i) {
    return {
      o::Region::Operation, {
      },i
    };
  }
  inline void loop(o::Program &p) {
    o::Region body;
    for(unsigned i=0;i<p.operations.size();++i)body.children.push_back(leaf(i));
    p.body= {
      o::Region::For, {
        body
      },0,true
    };
  }
}
// namespace phase_fixture
#endif
