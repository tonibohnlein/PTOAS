// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_PHASE_MODEL_H
#define PTO_OAHS_PHASE_MODEL_H
#include "PTO/Transforms/OAHS/Phases.h"
#include <algorithm>
#include <limits>
#include <map>
#include <set>
namespace mlir::pto::oahs::detail {
  inline bool phaseModelValid(const Program &p, std::string &reason) {
    auto fail=[&](const char *s) {
      reason=s;
      return false;
    };
    if (!p.finalBlocks) {
      for (const auto &op:p.operations) if(op.finalBlock) return fail("phase operation has no supplied final-block contract");
      return true;
    }
    const auto &profile=*p.finalBlocks;
    if(profile.name!="final-block-pair-v1" || profile.qualification!="reference-contract-only") return fail("unknown final-block profile or unregistered native qualification");
    if(!profile.entryWritable || !profile.orderedPerBlockService || !profile.enabledPhaseProgress || !profile.commonForwardControl) return fail("missing writable-entry, per-block service, progress or common-control premise");
    if(profile.groups.empty()) return fail("empty resource profile");
    if(p.invocation.retirement!=Program::InvocationContract::NoRetirement) return fail("phase invocation-retirement adapter is not qualified");
    if(std::any_of(p.target.synchronous.begin(),p.target.synchronous.end(),[](bool x) {
      return x;
    })) return fail("phase synchronous-engine adapter is not qualified");
    for(const auto &c:p.cells) {
      if(c.exclusive || c.unknownRange || c.addressSpace.empty() || c.ranges.size()!=1) return fail("phase contract requires exact nonexclusive canonical physical cells");
      const auto [base,size]=c.ranges[0];
      if(!size || base>std::numeric_limits<uint64_t>::max()-size) return fail("invalid or overflowing phase cell extent");
    }
    for(std::size_t a=0;a<p.cells.size();++a) for(std::size_t b=0;b<a;++b) {
      const auto &x=p.cells[a],&y=p.cells[b];
      if(x.addressSpace==y.addressSpace && std::max(x.ranges[0].first,y.ranges[0].first)< std::min(x.ranges[0].first+x.ranges[0].second,y.ranges[0].first+y.ranges[0].second)) return fail("unpartitioned physical overlap in phase profile");
    }
    std::set<unsigned> protectedCells;
    std::set<std::string> names;
    auto exactBlock=[&](unsigned c,const char *domain) {
      return c<p.cells.size() && p.cells[c].addressSpace==domain && p.cells[c].ranges[0].first%512==0 && p.cells[c].ranges[0].second==512;
    };
    for(const auto &g:profile.groups) {
      if(g.name.empty() || !names.insert(g.name).second || g.blocks.empty() || unsigned(g.producer)>=PipeCount || unsigned(g.consumer)>=PipeCount || g.producer==g.consumer || !p.target.supported[unsigned(g.producer)] || !p.target.supported[unsigned(g.consumer)]) return fail("invalid resource participants or group");
      for(auto b:g.blocks) if(!exactBlock(b,"ACC") || !protectedCells.insert(b).second) return fail("protected map must contain unique exact aligned ACC blocks");
    }
    using Effect=std::pair<bool,bool>;
    for(const auto &op:p.operations) {
      if(!op.complete || unsigned(op.pipe)>=PipeCount || !p.target.supported[unsigned(op.pipe)]) return fail("incomplete or unsupported phase operation");
      if(!op.resources.empty() || !op.visibility.empty() || !op.authoredEvents.empty() || !op.internalTransfers.empty()) return fail("unexplained resource, visibility or private event effects in phase operation");
      std::map<unsigned,Effect> actual;
      for(const auto &a:op.accesses) {
        if(a.cell>=p.cells.size() || (!a.read&&!a.write) || (a.definiteWrite&&!a.write)) return fail("invalid original phase effect");
        auto &e=actual[a.cell];
        e.first|=a.read;
        e.second|=a.write;
      }
      if(!op.finalBlock) {
        for(const auto &[cell,e]:actual) {
          (void)e;
          if(protectedCells.count(cell)) return fail("ordinary access to protected block");
        }
        continue;
      }
      const auto &f=*op.finalBlock;
      if((f.role!=FinalBlockOperation::Role::Producer && f.role!=FinalBlockOperation::Role::Consumer) || f.mode!="final" || f.layout!="exact-block-map-v1" || f.group>=profile.groups.size()) return fail("KEEP, inferred mode, unknown layout or resource group is outside supplied profile");
      const auto &g=profile.groups[f.group];
      const bool prod=f.role==FinalBlockOperation::Role::Producer;
      if(f.blocks!=g.blocks || op.pipe!=(prod?g.producer:g.consumer)) return fail("partial/permuted block map or wrong resource-role engine");
      std::map<unsigned,Effect> expected;
      for(auto b:g.blocks) expected[b]= {
        !prod,prod
      };
      if(prod) {
        if(f.operands.size()!=2 || !f.outputs.empty() || f.operands[0]>=p.cells.size() || f.operands[1]>=p.cells.size() || p.cells[f.operands[0]].addressSpace!="LEFT" || p.cells[f.operands[1]].addressSpace!="RIGHT") return fail("final producer needs complete LEFT/RIGHT reads and no accumulating read");
        for(auto c:f.operands) expected[c]= {
          true,false
        };
      }
      else {
        if(!f.operands.empty() || f.outputs.size()!=g.blocks.size()) return fail("final consumer needs one complete output per block");
        std::set<unsigned> seen;
        for(auto c:f.outputs) if(!exactBlock(c,"GM") || !seen.insert(c).second) return fail("consumer outputs must be unique exact aligned GM blocks");
        for(auto c:f.outputs) expected[c]= {
          false,true
        };
      }
      if(actual!=expected) return fail("original phase effects disagree with complete supplied signature");
    }
    return true;
  }
  inline PhaseFragment buildPhaseFragment(const Program &p,std::size_t id) {
    const auto &op=p.operations.at(id);
    PhaseFragment f;
    f.endpoints= {
      "I","C"
    };
    f.edges= {
      {
        0,1
      }
    };
    if(!op.finalBlock) {
      std::map<unsigned,Access> merged;
      for(const auto &a:op.accesses) {
        auto &b=merged[a.cell];
        b.cell=a.cell;
        b.read|=a.read;
        b.write|=a.write;
        b.definiteWrite|=a.definiteWrite;
      }
      for(const auto &[c,a]:merged) {
        (void)c;
        f.accesses.push_back( {
          a,0,1,false
        });
      }
      return f;
    }
    const auto &q=*op.finalBlock;
    bool prod=q.role==FinalBlockOperation::Role::Producer;
    if(prod) for(auto c:q.operands) f.accesses.push_back( {
      {
        c,true,false,false
      },0,1,false
    });
    for(std::size_t j=0;j<q.blocks.size();++j) {
      const auto b=f.endpoints.size(),e=b+1;
      f.endpoints.push_back("b"+std::to_string(j));
      f.endpoints.push_back("e"+std::to_string(j));
      f.edges.insert(f.edges.end(), {
        {
          0,b
        }, {
          b,e
        }, {
          e,1
        }
      });
      f.accesses.push_back( {
        {
          q.blocks[j],!prod,prod,prod
        },b,e,true
      });
      if(!prod) {
        const auto ob=f.endpoints.size(),oe=ob+1;
        f.endpoints.push_back("o"+std::to_string(j)+"b");
        f.endpoints.push_back("o"+std::to_string(j)+"e");
        // Streaming identity copy: read END -> output END, not output BEGIN.
        f.edges.insert(f.edges.end(), {
          {
            0,ob
          }, {
            ob,oe
          }, {
            oe,1
          }, {
            e,oe
          }
        });
        f.accesses.push_back( {
          {
            q.outputs[j],false,true,true
          },ob,oe,false
        });
      }
    }
    return f;
  }
}
// namespace mlir::pto::oahs::detail
#endif
