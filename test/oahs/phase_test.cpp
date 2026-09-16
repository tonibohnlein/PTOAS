// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PhaseFixtures.h"
#include "PTO/Transforms/OAHS/Bundles.h"
#include "../../lib/PTO/Transforms/OAHS/PhaseTransfer.h"
#include <cstdlib>
#include <iostream>
namespace o=mlir::pto::oahs;
using namespace phase_fixture;
unsigned assertions=0;
void check(bool ok,unsigned line) {
  ++assertions;
  if(!ok) {
    std::cerr<<"phase assertion failed at "<<line<<'\n';
    std::abort();
  }
}
#define CHECK(x) check(bool(x),__LINE__)
int main() {
  {
    auto p=program();
    p.operations= {
      producer(p),consumer(p)
    };
    auto a=o::analyze(p);
    CHECK(a.verified());
    CHECK(!a.cuts[1].incoming->phaseResources[0].completedOperations[0]);
    CHECK(a.cuts[2].outgoing->phaseResources[0].possiblePermission==1);
    CHECK(a.cuts[2].outgoing->pending[F][0]);
    // block ready is not full M completion
    auto q=o::checkPhaseOrder(p,o::Commands(3));
    CHECK(q.safe&&q.exact);
    auto c=o::Commands(3);
    c[1]=handoff(M,F);
    q=o::checkPhaseOrder(p,c);
    CHECK(q.safe&&!q.exact&&!q.excess.empty());
    o::PhaseFragment fragment;
    std::string why;
    CHECK(o::phaseFragment(p,1,fragment,why));
    // Streaming dataflow end->end, never whole read end->output begin.
    CHECK(std::find(fragment.edges.begin(),fragment.edges.end(),std::make_pair(std::size_t(3),std::size_t(5)))!=fragment.edges.end());
    CHECK(std::find(fragment.edges.begin(),fragment.edges.end(),std::make_pair(std::size_t(3),std::size_t(4)))==fragment.edges.end());
  }
  {
    auto p=program();
    p.operations= {
      ordinary(L, {
      }, {
        0,1
      }),producer(p),consumer(p)
    };
    auto a=o::analyze(p);
    CHECK(a.complete&&!a.verified()&&a.residuals.size()>=2);
    o::Commands c(4);
    c[1]=handoff(L,M);
    CHECK(o::analyze(p,c).verified());
    CHECK(o::checkPhaseOrder(p,c).exact);
  }
  {
    auto p=program();
    p.operations= {
      producer(p),consumer(p),ordinary(Q, {
        5,6
      })
    };
    o::Commands c(4);
    CHECK(!o::analyze(p,c).verified());
    c[2]=handoff(F,Q);
    CHECK(o::analyze(p,c).verified());
    o::BundleQuery b(p,o::Commands(4));
    auto r=b.evaluate(c);
    CHECK(r.complete&&r.analysis.verified()&&!r.discharged.empty());
    for(auto &x:r.discharged)CHECK(x.producerEndpoint!=o::NoAnalysisId&&x.consumerEndpoint==0);
  }
  {
    auto p=program();
    p.operations= {
      producer(p),consumer(p),ordinary(M, {
      }, {
        0
      })
    };
    o::Commands c(4);
    c[2]=handoff(F,M);
    auto a=o::analyze(p,c);
    CHECK(!a.verified());
    bool operand=false;
    for(auto r:a.residuals)operand|=r.demand.cell==0;
    CHECK(operand);
    c[2].push_back( {
      o::Command::Barrier,o::Pipe(M)
    });
    CHECK(o::verify(p,c).success);
  }
  {
    auto p=program();
    p.operations= {
      producer(p),ordinary(M, {
      }, {
        2
      }),consumer(p),ordinary(M, {
        2
      })
    };
    o::Commands c(5);
    c[3]=handoff(F,M);
    auto a=o::analyze(p,c);
    CHECK(!a.verified());
    CHECK(a.cuts[3].beforeIssue->pending[M][1]);
  }
  {
    auto p=program();
    p.operations= {
      producer(p),consumer(p)
    };
    loop(p);
    auto a=o::analyze(p);
    CHECK(a.complete&&!a.verified()&&a.phaseResources.empty());
    for(auto &r:a.residuals)CHECK(r.demand.cell>=5);
    // only output WAW remains
    o::Commands c(3);
    c[1]= {
      {
        o::Command::Barrier,o::Pipe(F)
      }
    };
    CHECK(o::verify(p,c).success);
  }
  {
    auto p=program();
    p.operations= {
      producer(p),consumer(p)
    };
    loop(p);
    o::Commands c(3);
    c[0]=handoff(Q,M);
    c[1]= {
      {
        o::Command::Barrier,o::Pipe(F)
      }
    };
    c[2]=handoff(F,Q);
    // Final exit is outside the body; put the return at a real body cut instead.
    p.operations.push_back(ordinary(Q));
    loop(p);
    c.resize(4);
    c[2]=handoff(F,Q);
    c[3].clear();
    auto a=o::analyze(p,c);
    CHECK(a.verified());
    CHECK(a.cuts[2].outgoing->events[0].consumptionKnownAt || a.cuts[2].outgoing->events[1].consumptionKnownAt);
    auto bad=c;
    bad[1].clear();
    a=o::analyze(p,bad);
    CHECK(!a.verified()&&a.protocol.empty());
  }
  {
    auto p=program();
    p.operations= {
      producer(p),consumer(p),producer(p),consumer(p)
    };
    o::Commands c(5);
    c[2]= {
      {
        o::Command::Barrier,o::Pipe(F)
      }
    };
    CHECK(o::verify(p,c).success);
    p.operations[1]=producer(p);
    auto a=o::analyze(p,c);
    CHECK(a.complete&&!a.phaseResources.empty()&&!a.verified());
  }
  {
    auto p=program();
    p.operations= {
      producer(p)
    };
    auto a=o::analyze(p);
    CHECK(a.complete&&!a.verified()&&!a.phaseResources.empty());
    CHECK(a.phaseResources.front().kind==o::PhaseResourceObligation::UnconsumedBlock);
  }
  {
    auto p=program();
    p.operations= {
      ordinary(L, {
      }, {
        2
      }),ordinary(Q, {
        2
      })
    };
    o::Commands c(3);
    c[1]= {
      {
        o::Command::Acquire,o::Pipe(L),o::Pipe(Q),0
      }
    };
    auto a=o::analyze(p,c);
    CHECK(a.complete&&!a.verified()&&!a.protocol.empty()&&!a.residuals.empty());
    CHECK(a.cuts[1].beforeIssue->pending[Q][0]);
  }
  {
    auto p=program();
    p.operations= {
      producer(p),consumer(p)
    };
    std::string why;
    CHECK(o::validatePhaseContract(p,why));
    auto q=p;
    q.finalBlocks->orderedPerBlockService=false;
    CHECK(!o::validatePhaseContract(q,why));
    q=p;
    q.finalBlocks->qualification="native-qualified";
    CHECK(!o::validatePhaseContract(q,why));
    q=p;
    q.operations[0].finalBlock->mode="keep";
    CHECK(!o::validatePhaseContract(q,why));
    q=p;
    q.operations[0].accesses.pop_back();
    CHECK(!o::validatePhaseContract(q,why));
    q=p;
    q.cells[3].unknownRange=true;
    CHECK(!o::validatePhaseContract(q,why));
    q=p;
    q.operations[1].finalBlock->outputs= {
      5,5
    };
    CHECK(!o::validatePhaseContract(q,why));
    q=p;
    q.operations[0].finalBlock->blocks= {
      4,3
    };
    CHECK(!o::validatePhaseContract(q,why));
    q=p;
    q.cells[4].ranges[0].first=0;
    CHECK(!o::validatePhaseContract(q,why));
    q=p;
    q.operations.push_back(ordinary(L, {
      3
    }));
    CHECK(!o::validatePhaseContract(q,why));
    q=p;
    q.operations[0].finalBlock->role=static_cast<o::FinalBlockOperation::Role>(100);
    CHECK(!o::validatePhaseContract(q,why));
    q=p;
    q.finalBlocks.reset();
    CHECK(!o::analyze(q).complete);
    CHECK(!o::phaseNativeQualification().enabled&&!o::phaseNativeQualification().missing.empty());
    q = p; q.target.barrierAll = true;
    CHECK(!o::validatePhaseContract(q, why));
    q = p; q.target.synchronous[0] = true;
    CHECK(!o::validatePhaseContract(q, why));
    q = p; q.invocation.retirement = o::Program::InvocationContract::DrainAllAtReturn;
    CHECK(!o::validatePhaseContract(q, why));

  }
  {
    // Endpoint-complete monitor: isolated completion and intra-operation order.
    auto p=program();
    p.operations= {
      ordinary(M, {
      }, {
        2
      }),ordinary(Q, {
        0
      }),producer(p),consumer(p)
    };
    o::Commands c(5);
    using namespace o::detail::phase;
    Machine m(p,c,true);
    Issues x;
    auto s=m.step(m.initial(),0,0,nullptr,o::NoAnalysisId,x);
    CHECK(x.order.empty());
    x= {
    };
    (void)m.step(s,1,1,nullptr,o::NoAnalysisId,x,false,false, {
      {
        m.T(M),1
      }
    });
    CHECK(!x.order.empty()&&x.order.front().target=="C");
    x= {
    };
    s=m.step(m.initial(),2,2,nullptr,o::NoAnalysisId,x);
    CHECK(x.order.empty());
    x= {
    };
    (void)m.step(s,3,3,nullptr,o::NoAnalysisId,x,false,false, {
    }, {
      {
        3,6
      }
    });
    CHECK(!x.order.empty());
    // An extra access-begin predecessor can leave the operation's issue alone.
    x = {};
    s = m.step(m.initial(), 0, 0, nullptr, o::NoAnalysisId, x);
    s = m.step(s, 2, 2, nullptr, o::NoAnalysisId, x);
    CHECK(x.order.empty());
    x = {};
    (void)m.step(s, 3, 3, nullptr, o::NoAnalysisId, x, false, false,
                 {{m.T(M), 2}});
    CHECK(std::any_of(x.order.begin(), x.order.end(), [](const auto &w) {
      return w.target == "b0";
    }));
  }
  { // Independent supplied resource groups retain separate generations.
    auto p = program(2);
    auto other = p.finalBlocks->groups.front();
    p.finalBlocks->groups.front().blocks = {3};
    other.name = "second"; other.blocks = {4};
    p.finalBlocks->groups.push_back(other);
    auto make = [&](std::size_t g, bool prod) {
      o::Operation a = ordinary(prod ? M : F);
      o::FinalBlockOperation f;
      f.group = g; f.blocks = p.finalBlocks->groups[g].blocks;
      f.role = prod ? o::FinalBlockOperation::Role::Producer : o::FinalBlockOperation::Role::Consumer;
      if (prod) {
        f.operands = {0, 1};
        a.accesses = {{0, true, false}, {1, true, false}, {f.blocks[0], false, true}};
      } else {
        const auto output = unsigned(5 + g);
        f.outputs = {output}; a.accesses = {{f.blocks[0], true, false}, {output, false, true}};
      }
      a.finalBlock = f; return a;
    };
    p.operations = {make(0, true), make(1, true), make(0, false), make(1, false)};
    CHECK(o::analyze(p).verified());
    CHECK(o::checkPhaseOrder(p, o::Commands(5)).exact);
    std::swap(p.operations[0], p.operations[2]);
    CHECK(!o::analyze(p).verified());
  }
  for(unsigned blocks: {
    1u,2u,4u,8u
  }) {
    auto p=program(blocks);
    p.operations= {
      producer(p),consumer(p)
    };
    CHECK(o::checkPhaseOrder(p,o::Commands(3)).exact);
    loop(p);
    o::Commands c(3);
    c[1]= {
      {
        o::Command::Barrier,o::Pipe(F)
      }
    };
    CHECK(o::verify(p,c).success);
  }
  std::cout<<assertions<<" phase contract/causality assertions\n";
}
