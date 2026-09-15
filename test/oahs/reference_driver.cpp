// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Test-only serialization boundary for pinned reference differential checks.
#include "PTO/Transforms/OAHS/Analysis.h"
#include <iostream>
#include <string>
namespace o=mlir::pto::oahs;
namespace {
bool readRegion(o::Region &r) {
 unsigned kind=0,zero=0;std::size_t children=0;
 if(!(std::cin>>kind>>r.operation>>zero>>children)||kind>o::Region::Operation||zero>1)return false;
 r.kind=o::Region::Kind(kind);r.zeroTripPossible=zero;r.children.resize(children);
 for(auto &c:r.children)if(!readRegion(c))return false;
 return true;
}
void quoted(const std::string &s) {
 std::cout<<'"';for(unsigned char c:s){
  if(c=='"'||c=='\\')std::cout<<'\\'<<c;
  else if(c=='\n')std::cout<<"\\n";
  else if(c=='\r')std::cout<<"\\r";
  else if(c=='\t')std::cout<<"\\t";
  else if(c<32)std::cout<<'?';else std::cout<<c;
 }std::cout<<'"';
}
void bits(const o::AnalysisBits &b) {std::cout<<'[';for(std::size_t i=0;i<b.size();++i){if(i)std::cout<<',';std::cout<<unsigned(b[i]);}std::cout<<']';}
void fact(const std::optional<o::BoundaryFacts> &s) {
 if(!s){std::cout<<"null";return;}
 std::cout<<"{\"pending\":[";for(unsigned q=0;q<o::PipeCount;++q){if(q)std::cout<<',';bits(s->pending[q]);}
 std::cout<<"],\"events\":[";bool first=true;for(auto &e:s->events){if(!first)std::cout<<',';first=false;
  std::cout<<"{\"o\":"<<unsigned(e.possibleOccupancy)<<",\"v\":"<<(e.receiptValidOnEveryPath?"true":"false")<<",\"u\":";bits(e.uncoveredOperations);
  std::cout<<",\"j\":";bits(e.carriedConsumptions);std::cout<<",\"k\":"<<unsigned(e.consumptionKnownAt)<<'}';}
 std::cout<<"]}";
}
}
int main() {
 unsigned version=0;std::size_t n=0,cells=0;
 if(!(std::cin>>version>>n>>cells)||version!=1)return 2;
 o::Program p;p.cells.resize(cells);p.operations.resize(n);p.target.contract="reference bridge declared core";
 for(auto *population:{&p.target.supported,&p.target.barriers,&p.target.synchronous})
  for(auto &b:*population){unsigned x;if(!(std::cin>>x)||x>1)return 2;b=x;}
 unsigned all=0,retirement=0;if(!(std::cin>>all>>retirement)||all>1||retirement>1)return 2;
 p.target.barrierAll=all;p.invocation.retirement=o::Program::InvocationContract::Retirement(retirement);
 std::size_t keys=0;if(!(std::cin>>keys))return 2;
 for(std::size_t i=0;i<keys;++i){unsigned a,b,k;if(!(std::cin>>a>>b>>k)||a>=o::PipeCount||b>=o::PipeCount)return 2;p.target.keys[a][b].push_back(k);}
 if(!(std::cin>>keys))return 2;
 for(std::size_t i=0;i<keys;++i){unsigned a,b,k;if(!(std::cin>>a>>b>>k)||a>=o::PipeCount||b>=o::PipeCount)return 2;p.reservations.push_back({o::Pipe(a),o::Pipe(b),k,false});}
 for(std::size_t i=0;i<n;++i){unsigned lane;std::size_t accesses;if(!(std::cin>>lane>>accesses))return 2;
  auto &a=p.operations[i];a.pipe=o::Pipe(lane);a.original=i;a.complete=true;
  for(std::size_t j=0;j<accesses;++j){unsigned cell,read,write;if(!(std::cin>>cell>>read>>write)||read>1||write>1)return 2;a.accesses.push_back({cell,bool(read),bool(write)});}
 }
 if(!readRegion(p.body))return 2;
 unsigned generate=0;if(!(std::cin>>generate)||generate>1)return 2;
 o::Commands commands(n+1);bool success=false;std::string constructionReason;
 if(generate){auto plan=o::construct(p);success=plan.success;constructionReason=plan.reason;if(success)commands=std::move(plan.commands);}
 else for(auto &at:commands){std::size_t count;if(!(std::cin>>count))return 2;
  for(std::size_t i=0;i<count;++i){unsigned k,a,b,e;if(!(std::cin>>k>>a>>b>>e))return 2;at.push_back({o::Command::Kind(k),o::Pipe(a),o::Pipe(b),e});}
 }
 std::string trailing;if(std::cin>>trailing)return 2;
 const auto analysis=o::analyze(p,commands);
 std::cout<<"{\"schema\":\"oahs.compact-report.v1\",\"constructed\":"<<(success?"true":"false")<<",\"construction_reason\":";quoted(constructionReason);
 std::cout<<",\"complete\":"<<(analysis.complete?"true":"false")<<",\"verified\":"<<(analysis.verified()?"true":"false")<<",\"reason\":";quoted(analysis.reason);
 std::cout<<",\"commands\":[";
 for(std::size_t at=0;at<commands.size();++at){if(at)std::cout<<',';std::cout<<'[';
  for(std::size_t j=0;j<commands[at].size();++j){if(j)std::cout<<',';const auto &x=commands[at][j];std::cout<<'['<<x.kind<<','<<unsigned(x.source)<<','<<unsigned(x.observer)<<','<<x.key<<']';}std::cout<<']';}
 std::cout<<"],\"keys\":[";for(std::size_t i=0;i<analysis.keys.size();++i){if(i)std::cout<<',';auto &k=analysis.keys[i];std::cout<<'['<<unsigned(k.source)<<','<<unsigned(k.observer)<<','<<k.key<<']';}
 std::cout<<"],\"cuts\":[";for(std::size_t i=0;i<analysis.cuts.size();++i){if(i)std::cout<<',';auto &c=analysis.cuts[i];
  std::cout<<"{\"reachable\":"<<(c.reachable?"true":"false")<<",\"incoming\":";fact(c.incoming);std::cout<<",\"before\":";fact(c.beforeIssue);std::cout<<",\"outgoing\":";fact(c.outgoing);std::cout<<'}';}
 std::cout<<"],\"residuals\":[";for(std::size_t i=0;i<analysis.residuals.size();++i){if(i)std::cout<<',';auto &x=analysis.residuals[i];std::cout<<'['<<x.kind<<','<<x.demand.producer<<','<<x.demand.consumer<<','<<x.demand.cell<<']';}
 std::cout<<"],\"protocol\":[";for(std::size_t i=0;i<analysis.protocol.size();++i){if(i)std::cout<<',';auto &x=analysis.protocol[i];std::cout<<'['<<x.kind<<','<<x.cut<<','<<unsigned(x.event.source)<<','<<unsigned(x.event.observer)<<','<<x.event.key<<']';}
 std::cout<<"]}\n";
}
