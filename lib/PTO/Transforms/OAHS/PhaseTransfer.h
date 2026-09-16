// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_PHASE_TRANSFER_H
#define PTO_OAHS_PHASE_TRANSFER_H
#include "Control.h"
#include "PhaseModel.h"
#include <deque>
#include <map>
#include <set>
#include <tuple>
namespace mlir::pto::oahs::detail::phase {
  using Bits=std::vector<uint64_t>;
  using Rows=std::vector<Bits>;
  using Family=std::vector<Bits>;
  using Pair=std::pair<Bits,Bits>;
  using Key=std::tuple<Pipe,Pipe,unsigned>;
  inline Bits zeros(std::size_t n) {
    return Bits((n+63)/64);
  }
  inline void put(Bits &b,std::size_t i) {
    b.at(i/64)|=uint64_t(1)<<(i%64);
  }
  inline bool has(const Bits &b,std::size_t i) {
    return i/64<b.size() && ((b[i/64]>>(i%64))&1);
  }
  inline void joinBits(Bits &a,const Bits &b) {
    for(std::size_t i=0;i<a.size();++i)a[i]|=b[i];
  }
  inline bool subset(const Bits &a,const Bits &b) {
    for(std::size_t i=0;i<a.size();++i)if(a[i]&~b[i])return false;
    return true;
  }
  inline Family minimize(Family all) {
    std::sort(all.begin(),all.end());
    all.erase(std::unique(all.begin(),all.end()),all.end());
    Family out;
    for(std::size_t i=0;i<all.size();++i) {
      bool dominated=false;
      for(std::size_t j=0;j<all.size();++j)if(i!=j&&subset(all[j],all[i])) {
        dominated=true;
        break;
      }
      if(!dominated)out.push_back(all[i]);
    }
    return out;
  }
  inline std::vector<Pair> minimizePairs(std::vector<Pair> all) {
    std::sort(all.begin(),all.end());
    all.erase(std::unique(all.begin(),all.end()),all.end());
    std::vector<Pair> out;
    for(std::size_t i=0;i<all.size();++i) {
      bool dominated=false;
      for(std::size_t j=0;j<all.size();++j)if(i!=j&&subset(all[i].first,all[j].first)&&subset(all[j].second,all[i].second)) {
        dominated=true;
        break;
      }
      if(!dominated)out.push_back(all[i]);
    }
    return out;
  }
  inline void close(Rows &r) {
    for(std::size_t k=0;k<r.size();++k)for(auto &row:r)if(has(row,k))joinBits(row,r[k]);
  }
  inline Rows extend(const Rows &old,std::size_t count) {
    Rows r=old;
    const auto n=r.size()+count;
    for(auto &row:r)row.resize((n+63)/64);
    while(r.size()<n) {
      r.push_back(zeros(n));
      put(r.back(),r.size()-1);
    }
    return r;
  }
  inline Bits project(const Rows &r,const std::vector<std::size_t> &mapping,std::size_t source) {
    auto out=zeros(mapping.size());
    if(source==NoAnalysisId)return out;
    for(std::size_t i=0;i<mapping.size();++i)if(mapping[i]!=NoAnalysisId&&has(r[source],mapping[i]))put(out,i);
    return out;
  }
  inline Bits expandSignature(const Bits &s,const Rows &r,std::size_t ports) {
    auto out=zeros(r.size());
    for(std::size_t i=0;i<ports;++i)if(has(s,i))joinBits(out,r[i]);
    return out;
  }
  inline Bits image(const Bits &s,const Rows &r,const std::vector<std::size_t>&mapping) {
    const auto full=expandSignature(s,r,mapping.size());
    auto out=zeros(mapping.size());
    for(std::size_t i=0;i<mapping.size();++i)if(mapping[i]!=NoAnalysisId&&has(full,mapping[i]))put(out,i);
    return out;
  }
  struct State {
    Rows reach;
    std::vector<uint8_t> live,permission;
    // 0 empty/writable; 1 full/readable; 2 poisoned
    std::vector<Family> history;
    Rows reference;
    std::vector<Pair> pairs;
    bool operator<(const State &b) const {
      return std::tie(reach,live,permission,history,reference,pairs)<std::tie(b.reach,b.live,b.permission,b.history,b.reference,b.pairs);
    }
  };
  struct Origin {
    std::size_t operation,access;
  };
  struct Issues {
    std::vector<CompletionRequirement> memory;
    std::vector<ProtocolObligation> protocol;
    std::vector<PhaseResourceObligation> resources;
    std::vector<PhaseOrderWitness> order;
  };
  // Whole-state collecting extension of the SUPPLIED phase contract, not an
  // automatic fallback from the compact ordinary domain. Its potentially large
  // state space is explicit. No dynamic occurrence expansion or numerical cutoff.
  class Machine {
    public: const Program &p;
    std::vector<Key> keys;
    std::map<Key,std::size_t> keyIds;
    std::vector<unsigned> blocks;
    std::map<unsigned,std::size_t> blockIds;
    std::vector<PhaseFragment> fragments;
    std::vector<Origin> origins;
    std::vector<std::vector<std::size_t>> accessIds;
    std::vector<std::size_t> completionIds;
    const bool order;
    std::size_t m,n;
    Machine(const Program &program,const Commands &commands,bool quality=false):p(program),order(quality) {
      std::set<Key> kk;
      for(const auto &word:commands)for(const auto &c:word)if(c.kind==Command::Publish||c.kind==Command::Acquire)kk.emplace(c.source,c.observer,c.key);
      keys.assign(kk.begin(),kk.end());
      for(std::size_t i=0;i<keys.size();++i)keyIds[keys[i]]=i;
      for(const auto &g:p.finalBlocks->groups)for(auto c:g.blocks) {
        blockIds[c]=blocks.size();
        blocks.push_back(c);
      }
      for(std::size_t i=0;i<p.operations.size();++i) {
        fragments.push_back(buildPhaseFragment(p,i));
        accessIds.emplace_back();
        for(std::size_t a=0;a<fragments.back().accesses.size();++a) {
          accessIds.back().push_back(origins.size());
          origins.push_back( {
            i,a
          });
        }
      }
      for(std::size_t i=0;i<p.operations.size();++i) {
        completionIds.push_back(origins.size());
        origins.push_back( {
          i,NoAnalysisId
        });
      }
      m=2*PipeCount+2*keys.size()+blocks.size();
      n=PipeCount+2*p.cells.size();
    }
    std::size_t A(unsigned p_)const {
      return p_;
    }
    std::size_t T(unsigned p_)const {
      return PipeCount+p_;
    }
    std::size_t S(std::size_t e)const {
      return 2*PipeCount+e;
    }
    std::size_t D(std::size_t e)const {
      return 2*PipeCount+keys.size()+e;
    }
    std::size_t U(std::size_t b)const {
      return 2*PipeCount+2*keys.size()+b;
    }
    std::size_t H(unsigned cell,bool write)const {
      return PipeCount+2*cell+(write?1:0);
    }
    State initial()const {
      State s;
      s.reach.assign(m,zeros(m));
      s.live.assign(keys.size(),0);
      s.permission.assign(blocks.size(),0);
      s.history.resize(origins.size());
      auto active=zeros(m);
      for(std::size_t i=0;i<m;++i)if(i<2*PipeCount||i>=2*PipeCount+keys.size())put(active,i);
      for(std::size_t i=0;i<m;++i)if(has(active,i))s.reach[i]=active;
      if(order) {
        s.reference.assign(n,zeros(n));
        for(auto &row:s.reference)for(std::size_t j=0;j<n;++j)put(row,j);
      }
      return s;
    }
    static void mergeSnapshot(std::optional<BoundaryFacts> &a,BoundaryFacts b) {
      if(!a) {
        a=std::move(b);
        return;
      }
      for(unsigned q=0;q<PipeCount;++q)for(std::size_t i=0;i<b.pending[q].size();++i)a->pending[q][i]|=b.pending[q][i];
      for(std::size_t e=0;e<b.events.size();++e) {
        auto &x=a->events[e];
        const auto &y=b.events[e];
        x.possibleOccupancy|=y.possibleOccupancy;
        x.receiptValidOnEveryPath&=y.receiptValidOnEveryPath;
        x.consumptionKnownAt&=y.consumptionKnownAt;
        for(std::size_t i=0;i<x.uncoveredOperations.size();++i)x.uncoveredOperations[i]|=y.uncoveredOperations[i];
        for(std::size_t i=0;i<x.carriedConsumptions.size();++i)x.carriedConsumptions[i]&=y.carriedConsumptions[i];
      }
      for(std::size_t u=0;u<b.phaseResources.size();++u) {
        auto &x=a->phaseResources[u];
        const auto &y=b.phaseResources[u];
        x.possiblePermission|=y.possiblePermission;
        for(std::size_t i=0;i<x.completedOperations.size();++i)x.completedOperations[i]&=y.completedOperations[i];
        for(std::size_t i=0;i<x.carriedConsumptions.size();++i)x.carriedConsumptions[i]&=y.carriedConsumptions[i];
      }
    }
    bool covers(const Family &f,std::size_t port)const {
      for(const auto &s:f)if(!has(s,port))return false;
      return true;
    }
    BoundaryFacts snapshot(const State &s)const {
      BoundaryFacts f;
      for(unsigned q=0;q<PipeCount;++q)for(auto h:completionIds)f.pending[q].push_back(!covers(s.history[h],A(q)));
      for(std::size_t e=0;e<keys.size();++e) {
        EventFacts v;
        v.possibleOccupancy=s.live[e]==2?3:1u<<s.live[e];
        v.receiptValidOnEveryPath=s.live[e]==1;
        for(auto h:completionIds)v.uncoveredOperations.push_back(!v.receiptValidOnEveryPath||!covers(s.history[h],S(e)));
        for(unsigned q=0;q<PipeCount;++q)if(s.live[e]!=2 && has(s.reach[D(e)],A(q)))v.consumptionKnownAt|=1u<<q;
        for(std::size_t k=0;k<keys.size();++k)v.carriedConsumptions.push_back(v.receiptValidOnEveryPath&&s.live[k]!=2&&has(s.reach[D(k)],S(e)));
        f.events.push_back(std::move(v));
      }
      for(std::size_t u=0;u<blocks.size();++u) {
        PhaseResourceFacts v;
        v.cell=blocks[u];
        v.possiblePermission=s.permission[u]==2?3:1u<<s.permission[u];
        for(auto h:completionIds)v.completedOperations.push_back(s.permission[u]!=2&&covers(s.history[h],U(u)));
        for(std::size_t k=0;k<keys.size();++k)v.carriedConsumptions.push_back(s.permission[u]!=2&&s.live[k]!=2&&has(s.reach[D(k)],U(u)));
        f.phaseResources.push_back(std::move(v));
      }
      return f;
    }
    // Internal test hooks are not accepted by Program or serialized commands.
    // Each pair adds an actual OLD port -> fresh endpoint, or a within-fragment edge.
    State step(const State &s,Cut cut,std::size_t operation,const Command *cmd,std::size_t commandIndex, Issues &issues,bool provisional=false, bool skipRearm=false, const std::vector<std::pair<std::size_t,std::size_t>>& oldEdges= {
    }, const std::vector<std::pair<std::size_t,std::size_t>>& freshEdges= {
    })const {
      PhaseFragment f;
      unsigned pipe;
      if(cmd) {
        f.endpoints= {
          "I","C"
        };
        f.edges= {
          {
            0,1
          }
        };
        pipe=unsigned(cmd->kind==Command::Acquire?cmd->observer:cmd->source);
      }
      else {
        f=fragments.at(operation);
        pipe=unsigned(p.operations[operation].pipe);
      }
      auto r=extend(s.reach,f.endpoints.size()+1);
      const auto fresh=m,finish=fresh+1,aggregate=r.size()-1;
      for(const auto &[a,b]:f.edges)put(r[fresh+a],fresh+b);
      put(r[A(pipe)],fresh);
      put(r[T(pipe)],aggregate);
      put(r[finish],aggregate);
      std::vector<std::size_t> map(m);
      for(std::size_t i=0;i<m;++i)map[i]=i;
      map[A(pipe)]=fresh;
      map[T(pipe)]=aggregate;
      State out=s;
      std::size_t e=NoAnalysisId;
      bool invalid=false;
      auto problem=[&](ProtocolObligation::Kind kind,const char *text) {
        issues.protocol.push_back( {
          kind,cut,commandIndex,0, {
            cmd->source,cmd->observer,cmd->key,false
          },text
        });
        invalid=true;
      };
      if(cmd) {
        if(cmd->kind==Command::Barrier) {
          put(r[T(pipe)],finish);
          map[A(pipe)]=finish;
        }
        else {
          e=keyIds.at( {
            cmd->source,cmd->observer,cmd->key
          });
          if(!provisional) {
            if(cmd->kind==Command::Publish && s.live[e]!=0)problem(ProtocolObligation::PublicationNotEmpty,"publication has no empty-key proof");
            if(cmd->kind==Command::Acquire && s.live[e]!=1)problem(ProtocolObligation::AcquisitionNotFull,"acquisition has no live matching publication");
          }
          if(!invalid) {
            if(cmd->kind==Command::Publish) {
              put(r[T(pipe)],finish);
              map[S(e)]=finish;
              out.live[e]=1;
            }
            else {
              put(r[S(e)],finish);
              map[S(e)]=NoAnalysisId;
              map[D(e)]=finish;
              map[A(pipe)]=finish;
              out.live[e]=0;
            }
          }
        }
      }
      for(const auto &a:f.accesses)if(a.permission) {
        auto u=blockIds.at(a.effect.cell);
        const bool read=a.effect.read;
        if(s.permission[u]!=(read?1:0)) {
          issues.resources.push_back( {
            PhaseResourceObligation::WrongRole,cut,operation,a.effect.cell,"resource role not justified by the current permission generation"
          });
          out.permission[u]=2;
          // No resource edge or later permission credit from an invalid request.
          map[U(u)]=NoAnalysisId;
        }
        else {
          put(r[U(u)],fresh+a.begin);
          map[U(u)]=fresh+a.end;
          out.permission[u]=read?0:1;
        }
      }
      for(const auto &[a,b]:oldEdges)put(r.at(a),fresh+b);
      for(const auto &[a,b]:freshEdges)put(r.at(fresh+a),fresh+b);
      close(r);
      if(cmd && e!=NoAnalysisId && cmd->kind==Command::Publish && !provisional && !skipRearm && !invalid && !has(r[D(e)],finish)) problem(ProtocolObligation::ConsumptionNotEstablished,"latest consumption does not causally precede republication");
      if(invalid) {
        // Rebuild with no completion-transfer edge from this unproved primitive.
        // An independent fresh D forgets old acknowledgment knowledge. Poison is
        // sticky for this invalid key; a later endpoint cannot rehabilitate it by name.
        r=extend(s.reach,4);
        const auto i=m,fc=m+1,tc=m+2,unknown=m+3;
        put(r[A(pipe)],i);
        put(r[i],fc);
        put(r[T(pipe)],tc);
        put(r[fc],tc);
        close(r);
        for(std::size_t j=0;j<m;++j)map[j]=j;
        map[A(pipe)]=i;
        map[T(pipe)]=tc;
        map[S(e)]=NoAnalysisId;
        map[D(e)]=unknown;
        out.live[e]=2;
      }
      for(std::size_t k=0;k<keys.size();++k)if(out.live[k]!=1)map[S(k)]=NoAnalysisId;
      // Query every original conflict at its true begin endpoint. No required edge
      // is added to the actual graph to satisfy this test.
      if(!cmd) {
        for(std::size_t ai=0;ai<f.accesses.size();++ai) {
          const auto &a=f.accesses[ai];
          for(std::size_t h=0;h<origins.size();++h) {
            const auto &o=origins[h];
            if(o.access==NoAnalysisId)continue;
            const auto &old=fragments[o.operation].accesses[o.access];
            if(old.effect.cell!=a.effect.cell || !(old.effect.write||a.effect.write))continue;
            bool missing=false;
            for(const auto &sig:s.history[h])if(!has(expandSignature(sig,r,m),fresh+a.begin)) {
              missing=true;
              break;
            }
            if(!missing)continue;
            auto add=[&](CompletionRequirement::Kind kind) {
              CompletionRequirement q;
              q.kind=kind;
              q.demand= {
                o.operation,operation,a.effect.cell,Property::ByteCompletion
              };
              q.producerAccess=old.effect;
              q.consumerAccess=a.effect;
              q.consumerCut=cut;
              q.producerEndpoint=old.end;
              q.consumerEndpoint=a.begin;
              issues.memory.push_back(std::move(q));
            };
            if(old.effect.write&&a.effect.read)add(CompletionRequirement::RAW);
            if(old.effect.read&&a.effect.write)add(CompletionRequirement::WAR);
            if(old.effect.write&&a.effect.write)add(CompletionRequirement::WAW);
          }
        }
      }
      for(std::size_t h=0;h<s.history.size();++h) {
        Family bucket;
        for(const auto &sig:s.history[h])bucket.push_back(image(sig,r,map));
        out.history[h]=minimize(std::move(bucket));
      }
      if(!cmd) {
        for(std::size_t a=0;a<f.accesses.size();++a) {
          auto &bucket=out.history[accessIds[operation][a]];
          bucket.push_back(project(r,map,fresh+f.accesses[a].end));
          bucket=minimize(std::move(bucket));
        }
        auto &bucket=out.history[completionIds[operation]];
        bucket.push_back(project(r,map,finish));
        bucket=minimize(std::move(bucket));
      }
      if(order) {
        Rows rr=s.reference;
        std::vector<std::size_t> rm(n);
        for(std::size_t i=0;i<n;++i)rm[i]=i;
        std::vector<std::size_t> rid;
        if(!cmd) {
          rr=extend(rr,f.endpoints.size());
          for(std::size_t i=0;i<f.endpoints.size();++i)rid.push_back(n+i);
          for(const auto &[a,b]:f.edges)put(rr[rid[a]],rid[b]);
          put(rr[pipe],rid[0]);
          rm[pipe]=rid[0];
          for(const auto &a:f.accesses) {
            put(rr[rm[H(a.effect.cell,true)]],rid[a.begin]);
            if(a.effect.write)put(rr[rm[H(a.effect.cell,false)]],rid[a.begin]);
            for(unsigned mode=0;mode<2;++mode)if(mode?a.effect.write:a.effect.read) {
              auto h=H(a.effect.cell,mode);
              const auto newh=rr.size();
              rr=extend(rr,1);
              put(rr[rm[h]],newh);
              put(rr[rid[a.end]],newh);
              rm[h]=newh;
            }
          }
          close(rr);
          for(const auto &[act,ideal]:s.pairs) {
            const auto ar=expandSignature(act,r,m),ir=expandSignature(ideal,rr,n);
            for(std::size_t v=0;v<f.endpoints.size();++v)if(has(ar,fresh+v)&&!has(ir,rid[v]))issues.order.push_back( {
              cut,operation,f.endpoints[v],false
            });
          }
          for(std::size_t a=0;a<f.endpoints.size();++a)for(std::size_t b=0;b<f.endpoints.size();++b) if(a!=b&&has(r[fresh+a],fresh+b)&&!has(rr[rid[a]],rid[b]))issues.order.push_back( {
            cut,operation,f.endpoints[b],true
          });
        }
        std::vector<Pair> pairs;
        for(const auto &[a,b]:s.pairs)pairs.emplace_back(image(a,r,map),image(b,rr,rm));
        if(!cmd)for(std::size_t v=0;v<f.endpoints.size();++v)pairs.emplace_back(project(r,map,fresh+v),project(rr,rm,rid[v]));
        out.pairs=minimizePairs(std::move(pairs));
        out.reference.clear();
        for(auto v:rm)out.reference.push_back(project(rr,rm,v));
      }
      out.reach.clear();
      for(auto v:map)out.reach.push_back(project(r,map,v));
      return out;
    }
  };
  struct Collected {
    AnalysisResult analysis;
    std::vector<PhaseOrderWitness> excess;
  };
  inline Collected collect(const Program &p,const Commands &commands,AnalysisOptions options= {
  },bool order=false,bool provisional=false,bool skipRearm=false) {
    Collected result;
    auto &out=result.analysis;
    Machine m(p,commands,order);
    auto graph=buildControlGraph(p);
    // A static operation may occur in several original analysis contexts.
    // Preserve a unique source context only when it is unambiguous; origin
    // signatures do not establish a correlated producer occurrence by themselves.
    std::vector<std::size_t> sourceContexts(p.operations.size(), NoAnalysisId);
    std::vector<bool> sourceSeen(p.operations.size(), false), sourceAmbiguous(p.operations.size(), false);
    for (std::size_t at = 0; at < graph.operations.size(); ++at) {
      const auto op = graph.operations[at];
      if (op == NoAnalysisId) continue;
      const auto context = graph.cutContexts[at];
      if (!sourceSeen[op]) sourceContexts[op] = context;
      else if (sourceContexts[op] != context) sourceAmbiguous[op] = true;
      sourceSeen[op] = true;
    }
    for (std::size_t op = 0; op < sourceContexts.size(); ++op)
      if (sourceAmbiguous[op]) sourceContexts[op] = NoAnalysisId;
    out.contexts=graph.contexts;
    out.cuts.resize(commands.size());
    out.stats.staticSites=graph.sites.size();
    out.stats.certificationPasses=1;
    for(auto [a,b,k]:m.keys)out.keys.push_back( {
      a,b,k,false
    });
    std::vector<std::vector<bool>> commandOK(commands.size());
    for(Cut c=0;c<commands.size();++c) {
      out.cuts[c].cut=c;
      out.cuts[c].context=graph.cutContexts[c];
      commandOK[c].assign(commands[c].size(),true);
    }
    std::vector<std::set<State>> seen(graph.sites.size());
    std::deque<std::pair<Cut,State>> queue;
    auto initial=m.initial();
    seen[graph.entry].insert(initial);
    queue.emplace_back(graph.entry,std::move(initial));
    std::set<std::tuple<Cut,std::size_t,std::size_t,unsigned,unsigned,std::size_t,std::size_t>> residualKeys;
    std::set<std::tuple<Cut,std::size_t,unsigned>> protocolKeys;
    std::set<std::tuple<Cut,std::size_t,unsigned,unsigned>> resourceKeys;
    std::set<std::tuple<Cut,std::size_t,std::string,bool>> orderKeys;
    auto append=[&](Issues &x,Cut at) {
      for(auto &r:x.memory) {
        r.producerContext=sourceContexts[r.demand.producer];
        r.consumerContext=graph.cutContexts[at];
        auto key=std::make_tuple(at,r.demand.producer,r.demand.consumer,r.demand.cell,unsigned(r.kind),r.producerEndpoint,r.consumerEndpoint);
        if(residualKeys.insert(key).second)out.residuals.push_back(std::move(r));
      }
      for(auto &r:x.protocol) {
        r.context=graph.cutContexts[at];
        if(protocolKeys.emplace(at,r.command,unsigned(r.kind)).second)out.protocol.push_back(std::move(r));
      }
      for(auto &r:x.resources)if(resourceKeys.emplace(at,r.operation,r.cell,unsigned(r.kind)).second)out.phaseResources.push_back(std::move(r));
      for(auto &r:x.order)if(orderKeys.emplace(at,r.operation,r.target,r.withinOperation).second)result.excess.push_back(std::move(r));
    };
    while(!queue.empty()) {
      auto [at,s]=std::move(queue.front());
      queue.pop_front();
      ++out.stats.siteEvaluations;
      if(at<commands.size()) {
        auto &facts=out.cuts[at];
        facts.reachable=true;
        if(options.captureStates)Machine::mergeSnapshot(facts.incoming,m.snapshot(s));
        for(std::size_t c=0;c<commands[at].size();++c) {
          Issues x;
          s=m.step(s,at,NoAnalysisId,&commands[at][c],c,x,provisional,skipRearm);
          if(!x.protocol.empty())commandOK[at][c]=false;
          append(x,at);
        }
        if(options.captureStates)Machine::mergeSnapshot(facts.beforeIssue,m.snapshot(s));
      }
      const auto op=graph.operations[at];
      if(op!=NoAnalysisId) {
        Issues x;
        s=m.step(s,at,op,nullptr,NoAnalysisId,x,provisional,skipRearm);
        append(x,at);
      }
      if(at==graph.exit) {
        Issues x;
        for(std::size_t e=0;e<m.keys.size();++e)if(s.live[e]!=0)x.protocol.push_back( {
          ProtocolObligation::UnconsumedAtExit,at,NoAnalysisId,0,out.keys[e],"software event not empty at closed phase exit"
        });
        for(std::size_t u=0;u<m.blocks.size();++u)if(s.permission[u]!=0)x.resources.push_back( {
          PhaseResourceObligation::UnconsumedBlock,at,NoAnalysisId,m.blocks[u],"protected block not writable at closed exit"
        });
        append(x,at);
      }
      if(at<commands.size()&&options.captureStates)Machine::mergeSnapshot(out.cuts[at].outgoing,m.snapshot(s));
      for(auto next:graph.sites[at].successors)if(seen[next].insert(s).second) {
        queue.emplace_back(next,s);
        ++out.stats.merges;
      }
    }
    for(Cut at=0;at<commands.size();++at)if(out.cuts[at].reachable)for(std::size_t c=0;c<commands[at].size();++c) out.commands.push_back( {
      at,c,graph.cutContexts[at],commands[at][c],commandOK[at][c]
    });
    out.complete=true;
    if(!out.phaseResources.empty())out.reason=out.phaseResources.front().reason;
    else if(!out.protocol.empty())out.reason=out.protocol.front().reason;
    else if(!out.residuals.empty())out.reason="uncovered original phase-access requirements";
    for(const auto &s:seen) {
      out.stats.phaseStateCount+=s.size();
      out.stats.maxPhaseStatesPerSite=std::max(out.stats.maxPhaseStatesPerSite,s.size());
    }
    return result;
  }
}
// namespace mlir::pto::oahs::detail::phase
#endif
