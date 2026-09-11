// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/InsertSync/StructuredSyncCore.h"
#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <tuple>
#include <utility>

using namespace mlir::pto::structured_sync;
namespace {
bool hasBoundaries(const Model &m) {
    return std::any_of(m.atoms.begin(),m.atoms.end(),[](const Atom &a){return a.segment!=Segment::Body;});
}
Result constructRegion(const Model &);
Result verifyRegion(const Model &,const std::vector<Action> &);
bool suppliesRegion(const Model &,const Plan &,const Requirement &);
constexpr uint64_t Inf = std::numeric_limits<uint64_t>::max();
uint64_t plus(uint64_t a, uint64_t b) {
    return a == Inf || b == Inf || a > Inf - b ? Inf : a + b;
}
bool before(const Atom &a, const Atom &b) {
    return std::tie(a.segment, a.residue, a.order) < std::tie(b.segment, b.residue, b.order);
}

// A cell d represents the guarded cut p[k-d] complete at q[k]. Infinity is
// absence of a guarantee. Min/plus are exact for this prefix-periodic fragment.
// This is a finite graph closure, NOT an expanded integer occurrence relation.
class Cuts {
    std::size_t n;
    std::vector<std::vector<uint64_t>> cells;
public:
    uint64_t relaxations = 0;
    explicit Cuts(std::size_t size) : n(size) {
        cells.assign(n, std::vector<uint64_t>(n, Inf));
        for (std::size_t i = 0; i < n; ++i) cells[i][i] = 0;
    }
    uint64_t get(std::size_t a, std::size_t b) const { return cells[a][b]; }
    void edge(std::size_t a, std::size_t b, uint64_t d) {
        cells[a][b] = std::min(cells[a][b], d);
    }
    void close() {
        for (std::size_t k = 0; k < n; ++k)
            for (std::size_t i = 0; i < n; ++i) {
                if (get(i,k) == Inf) continue;
                for (std::size_t j = 0; j < n; ++j) {
                    if (get(k,j) == Inf) continue;
                    ++relaxations;
                    cells[i][j] = std::min(get(i,j), plus(get(i,k),get(k,j)));
                }
            }
    }
    // The old matrix is closed. With nonnegative distances a shortest path
    // need use this newly inserted edge at most once. Preserve old row/column
    // before updating, so this operation has a fixed finite O(n^2) loop.
    void insert(std::size_t a, std::size_t b, uint64_t d) {
        if (get(a,b) <= d) return;
        std::vector<uint64_t> incoming(n), outgoing(n);
        for (std::size_t i=0;i<n;++i) incoming[i]=get(i,a), outgoing[i]=get(b,i);
        for (std::size_t i=0;i<n;++i) {
            if (incoming[i] == Inf) continue;
            for (std::size_t j=0;j<n;++j) {
                if (outgoing[j] == Inf) continue;
                ++relaxations;
                cells[i][j] = std::min(get(i,j), plus(plus(incoming[i],d),outgoing[j]));
            }
        }
    }
};

bool validModel(const Model &m, std::string &why) {
    if (!m.period) { why="zero execution period"; return false; }
    if (m.atoms.size() > std::numeric_limits<std::size_t>::max()/2) {
        why="phase identity overflow"; return false;
    }
    std::set<std::tuple<Segment,uint64_t,uint64_t>> ranks;
    for (const Atom &a:m.atoms) {
        if (a.residue>=m.period || !m.target.supports(a.lane) ||
            (a.segment!=Segment::Prelude && a.segment!=Segment::Body && a.segment!=Segment::Epilogue) ||
            (a.segment!=Segment::Body && (!m.recurring || a.residue)) ||
            !ranks.insert({a.segment,a.residue,a.order}).second) {
            why="invalid physical lane, residue or original schedule rank"; return false;
        }
    }
    for (const Requirement &r:m.requirements) {
        if (r.property!=Property::Completion && r.property!=Property::AccResource &&
            r.property!=Property::Visibility) {
            why="unknown required property"; return false;
        }
        if (r.source>=m.atoms.size() || r.target>=m.atoms.size()) {
            why="invalid requirement endpoint"; return false;
        }
        auto d=priorDistance(m,r.source,r.target);
        if (!d || r.distance!=*d) {
            why="requirement outside latest ordinary periodic occurrence fragment"; return false;
        }
    }
    std::set<unsigned> keys;
    for (unsigned key:m.target.compilerKeys)
        if (key>=8 || !keys.insert(key).second) { why="invalid compiler event pool"; return false; }
    return true;
}
std::vector<std::size_t> schedule(const Model &m) {
    std::vector<std::size_t> ids(m.atoms.size());
    std::iota(ids.begin(),ids.end(),std::size_t(0));
    std::sort(ids.begin(),ids.end(),[&](auto a,auto b){return before(m.atoms[a],m.atoms[b]);});
    return ids;
}

// Two layers prevent issue order alone being used as completed work. Layer 0
// has not crossed a completing mechanism; layer 1 has. A barrier/handoff
// crosses 0->1 and can propagate already acquired facts within layer 1.
class Completion {
    const Model &m;
    std::size_t n;
    Cuts cuts;
public:
    explicit Completion(const Model &model) : m(model),n(m.atoms.size()),cuts(2*n) {
        std::map<Lane,std::vector<std::size_t>> lanes;
        for (auto p:schedule(m)) lanes[m.atoms[p].lane].push_back(p);
        for (const auto &entry:lanes) {
            const auto &v=entry.second;
            for (std::size_t i=1;i<v.size();++i) {
                cuts.edge(v[i-1],v[i],0); cuts.edge(n+v[i-1],n+v[i],0);
            }
            if (m.recurring && !v.empty()) {
                cuts.edge(v.back(),v.front(),1); cuts.edge(n+v.back(),n+v.front(),1);
            }
        }
        cuts.close();
    }
    void handoff(const Handoff &e) {
        cuts.insert(e.source,n+e.target,e.distance);
        cuts.insert(n+e.source,n+e.target,e.distance);
    }
    void barrier(std::size_t q) {
        for (std::size_t p=0;p<n;++p) if (m.atoms[p].lane==m.atoms[q].lane) {
            auto d=priorDistance(m,p,q);
            if (d) handoff({p,q,*d,0});
        }
    }
    bool has(const Requirement &r) const {
        if (r.property==Property::Visibility) return false;
        if (m.atoms[r.source].lane == m.atoms[r.target].lane &&
            m.target.synchronous(m.atoms[r.source].lane)) return true;
        return cuts.get(r.source,n+r.target)<=r.distance;
    }
    uint64_t work() const { return cuts.relaxations; }
};

using Position=std::tuple<uint64_t,uint64_t,unsigned,uint64_t>;
Position position(const Model &m,const Action &a) {
    const Atom &p=m.atoms[a.anchor];
    return {p.residue,p.order,a.after?1u:0u,a.order};
}
std::optional<uint64_t> nextDistance(const Model &m,const Action &a,const Action &b) {
    if (position(m,a)<position(m,b)) return 0;
    if (m.recurring) return 1;
    return {};
}
struct Episode { std::size_t set,wait; Handoff edge; };

// Relies only on RECOVERED endpoint domains, pipes and keys, not stream tags.
bool matchActions(const Model &m,const std::vector<Action> &actions,
                  std::vector<Episode> &episodes,std::string &why) {
    std::map<std::tuple<std::size_t,bool,uint64_t>,bool> unique;
    for (const Action &a:actions) {
        if (a.participation!=Action::Every || a.guardResidue ||
            (a.kind!=Action::Set && a.kind!=Action::Wait && a.kind!=Action::Barrier) ||
            a.anchor>=m.atoms.size() || !unique.emplace(
                std::make_tuple(a.anchor,a.after,a.order),true).second) {
            why="duplicate or unavailable concrete action point"; return false;
        }
        if (a.kind==Action::Barrier) {
            if (a.after || a.distanceInIterations || a.source!=a.target ||
                a.source!=m.atoms[a.anchor].lane || !m.target.barrier(a.source)) {
                why="invalid named barrier"; return false;
            }
        } else if (!m.target.event(a.source,a.target) ||
                   !m.target.available(a.source,a.target,a.key) ||
                   (a.kind==Action::Set ? (!a.after || a.source!=m.atoms[a.anchor].lane)
                                       : (a.after || a.target!=m.atoms[a.anchor].lane))) {
            why="illegal event direction, reservation, lane or cut"; return false;
        }
    }
    std::vector<bool> used(actions.size(),false);
    for (std::size_t s=0;s<actions.size();++s) {
        const auto &set=actions[s];
        if (set.kind!=Action::Set) continue;
        std::optional<Episode> matching;
        for (std::size_t w=0;w<actions.size();++w) {
            const auto &wait=actions[w];
            if (wait.kind!=Action::Wait || wait.source!=set.source || wait.target!=set.target ||
                wait.key!=set.key || wait.distanceInIterations!=set.distanceInIterations) continue;
            uint64_t destination=plus(m.atoms[set.anchor].residue,set.distanceInIterations);
            if (destination==Inf || destination%m.period!=m.atoms[wait.anchor].residue) continue;
            uint64_t distance=destination/m.period;
            if ((!m.recurring && distance) || (!distance &&
                !(position(m,set)<position(m,wait)))) continue;
            // Recover the first consuming occurrence in the actual command
            // order. Causal recycling is checked separately below; textual
            // order alone is not accepted as a safe reuse argument.
            if (!matching || std::make_pair(distance, position(m,wait)) <
                std::make_pair(matching->edge.distance,position(m,actions[matching->wait])))
                matching=Episode{s,w,{set.anchor,wait.anchor,distance,set.key}};
        }
        if (!matching) { why="publication has no matching acquisition domain"; return false; }
        if (used[matching->wait]) { why="two publications target one consuming occurrence"; return false; }
        used[matching->wait]=true; used[s]=true; episodes.push_back(*matching);
    }
    for (std::size_t i=0;i<actions.size();++i)
        if (actions[i].kind!=Action::Barrier && !used[i]) {
            why="acquisition has no unique publication"; return false;
        }
    return true;
}

// Event-generation causality is separate from memory coverage. Set FIRE does
// not block subsequent source issue. Consequently there is NO implicit edge
// between consecutive Set FIRE nodes. A consumed wait does order later issue
// on its destination lane, and a matching set precedes that wait.
class EventCausality {
    const Model &m;
    const std::vector<Action> &actions;
    const std::vector<Episode> &episodes;
    Cuts causal;
public:
    EventCausality(const Model &m,const std::vector<Action> &actions,
                   const std::vector<Episode> &episodes)
        : m(m),actions(actions),episodes(episodes),causal(actions.size()) {
        for (const auto &e:episodes) causal.edge(e.set,e.wait,e.edge.distance);
        for (std::size_t w=0;w<actions.size();++w) if (actions[w].kind==Action::Wait)
            for (std::size_t s=0;s<actions.size();++s) if (actions[s].kind==Action::Set &&
                    actions[w].target==actions[s].source) {
                auto d=nextDistance(m,actions[w],actions[s]);
                if (d) causal.edge(w,s,*d);
            }
        causal.close();
    }
    uint64_t work() const { return causal.relaxations; }
    bool recycles(std::vector<std::size_t> ids) const {
        std::sort(ids.begin(),ids.end(),[&](auto a,auto b) {
            return position(m,actions[episodes[a].set])<position(m,actions[episodes[b].set]);
        });
        const std::size_t checks=m.recurring?ids.size():(ids.empty()?0:ids.size()-1);
        for (std::size_t i=0;i<checks;++i) {
            const auto &a=episodes[ids[i]], &b=episodes[ids[(i+1)%ids.size()]];
            auto next=nextDistance(m,actions[a.set],actions[b.set]);
            if (!next || a.edge.distance>*next ||
                causal.get(a.wait,b.set)>*next-a.edge.distance) return false;
        }
        return true;
    }
    bool allFamilies() const {
        using Key=std::tuple<Lane,Lane,unsigned>;
        std::map<Key,std::vector<std::size_t>> families;
        for (std::size_t i=0;i<episodes.size();++i) {
            const auto &a=actions[episodes[i].set];
            families[{a.source,a.target,a.key}].push_back(i);
        }
        for (const auto &family:families) if (!recycles(family.second)) return false;
        return true;
    }
};

} // namespace

bool Target::supports(Lane l) const {
    if (l.core!=Core::AIC && l.core!=Core::AIV) return false;
    if (l.core==Core::AIV) return l.pipe==Pipe::S || l.pipe==Pipe::V ||
        l.pipe==Pipe::MTE2 || l.pipe==Pipe::MTE3;
    return l.pipe==Pipe::S || l.pipe==Pipe::M || l.pipe==Pipe::MTE1 ||
        l.pipe==Pipe::MTE2 || l.pipe==Pipe::MTE3 || l.pipe==Pipe::FIX;
}
bool Target::event(Lane a,Lane b) const {
    if (a.core!=b.core || a==b || !supports(a) || !supports(b)) return false;
    if (a.core==Core::AIV) return true;
    if (a.pipe==Pipe::S || b.pipe==Pipe::S) return false;
    if (a.pipe==Pipe::M) return b.pipe==Pipe::MTE1 || b.pipe==Pipe::MTE2 || b.pipe==Pipe::FIX;
    if (a.pipe==Pipe::MTE3) return b.pipe==Pipe::MTE1 || b.pipe==Pipe::MTE2 || b.pipe==Pipe::FIX;
    return true; // MTE1, MTE2, FIX rows of the selected NPU2201 profile.
}
bool Target::barrier(Lane lane) const { return supports(lane) && lane.pipe!=Pipe::S; }
bool Target::available(Lane a,Lane b,unsigned key) const {
    return std::find(compilerKeys.begin(),compilerKeys.end(),key)!=compilerKeys.end() &&
        std::none_of(reservations.begin(),reservations.end(),[&](const auto &r) {
            return r.source==a && r.target==b && r.key==key;
        });
}
std::optional<uint64_t> mlir::pto::structured_sync::priorDistance(
    const Model &m,std::size_t p,std::size_t q) {
    if (p>=m.atoms.size() || q>=m.atoms.size()) return {};
    if (before(m.atoms[p],m.atoms[q])) return 0;
    if (m.recurring && m.atoms[p].segment==Segment::Body && m.atoms[q].segment==Segment::Body) return 1;
    return {};
}
std::optional<uint64_t> mlir::pto::structured_sync::iterationDistance(
    const Model &m,const Handoff &e) {
    if (!m.period || e.source>=m.atoms.size() || e.target>=m.atoms.size() ||
        e.distance>Inf/m.period) return {};
    if (m.atoms[e.source].segment!=Segment::Body || m.atoms[e.target].segment!=Segment::Body) {
        if (e.distance || !before(m.atoms[e.source],m.atoms[e.target])) return {};
        return 0;
    }
    if ((!m.recurring && e.distance) || (!e.distance && !before(m.atoms[e.source],m.atoms[e.target]))) return {};
    uint64_t target=plus(e.distance*m.period,m.atoms[e.target].residue);
    if (target==Inf || target<m.atoms[e.source].residue) return {};
    return target-m.atoms[e.source].residue;
}
std::vector<Action> mlir::pto::structured_sync::actionsForPlan(const Model &m,const Plan &p) {
    std::vector<Action> out;
    uint64_t ordinal=0;
    // Native lowering preserves this per-boundary order; the checker recovers
    // the order again instead of assuming that tags establish matching.
    for (auto q:p.barriers) {
        if (q>=m.atoms.size()) return {};
        out.push_back({Action::Barrier,q,false,ordinal++,m.atoms[q].lane,m.atoms[q].lane,0,0});
    }
    for (auto q:p.firstBarriers) {
        if (q>=m.atoms.size()) return {};
        Action a{Action::Barrier,q,false,ordinal++,m.atoms[q].lane,m.atoms[q].lane,0,0};
        a.participation=Action::First; out.push_back(a);
    }
    for (const auto &e:p.handoffs) {
        auto delta=iterationDistance(m,e);
        if (!delta) return {};
        auto source=m.atoms[e.source].lane,target=m.atoms[e.target].lane;
        Action set{Action::Set,e.source,true,ordinal++,source,target,e.key,*delta};
        Action wait{Action::Wait,e.target,false,ordinal++,source,target,e.key,*delta};
        if (m.atoms[e.source].segment==Segment::Prelude && m.atoms[e.target].segment==Segment::Body) {
            set.participation=Action::IfBody; set.guardResidue=m.atoms[e.target].residue;
            wait.participation=Action::First;
        } else if (m.atoms[e.source].segment==Segment::Body && m.atoms[e.target].segment==Segment::Epilogue) {
            set.participation=Action::Last;
            wait.participation=Action::IfBody; wait.guardResidue=m.atoms[e.source].residue;
        }
        out.push_back(set); out.push_back(wait);
    }
    return out;
}
bool mlir::pto::structured_sync::supplies(const Model &m,const Plan &p,const Requirement &r) {
    if (hasBoundaries(m)) return suppliesRegion(m,p,r);
    std::string why;
    if (!p.firstBarriers.empty() || !validModel(m,why) || r.source>=m.atoms.size() || r.target>=m.atoms.size()) return false;
    for (const auto &e:p.handoffs)
        if (!iterationDistance(m,e)) return false;
    for (auto q:p.barriers) if (q>=m.atoms.size()) return false;
    Completion c(m);
    for (const auto &e:p.handoffs) c.handoff(e);
    for (auto q:p.barriers) c.barrier(q);
    return c.has(r);
}
Result mlir::pto::structured_sync::verify(const Model &m,const std::vector<Action> &actions) {
    if (hasBoundaries(m)) return verifyRegion(m,actions);
    Result out;
    if (!validModel(m,out.reason)) return out;
    std::vector<Episode> episodes;
    if (!matchActions(m,actions,episodes,out.reason)) {
        out.status=Status::InvalidPlan; return out;
    }
    EventCausality events(m,actions,episodes);
    if (!events.allFamilies()) {
        out.reason="consumption is not causally before physical-key rearm";
        out.status=Status::InvalidPlan; return out;
    }
    out.eventRelaxations=events.work();
    Completion c(m);
    for (const auto &e:episodes) { c.handoff(e.edge); out.plan.handoffs.push_back(e.edge); }
    for (const auto &a:actions) if (a.kind==Action::Barrier) {
        c.barrier(a.anchor); out.plan.barriers.push_back(a.anchor);
    }
    for (const Requirement &r:m.requirements) if (!c.has(r)) {
        out.reason="required completion/resource/visibility property not supplied";
        out.status=Status::InvalidPlan; return out;
    }
    out.completionRelaxations=c.work(); out.status=Status::Applied;
    out.reason="prefix-periodic requirements and event episodes verified";
    return out;
}
namespace {
Result selectPeriodic(const Model &m) {
    Result out;
    if (!validModel(m,out.reason)) return out;
    for (const auto &r:m.requirements) if (r.property==Property::Visibility) {
        out.reason="visibility has no qualified structured realization"; return out;
    }
    Completion c(m);
    auto order=schedule(m);
    std::vector<std::size_t> rank(m.atoms.size());
    for (std::size_t i=0;i<order.size();++i) rank[order[i]]=i;
    for (auto q:order) {
        std::map<Lane,Requirement> missing;
        for (const auto &r:m.requirements) if (r.target==q &&
                m.atoms[r.source].lane!=m.atoms[q].lane && !c.has(r)) {
            auto lane=m.atoms[r.source].lane;
            auto found=missing.find(lane);
            if (found==missing.end() || r.distance<found->second.distance ||
                (r.distance==found->second.distance && rank[r.source]>rank[found->second.source]))
                missing[lane]=r;
        }
        for (const auto &item:missing) {
            const auto &r=item.second;
            if (c.has(r)) continue; // a preceding choice may now supply it transitively
            if (!m.target.event(m.atoms[r.source].lane,m.atoms[r.target].lane)) {
                out.reason="required directed hardware event is unavailable"; return out;
            }
            Handoff e{r.source,r.target,r.distance,0};
            out.plan.handoffs.push_back(e); c.handoff(e);
        }
    }
    for (auto q:order) {
        bool needs=false;
        for (const auto &r:m.requirements) if (r.target==q &&
                m.atoms[r.source].lane==m.atoms[q].lane && !c.has(r)) needs=true;
        if (!needs) continue;
        if (!m.target.barrier(m.atoms[q].lane)) {
            out.reason="required same-lane completion has no qualified barrier"; return out;
        }
        out.plan.barriers.push_back(q); c.barrier(q);
    }
    for (const auto &r:m.requirements) if (!c.has(r)) {
        out.reason="constructed compact plan does not supply original requirements";
        out.status=Status::InvalidPlan; return out;
    }
    out.completionRelaxations=c.work();
    out.status=Status::Applied;
    return out;
}
Result allocatePeriodic(const Model &m,Result out) {
    using Domain=std::pair<Lane,Lane>;
    std::map<Domain,std::size_t> population;
    for (const auto &e:out.plan.handoffs) ++population[{m.atoms[e.source].lane,m.atoms[e.target].lane}];
    // Compute the key-independent causal cuts ONCE. Assignment groups episodes
    // without changing the plan; it must not repeat the fixed point per key.
    auto logicalActions=actionsForPlan(m,out.plan);
    std::vector<Episode> episodes;
    for (std::size_t j=0;j<out.plan.handoffs.size();++j) {
        std::size_t s=out.plan.barriers.size()+2*j;
        episodes.push_back({s,s+1,out.plan.handoffs[j]});
    }
    EventCausality causal(m,logicalActions,episodes);
    out.eventRelaxations=causal.work();
    std::map<Domain,std::map<unsigned,std::vector<std::size_t>>> assigned;
    for (std::size_t i=0;i<out.plan.handoffs.size();++i) {
        auto &e=out.plan.handoffs[i];
        Domain d{m.atoms[e.source].lane,m.atoms[e.target].lane};
        std::vector<unsigned> keys;
        for (unsigned k:m.target.compilerKeys) if (m.target.available(d.first,d.second,k)) keys.push_back(k);
        if (population[d]<=keys.size()) std::stable_sort(keys.begin(),keys.end(),[&](unsigned a,unsigned b) {
            return assigned[d].count(a)<assigned[d].count(b);
        });
        bool found=false;
        for (unsigned key:keys) {
            auto occupied=assigned[d].find(key);
            auto candidate=occupied==assigned[d].end()?std::vector<std::size_t>{}:occupied->second;
            candidate.push_back(i);
            if (causal.recycles(candidate)) {
                e.key=key; assigned[d][key]=std::move(candidate); found=true; break;
            }
        }
        if (!found) { out.reason="no boundary-preserving key assignment established";
            out.status=Status::AllocationFailure; return out; }
    }
    auto checked=verify(m,actionsForPlan(m,out.plan));
    if (checked.status!=Status::Applied) return checked;
    out.status=Status::Applied; out.reason="deterministic structured staircase";
    return out;
}

} // namespace

Result mlir::pto::structured_sync::construct(const Model &m) {
    if (hasBoundaries(m)) return constructRegion(m);
    auto selected=selectPeriodic(m);
    if (selected.status!=Status::Applied) return selected;
    return allocatePeriodic(m,std::move(selected));
}

namespace {
bool periodicEdge(const Model &m,const Handoff &h) {
    return m.atoms[h.source].segment==Segment::Body && m.atoms[h.target].segment==Segment::Body;
}
struct BodyProjection {
    Model model;
    std::vector<std::size_t> original, local;
    explicit BodyProjection(const Model &m):local(m.atoms.size(),std::size_t(-1)) {
        model.period=m.period; model.recurring=m.recurring; model.target=m.target;
        for (std::size_t i=0;i<m.atoms.size();++i) if (m.atoms[i].segment==Segment::Body) {
            local[i]=original.size(); original.push_back(i); model.atoms.push_back(m.atoms[i]);
        }
        for (auto r:m.requirements) if (m.atoms[r.source].segment==Segment::Body &&
                                       m.atoms[r.target].segment==Segment::Body) {
            r.source=local[r.source]; r.target=local[r.target]; model.requirements.push_back(r);
        }
    }
};

// A finite first/last-occurrence view of one loop invocation. First views are
// prefix-closed: when a target exists, every earlier first node used on its
// causal path exists. Last views split N=D*K+s only at represented residues.
// K=0 and K>=1 are checked separately; no trip count or numeric period is
// enumerated. A node denotes a selected occurrence, NEVER the entire region.
class BoundaryCuts {
    const Model &original;
    std::unique_ptr<Model> projected;
    std::unique_ptr<Completion> completion;
    std::vector<std::size_t> local;
    std::vector<int> epoch;
    bool first;
public:
    BoundaryCuts(const Model &m,const Plan &plan,bool firstView,bool full=false,uint64_t tail=0)
        :original(m),projected(new Model),local(m.atoms.size(),std::size_t(-1)),
         epoch(m.atoms.size(),0),first(firstView) {
        projected->recurring=false; projected->target=m.target;
        std::vector<std::size_t> ids;
        for (std::size_t i=0;i<m.atoms.size();++i) {
            const auto &a=m.atoms[i];
            if (first && a.segment==Segment::Epilogue) continue;
            if (!first && a.segment==Segment::Body && !full && a.residue>=tail) continue;
            if (!first && a.segment==Segment::Body) epoch[i]=a.residue<tail?0:-1;
            ids.push_back(i);
        }
        std::sort(ids.begin(),ids.end(),[&](auto p,auto q) {
            const auto &a=m.atoms[p],&b=m.atoms[q];
            return std::tie(a.segment,epoch[p],a.residue,a.order)<
                   std::tie(b.segment,epoch[q],b.residue,b.order);
        });
        for (auto i:ids) {
            local[i]=projected->atoms.size();
            projected->atoms.push_back({0,local[i],m.atoms[i].lane});
        }
        completion.reset(new Completion(*projected));
        // Only an actual body handoff whose selected source reaches the last
        // source occurrence can become a last-view edge. A distance-one event
        // is absent at the first target occurrence and cannot seed First.
        for (const auto &h:plan.handoffs) if (periodicEdge(m,h)) {
            if (!present(h.source) || !present(h.target)) continue;
            int d=first?0:epoch[h.target]-epoch[h.source];
            if (d>=0 && h.distance<=uint64_t(d)) add(h);
        }
        for (auto q:plan.barriers) if (m.atoms[q].segment==Segment::Body) barrier(q,false);
        for (const auto &h:plan.handoffs) if (!periodicEdge(m,h)) add(h);
        for (auto q:plan.barriers) if (m.atoms[q].segment!=Segment::Body) barrier(q,false);
        for (auto q:plan.firstBarriers) barrier(q,true);
    }
    bool present(std::size_t p) const { return local[p]!=std::size_t(-1); }
    void add(const Handoff &h) {
        if (present(h.source) && present(h.target))
            completion->handoff({local[h.source],local[h.target],0,0});
    }
    void barrier(std::size_t q,bool firstOnly) {
        if (!present(q)) return;
        if (!firstOnly || first) { completion->barrier(local[q]); return; }
        // A first-only barrier's PRELUDE knowledge persists to the last q.
        // It does not complete body work issued later than that first barrier.
        for (std::size_t p=0;p<original.atoms.size();++p)
            if (original.atoms[p].segment==Segment::Prelude &&
                original.atoms[p].lane==original.atoms[q].lane)
                completion->handoff({local[p],local[q],0,0});
    }
    bool has(const Requirement &r) const {
        if (!present(r.target) || !present(r.source)) return true; // absent-body requirement is vacuous
        return completion->has({local[r.source],local[r.target],0,r.property});
    }
    uint64_t work() const { return completion->work(); }
};
using Tails=std::vector<std::unique_ptr<BoundaryCuts>>;
Tails tailCuts(const Model &m,const Plan &plan) {
    if (std::none_of(m.atoms.begin(),m.atoms.end(),[](const Atom &a){return a.segment==Segment::Epilogue;})) return {};
    std::set<uint64_t> tails{0};
    for (const auto &a:m.atoms) if (a.segment==Segment::Body && a.residue+1<m.period)
        tails.insert(a.residue+1);
    Tails result;
    for (uint64_t s:tails) for (bool full:{false,true})
        result.emplace_back(new BoundaryCuts(m,plan,false,full,s));
    return result;
}
bool boundaryRequirement(const Model &m,const Requirement &r) {
    return m.atoms[r.source].segment!=Segment::Body || m.atoms[r.target].segment!=Segment::Body;
}
bool allHave(const Tails &tails,const Requirement &r) {
    return std::all_of(tails.begin(),tails.end(),[&](const auto &c){return c->has(r);});
}
bool boundaryCoverage(const Model &m,const Plan &p,uint64_t &work) {
    BoundaryCuts first(m,p,true); auto tails=tailCuts(m,p);
    for (const auto &r:m.requirements) if (boundaryRequirement(m,r)) {
        if (m.atoms[r.target].segment==Segment::Epilogue ? !allHave(tails,r) : !first.has(r)) return false;
    }
    work=first.work(); for (const auto &tail:tails) work+=tail->work();
    return true;
}
bool suppliesRegion(const Model &m,const Plan &p,const Requirement &r) {
    std::string why;
    if (!validModel(m,why) || r.source>=m.atoms.size() || r.target>=m.atoms.size()) return false;
    auto expected=priorDistance(m,r.source,r.target);
    if (!expected || *expected!=r.distance) return false;
    // A supply query is not whole-plan acceptance. Validate the actual
    // protocol separately, then answer only the requested obligation.
    Model protocol=m;protocol.requirements.clear();
    auto checked=verifyRegion(protocol,actionsForPlan(m,p));
    if (checked.status!=Status::Applied) return false;
    if (boundaryRequirement(m,r)) {
        if (m.atoms[r.target].segment==Segment::Epilogue) return allHave(tailCuts(m,p),r);
        return BoundaryCuts(m,p,true).has(r);
    }
    BodyProjection body(m); Plan inner;
    for (auto e:p.handoffs) if (periodicEdge(m,e)) {
        e.source=body.local[e.source];e.target=body.local[e.target];inner.handoffs.push_back(e);
    }
    for (auto q:p.barriers) if (m.atoms[q].segment==Segment::Body) inner.barriers.push_back(body.local[q]);
    return supplies(body.model,inner,{body.local[r.source],body.local[r.target],r.distance,r.property});
}

// Reconstruction accepts only complete one-shot boundary pairs. Until a later
// explicitly checked lifetime-sharing extension, each owns a distinct key in
// its directed domain, disjoint from recurring keys. This is an allocation
// restriction, NOT a serialization fallback or a hardware-capacity claim.
Result verifyRegion(const Model &m,const std::vector<Action> &actions) {
    Result out;
    if (!validModel(m,out.reason)) return out;
    out.status=Status::InvalidPlan;
    BodyProjection body(m); std::vector<Action> inner;
    using Key=std::tuple<Lane,Lane,unsigned>;
    std::map<Key,std::vector<const Action*>> boundary;
    std::set<Key> recurringKeys;
    std::set<std::tuple<std::size_t,bool,uint64_t>> points;
    for (const auto &a:actions) {
        if (a.anchor>=m.atoms.size() || !points.emplace(a.anchor,a.after,a.order).second) {
            out.reason="invalid or duplicate reconstructed action point";return out;
        }
        const auto &atom=m.atoms[a.anchor];
        if (atom.segment==Segment::Body && a.participation==Action::Every) {
            auto remapped=a;remapped.anchor=body.local[a.anchor];inner.push_back(remapped);
            if (a.kind!=Action::Barrier) recurringKeys.insert({a.source,a.target,a.key});
            continue;
        }
        if (a.distanceInIterations || (a.participation!=Action::IfBody && a.guardResidue)) {
            out.reason="boundary action has an unqualified execution domain";return out;
        }
        if (a.kind==Action::Barrier) {
            bool firstOnly=atom.segment==Segment::Body && a.participation==Action::First;
            bool once=atom.segment!=Segment::Body && a.participation==Action::Every;
            if ((!firstOnly&&!once) || a.after || a.source!=a.target || a.source!=atom.lane ||
                !m.target.barrier(a.source)) {out.reason="invalid boundary barrier";return out;}
            (firstOnly?out.plan.firstBarriers:out.plan.barriers).push_back(a.anchor);
            continue;
        }
        if ((a.kind!=Action::Set&&a.kind!=Action::Wait) || !m.target.event(a.source,a.target) ||
            !m.target.available(a.source,a.target,a.key) ||
            (a.kind==Action::Set?(!a.after||a.source!=atom.lane):(a.after||a.target!=atom.lane))) {
            out.reason="illegal boundary event direction, key or physical cut";return out;
        }
        boundary[{a.source,a.target,a.key}].push_back(&a);
    }
    for (const auto &family:boundary) {
        if (family.second.size()!=2 || recurringKeys.count(family.first)) {
            out.reason="boundary key must own exactly one pair and no recurring episodes";return out;
        }
        const Action *s=family.second[0],*w=family.second[1];
        if (s->kind==Action::Wait) std::swap(s,w);
        if (s->kind!=Action::Set || w->kind!=Action::Wait) {
            out.reason="missing boundary publication or acquisition";return out;
        }
        const auto &p=m.atoms[s->anchor],&q=m.atoms[w->anchor];
        bool once=p.segment!=Segment::Body && q.segment!=Segment::Body &&
                  s->participation==Action::Every && w->participation==Action::Every;
        bool entry=p.segment==Segment::Prelude && q.segment==Segment::Body &&
                   s->participation==Action::IfBody && s->guardResidue==q.residue &&
                   w->participation==Action::First;
        bool exit=p.segment==Segment::Body && q.segment==Segment::Epilogue &&
                  s->participation==Action::Last && w->participation==Action::IfBody &&
                  w->guardResidue==p.residue;
        if ((!once&&!entry&&!exit) || !before(p,q)) {
            out.reason="boundary first/last/empty-path participation does not match";return out;
        }
        out.plan.handoffs.push_back({s->anchor,w->anchor,0,s->key});
    }
    auto inside=verify(body.model,inner);
    if (inside.status!=Status::Applied) {out.reason="loop body: "+inside.reason;return out;}
    for (auto h:inside.plan.handoffs) {
        h.source=body.original[h.source];h.target=body.original[h.target];out.plan.handoffs.push_back(h);
    }
    for (auto q:inside.plan.barriers) out.plan.barriers.push_back(body.original[q]);
    uint64_t work=0;
    if (!boundaryCoverage(m,out.plan,work)) {
        out.reason="entry, exit or zero-trip requirement is not supplied";return out;
    }
    out.status=Status::Applied;out.reason="periodic body and guarded boundary transfers verified";
    out.completionRelaxations=inside.completionRelaxations+work;out.eventRelaxations=inside.eventRelaxations;
    return out;
}

Result constructRegion(const Model &m) {
    Result out;
    if (!validModel(m,out.reason)) return out;
    for (const auto &r:m.requirements) if (r.property==Property::Visibility) {
        out.reason="visibility has no qualified structured realization";return out;
    }
    BodyProjection body(m);auto selected=selectPeriodic(body.model);
    if (selected.status!=Status::Applied) return selected;
    for (auto h:selected.plan.handoffs) {
        h.source=body.original[h.source];h.target=body.original[h.target];out.plan.handoffs.push_back(h);
    }
    for (auto q:selected.plan.barriers) out.plan.barriers.push_back(body.original[q]);
    BoundaryCuts first(m,out.plan,true);
    // SAME prefix-advance rule as S1, applied to first acquisitions. A first
    // acquisition is persistent knowledge, not one consuming wait per reader.
    for (auto q:schedule(m)) {
        if (m.atoms[q].segment==Segment::Epilogue) continue;
        std::map<Lane,Requirement> latest;
        for (const auto &r:m.requirements) if (r.target==q && boundaryRequirement(m,r) &&
                m.atoms[r.source].lane!=m.atoms[q].lane && !first.has(r)) {
            auto lane=m.atoms[r.source].lane;auto it=latest.find(lane);
            if (it==latest.end() || m.atoms[r.source].order>m.atoms[it->second.source].order) latest[lane]=r;
        }
        for (const auto &item:latest) {
            const auto &r=item.second;if (first.has(r)) continue;
            if (!m.target.event(m.atoms[r.source].lane,m.atoms[q].lane)) {
                out.reason="boundary event direction unavailable";return out;
            }
            Handoff h{r.source,q,0,0};out.plan.handoffs.push_back(h);first.add(h);
        }
    }
    for (auto q:schedule(m)) {
        if (m.atoms[q].segment==Segment::Epilogue) continue;
        bool missing=false;
        for (const auto &r:m.requirements) if (r.target==q && boundaryRequirement(m,r) &&
                m.atoms[r.source].lane==m.atoms[q].lane && !first.has(r)) missing=true;
        if (!missing) continue;
        if (!m.target.barrier(m.atoms[q].lane)) {out.reason="boundary barrier unavailable";return out;}
        bool firstOnly=m.atoms[q].segment==Segment::Body;
        (firstOnly?out.plan.firstBarriers:out.plan.barriers).push_back(q);first.barrier(q,firstOnly);
    }
    auto tails=tailCuts(m,out.plan);
    for (auto q:schedule(m)) if (m.atoms[q].segment==Segment::Epilogue) {
        // Last occurrences at DIFFERENT residues are not uniformly comparable
        // for all N. Keep their separate guarded cuts. Same-residue sources on
        // one lane admit the ordinary latest-prefix compression.
        using Group=std::tuple<Lane,Segment,uint64_t>;
        std::map<Group,Requirement> latest;
        for (const auto &r:m.requirements) if (r.target==q && m.atoms[r.source].lane!=m.atoms[q].lane &&
                !allHave(tails,r)) {
            const auto &p=m.atoms[r.source];Group g{p.lane,p.segment,p.residue};auto it=latest.find(g);
            if (it==latest.end() || p.order>m.atoms[it->second.source].order) latest[g]=r;
        }
        for (const auto &item:latest) {
            const auto &r=item.second;if (allHave(tails,r)) continue;
            if (!m.target.event(m.atoms[r.source].lane,m.atoms[q].lane)) {
                out.reason="exit event direction unavailable";return out;
            }
            Handoff h{r.source,q,0,0};out.plan.handoffs.push_back(h);for (auto &c:tails)c->add(h);
        }
    }
    for (auto q:schedule(m)) if (m.atoms[q].segment==Segment::Epilogue) {
        bool missing=false;
        for (const auto &r:m.requirements) if (r.target==q && m.atoms[r.source].lane==m.atoms[q].lane &&
                !allHave(tails,r)) missing=true;
        if (!missing) continue;
        if (!m.target.barrier(m.atoms[q].lane)) {out.reason="exit barrier unavailable";return out;}
        out.plan.barriers.push_back(q);for (auto &c:tails)c->barrier(q,false);
    }
    using Domain=std::pair<Lane,Lane>;
    std::map<Domain,std::set<unsigned>> boundaryKeys;
    // One assignment owns all domains. Boundary episodes are deliberately not
    // merged with recurring families; reserve their actual keys before invoking
    // S1's recurrence allocator. A refusal is not a proof of infeasibility.
    for (auto &h:out.plan.handoffs) if (!periodicEdge(m,h)) {
        Domain d{m.atoms[h.source].lane,m.atoms[h.target].lane};bool assigned=false;
        for (unsigned k:m.target.compilerKeys) if (m.target.available(d.first,d.second,k) &&
                !boundaryKeys[d].count(k)) {
            h.key=k;boundaryKeys[d].insert(k);body.model.target.reservations.push_back({d.first,d.second,k});
            assigned=true;break;
        }
        if (!assigned) {out.status=Status::AllocationFailure;out.reason="distinct boundary keys do not fit";return out;}
    }
    auto allocated=allocatePeriodic(body.model,std::move(selected));
    if (allocated.status!=Status::Applied) return allocated;
    std::size_t index=0;
    for (auto &h:out.plan.handoffs) if (periodicEdge(m,h)) h.key=allocated.plan.handoffs[index++].key;
    auto checked=verifyRegion(m,actionsForPlan(m,out.plan));
    if (checked.status!=Status::Applied) return checked;
    out.status=Status::Applied;out.reason="structured first/last boundary construction";
    out.completionRelaxations=checked.completionRelaxations;out.eventRelaxations=checked.eventRelaxations;
    return out;
}
} // namespace
