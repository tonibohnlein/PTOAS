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
#include <numeric>
#include <set>
#include <tuple>
#include <utility>

using namespace mlir::pto::structured_sync;
namespace {
constexpr uint64_t Inf = std::numeric_limits<uint64_t>::max();
uint64_t plus(uint64_t a, uint64_t b) {
    return a == Inf || b == Inf || a > Inf - b ? Inf : a + b;
}
bool before(const Atom &a, const Atom &b) {
    return std::tie(a.residue, a.order) < std::tie(b.residue, b.order);
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
    std::set<std::pair<uint64_t,uint64_t>> ranks;
    for (const Atom &a:m.atoms) {
        if (a.residue>=m.period || !m.target.supports(a.lane) ||
            !ranks.insert({a.residue,a.order}).second) {
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
        if ((a.kind!=Action::Set && a.kind!=Action::Wait && a.kind!=Action::Barrier) ||
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
    if (m.recurring) return 1;
    return {};
}
std::optional<uint64_t> mlir::pto::structured_sync::iterationDistance(
    const Model &m,const Handoff &e) {
    if (!m.period || e.source>=m.atoms.size() || e.target>=m.atoms.size() ||
        e.distance>Inf/m.period) return {};
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
    for (const auto &e:p.handoffs) {
        auto delta=iterationDistance(m,e);
        if (!delta) return {};
        auto source=m.atoms[e.source].lane,target=m.atoms[e.target].lane;
        out.push_back({Action::Set,e.source,true,ordinal++,source,target,e.key,*delta});
        out.push_back({Action::Wait,e.target,false,ordinal++,source,target,e.key,*delta});
    }
    return out;
}
bool mlir::pto::structured_sync::supplies(const Model &m,const Plan &p,const Requirement &r) {
    std::string why;
    if (!validModel(m,why) || r.source>=m.atoms.size() || r.target>=m.atoms.size()) return false;
    for (const auto &e:p.handoffs)
        if (!iterationDistance(m,e)) return false;
    for (auto q:p.barriers) if (q>=m.atoms.size()) return false;
    Completion c(m);
    for (const auto &e:p.handoffs) c.handoff(e);
    for (auto q:p.barriers) c.barrier(q);
    return c.has(r);
}
Result mlir::pto::structured_sync::verify(const Model &m,const std::vector<Action> &actions) {
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
Result mlir::pto::structured_sync::construct(const Model &m) {
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
