// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/InsertSync/StructuredSyncCore.h"
#include <algorithm>
#include <functional>
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
bool mixedBoundaryReuse(const Model &,const Plan &,const std::vector<Action> * = nullptr);
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
    for (const auto &a:actions)
        if (a.invocation!=Action::Local || a.invocationBodyGuard || a.invocationGuardResidue) {
            Result rejected; rejected.status=Status::InvalidPlan;
            rejected.reason="invocation action passed to a one-invocation verifier"; return rejected;
        }
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
Result mlir::pto::structured_sync::refinePeriodicHandoffs(const Model &m,const Plan &input) {
    Result out;
    if (!validModel(m,out.reason)) return out;
    if (!m.recurring || hasBoundaries(m) || !input.firstBarriers.empty()) {
        out.reason="handoff refinement requires one local periodic body";return out;
    }
    // Do not let an invalid endpoint be silently converted to an empty action
    // population by actionsForPlan, even when there are no memory demands.
    for (const auto &h:input.handoffs) if (!iterationDistance(m,h)) {
        out.status=Status::InvalidPlan;out.reason="invalid refinement handoff";return out;
    }
    for (auto q:input.barriers) if (q>=m.atoms.size()) {
        out.status=Status::InvalidPlan;out.reason="invalid refinement barrier";return out;
    }
    out=verify(m,actionsForPlan(m,input));
    if (out.status!=Status::Applied) return out;
    out.plan=input;
    // Reverse order gives each originally selected handoff at most one trial.
    // The trial checker is rebuilt WITHOUT the removed pair. It must not reuse
    // a completion receipt or token-order path supplied by that pair itself.
    for (std::size_t i=input.handoffs.size();i>0;--i) {
        Plan trial=out.plan;
        trial.handoffs.erase(trial.handoffs.begin()+std::ptrdiff_t(i-1));
        auto checked=verify(m,actionsForPlan(m,trial));
        ++out.refinementAttempts;
        out.completionRelaxations=plus(out.completionRelaxations,checked.completionRelaxations);
        out.eventRelaxations=plus(out.eventRelaxations,checked.eventRelaxations);
        if (checked.status==Status::Applied) {
            out.plan=std::move(trial);++out.removedHandoffs;
        }
    }
    out.reason="verified periodic handoff reverse deletion";return out;
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
Result allocatePeriodic(const Model &m,Result out,
                        const std::function<bool(const Plan &,std::size_t)> &boundaryCheck = {}) {
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
                const unsigned before=e.key;e.key=key;
                if(boundaryCheck&&!boundaryCheck(out.plan,i+1)){e.key=before;continue;}
                assigned[d][key]=std::move(candidate); found=true; break;
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
bool hasExitRequirements(const Model &m) {
    return std::any_of(m.requirements.begin(),m.requirements.end(),[&](const Requirement &r) {
        return m.atoms[r.target].segment==Segment::Epilogue;
    });
}
bool boundaryCoverage(const Model &m,const Plan &p,uint64_t &work) {
    BoundaryCuts first(m,p,true);
    auto tails=hasExitRequirements(m)?tailCuts(m,p):Tails{};
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
        if (family.second.size()!=2 || (!m.allowBoundaryKeyReuse&&recurringKeys.count(family.first))) {
            out.reason="boundary key must own exactly one pair; recurring continuation is not enabled";return out;
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
    if(m.allowBoundaryKeyReuse&&!mixedBoundaryReuse(m,out.plan,&actions)) {
        out.reason="boundary/recurring notification consumption before reuse is not established";return out;
    }
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
    auto tails=hasExitRequirements(m)?tailCuts(m,out.plan):Tails{};
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
            h.key=k;boundaryKeys[d].insert(k);
            if(!m.allowBoundaryKeyReuse)body.model.target.reservations.push_back({d.first,d.second,k});
            assigned=true;break;
        }
        if (!assigned) {out.status=Status::AllocationFailure;out.reason="distinct boundary keys do not fit";return out;}
    }
    std::function<bool(const Plan &,std::size_t)> continuation;
    if(m.allowBoundaryKeyReuse)continuation=[&](const Plan &inside,std::size_t assignedCount) {
        Plan trial=out.plan;std::size_t i=0;
        for(auto &h:trial.handoffs)if(periodicEdge(m,h)) {
            // Unassigned streams participate in logical causality but do not
            // collide with a proposed physical family. The sentinel is NEVER
            // emitted or accepted by the final physical-key validator.
            h.key=i<assignedCount?inside.handoffs[i].key:unsigned(8);
            ++i;
        }
        return mixedBoundaryReuse(m,trial);
    };
    auto allocated=allocatePeriodic(body.model,std::move(selected),continuation);
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

//===----------------------------------------------------------------------===//
// S3: re-entry of an S2 region. No Presburger sets, trip unrolling, quota,
// solver, all-path enumeration or iteration drain participates in this code.
//===----------------------------------------------------------------------===//
namespace {
bool loopAtom(const Model &m,std::size_t p) {
    return m.recurring && m.atoms[p].segment==Segment::Body;
}
std::optional<uint64_t> pairExistence(const Model &m,std::size_t p,std::size_t q) {
    std::optional<uint64_t> r;
    for (auto a:{p,q}) if (loopAtom(m,a)) r=r?std::max(*r,m.atoms[a].residue):m.atoms[a].residue;
    return r;
}
bool validInvocationModel(const InvocationModel &m,std::string &why) {
    if (!validModel(m.region,why)) return false;
    if (!m.region.recurring && (m.region.period!=1 ||
        std::any_of(m.region.atoms.begin(),m.region.atoms.end(),[](const Atom &a){return a.residue!=0;}))) {
        why="a repeated straight-line region must have one static occurrence per atom";return false;
    }
    for (const auto &r:m.carried) {
        if (r.source>=m.region.atoms.size() || r.target>=m.region.atoms.size()) {
            why="cross-invocation requirement has an invalid endpoint";return false;
        }
        if (r.property!=Property::Completion && r.property!=Property::AccResource &&
            r.property!=Property::Visibility) {
            why="unknown cross-invocation property";return false;
        }
    }
    return true;
}

// Body occurrence epochs are a*K+b, a in {0,1}, for N=D*K+s. Retained
// occurrences are first/last payloads and first/last actual event episodes.
// Comparisons and existence can change only at integer roots of differences
// of these affine forms. Cuts below partition K by those roots; they do NOT
// approximate arbitrary N by executing a chosen number of iterations.
struct Epoch {
    int a=0;
    int64_t b=0;
};
struct InterfaceCase { uint64_t tail=0;int64_t k=0; };
struct Point {
    std::size_t atom=0;
    int64_t epoch=0;
    bool operator<(const Point &b) const {return std::tie(atom,epoch)<std::tie(b.atom,b.epoch);}
    bool operator==(const Point &b) const {return atom==b.atom&&epoch==b.epoch;}
};
struct End {
    Point point;
    bool after=false;
    uint64_t order=0;
};
struct LocalEpisode { End set,wait; Lane source,target;unsigned key=0;bool periodic=false; };
struct BarrierPort { End at;Lane lane; };
struct Interface {
    std::vector<Point> payloads;
    std::vector<LocalEpisode> episodes;
    std::vector<BarrierPort> barriers;
    std::vector<std::optional<Point>> first,last;
};

int64_t lastEpoch(const Model &m,std::size_t p,const InterfaceCase &c) {
    return loopAtom(m,p)?c.k-(m.atoms[p].residue>=c.tail?1:0):0;
}
bool exists(const Model &m,std::size_t p,const InterfaceCase &c) {
    return !loopAtom(m,p)||lastEpoch(m,p,c)>=0;
}
bool residueExists(const Model &m,uint64_t r,const InterfaceCase &c) {
    return m.recurring && (c.k>0 || r<c.tail);
}
using EndRank=std::tuple<Segment,int64_t,uint64_t,uint64_t,unsigned,uint64_t>;
EndRank endRank(const Model &m,const End &e) {
    const auto &a=m.atoms[e.point.atom];
    return {a.segment,e.point.epoch,a.residue,a.order,e.after?2u:0u,e.order};
}

std::vector<InterfaceCase> interfaceCases(const Model &m,const InvocationPlan &plan) {
    const auto &p=plan.local;
    if (!m.recurring) return {{0,0}};
    std::set<uint64_t> tails{0};
    for (const auto &a:m.atoms) if (a.segment==Segment::Body && a.residue+1<m.period)
        tails.insert(a.residue+1);
    for (const auto &b:plan.barriers) if (b.bodyGuard && b.guardResidue+1<m.period)
        tails.insert(b.guardResidue+1);
    std::vector<InterfaceCase> out;
    for (uint64_t tail:tails) {
        std::vector<Epoch> forms{{0,0}};
        for (std::size_t i=0;i<m.atoms.size();++i) if (loopAtom(m,i))
            forms.push_back({1,m.atoms[i].residue<tail?0:-1});
        for (const auto &h:p.handoffs) if (periodicEdge(m,h)) {
            // construct/verify admits latest ordinary sources only (0 or 1).
            // Avoid narrowing a malformed distance before the model gate.
            if (h.distance>1) return {};
            int64_t d=int64_t(h.distance);
            forms.push_back({0,d});
            forms.push_back({1,(m.atoms[h.target].residue<tail?0:-1)-d});
        }
        std::set<int64_t> cuts{0};
        for (const auto &x:forms) for (const auto &y:forms) if (x.a!=y.a) {
            // Both slopes are 0/1, so this division is exact. Offsets are
            // -2..1 for the admitted one-invocation transfers.
            const int64_t root=(y.b-x.b)/(x.a-y.a);
            if (root>=0) cuts.insert(root);
            if (root>=-1) cuts.insert(root+1);
        }
        for (auto k:cuts) out.push_back({tail,k});
    }
    return out;
}

Interface interfaceFor(const Model &m,const Plan &p,const InterfaceCase &c,const std::vector<Action> &actions) {
    Interface out;out.first.resize(m.atoms.size());out.last.resize(m.atoms.size());
    std::set<Point> points;
    for (std::size_t a=0;a<m.atoms.size();++a) if (exists(m,a,c)) {
        out.first[a]=Point{a,0};out.last[a]=Point{a,lastEpoch(m,a,c)};
        points.insert(*out.first[a]);points.insert(*out.last[a]);
    }
    const std::size_t handoffBase=p.barriers.size()+p.firstBarriers.size();
    for (std::size_t j=0;j<p.handoffs.size();++j) {
        const auto &h=p.handoffs[j];
        if (!out.first[h.source]||!out.first[h.target]) continue;
        auto append=[&](Point s,Point w) {
            points.insert(s);points.insert(w);
            out.episodes.push_back({{s,true,actions[handoffBase+2*j].order},
                                    {w,false,actions[handoffBase+2*j+1].order},
                                    m.atoms[h.source].lane,m.atoms[h.target].lane,h.key,periodicEdge(m,h)});
        };
        if (periodicEdge(m,h) && m.recurring) {
            const int64_t d=int64_t(h.distance),end=lastEpoch(m,h.target,c)-d;
            if (end<0) continue; // no matching publication/acquisition exists
            append({h.source,0},{h.target,d});
            if (end>0) append({h.source,end},{h.target,end+d});
        } else if (loopAtom(m,h.source)) append(*out.last[h.source],*out.first[h.target]);
        else append(*out.first[h.source],*out.first[h.target]);
    }
    for (std::size_t j=0;j<p.barriers.size();++j) {
        const auto a=p.barriers[j];if (!out.first[a])continue;
        out.barriers.push_back({{*out.first[a],false,actions[j].order},m.atoms[a].lane});
        if (!(*out.first[a]==*out.last[a]))
            out.barriers.push_back({{*out.last[a],false,actions[j].order},m.atoms[a].lane});
    }
    for (std::size_t j=0;j<p.firstBarriers.size();++j) {
        const auto a=p.firstBarriers[j];if(out.first[a])
            out.barriers.push_back({{*out.first[a],false,actions[p.barriers.size()+j].order},m.atoms[a].lane});
    }
    out.payloads.assign(points.begin(),points.end());return out;
}

// Mathematical episode reconstruction may group by key. Graph construction
// must nevertheless use the ACTUAL order recovered at each payload boundary.
// A signature identifies a command; it does not establish its execution order.
using InvocationSignature=std::tuple<unsigned,std::size_t,bool,Lane,Lane,unsigned,uint64_t,
    unsigned,uint64_t,unsigned,bool,uint64_t>;
InvocationSignature invocationSignature(const Action &a) {
    return {unsigned(a.kind),a.anchor,a.after,a.source,a.target,a.key,a.distanceInIterations,
            unsigned(a.participation),a.guardResidue,unsigned(a.invocation),a.invocationBodyGuard,a.invocationGuardResidue};
}
std::vector<Action> orderedInvocationActions(const Model &m,const InvocationPlan &p,
                                            const std::vector<Action> *recovered) {
    auto result=actionsForInvocations({m,{}},p);
    if(recovered) {
        std::map<InvocationSignature,uint64_t> order;
        for(const auto &a:*recovered)order.emplace(invocationSignature(a),a.order);
        for(auto &a:result)a.order=order.at(invocationSignature(a));
    }
    return result;
}

// Finite proof graph of TWO symbolic adjacent invocations. No incoming
// cross-invocation wait is assumed in the first copy. This makes the proof
// valid for invocation zero as well as for every induction step. Outgoing
// publication fire in copy 1 is retained to check consume-before-next-rearm.
// Missing intermediate inner occurrences only LOSE paths; every retained
// edge has an actual target-semantic counterpart for the complete K cell.
class InvocationGraph {
    const Model &m;
    std::vector<Action> orderedActions;
    Interface ports;
    std::vector<std::vector<std::size_t>> graph;
    mutable std::map<std::size_t,std::vector<bool>> cache;
    std::map<std::tuple<unsigned,std::size_t,int64_t>,std::pair<std::size_t,std::size_t>> payload;
    using Key=std::tuple<Lane,Lane,unsigned>;
    struct EpisodeNodes { End source;std::size_t fire=0,take=0;bool periodic=false; };
    std::map<Key,std::vector<EpisodeNodes>> local[2];
    std::vector<std::pair<std::size_t,std::size_t>> reentryReuse;
    static constexpr std::size_t Absent=std::numeric_limits<std::size_t>::max();
    std::size_t node() {graph.emplace_back();return graph.size()-1;}
    void edge(std::size_t a,std::size_t b) {graph[a].push_back(b);}
public:
    mutable uint64_t visits=0;
    explicit InvocationGraph(const Model &model,const InvocationPlan &p,const InterfaceCase &c,
                             const std::vector<Action> *recovered=nullptr)
        :m(model),orderedActions(orderedInvocationActions(m,p,recovered)),
         ports(interfaceFor(m,p.local,c,orderedActions)) {
        struct Item { unsigned invocation=0;End at;int kind=0;Lane source,target;unsigned key=0;
                      std::size_t episode=0;bool reentry=false; };
        std::vector<Item> items;
        const auto &allActions=orderedActions;
        const auto localCount=actionsForPlan(m,p.local).size();
        std::vector<std::pair<std::size_t,std::size_t>> localNodes[2];
        localNodes[0].resize(ports.episodes.size(),{Absent,Absent});
        localNodes[1].resize(ports.episodes.size(),{Absent,Absent});
        std::vector<std::size_t> carrySets[2],carryWaits;
        carrySets[0].assign(p.handoffs.size(),Absent);carrySets[1].assign(p.handoffs.size(),Absent);
        carryWaits.assign(p.handoffs.size(),Absent);
        for (unsigned inv=0;inv<2;++inv) {
            for (auto pt:ports.payloads) items.push_back({inv,{pt,false,0},0,m.atoms[pt.atom].lane,{},0,0,false});
            for (std::size_t j=0;j<ports.episodes.size();++j) {
                const auto &e=ports.episodes[j];
                items.push_back({inv,e.set,1,e.source,e.target,e.key,j,false});
                items.push_back({inv,e.wait,2,e.source,e.target,e.key,j,false});
            }
            for (const auto &b:ports.barriers)items.push_back({inv,b.at,3,b.lane,b.lane,0,0,false});
            for (std::size_t j=0;j<p.handoffs.size();++j) {
                const auto &h=p.handoffs[j];
                if (!ports.last[h.source]||!ports.first[h.target]) continue;
                const auto &s=allActions[localCount+2*j],&w=allActions[localCount+2*j+1];
                items.push_back({inv,{*ports.last[h.source],true,s.order},1,s.source,s.target,h.key,j,true});
                if (inv)items.push_back({inv,{*ports.first[h.target],false,w.order},2,w.source,w.target,h.key,j,true});
            }
            if (inv)for (std::size_t j=0;j<p.barriers.size();++j) {
                const auto &b=p.barriers[j];
                if (!ports.first[b.target] || (b.bodyGuard&&!residueExists(m,b.guardResidue,c)))continue;
                const auto &a=allActions[localCount+2*p.handoffs.size()+j];
                items.push_back({inv,{*ports.first[b.target],false,a.order},3,a.source,a.target,0,0,true});
            }
        }
        std::sort(items.begin(),items.end(),[&](const auto &a,const auto &b) {
            const auto rank=[&](const auto &item) {
                const auto &atom=m.atoms[item.at.point.atom];
                return std::make_tuple(item.invocation,atom.segment,item.at.point.epoch,atom.residue,
                    atom.order,item.kind==0?1u:(item.at.after?2u:0u),item.at.order);
            };return rank(a)<rank(b);
        });
        std::map<Lane,std::size_t> previous;
        std::map<Lane,std::vector<std::size_t>> completions;
        auto issue=[&](Lane lane) {
            auto n=node();auto it=previous.find(lane);if(it!=previous.end())edge(it->second,n);
            previous[lane]=n;return n;
        };
        for (const auto &item:items) {
            if (item.kind==0) {
                auto start=issue(item.source),done=node();edge(start,done);
                if(m.target.synchronous(item.source))previous[item.source]=done;
                completions[item.source].push_back(done);
                payload[{item.invocation,item.at.point.atom,item.at.point.epoch}]={start,done};
            } else if (item.kind==1) {
                auto queued=issue(item.source),fire=node();edge(queued,fire);
                for(auto done:completions[item.source])edge(done,fire);
                if(item.reentry)carrySets[item.invocation][item.episode]=fire;
                else localNodes[item.invocation][item.episode].first=fire;
                // No fire -> next source issue edge: set is not a drain.
            } else if (item.kind==2) {
                auto take=issue(item.target);
                if(item.reentry)carryWaits[item.episode]=take;
                else localNodes[item.invocation][item.episode].second=take;
            } else {
                auto pass=issue(item.source);for(auto done:completions[item.source])edge(done,pass);
            }
        }
        for(unsigned inv=0;inv<2;++inv)for(std::size_t j=0;j<ports.episodes.size();++j) {
            const auto &e=ports.episodes[j];const auto nodes=localNodes[inv][j];
            edge(nodes.first,nodes.second);
            local[inv][{e.source,e.target,e.key}].push_back({e.set,nodes.first,nodes.second,e.periodic});
        }
        for(std::size_t j=0;j<p.handoffs.size();++j)if(carrySets[0][j]!=Absent) {
            edge(carrySets[0][j],carryWaits[j]);
            reentryReuse.push_back({carryWaits[j],carrySets[1][j]});
        }
    }
    bool reaches(std::size_t a,std::size_t b) const {
        auto it=cache.find(a);
        if(it==cache.end()) {
            std::vector<bool> seen(graph.size(),false);std::vector<std::size_t> pending{a};seen[a]=true;
            while(!pending.empty()) {auto p=pending.back();pending.pop_back();
                for(auto q:graph[p]) {++visits;if(!seen[q]){seen[q]=true;pending.push_back(q);}}
            }
            it=cache.emplace(a,std::move(seen)).first;
        }
        return it->second[b];
    }
    bool has(const InvocationRequirement &r) const {
        if(!ports.last[r.source]||!ports.first[r.target])return true;
        if(r.property==Property::Visibility)return false;
        const auto s=*ports.last[r.source],t=*ports.first[r.target];
        return reaches(payload.at({0,s.atom,s.epoch}).second,payload.at({1,t.atom,t.epoch}).first);
    }
    bool mixedLocalReuse() const {
        for(const auto &family:local[0]) {
            auto episodes=family.second;
            const bool boundary=std::any_of(episodes.begin(),episodes.end(),[](const auto &e){return !e.periodic;});
            const bool periodic=std::any_of(episodes.begin(),episodes.end(),[](const auto &e){return e.periodic;});
            if(!boundary||!periodic)continue;
            std::sort(episodes.begin(),episodes.end(),[&](const auto &a,const auto &b){return endRank(m,a.source)<endRank(m,b.source);});
            for(std::size_t j=1;j<episodes.size();++j) {
                // Body/body successors are already proved for ALL epochs by
                // EventCausality. The reduced first/last interface may omit
                // intermediate body episodes, so it must not pretend these
                // two retained endpoints are immediate recurring successors.
                if(episodes[j-1].periodic&&episodes[j].periodic)continue;
                if(!reaches(episodes[j-1].take,episodes[j].fire))return false;
            }
        }
        return true;
    }
    bool recycles() const {
        for(const auto &f:local[0]) {
            const auto order=[&](const auto &a,const auto &b){return endRank(m,a.source)<endRank(m,b.source);};
            const auto &next=local[1].at(f.first);
            const auto &last=*std::max_element(f.second.begin(),f.second.end(),order);
            const auto &first=*std::min_element(next.begin(),next.end(),order);
            if(!reaches(last.take,first.fire))return false;
        }
        for(auto e:reentryReuse)if(!reaches(e.first,e.second))return false;
        return true;
    }
    bool acyclic() const {
        std::vector<std::size_t> indegree(graph.size(),0),ready;
        for(const auto &row:graph)for(auto q:row)++indegree[q];
        for(std::size_t i=0;i<graph.size();++i)if(!indegree[i])ready.push_back(i);
        std::size_t n=0;while(!ready.empty()){auto p=ready.back();ready.pop_back();++n;
            for(auto q:graph[p])if(--indegree[q]==0)ready.push_back(q);}
        return n==graph.size();
    }
};

std::vector<std::unique_ptr<InvocationGraph>> invocationGraphs(const Model &m,const InvocationPlan &p,
                                                                const std::vector<Action> *recovered=nullptr) {
    std::vector<std::unique_ptr<InvocationGraph>> result;
    for(const auto &c:interfaceCases(m,p))result.emplace_back(new InvocationGraph(m,p,c,recovered));
    return result;
}
bool mixedBoundaryReuse(const Model &m,const Plan &p,const std::vector<Action> *recovered) {
    using Key=std::tuple<Lane,Lane,unsigned>;
    std::map<Key,unsigned> populations;
    for(const auto &h:p.handoffs)
        populations[{m.atoms[h.source].lane,m.atoms[h.target].lane,h.key}]|=periodicEdge(m,h)?1u:2u;
    bool mixed=false;for(const auto &entry:populations)mixed|=entry.second==3;
    if(!mixed)return true;
    InvocationPlan projected;projected.local=p;
    if(recovered) {
        std::set<InvocationSignature> expected,actual;
        auto commands=actionsForPlan(m,p);
        for(const auto &a:commands)expected.insert(invocationSignature(a));
        for(const auto &a:*recovered)actual.insert(invocationSignature(a));
        if(expected.size()!=commands.size()||actual.size()!=recovered->size()||expected!=actual)return false;
    }
    auto cases=interfaceCases(m,projected);if(cases.empty())return false;
    for(const auto &c:cases) {
        InvocationGraph graph(m,projected,c,recovered);
        if(!graph.acyclic()||!graph.mixedLocalReuse())return false;
    }
    return true;
}

bool invocationCovered(const std::vector<std::unique_ptr<InvocationGraph>> &g,const InvocationRequirement &r) {
    return !g.empty()&&std::all_of(g.begin(),g.end(),[&](const auto &x){return x->has(r);});
}
bool sameLocalShape(const Plan &a,const Plan &b) {
    if(a.barriers!=b.barriers||a.firstBarriers!=b.firstBarriers||a.handoffs.size()!=b.handoffs.size())return false;
    for(std::size_t i=0;i<a.handoffs.size();++i) {
        const auto &x=a.handoffs[i],&y=b.handoffs[i];
        if(std::tie(x.source,x.target,x.distance)!=std::tie(y.source,y.target,y.distance))return false;
    }
    return true;
}
} // namespace

std::vector<Action> mlir::pto::structured_sync::actionsForInvocations(
    const InvocationModel &m,const InvocationPlan &p) {
    auto actions=actionsForPlan(m.region,p.local);uint64_t order=actions.size();
    for(const auto &h:p.handoffs) {
        if(h.source>=m.region.atoms.size()||h.target>=m.region.atoms.size())return {};
        Lane s=m.region.atoms[h.source].lane,t=m.region.atoms[h.target].lane;
        Action set{Action::Set,h.source,true,order++,s,t,h.key,0};
        Action wait{Action::Wait,h.target,false,order++,s,t,h.key,0};
        set.invocation=Action::ToNextInvocation;wait.invocation=Action::FromPreviousInvocation;
        if(loopAtom(m.region,h.source))set.participation=Action::Last;
        if(loopAtom(m.region,h.target))wait.participation=Action::First;
        if(auto r=pairExistence(m.region,h.source,h.target)) {
            set.invocationBodyGuard=wait.invocationBodyGuard=true;
            set.invocationGuardResidue=wait.invocationGuardResidue=*r;
        }
        actions.push_back(set);actions.push_back(wait);
    }
    for(const auto &b:p.barriers) {
        if(b.target>=m.region.atoms.size())return {};
        auto lane=m.region.atoms[b.target].lane;
        Action a{Action::Barrier,b.target,false,order++,lane,lane,0,0};
        a.invocation=Action::FromPreviousInvocation;
        if(loopAtom(m.region,b.target))a.participation=Action::First;
        a.invocationBodyGuard=b.bodyGuard;a.invocationGuardResidue=b.guardResidue;
        actions.push_back(a);
    }
    return actions;
}

InvocationResult mlir::pto::structured_sync::verifyInvocations(
    const InvocationModel &m,const std::vector<Action> &actions) {
    InvocationResult out;if(!validInvocationModel(m,out.reason))return out;
    out.status=Status::InvalidPlan;
    using Key=std::tuple<Lane,Lane,unsigned>;
    std::vector<Action> local;std::map<Key,std::vector<const Action*>> carried;std::set<Key> localKeys;
    std::set<std::tuple<std::size_t,bool,uint64_t>> positions;
    for(const auto &a:actions) {
        if(a.anchor>=m.region.atoms.size()||!positions.emplace(a.anchor,a.after,a.order).second) {
            out.reason="invalid or duplicate invocation action point";return out;
        }
        if(a.invocation==Action::Local) {
            if(a.invocationBodyGuard||a.invocationGuardResidue) {
                out.reason="unexpected invocation predicate on local action";return out;
            }
            local.push_back(a);if(a.kind!=Action::Barrier)localKeys.insert({a.source,a.target,a.key});continue;
        }
        if((a.invocation!=Action::ToNextInvocation&&a.invocation!=Action::FromPreviousInvocation)||
            a.distanceInIterations||a.guardResidue||(!a.invocationBodyGuard&&a.invocationGuardResidue)||
            (a.invocationBodyGuard&&(!m.region.recurring||a.invocationGuardResidue>=m.region.period))) {
            out.reason="invalid invocation participation domain";return out;
        }
        const auto expected=loopAtom(m.region,a.anchor)?(a.kind==Action::Set?Action::Last:Action::First):Action::Every;
        if(a.participation!=expected) {out.reason="re-entry endpoint lost its first/last occurrence";return out;}
        if(a.kind==Action::Barrier) {
            if(a.invocation!=Action::FromPreviousInvocation||a.after||a.source!=a.target||
                a.source!=m.region.atoms[a.anchor].lane||!m.region.target.barrier(a.source)) {
                out.reason="invalid re-entry barrier";return out;
            }
            out.plan.barriers.push_back({a.anchor,a.invocationBodyGuard,a.invocationGuardResidue});continue;
        }
        if((a.kind!=Action::Set&&a.kind!=Action::Wait)||!m.region.target.event(a.source,a.target)||
            !m.region.target.available(a.source,a.target,a.key)||
            (a.kind==Action::Set?(!a.after||a.source!=m.region.atoms[a.anchor].lane||a.invocation!=Action::ToNextInvocation):
                (a.after||a.target!=m.region.atoms[a.anchor].lane||a.invocation!=Action::FromPreviousInvocation))) {
            out.reason="invalid re-entry event direction or key";return out;
        }
        carried[{a.source,a.target,a.key}].push_back(&a);
    }
    auto inside=verify(m.region,local);
    if(inside.status!=Status::Applied){out.reason="local invocation: "+inside.reason;return out;}
    out.plan.local=std::move(inside.plan);
    for(const auto &entry:carried) {
        if(entry.second.size()!=2||localKeys.count(entry.first)) {
            out.reason="re-entry key must own one pair and remain disjoint from local keys";return out;
        }
        auto *s=entry.second[0],*w=entry.second[1];if(s->kind==Action::Wait)std::swap(s,w);
        if(s->kind!=Action::Set||w->kind!=Action::Wait) {out.reason="incomplete re-entry pair";return out;}
        auto r=pairExistence(m.region,s->anchor,w->anchor);
        if(s->invocationBodyGuard!=bool(r)||w->invocationBodyGuard!=bool(r)||
            (r&&(s->invocationGuardResidue!=*r||w->invocationGuardResidue!=*r))) {
            out.reason="re-entry endpoints disagree on existence";return out;
        }
        out.plan.handoffs.push_back({s->anchor,w->anchor,s->key});
    }
    const auto expected=actionsForInvocations(m,out.plan);
    std::set<InvocationSignature> wanted,actual;
    for(const auto &a:expected)wanted.insert(invocationSignature(a));
    for(const auto &a:actions)actual.insert(invocationSignature(a));
    if(wanted.size()!=expected.size() || actual.size()!=actions.size() || wanted!=actual) {
        out.reason="reconstructed invocation command population is ambiguous";return out;
    }
    auto graphs=invocationGraphs(m.region,out.plan,&actions);out.proofViews=graphs.size();
    if(graphs.empty()){out.reason="unsupported invocation interface";return out;}
    for(const auto &r:m.carried)if(!invocationCovered(graphs,r)) {
        out.failure=InvocationFailure::Completion;
        out.reason="cross-invocation reader/write requirement not supplied";return out;
    }
    for(const auto &g:graphs) {
        if(!g->acyclic()) {
            out.failure=InvocationFailure::Progress;
            out.reason="cross-invocation causal constraints are cyclic";return out;
        }
        if(!g->recycles()) {
            out.failure=InvocationFailure::EventReuse;
            out.reason="cross-invocation consumption-before-rearm not established";return out;
        }
        out.graphVisits+=g->visits;
    }
    out.status=Status::Applied;out.reason="local transfer and inductive invocation interface verified";return out;
}

bool mlir::pto::structured_sync::suppliesInvocation(
    const InvocationModel &m,const InvocationPlan &p,const InvocationRequirement &r) {
    std::string why;if(!validInvocationModel(m,why)||r.source>=m.region.atoms.size()||r.target>=m.region.atoms.size())return false;
    // This query requires a well-formed local plan. It is not a safety override.
    if(verify(m.region,actionsForPlan(m.region,p.local)).status!=Status::Applied)return false;
    for(const auto &h:p.handoffs)if(h.source>=m.region.atoms.size()||h.target>=m.region.atoms.size())return false;
    for(const auto &b:p.barriers)if(b.target>=m.region.atoms.size())return false;
    return invocationCovered(invocationGraphs(m.region,p),r);
}

InvocationResult mlir::pto::structured_sync::constructInvocations(const InvocationModel &m) {
    InvocationResult out;if(!validInvocationModel(m,out.reason))return out;
    for(const auto &r:m.carried)if(r.property==Property::Visibility) {
        out.reason="visibility has no qualified invocation realization";return out;
    }
    auto inner=construct(m.region);
    if(inner.status!=Status::Applied){out.status=inner.status;out.reason=inner.reason;return out;}
    out.plan.local=inner.plan;
    auto graphs=invocationGraphs(m.region,out.plan);
    for(auto q:schedule(m.region)) {
        using Group=std::tuple<Lane,Segment,uint64_t>;
        std::map<Group,InvocationRequirement> latest;
        for(const auto &r:m.carried)if(r.target==q && m.region.atoms[r.source].lane!=m.region.atoms[q].lane &&
                                     !invocationCovered(graphs,r)) {
            const auto &a=m.region.atoms[r.source];Group group{a.lane,a.segment,a.residue};auto it=latest.find(group);
            if(it==latest.end()||a.order>m.region.atoms[it->second.source].order)latest[group]=r;
        }
        for(const auto &entry:latest) {
            const auto &r=entry.second;if(invocationCovered(graphs,r))continue;
            if(!m.region.target.event(m.region.atoms[r.source].lane,m.region.atoms[q].lane)) {
                out.reason="cross-invocation event direction unavailable";return out;
            }
            out.plan.handoffs.push_back({r.source,q,0});graphs=invocationGraphs(m.region,out.plan);
        }
    }
    for(auto q:schedule(m.region)) {
        bool missing=false,unconditional=false;std::optional<uint64_t> residue;
        for(const auto &r:m.carried)if(r.target==q&&m.region.atoms[r.source].lane==m.region.atoms[q].lane&&
                                     !invocationCovered(graphs,r)) {
            missing=true;auto e=pairExistence(m.region,r.source,r.target);
            if(!e)unconditional=true;else residue=residue?std::min(*residue,*e):*e;
        }
        if(!missing)continue;
        if(!m.region.target.barrier(m.region.atoms[q].lane)) {out.reason="cross-invocation barrier unavailable";return out;}
        out.plan.barriers.push_back({q,!unconditional&&bool(residue),unconditional?0:residue.value_or(0)});
        graphs=invocationGraphs(m.region,out.plan);
    }
    for(const auto &r:m.carried)if(!invocationCovered(graphs,r)) {
        out.status=Status::InvalidPlan;out.reason="invocation staircase left an original requirement";return out;
    }
    // Both populations are now selected. Keep re-entry streams on distinct
    // keys and reassign the unchanged local plan from the remaining pool.
    // This is conservative allocation, not a reset, acknowledgement or drain.
    auto reserved=m.region;using Domain=std::pair<Lane,Lane>;
    std::map<Domain,std::set<unsigned>> used;
    for(auto &h:out.plan.handoffs) {
        Domain d{m.region.atoms[h.source].lane,m.region.atoms[h.target].lane};bool assigned=false;
        for(unsigned k:m.region.target.compilerKeys)if(m.region.target.available(d.first,d.second,k)&&!used[d].count(k)) {
            h.key=k;used[d].insert(k);reserved.target.reservations.push_back({d.first,d.second,k});assigned=true;break;
        }
        if(!assigned){out.status=Status::AllocationFailure;out.reason="distinct re-entry keys do not fit";return out;}
    }
    auto local=construct(reserved);
    if(local.status!=Status::Applied){out.status=local.status;out.reason="local reallocation: "+local.reason;return out;}
    if(!sameLocalShape(inner.plan,local.plan)) {
        out.status=Status::InvalidPlan;out.reason="allocation changed the local occurrence boundaries";return out;
    }
    out.plan.local=std::move(local.plan);
    auto checked=verifyInvocations(m,actionsForInvocations(m,out.plan));
    if(checked.status!=Status::Applied) {
        // Recycling failure is not hardware infeasibility. Other verifier
        // failures are NOT allocation outcomes and must retain their severity.
        checked.status=checked.constructionStatus();return checked;
    }
    out.status=Status::Applied;out.reason="structured nested invocation composition";
    out.proofViews=checked.proofViews;out.graphVisits=checked.graphVisits;return out;
}


//===----------------------------------------------------------------------===//
// S6: complete local-region refinement and original-cut startup emission.
// These utilities add no target premise and do not run on re-entry components.
//===----------------------------------------------------------------------===//
namespace {
bool renderRegionPlan(const Model &m,const Plan &p,std::vector<Action> &actions) {
    std::string reason;if(!validModel(m,reason))return false;
    for(auto q:p.barriers)if(q>=m.atoms.size())return false;
    for(auto q:p.firstBarriers)if(q>=m.atoms.size())return false;
    for(const auto &h:p.handoffs)if(!iterationDistance(m,h))return false;
    actions=actionsForPlan(m,p);
    return true;
}
Result badRegionPlan() {
    Result out;out.status=Status::InvalidPlan;out.reason="invalid complete-region plan endpoints";return out;
}
}
Result mlir::pto::structured_sync::refineRegionHandoffs(const Model &m,const Plan &input) {
    std::vector<Action> inputActions;
    if(!renderRegionPlan(m,input,inputActions))return badRegionPlan();
    auto checked=verify(m,inputActions);
    if(checked.status!=Status::Applied)return checked;
    Result out;out.status=Status::Applied;out.plan=input;
    for(std::size_t i=out.plan.handoffs.size();i>0;--i) {
        Plan trial=out.plan;
        trial.handoffs.erase(trial.handoffs.begin()+std::ptrdiff_t(i-1));
        ++out.refinementAttempts;
        auto result=verify(m,actionsForPlan(m,trial));
        out.completionRelaxations=plus(out.completionRelaxations,result.completionRelaxations);
        out.eventRelaxations=plus(out.eventRelaxations,result.eventRelaxations);
        if(result.status==Status::Applied) {
            out.plan=std::move(trial);++out.removedHandoffs;
        }
    }
    out.reason="complete local-region reverse deletion verified";
    return out;
}

namespace {
bool plainSiteAction(const Action &a) {
    return a.invocation==Action::Local && !a.invocationBodyGuard && !a.invocationGuardResidue &&
        a.participation==Action::Every && !a.distanceInIterations && !a.guardResidue;
}
bool matchingSiteActions(const Action &a,const Action &b) {
    return plainSiteAction(a)&&plainSiteAction(b) &&
        std::tie(a.kind,a.after,a.source,a.target,a.key)==
        std::tie(b.kind,b.after,b.source,b.target,b.key);
}
bool validSiteOrigins(const Model &m,const std::vector<SiteOrigin> &origins) {
    if(origins.size()!=m.atoms.size())return false;
    std::map<std::pair<std::size_t,SiteOrigin::Phase>,std::size_t> identities;
    std::map<std::size_t,Lane> lanes;
    for(std::size_t i=0;i<origins.size();++i) {
        const auto &o=origins[i];const auto &a=m.atoms[i];
        if(o.phase==SiteOrigin::Other)continue;
        if(m.period!=1||!m.recurring||a.residue ||
           (o.phase!=SiteOrigin::Initial&&o.phase!=SiteOrigin::Steady) ||
           a.segment!=(o.phase==SiteOrigin::Initial?Segment::Prelude:Segment::Body) ||
           !identities.emplace(std::make_pair(o.payload,o.phase),i).second)return false;
        auto lane=lanes.emplace(o.payload,a.lane);
        if(!lane.second&&lane.first->second!=a.lane)return false;
    }
    return true;
}
}

std::optional<std::vector<EmissionSite>> mlir::pto::structured_sync::startupEmissionSites(
    const Model &m,const std::vector<SiteOrigin> &origins,const std::vector<Action> &actions) {
    if(!validSiteOrigins(m,origins))return {};
    // Processing order is stable: original integer identities, never pointers.
    std::map<std::pair<std::size_t,bool>,std::vector<std::size_t>> points;
    std::set<std::tuple<std::size_t,bool,uint64_t>> positions;
    for(std::size_t i=0;i<actions.size();++i) {
        const auto &a=actions[i];
        if(a.anchor>=origins.size() || !positions.emplace(a.anchor,a.after,a.order).second)return {};
        points[{origins[a.anchor].payload,a.after}].push_back(i);
    }
    std::vector<EmissionSite> result;
    for(auto &point:points) {
        auto &indices=point.second;
        std::stable_sort(indices.begin(),indices.end(),[&](auto a,auto b){return actions[a].order<actions[b].order;});
        std::vector<std::size_t> initial,steady;
        bool other=false;
        for(auto index:indices) {
            auto phase=origins[actions[index].anchor].phase;
            if(phase==SiteOrigin::Other){other=true;break;}
            (phase==SiteOrigin::Initial?initial:steady).push_back(index);
        }
        if(other||initial.empty()||steady.empty()) {
            for(auto i:indices)result.push_back({{i}});
            continue;
        }
        // Longest common subsequence of MATCHABLE commands; merging arbitrary
        // equal commands without this alignment could reorder a same-key FIFO.
        const auto n=initial.size(),k=steady.size();
        std::vector<std::vector<std::size_t>> length(n+1,std::vector<std::size_t>(k+1));
        for(std::size_t i=n;i>0;--i)for(std::size_t j=k;j>0;--j) {
            bool equal=matchingSiteActions(actions[initial[i-1]],actions[steady[j-1]]);
            length[i-1][j-1]=equal?1+length[i][j]:std::max(length[i][j-1],length[i-1][j]);
        }
        std::size_t i=0,j=0;
        while(i<n||j<k) {
            if(i<n&&j<k&&matchingSiteActions(actions[initial[i]],actions[steady[j]])) {
                result.push_back({{initial[i++],steady[j++]}});
            } else if(i<n&&(j==k||length[i+1][j]>=length[i][j+1]))result.push_back({{initial[i++]}});
            else result.push_back({{steady[j++]}});
        }
    }
    return result;
}

Result mlir::pto::structured_sync::coalesceStartupHandoffs(
    const Model &m,const std::vector<SiteOrigin> &origins,const Plan &input) {
    std::vector<Action> inputActions;
    if(!renderRegionPlan(m,input,inputActions))return badRegionPlan();
    auto valid=verify(m,inputActions);if(valid.status!=Status::Applied)return valid;
    Result out;out.status=Status::Applied;out.plan=input;
    auto sites=startupEmissionSites(m,origins,actionsForPlan(m,input));
    if(!sites){out.status=Status::InvalidPlan;out.reason="invalid original-cut startup mapping";return out;}
    std::size_t siteCount=sites->size();
    // One pass over boundary pairs. The population never grows; a trial only
    // adopts an already selected recurring key with a shared ORIGINAL endpoint.
    if(m.allowBoundaryKeyReuse&&m.recurring&&m.period==1) {
        for(std::size_t i=0;i<out.plan.handoffs.size();++i) {
            const auto boundary=out.plan.handoffs[i];
            if(m.atoms[boundary.source].segment==Segment::Body||
               m.atoms[boundary.target].segment==Segment::Body)continue;
            for(std::size_t j=0;j<out.plan.handoffs.size();++j) {
                const auto periodic=out.plan.handoffs[j];
                if(m.atoms[periodic.source].segment!=Segment::Body||
                   m.atoms[periodic.target].segment!=Segment::Body||periodic.distance ||
                   m.atoms[boundary.source].lane!=m.atoms[periodic.source].lane||
                   m.atoms[boundary.target].lane!=m.atoms[periodic.target].lane||
                   out.plan.handoffs[i].key==periodic.key)continue;
                auto common=[&](std::size_t a,std::size_t b) {
                    return origins[a].phase==SiteOrigin::Initial&&origins[b].phase==SiteOrigin::Steady&&
                           origins[a].payload==origins[b].payload;
                };
                if(!common(boundary.source,periodic.source)&&!common(boundary.target,periodic.target))continue;
                Plan trial=out.plan;trial.handoffs[i].key=periodic.key;
                auto actions=actionsForPlan(m,trial);auto candidate=startupEmissionSites(m,origins,actions);
                if(!candidate||candidate->size()>=siteCount)continue;
                ++out.rekeyAttempts;
                auto result=verify(m,actions);
                out.completionRelaxations=plus(out.completionRelaxations,result.completionRelaxations);
                out.eventRelaxations=plus(out.eventRelaxations,result.eventRelaxations);
                if(result.status!=Status::Applied)continue;
                out.plan=std::move(trial);siteCount=candidate->size();++out.rekeyedHandoffs;
                break;
            }
        }
    }
    out.coalescedSites=actionsForPlan(m,out.plan).size()-siteCount;
    // Reconstruct the proposed command order independently of its old ordinals.
    // Paired members execute in disjoint phases, but each phase's FIFO remains.
    auto actions=actionsForPlan(m,out.plan);
    sites=startupEmissionSites(m,origins,actions);
    std::vector<Action> recovered;
    for(std::size_t order=0;order<sites->size();++order)for(auto member:(*sites)[order].members) {
        auto a=actions[member];a.order=order;recovered.push_back(a);
    }
    auto checked=verify(m,recovered);
    if(checked.status!=Status::Applied)return checked;
    out.reason="same-original-cut startup sites and whole-region key continuation verified";
    return out;
}


std::optional<std::vector<HandoffAudit>> mlir::pto::structured_sync::auditRegionHandoffs(
    const Model &m,const Plan &input) {
    std::vector<Action> actions;
    if(!renderRegionPlan(m,input,actions)||verify(m,actions).status!=Status::Applied)return {};
    std::vector<HandoffAudit> out;
    for(std::size_t i=0;i<input.handoffs.size();++i) {
        Plan trial=input;trial.handoffs.erase(trial.handoffs.begin()+std::ptrdiff_t(i));
        HandoffAudit a;a.handoff=i;
        Model protocol=m;protocol.requirements.clear();
        auto checked=verify(protocol,actionsForPlan(protocol,trial));
        a.protocolWithout=checked.status==Status::Applied;a.protocolReason=checked.reason;
        if(!hasBoundaries(m)) {
            Completion c(m);
            for(const auto &h:trial.handoffs)c.handoff(h);
            for(auto q:trial.barriers)c.barrier(q);
            for(std::size_t r=0;r<m.requirements.size();++r)if(!c.has(m.requirements[r]))a.completionLost.push_back(r);
        } else {
            BodyProjection body(m);Completion inside(body.model);
            for(auto h:trial.handoffs)if(periodicEdge(m,h)) {
                h.source=body.local[h.source];h.target=body.local[h.target];inside.handoff(h);
            }
            for(auto q:trial.barriers)if(m.atoms[q].segment==Segment::Body)inside.barrier(body.local[q]);
            BoundaryCuts first(m,trial,true);auto tails=hasExitRequirements(m)?tailCuts(m,trial):Tails{};
            for(std::size_t r=0;r<m.requirements.size();++r) {
                auto x=m.requirements[r];bool has=false;
                if(boundaryRequirement(m,x))has=m.atoms[x.target].segment==Segment::Epilogue?allHave(tails,x):first.has(x);
                else {x.source=body.local[x.source];x.target=body.local[x.target];has=inside.has(x);}
                if(!has)a.completionLost.push_back(r);
            }
        }
        out.push_back(std::move(a));
    }
    return out;
}
