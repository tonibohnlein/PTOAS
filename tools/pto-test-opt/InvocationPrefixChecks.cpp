// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Interpret emitted control independently; compare payload closure and exact token matching.
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/InvocationPrefix.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>
#include <tuple>
using namespace mlir;
namespace fs = pto::frontiersynch;
namespace {
constexpr unsigned P = unsigned(pto::PIPE::PIPE_MTE2);
constexpr unsigned R = unsigned(pto::PIPE::PIPE_MTE3);
constexpr unsigned Q = unsigned(pto::PIPE::PIPE_V);
constexpr unsigned S = unsigned(pto::PIPE::PIPE_S);
struct Event {
    unsigned kind = 0, role = 0, source = 0, target = 0;
    int64_t ordinal = 0, record = 0, plan = 0;
};
struct Fixture {
    OwningOpRef<ModuleOp> module;
    func::FuncOp function;
    void body(OpBuilder& b, ValueRange flags) {
        auto loc = b.getUnknownLoc();
        for (unsigned choice = 0; choice < 4; ++choice) {
            auto branch = b.create<scf::IfOp>(loc, flags[choice], false);
            OpBuilder::InsertionGuard arm(b); b.setInsertionPointToStart(branch.thenBlock());
            unsigned role = choice < 3 ? choice : 4;
            auto tag = b.create<arith::ConstantIndexOp>(loc, role);
            auto call = b.create<func::CallOp>(loc, "payload", TypeRange{}, ValueRange{tag});
            call->setAttr("test.role", b.getI32IntegerAttr(role));
            if (choice == 2) {
                auto older = b.create<arith::ConstantIndexOp>(loc, 3);
                auto second = b.create<func::CallOp>(loc, "payload", TypeRange{}, ValueRange{older});
                second->setAttr("test.role", b.getI32IntegerAttr(3));
            }
        }
    }
    void siblings(OpBuilder& b, Value zero, Value one) {
        auto loc = b.getUnknownLoc();
        for (unsigned visit = 0; visit < 3; ++visit) {
            auto position = b.create<arith::ConstantIndexOp>(loc, visit);
            auto active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                position, function.getArgument(0));
            auto outer = b.create<scf::IfOp>(loc, active, false);
            OpBuilder::InsertionGuard restore(b); b.setInsertionPointToStart(outer.thenBlock());
            auto loop = b.create<scf::ForOp>(loc, zero, one, one);
            b.setInsertionPointToStart(loop.getBody());
            body(b, function.getArguments().slice(1 + visit*4, 4));
        }
    }
    void repeated(OpBuilder& b, Value zero, Value one) {
        auto loc = b.getUnknownLoc();
        auto outer = b.create<scf::ForOp>(loc, zero, function.getArgument(0), one);
        OpBuilder::InsertionGuard restore(b); b.setInsertionPointToStart(outer.getBody());
        SmallVector<Value> flags;
        for (unsigned choice = 0; choice < 4; ++choice) {
            Value flag = function.getArgument(9 + choice);
            for (int visit = 1; visit >= 0; --visit) {
                auto position = b.create<arith::ConstantIndexOp>(loc, visit);
                auto active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                    outer.getInductionVar(), position);
                flag = b.create<arith::SelectOp>(loc, active, function.getArgument(1 + visit*4 + choice), flag);
            }
            flags.push_back(flag);
        }
        auto inner = b.create<scf::ForOp>(loc, zero, one, one);
        b.setInsertionPointToStart(inner.getBody()); body(b, flags);
    }
    explicit Fixture(MLIRContext& context, bool loops = false)
        : module(ModuleOp::create(UnknownLoc::get(&context))) {
        OpBuilder b(&context); auto loc = b.getUnknownLoc();
        b.setInsertionPointToEnd(module->getBody());
        auto payload = b.create<func::FuncOp>(loc, "payload", b.getFunctionType({b.getIndexType()}, {}));
        payload.setPrivate();
        SmallVector<Type> types(13, b.getI1Type()); types[0] = b.getIndexType();
        function = b.create<func::FuncOp>(loc, "prefix", b.getFunctionType(types, {}));
        b.setInsertionPointToStart(function.addEntryBlock());
        auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
        auto one = b.create<arith::ConstantIndexOp>(loc, 1);
        if (loops) { repeated(b, zero, one); } else { siblings(b, zero, one); }
        b.create<func::ReturnOp>(loc);
    }
};
unsigned pipe(unsigned role) { return role == 0 ? P : role == 1 ? R : role == 4 ? S : Q; }
fs::InvocationPrefixRecipe recipe(Fixture& f)
{
    fs::InvocationPrefixRecipe result; result.privateScalarControl = true; result.planId = 19;
    result.sourceBounds = {{P, 3}, {R, 3}}; result.directions = {{P,Q}, {P,S}, {R,Q}};
    Builder b(f.function.getContext());
    for (unsigned i = 0; i < 3; ++i) { result.slots.push_back({b.getIndexType(), b.getIndexAttr(0)}); }
    f.function.walk([&](func::CallOp call) {
        const auto role = unsigned(call->getAttrOfType<IntegerAttr>("test.role").getInt());
        if (role < 2) {
            result.sources.push_back({call, pipe(role), [role](OpBuilder&, const IRMapping&,
                MutableArrayRef<Value> state, Value ordinal, std::string&) {
                    if (role == 0) { state[1] = state[0]; state[0] = ordinal; }
                    else { state[2] = ordinal; }
                    return success();
                }});
        } else {
            auto incoming = [&](unsigned source, unsigned slot) {
                result.consumers.push_back({call, {source, pipe(role)},
                    [slot](OpBuilder&, const IRMapping&, ArrayRef<Value> state, std::string&) -> FailureOr<Value> {
                        return state[slot];
                    }});
            };
            incoming(P, role == 3 ? 1 : 0);
            if (role == 2) { incoming(R, 2); }
        }
    });
    return result;
}
struct Execution {
    llvm::DenseMap<Value, int64_t> values;
    std::vector<Event> events;
    bool valid = true;
    int64_t get(Value v) {
        auto it = values.find(v); if (it == values.end()) { valid = false; return 0; } return it->second;
    }
    SmallVector<int64_t> read(ValueRange operands) {
        SmallVector<int64_t> out; for (auto v : operands) { out.push_back(get(v)); } return out;
    }
    void assign(ValueRange targets, ArrayRef<int64_t> data) {
        const bool invalid134 = targets.size() != data.size();
        if (invalid134) { valid = false; return; }
        for (auto [v, n] : llvm::zip(targets, data)) { values[v] = n; }
    }
    bool scalar(Operation& op) {
        if (auto c = dyn_cast<arith::ConstantOp>(op)) {
            auto n = dyn_cast<IntegerAttr>(c.getValue()); if (!n) { return false; }
            values[c] = n.getValue().getSExtValue();
        } else if (auto add = dyn_cast<arith::AddIOp>(op)) {
            values[add] = get(add.getLhs()) + get(add.getRhs());
        } else if (auto select = dyn_cast<arith::SelectOp>(op)) {
            values[select] = get(get(select.getCondition()) ? select.getTrueValue() : select.getFalseValue());
        } else if (auto compare = dyn_cast<arith::CmpIOp>(op)) {
            auto a = get(compare.getLhs()), b = get(compare.getRhs());
            const bool invalid147 = compare.getPredicate() == arith::CmpIPredicate::slt;
            if (invalid147) { values[compare] = a < b; }
            else if (compare.getPredicate() == arith::CmpIPredicate::ugt) {
                values[compare] = uint64_t(a) > uint64_t(b);
            } else if (compare.getPredicate() == arith::CmpIPredicate::eq) { values[compare] = a == b; }
            else { return false; }
        } else { return false; }
        return true;
    }
    SmallVector<int64_t> block(Block& body) {
        for (auto& op : body) {
            if (auto call = dyn_cast<func::CallOp>(op)) {
                const auto role = unsigned(get(call.getOperand(0)));
                events.push_back({0, role, pipe(role)});
            } else if (auto loop = dyn_cast<scf::ForOp>(op)) {
                auto carried = read(loop.getInitArgs());
                const auto lower = get(loop.getLowerBound()), upper = get(loop.getUpperBound());
                const auto step = get(loop.getStep());
                if (step <= 0 || upper > 32 || lower < 0) { valid = false; return {}; }
                for (auto i = lower; i < upper; i += step) {
                    values[loop.getInductionVar()] = i; assign(loop.getRegionIterArgs(), carried);
                    carried = block(*loop.getBody());
                }
                assign(loop.getResults(), carried);
            } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
                auto& arm = get(branch.getCondition()) ? branch.getThenRegion() : branch.getElseRegion();
                auto output = arm.empty() ? SmallVector<int64_t>{} : block(arm.front());
                assign(branch.getResults(), output);
            } else if (isa<scf::YieldOp, func::ReturnOp>(op)) { return read(op.getOperands()); }
            else if (op.getName().getStringRef() == "pto.logical_set" ||
                     op.getName().getStringRef() == "pto.logical_wait") {
                auto source = op.getAttrOfType<pto::PipeAttr>("src_pipe");
                auto target = op.getAttrOfType<pto::PipeAttr>("dst_pipe");
                const bool invalid179 = !source || !target || op.getNumOperands() != 1;
                if (invalid179) { valid = false; return {}; }
                events.push_back({op.getName().getStringRef() == "pto.logical_set" ? 1U : 2U, 0,
                    unsigned(source.getPipe()), unsigned(target.getPipe()), get(op.getOperand(0)),
                    op.getAttrOfType<IntegerAttr>("record_id").getInt(),
                    op.getAttrOfType<IntegerAttr>("plan_id").getInt()});
            } else if (!scalar(op)) { valid = false; return {}; }
        }
        valid = false; return {};
    }
    void run(func::FuncOp f, unsigned trips, unsigned bits) {
        values[f.getArgument(0)] = trips;
        for (unsigned i = 0; i < 12; ++i) { values[f.getArgument(i + 1)] = (bits >> i) & 1; }
        block(f.front());
    }
};
using Graph = std::vector<uint64_t>;
void edge(Graph& graph, unsigned from, unsigned to) { graph[from] |= uint64_t(1) << to; }
Graph native(ArrayRef<unsigned> roles) {
    Graph graph(2 * roles.size()); std::map<unsigned, unsigned> previous;
    for (unsigned i = 0; i < roles.size(); ++i) {
        edge(graph, 2*i, 2*i+1); auto p = pipe(roles[i]);
        if (auto it = previous.find(p); it != previous.end()) {
            edge(graph, 2*it->second, 2*i); edge(graph, 2*it->second+1, 2*i+1);
        }
        previous[p] = i;
    }
    return graph;
}
void close(Graph& graph) {
    for (unsigned k = 0; k < graph.size(); ++k) {
        for (auto& row : graph) {
            const bool reachable = (row >> k) & 1;
            if (reachable) { row |= graph[k]; }
        }
    }
}
Graph reference(ArrayRef<unsigned> roles) {
    auto graph = native(roles); std::vector<unsigned> p, r;
    for (unsigned i = 0; i < roles.size(); ++i) {
        auto role = roles[i];
        if (role == 0) { p.push_back(i); }
        if (role == 1) { r.push_back(i); }
        if (role >= 2) {
            const auto needed = role == 3 && !p.empty() ? p.size()-1 : p.size();
            if (needed) { edge(graph, 2*p[needed-1]+1, 2*i); }
            if (role == 2 && !r.empty()) { edge(graph, 2*r.back()+1, 2*i); }
        }
    }
    close(graph); return graph;
}
// A WAIT gates the next payload start on its destination; earlier WAITs also
// gate later starts through native start order. Tail waits have no payload sink.
bool verifyTrace(ArrayRef<Event> events, ArrayRef<unsigned> roles) {
    const bool invalid229 = roles.size() > 31;
    if (invalid229) { return false; }
    auto graph = native(roles);
    using Key = std::tuple<unsigned,unsigned,int64_t,int64_t,int64_t>;
    std::map<Key,unsigned> publications; std::set<Key> consumed;
    std::map<unsigned,unsigned> last, counts;
    std::map<unsigned,std::vector<unsigned>> pending;
    std::map<std::pair<unsigned,unsigned>,int64_t> records;
    unsigned next = 0;
    for (auto event : events) {
        if (event.kind == 0) {
            const bool invalid239 = next >= roles.size() || event.role != roles[next];
            if (invalid239) { return false; }
            for (auto source : pending[event.source]) { edge(graph, 2*source+1, 2*next); }
            pending[event.source].clear(); last[event.source] = next++; ++counts[event.source];
            continue;
        }
        Key key{event.source,event.target,event.ordinal,event.record,event.plan};
        auto direction = std::make_pair(event.source,event.target);
        auto entry = records.emplace(direction,event.record).first;
        if (entry->second != event.record || event.plan != 19 || event.ordinal <= 0) { return false; }
        const bool allowed = direction == std::make_pair(P,Q) || direction == std::make_pair(P,S) ||
                             direction == std::make_pair(R,Q);
        if (!allowed) { return false; }
        if (event.kind == 1) {
            const bool invalid252 = !last.count(event.source) || event.ordinal != counts[event.source] ||
                !publications.emplace(key,last[event.source]).second;
            if (invalid252) { return false; }
        } else {
            auto found = publications.find(key);
            const bool invalid256 = found == publications.end() || !consumed.insert(key).second;
            if (invalid256) { return false; }
            pending[event.target].push_back(found->second);
        }
    }
    std::set<int64_t> recordIds;
    for (auto [direction, record] : records) {
        if (!recordIds.insert(record).second) { return false; }
    }
    const bool invalid264 = next != roles.size() || publications.size() != consumed.size() ||
        publications.size() != 2*counts[P]+counts[R];
    if (invalid264) { return false; }
    close(graph);
    for (unsigned i = 0; i < graph.size(); ++i) { if ((graph[i] >> i) & 1) { return false; } }
    return graph == reference(roles);
}
bool mutationChecks(ArrayRef<Event> original, ArrayRef<unsigned> roles) {
    std::vector<Event> events(original.begin(), original.end());
    auto wait = llvm::find_if(events, [](const Event& e) { return e.kind == 2; });
    if (wait == events.end()) { return false; }
    const auto index = std::size_t(wait - events.begin());
    events.erase(events.begin() + index);
    if (verifyTrace(events, roles)) { return false; }
    events.assign(original.begin(), original.end());
    events.insert(events.begin() + index, events[index]);
    if (verifyTrace(events, roles)) { return false; }
    events.assign(original.begin(), original.end()); events[index].ordinal = 99;
    return !verifyTrace(events, roles);
}
std::string render(Operation* op) {
    std::string text; llvm::raw_string_ostream out(text); op->print(out); return text;
}
bool runCases(MLIRContext& context, bool loops, unsigned& cases) {
    Fixture f(context, loops); fs::PrivateStateError error; auto spec = recipe(f);
    auto original = render(f.module->getOperation());
    auto prepared = fs::materializeInvocationPrefix(f.function,spec,error);
    const bool invalid290 = failed(prepared);
    if (invalid290) {
        llvm::errs() << error.message << "\n";
        return false;
    }
    auto metadata = (*prepared)->function->getAttrOfType<DictionaryAttr>("pto.prefix_protocol");
    const bool invalid292 = !metadata || metadata.getAs<IntegerAttr>("version").getInt() != 1;
    if (invalid292) { return false; }
    for (unsigned trips = 0; trips <= 3; ++trips) {
        for (unsigned bits = 0; bits < (1U << (4*trips)); ++bits) {
            Execution baseline, actual;
            baseline.run(f.function,trips,bits); actual.run((*prepared)->function,trips,bits);
            std::vector<unsigned> roles; for (auto event : baseline.events) { roles.push_back(event.role); }
            if (!baseline.valid || !actual.valid || !verifyTrace(actual.events,roles)) {
                llvm::errs() << "prefix execution failed loops=" << loops << " trips=" << trips
                             << " choices=" << bits << "\n";
                return false;
            }
            if (trips == 3 && bits == 4095 && !mutationChecks(actual.events,roles)) { return false; }
            ++cases;
        }
    }
    auto reject = [&](fs::InvocationPrefixRecipe invalid) {
        auto bad = fs::materializeInvocationPrefix(f.function,invalid,error);
        return failed(bad) && original == render(f.module->getOperation());
    };
    auto invalid = spec; invalid.privateScalarControl = false;
    if (!reject(invalid)) { return false; }
    invalid = spec; invalid.sourceBounds[P] = uint64_t(INT64_MAX);
    if (!reject(invalid)) { return false; }
    Fixture foreign(context); invalid = spec;
    invalid.sources.front().original = foreign.function.front().getTerminator();
    if (!reject(invalid)) { return false; }
    invalid = spec;
    invalid.consumers.front().incoming = [](OpBuilder&, const IRMapping&, ArrayRef<Value>,
                                          std::string&) -> FailureOr<Value> { return failure(); };
    if (!reject(invalid)) { return false; }
    invalid = spec;
    invalid.consumers.front().incoming = [](OpBuilder&, const IRMapping&, ArrayRef<Value>,
                                          std::string&) -> FailureOr<Value> { return Value{}; };
    return reject(invalid) && original == render(f.module->getOperation());
}
} // namespace
bool runInvocationPrefixChecks() {
    MLIRContext context; context.disableMultithreading();
    context.loadDialect<func::FuncDialect, scf::SCFDialect, arith::ArithDialect, pto::PTODialect>();
    unsigned cases = 0;
    const bool invalid332 = !runCases(context, false, cases) || !runCases(context, true, cases);
    if (invalid332) { return false; }
    llvm::outs() << "invocation prefix checks passed executions=" << cases << "\n";
    return true;
}
