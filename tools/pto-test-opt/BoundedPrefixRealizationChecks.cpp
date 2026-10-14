// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Unfold original opaque evaluations independently from retained-window history emission.
#include "../../lib/PTO/Transforms/FrontierSynch/Structured/BoundedPrefixRealization.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/IterationPredicates.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/PrefixExpressionProfile.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/SequenceAnalysisInternal.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/RegionalCapabilitiesInternal.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <map>
#include <set>
#include <tuple>
using namespace mlir;
namespace fs = pto::frontiersynch;
namespace {
constexpr unsigned P = unsigned(pto::PIPE::PIPE_MTE2), Q = unsigned(pto::PIPE::PIPE_V);
struct Fixture {
    OwningOpRef<ModuleOp> module;
    func::FuncOp function;
    scf::ForOp outer, loop;
    Value defined, predicate, present;
    Operation *source = nullptr, *target = nullptr;
    std::vector<std::unique_ptr<pto::CompoundInstanceElement>> phases;
    unsigned trips, invocations;
    Fixture(MLIRContext& context, unsigned trips, unsigned invocations)
        : module(ModuleOp::create(UnknownLoc::get(&context))), trips(trips), invocations(invocations)
    {
        OpBuilder b(&context);
        auto loc = b.getUnknownLoc();
        b.setInsertionPointToEnd(module->getBody());
        auto opaqueType = b.getFunctionType({b.getIndexType(), b.getIndexType(), b.getI64Type()}, {b.getI1Type()});
        for (auto name : {"defined", "value", "present"}) {
            b.create<func::FuncOp>(loc, name, opaqueType).setPrivate();
        }
        b.create<func::FuncOp>(
             loc, "payload", b.getFunctionType({b.getIndexType(), b.getIndexType(), b.getIndexType()}, {}))
            .setPrivate();
        function = b.create<func::FuncOp>(
            loc, "bounded_prefix", b.getFunctionType({b.getI64Type(), b.getI64Type(), b.getI64Type()}, {}));
        b.setInsertionPointToStart(function.addEntryBlock());
        auto zero = b.create<arith::ConstantIndexOp>(loc, 0), one = b.create<arith::ConstantIndexOp>(loc, 1);
        auto count = b.create<arith::ConstantIndexOp>(loc, invocations);
        auto lower = b.create<arith::ConstantIndexOp>(loc, -5);
        auto upper = b.create<arith::ConstantIndexOp>(loc, -5 + 3 * int64_t(trips));
        auto step = b.create<arith::ConstantIndexOp>(loc, 3);
        outer = b.create<scf::ForOp>(loc, zero, count, one);
        b.setInsertionPointToStart(outer.getBody());
        loop = b.create<scf::ForOp>(loc, lower, upper, step);
        b.setInsertionPointToStart(loop.getBody());
        auto opaque = [&](StringRef name, unsigned argument) {
            return b
                .create<func::CallOp>(
                    loc, name, TypeRange{b.getI1Type()},
                    ValueRange{outer.getInductionVar(), loop.getInductionVar(), function.getArgument(argument)})
                .getResult(0);
        };
        defined = opaque("defined", 0);
        auto branch = b.create<scf::IfOp>(loc, defined, false);
        {
            OpBuilder::InsertionGuard restore(b);
            b.setInsertionPointToStart(branch.thenBlock());
            predicate = opaque("value", 1);
        }
        present = opaque("present", 2);
        auto selected = b.create<scf::IfOp>(loc, present, false);
        {
            OpBuilder::InsertionGuard restore(b);
            b.setInsertionPointToStart(selected.thenBlock());
            source = b.create<func::CallOp>(
                loc, "payload", TypeRange{}, ValueRange{zero, outer.getInductionVar(), loop.getInductionVar()});
        }
        target = b.create<func::CallOp>(
            loc, "payload", TypeRange{}, ValueRange{one, outer.getInductionVar(), loop.getInductionVar()});
        b.setInsertionPointAfter(outer);
        b.create<func::ReturnOp>(loc);
        phases.push_back(
            std::make_unique<pto::CompoundInstanceElement>(
                0, SmallVector<const pto::BaseMemInfo*>{}, SmallVector<const pto::BaseMemInfo*>{},
                pto::PipelineType::PIPE_MTE2, source->getName()));
        phases.push_back(
            std::make_unique<pto::CompoundInstanceElement>(
                1, SmallVector<const pto::BaseMemInfo*>{}, SmallVector<const pto::BaseMemInfo*>{},
                pto::PipelineType::PIPE_V, target->getName()));
        phases[0]->elementOp = source;
        phases[1]->elementOp = target;
    }
    std::shared_ptr<fs::BoundedLifetimeDemandResult> demands(bool masked)
    {
        auto result = std::make_shared<fs::BoundedLifetimeDemandResult>(std::make_shared<fs::RegionExpressions>());
        result->function = function;
        result->loop = loop;
        auto& e = result->expressions;
        result->base = e.input(result->symbols.addArgument(IndexType::get(function.getContext()), function.getLoc()));
        result->ordinal = fs::CountedLoop::get(loop)->ordinal(e);
        result->predicates =
            std::make_unique<fs::IterationPredicates>(loop, e, llvm::ArrayRef<const pto::CompoundInstanceElement*>{});
        std::array<fs::RegionExpressions::Id, 2> d, v, p;
        for (unsigned offset = 0; offset < 2; ++offset) {
            auto at = e.add(result->base, e.constant(offset));
            d[offset] = result->predicates->at(defined, at);
            v[offset] = result->predicates->at(predicate, at);
            p[offset] = result->predicates->at(present, at);
            result->predicateBindings.push_back({d[offset], {defined, offset}});
            result->predicateBindings.push_back({v[offset], {predicate, offset}});
            result->predicateBindings.push_back({p[offset], {present, offset}});
        }
        auto live = e.lt(result->base, e.constant(trips));
        auto next = e.land(live, e.lt(e.constant(1), e.sub(e.constant(trips), result->base)));
        auto sourcePresence = e.land(live, p[0]);
        result->window.sites = 2;
        result->window.span = 1;
        result->window.payloads = {{0, sourcePresence}, {1, live}, {0, e.land(next, p[1])}, {1, next}};
        result->unconditional = {0, 1};
        auto predicateGuard = masked ? e.select(d[0], v[0], e.boolean(false)) : v[0];
        auto guard = e.land(e.land(sourcePresence, next), predicateGuard);
        result->analysis.sourceDemands = {{0, 3, guard}};
        result->anchors = {
            {phases[0].get(), {}, {source->getBlock(), source}, {source->getBlock(), source->getNextNode()}},
            {phases[1].get(), {}, {target->getBlock(), target}, {target->getBlock(), target->getNextNode()}}};
        return result;
    }
};
struct Event {
    unsigned kind, role;
    uint64_t outer, iv, ordinal;
};
struct Execution {
    explicit Execution(unsigned trips) : trips(trips) {}
    unsigned trips;
    bool valid = true;
    llvm::DenseMap<Value, uint64_t> values;
    std::vector<Event> events;
    std::vector<std::tuple<std::string, uint64_t, uint64_t, uint64_t>> evaluations;
    uint64_t get(Value v)
    {
        auto it = values.find(v);
        if (it == values.end()) {
            valid = false;
            return 0;
        }
        return it->second;
    }
    SmallVector<uint64_t> read(ValueRange operands)
    {
        SmallVector<uint64_t> out;
        for (auto v : operands) {
            out.push_back(get(v));
        }
        return out;
    }
    void assign(ValueRange targets, ArrayRef<uint64_t> data)
    {
        if (targets.size() != data.size()) {
            valid = false;
            return;
        }
        for (auto [v, n] : llvm::zip(targets, data)) {
            values[v] = n;
        }
    }
    bool compare(arith::CmpIOp op)
    {
        auto a = get(op.getLhs()), b = get(op.getRhs());
        bool result;
        switch (op.getPredicate()) {
            case arith::CmpIPredicate::eq:
                result = a == b;
                break;
            case arith::CmpIPredicate::ne:
                result = a != b;
                break;
            case arith::CmpIPredicate::ult:
                result = a < b;
                break;
            case arith::CmpIPredicate::ule:
                result = a <= b;
                break;
            case arith::CmpIPredicate::ugt:
                result = a > b;
                break;
            case arith::CmpIPredicate::uge:
                result = a >= b;
                break;
            case arith::CmpIPredicate::slt:
                result = APInt(64, a).slt(APInt(64, b));
                break;
            case arith::CmpIPredicate::sle:
                result = APInt(64, a).sle(APInt(64, b));
                break;
            default:
                return false;
        }
        values[op] = result;
        return true;
    }
    bool scalar(Operation& op)
    {
        if (auto c = dyn_cast<arith::ConstantOp>(op)) {
            auto value = dyn_cast<IntegerAttr>(c.getValue());
            if (!value) {
                return false;
            }
            auto bits = value.getValue();
            values[c] = (bits.getBitWidth() == 1 ? bits.zextOrTrunc(64) : bits.sextOrTrunc(64)).getZExtValue();
        } else if (auto a = dyn_cast<arith::AddIOp>(op)) {
            values[a] = get(a.getLhs()) + get(a.getRhs());
        } else if (auto a = dyn_cast<arith::SubIOp>(op)) {
            values[a] = get(a.getLhs()) - get(a.getRhs());
        } else if (auto a = dyn_cast<arith::DivUIOp>(op)) {
            auto divisor = get(a.getRhs());
            if (!divisor) {
                return false;
            }
            values[a] = get(a.getLhs()) / divisor;
        } else if (auto a = dyn_cast<arith::AndIOp>(op)) {
            values[a] = get(a.getLhs()) & get(a.getRhs());
        } else if (auto a = dyn_cast<arith::OrIOp>(op)) {
            values[a] = get(a.getLhs()) | get(a.getRhs());
        } else if (auto a = dyn_cast<arith::XOrIOp>(op)) {
            values[a] = get(a.getLhs()) ^ get(a.getRhs());
        } else if (auto a = dyn_cast<arith::SelectOp>(op)) {
            values[a] = get(get(a.getCondition()) ? a.getTrueValue() : a.getFalseValue());
        } else if (auto a = dyn_cast<arith::CmpIOp>(op)) {
            return compare(a);
        } else if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(op)) {
            values[op.getResult(0)] = get(op.getOperand(0));
        } else {
            return false;
        }
        return true;
    }
    bool call(func::CallOp op)
    {
        auto args = read(op.getOperands());
        if (op.getCallee() == "payload") {
            events.push_back({0, unsigned(args[0]), args[1], args[2], 0});
            return true;
        }
        if (args.size() != 3 || op.getNumResults() != 1) {
            return false;
        }
        const auto ordinal = (APInt(64, args[1]).getSExtValue() + 5) / 3;
        const auto index = args[0] * trips + ordinal;
        if (index >= 64) {
            return false;
        }
        const auto value = (args[2] >> index) & 1;
        evaluations.emplace_back(op.getCallee().str(), args[0], args[1], value);
        values[op.getResult(0)] = value;
        return true;
    }
    SmallVector<uint64_t> block(Block& source)
    {
        for (auto& op : source) {
            if (auto invocation = dyn_cast<func::CallOp>(op)) {
                if (!call(invocation)) {
                    valid = false;
                    return {};
                }
            } else if (auto loop = dyn_cast<scf::ForOp>(op)) {
                auto carried = read(loop.getInitArgs());
                auto lower = APInt(64, get(loop.getLowerBound())).getSExtValue();
                auto upper = APInt(64, get(loop.getUpperBound())).getSExtValue();
                auto step = APInt(64, get(loop.getStep())).getSExtValue();
                if (step <= 0 || lower < -5 || upper > 16) {
                    valid = false;
                    return {};
                }
                for (auto i = lower; i < upper; i += step) {
                    values[loop.getInductionVar()] = uint64_t(i);
                    assign(loop.getRegionIterArgs(), carried);
                    carried = block(*loop.getBody());
                }
                assign(loop.getResults(), carried);
            } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
                auto& arm = get(branch.getCondition()) ? branch.getThenRegion() : branch.getElseRegion();
                auto output = arm.empty() ? SmallVector<uint64_t>{} : block(arm.front());
                assign(branch.getResults(), output);
            } else if (isa<scf::YieldOp, func::ReturnOp>(op)) {
                return read(op.getOperands());
            } else if (
                op.getName().getStringRef() == "pto.logical_set" || op.getName().getStringRef() == "pto.logical_wait") {
                auto p = op.getAttrOfType<pto::PipeAttr>("src_pipe");
                auto q = op.getAttrOfType<pto::PipeAttr>("dst_pipe");
                if (!p || !q || unsigned(p.getPipe()) != P || unsigned(q.getPipe()) != Q) {
                    valid = false;
                    return {};
                }
                events.push_back(
                    {op.getName().getStringRef() == "pto.logical_set" ? 1U : 2U, 0, 0, 0, get(op.getOperand(0))});
            } else if (!scalar(op)) {
                valid = false;
                return {};
            }
        }
        valid = false;
        return {};
    }
    void run(func::FuncOp function, uint64_t d, uint64_t v, uint64_t p)
    {
        values[function.getArgument(0)] = d;
        values[function.getArgument(1)] = v;
        values[function.getArgument(2)] = p;
        block(function.front());
    }
};
using Graph = std::vector<uint64_t>;
void edge(Graph& graph, unsigned a, unsigned b) { graph[a] |= uint64_t(1) << b; }
void close(Graph& graph)
{
    for (unsigned k = 0; k < graph.size(); ++k) {
        for (auto& row : graph) {
            if ((row >> k) & 1) {
                row |= graph[k];
            }
        }
    }
}
Graph native(ArrayRef<Event> payloads)
{
    Graph graph(payloads.size() * 2);
    std::map<unsigned, unsigned> previous;
    for (unsigned i = 0; i < payloads.size(); ++i) {
        edge(graph, 2 * i, 2 * i + 1);
        auto role = payloads[i].role;
        if (auto found = previous.find(role); found != previous.end()) {
            edge(graph, 2 * found->second, 2 * i);
            edge(graph, 2 * found->second + 1, 2 * i + 1);
        }
        previous[role] = i;
    }
    return graph;
}
bool oracle(const Execution& baseline, const Execution& actual, unsigned trips, uint64_t d, uint64_t v)
{
    auto fail = [](StringRef reason) {
        llvm::errs() << "bounded trace mismatch: " << reason << "\n";
        return false;
    };
    if (!baseline.valid || !actual.valid) {
        return fail("unsupported or invalid interpreted operation");
    }
    if (baseline.evaluations != actual.evaluations) {
        return fail("original opaque evaluations changed");
    }
    const auto& payloads = baseline.events;
    if (payloads.size() > 31) {
        return fail("oracle graph capacity");
    }
    auto expected = native(payloads), observed = native(payloads);
    std::map<std::pair<uint64_t, uint64_t>, unsigned> sources;
    for (unsigned i = 0; i < payloads.size(); ++i) {
        const auto& event = payloads[i];
        const auto ordinal = uint64_t((APInt(64, event.iv).getSExtValue() + 5) / 3);
        if (event.role == 0) {
            sources[{event.outer, ordinal}] = i;
            continue;
        }
        if (!ordinal) {
            continue;
        }
        const auto bit = event.outer * trips + ordinal - 1;
        auto source = sources.find({event.outer, ordinal - 1});
        if (source != sources.end() && ((d >> bit) & 1) && ((v >> bit) & 1)) {
            edge(expected, 2 * source->second + 1, 2 * i);
        }
    }
    std::map<uint64_t, unsigned> published;
    std::set<uint64_t> consumed;
    std::vector<unsigned> pending;
    unsigned cursor = 0, count = 0, last = 0;
    for (auto event : actual.events) {
        if (!event.kind) {
            if (cursor >= payloads.size()) {
                return fail("extra original payload");
            }
            const auto& original = payloads[cursor];
            if (std::tie(event.role, event.outer, event.iv) != std::tie(original.role, original.outer, original.iv)) {
                return fail("payload role or original visit coordinates changed");
            }
            if (event.role == 0) {
                ++count;
                last = cursor;
            } else {
                for (auto source : pending) {
                    edge(observed, 2 * source + 1, 2 * cursor);
                }
                pending.clear();
            }
            ++cursor;
        } else if (event.kind == 1) {
            if (!count || event.ordinal != count || !published.emplace(event.ordinal, last).second) {
                return fail("publication is duplicate or differs from executed source ordinal");
            }
        } else {
            auto source = published.find(event.ordinal);
            if (source == published.end() || !consumed.insert(event.ordinal).second) {
                return fail("acquisition is duplicate or lacks a publication");
            }
            pending.push_back(source->second);
        }
    }
    // A retained direction publishes every executed source, even when this
    // execution has no selected edge. Terminal cleanup consumes that suffix.
    const bool balanced = published.size() == count && consumed.size() == published.size();
    if (cursor != payloads.size() || !balanced) {
        llvm::errs() << "payloads=" << cursor << "/" << payloads.size() << " sources=" << count
                     << " publications=" << published.size() << " acquisitions=" << consumed.size() << "\n";
        return fail("missing payload or unmatched terminal suffix");
    }
    close(expected);
    close(observed);
    for (unsigned i = 0; i < observed.size(); ++i) {
        if ((observed[i] >> i) & 1) {
            return fail("emitted dependency cycle");
        }
        if (expected[i] != observed[i]) {
            llvm::errs() << "endpoint=" << i << " expected=" << expected[i] << " observed=" << observed[i] << "\n";
            return fail("payload closure differs");
        }
    }
    return true;
}
std::string render(Operation* op)
{
    std::string text;
    llvm::raw_string_ostream out(text);
    op->print(out);
    return text;
}
bool rejectsFutureDefinition(MLIRContext& context)
{
    Fixture f(context, 2, 1);
    f.predicate.getDefiningOp()->getParentOp()->moveAfter(f.target);
    auto owner = f.demands(true);
    owner->analysis.sourceDemands[0].target = 1;
    auto before = render(f.module->getOperation());
    fs::InvocationPrefixCollection collection(f.function);
    std::string error;
    return failed(fs::appendBoundedPrefixRealization(owner, collection, error)) &&
           before == render(f.module->getOperation());
}
bool runCase(MLIRContext& context, unsigned trips, unsigned invocations, unsigned& cases)
{
    Fixture f(context, trips, invocations);
    if (failed(verify(*f.module))) {
        return false;
    }
    auto owner = f.demands(true);
    auto before = render(f.module->getOperation());
    auto records = owner->analysis.sourceDemands;
    // Exercise a detached regional capability rather than directly calling the
    // bounded primitive. The source model identity survives wrapper destruction.
    auto input = std::make_shared<pto::SyncInput>(pto::GMAliasPolicy::MayNotAlias);
    auto retained = std::make_shared<fs::MathematicalResult>();
    retained->input = input; retained->boundedDemands = owner;
    auto state = std::make_shared<fs::SequenceAnalysisState>(f.function, owner->arena);
    fs::Child child;
    child.prefixMathematics = retained;
    child.anchors = owner->anchors;
    state->children.push_back(std::move(child));
    fs::RegionalAnalysis regional;
    regional.expressions = owner->arena;
    auto mismatched = regional;
    mismatched.expressions = std::make_shared<fs::RegionExpressions>();
    if (fs::sequencePrefixContribution(mismatched, state)) { return false; }
    // A sibling without retained provenance cannot borrow another child's
    // owner, even when its original operations belong to the same function.
    fs::Child unowned;
    unowned.anchors = owner->anchors;
    state->children.push_back(std::move(unowned));
    if (fs::sequencePrefixContribution(regional, state)) { return false; }
    auto foreignOwner = std::make_shared<fs::MathematicalResult>();
    foreignOwner->input = std::make_shared<pto::SyncInput>();
    state->children.back().prefixMathematics = foreignOwner;
    if (fs::sequencePrefixContribution(regional, state)) { return false; }
    state->children.pop_back();
    foreignOwner.reset();
    regional.invocationPrefix = fs::sequencePrefixContribution(regional, state);
    if (!regional.invocationPrefix) { return false; }
    std::weak_ptr<const pto::SyncInput> lifetime = input;
    state.reset(); retained.reset(); input.reset();
    if (lifetime.expired()) { return false; }
    std::string error;
    fs::InvocationPrefixCollection foreign(f.function, std::make_shared<pto::SyncInput>());
    if (succeeded(fs::collectRegionalInvocationPrefix(regional, foreign, error)) ||
        !foreign.recipe().sources.empty()) { return false; }
    auto changed = regional;
    changed.expressions = std::make_shared<fs::RegionExpressions>();
    fs::InvocationPrefixCollection wrongArena(f.function);
    if (succeeded(fs::collectRegionalInvocationPrefix(changed, wrongArena, error))) { return false; }
    auto guarded = fs::guardRegionalArm(regional, owner->expressions.boolean(true));
    if (guarded.invocationPrefix) { return false; }
    fs::invalidateDerivedRegionalCertificates(changed);
    if (changed.invocationPrefix) { return false; }
    fs::InvocationPrefixCollection collection(f.function);
    if (failed(fs::collectRegionalInvocationPrefix(regional, collection, error))) {
        llvm::errs() << "bounded regional contribution: " << error << "\n";
        return false;
    }
    auto& recipe = collection.recipe();
    recipe.privateScalarControl = true;
    recipe.planId = 23;
    recipe.sourceBounds = {{P, trips * invocations}, {Q, trips * invocations}};
    fs::PrivateStateError diagnostic;
    auto prepared = fs::materializeInvocationPrefix(f.function, recipe, diagnostic);
    if (failed(prepared)) {
        llvm::errs() << diagnostic.message << "\n";
        return false;
    }
    if (!recipe.expressionProfile) { return false; }
    const auto profile = recipe.expressionProfile->report();
    const int64_t expected = trips ? 1 : 0;
    if (profile.getInteger("requests") != expected || profile.getInteger("emissions") != expected ||
        (expected && profile.getInteger("constructed_operations").value_or(0) <= 0) ||
        profile.getInteger("constructed_operations") != profile.getInteger("retained_operations")) {
        llvm::errs() << "bounded profile trips=" << trips << " " << profile.getInteger("requests").value_or(-1) << "\n";
        return false;
    }
    const auto masks = uint64_t(1) << (trips * invocations);
    for (uint64_t d = 0; d < masks; ++d) {
        for (uint64_t v = 0; v < masks; ++v) {
            for (uint64_t p = 0; p < masks; ++p) {
                Execution baseline(trips), actual(trips);
                baseline.run(f.function, d, v, p);
                actual.run((*prepared)->function, d, v, p);
                if (!oracle(baseline, actual, trips, d, v)) {
                    llvm::errs() << "bounded oracle trips=" << trips << " invocations=" << invocations << " masks=" << d
                                 << "," << v << "," << p << "\n";
                    return false;
                }
                ++cases;
            }
        }
    }
    const bool intact = before == render(f.module->getOperation()) && owner->analysis.sourceDemands.size() == 1 &&
                        owner->analysis.sourceDemands[0].guard == records[0].guard;
    if (!intact) {
        return false;
    }
    if (trips > 1) {
        fs::InvocationPrefixCollection rejected(f.function);
        auto unmasked = f.demands(false);
        if (succeeded(fs::appendBoundedPrefixRealization(unmasked, rejected, error))) {
            return false;
        }
    }
    return before == render(f.module->getOperation());
}
} // namespace
bool runBoundedPrefixRealizationChecks()
{
    MLIRContext context;
    context.disableMultithreading();
    context.loadDialect<func::FuncDialect, scf::SCFDialect, arith::ArithDialect, pto::PTODialect>();
    if (!rejectsFutureDefinition(context)) {
        return false;
    }
    unsigned cases = 0;
    for (unsigned trips = 0; trips <= 3; ++trips) {
        if (!runCase(context, trips, 1, cases)) {
            return false;
        }
    }
    for (unsigned trips : {1U, 2U}) {
        if (!runCase(context, trips, 2, cases)) {
            return false;
        }
    }
    llvm::outs() << "bounded prefix realization checks passed executions=" << cases << "\n";
    return true;
}
