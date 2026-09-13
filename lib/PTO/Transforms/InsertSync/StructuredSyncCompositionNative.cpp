// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/InsertSync/StructuredSyncPlan.h"
#include "PTO/Transforms/InsertSync/StructuredSyncComposition.h"
#include "PTO/Transforms/InsertSync/StructuredSyncCoverage.h"
#include "PTO/Transforms/InsertSync/StructuredSyncPeriodicScalar.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncPayloadSnapshot.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <map>
#include <set>

using namespace mlir;
using namespace mlir::pto;
namespace ss = mlir::pto::structured_sync;
namespace c = ss::composition;
using Outcome = logical_sync::ConstructionResult;
namespace {
std::optional<unsigned> lane(PIPE pipe)
{
    switch (pipe) {
        case PIPE::PIPE_S:
            return unsigned(ss::Pipe::S);
        case PIPE::PIPE_V:
            return unsigned(ss::Pipe::V);
        case PIPE::PIPE_M:
            return unsigned(ss::Pipe::M);
        case PIPE::PIPE_MTE1:
            return unsigned(ss::Pipe::MTE1);
        case PIPE::PIPE_MTE2:
            return unsigned(ss::Pipe::MTE2);
        case PIPE::PIPE_MTE3:
            return unsigned(ss::Pipe::MTE3);
        case PIPE::PIPE_FIX:
            return unsigned(ss::Pipe::FIX);
        default:
            return {};
    }
}
PIPE pipe(unsigned id)
{
    static constexpr PIPE pipes[] = {PIPE::PIPE_S,    PIPE::PIPE_V,    PIPE::PIPE_M,  PIPE::PIPE_MTE1,
                                     PIPE::PIPE_MTE2, PIPE::PIPE_MTE3, PIPE::PIPE_FIX};
    return pipes[id];
}
bool sync(Operation* op) { return isa<SetFlagOp, WaitFlagOp, BarrierOp>(op); }

// Stage the immutable symbol closure as well as the kernel. Physical helper
// contracts and pure bodies are resolved by the translator/importer through
// ordinary symbol lookup, including during fresh reconstruction. Only the
// working kernel body is ever committed back to the original module.
bool stageDependencies(Operation* root, llvm::DenseMap<Operation*, Operation*>& copies, std::string& reason)
{
    std::function<Operation*(Operation*)> scope = [&](Operation* original) -> Operation* {
        auto found = copies.find(original);
        if (found != copies.end())
            return found->second;
        auto module = dyn_cast_or_null<ModuleOp>(original);
        if (!module)
            return nullptr;
        Operation* parent = scope(module->getParentOp());
        if (!parent || !isa<ModuleOp>(parent))
            return nullptr;
        auto clone = ModuleOp::create(module.getLoc());
        clone->setAttrs(module->getAttrs());
        cast<ModuleOp>(parent).getBody()->push_back(clone);
        copies[original] = clone;
        return clone;
    };
    std::vector<Operation*> pending{root};
    while (!pending.empty()) {
        Operation* original = pending.back();
        pending.pop_back();
        auto uses = SymbolTable::getSymbolUses(original);
        if (!uses) {
            reason = "cannot enumerate staged helper symbol dependencies";
            return false;
        }
        for (const auto& use : *uses) {
            Operation* symbol = SymbolTable::lookupNearestSymbolFrom(use.getUser(), use.getSymbolRef());
            if (!symbol) {
                reason = "unresolved staged helper symbol dependency";
                return false;
            }
            if (copies.count(symbol))
                continue;
            // Symbols defined inside an already cloned definition travelled
            // with it; do not clone those a second time into an outer scope.
            bool contained = false;
            for (Operation* ancestor = symbol->getParentOp(); ancestor; ancestor = ancestor->getParentOp())
                if (copies.count(ancestor) && !isa<ModuleOp>(ancestor)) {
                    contained = true;
                    break;
                }
            if (contained)
                continue;
            Operation* parent = scope(symbol->getParentOp());
            if (!parent || !isa<ModuleOp>(parent)) {
                reason = "unsupported staged helper symbol scope";
                return false;
            }
            Operation* clone = symbol->clone();
            cast<ModuleOp>(parent).getBody()->push_back(clone);
            copies[symbol] = clone;
            pending.push_back(symbol);
        }
    }
    return true;
}

struct Inventory {
    func::FuncOp function;
    MemoryDependentAnalyzer memory;
    SyncIRs ir;
    Buffer2MemInfoMap buffers;
    SyncPhysicalFacts physical;
    std::string reason;
    explicit Inventory(func::FuncOp f) : function(f) {}
    bool build(bool actual = false)
    {
        PTOIRTranslator translator(ir, memory, buffers, function, SyncAnalysisMode::NORMALSYNC, true);
        if (failed(translator.Build())) {
            reason = "physical translation failed";
            return false;
        }
        physical = importCoverageStructuredSyncPhysicalFacts(function, ir, true, actual);
        reason = physical.reason;
        return physical.status == SyncPhysicalFacts::Status::Complete;
    }
};

struct Cell {
    AddressSpace space;
    uint64_t lower = 0, upper = 0;
    bool whole = false;
    SmallVector<Value> gmRoots;
};
struct Tree {
    Inventory& inventory;
    c::Program program;
    std::vector<Cell> cells;
    std::vector<Operation*> anchors;
    llvm::DenseMap<Operation*, unsigned> ids;
    const llvm::SmallPtrSetImpl<Operation*>* ignored = nullptr;
    llvm::DenseMap<Operation*, const CompoundInstanceElement*> phases;
    std::string reason;
    uint64_t widenedSpaces = 0;
    uint64_t periodicScalarWork = 0;
    uint64_t periodicDagVisits = 0, periodicEvaluations = 0;
    InsertSyncGMAliasMode gm;
    llvm::DenseMap<const BaseMemInfo*, InsertSyncGMRoots> gmAccesses;
    std::set<std::pair<unsigned, unsigned>> disjointPairs;

    Tree(Inventory& i, ss::HardwareContract hardware, InsertSyncGMAliasMode alias) : inventory(i), gm(alias)
    {
        program.core = i.physical.cube ? ss::Core::AIC : ss::Core::AIV;
        program.target.hardware = hardware;
    }
    bool gmContracts()
    {
        if (auto attribute = inventory.function->getAttr("pto.noalias_pairs")) {
            auto pairs = dyn_cast<DenseI64ArrayAttr>(attribute);
            if (!pairs || pairs.size() % 2) {
                reason = "invalid pairwise GM alias contract";
                return false;
            }
            for (unsigned i = 0; i < pairs.size(); i += 2) {
                auto a = pairs[i], b = pairs[i + 1];
                if (a < 0 || b < 0 || a == b || uint64_t(a) >= inventory.function.getNumArguments() ||
                    uint64_t(b) >= inventory.function.getNumArguments() ||
                    !isa<PtrType, TensorViewType, PartitionTensorViewType>(
                        inventory.function.getArgument(a).getType()) ||
                    !isa<PtrType, TensorViewType, PartitionTensorViewType>(
                        inventory.function.getArgument(b).getType())) {
                    reason = "invalid pairwise GM alias contract";
                    return false;
                }
                disjointPairs.emplace(std::min(a, b), std::max(a, b));
            }
        }
        return true;
    }
    std::vector<Cell> gmCells()
    {
        SmallVector<Value> roots;
        bool unknown = false;
        for (auto* p : inventory.physical.phases) {
            auto collect = [&](const auto& entries) {
                for (auto* a : entries)
                    if (a->scope == AddressSpace::GM) {
                        if (gmAccesses.count(a))
                            continue;
                        auto traced = a->aliasesUnknownRange ?
                                          InsertSyncGMRoots{} :
                                          traceInsertSyncGMRoots(inventory.function, a->rootBuffer);
                        unknown |= !traced.complete;
                        for (Value root : traced.arguments)
                            if (roots.size() <= c::MaxCells && !llvm::is_contained(roots, root))
                                roots.push_back(root);
                        gmAccesses[a] = std::move(traced);
                    }
            };
            collect(p->useVec);
            collect(p->defVec);
        }
        if (unknown || roots.size() > c::MaxCells || roots.empty())
            return {{AddressSpace::GM, 0, 0, true, {}}};
        std::vector<unsigned> parent(roots.size());
        for (unsigned i = 0; i < roots.size(); ++i)
            parent[i] = i;
        auto representative = [&](unsigned i) {
            while (parent[i] != i) {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            return i;
        };
        // Merge possibly aliasing roots; transitive merging only loses precision.
        // The bounded root table prevents an access-pair dependence campaign.
        for (unsigned i = 0; i < roots.size(); ++i)
            for (unsigned j = i + 1; j < roots.size(); ++j) {
                unsigned a = cast<BlockArgument>(roots[i]).getArgNumber();
                unsigned b = cast<BlockArgument>(roots[j]).getArgNumber();
                bool disjoint = gm == InsertSyncGMAliasMode::DisjointArguments ||
                                disjointPairs.count({std::min(a, b), std::max(a, b)});
                if (!disjoint)
                    parent[representative(j)] = representative(i);
            }
        std::map<unsigned, Cell> groups;
        for (unsigned i = 0; i < roots.size(); ++i) {
            auto& cell = groups[representative(i)];
            cell.space = AddressSpace::GM;
            cell.whole = true;
            cell.gmRoots.push_back(roots[i]);
        }
        std::vector<Cell> result;
        for (auto& [id, cell] : groups)
            result.push_back(std::move(cell));
        return result;
    }
    void partition()
    {
        auto globalCells = gmCells();
        std::map<AddressSpace, std::set<uint64_t>> cuts;
        std::set<AddressSpace> coarse;
        for (auto* p : inventory.physical.phases) {
            phases[p->elementOp] = p;
            auto collect = [&](const auto& accesses) {
                for (auto* a : accesses) {
                    auto& bounds = cuts[a->scope];
                    if (coarse.count(a->scope))
                        continue;
                    if (a->scope == AddressSpace::GM || a->aliasesUnknownRange || !a->hasKnownPhysicalAddresses ||
                        !a->allocateSize || a->baseAddresses.empty() || a->baseAddresses.size() > c::MaxCells) {
                        coarse.insert(a->scope);
                        bounds.clear();
                        continue;
                    }
                    for (uint64_t base : a->baseAddresses) {
                        if (base > UINT64_MAX - a->allocateSize) {
                            coarse.insert(a->scope);
                            break;
                        }
                        bounds.insert(base);
                        bounds.insert(base + a->allocateSize);
                        if (bounds.size() > c::MaxCells) {
                            coarse.insert(a->scope);
                            break;
                        }
                    }
                    if (coarse.count(a->scope))
                        bounds.clear();
                }
            };
            collect(p->useVec);
            collect(p->defVec);
        }
        unsigned total = 0;
        for (const auto& [space, bounds] : cuts)
            total += space == AddressSpace::GM ? globalCells.size() : coarse.count(space) ? 1 : bounds.size() - 1;
        if (total > c::MaxCells) {
            for (const auto& [space, bounds] : cuts)
                coarse.insert(space);
            globalCells = {{AddressSpace::GM, 0, 0, true, {}}};
        }
        for (const auto& [space, bounds] : cuts) {
            if (space == AddressSpace::GM) {
                for (auto& cell : globalCells)
                    cells.push_back(std::move(cell));
                continue;
            }
            if (coarse.count(space)) {
                cells.push_back({space, 0, 0, true});
                ++widenedSpaces;
            } else {
                for (auto it = bounds.begin(); it != bounds.end(); ++it) {
                    auto next = std::next(it);
                    if (next != bounds.end())
                        cells.push_back({space, *it, *next, false});
                }
            }
        }
        // A scalar-only qualified function still has a well-formed empty state.
        if (cells.empty())
            cells.push_back({AddressSpace::GM, 0, 0, true});
        program.cells = cells.size();
        for (const auto& cell : cells)
            program.globalMemory.push_back(cell.space == AddressSpace::GM);
    }
    unsigned add(c::Node n, Operation* anchor = nullptr)
    {
        unsigned id = program.nodes.size();
        program.nodes.push_back(std::move(n));
        anchors.push_back(anchor);
        if (anchor)
            ids[anchor] = id;
        return id;
    }
    c::Node node(c::Node::Kind kind)
    {
        c::Node n;
        n.kind = kind;
        n.effects.resize(program.cells);
        return n;
    }
    bool region(Region& body, unsigned& id)
    {
        auto sequence = node(c::Node::Sequence);
        if (body.empty()) {
            id = add(std::move(sequence));
            return true;
        }
        if (!llvm::hasSingleElement(body)) {
            reason = "unsupported multi-block region";
            return false;
        }
        for (Operation& op : body.front()) {
            if (sync(&op) || (ignored && ignored->contains(&op)))
                continue;
            c::Node current = node(c::Node::Sequence);
            if (auto loop = dyn_cast<scf::ForOp>(op)) {
                if (!isInsertSyncScalarPrerequisite(loop.getLowerBound()) ||
                    !isInsertSyncScalarPrerequisite(loop.getUpperBound()) ||
                    !isInsertSyncScalarPrerequisite(loop.getStep())) {
                    reason = "unsupported asynchronous for control prerequisite";
                    return false;
                }
                current.kind = c::Node::For;
            } else if (auto loop = dyn_cast<scf::WhileOp>(op)) {
                if (!isInsertSyncScalarPrerequisite(loop.getConditionOp().getCondition())) {
                    reason = "unsupported asynchronous while condition";
                    return false;
                }
                current.kind = c::Node::While;
            } else if (auto choice = dyn_cast<scf::IfOp>(op)) {
                if (!isInsertSyncScalarPrerequisite(choice.getCondition())) {
                    reason = "unsupported asynchronous branch condition";
                    return false;
                }
                current.kind = c::Node::Choice;
            }
            for (Region& child : op.getRegions()) {
                unsigned childId;
                if (!region(child, childId))
                    return false;
                current.children.push_back(childId);
            }
            auto found = phases.find(&op);
            if (found != phases.end()) {
                auto source = lane(static_cast<PIPE>(found->second->kPipeValue));
                if (!source) {
                    reason = "unsupported physical lane";
                    return false;
                }
                current.kind = c::Node::Operation;
                current.lane = *source;
                auto access = [&](const auto& entries, bool write) {
                    for (auto* a : entries)
                        for (unsigned j = 0; j < cells.size(); ++j) {
                            const auto& cell = cells[j];
                            if (cell.space != a->scope)
                                continue;
                            bool overlaps = cell.whole;
                            if (cell.space == AddressSpace::GM && !cell.gmRoots.empty()) {
                                const auto& roots = gmAccesses.find(a)->second;
                                overlaps = !roots.complete || llvm::any_of(roots.arguments, [&](Value root) {
                                    return llvm::is_contained(cell.gmRoots, root);
                                });
                            }
                            if (!overlaps)
                                for (uint64_t base : a->baseAddresses)
                                    overlaps |= base < cell.upper && cell.lower < base + a->allocateSize;
                            if (!overlaps)
                                continue;
                            if (write)
                                current.effects[j].writers |= 1u << *source;
                            else
                                current.effects[j].readers |= 1u << *source;
                            // ACC read/read resource exclusion is stronger than ordinary RAW.
                            // The baseline uses full completion, never upgrades typed MMAD
                            // ordering evidence into operand release or visibility.
                            if (a->scope == AddressSpace::ACC)
                                current.effects[j].writers |= 1u << *source;
                        }
                };
                access(found->second->useVec, false);
                access(found->second->defVec, true);
            }
            sequence.children.push_back(add(std::move(current), &op));
        }
        id = add(std::move(sequence));
        return true;
    }
    bool build(bool periodicPrecision)
    {
        if (!gmContracts())
            return false;
        partition();
        unsigned root;
        auto* scope = inventory.physical.lifetimeScope;
        if (!region(
                scope == inventory.function.getOperation() ? inventory.function.getBody() : scope->getRegion(0), root))
            return false;
        // This optional guard contract is deliberately narrower than For
        // admission. Other steps/bounds still use conservative composition.
        // Publication remains in the loop's own parent Sequence. Choose its
        // earliest cut dominated by both original bounds; demand analysis then
        // moves it after the last required producer, not after unrelated loads.
        DominanceInfo dominance(inventory.function);
        for (unsigned parent = 0; parent < program.nodes.size(); ++parent) {
            const auto& children = program.nodes[parent].children;
            if (program.nodes[parent].kind != c::Node::Sequence)
                continue;
            for (unsigned index = 0; index < children.size(); ++index) {
                unsigned id = children[index];
                if (program.nodes[id].kind != c::Node::For || index + 1 == children.size() ||
                    program.nodes[id].children.size() != 1)
                    continue;
                auto loop = cast<scf::ForOp>(anchors[id]);
                unsigned body = program.nodes[id].children[0];
                if (program.nodes[body].kind != c::Node::Sequence || program.nodes[body].children.empty() ||
                    program.nodes[body].children.front() >= anchors.size())
                    continue;
                APInt step;
                if (!loop.getInductionVar().getType().isIndex() ||
                    !matchPattern(loop.getStep(), m_ConstantInt(&step)) || step != 1)
                    continue;
                for (unsigned cut = 0; cut <= index; ++cut) {
                    Operation* anchor = anchors[children[cut]];
                    if (anchor && dominance.properlyDominates(loop.getLowerBound(), anchor) &&
                        dominance.properlyDominates(loop.getUpperBound(), anchor)) {
                        program.nodes[id].entryGuardStart = children[cut];
                        break;
                    }
                }
            }
        }
        if (periodicPrecision)
            qualifyPeriodicWords();
        return true;
    }

    void qualifyPeriodicWords()
    {
        // Optional metadata only. One bounded scalar DAG and at most 32
        // ordinal evaluations per owner, never a Boolean path product.
        uint64_t& work = periodicScalarWork;
        constexpr uint64_t MaxWork = 1u << 20;
        for (unsigned owner = 0; owner < program.nodes.size(); ++owner) {
            if (work >= MaxWork)
                break;
            const auto& n = program.nodes[owner];
            if (n.kind != c::Node::For || n.entryGuardStart == ~0u)
                continue;
            auto loop = cast<scf::ForOp>(anchors[owner]);
            APInt lower;
            if (loop->hasAttr("unsignedCmp") || !matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
                !lower.isSignedIntN(64) || lower.isNegative())
                continue;
            ss::LoopOrdinal induction{lower.getSExtValue(), 1};
            ss::PeriodicScalar scalar(loop.getInductionVar(), induction);
            std::vector<unsigned> nodes, choices;
            bool qualified = true;
            std::function<void(unsigned, unsigned)> collect = [&](unsigned id, unsigned nesting) {
                if (!qualified || work >= MaxWork || nesting >= 64) {
                    qualified = false;
                    return;
                }
                ++work;
                const auto& child = program.nodes[id];
                if (child.kind == c::Node::For || child.kind == c::Node::While) {
                    qualified = false;
                    return;
                }
                nodes.push_back(id);
                if (child.kind == c::Node::Choice)
                    choices.push_back(id);
                for (unsigned next : child.children)
                    collect(next, nesting + 1);
            };
            collect(n.children[0], 0);
            if (!qualified || choices.empty())
                continue;
            llvm::DenseMap<Value, unsigned> depths;
            std::function<std::optional<unsigned>(Value, unsigned)> depth =
                [&](Value value, unsigned nesting) -> std::optional<unsigned> {
                if (work >= MaxWork || nesting >= 64 || depths.size() >= 256)
                    return {};
                ++work;
                ++periodicDagVisits;
                if (value == loop.getInductionVar())
                    return 0;
                auto found = depths.find(value);
                if (found != depths.end())
                    return found->second;
                auto* op = value.getDefiningOp();
                if (!op)
                    return {};
                // Reserve before descending so a deep expression is refused
                // before PeriodicScalar's recursive matcher is invoked.
                depths[value] = 64;
                unsigned result = 0;
                for (Value operand : op->getOperands()) {
                    auto nested = depth(operand, nesting + 1);
                    if (!nested || *nested >= 63)
                        return {};
                    result = std::max(result, *nested + 1);
                }
                depths[value] = result;
                return result;
            };
            uint64_t period = 1;
            for (unsigned id : choices) {
                Value condition = cast<scf::IfOp>(anchors[id]).getCondition();
                if (!depth(condition, 0)) {
                    qualified = false;
                    break;
                }
                auto p = scalar.period(condition);
                if (!p || !*p || *p > 32) {
                    qualified = false;
                    break;
                }
                period = period / std::gcd(period, *p) * *p;
                if (period > 32) {
                    qualified = false;
                    break;
                }
            }
            uint64_t evaluation = nodes.size() + choices.size() * depths.size();
            if (!qualified || evaluation > (MaxWork - std::min(work, MaxWork)) / period)
                continue;
            work += evaluation * period;
            std::map<unsigned, uint32_t> truth;
            for (unsigned id : choices) {
                Value condition = cast<scf::IfOp>(anchors[id]).getCondition();
                for (unsigned ordinal = 0; ordinal < period; ++ordinal) {
                    ++periodicEvaluations;
                    auto value = scalar.evaluate(condition, ordinal);
                    if (!value) {
                        qualified = false;
                        break;
                    }
                    if (*value)
                        truth[id] |= uint32_t(1) << ordinal;
                }
                if (!qualified)
                    break;
            }
            if (!qualified)
                continue;
            uint32_t all = period == 32 ? UINT32_MAX : (uint32_t(1) << period) - 1;
            std::function<void(unsigned, uint32_t)> record = [&](unsigned id, uint32_t active) {
                auto& child = program.nodes[id];
                if (child.kind == c::Node::Sequence || child.kind == c::Node::Choice) {
                    child.periodicOwner = owner;
                    child.periodicPeriod = unsigned(period);
                    child.periodicLower = induction.lower;
                    child.periodicResidues = child.kind == c::Node::Choice ? truth[id] : active;
                }
                if (child.kind == c::Node::Choice) {
                    record(child.children[0], active & truth[id]);
                    record(child.children[1], active & ~truth[id]);
                } else
                    for (unsigned next : child.children)
                        record(next, active);
            };
            record(n.children[0], all);
        }
    }
};

void emit(Tree& tree, const c::Result& plan)
{
    auto* context = tree.inventory.function.getContext();
    for (unsigned id = 0; id < plan.before.size(); ++id) {
        if (plan.before[id].empty())
            continue;
        OpBuilder b(tree.anchors[id]);
        auto loc = tree.anchors[id]->getLoc();
        for (unsigned index = 0; index < plan.before[id].size(); ++index) {
            const auto& m = plan.before[id][index];
            if (m.participation != c::Mechanism::Every) {
                auto loop = cast<scf::ForOp>(tree.anchors[m.loop]);
                bool first = m.participation == c::Mechanism::First;
                bool previous = m.participation == c::Mechanism::Previous;
                Value boundary = loop.getLowerBound();
                if (m.word != ~0u)
                    boundary = b.create<arith::ConstantIndexOp>(loc, *tree.program.nodes[m.word].firstActive());
                auto condition = b.create<arith::CmpIOp>(
                    loc,
                    first    ? arith::CmpIPredicate::eq :
                    previous ? arith::CmpIPredicate::ne :
                               arith::CmpIPredicate::slt,
                    first || previous ? loop.getInductionVar() : boundary,
                    first || previous ? boundary : loop.getUpperBound());
                auto branch = b.create<scf::IfOp>(loc, condition, false);
                OpBuilder nested = OpBuilder::atBlockBegin(&branch.getThenRegion().front());
                auto emitEvent = [&](const c::Mechanism& command) {
                    auto a = PipeAttr::get(context, pipe(command.first)),
                         d = PipeAttr::get(context, pipe(command.second));
                    auto key = EventAttr::get(context, static_cast<EVENT>(command.forwardKey));
                    if (command.kind == c::Mechanism::Publish)
                        nested.create<SetFlagOp>(loc, a, d, key);
                    else
                        nested.create<WaitFlagOp>(loc, a, d, key);
                };
                emitEvent(m);
                if (!first && m.kind == c::Mechanism::Publish && index + 1 < plan.before[id].size()) {
                    const auto& next = plan.before[id][index + 1];
                    if (next.kind == c::Mechanism::Acquire && next.participation == m.participation &&
                        next.loop == m.loop && next.first == m.first && next.second == m.second &&
                        next.forwardKey == m.forwardKey) {
                        emitEvent(next);
                        ++index;
                    }
                }
                continue;
            }
            if (m.kind == c::Mechanism::Barrier) {
                b.create<BarrierOp>(loc, PipeAttr::get(context, pipe(m.first)));
                continue;
            }
            auto first = PipeAttr::get(context, pipe(m.first));
            auto second = PipeAttr::get(context, pipe(m.second));
            auto forward = EventAttr::get(context, static_cast<EVENT>(m.forwardKey));
            auto reverse = EventAttr::get(context, static_cast<EVENT>(m.reverseKey));
            if (m.kind == c::Mechanism::Publish) {
                b.create<SetFlagOp>(loc, first, second, forward);
                continue;
            }
            if (m.kind == c::Mechanism::Acquire) {
                b.create<WaitFlagOp>(loc, first, second, forward);
                continue;
            }
            b.create<SetFlagOp>(loc, first, second, forward);
            b.create<WaitFlagOp>(loc, first, second, forward);
            b.create<SetFlagOp>(loc, second, first, reverse);
            b.create<WaitFlagOp>(loc, second, first, reverse);
        }
    }
}

// Recover guarded participation from actual IR and the immutable original
// loop/bound identities. No selected entry-demand or completion receipt enters
// this classifier. Only its exactly checked added operations pass the snapshot.
struct ParticipationGuards {
    llvm::SmallPtrSet<Operation*, 32> operations;
    llvm::DenseMap<Operation*, std::vector<c::Mechanism>> commands;
    llvm::DenseMap<Operation*, Operation*> targets;
    bool build(Tree& original, const llvm::SmallPtrSetImpl<Operation*>& generated, std::string& reason)
    {
        using Key = std::tuple<unsigned, unsigned, unsigned>;
        std::map<Key, unsigned> firstLoops, previousLoops, previousWords;
        std::vector<scf::IfOp> candidates;
        std::vector<unsigned> parent(original.program.nodes.size(), ~0u), position(parent.size());
        for (unsigned id = 0; id < original.program.nodes.size(); ++id)
            for (unsigned i = 0; i < original.program.nodes[id].children.size(); ++i) {
                auto child = original.program.nodes[id].children[i];
                parent[child] = id;
                position[child] = i;
            }
        auto fail = [&]() {
            reason = "invalid native participation guard or original loop binding";
            return false;
        };
        auto exactBoundary = [&](Value value, unsigned word, Operation* cmp) {
            if (word >= original.program.nodes.size())
                return false;
            auto first = original.program.nodes[word].firstActive();
            auto constant = value.getDefiningOp<arith::ConstantIndexOp>();
            if (!first || !constant || constant.value() != *first || !generated.contains(constant) ||
                !constant.getResult().hasOneUse() || constant->getNextNode() != cmp)
                return false;
            operations.insert(constant);
            return true;
        };
        auto nextOriginal = [&](Operation* from) {
            while (from && !original.ids.count(from))
                from = from->getNextNode();
            return from;
        };
        auto raw = [&](Operation* op, c::Mechanism& m) {
            auto decode = [&](auto event, c::Mechanism::Kind kind) {
                auto a = lane(event.getSrcPipe().getPipe()), b = lane(event.getDstPipe().getPipe());
                if (!a || !b)
                    return false;
                m = {kind, *a, *b, unsigned(event.getEventId().getEvent()), 0};
                return true;
            };
            if (auto set = dyn_cast<SetFlagOp>(op))
                return decode(set, c::Mechanism::Publish);
            if (auto wait = dyn_cast<WaitFlagOp>(op))
                return decode(wait, c::Mechanism::Acquire);
            return false;
        };
        original.inventory.function.walk([&](scf::IfOp branch) {
            if (branch.getNumResults() || !branch.getElseRegion().empty() ||
                !llvm::hasSingleElement(branch.getThenRegion()))
                return;
            bool event = false, pure = true;
            for (Operation& op : branch.getThenRegion().front()) {
                event |= isa<SetFlagOp, WaitFlagOp>(op);
                pure &= isa<SetFlagOp, WaitFlagOp, scf::YieldOp>(op);
            }
            if (event && pure)
                candidates.push_back(branch);
        });
        for (auto branch : candidates) {
            if (!generated.contains(branch))
                return fail();
            auto cmp = branch.getCondition().getDefiningOp<arith::CmpIOp>();
            if (!cmp || cmp->getNextNode() != branch.getOperation() || !cmp.getResult().hasOneUse())
                return fail();
            auto& list = commands[branch];
            for (Operation& op : branch.getThenRegion().front().without_terminator()) {
                c::Mechanism m;
                if (!raw(&op, m))
                    return fail();
                list.push_back(m);
            }
            targets[branch] = nextOriginal(branch->getNextNode());
            if (!targets[branch])
                return fail();
            bool previous = cmp.getPredicate() == arith::CmpIPredicate::ne;
            if (cmp.getPredicate() != arith::CmpIPredicate::eq && !previous)
                continue;
            auto iv = dyn_cast<BlockArgument>(cmp.getLhs());
            auto loop = iv ? dyn_cast_or_null<scf::ForOp>(iv.getOwner()->getParentOp()) : scf::ForOp{};
            auto found = loop ? original.ids.find(loop) : original.ids.end();
            if (!loop || found == original.ids.end() || original.program.nodes[found->second].entryGuardStart == ~0u ||
                cmp.getLhs() != loop.getInductionVar() || list.size() != 1 || list[0].kind != c::Mechanism::Acquire)
                return fail();
            auto& m = list[0];
            if (branch->getBlock() == loop.getBody()) {
                if (cmp.getRhs() != loop.getLowerBound())
                    return fail();
            } else if (!previous) {
                // A shared incoming episode may acquire at mutually exclusive
                // first consumers under original choices. Bind only the exact
                // original owner predicate here; the core independently checks
                // the complete actual wait population and path cardinality.
                unsigned cut = original.ids.lookup(targets[branch]);
                if (cmp.getRhs() != loop.getLowerBound() || original.program.nodes[cut].kind != c::Node::Operation)
                    return fail();
                unsigned ancestor = parent[cut];
                unsigned body = original.program.nodes[found->second].children[0];
                bool choice = false;
                while (ancestor != body && ancestor != ~0u) {
                    auto kind = original.program.nodes[ancestor].kind;
                    if (kind != c::Node::Sequence && kind != c::Node::Choice)
                        return fail();
                    choice |= kind == c::Node::Choice;
                    ancestor = parent[ancestor];
                }
                if (ancestor != body || !choice)
                    return fail();
            } else {
                unsigned cut = original.ids.lookup(targets[branch]);
                unsigned word = parent[cut];
                if (!previous || word == ~0u || original.program.nodes[word].periodicOwner != found->second ||
                    !exactBoundary(cmp.getRhs(), word, cmp))
                    return fail();
                m.word = word;
            }
            m.participation = previous ? c::Mechanism::Previous : c::Mechanism::First;
            m.loop = found->second;
            auto& loops = previous ? previousLoops : firstLoops;
            auto binding = loops.emplace(Key{m.first, m.second, m.forwardKey}, m.loop);
            if (!binding.second && (previous || binding.first->second != m.loop))
                return fail();
            if (previous)
                previousWords[{m.first, m.second, m.forwardKey}] = m.word;
        }
        for (auto branch : candidates) {
            auto cmp = branch.getCondition().getDefiningOp<arith::CmpIOp>();
            auto& list = commands[branch];
            if (list[0].participation != c::Mechanism::First && list[0].participation != c::Mechanism::Previous) {
                if (cmp.getPredicate() != arith::CmpIPredicate::slt)
                    return fail();
                unsigned loopId = ~0u;
                bool loopExit = list.size() == 1 && list[0].kind == c::Mechanism::Acquire;
                if (loopExit) {
                    auto found = previousLoops.find({list[0].first, list[0].second, list[0].forwardKey});
                    if (found == previousLoops.end())
                        return fail();
                    loopId = found->second;
                    unsigned cut = original.ids.lookup(targets[branch]);
                    if (parent[cut] == ~0u || !position[cut] ||
                        original.program.nodes[parent[cut]].children[position[cut] - 1] != loopId)
                        return fail();
                } else if (list.size() == 1 && list[0].kind == c::Mechanism::Publish) {
                    auto found = firstLoops.find({list[0].first, list[0].second, list[0].forwardKey});
                    if (found == firstLoops.end())
                        return fail();
                    loopId = found->second;
                } else if (
                    list.size() == 2 && list[0].kind == c::Mechanism::Publish &&
                    list[1].kind == c::Mechanism::Acquire && list[0].first == list[1].first &&
                    list[0].second == list[1].second && list[0].forwardKey == list[1].forwardKey) {
                    unsigned cut = original.ids.lookup(targets[branch]);
                    if (parent[cut] == ~0u || !position[cut])
                        return fail();
                    loopId = original.program.nodes[parent[cut]].children[position[cut] - 1];
                } else
                    return fail();
                if (original.program.nodes[loopId].kind != c::Node::For ||
                    original.program.nodes[loopId].entryGuardStart == ~0u)
                    return fail();
                auto loop = cast<scf::ForOp>(original.anchors[loopId]);
                unsigned word = loopExit ? previousWords.at({list[0].first, list[0].second, list[0].forwardKey}) : ~0u;
                if ((word == ~0u ? cmp.getLhs() != loop.getLowerBound() : !exactBoundary(cmp.getLhs(), word, cmp)) ||
                    cmp.getRhs() != loop.getUpperBound() || branch->getBlock() != loop->getBlock())
                    return fail();
                for (auto& m : list) {
                    m.participation = loopExit ? c::Mechanism::LoopExit : c::Mechanism::NonEmpty;
                    m.loop = loopId;
                    m.word = word;
                }
            }
            operations.insert(branch);
            operations.insert(cmp);
            for (Operation& op : branch.getThenRegion().front())
                operations.insert(&op);
        }
        return true;
    }
};

// Recognize the actual reusable protocol, with no attributes or selected-plan
// receipt. All four commands must be adjacent in the same original block.
bool parsePacket(Operation*& cursor, c::Mechanism& m, const c::Program& program, bool precision)
{
    if (auto barrier = dyn_cast<BarrierOp>(cursor)) {
        auto p = lane(barrier.getPipe().getPipe());
        if (!p)
            return false;
        m = {c::Mechanism::Barrier, *p, *p};
        cursor = cursor->getNextNode();
        return true;
    }
    auto independent = [&](auto op, c::Mechanism::Kind kind) {
        auto a = lane(op.getSrcPipe().getPipe()), b = lane(op.getDstPipe().getPipe());
        if (!a || !b)
            return false;
        auto key = unsigned(op.getEventId().getEvent());
        if (precision) {
            m = {kind, *a, *b, key, 0};
            cursor = cursor->getNextNode();
            return true;
        }
        for (unsigned fallback : program.target.compilerKeys)
            if (program.target.available(
                    {program.core, static_cast<ss::Pipe>(*a)}, {program.core, static_cast<ss::Pipe>(*b)}, fallback)) {
                if (key == fallback)
                    return false;
                m = {kind, *a, *b, key, 0};
                cursor = cursor->getNextNode();
                return true;
            }
        return false;
    };
    if (auto s = dyn_cast<SetFlagOp>(cursor))
        if (independent(s, c::Mechanism::Publish))
            return true;
    if (auto w = dyn_cast<WaitFlagOp>(cursor))
        if (independent(w, c::Mechanism::Acquire))
            return true;
    auto s = dyn_cast<SetFlagOp>(cursor);
    if (!s)
        return false;
    auto* n = cursor->getNextNode();
    auto w = dyn_cast_or_null<WaitFlagOp>(n);
    n = n ? n->getNextNode() : nullptr;
    auto reply = dyn_cast_or_null<SetFlagOp>(n);
    n = n ? n->getNextNode() : nullptr;
    auto ack = dyn_cast_or_null<WaitFlagOp>(n);
    if (!w || !reply || !ack)
        return false;
    auto a = lane(s.getSrcPipe().getPipe()), b = lane(s.getDstPipe().getPipe());
    if (!a || !b || *a >= *b || w.getSrcPipe() != s.getSrcPipe() || w.getDstPipe() != s.getDstPipe() ||
        w.getEventId() != s.getEventId() || reply.getSrcPipe() != s.getDstPipe() ||
        reply.getDstPipe() != s.getSrcPipe() || ack.getSrcPipe() != reply.getSrcPipe() ||
        ack.getDstPipe() != reply.getDstPipe() || ack.getEventId() != reply.getEventId())
        return false;
    m = {
        c::Mechanism::Rendezvous, *a, *b, unsigned(s.getEventId().getEvent()), unsigned(reply.getEventId().getEvent())};
    cursor = ack->getNextNode();
    return true;
}

bool reconstruct(
    Tree& tree, Operation* drain, std::vector<std::vector<c::Mechanism>>& actual, std::string& reason, bool precision,
    const ParticipationGuards& guards)
{
    actual.resize(tree.program.nodes.size());
    bool valid = true;
    tree.inventory.function.walk([&](Operation* parent) {
        if (guards.commands.count(parent))
            return;
        for (Region& region : parent->getRegions())
            for (Block& block : region) {
                Operation* cursor = block.empty() ? nullptr : &block.front();
                while (cursor) {
                    auto guard = guards.commands.find(cursor);
                    if (guard != guards.commands.end()) {
                        auto found = tree.ids.find(guards.targets.lookup(cursor));
                        if (found == tree.ids.end()) {
                            valid = false;
                            break;
                        }
                        auto& out = actual[found->second];
                        out.insert(out.end(), guard->second.begin(), guard->second.end());
                        cursor = cursor->getNextNode();
                        continue;
                    }
                    if (!sync(cursor) || cursor == drain) {
                        cursor = cursor->getNextNode();
                        continue;
                    }
                    std::vector<c::Mechanism> packets;
                    while (cursor && sync(cursor) && cursor != drain) {
                        c::Mechanism m;
                        if (!parsePacket(cursor, m, tree.program, precision)) {
                            valid = false;
                            break;
                        }
                        packets.push_back(m);
                    }
                    if (!valid)
                        break;
                    // The retirement drain is not an original cut or completion receipt.
                    // Exit acquisitions immediately before it still belong to the
                    // original terminator boundary.
                    if (cursor == drain)
                        cursor = cursor->getNextNode();
                    auto* target = cursor;
                    while (target && !tree.ids.count(target))
                        target = target->getNextNode();
                    auto found = tree.ids.find(target);
                    if (!target || found == tree.ids.end()) {
                        valid = false;
                        break;
                    }
                    auto& out = actual[found->second];
                    out.insert(out.end(), packets.begin(), packets.end());
                }
            }
    });
    if (!valid)
        reason = "malformed, unbalanced, or misplaced compositional protocol";
    return valid;
}
} // namespace

Outcome ss::constructCompositionalSync(
    func::FuncOp function, InsertSyncGMAliasMode gm, HardwareContract hardware, bool enablePrecision)
{
    return testing::constructCompositionalSync(
        function, gm, {}, hardware,
        enablePrecision ? testing::CompositionConstructor::Demands : testing::CompositionConstructor::Conservative);
}

Outcome ss::testing::constructCompositionalSync(
    func::FuncOp function, InsertSyncGMAliasMode gm, llvm::function_ref<void(func::FuncOp)> mutate,
    HardwareContract hardware, CompositionConstructor constructor)
{
    const bool precision = constructor == CompositionConstructor::Cuts;
    const bool rejectRefinement = constructor == CompositionConstructor::DemandsRejectRefinement;
    const bool rejectEntry = constructor == CompositionConstructor::DemandsRejectEntryProposal;
    const bool rejectDeferred = constructor == CompositionConstructor::DemandsRejectDeferredRings;
    const bool fallbackOnly = constructor == CompositionConstructor::DemandsFallbackOnly;
    const bool withoutReplay = constructor == CompositionConstructor::DemandsWithoutAllocationReplay;
    const bool rejectReplay = constructor == CompositionConstructor::DemandsRejectAllocationReplay;
    const bool withoutLateEntry = constructor == CompositionConstructor::DemandsWithoutLateEntry;
    const bool rejectLateEntry = constructor == CompositionConstructor::DemandsRejectLateEntry;
    const bool withoutChoice = constructor == CompositionConstructor::DemandsWithoutChoiceDemands;
    const bool rejectChoice = constructor == CompositionConstructor::DemandsRejectChoiceDemands;
    const bool demandPlacement = constructor == CompositionConstructor::Demands || rejectRefinement || fallbackOnly ||
                                 withoutReplay || rejectReplay || rejectEntry || rejectDeferred || withoutLateEntry ||
                                 rejectLateEntry || withoutChoice || rejectChoice;
    Outcome out;
    if (function.isDeclaration() || !llvm::hasSingleElement(function.getBody())) {
        out.reason = "composition requires a single function block";
        return out;
    }
    if (hardware != HardwareContract::Conservative && hardware != HardwareContract::A2A3MmadAccV1) {
        out.reason = "unknown structured hardware contract";
        return out;
    }
    auto start = std::chrono::steady_clock::now();
    OwningOpRef<ModuleOp> stage(ModuleOp::create(function.getLoc()));
    SmallVector<ModuleOp> ancestors;
    for (Operation* op = function->getParentOp(); op; op = op->getParentOp())
        if (auto module = dyn_cast<ModuleOp>(op))
            ancestors.push_back(module);
    ModuleOp parent = *stage;
    llvm::DenseMap<Operation*, Operation*> stagedSymbols;
    for (auto module : llvm::reverse(ancestors)) {
        auto child = ModuleOp::create(module.getLoc());
        child->setAttrs(module->getAttrs());
        parent.getBody()->push_back(child);
        parent = child;
        stagedSymbols[module] = child;
    }
    IRMapping mapping;
    auto working = cast<func::FuncOp>(function->clone(mapping));
    parent.getBody()->push_back(working);
    stagedSymbols[function] = working;
    if (!stageDependencies(function, stagedSymbols, out.reason))
        return out;
    Inventory inventory(working);
    if (!inventory.build()) {
        out.reason = inventory.reason;
        return out;
    }
    Tree tree(inventory, hardware, gm);
    if (!tree.build(demandPlacement)) {
        out.reason = tree.reason;
        return out;
    }
    if (fallbackOnly)
        tree.program.target.compilerKeys = {0};
    auto selected = withoutChoice    ? c::testing::constructDemandsWithoutChoiceDemands(tree.program) :
                    rejectChoice     ? c::testing::constructDemandsRejectingChoiceDemands(tree.program) :
                    withoutLateEntry ? c::testing::constructDemandsWithoutLateEntry(tree.program) :
                    rejectLateEntry  ? c::testing::constructDemandsRejectingLateEntry(tree.program) :
                    rejectDeferred   ? c::testing::constructDemandsRejectingDeferredRings(tree.program) :
                    rejectEntry      ? c::testing::constructDemandsRejectingEntryProposal(tree.program) :
                    withoutReplay    ? c::testing::constructDemandsWithoutAllocationReplay(tree.program) :
                    rejectReplay     ? c::testing::constructDemandsRejectingAllocationReplay(tree.program) :
                    rejectRefinement ? c::testing::constructDemandsRejectingRefinement(tree.program) :
                    demandPlacement  ? c::constructDemands(tree.program) :
                    precision        ? c::constructCuts(tree.program) :
                                       c::construct(tree.program);
    if (!selected.success) {
        out.reason = selected.reason;
        return out;
    }
    SyncPayloadSnapshot snapshot(working);
    llvm::SmallPtrSet<Operation*, 32> originalOperations, generatedOperations;
    working.walk([&](Operation* op) { originalOperations.insert(op); });
    emit(tree, selected);
    Operation* scope = inventory.physical.lifetimeScope;
    Block& last = scope == working.getOperation() ? working.getBody().front() : scope->getRegion(0).front();
    Operation* terminator = !last.empty() && last.back().hasTrait<OpTrait::IsTerminator>() ? &last.back() : nullptr;
    OpBuilder b(working.getContext());
    if (terminator)
        b.setInsertionPoint(terminator);
    else
        b.setInsertionPointToEnd(&last);
    b.create<BarrierOp>(working.getLoc(), PipeAttr::get(working.getContext(), PIPE::PIPE_ALL));
    working.walk([&](Operation* op) {
        if (!originalOperations.contains(op))
            generatedOperations.insert(op);
    });
    if (mutate)
        mutate(working);
    out.status = Outcome::InternalError;
    ParticipationGuards guards;
    if (failed(mlir::verify(working)) || (demandPlacement && !guards.build(tree, generatedOperations, out.reason)) ||
        !snapshot.preserved(working, [&](Operation* op) {
            return generatedOperations.contains(op) && (sync(op) || guards.operations.contains(op));
        })) {
        out.reason = "compositional emission changed original payload or control";
        return out;
    }
    Operation* drain = nullptr;
    unsigned drains = 0;
    working.walk([&](BarrierOp barrier) {
        if (barrier.getPipe().getPipe() == PIPE::PIPE_ALL) {
            drain = barrier;
            ++drains;
        }
    });
    if (drains != 1 || drain->getBlock() != &last || drain->getNextNode() != terminator ||
        drain->hasAttr("pto.auto_sync_tail_barrier") || drain->hasAttr("pto.auto_sync_tail_hint")) {
        out.reason = "one explicit unconditional physical-context retirement drain is required";
        return out;
    }
    Inventory fresh(working);
    if (!fresh.build(true)) {
        out.reason = fresh.reason;
        return out;
    }
    Tree rebuilt(fresh, hardware, gm);
    rebuilt.ignored = &guards.operations;
    if (!rebuilt.build(demandPlacement)) {
        out.reason = rebuilt.reason;
        return out;
    }
    // Re-extract the original phase population after emission, including all
    // sibling and loop-carried effects. Never trust selected child requirements.
    if (fresh.physical.phases.size() != inventory.physical.phases.size()) {
        out.reason = "physical phase population changed";
        return out;
    }
    for (unsigned i = 0; i < fresh.physical.phases.size(); ++i) {
        auto *a = inventory.physical.phases[i], *d = fresh.physical.phases[i];
        auto equal = [](const auto& x, const auto& y) {
            if (x.size() != y.size())
                return false;
            for (unsigned j = 0; j < x.size(); ++j)
                if (!(*x[j] == *y[j]))
                    return false;
            return true;
        };
        if (a->elementOp != d->elementOp || a->kPipeValue != d->kPipeValue || !equal(a->useVec, d->useVec) ||
            !equal(a->defVec, d->defVec)) {
            out.reason = "physical effect contract changed";
            return out;
        }
    }
    if (fallbackOnly)
        rebuilt.program.target.compilerKeys = {0};
    std::vector<std::vector<c::Mechanism>> actual;
    if (!reconstruct(rebuilt, drain, actual, out.reason, precision, guards))
        return out;
    auto checked = demandPlacement ? c::verifyDemands(rebuilt.program, actual) :
                   precision       ? c::verifyCuts(rebuilt.program, actual) :
                                     c::verify(rebuilt.program, actual);
    if (!checked.success) {
        out.reason = checked.reason;
        return out;
    }
    unsigned rendezvousPackets = 0;
    for (const auto& site : actual)
        for (const auto& m : site) {
            if (m.kind == c::Mechanism::Barrier)
                ++out.barriers;
            else if (m.kind == c::Mechanism::Rendezvous) {
                ++rendezvousPackets;
                out.handoffs += 2;
            } else if (m.kind == c::Mechanism::Publish)
                ++out.handoffs;
        }
    out.requirements = selected.acquisitions;
    out.work = selected.cellVisits + checked.cellVisits;
    out.work += tree.periodicScalarWork + rebuilt.periodicScalarWork;
    out.status = Outcome::Applied;
    out.reason = "compositional storage, completion, reusable protocols and retirement verified";
    function.getBody().takeBody(working.getBody());
    if (std::getenv("PTOAS_LOGICAL_TRACE") && !selected.deferredRejectionStage.empty())
        llvm::errs() << "structured deferred_rejection stage " << selected.deferredRejectionStage << " reason "
                     << selected.deferredRejectionReason << "\n";
    if (std::getenv("PTOAS_LOGICAL_TRACE"))
        llvm::errs()
            << "structured composition precision " << precision << " demands " << demandPlacement << " direct_handoffs "
            << selected.directHandoffs << " shared_acknowledgments " << selected.sharedAcknowledgments
            << " reused_acknowledgments " << selected.reusedAcknowledgments << " completion_refinements "
            << selected.completionRefinements << " rejected_refinements " << selected.rejectedRefinements
            << " owned_refinements " << checked.ownedRefinements << " protocol_keys " << selected.protocolKeys
            << " shared_protocol_keys " << selected.sharedProtocolKeys << " allocation_fallback_keys "
            << selected.allocationFallbackKeys << " allocation_replays " << selected.allocationReplays
            << " rejected_allocation_replays " << selected.rejectedAllocationReplays << " replay_commands_removed "
            << selected.replayCommandsRemoved << " replayed_fallback_demands " << selected.replayedFallbackDemands
            << " allocation_fallback_scopes " << selected.allocationFallbackScopes << " entry_episodes "
            << selected.entryEpisodes << " entry_reply_families " << selected.entryReplyFamilies
            << " rejected_entry_proposals " << selected.rejectedEntryProposals << " entry_summary_slots "
            << selected.entrySummarySlots << " entry_summary_scans " << selected.entrySummaryScans
            << " entry_storage_units " << selected.entryStorageUnits << " entry_candidate_pairs "
            << selected.entryCandidatePairs << " entry_witness_cells " << selected.entryWitnessCells
            << " entry_witnesses " << selected.entryWitnesses << " entry_source_overlap_rejections "
            << selected.entrySourceOverlapRejections << " entry_summary_skipped " << selected.entrySummarySkipped
            << " late_entry_candidates " << selected.lateEntryCandidates << " late_entry_families "
            << selected.lateEntryFamilies << " late_entry_sites " << selected.lateEntrySites
            << " rejected_late_entry_families " << selected.rejectedLateEntryFamilies << " choice_demand_candidates "
            << selected.choiceDemandCandidates << " choice_demand_families " << selected.choiceDemandFamilies
            << " rejected_choice_demands " << selected.rejectedChoiceDemands << " choice_demand_work "
            << selected.choiceDemandWork << " choice_demand_reserved_work " << selected.choiceDemandReservedWork
            << " choice_demand_analysis_work " << selected.choiceDemandAnalysisWork << " choice_demand_analysis_passes "
            << selected.choiceDemandAnalysisPasses << " choice_demand_budget_pass " << selected.choiceDemandBudgetPass
            << " ring_candidates " << selected.ringCandidates << " rejected_rings " << selected.rejectedRings
            << " ring_candidate_commands_removed " << selected.ringCandidateCommandsRemoved << " rendezvous_packets "
            << rendezvousPackets << " deferred_ring_candidates " << selected.deferredRingCandidates
            << " deferred_rings " << selected.deferredRings << " rejected_deferred_rings "
            << selected.rejectedDeferredRings << " periodic_deferred_rings " << selected.periodicDeferredRings
            << " periodic_write_overlap_rejections " << selected.periodicWriteOverlapRejections
            << " periodic_scalar_work " << tree.periodicScalarWork + rebuilt.periodicScalarWork
            << " periodic_dag_visits " << tree.periodicDagVisits + rebuilt.periodicDagVisits
            << " periodic_residue_evaluations " << tree.periodicEvaluations + rebuilt.periodicEvaluations
            << " deferred_discovery_work " << selected.deferredDiscoveryWork << " deferred_discovery_refusals "
            << selected.deferredDiscoveryRefusals << " deferred_discovery_limit " << c::DeferredDiscoveryLimit
            << " deferred_eligibility_work " << selected.deferredEligibilityWork + checked.deferredEligibilityWork
            << " deferred_receipt_cells " << selected.deferredReceiptCells + checked.deferredReceiptCells
            << " deferred_skipped_families " << selected.deferredSkippedFamilies << " deferred_protocol_steps "
            << selected.deferredProtocolSteps + checked.deferredProtocolSteps << " demand_fallbacks "
            << selected.demandFallbacks << " nodes " << tree.program.nodes.size() << " cells " << tree.program.cells
            << " widened_spaces " << tree.widenedSpaces << " node_visits " << selected.nodeVisits + checked.nodeVisits
            << " cell_visits " << selected.cellVisits + checked.cellVisits << " handoffs " << out.handoffs
            << " cut_cycles " << checked.cutCycles << " allocation_retries " << selected.allocationRetries
            << " barriers " << out.barriers << " seconds "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << "\n";
    return out;
}
