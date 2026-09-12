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
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncPayloadSnapshot.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
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
    llvm::DenseMap<Operation*, const CompoundInstanceElement*> phases;
    std::string reason;
    uint64_t widenedSpaces = 0;
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
            if (sync(&op))
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
    bool build()
    {
        if (!gmContracts())
            return false;
        partition();
        unsigned root;
        auto* scope = inventory.physical.lifetimeScope;
        return region(
            scope == inventory.function.getOperation() ? inventory.function.getBody() : scope->getRegion(0), root);
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
        for (const auto& m : plan.before[id]) {
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
    Tree& tree, Operation* drain, std::vector<std::vector<c::Mechanism>>& actual, std::string& reason, bool precision)
{
    actual.resize(tree.program.nodes.size());
    bool valid = true;
    tree.inventory.function.walk([&](Operation* parent) {
        for (Region& region : parent->getRegions())
            for (Block& block : region) {
                Operation* cursor = block.empty() ? nullptr : &block.front();
                while (cursor) {
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
                    auto found = tree.ids.find(cursor);
                    if (!cursor || found == tree.ids.end()) {
                        valid = false;
                        break;
                    }
                    actual[found->second] = std::move(packets);
                }
            }
    });
    if (!valid)
        reason = "malformed, unbalanced, or misplaced compositional protocol";
    return valid;
}
} // namespace

Outcome ss::testing::constructCompositionalSync(
    func::FuncOp function, InsertSyncGMAliasMode gm, llvm::function_ref<void(func::FuncOp)> mutate,
    HardwareContract hardware, bool precision)
{
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
    for (auto module : llvm::reverse(ancestors)) {
        auto child = ModuleOp::create(module.getLoc());
        child->setAttrs(module->getAttrs());
        parent.getBody()->push_back(child);
        parent = child;
    }
    IRMapping mapping;
    auto working = cast<func::FuncOp>(function->clone(mapping));
    parent.getBody()->push_back(working);
    Inventory inventory(working);
    if (!inventory.build()) {
        out.reason = inventory.reason;
        return out;
    }
    Tree tree(inventory, hardware, gm);
    if (!tree.build()) {
        out.reason = tree.reason;
        return out;
    }
    auto selected = precision ? c::constructCuts(tree.program) : c::construct(tree.program);
    if (!selected.success) {
        out.reason = selected.reason;
        return out;
    }
    SyncPayloadSnapshot snapshot(working);
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
    if (mutate)
        mutate(working);
    out.status = Outcome::InternalError;
    if (failed(mlir::verify(working)) || !snapshot.preserved(working, sync)) {
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
    if (!rebuilt.build()) {
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
    std::vector<std::vector<c::Mechanism>> actual;
    if (!reconstruct(rebuilt, drain, actual, out.reason, precision))
        return out;
    auto checked = precision ? c::verifyCuts(rebuilt.program, actual) : c::verify(rebuilt.program, actual);
    if (!checked.success) {
        out.reason = checked.reason;
        return out;
    }
    for (const auto& site : actual)
        for (const auto& m : site) {
            if (m.kind == c::Mechanism::Barrier)
                ++out.barriers;
            else if (m.kind == c::Mechanism::Rendezvous)
                out.handoffs += 2;
            else if (m.kind == c::Mechanism::Publish)
                ++out.handoffs;
        }
    out.requirements = selected.acquisitions;
    out.work = selected.cellVisits + checked.cellVisits;
    out.status = Outcome::Applied;
    out.reason = "compositional storage, completion, reusable protocols and retirement verified";
    function.getBody().takeBody(working.getBody());
    if (std::getenv("PTOAS_LOGICAL_TRACE"))
        llvm::errs() << "structured composition precision " << precision << " nodes " << tree.program.nodes.size()
                     << " cells " << tree.program.cells << " widened_spaces " << tree.widenedSpaces << " node_visits "
                     << selected.nodeVisits + checked.nodeVisits << " cell_visits " << out.work << " handoffs "
                     << out.handoffs << " cut_cycles " << checked.cutCycles << " allocation_retries "
                     << selected.allocationRetries << " barriers " << out.barriers << " seconds "
                     << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << "\n";
    return out;
}
