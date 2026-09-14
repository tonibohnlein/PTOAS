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
#include "PTO/Transforms/InsertSync/StructuredSyncMatrixContract.h"
#include "PTO/Transforms/InsertSync/StructuredSyncUnitFlagContract.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncPayloadSnapshot.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/JSON.h"
#include "mlir/IR/AsmState.h"
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
bool sync(Operation* op)
{
    return isa<SetFlagOp, WaitFlagOp, BarrierOp, CmoCacheInvalidOp, FenceBarrierAllOp, TNotifyOp, TWaitOp>(op);
}

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
    bool build(bool actual = false, bool authored = false)
    {
        PTOIRTranslator translator(ir, memory, buffers, function, SyncAnalysisMode::NORMALSYNC, true);
        if (failed(translator.Build())) {
            reason = "physical translation failed";
            return false;
        }
        physical = importCoverageStructuredSyncPhysicalFacts(function, ir, true, actual, authored);
        reason = physical.reason;
        return physical.status == SyncPhysicalFacts::Status::Complete;
    }
};

using UnitFlagMap = llvm::DenseMap<Operation*, ss::UnitFlagInfo>;

bool validateUnitFlagOwnership(Inventory& inventory, ss::OwnershipContract contract, UnitFlagMap& transitions,
                               std::string& reason)
{
    bool authored = false;
    inventory.function.walk([&](Operation* op) {
        if (auto phase = op->getAttrOfType<AccPhaseAttr>("accPhase"))
            authored |= phase.getValue() != AccPhase::Unspecified;
        if (auto phase = op->getAttrOfType<STPhaseAttr>("stPhase"))
            authored |= phase.getValue() != STPhase::Unspecified;
    });
    if (!authored)
        return true;
    if (contract != ss::OwnershipContract::A2A3UnitFlagPairedV1) {
        reason = "authored UnitFlag ownership requires a2a3-unitflag-paired-v1";
        return false;
    }
    if (!inventory.physical.cube) {
        reason = "a2a3 UnitFlag ownership requires an AIC physical context";
        return false;
    }

    std::vector<std::pair<uint64_t, uint64_t>> domains;
    std::map<std::pair<uint64_t, uint64_t>, std::pair<uint64_t, uint64_t>> domainGeometry;
    for (auto* phase : inventory.physical.phases) {
        bool enabled = false;
        if (auto attr = phase->elementOp->getAttrOfType<AccPhaseAttr>("accPhase"))
            enabled |= attr.getValue() != AccPhase::Unspecified;
        if (auto attr = phase->elementOp->getAttrOfType<STPhaseAttr>("stPhase"))
            enabled |= attr.getValue() != STPhase::Unspecified;
        if (!enabled)
            continue;
        auto info = ss::unitFlagLoweringFacts(*phase);
        if (!info || transitions.count(phase->elementOp)) {
            reason = "authored UnitFlag operation lacks exact paired lowering coverage";
            return false;
        }
        transitions[phase->elementOp] = info;
        auto domain = std::make_pair(info.base, info.base + info.bytes);
        auto [geometry, inserted] = domainGeometry.emplace(domain, std::make_pair(info.rows, info.cols));
        if (!inserted && geometry->second != std::make_pair(info.rows, info.cols)) {
            reason = "UnitFlag producer/store geometry does not match within its ownership domain";
            return false;
        }
        domains.push_back(domain);
    }
    llvm::sort(domains);
    domains.erase(std::unique(domains.begin(), domains.end()), domains.end());
    auto metadata = inventory.function->getAttrOfType<DenseI64ArrayAttr>("pto.unitflag_entry_writable");
    if (!metadata || metadata.size() % 2) {
        reason = "UnitFlag profile requires base/length pto.unitflag_entry_writable metadata";
        return false;
    }
    std::vector<std::pair<uint64_t, uint64_t>> declared;
    for (unsigned i = 0; i < metadata.size(); i += 2) {
        int64_t base = metadata[i], bytes = metadata[i + 1];
        if (base < 0 || bytes <= 0 || (base % 512) || (bytes % 512) ||
            uint64_t(base) > UINT64_MAX - uint64_t(bytes)) {
            reason = "invalid UnitFlag entry-writable interval";
            return false;
        }
        std::pair<uint64_t, uint64_t> interval{uint64_t(base), uint64_t(base) + uint64_t(bytes)};
        if (!declared.empty() && declared.back().second > interval.first) {
            reason = "UnitFlag entry-writable intervals overlap or are unsorted";
            return false;
        }
        declared.push_back(interval);
    }
    if (declared != domains) {
        reason = "UnitFlag entry-writable metadata does not exactly cover qualified domains";
        return false;
    }

    // Every ACC access overlapping a tracked domain must be exactly the
    // qualified transition.  Operand and GM effects are intentionally outside
    // this ownership state machine.
    for (auto* phase : inventory.physical.phases) {
        auto transition = transitions.find(phase->elementOp);
        auto inspect = [&](const auto& entries) {
            for (auto* memory : entries) {
                if (!memory || memory->scope != AddressSpace::ACC)
                    continue;
                for (uint64_t base : memory->baseAddresses)
                    for (auto domain : domains)
                        if (base < domain.second && domain.first < base + memory->allocateSize) {
                            if (transition == transitions.end() || memory->baseAddresses.size() != 1 ||
                                base != transition->second.base || memory->allocateSize != transition->second.bytes)
                                return false;
                        }
            }
            return true;
        };
        if (!inspect(phase->useVec) || !inspect(phase->defVec)) {
            reason = "unqualified or partial ACC access overlaps a UnitFlag ownership domain";
            return false;
        }
    }

    std::map<std::pair<uint64_t, uint64_t>, unsigned> index;
    for (unsigned i = 0; i < domains.size(); ++i)
        index[domains[i]] = i;
    using State = std::vector<uint8_t>;
    std::function<bool(Region&, State&)> region;
    std::function<bool(Operation*, State&)> operation = [&](Operation* op, State& state) {
        // This exact scope was already qualified by physical import. It
        // executes once; it is not an arbitrary region-control exemption.
        if (op == inventory.physical.lifetimeScope && op != inventory.function.getOperation()) {
            if (op->getNumRegions() != 1) {
                reason = "qualified UnitFlag lifetime scope must have one region";
                return false;
            }
            return region(op->getRegion(0), state);
        }
        auto found = transitions.find(op);
        if (found != transitions.end()) {
            auto domain = std::make_pair(found->second.base, found->second.base + found->second.bytes);
            unsigned id = index.at(domain);
            bool producer = found->second.kind == ss::UnitFlagInfo::ProducerFinal;
            if (state[id] != unsigned(producer)) {
                reason = producer ? "UnitFlag producer requires writable entry ownership" :
                                    "UnitFlag store requires readable producer ownership";
                return false;
            }
            state[id] = !producer;
            return true;
        }
        if (auto loop = dyn_cast<scf::ForOp>(op)) {
            auto body = state;
            if (!region(loop.getRegion(), body) || body != state) {
                if (reason.empty())
                    reason = "UnitFlag counted-loop ownership is not balanced";
                return false;
            }
            return true; // includes the original zero-trip edge
        }
        if (auto choice = dyn_cast<scf::IfOp>(op)) {
            auto yes = state, no = state;
            if (!region(choice.getThenRegion(), yes) || !region(choice.getElseRegion(), no) || yes != no) {
                if (reason.empty())
                    reason = "UnitFlag branch ownership does not agree on every successor";
                return false;
            }
            state = std::move(yes);
            return true;
        }
        if (auto loop = dyn_cast<scf::WhileOp>(op)) {
            auto before = state, after = state;
            if (!region(loop.getBefore(), before) || before != state || !region(loop.getAfter(), after) ||
                after != state) {
                if (reason.empty())
                    reason = "UnitFlag while ownership is not independently balanced";
                return false;
            }
            return true;
        }
        if (op->getNumRegions()) {
            bool contains = false;
            op->walk([&](Operation* nested) { contains |= nested != op && transitions.count(nested); });
            if (contains) {
                reason = "UnitFlag ownership appears in unsupported region control";
                return false;
            }
        }
        return true;
    };
    region = [&](Region& current, State& state) {
        if (current.empty())
            return true;
        if (!llvm::hasSingleElement(current)) {
            reason = "UnitFlag ownership requires single-block structured regions";
            return false;
        }
        for (Operation& op : current.front())
            if (!op.hasTrait<OpTrait::IsTerminator>() && !operation(&op, state))
                return false;
        return true;
    };
    State state(domains.size(), 1);
    if (!region(inventory.function.getBody(), state))
        return false;
    if (llvm::any_of(state, [](uint8_t writable) { return writable != 1; })) {
        reason = "UnitFlag ownership is not returned writable at function exit";
        return false;
    }
    return true;
}

struct Cell {
    AddressSpace space;
    uint64_t lower = 0, upper = 0;
    bool whole = false;
    SmallVector<Value> gmRoots;
    // Covers coordinates outside retained exact intervals. Known accesses
    // never touch it merely because another access has unknown geometry.
    bool remainder = false;
    // A may-alias pair has independent coordinates in each backing root.
    // Keeping both coordinates prevents B from merging disjoint A intervals.
    struct PeerRange {
        uint64_t lower, upper;
        bool whole, remainder;
    };
    std::optional<PeerRange> peerRange;
};
struct Tree {
    Inventory& inventory;
    c::Program program;
    std::vector<Cell> cells;
    std::vector<Operation*> anchors;
    // Demand construction needs a real post-body cut even for region kinds
    // with no implicit terminator.  Native emission binds this synthetic leaf
    // to the required retirement drain, and reconstruction maps commands
    // immediately preceding that drain back to the same leaf.
    unsigned terminal = ~0u;
    llvm::DenseMap<Operation*, unsigned> ids;
    llvm::SmallPtrSet<Operation*, 32> fixedOperations;
    const llvm::SmallPtrSetImpl<Operation*>* ignored = nullptr;
    llvm::DenseMap<Operation*, SmallVector<const CompoundInstanceElement*, 2>> phases;
    std::string reason;
    bool preserveFixedCuts = false;
    uint64_t widenedSpaces = 0;
    uint64_t gmGeometryWork = 0;
    bool gmGeometryExhausted = false;
    uint64_t periodicScalarWork = 0;
    uint64_t periodicDagVisits = 0, periodicEvaluations = 0;
    InsertSyncGMAliasMode gm;
    llvm::DenseMap<const BaseMemInfo*, InsertSyncGMRoots> gmAccesses;
    llvm::DenseMap<const BaseMemInfo*, std::optional<InsertSyncGMRange>> gmRanges;
    std::set<std::pair<unsigned, unsigned>> disjointPairs;
    struct ScalarArgumentContract {
        int64_t minimum = 0, maximum = 0, multiple = 1;
    };
    std::map<unsigned, ScalarArgumentContract> scalarArguments;
    const UnitFlagMap* unitFlags = nullptr;

    Tree(Inventory& i, ss::HardwareContract hardware, InsertSyncGMAliasMode alias,
         ss::OwnershipContract ownership = ss::OwnershipContract::None, bool ownershipCredit = true,
         const UnitFlagMap* qualifiedUnitFlags = nullptr)
        : inventory(i), gm(alias), unitFlags(qualifiedUnitFlags)
    {
        program.core = i.physical.cube ? ss::Core::AIC : ss::Core::AIV;
        program.target.hardware = hardware;
        program.target.ownership = ownership;
        program.target.ownershipCredit = ownershipCredit;
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
    bool scalarContracts()
    {
        auto raw = inventory.function->getAttr("pto.scalar_argument_preconditions");
        if (!raw)
            return true;
        auto values = dyn_cast<DenseI64ArrayAttr>(raw);
        if (!values || values.size() % 4) {
            reason = "scalar argument preconditions require arg/min/max/multiple quadruples";
            return false;
        }
        for (unsigned offset = 0; offset < values.size(); offset += 4) {
            int64_t argument = values[offset], minimum = values[offset + 1];
            int64_t maximum = values[offset + 2], multiple = values[offset + 3];
            if (argument < 0 || uint64_t(argument) >= inventory.function.getNumArguments() ||
                minimum > maximum || multiple <= 0) {
                reason = "invalid scalar argument precondition interval";
                return false;
            }
            auto type = dyn_cast<IntegerType>(inventory.function.getArgument(unsigned(argument)).getType());
            if (!type || !APInt(64, uint64_t(minimum), true).isSignedIntN(type.getWidth()) ||
                !APInt(64, uint64_t(maximum), true).isSignedIntN(type.getWidth()) ||
                !scalarArguments.emplace(unsigned(argument), ScalarArgumentContract{minimum, maximum, multiple}).second) {
                reason = "scalar argument precondition type or argument is invalid";
                return false;
            }
        }
        return true;
    }
    void qualifyNonEmptyLoops()
    {
        for (unsigned id = 0; id < program.nodes.size(); ++id) {
            auto& node = program.nodes[id];
            if (node.kind != c::Node::For || !anchors[id])
                continue;
            auto loop = cast<scf::ForOp>(anchors[id]);
            APInt lower, step, divisor;
            if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) || lower != 0 ||
                !matchPattern(loop.getStep(), m_ConstantInt(&step)) || step != 1)
                continue;
            auto division = loop.getUpperBound().getDefiningOp<arith::DivSIOp>();
            if (!division || !matchPattern(division.getRhs(), m_ConstantInt(&divisor)) ||
                !divisor.isStrictlyPositive() || !divisor.isSignedIntN(64))
                continue;
            auto cast = division.getLhs().getDefiningOp<arith::IndexCastOp>();
            auto argument = cast ? dyn_cast<BlockArgument>(cast.getIn()) : dyn_cast<BlockArgument>(division.getLhs());
            if (!argument || argument.getOwner() != &inventory.function.getBody().front())
                continue;
            auto contract = scalarArguments.find(argument.getArgNumber());
            int64_t width = divisor.getSExtValue();
            if (contract != scalarArguments.end() && contract->second.minimum >= width &&
                contract->second.multiple % width == 0)
                node.nonEmpty = true;
        }
    }
    std::vector<Cell> gmCells()
    {
        auto charge = [&](uint64_t amount) {
            if (amount > (1u << 20) - gmGeometryWork) {
                gmGeometryExhausted = true;
                return false;
            }
            gmGeometryWork += amount;
            return true;
        };
        SmallVector<Value> roots;
        llvm::DenseMap<Value, SmallVector<const BaseMemInfo*>> byRoot;
        bool unknown = false;
        for (auto* p : inventory.physical.phases) {
            auto collect = [&](const auto& entries) {
                for (auto* a : entries)
                    if (a->scope == AddressSpace::GM) {
                        // Translation facts are cached even if optional cell
                        // geometry exhausts its allowance: the whole GM cell
                        // still needs every access during reconstruction.
                        if (gmAccesses.count(a))
                            continue;
                        auto traced = traceInsertSyncGMRoots(inventory.function, a->rootBuffer);
                        unknown |= !traced.complete;
                        if (charge(1 + traced.arguments.size())) for (Value root : traced.arguments) {
                            if (roots.size() <= c::MaxCells && !byRoot.count(root))
                                roots.push_back(root);
                            if (roots.size() <= c::MaxCells) {
                                auto& group = byRoot[root];
                                if (traced.complete) group.push_back(a);
                            }
                        }
                        gmAccesses[a] = std::move(traced);
                        gmRanges[a] = gmGeometryExhausted ? std::optional<InsertSyncGMRange>{} :
                            traceInsertSyncGMRange(inventory.function, a->baseBuffer);
                    }
            };
            collect(p->useVec);
            collect(p->defVec);
        }
        if (gmGeometryExhausted || roots.size() > c::MaxCells || roots.empty())
            return {{AddressSpace::GM, 0, 0, true, {}}};
        SmallVector<std::pair<unsigned, unsigned>> aliasPairs;
        for (unsigned i = 0; i < roots.size(); ++i)
            for (unsigned j = i + 1; j < roots.size(); ++j) {
                if (!charge(1)) return {{AddressSpace::GM, 0, 0, true, {}}};
                unsigned a = cast<BlockArgument>(roots[i]).getArgNumber();
                unsigned b = cast<BlockArgument>(roots[j]).getArgNumber();
                bool disjoint = gm == InsertSyncGMAliasMode::DisjointArguments ||
                                disjointPairs.count({std::min(a, b), std::max(a, b)});
                if (!disjoint) aliasPairs.push_back({i, j});
            }
        // Reserve one cell for every known root, possible-alias group and the
        // unknown-only remainder. Extra geometry is optional within its own
        // group, so pressure coarsens that group without consuming a later
        // group's required cell.
        if (roots.size() + aliasPairs.size() + unsigned(unknown) > c::MaxCells)
            return {{AddressSpace::GM, 0, 0, true, {}}};
        const unsigned cellLimit = c::MaxCells - unsigned(unknown);
        const unsigned rootLimit = cellLimit - aliasPairs.size();
        std::vector<Cell> result;
        llvm::DenseMap<Value, std::pair<unsigned, unsigned>> rootCells;
        // One cell per origin plus one per possible alias pair. Unlike a
        // transitive may-alias component, this does not make A and B overlap
        // just because a third origin can overlap either of them.
        for (auto [rootIndex, root] : llvm::enumerate(roots)) {
            if (!charge(1)) return {{AddressSpace::GM, 0, 0, true, {}}};
            unsigned remainingRoots = roots.size() - rootIndex - 1;
            unsigned groupLimit = rootLimit - remainingRoots;
            assert(result.size() < groupLimit && "GM root cell capacity was not reserved");
            std::set<uint64_t> bounds;
            bool uncertain = false;
            bool coarsened = false;
            for (const auto* access : byRoot[root]) {
                if (!charge(1)) return {{AddressSpace::GM, 0, 0, true, {}}};
                const auto& range = gmRanges.find(access)->second;
                if (!range || range->root != root)
                    uncertain = true;
                else {
                    bounds.insert(range->lower);
                    bounds.insert(range->upper);
                    if (bounds.size() > groupLimit - result.size() + 1) {
                        coarsened = true;
                        break;
                    }
                }
            }
            unsigned begin = result.size();
            uint64_t pieces = bounds.empty() ? 1 : bounds.size() - 1 + unsigned(uncertain);
            if (coarsened || bounds.empty() || pieces > groupLimit - result.size()) {
                result.push_back({AddressSpace::GM, 0, 0, true, {root}});
            } else {
                for (auto it = bounds.begin(); std::next(it) != bounds.end(); ++it)
                    result.push_back({AddressSpace::GM, *it, *std::next(it), false, {root}});
                if (uncertain)
                    result.push_back({AddressSpace::GM, 0, 0, false, {root}, true});
            }
            rootCells[root] = {begin, result.size()};
        }
        for (auto [pairIndex, indices] : llvm::enumerate(aliasPairs)) {
            auto [i, j] = indices;
            unsigned remainingPairs = aliasPairs.size() - pairIndex - 1;
            unsigned groupLimit = cellLimit - remainingPairs;
            assert(result.size() < groupLimit && "GM alias cell capacity was not reserved");
            auto [leftBegin, leftEnd] = rootCells.lookup(roots[i]);
            auto [rightBegin, rightEnd] = rootCells.lookup(roots[j]);
            uint64_t count = uint64_t(leftEnd - leftBegin) * (rightEnd - rightBegin);
            if (count > groupLimit - result.size()) {
                // Only this alias group loses geometry at the limit.
                result.push_back({AddressSpace::GM, 0, 0, true, {roots[i], roots[j]}});
                continue;
            }
            if (!charge(count)) return {{AddressSpace::GM, 0, 0, true, {}}};
            for (unsigned left = leftBegin; left < leftEnd; ++left)
                for (unsigned right = rightBegin; right < rightEnd; ++right) {
                    Cell pair = result[left];
                    const auto& peer = result[right];
                    pair.gmRoots.push_back(roots[j]);
                    pair.peerRange = Cell::PeerRange{peer.lower, peer.upper, peer.whole, peer.remainder};
                    result.push_back(std::move(pair));
                }
        }
        if (unknown) {
            assert(result.size() < c::MaxCells && "unknown GM cell capacity was not reserved");
            result.push_back({AddressSpace::GM, 0, 0, true, {}, true});
        }
        return result;
    }
    void reportWitnesses(const c::Result& selected)
    {
        if (!std::getenv("PTOAS_COMPOSITION_WITNESSES"))
            return;
        llvm::json::Array physical, operations, accesses, commands, pairs;
        struct Access {
            const BaseMemInfo* memory;
            Operation* operation;
            unsigned id, node;
            PipelineType pipeline;
            bool write;
            std::set<unsigned> cells;
            SmallVector<Operation*, 4> loops;
        };
        std::vector<Access> recorded;
        std::vector<unsigned> writers, readers;
        uint64_t reportWork = 0;
        constexpr uint64_t ReportLimit = 1u << 20;
        bool exhausted = false, legacyAvailable = false;
        auto charge = [&](uint64_t amount) {
            if (exhausted || amount > ReportLimit - reportWork) {
                exhausted = true;
                return false;
            }
            reportWork += amount;
            return true;
        };
        AsmState names(inventory.function);
        auto name = [&](Value value) {
            std::string text;
            llvm::raw_string_ostream stream(text);
            if (value) value.printAsOperand(stream, names);
            return text;
        };
        // Export the bounded actual failure before optional pair enumeration
        // can exhaust its independent diagnostic allowance.
        llvm::json::Value failure = nullptr;
        if (selected.publicationFailureNode != ~0u) {
            llvm::json::Array failedCells;
            bool complete = charge(selected.publicationFailureCells.size());
            if (complete)
                for (unsigned cell : selected.publicationFailureCells) failedCells.push_back(cell);
            failure = llvm::json::Object{
                {"node", selected.publicationFailureNode},
                {"macro_phase", selected.publicationFailurePhase == ~0u ? llvm::json::Value(nullptr) :
                    llvm::json::Value(selected.publicationFailurePhase)},
                {"cells", std::move(failedCells)}, {"cells_complete", complete},
                {"producer", "MTE3"}, {"consumer", "MTE2"},
                {"evidence", "actual-construction-may-obligation"},
                {"reaching_writer_identified", false}};
        }
        auto gather = [&]() {
            for (unsigned index = 0; index < cells.size(); ++index) {
                const auto& cell = cells[index];
                if (!charge(1 + cell.gmRoots.size())) return;
                llvm::json::Array roots;
                for (Value root : cell.gmRoots) roots.push_back(name(root));
                llvm::json::Object description{
                    {"id", index}, {"space", unsigned(cell.space)}, {"whole", cell.whole},
                    {"remainder", cell.remainder}, {"lower", std::to_string(cell.lower)},
                    {"upper", std::to_string(cell.upper)}, {"roots", std::move(roots)}};
                if (cell.peerRange) {
                    const auto& peer = *cell.peerRange;
                    description["peer_range"] = llvm::json::Object{
                        {"lower", std::to_string(peer.lower)}, {"upper", std::to_string(peer.upper)},
                        {"whole", peer.whole}, {"remainder", peer.remainder}};
                }
                physical.push_back(std::move(description));
            }
            for (unsigned id = 0; id < program.nodes.size(); ++id) {
                const auto& node = program.nodes[id];
                if (!charge(1 + node.children.size() + node.effects.size())) return;
                llvm::json::Array effects, children;
                for (unsigned child : node.children) children.push_back(child);
                for (unsigned cell = 0; cell < node.effects.size(); ++cell)
                    if (node.effects[cell].readers || node.effects[cell].writers)
                        effects.push_back(llvm::json::Object{{"cell", cell},
                            {"readers", unsigned(node.effects[cell].readers)},
                            {"writers", unsigned(node.effects[cell].writers)},
                            {"byte_readers", unsigned(node.storageAccesses()[cell].readers)},
                            {"byte_writers", unsigned(node.storageAccesses()[cell].writers)}});
                std::string operation;
                llvm::raw_string_ostream stream(operation);
                if (anchors[id]) anchors[id]->print(stream, OpPrintingFlags().skipRegions());
                operations.push_back(llvm::json::Object{{"id", id}, {"kind", unsigned(node.kind)},
                    {"lane", node.lane}, {"operation", operation}, {"children", std::move(children)},
                    {"effects", std::move(effects)}});
            }
            for (auto* phase : inventory.physical.phases) {
                auto mapped = ids.find(phase->elementOp);
                unsigned nodeId = mapped == ids.end() ? ~0u : mapped->second;
                unsigned phaseIndex = ~0u;
                if (nodeId != ~0u && program.nodes[nodeId].kind == c::Node::Macro) {
                    // The supported two-phase macro model has bounded private
                    // event/transfer storage. Reserve its reconstruction too.
                    if (!charge(16 + program.nodes[nodeId].macroPhases.size())) return;
                    auto model = getSyncMacroModel(phase->elementOp);
                    if (model) {
                        if (!charge(model->phases.size())) return;
                        for (unsigned i = 0; i < model->phases.size(); ++i)
                            if (int(model->phases[i].phaseId) == phase->macroOpInstanceId)
                                phaseIndex = i;
                    }
                }
                auto append = [&](const auto& entries, bool write) {
                    for (const auto* access : entries) {
                        auto cached = gmAccesses.find(access);
                        uint64_t rootCount = cached == gmAccesses.end() ? 1 : cached->second.arguments.size();
                        if (!charge(1 + rootCount + cells.size() * (1 + 2 * rootCount + access->baseAddresses.size())))
                            return false;
                        llvm::json::Array roots, ranges, cellIds;
                        bool complete = !access->aliasesUnknownRange;
                        if (cached != gmAccesses.end()) {
                            complete = cached->second.complete;
                            for (Value root : cached->second.arguments) roots.push_back(name(root));
                        } else roots.push_back(name(access->rootBuffer));
                        Access record{access, phase->elementOp, unsigned(recorded.size()),
                            nodeId, phase->kPipeValue, write, {}};
                        for (Operation* owner = phase->elementOp->getParentOp();
                             owner && owner != inventory.function.getOperation(); owner = owner->getParentOp()) {
                            if (!charge(1)) return false;
                            if (isa<scf::ForOp, scf::WhileOp>(owner)) record.loops.push_back(owner);
                        }
                        for (unsigned cell = 0; cell < cells.size(); ++cell)
                            if (accessOverlaps(cells[cell], access)) {
                                record.cells.insert(cell);
                                cellIds.push_back(cell);
                            }
                        recorded.push_back(record);
                        if (access->scope == AddressSpace::GM) {
                            if (write && phase->kPipeValue == PipelineType::PIPE_MTE3) writers.push_back(record.id);
                            if (!write && phase->kPipeValue == PipelineType::PIPE_MTE2) readers.push_back(record.id);
                        }
                        for (uint64_t base : access->baseAddresses) ranges.push_back(std::to_string(base));
                        llvm::json::Object row{{"node", nodeId == ~0u ? llvm::json::Value(nullptr) : llvm::json::Value(nodeId)},
                            {"mapping_status", nodeId == ~0u ? "unmapped" : "mapped"},
                            {"macro_phase", phaseIndex == ~0u ? llvm::json::Value(nullptr) : llvm::json::Value(phaseIndex)},
                            {"native_phase_id", phase->macroOpInstanceId},
                            {"pipeline", unsigned(phase->kPipeValue)}, {"write", write},
                            {"base", name(access->baseBuffer)}, {"space", unsigned(access->scope)},
                            {"roots", std::move(roots)}, {"origins_complete", complete}, {"id", record.id}, {"cells", std::move(cellIds)},
                            {"offsets", std::move(ranges)}, {"bytes", std::to_string(access->allocateSize)},
                            {"unknown_range", access->aliasesUnknownRange},
                            {"physical_addresses", access->hasKnownPhysicalAddresses}};
                        auto found = gmRanges.find(access);
                        if (found != gmRanges.end() && found->second) {
                            row["gm_lower"] = std::to_string(found->second->lower);
                            row["gm_upper"] = std::to_string(found->second->upper);
                        }
                        accesses.push_back(std::move(row));
                    }
                    return true;
                };
                if (!append(phase->useVec, false) || !append(phase->defVec, true)) return;
            }
            for (unsigned site = 0; site < selected.before.size(); ++site)
                for (const auto& command : selected.before[site]) {
                    if (!charge(1)) return;
                    commands.push_back(llvm::json::Object{{"cut", site}, {"kind", unsigned(command.kind)},
                        {"source", command.first}, {"observer", command.second},
                        {"forward_key", command.forwardKey}, {"reverse_key", command.reverseKey},
                        {"loop", command.loop}, {"word", command.word},
                        {"participation", unsigned(command.participation)}});
                }
            // Legacy decisions are measured with its own forwarding mode on the
            // same original IR. They are evidence of behavior, not alias promises.
            SyncIRs legacyIR;
            Buffer2MemInfoMap legacyBuffers;
            MemoryDependentAnalyzer legacyMemory;
            // Reserve a bounded traversal before invoking the legacy translator.
            // Its original-operation scan includes scalar/view operations absent
            // from the composition tree; stop before translation if it cannot fit.
            auto preflight = inventory.function.walk([&](Operation* op) {
                return charge(1 + op->getNumOperands() + op->getNumResults()) ?
                    WalkResult::advance() : WalkResult::interrupt();
            });
            if (preflight.wasInterrupted() || !charge(recorded.size() * (1 + c::MaxCells))) return;
            {
                ScopedDiagnosticHandler diagnostics(inventory.function.getContext(), [](Diagnostic&) { return success(); });
                PTOIRTranslator legacy(legacyIR, legacyMemory, legacyBuffers, inventory.function,
                                        SyncAnalysisMode::NORMALSYNC, false);
                legacyAvailable = succeeded(legacy.Build());
            }
            for (unsigned writerId : writers) {
                const auto& writer = recorded[writerId];
                for (unsigned readerId : readers) {
                    const auto& reader = recorded[readerId];
                    if (!charge(1 + writer.cells.size())) return;
                    bool overlap = std::any_of(writer.cells.begin(), writer.cells.end(),
                        [&](unsigned cell) { return reader.cells.count(cell); });
                    if (!charge((1 + writer.loops.size()) * (1 + reader.loops.size()))) return;
                    bool loop = llvm::any_of(writer.loops, [&](Operation* owner) {
                        return llvm::is_contained(reader.loops, owner);
                    });
                    auto left = legacyBuffers.find(writer.memory->baseBuffer);
                    auto right = legacyBuffers.find(reader.memory->baseBuffer);
                    llvm::json::Object pair{{"writer", writer.id}, {"reader", reader.id},
                        {"native_overlap", overlap},
                        {"possible_relation", loop ? "same-iteration-or-backedge" : "same-invocation"}};
                    if (legacyAvailable && left != legacyBuffers.end() && right != legacyBuffers.end()) {
                        bool legacyOverlap = false;
                        for (const auto& a : left->second)
                            for (const auto& b : right->second) {
                                if (!charge(1 + uint64_t(a->baseAddresses.size()) * b->baseAddresses.size())) return;
                                legacyOverlap |= legacyMemory.MemAlias(a.get(), b.get());
                            }
                        pair["legacy_overlap"] = legacyOverlap;
                    }
                    pairs.push_back(std::move(pair));
                }
            }
        };
        gather();
        llvm::json::Array aliasPairs, disjointRootPairs;
        for (auto [first, second] : disjointPairs) {
            if (!charge(1)) break;
            aliasPairs.push_back(llvm::json::Array{first, second});
            disjointRootPairs.push_back(llvm::json::Array{
                name(inventory.function.getArgument(first)), name(inventory.function.getArgument(second))});
        }
        llvm::json::Object report{{"schema", "oahs.composition.witness.v1"},
            {"function", inventory.function.getSymName().str()},
            {"alias_contract", gm == InsertSyncGMAliasMode::DisjointArguments ? "assume-disjoint-arguments" : "may-alias"},
            {"success", selected.success}, {"reason", selected.reason},
            {"publication_failure", std::move(failure)},
            {"command_stage", "proposed-before-reconstruction"},
            {"pairwise_alias_contract", std::move(aliasPairs)},
            {"disjoint_root_pairs", std::move(disjointRootPairs)},
            {"gm_geometry_work", gmGeometryWork}, {"gm_geometry_exhausted", gmGeometryExhausted},
            {"cells", std::move(physical)}, {"nodes", std::move(operations)},
            {"accesses", std::move(accesses)}, {"commands", std::move(commands)},
            {"publication_pairs", std::move(pairs)}, {"legacy_translation_available", legacyAvailable},
            {"legacy_translation_accounting", "separate diagnostic; translator internals not charged"},
            {"families_discovered", selected.lifetimeCandidates},
            {"families_selected", selected.persistentLifetimes},
            {"families_residual", selected.residualLifetimes},
            {"retained_completion_groups", selected.retainedCompletionGroups},
            {"completion_group_retries", selected.lifetimeCompletionRetries},
            {"observation_work", selected.lifetimeObservationWork},
            {"observation_exhausted", selected.lifetimeObservationExhausted},
            {"selected_visibility_sites", selected.selectedCountsValid ?
                llvm::json::Value(selected.selectedVisibilitySites) : llvm::json::Value(nullptr)},
            {"families_rejected", selected.rejectedPersistentLifetimes},
            {"cleanup_trials", selected.lifetimeCleanupTrials}, {"cleanup_commands_removed", selected.lifetimeCleanupRemoved},
            {"cleanup_work", selected.lifetimeCleanupWork}, {"cleanup_budget_exhausted", selected.lifetimeCleanupBudgetExhausted},
            {"rejection", selected.deferredRejectionReason},
            {"report_work", reportWork}, {"report_exhausted", exhausted}};
        report["report_limit"] = ReportLimit;
        llvm::errs() << "OAHS_WITNESS " << llvm::json::Value(std::move(report)) << "\n";
    }
    bool remoteSignalSeparate(Operation* signalOp, Value signal)
    {
        auto signalRoots = traceInsertSyncGMRoots(inventory.function, signal);
        for (auto* phase : inventory.physical.phases) {
            // The signal operation's declared effect is the resource being
            // classified, not an independent local payload conflict.
            if (phase->elementOp == signalOp)
                continue;
            auto separate = [&](const auto& accesses) {
                for (const auto* access : accesses) {
                    if (access->scope != AddressSpace::GM)
                        continue;
                    const auto& payloadRoots = gmAccesses.find(access)->second;
                    if (!signalRoots.complete || !payloadRoots.complete)
                        return false;
                    for (Value signalRoot : signalRoots.arguments)
                        for (Value payloadRoot : payloadRoots.arguments) {
                            if (signalRoot == payloadRoot)
                                return false;
                            unsigned signalArg = cast<BlockArgument>(signalRoot).getArgNumber();
                            unsigned payloadArg = cast<BlockArgument>(payloadRoot).getArgNumber();
                            if (gm != InsertSyncGMAliasMode::DisjointArguments &&
                                !disjointPairs.count(
                                    {std::min(signalArg, payloadArg), std::max(signalArg, payloadArg)}))
                                return false;
                        }
                }
                return true;
            };
            if (!separate(phase->useVec) || !separate(phase->defVec))
                return false;
        }
        return true;
    }
    static bool knownRange(const BaseMemInfo* access)
    {
        return !access->aliasesUnknownRange && access->hasKnownPhysicalAddresses &&
            access->allocateSize && !access->baseAddresses.empty() &&
            access->baseAddresses.size() <= c::MaxCells &&
            llvm::all_of(access->baseAddresses, [&](uint64_t base) {
                return base <= UINT64_MAX - access->allocateSize;
            });
    }
    void partition()
    {
        auto globalCells = gmCells();
        std::map<AddressSpace, std::set<uint64_t>> cuts;
        std::set<AddressSpace> coarse, remainder;
        for (auto* p : inventory.physical.phases) {
            phases[p->elementOp].push_back(p);
            auto collect = [&](const auto& accesses) {
                for (auto* a : accesses) {
                    auto& bounds = cuts[a->scope];
                    if (a->scope == AddressSpace::GM || coarse.count(a->scope))
                        continue;
                    if (!knownRange(a)) {
                        remainder.insert(a->scope);
                        continue;
                    }
                    for (uint64_t base : a->baseAddresses) {
                        bounds.insert(base);
                        bounds.insert(base + a->allocateSize);
                        if (bounds.size() > c::MaxCells) {
                            coarse.insert(a->scope);
                            bounds.clear();
                            break;
                        }
                    }
                }
            };
            collect(p->useVec);
            collect(p->defVec);
        }
        auto size = [&](AddressSpace space, const auto& bounds) -> unsigned {
            if (space == AddressSpace::GM)
                return globalCells.size();
            return coarse.count(space) ? 1 :
                (bounds.empty() ? 0 : bounds.size() - 1) + remainder.count(space);
        };
        unsigned total = 0;
        for (const auto& [space, bounds] : cuts)
            total += size(space, bounds);
        // Deterministically coarsen the largest affected group first; retain
        // all other address spaces instead of collapsing the whole function.
        while (total > c::MaxCells) {
            auto largest = cuts.end();
            unsigned count = 1;
            for (auto it = cuts.begin(); it != cuts.end(); ++it)
                if (size(it->first, it->second) > count) {
                    largest = it;
                    count = size(it->first, it->second);
                }
            if (largest == cuts.end())
                break;
            if (largest->first == AddressSpace::GM)
                globalCells = {{AddressSpace::GM, 0, 0, true, {}}};
            else
                coarse.insert(largest->first);
            total -= count - 1;
        }
        for (const auto& [space, bounds] : cuts) {
            if (space == AddressSpace::GM) {
                for (auto& cell : globalCells)
                    cells.push_back(std::move(cell));
            } else if (coarse.count(space)) {
                cells.push_back({space, 0, 0, true});
                ++widenedSpaces;
            } else {
                for (auto it = bounds.begin(); it != bounds.end(); ++it) {
                    auto next = std::next(it);
                    if (next != bounds.end())
                        cells.push_back({space, *it, *next, false});
                }
                if (remainder.count(space))
                    cells.push_back({space, 0, 0, false, {}, true});
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
        program.fixedBefore.emplace_back();
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
    bool fixed(Operation* op, c::FixedAction& action)
    {
        if (auto barrier = dyn_cast<BarrierOp>(op)) {
            PIPE p = barrier.getPipe().getPipe();
            if (p == PIPE::PIPE_ALL) {
                action.kind = c::FixedAction::BarrierAll;
            } else {
                auto fixedLane = lane(p);
                if (!fixedLane || !program.target.barrier({program.core, static_cast<ss::Pipe>(*fixedLane)})) {
                    reason = "unsupported authored pipe barrier";
                    return false;
                }
                action.kind = c::FixedAction::Barrier;
                action.lane = *fixedLane;
            }
        } else if (auto cmo = dyn_cast<CmoCacheInvalidOp>(op)) {
            auto space = cmo.getSpace().getAddressSpace();
            if (space != AddressSpace::GM && space != AddressSpace::Zero) {
                reason = "unsupported authored cache-maintenance scope";
                return false;
            }
            action.kind = c::FixedAction::CacheMaintenance;
            action.cells.resize(program.cells);
            // Without cache-line geometry an addressed CMO is preserved but
            // cannot safely be credited to a whole abstract GM cell. This is
            // the same fail-closed boundary as the hardware reference model.
            if (!cmo.getAddr())
                for (unsigned cell = 0; cell < cells.size(); ++cell)
                    action.cells[cell] = cells[cell].space == AddressSpace::GM;
        } else if (auto fence = dyn_cast<FenceBarrierAllOp>(op)) {
            auto scope = fence.getScope().getScope();
            if (scope != FenceScope::GM && scope != FenceScope::All) {
                reason = "unsupported authored physical fence scope";
                return false;
            }
            action.kind = c::FixedAction::Fence;
        } else if (auto notify = dyn_cast<TNotifyOp>(op)) {
            if (program.core != ss::Core::AIV) {
                reason = "authored remote notify requires the qualified AIV contract";
                return false;
            }
            if (!remoteSignalSeparate(op, notify.getSignal())) {
                reason = "authored remote signal may alias local payload";
                return false;
            }
            action.kind = c::FixedAction::RemoteNotify;
        } else if (auto wait = dyn_cast<TWaitOp>(op)) {
            if (program.core != ss::Core::AIV) {
                reason = "authored remote wait requires the qualified AIV contract";
                return false;
            }
            if (!remoteSignalSeparate(op, wait.getSignal())) {
                reason = "authored remote signal may alias local payload";
                return false;
            }
            action.kind = c::FixedAction::RemoteWait;
        } else {
            reason = "authored event protocol has no qualified compositional summary";
            return false;
        }
        fixedOperations.insert(op);
        return true;
    }
    bool accessOverlaps(const Cell& cell, const BaseMemInfo* a) const
    {
        if (cell.space != a->scope) return false;
        bool overlaps = cell.whole;
        if (cell.space == AddressSpace::GM) {
            const auto& roots = gmAccesses.find(a)->second;
            if (!roots.complete) return true;
            if (cell.gmRoots.empty()) return !cell.remainder;
            const auto& range = gmRanges.find(a)->second;
            for (unsigned index = 0; index < cell.gmRoots.size(); ++index) {
                Value root = cell.gmRoots[index];
                if (!llvm::is_contained(roots.arguments, root)) continue;
                Cell::PeerRange coordinate{cell.lower, cell.upper, cell.whole, cell.remainder};
                if (index == 1 && cell.peerRange) coordinate = *cell.peerRange;
                if (coordinate.whole || !range || range->root != root) return true;
                if (!coordinate.remainder && range->lower < coordinate.upper && coordinate.lower < range->upper)
                    return true;
            }
            return false;
        } else if (!knownRange(a)) {
            overlaps = true;
        } else if (!cell.remainder && !overlaps) {
            for (uint64_t base : a->baseAddresses)
                overlaps |= base < cell.upper && cell.lower < base + a->allocateSize;
        }
        return overlaps;
    }
    bool addEffects(c::Effects& effects, unsigned source, const CompoundInstanceElement* phase,
                    c::Effects* byteEffects = nullptr)
    {
        auto access = [&](const auto& entries, bool write) {
            for (auto* a : entries)
                for (unsigned j = 0; j < cells.size(); ++j) {
                    const auto& cell = cells[j];
                    if (cell.space != a->scope)
                        continue;
                    bool overlaps = accessOverlaps(cell, a);
                    if (!overlaps)
                        continue;
                    if (write)
                        effects[j].writers |= 1u << source;
                    else
                        effects[j].readers |= 1u << source;
                    if (byteEffects) {
                        if (write)
                            (*byteEffects)[j].writers |= 1u << source;
                        else
                            (*byteEffects)[j].readers |= 1u << source;
                    }
                    // Keep the qualified exclusion in the proof obligations,
                    // but do not tell lifetime discovery that FIX writes ACC.
                    if (a->scope == AddressSpace::ACC)
                        effects[j].writers |= 1u << source;
                }
        };
        access(phase->useVec, false);
        access(phase->defVec, true);
        return true;
    }
    bool macro(c::Node& current, Operation* op,
               SmallVector<const CompoundInstanceElement*, 2> translated)
    {
        auto model = getSyncMacroModel(op);
        if (program.core != ss::Core::AIV || !model || !isa<TPutOp, TGetOp>(op) || model->phases.size() != 2 ||
            model->hiddenEvents.size() != 2 ||
            model->completionTransfers.size() != 1) {
            reason = "unsupported compositional macro contract";
            return false;
        }
        llvm::sort(translated, [](const auto* a, const auto* b) {
            return a->macroOpInstanceId < b->macroOpInstanceId;
        });
        current.kind = c::Node::Macro;
        for (unsigned i = 0; i < model->phases.size(); ++i) {
            const auto& contract = model->phases[i];
            auto source = lane(static_cast<PIPE>(contract.pipe));
            if (!source || translated[i]->macroOpInstanceId != int(contract.phaseId) ||
                translated[i]->kPipeValue != contract.pipe) {
                reason = "translated macro phase differs from its positive contract";
                return false;
            }
            c::MacroPhase phase{*source, c::Effects(program.cells)};
            addEffects(phase.effects, *source, translated[i]);
            current.macroPhases.push_back(std::move(phase));
        }
        const auto& transfer = model->completionTransfers.front();
        if (transfer.sourcePhaseId >= current.macroPhases.size() ||
            transfer.targetPhaseId >= current.macroPhases.size() ||
            transfer.sourcePhaseId >= transfer.targetPhaseId) {
            reason = "invalid ordered macro completion contract";
            return false;
        }
        current.macroTransfers.push_back(
            {transfer.sourcePhaseId, current.macroPhases[transfer.sourcePhaseId].lane,
             current.macroPhases[transfer.targetPhaseId].lane});
        // Hidden events reserve private resources. They do not themselves
        // establish a phase edge; that fact is explicit above.
        std::set<std::pair<unsigned, unsigned>> directions;
        for (const auto& hidden : model->hiddenEvents) {
            auto source = lane(static_cast<PIPE>(hidden.srcPipe));
            auto observer = lane(static_cast<PIPE>(hidden.dstPipe));
            if (!source || !observer || hidden.eventIds.empty() ||
                !program.target.event({program.core, static_cast<ss::Pipe>(*source)},
                                      {program.core, static_cast<ss::Pipe>(*observer)})) {
                reason = "unsupported hidden macro event contract";
                return false;
            }
            unsigned after = *source == current.macroPhases[0].lane ? 0 :
                             *source == current.macroPhases[1].lane ? 1 : ~0u;
            if (after == ~0u || !directions.insert({*source, *observer}).second) {
                reason = "ambiguous hidden macro event contract";
                return false;
            }
            for (unsigned eventKey : hidden.eventIds) {
                if (!llvm::is_contained(program.target.compilerKeys, eventKey)) {
                    reason = "hidden macro event key is outside the qualified pool";
                    return false;
                }
                ss::Reservation reservation{{program.core, static_cast<ss::Pipe>(*source)},
                                            {program.core, static_cast<ss::Pipe>(*observer)}, eventKey};
                if (std::find_if(program.target.reservations.begin(), program.target.reservations.end(),
                                 [&](const auto& old) {
                                     return old.source == reservation.source && old.target == reservation.target &&
                                            old.key == reservation.key;
                                 }) == program.target.reservations.end())
                    program.target.reservations.push_back(reservation);
            }
        }
        if (!directions.count({current.macroPhases[0].lane, current.macroPhases[1].lane}) ||
            !directions.count({current.macroPhases[1].lane, current.macroPhases[0].lane})) {
            reason = "P2P macro requires a bidirectional hidden event contract";
            return false;
        }
        return true;
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
        std::vector<c::FixedAction> pendingFixed;
        for (Operation& op : body.front()) {
            if (ignored && ignored->contains(&op))
                continue;
            if (sync(&op)) {
                c::FixedAction action;
                if (!fixed(&op, action))
                    return false;
                if (preserveFixedCuts) {
                    // Keep an insertion cut BEFORE the immutable action, then
                    // its own transfer node. Arbitrary authored event/CMO/fence
                    // order must not be regrouped at the next payload node.
                    unsigned cut = add(node(c::Node::Sequence), &op);
                    program.nodes[cut].notificationPrerequisite = action.kind == c::FixedAction::RemoteNotify;
                    sequence.children.push_back(cut);
                    unsigned fixedId = add(node(c::Node::Sequence));
                    program.fixedBefore[fixedId].push_back(std::move(action));
                    sequence.children.push_back(fixedId);
                } else
                    pendingFixed.push_back(std::move(action));
                continue;
            }
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
                if (found->second.size() > 1) {
                    if (!macro(current, &op, found->second))
                        return false;
                } else {
                    auto* phase = found->second.front();
                    auto source = lane(static_cast<PIPE>(phase->kPipeValue));
                    if (!source) {
                        reason = "unsupported physical lane";
                        return false;
                    }
                    current.kind = c::Node::Operation;
                    current.lane = *source;
                    current.byteEffects.resize(program.cells);
                    addEffects(current.effects, *source, phase, &current.byteEffects);
                    if (program.target.ownershipCredit && unitFlags) {
                        auto ownership = unitFlags->find(&op);
                        if (ownership != unitFlags->end())
                            for (unsigned cell = 0; cell < cells.size(); ++cell)
                                if (cells[cell].space == AddressSpace::ACC && !cells[cell].whole &&
                                    cells[cell].lower == ownership->second.base &&
                                    cells[cell].upper - cells[cell].lower == ownership->second.bytes) {
                                    // Only the exact ACC ownership edge is
                                    // discharged. LEFT/RIGHT operands and the
                                    // store's GM write remain in current.effects.
                                    current.effects[cell].readers &= ~(1u << *source);
                                    current.effects[cell].writers &= ~(1u << *source);
                                    // Ownership already handles this exact cell;
                                    // do not propose a redundant event lifetime.
                                    current.byteEffects[cell].readers &= ~(1u << *source);
                                    current.byteEffects[cell].writers &= ~(1u << *source);
                                }
                    }
                    if (program.core == ss::Core::AIC) {
                        auto matrix = ss::matrixLoweringFacts(*phase);
                        if (ss::qualifiedAccumulatorInfo(matrix))
                            for (unsigned cell = 0; cell < cells.size(); ++cell)
                                if (cells[cell].space == AddressSpace::ACC && !cells[cell].whole &&
                                    cells[cell].lower == matrix.accumulatorBase &&
                                    cells[cell].upper - cells[cell].lower == matrix.accumulatorBytes) {
                                    current.matrix = matrix;
                                    current.matrixCell = cell;
                                }
                    }
                }
            }
            unsigned child = add(std::move(current), &op);
            program.fixedBefore[child] = std::move(pendingFixed);
            pendingFixed.clear();
            sequence.children.push_back(child);
        }
        if (!pendingFixed.empty()) {
            unsigned child = add(node(c::Node::Sequence));
            program.fixedBefore[child] = std::move(pendingFixed);
            sequence.children.push_back(child);
        }
        id = add(std::move(sequence));
        return true;
    }
    bool build(bool periodicPrecision, bool wholeFunction = false)
    {
        preserveFixedCuts = wholeFunction;
        inventory.function.walk([&](TNotifyOp) { preserveFixedCuts = true; });
        if (!gmContracts() || !scalarContracts())
            return false;
        partition();
        unsigned root;
        auto* scope = inventory.physical.lifetimeScope;
        if (!region(
                wholeFunction || scope == inventory.function.getOperation() ? inventory.function.getBody() : scope->getRegion(0), root))
            return false;
        if (periodicPrecision) {
            unsigned tail = add(node(c::Node::Sequence));
            program.nodes[root].children.push_back(tail);
            // Program consumers deliberately use the final postorder node as
            // the root. Keep that invariant while placing the terminal leaf
            // immediately before it in the numbering.
            std::swap(program.nodes[root], program.nodes[tail]);
            std::swap(program.fixedBefore[root], program.fixedBefore[tail]);
            std::swap(anchors[root], anchors[tail]);
            for (auto& current : program.nodes)
                for (unsigned& child : current.children)
                    if (child == root)
                        child = tail;
                    else if (child == tail)
                        child = root;
            for (auto& [operation, id] : ids)
                if (id == root)
                    id = tail;
                else if (id == tail)
                    id = root;
            terminal = root;
            root = tail;
        }
        qualifyNonEmptyLoops();
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
                if (loop->hasAttr("unsignedCmp") || !loop.getInductionVar().getType().isIndex() ||
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
                if (child.kind == c::Node::For || child.kind == c::Node::While)
                    return; // Its induction domain is qualified independently.
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
            std::vector<unsigned> periodicChoices;
            for (unsigned id : choices) {
                Value condition = cast<scf::IfOp>(anchors[id]).getCondition();
                if (!depth(condition, 0))
                    continue;
                auto p = scalar.period(condition);
                if (!p || !*p || *p > 32)
                    continue;
                uint64_t combined = period / std::gcd(period, *p) * *p;
                if (combined > 32)
                    continue;
                period = combined;
                periodicChoices.push_back(id);
            }
            uint64_t evaluation = nodes.size() + periodicChoices.size() * depths.size();
            if (periodicChoices.empty() || evaluation > (MaxWork - std::min(work, MaxWork)) / period)
                continue;
            work += evaluation * period;
            std::map<unsigned, uint32_t> truth;
            for (unsigned id : periodicChoices) {
                Value condition = cast<scf::IfOp>(anchors[id]).getCondition();
                uint32_t mask = 0;
                bool exact = true;
                for (unsigned ordinal = 0; ordinal < period; ++ordinal) {
                    ++periodicEvaluations;
                    auto value = scalar.evaluate(condition, ordinal);
                    if (!value) {
                        exact = false;
                        break;
                    }
                    if (*value)
                        mask |= uint32_t(1) << ordinal;
                }
                if (exact)
                    truth[id] = mask;
            }
            uint32_t all = period == 32 ? UINT32_MAX : (uint32_t(1) << period) - 1;
            std::function<void(unsigned, std::optional<uint32_t>)> record =
                [&](unsigned id, std::optional<uint32_t> active) {
                auto& child = program.nodes[id];
                if (child.kind == c::Node::For || child.kind == c::Node::While)
                    return;
                auto known = truth.find(id);
                if ((child.kind == c::Node::Sequence && active) ||
                    (child.kind == c::Node::Choice && known != truth.end())) {
                    child.periodicOwner = owner;
                    child.periodicPeriod = unsigned(period);
                    child.periodicLower = induction.lower;
                    child.periodicResidues = child.kind == c::Node::Choice ? known->second : *active;
                }
                if (child.kind == c::Node::Choice) {
                    // An unknown condition retains both successors. Its child
                    // participation is unknown even if a sibling's parity is exact.
                    record(child.children[0], active && known != truth.end() ?
                        std::optional<uint32_t>(*active & known->second) : std::nullopt);
                    record(child.children[1], active && known != truth.end() ?
                        std::optional<uint32_t>(*active & ~known->second) : std::nullopt);
                } else
                    for (unsigned next : child.children)
                        record(next, active);
            };
            record(n.children[0], all);
        }
        // Qualify the separate next-iteration predicate directly from the
        // original SSA. Failure here does not erase an already established
        // residue fact, and residue failure does not affect this fact.
        for (unsigned id = 0; id < program.nodes.size(); ++id) {
            auto& node = program.nodes[id];
            if (node.kind != c::Node::Choice || !anchors[id])
                continue;
            auto choice = dyn_cast<scf::IfOp>(anchors[id]);
            auto loop = choice ? choice->getParentOfType<scf::ForOp>() : scf::ForOp{};
            if (!loop || loop->hasAttr("unsignedCmp") || ids.find(loop.getOperation()) == ids.end())
                continue;
            APInt step;
            if (!matchPattern(loop.getStep(), m_ConstantInt(&step)) || step != 1)
                continue;
            auto compare = choice.getCondition().getDefiningOp<arith::CmpIOp>();
            if (!compare || compare.getPredicate() != arith::CmpIPredicate::slt ||
                compare.getRhs() != loop.getUpperBound())
                continue;
            auto add = compare.getLhs().getDefiningOp<arith::AddIOp>();
            if (!add || !((add.getLhs() == loop.getInductionVar() && add.getRhs() == loop.getStep()) ||
                          (add.getRhs() == loop.getInductionVar() && add.getLhs() == loop.getStep())))
                continue;
            node.nextIterationOwner = ids.lookup(loop.getOperation());
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
            if (m.kind == c::Mechanism::Visibility) {
                auto cmo = [&]() {
                    b.create<CmoCacheInvalidOp>(
                        loc, Value(), AddressSpaceAttr::get(context, AddressSpace::GM));
                };
                auto fence = [&]() {
                    b.create<FenceBarrierAllOp>(loc, FenceScopeAttr::get(context, FenceScope::GM));
                };
                if (m.visibilityAction == c::VisibilityAction::CleanSource) {
                    cmo();
                    fence();
                } else if (m.visibilityAction == c::VisibilityAction::InvalidateTarget) {
                    fence();
                    cmo();
                } else
                    fence();
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
            Operation* terminal = original.terminal == ~0u ? nullptr : original.anchors[original.terminal];
            Operation* terminalFallback = nullptr;
            while (from && !original.ids.count(from)) {
                if (from == terminal)
                    terminalFallback = from;
                from = from->getNextNode();
            }
            return from ? from : terminalFallback;
        };
        auto targetCut = [&](Operation* target) {
            if (original.terminal != ~0u && target == original.anchors[original.terminal])
                return original.terminal;
            return original.ids.lookup(target);
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
                unsigned cut = targetCut(targets[branch]);
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
                unsigned cut = targetCut(targets[branch]);
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
                    unsigned cut = targetCut(targets[branch]);
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
                    unsigned cut = targetCut(targets[branch]);
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
bool parsePacket(Operation*& cursor, c::Mechanism& m, const c::Program& program,
                 const llvm::SmallPtrSetImpl<Operation*>& fixedOperations,
                 bool precision)
{
    // The outer reconstruction loop skips authored operations when they begin
    // a run. Enforce the same boundary after parsing a preceding generated
    // command so a contiguous run cannot consume an authored barrier/fence.
    if (fixedOperations.contains(cursor))
        return false;
    if (auto barrier = dyn_cast<BarrierOp>(cursor)) {
        auto p = lane(barrier.getPipe().getPipe());
        if (!p)
            return false;
        m = {c::Mechanism::Barrier, *p, *p};
        cursor = cursor->getNextNode();
        return true;
    }
    auto wholeGmCmo = [](Operation* op) {
        auto cmo = dyn_cast_or_null<CmoCacheInvalidOp>(op);
        return cmo && !cmo.getAddr() && cmo.getSpace().getAddressSpace() == AddressSpace::GM;
    };
    auto gmFence = [](Operation* op) {
        auto fence = dyn_cast_or_null<FenceBarrierAllOp>(op);
        return fence && fence.getScope().getScope() == FenceScope::GM;
    };
    Operation* next = cursor->getNextNode();
    if (!fixedOperations.contains(cursor) && !fixedOperations.contains(next) &&
        ((wholeGmCmo(cursor) && gmFence(next)) || (gmFence(cursor) && wholeGmCmo(next)))) {
        m = {};
        m.kind = c::Mechanism::Visibility;
        m.visibilityAction = wholeGmCmo(cursor) ? c::VisibilityAction::CleanSource :
                                                 c::VisibilityAction::InvalidateTarget;
        cursor = next->getNextNode();
        return true;
    }
    if (gmFence(cursor)) {
        m = {};
        m.kind = c::Mechanism::Visibility;
        m.visibilityAction = c::VisibilityAction::FenceOnly;
        cursor = cursor->getNextNode();
        return true;
    }
    // A canonical key is available to an independent protocol when the whole
    // bidirectional population fits without fallback. Identify fallback by
    // its actual adjacent four-command word, not by reserving one magic key
    // from otherwise legal reconstructed events.
    if (!precision)
        if (auto s = dyn_cast<SetFlagOp>(cursor)) {
            auto* n = cursor->getNextNode();
            auto w = dyn_cast_or_null<WaitFlagOp>(n);
            n = n ? n->getNextNode() : nullptr;
            auto reply = dyn_cast_or_null<SetFlagOp>(n);
            n = n ? n->getNextNode() : nullptr;
            auto ack = dyn_cast_or_null<WaitFlagOp>(n);
            auto a = lane(s.getSrcPipe().getPipe()), b = lane(s.getDstPipe().getPipe());
            auto canonical = [&](unsigned source, unsigned observer) -> std::optional<unsigned> {
                for (unsigned eventKey : program.target.compilerKeys)
                    if (program.target.available(
                            {program.core, static_cast<ss::Pipe>(source)},
                            {program.core, static_cast<ss::Pipe>(observer)}, eventKey))
                        return eventKey;
                return {};
            };
            auto forward = a && b ? canonical(*a, *b) : std::optional<unsigned>{};
            auto reverse = a && b ? canonical(*b, *a) : std::optional<unsigned>{};
            if (w && reply && ack && a && b && *a < *b && forward && reverse &&
                unsigned(s.getEventId().getEvent()) == *forward &&
                unsigned(reply.getEventId().getEvent()) == *reverse && w.getSrcPipe() == s.getSrcPipe() &&
                w.getDstPipe() == s.getDstPipe() && w.getEventId() == s.getEventId() &&
                reply.getSrcPipe() == s.getDstPipe() && reply.getDstPipe() == s.getSrcPipe() &&
                ack.getSrcPipe() == reply.getSrcPipe() && ack.getDstPipe() == reply.getDstPipe() &&
                ack.getEventId() == reply.getEventId()) {
                m = {c::Mechanism::Rendezvous, *a, *b, unsigned(s.getEventId().getEvent()),
                     unsigned(reply.getEventId().getEvent())};
                cursor = ack->getNextNode();
                return true;
            }
        }
    auto independent = [&](auto op, c::Mechanism::Kind kind) {
        auto a = lane(op.getSrcPipe().getPipe()), b = lane(op.getDstPipe().getPipe());
        if (!a || !b)
            return false;
        unsigned eventKey = unsigned(op.getEventId().getEvent());
        if (std::find(program.target.compilerKeys.begin(), program.target.compilerKeys.end(), eventKey) ==
                program.target.compilerKeys.end() ||
            !program.target.available(
                {program.core, static_cast<ss::Pipe>(*a)}, {program.core, static_cast<ss::Pipe>(*b)}, eventKey))
            return false;
        m = {kind, *a, *b, eventKey, 0};
        cursor = cursor->getNextNode();
        return true;
    };
    if (auto s = dyn_cast<SetFlagOp>(cursor))
        if (independent(s, c::Mechanism::Publish))
            return true;
    if (auto w = dyn_cast<WaitFlagOp>(cursor))
        if (independent(w, c::Mechanism::Acquire))
            return true;
    return false;
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
                        Operation* target = guards.targets.lookup(cursor);
                        unsigned cut = ~0u;
                        if (target == drain && tree.terminal != ~0u)
                            cut = tree.terminal;
                        else if (auto found = tree.ids.find(target); found != tree.ids.end())
                            cut = found->second;
                        if (cut == ~0u || cut >= actual.size()) {
                            valid = false;
                            break;
                        }
                        auto& out = actual[cut];
                        out.insert(out.end(), guard->second.begin(), guard->second.end());
                        cursor = cursor->getNextNode();
                        continue;
                    }
                    if (tree.fixedOperations.contains(cursor)) {
                        cursor = cursor->getNextNode();
                        continue;
                    }
                    if (!sync(cursor) || cursor == drain) {
                        cursor = cursor->getNextNode();
                        continue;
                    }
                    std::vector<c::Mechanism> packets;
                    while (cursor && sync(cursor) && cursor != drain &&
                           (!tree.preserveFixedCuts || !tree.fixedOperations.contains(cursor))) {
                        c::Mechanism m;
                        if (!parsePacket(cursor, m, tree.program, tree.fixedOperations, precision)) {
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
                    if (cursor == drain && tree.terminal != ~0u) {
                        auto& out = actual[tree.terminal];
                        out.insert(out.end(), packets.begin(), packets.end());
                        cursor = cursor->getNextNode();
                        continue;
                    }
                    if (cursor == drain && !tree.ids.count(cursor))
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

Outcome ss::testing::verifyAuthoredCompositionalSync(
    func::FuncOp function, InsertSyncGMAliasMode gm, HardwareContract hardware, OwnershipContract ownership,
    bool ownershipCredit)
{
    Outcome out;
    if (function.isDeclaration() || !llvm::hasSingleElement(function.getBody()) ||
        (hardware != HardwareContract::Conservative && hardware != HardwareContract::A2A3MmadAccV1) ||
        (ownership != OwnershipContract::None && ownership != OwnershipContract::A2A3UnitFlagPairedV1)) {
        out.reason = "invalid authored verification function or hardware contract";
        return out;
    }
    Inventory inventory(function);
    if (!inventory.build(true, true)) {
        out.reason = inventory.reason;
        return out;
    }
    UnitFlagMap unitFlags;
    if (!validateUnitFlagOwnership(inventory, ownership, unitFlags, out.reason))
        return out;
    Operation* scope = inventory.physical.lifetimeScope;
    Block& block = scope == function.getOperation() ? function.getBody().front() : scope->getRegion(0).front();
    Operation* end = block.empty() ? nullptr : &block.back();
    if (end && end->hasTrait<OpTrait::IsTerminator>())
        end = end->getPrevNode();
    auto drain = dyn_cast_or_null<BarrierOp>(end);
    if (!drain || drain.getPipe().getPipe() != PIPE::PIPE_ALL ||
        drain->hasAttr("pto.auto_sync_tail_barrier") || drain->hasAttr("pto.auto_sync_tail_hint")) {
        out.reason = "authored plan requires an unconditional physical-context retirement drain";
        return out;
    }
    // Authored priming/cleanup may occur outside the physical section. The
    // whole-function tree below must not credit this drain with retiring a
    // later issuing command, even when it is final inside its own section.
    bool passedDrain = false, trailingSync = false;
    function.walk<WalkOrder::PreOrder>([&](Operation* op) {
        if (op == drain.getOperation()) passedDrain = true;
        else if (passedDrain && sync(op)) trailingSync = true;
    });
    Tree tree(inventory, hardware, gm, ownership, ownershipCredit, &unitFlags);
    llvm::SmallPtrSet<Operation*, 32> commands;
    function.walk([&](Operation* op) {
        if (isa<SetFlagOp, WaitFlagOp>(op) ||
            (isa<BarrierOp>(op) && cast<BarrierOp>(op).getPipe().getPipe() != PIPE::PIPE_ALL))
            commands.insert(op);
    });
    tree.ignored = &commands;
    // Include the containing function: priming and cleanup outside a physical
    // section must participate in the same protocol verification population.
    if (!tree.build(true, true)) {
        out.reason = tree.reason;
        return out;
    }
    // Such authored plans can still prove explicit return acknowledgments;
    // they simply receive no end-of-invocation retirement credit.
    tree.program.terminalRetirementCut = trailingSync ? ~0u : tree.terminal;
    std::vector<std::vector<c::Mechanism>> actual;
    ParticipationGuards guards;
    if (!reconstruct(tree, drain, actual, out.reason, false, guards))
        return out;
    auto checked = c::testing::verifyOpenDemands(tree.program, actual);
    if (!checked.success) {
        out.reason = trailingSync ? "authored synchronization follows the physical-context retirement drain: " +
                                        checked.reason : checked.reason;
        return out;
    }
    out.status = Outcome::Applied;
    out.fixedSync = checked.fixedActions;
    out.work = checked.nodeVisits + checked.cellVisits;
    for (const auto& site : actual)
        for (const auto& m : site) {
            out.barriers += m.kind == c::Mechanism::Barrier;
            out.handoffs += m.kind == c::Mechanism::Publish ? 1 : m.kind == c::Mechanism::Rendezvous ? 2 : 0;
        }
    return out;
}

Outcome ss::constructCompositionalSync(
    func::FuncOp function, InsertSyncGMAliasMode gm, HardwareContract hardware, bool enablePrecision,
    OwnershipContract ownership, bool ownershipCredit)
{
    return testing::constructCompositionalSync(
        function, gm, {}, hardware,
        enablePrecision ? testing::CompositionConstructor::Demands : testing::CompositionConstructor::Conservative,
        ownership, ownershipCredit);
}

Outcome ss::testing::constructCompositionalSync(
    func::FuncOp function, InsertSyncGMAliasMode gm, llvm::function_ref<void(func::FuncOp)> mutate,
    HardwareContract hardware, CompositionConstructor constructor, OwnershipContract ownership,
    bool ownershipCredit)
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
    const bool withoutStructuredRings = constructor == CompositionConstructor::DemandsWithoutStructuredRings;
    const bool withoutChoiceOrStructuredRings =
        constructor == CompositionConstructor::DemandsWithoutChoiceOrStructuredRings;
    const bool rejectChoiceWithoutStructuredRings =
        constructor == CompositionConstructor::DemandsRejectChoiceWithoutStructuredRings;
    const bool withoutChildReturns = constructor == CompositionConstructor::DemandsWithoutChildReturns;
    const bool rejectChildReturns = constructor == CompositionConstructor::DemandsRejectChildReturns;
    const bool withoutAlternatives = constructor == CompositionConstructor::DemandsWithoutAlternativeChoices;
    const bool rejectAlternatives = constructor == CompositionConstructor::DemandsRejectAlternativeChoices;
    const bool demandPlacement = constructor == CompositionConstructor::Demands || rejectRefinement || fallbackOnly ||
                                 withoutReplay || rejectReplay || rejectEntry || rejectDeferred || withoutLateEntry ||
                                 rejectLateEntry || withoutChoice || rejectChoice || withoutStructuredRings ||
                                 withoutChoiceOrStructuredRings || rejectChoiceWithoutStructuredRings ||
                                 withoutChildReturns || rejectChildReturns || withoutAlternatives || rejectAlternatives;
    Outcome out;
    if (function.isDeclaration() || !llvm::hasSingleElement(function.getBody())) {
        out.reason = "composition requires a single function block";
        return out;
    }
    if (hardware != HardwareContract::Conservative && hardware != HardwareContract::A2A3MmadAccV1) {
        out.reason = "unknown structured hardware contract";
        return out;
    }
    if (ownership != OwnershipContract::None && ownership != OwnershipContract::A2A3UnitFlagPairedV1) {
        out.reason = "unknown structured ownership contract";
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
    UnitFlagMap unitFlags;
    if (!validateUnitFlagOwnership(inventory, ownership, unitFlags, out.reason))
        return out;
    Tree tree(inventory, hardware, gm, ownership, ownershipCredit, &unitFlags);
    if (!tree.build(demandPlacement)) {
        out.reason = tree.reason;
        return out;
    }
    // Construction promises a real final ALL. The fresh tree receives this
    // premise only after the emitted drain's presence and position are checked.
    tree.program.terminalRetirementCut = tree.terminal;
    // Demand construction executes macros atomically and reserves their
    // private keys globally. The older cut-only API still declines macros.
    const bool hasMacro = llvm::any_of(tree.program.nodes, [](const auto& node) {
        return node.kind == c::Node::Macro;
    });
    const bool useDemandPlacement = demandPlacement;
    const bool useCutPrecision = precision && !hasMacro;
    if (fallbackOnly)
        tree.program.target.compilerKeys = {0};
    auto selected = hasMacro && !demandPlacement ? c::construct(tree.program) :
                    withoutAlternatives ? c::testing::constructDemandsWithoutAlternativeChoices(tree.program) :
                    rejectAlternatives  ? c::testing::constructDemandsRejectingAlternativeChoices(tree.program) :
                    withoutChildReturns ? c::testing::constructDemandsWithoutChildReturns(tree.program) :
                    rejectChildReturns  ? c::testing::constructDemandsRejectingChildReturns(tree.program) :
                    withoutChoice       ? c::testing::constructDemandsWithoutChoiceDemands(tree.program) :
                    rejectChoice        ? c::testing::constructDemandsRejectingChoiceDemands(tree.program) :
                    withoutStructuredRings ? c::testing::constructDemandsWithoutStructuredRings(tree.program) :
                    withoutChoiceOrStructuredRings ?
                        c::testing::constructDemandsWithoutChoiceOrStructuredRings(tree.program) :
                    rejectChoiceWithoutStructuredRings ?
                        c::testing::constructDemandsRejectingChoiceWithoutStructuredRings(tree.program) :
                    withoutLateEntry    ? c::testing::constructDemandsWithoutLateEntry(tree.program) :
                    rejectLateEntry     ? c::testing::constructDemandsRejectingLateEntry(tree.program) :
                    rejectDeferred      ? c::testing::constructDemandsRejectingDeferredRings(tree.program) :
                    rejectEntry         ? c::testing::constructDemandsRejectingEntryProposal(tree.program) :
                    withoutReplay       ? c::testing::constructDemandsWithoutAllocationReplay(tree.program) :
                    rejectReplay        ? c::testing::constructDemandsRejectingAllocationReplay(tree.program) :
                    rejectRefinement    ? c::testing::constructDemandsRejectingRefinement(tree.program) :
                    fallbackOnly        ? c::testing::constructDemandsWithoutPersistentLifetimes(tree.program) :
                    useDemandPlacement  ? c::constructDemands(tree.program) :
                    useCutPrecision     ? c::constructCuts(tree.program) :
                                          c::construct(tree.program);
    tree.reportWitnesses(selected);
    if (!selected.success) {
        out.reason = selected.reason;
        return out;
    }
    SyncPayloadSnapshot snapshot(working);
    llvm::SmallPtrSet<Operation*, 32> originalOperations, generatedOperations;
    working.walk([&](Operation* op) { originalOperations.insert(op); });
    Operation* scope = inventory.physical.lifetimeScope;
    Block& last = scope == working.getOperation() ? working.getBody().front() : scope->getRegion(0).front();
    Operation* terminator = !last.empty() && last.back().hasTrait<OpTrait::IsTerminator>() ? &last.back() : nullptr;
    OpBuilder b(working.getContext());
    if (terminator)
        b.setInsertionPoint(terminator);
    else
        b.setInsertionPointToEnd(&last);
    auto retirement = b.create<BarrierOp>(working.getLoc(), PipeAttr::get(working.getContext(), PIPE::PIPE_ALL));
    // The synthetic terminal is a real original-control cut even though its
    // native anchor is generated. Participation-guard recovery must be able
    // to bind loop-exit packets when the loop is the final payload operation.
    if (tree.terminal != ~0u)
        tree.anchors[tree.terminal] = retirement.getOperation();
    emit(tree, selected);
    // Other exit-cut packets can share the original terminator anchor.  Emit
    // everything first, then restore the retirement drain as the final command
    // in the physical context; terminal packets remain immediately before it.
    if (terminator)
        retirement->moveBefore(terminator);
    else
        retirement->moveBefore(&last, last.end());
    working.walk([&](Operation* op) {
        if (!originalOperations.contains(op))
            generatedOperations.insert(op);
    });
    if (mutate)
        mutate(working);
    out.status = Outcome::InternalError;
    ParticipationGuards guards;
    if (failed(mlir::verify(working))) {
        out.reason = "compositional emission produced invalid IR";
        return out;
    }
    if (useDemandPlacement && !guards.build(tree, generatedOperations, out.reason))
        return out;
    if (!snapshot.preserved(working, [&](Operation* op) {
            return generatedOperations.contains(op) && (sync(op) || guards.operations.contains(op));
        })) {
        out.reason = "compositional emission changed original payload or control";
        return out;
    }
    Operation* drain = nullptr;
    unsigned drains = 0;
    working.walk([&](BarrierOp barrier) {
        if (generatedOperations.contains(barrier.getOperation()) &&
            barrier.getPipe().getPipe() == PIPE::PIPE_ALL) {
            drain = barrier;
            ++drains;
        }
    });
    if (drains != 1) {
        out.reason = "one explicit unconditional physical-context retirement drain is required: emitted " +
                     std::to_string(drains);
        return out;
    }
    if (drain->getBlock() != &last || drain->getNextNode() != terminator) {
        out.reason = "the retirement drain must be the final command in its physical context (block=" +
                     std::string(drain->getBlock() == &last ? "expected" : "other") + ", next=" +
                     (drain->getNextNode() ? drain->getNextNode()->getName().getStringRef().str() : "none") +
                     ", terminator=" +
                     (terminator ? terminator->getName().getStringRef().str() : "none") + ")";
        return out;
    }
    if (drain->hasAttr("pto.auto_sync_tail_barrier") || drain->hasAttr("pto.auto_sync_tail_hint")) {
        out.reason = "the retirement drain cannot be an inferred tail barrier";
        return out;
    }
    Inventory fresh(working);
    if (!fresh.build(true)) {
        out.reason = fresh.reason;
        return out;
    }
    UnitFlagMap rebuiltUnitFlags;
    if (!validateUnitFlagOwnership(fresh, ownership, rebuiltUnitFlags, out.reason))
        return out;
    Tree rebuilt(fresh, hardware, gm, ownership, ownershipCredit, &rebuiltUnitFlags);
    llvm::SmallPtrSet<Operation*, 32> rebuiltIgnored;
    rebuiltIgnored.insert(generatedOperations.begin(), generatedOperations.end());
    rebuiltIgnored.insert(guards.operations.begin(), guards.operations.end());
    rebuilt.ignored = &rebuiltIgnored;
    if (!rebuilt.build(demandPlacement)) {
        out.reason = rebuilt.reason;
        return out;
    }
    if (rebuilt.terminal != ~0u)
        rebuilt.anchors[rebuilt.terminal] = drain;
    rebuilt.program.terminalRetirementCut = rebuilt.terminal;
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
    if (!reconstruct(rebuilt, drain, actual, out.reason, useCutPrecision, guards))
        return out;
    auto checked = useDemandPlacement ? c::verifyDemands(rebuilt.program, actual) :
                   useCutPrecision    ? c::verifyCuts(rebuilt.program, actual) :
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
            else if (m.kind == c::Mechanism::Visibility)
                ++out.visibility;
            else if (m.kind == c::Mechanism::Rendezvous) {
                ++rendezvousPackets;
                out.handoffs += 2;
            } else if (m.kind == c::Mechanism::Publish)
                ++out.handoffs;
        }
    out.requirements = selected.acquisitions + selected.visibilityRequirements;
    out.fixedSync = selected.fixedActions;
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
            << "structured composition precision " << useCutPrecision << " demands " << useDemandPlacement
            << " ownership_contract "
            << (ownership == OwnershipContract::None ? "none" : "a2a3-unitflag-paired-v1")
            << " ownership_credit " << ownershipCredit
            << " macro_fallback " << (precision && hasMacro) << " direct_handoffs "
            << selected.directHandoffs << " visibility_requirements " << selected.visibilityRequirements
            << " visibility_actions " << out.visibility << " fixed_sync " << out.fixedSync
            << " shared_acknowledgments " << selected.sharedAcknowledgments
            << " reused_acknowledgments " << selected.reusedAcknowledgments << " completion_refinements "
            << selected.completionRefinements << " rejected_refinements " << selected.rejectedRefinements
            << " owned_refinements " << checked.ownedRefinements << " protocol_keys " << selected.protocolKeys
            << " shared_protocol_keys " << selected.sharedProtocolKeys << " allocation_fallback_keys "
            << selected.allocationFallbackKeys << " allocation_replays " << selected.allocationReplays
            << " dedicated_allocation_domains " << selected.dedicatedAllocationDomains
            << " dedicated_allocation_keys " << selected.dedicatedAllocationKeys
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
            << " child_return_candidates " << selected.childReturnCandidates << " child_return_acks_removed "
            << selected.childReturnAcksRemoved << " rejected_child_returns " << selected.rejectedChildReturns
            << " child_return_work " << selected.childReturnWork << " child_return_checks "
            << selected.childReturnChecks << " child_return_budget_check " << selected.childReturnBudgetCheck
            << " child_return_budget_exhausted " << selected.childReturnBudgetExhausted
            << " alternative_choice_candidates " << selected.alternativeChoiceCandidates
            << " alternative_choice_families " << selected.alternativeChoiceFamilies << " alternative_choice_sites "
            << selected.alternativeChoiceSites << " alternative_choice_sets_removed "
            << selected.alternativeChoiceSetsRemoved << " alternative_choice_source_scopes "
            << selected.alternativeChoiceSourceScopes << " alternative_choice_prefix_steps "
            << selected.alternativeChoicePrefixSteps << " rejected_alternative_choices "
            << selected.rejectedAlternativeChoices << " alternative_choice_work " << selected.alternativeChoiceWork
            << " alternative_choice_budget_exhausted " << selected.alternativeChoiceBudgetExhausted
            << " alternative_choice_continuation_demands " << selected.alternativeChoiceContinuationDemands
            << " alternative_choice_cost_rejections " << selected.alternativeChoiceCostRejections
            << " ring_candidates " << selected.ringCandidates << " rejected_rings " << selected.rejectedRings
            << " ring_candidate_commands_removed " << selected.ringCandidateCommandsRemoved
            << " ring_candidate_packets_removed " << selected.ringCandidatePacketsRemoved
            << " recurring_episode_words " << selected.recurringEpisodeWords << " recurring_episode_pairs "
            << selected.recurringEpisodePairs << " recurring_choice_endpoints " << selected.recurringChoiceEndpoints
            << " recurring_repeated_directions " << selected.recurringRepeatedDirections
            << " recurring_episode_work " << selected.recurringEpisodeWork << " rejected_recurring_episodes "
            << selected.rejectedRecurringEpisodes << " recurring_episode_budget_exhausted "
            << selected.recurringEpisodeBudgetExhausted << " rendezvous_packets " << rendezvousPackets
            << " deferred_ring_candidates " << selected.deferredRingCandidates
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
            << selected.deferredProtocolSteps + checked.deferredProtocolSteps << " lifetime_analysis_work "
            << selected.lifetimeAnalysisWork
            << " lifetime_storage_units " << selected.lifetimeStorageUnits << " lifetime_candidates "
            << selected.lifetimeCandidates << " persistent_lifetimes " << selected.persistentLifetimes
            << " persistent_reader_families " << selected.persistentReaderFamilies
            << " pre_lifetime_cut_cycles " << selected.preLifetimeCutCycles
            << " pre_lifetime_protocol_keys " << selected.preLifetimeProtocolKeys
            << " selected_counts_valid " << selected.selectedCountsValid
            << " selected_set_sites " << selected.selectedSetSites
            << " selected_wait_sites " << selected.selectedWaitSites
            << " selected_named_barriers " << selected.selectedNamedBarriers
            << " selected_cycle_count_known " << selected.selectedCycleCountKnown
            << " selected_accounting_work " << selected.selectedAccountingWork
            << " lifetime_eligibility_work " << selected.lifetimeEligibilityWork
            << " lifetime_cleanup_trials " << selected.lifetimeCleanupTrials
            << " lifetime_cleanup_removed " << selected.lifetimeCleanupRemoved
            << " lifetime_cleanup_work " << selected.lifetimeCleanupWork
            << " lifetime_cleanup_budget_exhausted " << selected.lifetimeCleanupBudgetExhausted
            << " lifetime_stronger_rejections " << selected.lifetimeStrongerRejections
            << " lifetime_protocol_rejections " << selected.lifetimeProtocolRejections
            << " lifetime_boundary_rejections " << selected.lifetimeBoundaryRejections
            << " lifetime_allocation_rejections " << selected.lifetimeAllocationRejections
            << " lifetime_overlap_rejections " << selected.lifetimeOverlapRejections
            << " lifetime_verification_rejections " << selected.lifetimeVerificationRejections
            << " rejected_persistent_lifetimes " << selected.rejectedPersistentLifetimes
            << " lifetime_budget_exhausted " << selected.lifetimeBudgetExhausted << " demand_fallbacks "
            << selected.demandFallbacks << " nodes " << tree.program.nodes.size() << " cells " << tree.program.cells
            << " widened_spaces " << tree.widenedSpaces << " node_visits " << selected.nodeVisits + checked.nodeVisits
            << " cell_visits " << selected.cellVisits + checked.cellVisits << " handoffs " << out.handoffs
            << " cut_cycles " << selected.cutCycles << " allocation_retries " << selected.allocationRetries
            << " barriers " << out.barriers << " seconds "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << "\n";
    return out;
}
