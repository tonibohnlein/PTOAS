// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/InsertSync/HandoffFacts.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncGMAlias.h"
#include "PTO/Transforms/InsertSync/SyncEffectCoverage.h"
#include "PTO/Transforms/SlotAffineAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/Path.h"
#include <limits>

using namespace mlir;
using namespace mlir::pto;
namespace {
using Object = llvm::json::Object;
using Array = llvm::json::Array;

std::string printed(Operation* operation) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    operation->print(stream);
    return text;
}
std::string digest(StringRef text) {
    llvm::MD5 hash;
    hash.update(text);
    llvm::MD5::MD5Result result;
    hash.final(result);
    llvm::SmallString<32> value;
    llvm::MD5::stringifyResult(result, value);
    return value.str().str();
}
Object provenance(func::FuncOp function) {
    Array modules, path;
    for (Operation* op = function; op; op = op->getParentOp()) {
        if (auto module = dyn_cast<ModuleOp>(op)) {
            std::string attributes;
            llvm::raw_string_ostream stream(attributes);
            module->getAttrDictionary().print(stream);
            modules.push_back(attributes);
        }
        if (Block* block = op->getBlock()) {
            unsigned position = 0, blockPosition = 0;
            for (Operation& sibling : *block) { if (&sibling == op) break; ++position; }
            for (Block& sibling : *block->getParent()) { if (&sibling == block) break; ++blockPosition; }
            path.push_back(Object{{"operation", position}, {"block", blockPosition},
                                 {"region", block->getParent()->getRegionNumber()}});
        }
    }
    return Object{{"module_attributes_nearest_first", std::move(modules)},
                  {"operation_path_leaf_first", std::move(path)}};
}
void addProvenance(Object& record, func::FuncOp function) {
    auto context = provenance(function);
    std::string encoding;
    llvm::raw_string_ostream stream(encoding);
    stream << llvm::formatv("{0}", llvm::json::Value(std::move(context)));
    record["context_json"] = encoding;
    record["context_md5"] = digest(encoding);
}
StringRef laneName(PipelineType lane) {
    switch (lane) {
    case PipelineType::PIPE_V: return "V";
    case PipelineType::PIPE_M: return "M";
    case PipelineType::PIPE_MTE1: return "MTE1";
    case PipelineType::PIPE_MTE2: return "MTE2";
    case PipelineType::PIPE_MTE3: return "MTE3";
    case PipelineType::PIPE_FIX: return "FIX";
    default: return "unsupported";
    }
}
StringRef spaceName(AddressSpace space) {
    switch (space) {
    case AddressSpace::VEC: return "VEC";
    case AddressSpace::MAT: return "MAT";
    case AddressSpace::LEFT: return "LEFT";
    case AddressSpace::RIGHT: return "RIGHT";
    case AddressSpace::ACC: return "ACC";
    case AddressSpace::GM: return "GM";
    default: return "unsupported";
    }
}
bool constantInt(Value value, int64_t& result) {
    llvm::APInt constant;
    if (!matchPattern(value, m_ConstantInt(&constant)) || !constant.isSignedIntN(64)) return false;
    result = constant.getSExtValue();
    return true;
}
Array strings(ArrayRef<std::string> values) {
    Array result;
    for (const auto& value : values) result.push_back(value);
    return result;
}
std::string conjunction(ArrayRef<std::string> clauses) {
    std::string text = "true";
    for (const auto& clause : clauses) text += " and (" + clause + ")";
    return text;
}

// Only occurrence-expression conversion lives here. The existing translator,
// descriptor transfer and local requirements remain the owners of effect facts.
// Unsupported scalar expressions make the projection unavailable, never opaque
// unconstrained parameters correlated across repeated executions.
class Occurrences {
public:
    explicit Occurrences(func::FuncOp function) : function(function) {}
    func::FuncOp function;
    llvm::DenseMap<Value, std::string> symbols;
    llvm::DenseMap<Value, std::pair<int64_t, int64_t>> ranges;
    Array parameters, parameterBindings;
    std::vector<std::string> context;
    unsigned nextLoop = 0;
    std::string failure;
    unsigned recursion = 0;

    std::optional<std::string> expression(Value value) {
        if (recursion >= 64) return {};
        llvm::SaveAndRestore<unsigned> depth(recursion, recursion + 1);
        if (auto found = symbols.find(value); found != symbols.end()) return found->second;
        int64_t constant;
        if (constantInt(value, constant)) return std::to_string(constant);
        if (auto arg = dyn_cast<BlockArgument>(value)) {
            if (arg.getOwner() != &function.getBody().front() ||
                !isa<IndexType, IntegerType>(arg.getType())) return {};
            unsigned width = isa<IndexType>(arg.getType()) ? 64 : cast<IntegerType>(arg.getType()).getWidth();
            if (width == 0 || width > 64) return {};
            int64_t low = width == 64 ? INT64_MIN : -(int64_t(1) << (width - 1));
            int64_t high = width == 64 ? INT64_MAX : (int64_t(1) << (width - 1)) - 1;
            std::string name = "arg" + std::to_string(arg.getArgNumber());
            symbols[value] = name;
            ranges[value] = {low, high};
            parameters.push_back(name);
            parameterBindings.push_back(Object{{"name", name}, {"argument", arg.getArgNumber()},
                                               {"signed_width", width}, {"origin", "function argument type"}});
            context.push_back(std::to_string(low) + " <= " + name + " <= " + std::to_string(high));
            return name;
        }
        if (auto rem = value.getDefiningOp<arith::RemUIOp>()) {
            int64_t modulus;
            if (!constantInt(rem.getRhs(), modulus) || modulus <= 0 || modulus > 16) return {};
            auto normalized = normalizeSlotSSA(value, modulus);
            if (!normalized) return {};
            auto induction = expression(normalized->induction);
            if (!induction) return {};
            return "((" + *induction + "+" + std::to_string(normalized->offset) + ") mod " +
                   std::to_string(modulus) + ")";
        }
        // Arithmetic is not silently interpreted over mathematical integers.
        // Only constant shifts with a proved range are exported in this stage.
        auto op = value.getDefiningOp();
        if (op && isa<arith::AddIOp, arith::SubIOp>(op)) {
            int64_t offset;
            if (!constantInt(op->getOperand(1), offset)) return {};
            auto lhs = expression(op->getOperand(0));
            auto range = ranges.find(op->getOperand(0));
            if (!lhs || range == ranges.end()) return {};
            __int128 shift = isa<arith::SubIOp>(op) ? -__int128(offset) : __int128(offset);
            __int128 low = __int128(range->second.first) + shift, high = __int128(range->second.second) + shift;
            unsigned width = isa<IndexType>(value.getType()) ? 64 : cast<IntegerType>(value.getType()).getWidth();
            if (width == 0 || width > 64) return {};
            __int128 minimum = -(__int128(1) << (width - 1)), maximum = (__int128(1) << (width - 1)) - 1;
            if (low < minimum || high > maximum) return {};
            ranges[value] = {int64_t(low), int64_t(high)};
            return "(" + *lhs + (isa<arith::SubIOp>(op) ? "-" : "+") + std::to_string(offset) + ")";
        }
        return {};
    }

    std::optional<std::string> predicate(Value value) {
        if (recursion >= 64) return {};
        llvm::SaveAndRestore<unsigned> depth(recursion, recursion + 1);
        int64_t constant;
        if (constantInt(value, constant)) return constant ? "true" : "false";
        if (auto op = value.getDefiningOp<arith::CmpIOp>()) {
            auto lhs = expression(op.getLhs()), rhs = expression(op.getRhs());
            if (!lhs || !rhs) return {};
            StringRef comparison;
            switch (op.getPredicate()) {
            case arith::CmpIPredicate::eq: comparison = "="; break;
            case arith::CmpIPredicate::ne: comparison = "!="; break;
            case arith::CmpIPredicate::slt: comparison = "<"; break;
            case arith::CmpIPredicate::sle: comparison = "<="; break;
            case arith::CmpIPredicate::sgt: comparison = ">"; break;
            case arith::CmpIPredicate::sge: comparison = ">="; break;
            default: return {};
            }
            return "(" + *lhs + ") " + comparison.str() + " (" + *rhs + ")";
        }
        auto op = value.getDefiningOp();
        if (op && isa<arith::AndIOp, arith::OrIOp>(op)) {
            auto lhs = predicate(op->getOperand(0)), rhs = predicate(op->getOperand(1));
            if (lhs && rhs) return "(" + *lhs + ")" + (isa<arith::AndIOp>(op) ? " and " : " or ") + "(" + *rhs + ")";
        }
        // A function-invariant i1 argument is a real parameter, not a guessed
        // independent boolean for each use. Loop-local unknown predicates fail.
        if (isa<BlockArgument>(value) && value.getType().isInteger(1)) {
            auto term = expression(value);
            if (term) return *term + " != 0";
        }
        return {};
    }
};
} // namespace

HandoffFacts mlir::pto::collectInsertSyncHandoffFacts(
    func::FuncOp function, const SyncIRs& ir, const InsertSyncLifecycleStructure& structure) {
    HandoffFacts result;
    result.record = Object{{"schema", "ptoas.handoff-facts.v1"}, {"function", function.getSymName()},
        {"input_ir_md5", digest(printed(function))}, {"input_ir", printed(function)}, {"status", "unsupported"},
        {"scope", "conservative-local-ordering-projection"}, {"native_transformation_proof", false}};
    std::string moduleAttributes;
    if (auto parent = function->getParentOfType<ModuleOp>()) {
        llvm::raw_string_ostream stream(moduleAttributes);
        parent->getAttrDictionary().print(stream);
    }
    result.record["parent_module_attributes"] = moduleAttributes;
    addProvenance(result.record, function);
    auto fail = [&](StringRef reason) { result.record["reason"] = reason.str(); };
    auto coverage = inspectInsertSyncEffectCoverage(function, ir, false);
    if (failed(coverage) || !*coverage) {
        fail("physical effect coverage is incomplete");
        return result;
    }
    if (structure.status != StorageFrontierSnapshot::Status::Complete || !structure.requirements ||
        !structure.requirements->complete) {
        fail(structure.reason.empty() ? "local requirements unavailable or budget exhausted" : structure.reason);
        return result;
    }
    auto& facts = *structure.requirements;
    Occurrences occurrences(function);
    llvm::DenseMap<Operation*, unsigned> phaseIds, operationIds;
    unsigned ordinal = 0;
    function.walk<WalkOrder::PreOrder>([&](Operation* op) { operationIds[op] = ordinal++; });
    for (unsigned p = 0; p < structure.phases.size(); ++p) {
        if (!phaseIds.try_emplace(structure.phases[p]->elementOp, p).second) {
            fail("multiple physical phases at one operation require an occurrence adapter");
            return result;
        }
    }
    Array statements, nativePhases, nativeAccesses, obligations, omitted, projections;
    bool representable = true;
    unsigned dimensions = 0;
    std::function<void(Region&, std::vector<std::string>, std::vector<std::string>, std::vector<std::string>)> visit;
    visit = [&](Region& region, std::vector<std::string> schedule,
                std::vector<std::string> iterators, std::vector<std::string> domain) {
        if (!llvm::hasSingleElement(region)) { representable = false; occurrences.failure = "multi-block occurrence region"; return; }
        unsigned position = 0;
        for (Operation& op : region.front()) {
            auto time = schedule;
            time.push_back(std::to_string(position++));
            if (auto loop = dyn_cast<scf::ForOp>(op)) {
                int64_t step;
                auto lower = occurrences.expression(loop.getLowerBound());
                auto upper = occurrences.expression(loop.getUpperBound());
                if (!lower || !upper || !constantInt(loop.getStep(), step) || step != 1 || loop->hasAttr("unsignedCmp")) {
                    representable = false; occurrences.failure = "occurrence export needs supported bounds and unit step"; continue;
                }
                std::string iv = "i" + std::to_string(occurrences.nextLoop++);
                occurrences.symbols[loop.getInductionVar()] = iv;
                int64_t lo = INT64_MIN, hi = INT64_MAX;
                constantInt(loop.getLowerBound(), lo);
                constantInt(loop.getUpperBound(), hi);
                occurrences.ranges[loop.getInductionVar()] = {lo, hi == INT64_MIN ? hi : hi - 1};
                auto bodyDomain = domain;
                bodyDomain.push_back("(" + *lower + ") <= " + iv + " < (" + *upper + ")");
                auto bodyIterators = iterators;
                bodyIterators.push_back(iv);
                time.push_back(iv);
                visit(loop.getRegion(), time, bodyIterators, bodyDomain);
                continue;
            }
            if (auto branch = dyn_cast<scf::IfOp>(op)) {
                auto condition = occurrences.predicate(branch.getCondition());
                if (!condition) { representable = false; occurrences.failure = "unqualified branch occurrence predicate"; continue; }
                auto thenDomain = domain;
                thenDomain.push_back(*condition);
                auto branchTime = time;
                branchTime.push_back("0");
                visit(branch.getThenRegion(), branchTime, iterators, thenDomain);
                if (!branch.getElseRegion().empty()) {
                    auto elseDomain = domain;
                    elseDomain.push_back("not (" + *condition + ")");
                    branchTime.back() = "1";
                    visit(branch.getElseRegion(), branchTime, iterators, elseDomain);
                }
                continue;
            }
            if (isa<SectionCubeOp, SectionVectorOp>(op)) {
                visit(op.getRegion(0), time, iterators, domain);
                continue;
            }
            auto phase = phaseIds.find(&op);
            if (phase == phaseIds.end()) {
                if (op.getNumRegions()) { representable = false; occurrences.failure = "unmodeled occurrence region"; }
                continue;
            }
            unsigned p = phase->second;
            auto* effect = structure.phases[p];
            StringRef lane = laneName(effect->kPipeValue);
            if (lane == "unsupported") { representable = false; occurrences.failure = "unmodeled physical lane"; }
            Array reads, writes;
            auto add = [&](const BaseMemInfo* memory, bool write) {
                if (!memory) { representable = false; occurrences.failure = "missing physical memory record"; return; }
                unsigned accessId = nativeAccesses.size();
                Array addresses;
                for (uint64_t address : memory->baseAddresses) addresses.push_back(std::to_string(address));
                nativeAccesses.push_back(Object{{"id", accessId}, {"phase", p}, {"write", write},
                    {"space", spaceName(memory->scope)}, {"addresses", std::move(addresses)},
                    {"allocation_bytes", std::to_string(memory->allocateSize)},
                    {"known_physical_addresses", memory->hasKnownPhysicalAddresses},
                    {"unknown_range", memory->aliasesUnknownRange}});
                if (memory->scope == AddressSpace::GM) {
                    omitted.push_back(Object{{"access", accessId}, {"reason", "GM geometry, aliasing and visibility remain native requirements"}});
                    return;
                }
                auto slice = LocalStorageRequirements::exact(memory);
                if (!slice || spaceName(memory->scope) == "unsupported") {
                    representable = false; occurrences.failure = "local access lacks a singleton conservative physical interval"; return;
                }
                // Preserve the native may-access contract. A full descriptor is
                // not by itself a target proof of definite byte replacement.
                auto projection = facts.project(*slice);
                Object access{{"space", spaceName(memory->scope)}, {"address", std::to_string(slice->begin)},
                    {"size", slice->bytes}, {"definite", false}, {"native_access", accessId},
                    {"footprint_precision", "conservative-allocation"},
                    {"native_whole_productions", projection && projection->wholeProductions}};
                (write ? writes : reads).push_back(std::move(access));
            };
            for (auto* memory : effect->useVec) add(memory, false);
            for (auto* memory : effect->defVec) add(memory, true);
            statements.push_back(Object{{"id", "P" + std::to_string(p)}, {"lane", lane}, {"core", "local"},
                {"iterators", strings(iterators)}, {"schedule", strings(time)}, {"domain", conjunction(domain)},
                {"reads", std::move(reads)}, {"writes", std::move(writes)}});
            nativePhases.push_back(Object{{"id", p}, {"operation_ordinal", operationIds[&op]},
                {"operation", op.getName().getStringRef()}, {"lane", lane}});
            dimensions = std::max(dimensions, unsigned(time.size()));
        }
    };
    visit(function.getBody(), {}, {}, {});
    for (auto& value : statements) {
        auto& time = *value.getAsObject()->getArray("schedule");
        while (time.size() < dimensions) time.push_back("0");
    }
    for (const auto& requirement : facts.obligations) {
        auto memory = [&](unsigned index) {
            const auto& access = facts.accesses[index];
            auto slice = LocalStorageRequirements::exact(&access.memory);
            return Object{{"space", spaceName(access.memory.scope)},
                {"begin", slice ? std::to_string(slice->begin) : "unknown"},
                {"bytes", slice ? std::to_string(slice->bytes) : "unknown"},
                {"read", access.read}, {"write", access.write}};
        };
        obligations.push_back(Object{{"source", requirement.source}, {"target", requirement.target},
            {"hazards", requirement.hazards}, {"occurrence", "all feasible ordered occurrences"},
            {"source_access", memory(requirement.sourceAccess)}, {"target_access", memory(requirement.targetAccess)},
            {"resource_order_outside_reference", bool(requirement.hazards & LocalStorageRequirements::AccReadOrder)}});
    }
    for (const auto& projection : facts.slices) {
        projections.push_back(Object{{"space", spaceName(projection.slice.space)},
            {"begin", std::to_string(projection.slice.begin)}, {"bytes", std::to_string(projection.slice.bytes)},
            {"complete", projection.complete}, {"whole_productions", projection.wholeProductions}});
    }
    result.record["native"] = Object{{"phases", std::move(nativePhases)}, {"accesses", std::move(nativeAccesses)},
        {"projections", std::move(projections)},
        {"obligations", std::move(obligations)}, {"omitted_effects", std::move(omitted)},
        {"parameter_bindings", std::move(occurrences.parameterBindings)},
        {"physical_context", structure.cube ? "cube" : "vector"},
        {"abstract_guard_nodes", structure.program.nodes.size()},
        {"abstract_nodes_are_occurrences", false}};
    if (statements.size() != structure.phases.size()) {
        representable = false;
        if (occurrences.failure.empty()) occurrences.failure = "not every physical phase has an occurrence mapping";
    }
    if (!representable || statements.empty()) {
        fail(occurrences.failure.empty() ? "no physical phases" : occurrences.failure);
        return result;
    }
    result.record["model"] = Object{{"name", function.getSymName()},
        {"parameters", std::move(occurrences.parameters)}, {"context", conjunction(occurrences.context)},
        {"statements", std::move(statements)}};
    result.record["status"] = "supported-local-projection";
    result.record["reason"] = "original structured occurrences and translated conservative local effects";
    result.supported = true;
    return result;
}

LogicalResult mlir::pto::exportInsertSyncHandoffFacts(func::FuncOp function, StringRef directory, uint64_t workLimit,
                                                    StringRef bypassReason) {
    OwningOpRef<ModuleOp> container(ModuleOp::create(function.getLoc()));
    SmallVector<ModuleOp> ancestry;
    for (auto* op = function->getParentOp(); op; op = op->getParentOp())
        if (auto module = dyn_cast<ModuleOp>(op)) ancestry.push_back(module);
    ModuleOp destination = *container;
    for (auto module : llvm::reverse(ancestry)) {
        auto child = ModuleOp::create(module.getLoc());
        child->setAttrs(module->getAttrs());
        destination.getBody()->push_back(child);
        destination = child;
    }
    auto clone = cast<func::FuncOp>(function->clone());
    destination.getBody()->push_back(clone);
    HandoffFacts facts;
    uint64_t used = 0;
    bool calls = false;
    clone.walk([&](func::CallOp) { calls = true; });
    if (!bypassReason.empty() || calls) {
        facts.record = Object{{"schema", "ptoas.handoff-facts.v1"}, {"function", function.getSymName()},
            {"status", bypassReason.empty() ? "unsupported" : "bypassed"},
            {"reason", bypassReason.empty() ? "helper effects and symbol contracts are not imported" : bypassReason.str()},
            {"native_transformation_proof", false},
            {"input_ir", printed(function)}, {"input_ir_md5", digest(printed(function))}};
    } else {
    MemoryDependentAnalyzer analyzer;
    auto contract = resolveInsertSyncGMAlias(function, "");
    if (failed(contract)) return failure();
    analyzer.setGMContract(clone, *contract);
    SyncIRs ir;
    Buffer2MemInfoMap memory;
    PTOIRTranslator translator(ir, analyzer, memory, clone, SyncAnalysisMode::NORMALSYNC);
    translator.enableGenerationFlow(true);
    translator.Build();
    insert_sync_frontier::Budget budget{workLimit};
    auto structure = buildInsertSyncLifecycleStructure(clone, ir, budget, true);
    facts = collectInsertSyncHandoffFacts(clone, ir, structure);
    used = workLimit - budget.left;
    facts.record["gm_contract"] = *contract == InsertSyncGMAliasMode::MayAlias ? "may-alias" : "assume-disjoint-arguments";
    }
    facts.record["analysis_work"] = used;
    // Use original ancestry/position, not the temporary analysis container.
    addProvenance(facts.record, function);
    if (auto error = llvm::sys::fs::create_directories(directory)) return function.emitError("handoff facts directory: ") << error.message();
    // Hash the symbol rather than interpolating it into a filesystem path.
    // Payload hash distinguishes same-named functions in nested modules.
    llvm::SmallString<256> path(directory), temporary;
    llvm::sys::path::append(path, digest(function.getSymName()) + "-" + digest(printed(function)) + "-" +
                                  facts.record.getString("context_md5")->str() + ".json");
    int fd;
    if (auto error = llvm::sys::fs::createUniqueFile(path + ".tmp-%%%%%%", fd, temporary))
        return function.emitError("handoff facts temporary: ") << error.message();
    llvm::raw_fd_ostream stream(fd, true);
    stream << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(facts.record)));
    stream.close();
    if (stream.has_error()) {
        stream.clear_error();
        llvm::sys::fs::remove(temporary);
        return function.emitError("writing handoff facts failed");
    }
    if (auto error = llvm::sys::fs::rename(temporary, path)) {
        llvm::sys::fs::remove(temporary);
        return function.emitError("handoff facts rename: ") << error.message();
    }
    return success();
}
