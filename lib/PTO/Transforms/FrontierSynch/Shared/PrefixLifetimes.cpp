// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Analyze only the actual coordinator protocol. Existing full-prefix waits and
// ALL barriers may certify reuse; this stage never inserts additional ordering.
#include "PrefixLifetimes.h"
#include "InvocationSourceBounds.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/ExecutionContexts.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOSyncUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/AsmState.h"
#include "llvm/Support/SHA256.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/IR/Verifier.h"
#include <map>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
constexpr llvm::StringLiteral lifetimeAttr = "pto.prefix_lifetimes";
struct Direction {
    uint32_t source = 0, target = 0;
    uint64_t bound = 0, width = 0;
    SmallVector<Operation*> sets, waits;
    scf::ForOp epoch;
};
using Inventory = std::map<int64_t, Direction>;
std::optional<int64_t> number(DictionaryAttr attribute, StringRef name)
{
    auto value = attribute ? attribute.getAs<IntegerAttr>(name) : IntegerAttr{};
    if (!value || !value.getType().isInteger(64)) {
        return std::nullopt;
    }
    return value.getInt();
}
bool one(Value value)
{
    APInt integer;
    return matchPattern(value, m_ConstantInt(&integer)) && integer == 1;
}
bool allBarrier(Operation* operation)
{
    auto barrier = dyn_cast_or_null<BarrierOp>(operation);
    return barrier && barrier.getPipe().getPipe() == PIPE::PIPE_ALL;
}
FailureOr<Inventory> inventory(func::FuncOp function, std::string& error)
{
    auto protocol = function->getAttrOfType<DictionaryAttr>("pto.prefix_protocol");
    auto version = number(protocol, "version"), plan = number(protocol, "plan");
    auto directions = protocol ? protocol.getAs<ArrayAttr>("directions") : ArrayAttr{};
    if (!version || *version != 1 || !plan || *plan < 0 || !directions || function->hasAttr(ActiveContextAttr) ||
        hasPhysicalSections(function)) {
        error = "prefix lifetime export requires a whole invocation coordinator";
        return failure();
    }
    Inventory result;
    std::set<std::pair<uint32_t, uint32_t>> seen;
    for (auto attribute : directions) {
        auto item = dyn_cast<DictionaryAttr>(attribute);
        auto source = number(item, "source"), target = number(item, "target");
        auto record = number(item, "record"), bound = number(item, "bound");
        if (!source || !target || *source < 0 || *target < 0 || *source > UINT32_MAX || *target > UINT32_MAX ||
            !isConcreteSyncPipe(static_cast<PIPE>(*source)) || !isConcreteSyncPipe(static_cast<PIPE>(*target)) ||
            *source == *target || !record || *record < 0 || *record > UINT32_MAX || !bound || *bound < 0 ||
            *bound >= INT64_MAX || !seen.emplace(*source, *target).second || result.count(*record)) {
            error = "malformed prefix direction lifetime";
            return failure();
        }
        result.emplace(*record, Direction{uint32_t(*source), uint32_t(*target), uint64_t(*bound), uint64_t(*bound)});
    }
    std::map<int64_t, unsigned> terminal;
    auto walk = function.walk([&](Operation* operation) -> WalkResult {
        if (!isa<LogicalSetOp, LogicalWaitOp>(operation)) {
            return WalkResult::advance();
        }
        auto record = operation->getAttrOfType<IntegerAttr>("record_id");
        auto identity = operation->getAttrOfType<IntegerAttr>("plan_id");
        auto source = operation->getAttrOfType<PipeAttr>("src_pipe");
        auto target = operation->getAttrOfType<PipeAttr>("dst_pipe");
        auto found = record ? result.find(record.getInt()) : result.end();
        if (found == result.end() || !identity || identity.getInt() != *plan || !source || !target ||
            uint32_t(source.getPipe()) != found->second.source || uint32_t(target.getPipe()) != found->second.target ||
            operation->getNumOperands() != 1 || !operation->getOperand(0).getType().isIndex()) {
            error = "prefix command inventory no longer matches its coordinator";
            return WalkResult::interrupt();
        }
        auto& direction = found->second;
        if (isa<LogicalSetOp>(operation)) {
            auto increment = operation->getOperand(0).getDefiningOp<arith::AddIOp>();
            if (!increment || !one(increment.getRhs())) {
                error = "prefix publication lost its executed-source increment";
                return WalkResult::interrupt();
            }
            direction.sets.push_back(operation);
        } else {
            auto loop = dyn_cast<scf::ForOp>(operation->getParentOp());
            auto owner = loop ? loop->getAttrOfType<IntegerAttr>("pto.prefix_acquisition") : IntegerAttr{};
            auto first = loop ? loop.getLowerBound().getDefiningOp<arith::AddIOp>() : arith::AddIOp{};
            auto end = loop ? loop.getUpperBound().getDefiningOp<arith::AddIOp>() : arith::AddIOp{};
            if (!loop || !owner || owner.getInt() != record.getInt() || !one(loop.getStep()) || !first || !end ||
                !one(first.getRhs()) || !one(end.getRhs()) || operation->getOperand(0) != loop.getInductionVar() ||
                !loop.getInitArgs().empty()) {
                error = "prefix acquisition lost its ordered watermark interval";
                return WalkResult::interrupt();
            }
            if (loop->hasAttr("pto.prefix_terminal_acquisition")) {
                if (loop->getBlock() != &function.front() || !loop->hasAttr("pto.prefix_full_acquisition")) {
                    error = "prefix cleanup is not a terminal full-prefix acquisition";
                    return WalkResult::interrupt();
                }
                ++terminal[record.getInt()];
            }
            direction.waits.push_back(operation);
        }
        return WalkResult::advance();
    });
    if (walk.wasInterrupted()) {
        return failure();
    }
    for (auto& [record, direction] : result) {
        if (terminal[record] != 1 || direction.waits.empty()) {
            error = "prefix lifetime export lacks unique terminal cleanup";
            return failure();
        }
        // The admitted invocation bound is producer-owned evidence, as for
        // other saved allocation certificates. Structural counter/matching
        // validation above is not a proof checker for arbitrary annotated IR.
    }
    return result;
}
// A full acquisition at the direct body cut followed by an existing ALL barrier
// closes all earlier publications before the next epoch. No source may follow
// that acquisition. Optional sources and surplus terminal cleanup remain valid.
scf::ForOp drainedEpoch(const Direction& direction)
{
    if (direction.sets.empty()) {
        return {};
    }
    for (auto candidate = direction.sets.front()->getParentOfType<scf::ForOp>(); candidate;
         candidate = candidate->getParentOfType<scf::ForOp>()) {
        if (!llvm::all_of(direction.sets, [&](Operation* op) { return candidate->isProperAncestor(op); })) {
            continue;
        }
        for (auto* wait : direction.waits) {
            auto acquire = cast<scf::ForOp>(wait->getParentOp());
            if (!acquire->hasAttr("pto.prefix_full_acquisition")) {
                continue;
            }
            auto* block = acquire->getBlock();
            auto* enclosing = acquire.getOperation();
            // A conditional arm may have no publications. On every arm that
            // does publish, all publications must precede this full drain.
            bool structured = true;
            while (enclosing->getBlock() != candidate.getBody()) {
                auto branch = dyn_cast<scf::IfOp>(enclosing->getParentOp());
                if (!branch) {
                    structured = false;
                    break;
                }
                enclosing = branch.getOperation();
            }
            if (!structured) {
                continue;
            }
            bool after = false, valid = true;
            for (auto* source : direction.sets) {
                auto* direct = source;
                while (direct->getBlock() != block && direct->getParentOp() != candidate) {
                    direct = direct->getParentOp();
                }
                if (direct->getBlock() != block || !direct->isBeforeInBlock(acquire)) {
                    valid = false;
                    break;
                }
            }
            if (!valid) {
                continue;
            }
            for (Operation& sibling : *candidate.getBody()) {
                after |= &sibling == enclosing;
                if (after && allBarrier(&sibling)) {
                    return candidate;
                }
            }
        }
    }
    return {};
}
FailureOr<Inventory> planned(func::FuncOp function, std::string& error)
{
    auto result = inventory(function, error);
    if (failed(result)) {
        return failure();
    }
    for (auto& [record, direction] : *result) {
        direction.epoch = drainedEpoch(direction);
        if (!direction.epoch) {
            continue;
        }
        std::vector<PrefixSource> sources;
        for (auto* op : direction.sets) {
            sources.push_back({op, 0, {}});
        }
        std::string optionalError;
        auto counts = inferInvocationSourceBounds(function, sources, optionalError, direction.epoch);
        if (failed(counts)) {
            direction.epoch = {};
            continue;
        }
        direction.width = std::min(direction.bound, (*counts)[0]);
    }
    return result;
}
class BodyDigest : public llvm::raw_ostream {
public:
    BodyDigest() : llvm::raw_ostream(true) {}
    std::string result() { return llvm::toHex(hash.final()); }

private:
    llvm::SHA256 hash;
    uint64_t position = 0;
    void write_impl(const char* data, std::size_t size) override
    {
        hash.update(StringRef(data, size));
        position += size;
    }
    uint64_t current_pos() const override { return position; }
};
std::string bodyDigest(func::FuncOp function)
{
    BodyDigest stream;
    AsmState state(function, OpPrintingFlags().useLocalScope().printGenericOpForm());
    function.front().print(stream, state);
    return stream.result();
}
DictionaryAttr encode(func::FuncOp function, const Inventory& inventory)
{
    Builder b(function.getContext());
    SmallVector<Attribute> directions;
    for (const auto& [record, direction] : inventory) {
        directions.push_back(b.getDictionaryAttr(
            {b.getNamedAttr("record", b.getI64IntegerAttr(record)),
             b.getNamedAttr("bound", b.getI64IntegerAttr(direction.bound)),
             b.getNamedAttr("width", b.getI64IntegerAttr(direction.width)),
             b.getNamedAttr("strategy", b.getStringAttr(direction.epoch ? "drained-epochs" : "fresh-invocation")),
             b.getNamedAttr("publication_sites", b.getI64IntegerAttr(direction.sets.size())),
             b.getNamedAttr("acquisition_sites", b.getI64IntegerAttr(direction.waits.size()))}));
    }
    return b.getDictionaryAttr(
        {b.getNamedAttr("version", b.getI64IntegerAttr(1)), b.getNamedAttr("directions", b.getArrayAttr(directions)),
         b.getNamedAttr("body_sha256", b.getStringAttr(bodyDigest(function)))});
}
LogicalResult lower(func::FuncOp function, const Inventory& inventory, ArrayRef<int64_t> eligible)
{
    DominanceInfo dominance(function);
    struct Endpoint {
        Operation* command;
        std::shared_ptr<const PhysicalEndpointMap> map;
        uint32_t source, target;
    };
    std::vector<Endpoint> endpoints;
    for (const auto& [record, direction] : inventory) {
        SmallVector<Operation*> commands(direction.sets.begin(), direction.sets.end());
        commands.append(direction.waits.begin(), direction.waits.end());
        for (auto* command : commands) {
            if (!direction.width) {
                endpoints.push_back({command, {}, direction.source, direction.target});
                continue;
            }
            PhysicalRecordAllocation assignment{record, direction.source, direction.target, 1, direction.width - 1, {}};
            assignment.ids.append(eligible.begin(), eligible.begin() + direction.width);
            SmallVector<TemplateCoordinate> point;
            auto map = planPhysicalEndpointMap(
                command, ArrayRef<PhysicalRecordAllocation>(&assignment, 1),
                ArrayRef<SmallVector<TemplateCoordinate>>(&point, 1), 1, false, dominance);
            if (failed(map)) {
                return function.emitError("prefix physical map is unavailable");
            }
            endpoints.push_back({command, *map, direction.source, direction.target});
        }
    }
    for (const auto& endpoint : endpoints) {
        auto* command = endpoint.command;
        if (endpoint.map) {
            OpBuilder b(command);
            Value ordinal = command->getOperand(0);
            auto id = emitPhysicalEndpointMap(b, command->getLoc(), *endpoint.map, ArrayRef<Value>(&ordinal, 1), {});
            emitPhysicalCommand(b, command->getLoc(), isa<LogicalSetOp>(command), endpoint.source, endpoint.target, id);
        }
        command->erase();
    }
    return success();
}
} // namespace
LogicalResult exportPrefixLifetimes(func::FuncOp function, std::string& error)
{
    auto result = planned(function, error);
    if (failed(result)) {
        return failure();
    }
    function->setAttr(lifetimeAttr, encode(function, *result));
    return success();
}
LogicalResult allocatePrefixLifetimes(func::FuncOp function, ArrayRef<int64_t> eligible)
{
    if (failed(preflightPhysicalOutput(function, {}))) {
        return failure();
    }
    std::set<int64_t> ids(eligible.begin(), eligible.end());
    if (eligible.size() > 6 || ids.size() != eligible.size() ||
        llvm::any_of(eligible, [](int64_t id) { return id < 0 || id >= 6; })) {
        return function.emitError("prefix allocation requires distinct eligible IDs in 0..5");
    }
    if (!function.getBody().hasOneBlock() || !allBarrier(function.front().getTerminator()->getPrevNode())) {
        return function.emitError("prefix allocation lacks completed terminal invocation closure");
    }
    std::string error;
    auto result = planned(function, error);
    if (failed(result)) {
        return function.emitError(error);
    }
    auto certificate = function->getAttrOfType<DictionaryAttr>(lifetimeAttr);
    if (!certificate || certificate != encode(function, *result)) {
        return function.emitError("prefix lifetime export is missing or no longer matches the command protocol");
    }
    for (const auto& [record, direction] : *result) {
        if (direction.width > eligible.size()) {
            return function.emitError("prefix physical assignment not certified within directed capacity: ")
                   << direction.width << " sufficient IDs, " << eligible.size()
                   << " supplied; reusable lifetime export unavailable for this protocol; no minimum-capacity claim";
        }
    }
    OwningOpRef<func::FuncOp> staged(cast<func::FuncOp>(function->clone()));
    auto stagedInventory = planned(*staged, error);
    if (failed(stagedInventory) || failed(lower(*staged, *stagedInventory, eligible)) || failed(verify(*staged))) {
        return function.emitError("prefix allocation failed staged lowering");
    }
    function.getBody().takeBody(staged->getBody());
    Builder b(function.getContext());
    function->setAttr(
        "pto.prefix_physical", b.getDictionaryAttr(
                                   {b.getNamedAttr("version", b.getI64IntegerAttr(1)),
                                    b.getNamedAttr("eligible", b.getDenseI64ArrayAttr(eligible))}));
    return success();
}
} // namespace mlir::pto::frontiersynch
