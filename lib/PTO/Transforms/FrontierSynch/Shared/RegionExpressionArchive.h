// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Included once by RegionExpressions.cpp, where the immutable node representation is available.
#ifndef PTO_FRONTIER_REGIONEXPRESSIONARCHIVE_H
#define PTO_FRONTIER_REGIONEXPRESSIONARCHIVE_H
namespace {
Attribute archiveInteger(Builder& b, const BoundInteger& value)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << value;
    return b.getStringAttr(text);
}
FailureOr<BoundInteger> restoreInteger(Attribute raw)
{
    auto text = dyn_cast_or_null<StringAttr>(raw);
    if (!text || text.getValue().empty()) { return failure(); }
    auto digits = text.getValue();
    const bool negative = digits.consume_front("-");
    if (digits.empty()) { return failure(); }
    BoundInteger value(0);
    for (char digit : digits) {
        if (digit < '0' || digit > '9') { return failure(); }
        value = value * BoundInteger(10) + BoundInteger(digit - '0');
    }
    return negative ? -value : value;
}
ArrayAttr archiveIntegers(Builder& b, ArrayRef<BoundInteger> values)
{
    SmallVector<Attribute> result;
    for (const auto& value : values) { result.push_back(archiveInteger(b, value)); }
    return b.getArrayAttr(result);
}
FailureOr<std::vector<BoundInteger>> restoreIntegers(Attribute raw)
{
    auto list = dyn_cast_or_null<ArrayAttr>(raw);
    if (!list) { return failure(); }
    std::vector<BoundInteger> result;
    for (auto entry : list) {
        auto value = restoreInteger(entry);
        if (failed(value)) { return failure(); }
        result.push_back(std::move(*value));
    }
    return result;
}
}
FailureOr<DictionaryAttr> RegionExpressions::archive(ArrayRef<Id> roots, MLIRContext* context,
    llvm::function_ref<Attribute(Value)> encodeValue) const
{
    if (!constructionMessage.empty() || !llvm::all_of(roots, [&](Id id) { return valid(id); })) {
        return failure();
    }
    Builder b(context);
    SmallVector<Id> order;
    visitReferenced(roots, [&](Id id) { order.push_back(id); });
    llvm::sort(order);
    llvm::DenseMap<Id, int64_t> ids;
    SmallVector<Attribute> encoded;
    auto number = [&](StringRef name, uint64_t value) { return b.getNamedAttr(name, b.getI64IntegerAttr(value)); };
    for (Id id : order) {
        if (evaluation(id)) { return failure(); }
        const auto& node = nodes[id];
        SmallVector<Id> operands;
        appendOperands(node, operands);
        SmallVector<int64_t> references;
        for (Id operand : operands) {
            auto found = ids.find(operand);
            if (found == ids.end()) { return failure(); }
            references.push_back(found->second);
        }
        SmallVector<NamedAttribute> payload;
        if (node.kind == Kind::Constant) { payload.push_back(number("literal", node.literal)); }
        else if (node.kind == Kind::Input) {
            auto value = encodeValue(node.value);
            if (!value) { return failure(); }
            payload.push_back(b.getNamedAttr("value", value));
        } else if (node.kind == Kind::Integer) {
            const auto& recipe = *node.integer;
            payload.push_back(number("period", recipe.period));
            payload.push_back(number("output_residue", recipe.outputResidue));
            payload.push_back(b.getNamedAttr("predicate", b.getBoolAttr(recipe.predicate)));
            SmallVector<int64_t> residues(recipe.residues.begin(), recipe.residues.end());
            payload.push_back(b.getNamedAttr("residues", b.getDenseI64ArrayAttr(residues)));
            payload.push_back(b.getNamedAttr("coefficients", archiveIntegers(b, recipe.numerator.coefficients)));
            payload.push_back(b.getNamedAttr("constant", archiveInteger(b, recipe.numerator.constant)));
            payload.push_back(b.getNamedAttr("denominator", archiveInteger(b, recipe.denominator)));
            payload.push_back(number("dimensions", recipe.system.dimensions()));
            SmallVector<Attribute> rows, congruences, coordinates;
            for (const auto& row : recipe.system.constraints()) {
                rows.push_back(b.getArrayAttr({archiveIntegers(b, row.coefficients), archiveInteger(b, row.bound)}));
            }
            for (const auto& row : recipe.system.congruences()) {
                congruences.push_back(b.getArrayAttr({archiveIntegers(b, row.coefficients),
                    archiveInteger(b, row.residue), archiveInteger(b, row.modulus)}));
            }
            // Known contradictions may have no stored rows. Preserve them explicitly.
            payload.push_back(b.getNamedAttr("empty", b.getBoolAttr(recipe.system.isKnownEmpty())));
            for (auto coordinate : recipe.coordinates) {
                coordinates.push_back(AffineMapAttr::get(AffineMap::get(recipe.inputs.size(), 0, coordinate)));
            }
            payload.push_back(b.getNamedAttr("rows", b.getArrayAttr(rows)));
            payload.push_back(b.getNamedAttr("congruences", b.getArrayAttr(congruences)));
            payload.push_back(b.getNamedAttr("coordinates", b.getArrayAttr(coordinates)));
        }
        encoded.push_back(b.getDictionaryAttr({number("opcode", static_cast<uint64_t>(node.kind)),
            b.getNamedAttr("boolean", b.getBoolAttr(node.boolean)),
            b.getNamedAttr("operands", b.getDenseI64ArrayAttr(references)),
            b.getNamedAttr("payload", b.getDictionaryAttr(payload))}));
        ids[id] = encoded.size() - 1;
    }
    SmallVector<int64_t> outputs;
    for (Id root : roots) { outputs.push_back(ids.lookup(root)); }
    return b.getDictionaryAttr({b.getNamedAttr("version", b.getI64IntegerAttr(1)),
        b.getNamedAttr("nodes", b.getArrayAttr(encoded)),
        b.getNamedAttr("roots", b.getDenseI64ArrayAttr(outputs))});
}
FailureOr<SmallVector<RegionExpressions::Id>> RegionExpressions::restoreArchive(DictionaryAttr data,
    llvm::function_ref<Value(Attribute)> decodeValue)
{
    auto encoded = data ? data.getAs<ArrayAttr>("nodes") : ArrayAttr{};
    auto roots = data ? data.getAs<DenseI64ArrayAttr>("roots") : DenseI64ArrayAttr{};
    auto version = data ? data.getAs<IntegerAttr>("version") : IntegerAttr{};
    if (!data || data.size() != 3 || !version || !version.getType().isInteger(64) ||
        version.getInt() != 1 || !encoded || !roots) { return failure(); }
    Transaction transaction(*this);
    SmallVector<Id> ids;
    for (auto raw : encoded) {
        auto node = dyn_cast<DictionaryAttr>(raw);
        auto opcode = node ? node.getAs<IntegerAttr>("opcode") : IntegerAttr{};
        auto type = node ? node.getAs<BoolAttr>("boolean") : BoolAttr{};
        auto refs = node ? node.getAs<DenseI64ArrayAttr>("operands") : DenseI64ArrayAttr{};
        auto payload = node ? node.getAs<DictionaryAttr>("payload") : DictionaryAttr{};
        if (!node || node.size() != 4 || !opcode || !opcode.getType().isInteger(64) ||
            opcode.getInt() < 0 || opcode.getInt() > static_cast<int64_t>(Kind::Integer) ||
            !type || !refs || !payload) { return failure(); }
        SmallVector<Id> inputs;
        for (int64_t ref : refs.asArrayRef()) {
            if (ref < 0 || static_cast<uint64_t>(ref) >= ids.size()) { return failure(); }
            inputs.push_back(ids[ref]);
        }
        const auto kind = static_cast<Kind>(opcode.getInt());
        Id result = invalid;
        if (kind == Kind::Constant) {
            auto literal = payload.getAs<IntegerAttr>("literal");
            if (!inputs.empty() || payload.size() != 1 || !literal || !literal.getType().isInteger(64) ||
                (type.getValue() && literal.getValue().getZExtValue() > 1)) { return failure(); }
            result = type.getValue() ? boolean(literal.getInt() != 0) : constant(literal.getValue().getZExtValue());
        } else if (kind == Kind::Input) {
            if (!inputs.empty() || payload.size() != 1 || !payload.get("value")) { return failure(); }
            auto value = decodeValue(payload.get("value"));
            if (!value) { return failure(); }
            result = input(value);
        } else if (kind == Kind::Integer) {
            auto predicate = payload.getAs<BoolAttr>("predicate"), empty = payload.getAs<BoolAttr>("empty");
            auto period = payload.getAs<IntegerAttr>("period"), residue = payload.getAs<IntegerAttr>("output_residue");
            auto dimensions = payload.getAs<IntegerAttr>("dimensions");
            auto residues = payload.getAs<DenseI64ArrayAttr>("residues");
            auto rows = payload.getAs<ArrayAttr>("rows"), congruences = payload.getAs<ArrayAttr>("congruences");
            auto coordinates = payload.getAs<ArrayAttr>("coordinates");
            auto numerator = restoreIntegers(payload.get("coefficients"));
            auto offset = restoreInteger(payload.get("constant"));
            auto denominator = restoreInteger(payload.get("denominator"));
            if (payload.size() != 12 || !predicate || !empty || !period || !residue || !dimensions ||
                !period.getType().isInteger(64) || !residue.getType().isInteger(64) ||
                !dimensions.getType().isInteger(64) || period.getInt() <= 0 || residue.getInt() < 0 ||
                dimensions.getInt() < 0 || static_cast<uint64_t>(dimensions.getInt()) > UINT_MAX ||
                !residues || !rows || !congruences || !coordinates || failed(numerator) ||
                failed(offset) || failed(denominator)) { return failure(); }
            const auto coordinateCount = coordinates.empty() ? inputs.size() : coordinates.size();
            // Dimensions are a schema assertion, never an allocation request.
            // Check against already owned operands/maps before constructing rows.
            if (predicate.getValue()) {
                if (static_cast<uint64_t>(dimensions.getInt()) != coordinateCount ||
                    static_cast<std::size_t>(residues.size()) != coordinateCount || !numerator->empty() ||
                    *offset != 0 || *denominator != 1 || residue.getInt() != 0) { return failure(); }
            } else if (dimensions.getInt() || !rows.empty() || !congruences.empty() || empty.getValue() ||
                !coordinates.empty() || numerator->size() != inputs.size() ||
                static_cast<std::size_t>(residues.size()) != inputs.size() ||
                *denominator <= 0 || residue.getInt() >= period.getInt()) { return failure(); }
            SmallVector<uint64_t> residueValues;
            for (int64_t value : residues.asArrayRef()) {
                if (value < 0 || value >= period.getInt()) { return failure(); }
                residueValues.push_back(value);
            }
            std::vector<IntegerConstraint> inequalities;
            std::vector<IntegerCongruence> divisibilities;
            for (auto rawRow : rows) {
                auto row = dyn_cast<ArrayAttr>(rawRow);
                if (!row || row.size() != 2) { return failure(); }
                auto coefficients = restoreIntegers(row[0]); auto bound = restoreInteger(row[1]);
                if (failed(coefficients) || failed(bound) || coefficients->size() != coordinateCount) {
                    return failure();
                }
                inequalities.push_back({std::move(*coefficients), std::move(*bound)});
            }
            for (auto rawRow : congruences) {
                auto row = dyn_cast<ArrayAttr>(rawRow);
                if (!row || row.size() != 3) { return failure(); }
                auto coefficients = restoreIntegers(row[0]); auto value = restoreInteger(row[1]);
                auto modulus = restoreInteger(row[2]);
                if (failed(coefficients) || failed(value) || failed(modulus) ||
                    coefficients->size() != coordinateCount || *modulus <= 0) { return failure(); }
                divisibilities.push_back({std::move(*coefficients), std::move(*value), std::move(*modulus)});
            }
            if (empty.getValue()) {
                inequalities.push_back(
                    {std::vector<BoundInteger>(dimensions.getInt(), BoundInteger(0)), BoundInteger(-1)});
            }
            auto system = IntegerSystem::create(dimensions.getInt(), inequalities, divisibilities);
            if (failed(system)) { return failure(); }
            SmallVector<AffineExpr> maps;
            for (auto rawMap : coordinates) {
                auto map = dyn_cast<AffineMapAttr>(rawMap);
                if (!map || map.getValue().getNumDims() != inputs.size() || map.getValue().getNumSymbols() ||
                    map.getValue().getNumResults() != 1) { return failure(); }
                maps.push_back(map.getValue().getResult(0));
            }
            if (predicate.getValue()) {
                if (maps.empty()) { result = integerPredicate(*system, inputs, period.getInt(), residueValues); }
                else {
                    std::string diagnostic;
                    auto mapped = integerMappedPredicate(
                        *system, inputs, maps, period.getInt(), residueValues, diagnostic);
                    if (!mapped) { return failure(); }
                    result = *mapped;
                }
            } else {
                if (!maps.empty() || dimensions.getInt() || !rows.empty() || !congruences.empty() || empty.getValue()) {
                    return failure();
                }
                result = integerWitness({std::move(*numerator), std::move(*offset)}, *denominator,
                    inputs, period.getInt(), residueValues, residue.getInt());
            }
        } else {
            if (!payload.empty()) { return failure(); }
            if (kind == Kind::Not && inputs.size() == 1) { result = lnot(inputs[0]); }
            else if (kind == Kind::Select && inputs.size() == 3) { result = select(inputs[0], inputs[1], inputs[2]); }
            else if (kind != Kind::Not && kind != Kind::Select && inputs.size() == 2) {
                result = binary(kind, inputs[0], inputs[1]);
            }
        }
        if (result == invalid || isBoolean(result) != type.getValue()) { return failure(); }
        ids.push_back(result);
    }
    SmallVector<Id> outputs;
    for (int64_t ref : roots.asArrayRef()) {
        if (ref < 0 || static_cast<uint64_t>(ref) >= ids.size()) { return failure(); }
        outputs.push_back(ids[ref]);
    }
    transaction.commit();
    return outputs;
}
#endif
