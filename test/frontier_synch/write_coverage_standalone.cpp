// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-2.0
// Dependency-free core test. The runner supplies only a minimal OriginalStructure
// DTO; the coverage and partition algorithms themselves are production headers.
// This is NOT the native importer or either synchronization pass.
#include "PTO/Transforms/FrontierSynch/StorageWitnesses.h"
#include <cstdlib>
#include <iostream>
#include <set>

namespace fs = mlir::pto::frontiersynch;
static std::size_t checks = 0;
static void check(bool ok)
{
    ++checks;
    if (!ok) {
        std::cerr << "check failed: " << checks << '\n';
        std::exit(1);
    }
}
int main()
{
    // Existing synthetic fixtures omit the new evidence field. Preserve their
    // aggregate-initializer and warning behavior when the shared record grows.
    [[maybe_unused]] fs::FootprintUse oldUse{0, false, true, true};
    [[maybe_unused]] fs::Access oldAccess{0, true, false, false};
    // Independent concrete byte enumeration, including empty rectangles and
    // gaps. Neither factory nor covers() is used to compute the oracle.
    for (uint64_t base = 0; base < 4; ++base) {
        for (uint64_t blocks = 0; blocks <= 5; ++blocks) {
            for (uint64_t width = 0; width <= 5; ++width) {
                for (uint64_t gap = 0; gap <= 3; ++gap) {
                    const uint64_t stride = width + gap;
                    auto p = fs::StridedWriteRange::get(base, blocks, width, stride);
                    check(bool(p));
                    std::set<uint64_t> written;
                    for (uint64_t i = 0; i < blocks; ++i) {
                        for (uint64_t j = 0; j < width; ++j) {
                            written.insert(base + i * stride + j);
                        }
                    }
                    for (uint64_t begin = 0; begin <= 48; ++begin) {
                        for (uint64_t size = 0; size <= 48; ++size) {
                            bool all = size != 0;
                            for (uint64_t byte = begin; byte < begin + size && all; ++byte) {
                                all = written.count(byte);
                            }
                            check(p->covers(begin, size) == all);
                        }
                    }
                }
            }
        }
    }
    const auto max = std::numeric_limits<uint64_t>::max();
    check(!fs::StridedWriteRange::get(max, 1, 1, 1));
    check(!fs::StridedWriteRange::get(0, max, 2, 2));
    check(!fs::StridedWriteRange::get(max - 15, 2, 8, 8));
    check(!fs::StridedWriteRange::get(0, 2, 8, 7));
    auto end = fs::StridedWriteRange::get(max - 16, 2, 8, 8);
    check(bool(end) && end->covers(max - 16, 16));
    check(!end->covers(max - 16, 17));
    check(!fs::StridedWriteRange{0, max, 2, 2}.covers(0, 1));
    // A huge rectangular domain is represented in constant space.
    auto huge = fs::StridedWriteRange::get(0, uint64_t(1) << 40, 16, 32);
    check(bool(huge) && huge->covers((uint64_t(1) << 40) * 32 - 32, 16));
    check(!huge->covers(15, 2));

    // The same broad may footprint has per-cell coverage, not one operation-wide
    // boolean. A third, unknown alias creates witnesses but cannot merge atoms.
    fs::OriginalStructure original;
    original.operations.resize(5);
    fs::FootprintGroup broad, lo, hi, unknown;
    broad.description.addressSpace = "vec";
    broad.description.coordinateSpace = "physical-local";
    broad.description.ranges = {{0, 32}};
    fs::WriteCoverage prefix{fs::WriteCoverage::Status::Qualified, *fs::StridedWriteRange::get(0, 1, 16, 16)};
    fs::WriteCoverage full{fs::WriteCoverage::Status::Qualified, *fs::StridedWriteRange::get(0, 1, 32, 32)};
    broad.uses.push_back({0, false, true, false, nullptr, fs::NoControlId, prefix});
    broad.uses.push_back({1, true, true, false, nullptr, fs::NoControlId, full});
    lo.description = broad.description;
    lo.description.ranges = {{0, 16}};
    lo.uses.push_back({2, true, false, false, nullptr, fs::NoControlId, {}});
    hi.description = broad.description;
    hi.description.ranges = {{16, 16}};
    hi.uses.push_back({3, true, false, false, nullptr, fs::NoControlId, {}});
    unknown.description.addressSpace = "vec";
    unknown.description.unknownRange = true;
    // Even a supplied footprint-wide proof cannot strengthen an alias witness.
    unknown.uses.push_back({4, true, true, true, nullptr, fs::NoControlId, full});
    const auto counts = fs::appendCanonicalStorage(original, {broad, lo, hi, unknown}, [](auto, auto) { return true; });
    check(counts.canonicalCells == 2 && counts.pairCells == 3);
    for (std::size_t op = 0; op < original.operations.size(); ++op) {
        for (const auto& access : original.operations[op].accesses) {
            const auto& cell = original.cells[access.cell];
            const bool exact = cell.storage == fs::Cell::Storage::CanonicalInterval;
            if (!exact) {
                check(!access.definiteWrite);
            } else if (op == 0) {
                check(access.definiteWrite == (cell.ranges.front().first == 0));
            } else if (op == 1) {
                check(access.read && access.write && access.definiteWrite);
            } else {
                check(access.read && !access.write && !access.definiteWrite);
            }
        }
    }
    std::cout << "coverage-and-partition: " << checks << " checks passed\n";
}
