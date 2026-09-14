// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#ifndef PTO_TRANSFORMS_INSERTSYNC_STRUCTUREDSYNCLIFETIMESUMMARY_H
#define PTO_TRANSFORMS_INSERTSYNC_STRUCTUREDSYNCLIFETIMESUMMARY_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace mlir::pto::structured_sync::composition {
constexpr unsigned LaneCount = 7;

struct AccessFrontier {
    std::array<unsigned, 8> cuts{};
    unsigned count = 0;
    bool mayEmpty = true, valid = true;
};
struct StorageLifetimeSummary {
    unsigned scope = ~0u, entry = ~0u, exit = ~0u;
    std::vector<unsigned> cells;
    unsigned producer = ~0u;
    AccessFrontier firstWrite, lastWrite;
    std::array<AccessFrontier, LaneCount> firstRead, lastRead;
    uint8_t readers = 0;
    bool maySkip = true;
    bool wholeProgram = false;
};

// An exact summary index and a separately bucketed producer-envelope merge.
// Charge is bool(uint64_t), supplied by LifetimeDiscovery's one allowance.
// No comparison stops halfway through a map operation: reserve the red/black
// tree's worst-case comparison work BEFORE lookup/insertion. Failure discards
// the optional discovery; callers must not use its partially merged result.
class LifetimeSummaryIndex {
    using Key = std::vector<unsigned>;
    std::map<Key, unsigned> exact;
    static constexpr unsigned NoCut = ~0u;
    // Fields in the fixed part and maximum encoded frontiers (count/flags/cuts).
    static constexpr uint64_t MaxSignature = 9 + (2 + 2 * LaneCount) * 11;

    template <typename Charge>
    static bool reserveLookup(size_t count, size_t width, Charge& charge)
    {
        uint64_t logarithm = 0;
        for (size_t n = count; n; n >>= 1)
            ++logarithm;
        // Two searches (find then emplace), comparisons and node/key copying.
        return charge((4 * logarithm + 12) * uint64_t(width + 1));
    }
    static bool validFrontier(const AccessFrontier& f)
    {
        return f.valid && f.count <= f.cuts.size();
    }
    static void append(Key& key, const AccessFrontier& f)
    {
        key.push_back(f.count);
        key.push_back(f.mayEmpty);
        key.push_back(f.valid);
        key.insert(key.end(), f.cuts.begin(), f.cuts.begin() + f.count);
    }
    static Key common(const StorageLifetimeSummary& s)
    {
        Key key{s.scope, s.entry, s.exit, s.producer, s.readers,
                unsigned(s.maySkip), unsigned(s.wholeProgram)};
        for (unsigned reader = 0; reader < LaneCount; ++reader) {
            append(key, s.firstRead[reader]);
            append(key, s.lastRead[reader]);
        }
        return key;
    }
    static bool valid(const StorageLifetimeSummary& s)
    {
        if (s.scope == NoCut || s.entry == NoCut || s.exit == NoCut ||
            s.producer >= LaneCount || !s.readers ||
            (s.readers & ~((1u << LaneCount) - 1)) ||
            (s.readers & (1u << s.producer)) ||
            !validFrontier(s.firstWrite) || !validFrontier(s.lastWrite))
            return false;
        for (unsigned reader = 0; reader < LaneCount; ++reader)
            if (!validFrontier(s.firstRead[reader]) || !validFrontier(s.lastRead[reader]))
                return false;
        return true;
    }
    static Key exactKey(const StorageLifetimeSummary& s)
    {
        Key key = common(s);
        append(key, s.firstWrite);
        append(key, s.lastWrite);
        return key;
    }
    static bool producerParents(Key& key, const AccessFrontier& f,
                                const std::vector<unsigned>& parent,
                                const std::vector<uint8_t>& sequence)
    {
        key.push_back(f.count);
        key.push_back(f.mayEmpty);
        for (unsigned i = 0; i < f.count; ++i) {
            unsigned cut = f.cuts[i];
            if (cut >= parent.size() || parent[cut] >= sequence.size() || !sequence[parent[cut]])
                return false;
            key.push_back(parent[cut]);
        }
        return true;
    }
    static void envelope(AccessFrontier& into, const AccessFrontier& other,
                         const std::vector<unsigned>& position, bool earliest)
    {
        for (unsigned i = 0; i < into.count; ++i) {
            bool before = position[other.cuts[i]] < position[into.cuts[i]];
            if (before == earliest)
                into.cuts[i] = other.cuts[i];
        }
    }
public:
    template <typename Charge>
    bool insert(StorageLifetimeSummary summary, std::vector<StorageLifetimeSummary>& result, Charge& charge)
    {
        // Frontier validation/signature construction is bounded, including
        // unused reader entries. Input cells are copied only after charging.
        if (!charge(MaxSignature + summary.cells.size()) || !valid(summary))
            return false;
        auto key = exactKey(summary);
        if (!reserveLookup(exact.size(), key.size(), charge))
            return false;
        auto found = exact.find(key);
        if (found != exact.end()) {
            auto& cells = result[found->second].cells;
            cells.insert(cells.end(), summary.cells.begin(), summary.cells.end());
        } else {
            exact.emplace(std::move(key), unsigned(result.size()));
            result.push_back(std::move(summary));
        }
        return true;
    }

    template <typename Charge>
    static bool mergeProducerEnvelopes(std::vector<StorageLifetimeSummary>& input,
                                      const std::vector<unsigned>& parent,
                                      const std::vector<unsigned>& position,
                                      const std::vector<uint8_t>& sequence,
                                      Charge& charge)
    {
        if (parent.size() != position.size() || parent.size() != sequence.size() ||
            !charge(input.size()))
            return false;
        std::map<Key, unsigned> buckets;
        std::vector<StorageLifetimeSummary> merged;
        merged.reserve(input.size());
        for (auto& summary : input) {
            if (!charge(MaxSignature + summary.cells.size()) || !valid(summary))
                return false;
            Key key = common(summary);
            bool eligible = producerParents(key, summary.firstWrite, parent, sequence) &&
                            producerParents(key, summary.lastWrite, parent, sequence);
            if (!eligible) {
                // An unmergeable frontier remains a valid individual family.
                merged.push_back(std::move(summary));
                continue;
            }
            if (!reserveLookup(buckets.size(), key.size(), charge))
                return false;
            auto found = buckets.find(key);
            if (found == buckets.end()) {
                buckets.emplace(std::move(key), unsigned(merged.size()));
                merged.push_back(std::move(summary));
            } else {
                auto& into = merged[found->second];
                envelope(into.firstWrite, summary.firstWrite, position, true);
                envelope(into.lastWrite, summary.lastWrite, position, false);
                into.cells.insert(into.cells.end(), summary.cells.begin(), summary.cells.end());
            }
        }
        input = std::move(merged);
        return true;
    }
};
} // namespace mlir::pto::structured_sync::composition
#endif
