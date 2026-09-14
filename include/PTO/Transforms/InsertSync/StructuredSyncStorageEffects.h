// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#ifndef PTO_TRANSFORMS_INSERTSYNC_STRUCTUREDSYNCSTORAGEEFFECTS_H
#define PTO_TRANSFORMS_INSERTSYNC_STRUCTUREDSYNCSTORAGEEFFECTS_H
#include <cstdint>
#include <vector>

namespace mlir::pto::structured_sync::composition {
// These bits describe may-accesses, not definite initialization. An Effects
// used as a completion obligation may additionally encode a qualified
// read/read exclusion as a writer-like access. That is not a physical write.
struct History {
    uint8_t readers = 0, writers = 0;
};
using Effects = std::vector<History>;

// Discovery alone uses this projection. Every proof and completion receipt
// still uses the full obligation vector. Empty bytes means the older caller
// supplies no distinct projection; preserve its conservative behavior.
inline const Effects& physicalByteEffects(const Effects& obligations, const Effects& bytes)
{
    return bytes.empty() ? obligations : bytes;
}
inline bool validPhysicalByteEffects(const Effects& obligations, const Effects& bytes, unsigned lane)
{
    if (bytes.empty())
        return true;
    if (lane >= 7 || bytes.size() != obligations.size())
        return false;
    const uint8_t allowed = uint8_t(1u << lane);
    for (unsigned cell = 0; cell < bytes.size(); ++cell) {
        const auto b = bytes[cell], e = obligations[cell];
        if (((b.readers | b.writers) & ~allowed) || b.readers != e.readers ||
            (b.writers & ~e.writers) ||
            (e.writers & ~(b.writers | b.readers)))
            return false;
    }
    return true;
}
} // namespace mlir::pto::structured_sync::composition
#endif
