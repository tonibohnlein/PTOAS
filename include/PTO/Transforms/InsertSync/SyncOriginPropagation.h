// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_INSERTSYNC_SYNCORIGINPROPAGATION_H
#define PTO_INSERTSYNC_SYNCORIGINPROPAGATION_H
#include <cstddef>
#include <deque>
#include <vector>

namespace mlir::pto {
// Shared finite root-set propagation, independent of MLIR. Node supplies users,
// roots and unknown. Each newly added (node, root) fact crosses each outgoing
// edge once; unknown propagates once too. No iteration/trip/work cutoff changes
// the result. An origin-less cycle remains empty for the caller to classify as
// unknown; it is never interpreted as proof of non-aliasing.
template <typename Node>
std::size_t propagateSyncOrigins(std::vector<Node> &nodes) {
  std::vector<std::vector<unsigned>> pending(nodes.size());
  std::vector<bool> pendingUnknown(nodes.size()), queued(nodes.size());
  std::deque<unsigned> queue;
  for (unsigned i = 0; i < nodes.size(); ++i) {
    pending[i].assign(nodes[i].roots.begin(), nodes[i].roots.end());
    pendingUnknown[i] = nodes[i].unknown;
    if (!pending[i].empty() || pendingUnknown[i]) {
      queue.push_back(i); queued[i] = true;
    }
  }
  std::size_t visits = 0;
  while (!queue.empty()) {
    const auto i = queue.front(); queue.pop_front(); queued[i] = false;
    std::vector<unsigned> delta; delta.swap(pending[i]);
    const bool unknown = pendingUnknown[i]; pendingUnknown[i] = false;
    for (unsigned user : nodes[i].users) {
      auto &target = nodes.at(user);
      for (unsigned root : delta) {
        ++visits;
        if (target.roots.insert(root).second) pending[user].push_back(root);
      }
      if (unknown && !target.unknown) {
        target.unknown = true; pendingUnknown[user] = true;
      }
      if ((!pending[user].empty() || pendingUnknown[user]) && !queued[user]) {
        queue.push_back(user); queued[user] = true;
      }
    }
  }
  return visits;
}
} // namespace mlir::pto
#endif
