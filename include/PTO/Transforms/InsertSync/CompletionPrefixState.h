// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_COMPLETIONPREFIXSTATE_H
#define PTO_TRANSFORMS_INSERTSYNC_COMPLETIONPREFIXSTATE_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <tuple>

namespace mlir::pto::insert_sync_detail {

// A small, memory-layout-independent completion analysis. Indexing by source
// lane is valid ONLY inside the single-core, straight-line, static-event domain
// checked by the native adapter. No branch/loop/path facts are invented here.
//
// The sentinel epoch represents potentially outstanding incoming work. A lane
// is not initially "complete" merely because this scan has seen no phase yet.
// Epochs denote all physical work issued on a lane; ordinary issue order does
// not advance completion. Signals capture the preceding source prefix and
// acquired knowledge. Waits transfer that knowledge to the destination.
// A second vector clock tracks action causality, not memory completion, so a
// lexically earlier wait cannot by itself justify reuse of a hardware key.
template <unsigned Lanes> class CompletionPrefixState {
public:
  using Clock = std::array<uint64_t, Lanes>;
  using Key = std::tuple<unsigned, unsigned, unsigned>;

  CompletionPrefixState() { issued.fill(1); }

  bool issue(unsigned pipe) {
    if (pipe >= Lanes || issued[pipe] == std::numeric_limits<uint64_t>::max()) {
      return false;
    }
    ++issued[pipe];
    return advance(pipe);
  }

  bool signal(unsigned source, unsigned target, unsigned id) {
    if (!validDomain(source, target)) {
      return false;
    }
    Key key{source, target, id};
    auto consumed = lastWait.find(key);
    if (tokens.count(key) ||
        (consumed != lastWait.end() &&
         causal[source][target] < consumed->second)) {
      return false;
    }
    if (!advance(source)) {
      return false;
    }
    Clock completed = known[source];
    completed[source] = std::max(completed[source], issued[source]);
    tokens.emplace(key, Token{completed, causal[source]});
    return true;
  }

  bool wait(unsigned source, unsigned target, unsigned id) {
    if (!validDomain(source, target)) {
      return false;
    }
    Key key{source, target, id};
    auto found = tokens.find(key);
    if (found == tokens.end()) {
      return false;
    }
    merge(known[target], found->second.completed);
    merge(causal[target], found->second.actions);
    if (!advance(target)) {
      return false;
    }
    tokens.erase(found);
    lastWait[key] = causal[target][target];
    return true;
  }

  bool canOmitBarrier(unsigned pipe) const {
    return pipe < Lanes && known[pipe][pipe] >= issued[pipe];
  }

  bool keepBarrier(unsigned pipe) {
    if (pipe >= Lanes) {
      return false;
    }
    known[pipe][pipe] = issued[pipe];
    return advance(pipe);
  }

  bool allEventsConsumed() const { return tokens.empty(); }

private:
  struct Token {
    Clock completed;
    Clock actions;
  };

  static bool validDomain(unsigned source, unsigned target) {
    return source < Lanes && target < Lanes && source != target;
  }

  static void merge(Clock &destination, const Clock &source) {
    for (unsigned i = 0; i < Lanes; ++i) {
      destination[i] = std::max(destination[i], source[i]);
    }
  }

  bool advance(unsigned pipe) {
    if (causal[pipe][pipe] == std::numeric_limits<uint64_t>::max()) {
      return false;
    }
    ++causal[pipe][pipe];
    return true;
  }

  Clock issued{};
  std::array<Clock, Lanes> known{};
  std::array<Clock, Lanes> causal{};
  std::map<Key, Token> tokens;
  std::map<Key, uint64_t> lastWait;
};
} // namespace mlir::pto::insert_sync_detail
#endif
