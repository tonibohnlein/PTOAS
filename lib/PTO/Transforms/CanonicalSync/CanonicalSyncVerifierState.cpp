// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncVerifier.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

bool containsKey(ArrayRef<VerifierEffectKey> keys,
                 const VerifierEffectKey &key) {
  return llvm::is_contained(keys, key);
}

bool containsEffect(ArrayRef<VerifierEffect> effects,
                    const VerifierEffectKey &key) {
  return llvm::any_of(
      effects, [&](const VerifierEffect &effect) { return effect.key == key; });
}

VerifierResourceState *findResource(VerifierState &state,
                                    CanonicalPhysicalResource resource) {
  auto iterator =
      llvm::find_if(state.resources, [&](const VerifierResourceState &entry) {
        return entry.resource == resource;
      });
  if (iterator == state.resources.end()) {
    state.resources.push_back({resource, {}, {}, {}, {}});
    return &state.resources.back();
  }
  return &*iterator;
}

const VerifierResourceState *findResource(const VerifierState &state,
                                          CanonicalPhysicalResource resource) {
  auto iterator =
      llvm::find_if(state.resources, [&](const VerifierResourceState &entry) {
        return entry.resource == resource;
      });
  return iterator == state.resources.end() ? nullptr : &*iterator;
}

template <typename T, typename Equal>
bool unorderedEqual(ArrayRef<T> first, ArrayRef<T> second, Equal equal) {
  return first.size() == second.size() &&
         llvm::all_of(first, [&](const T &value) {
           return llvm::any_of(
               second, [&](const T &other) { return equal(value, other); });
         });
}

bool tokenEqual(const VerifierToken &first, const VerifierToken &second) {
  return first.source == second.source && first.target == second.target &&
         first.eventId == second.eventId &&
         unorderedEqual<VerifierEffectKey>(
             first.payload, second.payload,
             [](const VerifierEffectKey &left, const VerifierEffectKey &right) {
               return left == right;
             });
}

bool cacheActionEqual(const VerifierCacheAction &first,
                      const VerifierCacheAction &second) {
  return first.operation == second.operation && first.allGm == second.allGm &&
         first.access.value == second.access.value &&
         first.access.aliasRoot == second.access.aliasRoot;
}

template <typename T, typename Equal>
void appendUnion(SmallVectorImpl<T> &result, ArrayRef<T> values, Equal equal) {
  for (const T &value : values) {
    if (!llvm::any_of(result,
                      [&](const T &other) { return equal(value, other); })) {
      result.push_back(value);
    }
  }
}

SmallVector<VerifierEffectKey, 16>
intersectKeys(ArrayRef<VerifierEffectKey> first,
              ArrayRef<VerifierEffectKey> second) {
  SmallVector<VerifierEffectKey, 16> result;
  for (const VerifierEffectKey &key : first) {
    if (containsKey(second, key)) {
      result.push_back(key);
    }
  }
  return result;
}

} // namespace

bool mlir::pto::canonical_sync_detail::operator==(const VerifierState &first,
                                                  const VerifierState &second) {
  if (!unorderedEqual<VerifierEffectKey>(
          first.globalKnown, second.globalKnown,
          [](const VerifierEffectKey &left, const VerifierEffectKey &right) {
            return left == right;
          }) ||
      !unorderedEqual<VerifierEffectKey>(
          first.globalVisible, second.globalVisible,
          [](const VerifierEffectKey &left, const VerifierEffectKey &right) {
            return left == right;
          }) ||
      !unorderedEqual<VerifierEffectKey>(
          first.cacheMaintained, second.cacheMaintained,
          [](const VerifierEffectKey &left, const VerifierEffectKey &right) {
            return left == right;
          }) ||
      !unorderedEqual<VerifierCacheAction>(first.cacheInvalidations,
                                           second.cacheInvalidations,
                                           cacheActionEqual) ||
      !unorderedEqual<VerifierEffectKey>(
          first.exitComplete, second.exitComplete,
          [](const VerifierEffectKey &left, const VerifierEffectKey &right) {
            return left == right;
          }) ||
      !unorderedEqual<VerifierEffectKey>(
          first.loopCarried, second.loopCarried,
          [](const VerifierEffectKey &left, const VerifierEffectKey &right) {
            return left == right;
          }) ||
      !unorderedEqual<VerifierToken>(first.tokens, second.tokens, tokenEqual) ||
      first.resources.size() != second.resources.size()) {
    return false;
  }
  return llvm::all_of(first.resources, [&](const VerifierResourceState &left) {
    const VerifierResourceState *right = findResource(second, left.resource);
    return right &&
           unorderedEqual<VerifierEffect>(
               left.pending, right->pending,
               [](const VerifierEffect &one, const VerifierEffect &two) {
                 return one.key == two.key;
               }) &&
           unorderedEqual<VerifierEffectKey>(
               left.pendingCompletions, right->pendingCompletions,
               [](const VerifierEffectKey &one, const VerifierEffectKey &two) {
                 return one == two;
               }) &&
           unorderedEqual<VerifierEffectKey>(
               left.known, right->known,
               [](const VerifierEffectKey &one, const VerifierEffectKey &two) {
                 return one == two;
               }) &&
           unorderedEqual<VerifierEffectKey>(
               left.visible, right->visible,
               [](const VerifierEffectKey &one, const VerifierEffectKey &two) {
                 return one == two;
               });
  });
}

VerifierState mlir::pto::canonical_sync_detail::mergeVerifierStates(
    const VerifierState &first, const VerifierState &second) {
  VerifierState result;
  for (const VerifierResourceState &source : first.resources) {
    result.resources.push_back(
        {source.resource, source.pending, source.pendingCompletions, {}, {}});
  }
  for (const VerifierResourceState &source : second.resources) {
    VerifierResourceState *destination = findResource(result, source.resource);
    for (const VerifierEffect &effect : source.pending) {
      if (!containsEffect(destination->pending, effect.key)) {
        destination->pending.push_back(effect);
      }
    }
    appendUnion<VerifierEffectKey>(
        destination->pendingCompletions, source.pendingCompletions,
        [](const VerifierEffectKey &left, const VerifierEffectKey &right) {
          return left == right;
        });
  }
  for (VerifierResourceState &destination : result.resources) {
    const VerifierResourceState *left =
        findResource(first, destination.resource);
    const VerifierResourceState *right =
        findResource(second, destination.resource);
    if (left && right) {
      SmallVector<VerifierEffectKey, 16> known =
          intersectKeys(left->known, right->known);
      SmallVector<VerifierEffectKey, 16> visible =
          intersectKeys(left->visible, right->visible);
      destination.known.assign(known.begin(), known.end());
      destination.visible.assign(visible.begin(), visible.end());
    }
  }
  for (const VerifierToken &token : first.tokens) {
    if (llvm::any_of(second.tokens, [&](const VerifierToken &other) {
          return tokenEqual(token, other);
        })) {
      result.tokens.push_back(token);
    }
  }
  result.globalKnown = intersectKeys(first.globalKnown, second.globalKnown);
  result.globalVisible =
      intersectKeys(first.globalVisible, second.globalVisible);
  result.cacheMaintained =
      intersectKeys(first.cacheMaintained, second.cacheMaintained);
  for (const VerifierCacheAction &action : first.cacheInvalidations) {
    if (llvm::any_of(second.cacheInvalidations,
                     [&](const VerifierCacheAction &other) {
                       return cacheActionEqual(action, other);
                     })) {
      result.cacheInvalidations.push_back(action);
    }
  }
  result.exitComplete = intersectKeys(first.exitComplete, second.exitComplete);
  result.loopCarried = first.loopCarried;
  appendUnion<VerifierEffectKey>(
      result.loopCarried, second.loopCarried,
      [](const VerifierEffectKey &left, const VerifierEffectKey &right) {
        return left == right;
      });
  return result;
}
