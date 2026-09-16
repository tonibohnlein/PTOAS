// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_OAHS_STORAGEFRONTIERS_H
#define PTO_TRANSFORMS_OAHS_STORAGEFRONTIERS_H
#include "PTO/Transforms/OAHS/Analysis.h"
#include <memory>

namespace mlir::pto::oahs {
struct StorageOrigin {
  std::size_t site = NoAnalysisId, operation = NoAnalysisId;
  Cut cut = NoAnalysisId;
  std::size_t context = NoAnalysisId;
};
struct StorageRelationship {
  enum Kind { RAW, WAR, WAW } kind = RAW;
  unsigned cell = 0;
  StorageOrigin source, target;
};
struct StoragePath {
  bool exists = false;
  bool definiteWriteFree = false;
  std::vector<std::size_t> sites, crossedLoopOwners;
};
struct ReaderBoundary {
  bool boundary = false;
  std::vector<StorageOrigin> readers;
  bool ambiguous() const { return boundary && !readers.empty(); }
};
// Static-origin marginal summary. No causal or event fact is represented here.
struct SuccessionSummary {
  bool preservesIncoming = true;
  std::vector<std::size_t> writers, readers;
  static SuccessionSummary sequence(const SuccessionSummary &,
                                    const SuccessionSummary &);
  static SuccessionSummary choice(const SuccessionSummary &,
                                  const SuccessionSummary &);
  static SuccessionSummary repeat(const SuccessionSummary &);
  std::pair<std::vector<std::size_t>, std::vector<std::size_t>>
  apply(const std::vector<std::size_t> &,
        const std::vector<std::size_t> &) const;
};
struct StorageFrontierStats {
  std::size_t staticSites = 0, originIncidences = 0, storageWords = 0;
  std::size_t forwardEvaluations = 0, backwardEvaluations = 0;
};
// Owns ORIGINAL effects/control; no candidate commands are an input.
// Marginal witnesses are existential paths, never completion certificates.
class StorageFrontierAnalysis {
public:
  explicit StorageFrontierAnalysis(Program);
  ~StorageFrontierAnalysis();
  StorageFrontierAnalysis(StorageFrontierAnalysis &&) noexcept;
  StorageFrontierAnalysis &operator=(StorageFrontierAnalysis &&) noexcept;
  StorageFrontierAnalysis(const StorageFrontierAnalysis &) = delete;
  StorageFrontierAnalysis &operator=(const StorageFrontierAnalysis &) = delete;
  bool complete() const;
  const std::string &reason() const;
  const StorageFrontierStats &stats() const;
  std::vector<std::size_t> sitesForOperation(std::size_t) const;
  std::vector<StorageOrigin> previousWriters(std::size_t site,
                                             unsigned cell) const;
  std::vector<StorageOrigin> previousReaders(std::size_t site,
                                             unsigned cell) const;
  std::vector<StorageOrigin> nextWriters(std::size_t site, unsigned cell) const;
  std::vector<StorageOrigin> nextReaders(std::size_t site, unsigned cell) const;
  std::vector<StorageRelationship> relationshipsAt(std::size_t site) const;
  // Positive length, including source==target through recurrence. Weak writes
  // retain incoming origins; definiteWriteFree records that loss of precision.
  StoragePath witness(std::size_t source, std::size_t target,
                      unsigned cell) const;
  ReaderBoundary readerBoundary(std::size_t site, unsigned cell, Pipe,
                                bool backward) const;
  std::vector<Cut> corridor(std::size_t site, Pipe, bool backward,
                            bool nearestOnly) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace mlir::pto::oahs
#endif
