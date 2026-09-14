// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "PTO/Transforms/InsertSync/StructuredSyncLifetimeSummary.h"
#include "PTO/Transforms/InsertSync/StructuredSyncStorageEffects.h"
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <set>
namespace c = mlir::pto::structured_sync::composition;
static unsigned assertions = 0;
#define CHECK(x) do { ++assertions; if (!(x)) { std::cerr << "check failed: " #x " at " << __LINE__ << '\n'; std::abort(); } } while (false)
struct Budget {
    uint64_t limit = UINT64_MAX, used = 0;
    bool operator()(uint64_t amount) {
        if (amount > limit - used) return false;
        used += amount;
        return true;
    }
};
static c::AccessFrontier cut(unsigned id) {
    c::AccessFrontier result;
    result.cuts[0] = id; result.count = 1; result.mayEmpty = false;
    return result;
}
static c::StorageLifetimeSummary summary(unsigned scope, unsigned cell = 0) {
    c::StorageLifetimeSummary s;
    s.scope = scope; s.entry = 0; s.exit = 15; s.producer = 2;
    s.readers = 1u << 6; s.cells = {cell};
    s.firstWrite = cut(3); s.lastWrite = cut(5);
    s.firstRead[6] = cut(10); s.lastRead[6] = cut(10);
    return s;
}
int main() {
    std::vector<unsigned> parent(16, 0), position(16);
    std::vector<uint8_t> sequence(16, 0); sequence[0] = 1;
    for (unsigned i=0; i<16; ++i) position[i]=i;
    {
        Budget b;
        c::LifetimeSummaryIndex index;
        std::vector<c::StorageLifetimeSummary> values;
        CHECK(index.insert(summary(1,0), values,b));
        CHECK(index.insert(summary(1,1), values,b));
        CHECK(values.size()==1 && values[0].cells==std::vector<unsigned>({0,1}));
        CHECK(index.insert(summary(2,2), values,b));
        CHECK(values.size()==2);
        auto split=summary(1,3); split.firstRead[6]=cut(11);
        CHECK(index.insert(split,values,b)); CHECK(values.size()==3);
        CHECK(c::LifetimeSummaryIndex::mergeProducerEnvelopes(values,parent,position,sequence,b));
        CHECK(values.size()==3); // neither a different scope nor reader can merge
    }
    {
        Budget b; c::LifetimeSummaryIndex index;
        std::vector<c::StorageLifetimeSummary> values;
        auto a=summary(1,0), d=summary(1,1);
        d.firstWrite=cut(2); d.lastWrite=cut(7);
        CHECK(index.insert(a,values,b)); CHECK(index.insert(d,values,b));
        CHECK(values.size()==2);
        CHECK(c::LifetimeSummaryIndex::mergeProducerEnvelopes(values,parent,position,sequence,b));
        CHECK(values.size()==1 && values[0].cells.size()==2);
        CHECK(values[0].firstWrite.cuts[0]==2 && values[0].lastWrite.cuts[0]==7);
    }
    {
        Budget b; c::LifetimeSummaryIndex index;
        std::vector<c::StorageLifetimeSummary> values;
        auto a=summary(1,0), d=summary(1,1); d.firstWrite=cut(2);
        CHECK(index.insert(a,values,b)); CHECK(index.insert(d,values,b));
        auto otherParent=parent; otherParent[2]=1; auto otherSequence=sequence; otherSequence[1]=1;
        CHECK(c::LifetimeSummaryIndex::mergeProducerEnvelopes(values,otherParent,position,otherSequence,b));
        CHECK(values.size()==2); // cross-control-domain envelopes are forbidden
    }
    {
        Budget b; c::LifetimeSummaryIndex index;
        std::vector<c::StorageLifetimeSummary> values;
        auto a=summary(1), d=summary(1,1); d.maySkip=false;
        CHECK(index.insert(a,values,b)); CHECK(index.insert(d,values,b));
        CHECK(c::LifetimeSummaryIndex::mergeProducerEnvelopes(values,parent,position,sequence,b));
        CHECK(values.size()==2);
    }
    {
        Budget b{0,0}; c::LifetimeSummaryIndex index;
        std::vector<c::StorageLifetimeSummary> values;
        CHECK(!index.insert(summary(1),values,b)); CHECK(values.empty());
        Budget enough; CHECK(index.insert(summary(1),values,enough));
        CHECK(!c::LifetimeSummaryIndex::mergeProducerEnvelopes(values,parent,position,sequence,b));
    }
    {
        Budget b; c::LifetimeSummaryIndex index;
        std::vector<c::StorageLifetimeSummary> values;
        auto malformed=summary(1); malformed.lastRead[6].count=9;
        CHECK(!index.insert(malformed,values,b)); CHECK(values.empty());
        malformed=summary(1); malformed.firstWrite.valid=false;
        CHECK(!index.insert(malformed,values,b));
    }
    // ACC exclusion stays in the obligation vector: only discovery sees FIX
    // as a byte reader rather than a fictitious second writer.
    {
        const unsigned fix=6;
        c::Effects requirements(1), bytes(1);
        requirements[0]={uint8_t(1u<<fix),uint8_t(1u<<fix)};
        bytes[0]={uint8_t(1u<<fix),0};
        CHECK(c::validPhysicalByteEffects(requirements,bytes,fix));
        CHECK(c::physicalByteEffects(requirements,bytes)[0].writers==0);
        CHECK(requirements[0].writers==(1u<<fix));
        CHECK(&c::physicalByteEffects(requirements,{})==&requirements);
        auto invalid=bytes; invalid[0].writers=1u<<2;
        CHECK(!c::validPhysicalByteEffects(requirements,invalid,fix));
        invalid=bytes; invalid[0].readers=0;
        CHECK(!c::validPhysicalByteEffects(requirements,invalid,fix));
        c::Effects trueWrite(1); trueWrite[0].writers=1u<<fix;
        CHECK(!c::validPhysicalByteEffects(trueWrite,bytes,fix));
        CHECK(!c::validPhysicalByteEffects(requirements,c::Effects(2),fix));
    }
    // Compare the indexed merge with an independent direct specification on
    // bounded randomized populations; preserve every witness and endpoint.
    std::mt19937 random(7142026);
    for (unsigned trial=0;trial<128;++trial) {
        Budget b; c::LifetimeSummaryIndex index;
        std::vector<c::StorageLifetimeSummary> values;
        std::map<unsigned,std::set<unsigned>> expected;
        std::map<unsigned,std::pair<unsigned,unsigned>> extents;
        for (unsigned i=0;i<64;++i) {
            unsigned scope=1+random()%12;
            auto s=summary(scope,i);
            s.firstWrite=cut(2+random()%3); s.lastWrite=cut(5+random()%3);
            expected[scope].insert(i);
            auto it=extents.find(scope);
            if(it==extents.end()) extents[scope]={s.firstWrite.cuts[0],s.lastWrite.cuts[0]};
            else {
                it->second.first=std::min(it->second.first,s.firstWrite.cuts[0]);
                it->second.second=std::max(it->second.second,s.lastWrite.cuts[0]);
            }
            CHECK(index.insert(s,values,b));
        }
        CHECK(c::LifetimeSummaryIndex::mergeProducerEnvelopes(values,parent,position,sequence,b));
        CHECK(values.size()==expected.size());
        for(const auto& s:values) {
            CHECK(std::set<unsigned>(s.cells.begin(),s.cells.end())==expected[s.scope]);
            CHECK(s.firstWrite.cuts[0]==extents[s.scope].first);
            CHECK(s.lastWrite.cuts[0]==extents[s.scope].second);
        }
    }
    // Same-cell sibling scopes cannot merge. This is the old quadratic case.
    for(unsigned count:{1000u,4000u,16000u}) {
        Budget b; c::LifetimeSummaryIndex index;
        std::vector<c::StorageLifetimeSummary> values;
        for(unsigned i=0;i<count;++i) CHECK(index.insert(summary(i),values,b));
        CHECK(c::LifetimeSummaryIndex::mergeProducerEnvelopes(values,parent,position,sequence,b));
        CHECK(values.size()==count);
        CHECK(b.used < uint64_t(count)*30000);
        std::cout << "{\"summaries\":" << count << ",\"reserved_work\":" << b.used << "}\n";
    }
    {
        Budget b{1u<<24,0}; c::LifetimeSummaryIndex index;
        std::vector<c::StorageLifetimeSummary> values;
        unsigned added=0;
        while(added<16000 && index.insert(summary(added),values,b)) ++added;
        CHECK(added<16000); CHECK(b.used<=b.limit);
        // This is the caller's transactional boundary on any failed reserve.
        values.clear(); CHECK(values.empty());
    }
    std::cout << "PASS " << assertions << " production-index/projection assertions\n";
}
