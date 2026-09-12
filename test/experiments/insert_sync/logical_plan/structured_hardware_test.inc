// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
// S7 tests compile the actual core. The finite reference below DOES NOT call
// intrinsicAccumulatorOrder and never adds hardware edges to completion/reuse.
static uint64_t hardwareModels=0,hardwareFiniteRuns=0,hardwareRequirements=0;
static MmadInfo s7Matrix(bool accum,uint64_t m=16,uint64_t n=256,uint64_t k=64,uint64_t base=0) {
    MmadInfo a;a.kind=accum?MmadInfo::Accumulate:MmadInfo::Initialize;
    a.accumulatorBase=base;a.accumulatorBytes=4*m*n;a.m=m;a.n=n;a.k=k;a.input=MmadInfo::F16;
    return a;
}
static bool s7Finite(const Model &m,const std::vector<Action> &actions,uint64_t trips) {
    ++hardwareFiniteRuns;
    auto execution=expand(m,actions,trips);
    if(!execution.valid||!execution.acyclic())return false;
    using Occ=std::pair<std::size_t,uint64_t>;
    std::vector<Occ> matrix;
    auto position=[&](Occ a) {
        const auto &p=m.atoms[a.first];
        return std::make_tuple(p.segment,a.second*m.period+p.residue,p.order);
    };
    for(const auto &p:execution.payload)if(m.atoms[p.first.first].lane==Lane{Core::AIC,Pipe::M})matrix.push_back(p.first);
    std::sort(matrix.begin(),matrix.end(),[&](auto a,auto b){return position(a)<position(b);});
    // Independent, concrete instruction-stage specification. Accumulator
    // accesses may be ordered internally while whole-M completion stays pending.
    auto full=[](const MmadInfo &a) {
        return (a.kind==MmadInfo::Initialize||a.kind==MmadInfo::Accumulate) &&
            (a.input==MmadInfo::F16||a.input==MmadInfo::BF16) &&
            a.m>0&&a.n>0&&a.k>0&&a.m<=4095&&a.n<=4095&&a.k<=4095&&
            a.m%16==0&&a.n%16==0&&a.k%16==0&&a.m*a.n>2560&&
            a.accumulatorBase%1024==0&&a.accumulatorBytes==a.m*a.n*4&&
            a.accumulatorBase<=uint64_t(INT64_MAX)-a.accumulatorBytes;
    };
    auto narrow=[&](Occ a,Occ b) {
        if(m.target.hardware!=HardwareContract::A2A3MmadAccV1)return false;
        auto first=std::find(matrix.begin(),matrix.end(),a),last=std::find(matrix.begin(),matrix.end(),b);
        if(first==matrix.end()||last==matrix.end()||first>=last)return false;
        const auto &base=m.atoms[a.first].matrix;
        if(!full(base))return false;
        for(auto next=first+1;next<=last;++next) {
            const auto &part=m.atoms[next->first].matrix;
            if(!full(part)||part.kind!=MmadInfo::Accumulate||part.accumulatorBase!=base.accumulatorBase||
               part.accumulatorBytes!=base.accumulatorBytes||part.m!=base.m||part.n!=base.n||part.input!=base.input)
                return false;
        }
        return true;
    };
    for(const auto &r:m.requirements)for(const auto &q:execution.payload)if(q.first.first==r.target)
        for(const auto &p:execution.payload)if(p.first.first==r.source && position(p.first)<position(q.first)) {
            if(execution.reaches(p.second.second,q.second.first))continue;
            if(r.property!=Property::AccumulatorUpdate || !r.hasStorageWitness ||
               r.storageBase!=m.atoms[r.source].matrix.accumulatorBase ||
               r.storageBytes!=m.atoms[r.source].matrix.accumulatorBytes ||
               !narrow(p.first,q.first))return false;
        }
    for(const auto &key:execution.episodes)for(std::size_t i=1;i<key.second.size();++i)
        if(!execution.reaches(key.second[i-1].second,key.second[i].first))return false;
    return true;
}
static void s7HardwareTests() {
    Model two;two.recurring=false;
    two.atoms={{0,0,{Core::AIC,Pipe::M},Segment::Body,s7Matrix(false)},
               {0,1,{Core::AIC,Pipe::M},Segment::Body,s7Matrix(true)}};
    hazard(two,0,1,Property::AccumulatorUpdate);
    auto baseline=construct(two);require(baseline.status==Status::Applied&&baseline.plan.barriers.size()==1,"S7 conservative baseline");
    two.target.hardware=HardwareContract::A2A3MmadAccV1;
    auto fast=construct(two);require(fast.status==Status::Applied&&fast.plan.barriers.empty(),"S7 intrinsic did not reach construction");
    require(intrinsicAccumulatorOrder(two,two.requirements[0]),"S7 narrow property unavailable");
    require(!supplies(two,fast.plan,{0,1,0,Property::Completion}),"S7 promoted accumulator order to M completion");
    require(!supplies(two,fast.plan,{0,1,0,Property::AccResource}),"S7 promoted accumulator order to general ACC ownership");
    require(!supplies(two,fast.plan,{0,1,0,Property::Visibility}),"S7 promoted accumulator order to visibility");
    require(s7Finite(two,actionsForPlan(two,fast.plan),1),"S7 instruction-stage oracle");
    auto revoked=two;revoked.target.hardware=HardwareContract::Conservative;
    require(verify(revoked,actionsForPlan(two,fast.plan)).status!=Status::Applied,"S7 stale hardware premise accepted");
    // Shape, operation mode, dtype, exact footprint and physical identity matter.
    for(unsigned mutation=0;mutation<12;++mutation) {
        auto bad=two;auto &a=bad.atoms[1].matrix;
        switch(mutation) {
        case 0:a.m=1;break;
        case 1:a.n=128;a.accumulatorBytes=4*a.m*a.n;break;
        case 2:a.n=160;a.accumulatorBytes=4*a.m*a.n;break; // threshold boundary not assumed
        case 3:a.k=0;break;
        case 4:a.k=4096;break;
        case 5:a.accumulatorBase=1024;break;
        case 6:a.accumulatorBytes-=512;break;
        case 7:a.kind=MmadInfo::Initialize;break;
        case 8:a.kind=MmadInfo::Unknown;break;
        case 9:a.input=MmadInfo::Unsupported;break;
        case 10:a.input=MmadInfo::BF16;break;
        case 11:a.accumulatorBase=uint64_t(INT64_MAX)-511;break;
        }
        require(!intrinsicAccumulatorOrder(bad,bad.requirements[0]),"S7 inapplicable hardware witness used");
        require(verify(bad,actionsForPlan(two,fast.plan)).status!=Status::Applied,"S7 missing completion hidden by bad descriptor");
    }
    // An intervening, unqualified M instruction must not be skipped. Non-M
    // work does not change which consecutive matrix calls the rule relates.
    auto interleaved=two;interleaved.atoms.push_back({0,2,{Core::AIC,Pipe::M},Segment::Body,s7Matrix(true)});
    interleaved.atoms[1].matrix={};interleaved.requirements.clear();hazard(interleaved,0,2,Property::AccumulatorUpdate);
    require(!intrinsicAccumulatorOrder(interleaved,interleaved.requirements[0]),"S7 skipped unknown intervening M operation");

    // The exact failure we must avoid: M1->M2 internal ACC order cannot release
    // the operands of M2, nor can it act as the acknowledgement of a flag.
    Model reuse;reuse.recurring=false;reuse.target.hardware=HardwareContract::A2A3MmadAccV1;
    reuse.atoms={{0,0,{Core::AIC,Pipe::MTE1}},
                 {0,1,{Core::AIC,Pipe::M},Segment::Body,s7Matrix(false)},
                 {0,2,{Core::AIC,Pipe::M},Segment::Body,s7Matrix(true)},
                 {0,3,{Core::AIC,Pipe::MTE1}},
                 {0,4,{Core::AIC,Pipe::FIX}}};
    hazard(reuse,0,1);hazard(reuse,0,2);hazard(reuse,1,2,Property::AccumulatorUpdate);
    hazard(reuse,1,3);hazard(reuse,2,3);hazard(reuse,2,4);
    auto rp=construct(reuse);require(rp.status==Status::Applied,"S7 operand-reuse model refused");
    require(s7Finite(reuse,actionsForPlan(reuse,rp.plan),1),"S7 operand/FIX release proof");
    for(Pipe target:{Pipe::MTE1,Pipe::FIX}) {
        auto actions=actionsForPlan(reuse,rp.plan);
        auto wait=std::find_if(actions.begin(),actions.end(),[&](const Action &a){return a.kind==Action::Wait&&a.source.pipe==Pipe::M&&a.target.pipe==target;});
        require(wait!=actions.end(),"S7 required release absent");actions.erase(wait);
        require(verify(reuse,actions).status!=Status::Applied&&!s7Finite(reuse,actions,1),"S7 hardware rule hid a missing release");
    }
    // Many local sequences, period and tail lengths. Numeric phase IDs are not
    // sorted order. Compare every qualifying plan to the separate finite model.
    std::mt19937 random(0x7007);
    for(unsigned test=0;test<160;++test) {
        Model m;m.recurring=test%2;m.period=1+random()%3;
        m.target.hardware=HardwareContract::A2A3MmadAccV1;
        unsigned count=2+random()%6;
        for(unsigned i=0;i<count;++i) {
            auto info=s7Matrix(m.recurring||i!=0,16,256,64);
            m.atoms.push_back({random()%m.period,i,{Core::AIC,Pipe::M},Segment::Body,info});
        }
        for(unsigned i=0;i<count;++i)for(unsigned j=0;j<count;++j)hazard(m,i,j,Property::AccumulatorUpdate);
        auto plan=construct(m);require(plan.status==Status::Applied,"S7 random matrix sequence refused");
        ++hardwareModels;hardwareRequirements+=m.requirements.size();
        for(uint64_t n:{0,1,2,3,5,8,11})require(s7Finite(m,actionsForPlan(m,plan.plan),n),"S7 random matrix order differs from finite model");
    }
    // Real-source-shaped diagnostic, still explicitly NOT a native compilation.
    auto q=s6QProjection();
    for(std::size_t i=0;i<q.model.atoms.size();++i)if(q.model.atoms[i].lane.pipe==Pipe::M)
        q.model.atoms[i].matrix=s7Matrix(i!=8,16,256,64);
    for(auto &r:q.model.requirements)if(q.model.atoms[r.source].lane.pipe==Pipe::M&&q.model.atoms[r.target].lane.pipe==Pipe::M) {
        r.property=Property::AccumulatorUpdate;
        r.hasStorageWitness=true;
        r.storageBase=q.model.atoms[r.source].matrix.accumulatorBase;
        r.storageBytes=q.model.atoms[r.source].matrix.accumulatorBytes;
    }
    q.model.target.hardware=HardwareContract::A2A3MmadAccV1;
    auto qr=construct(q.model);
    require(qr.status==Status::Applied,"S7 Q-projection matrix premise broke construction");
    auto clean=refineRegionHandoffs(q.model,qr.plan);
    require(clean.status==Status::Applied,"S7 Q-projection refinement");
    auto shared=coalesceStartupHandoffs(q.model,q.origins,clean.plan);
    require(shared.status==Status::Applied,"S7 Q-projection coalescing");
    for(uint64_t n:{0,1,2,3,7,31})require(s7Finite(q.model,actionsForPlan(q.model,shared.plan),n),"S7 Q-projection finite obligations");
    auto sites=startupEmissionSites(q.model,q.origins,actionsForPlan(q.model,shared.plan));
    require(sites&&shared.plan.barriers.size()<7,"S7 Q-projection barriers not improved");
    std::cerr<<"S7_QPROJ_MODEL handoffs="<<shared.plan.handoffs.size()<<" barriers="<<shared.plan.barriers.size()
             <<" first_barriers="<<shared.plan.firstBarriers.size()<<" sites_excluding_retirement="<<sites->size()<<"\n";
    std::cerr<<"S7_HARDWARE models="<<hardwareModels<<" requirements="<<hardwareRequirements
             <<" finite_checks="<<hardwareFiniteRuns<<"\n";
}
