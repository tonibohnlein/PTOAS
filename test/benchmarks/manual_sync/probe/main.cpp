// Execute the unchanged upstream C++ control with non-numerical trace intrinsics.
#define __COSTMODEL
#include ORIGINAL_SOURCE
#include <cstdlib>
#include <memory>
int main(int argc, char** argv) {
    probe_core = argc > 1 ? std::stoul(argv[1]) : 0;
    constexpr unsigned cores = (BENCH_M / BENCH_CM) * (BENCH_N / BENCH_CN);
    if (probe_core >= cores) return 2;
    // Real arrays keep the source's pointer arithmetic defined. No data touched.
    std::unique_ptr<half[]> a(new half[size_t(BENCH_M)*BENCH_K]);
    std::unique_ptr<half[]> b(new half[size_t(BENCH_K)*BENCH_N]);
    std::unique_ptr<float[]> c(new float[size_t(BENCH_M)*BENCH_N]);
    probe_a=a.get(); probe_b=b.get(); probe_c=c.get();
    RunGemmE2E<float,half,half,float,cores,BENCH_M,BENCH_K,BENCH_N,
        BENCH_M,BENCH_K,BENCH_N,BENCH_CM,BENCH_K,BENCH_CN,128,64,256,1,4,4,1>(
        c.get(),a.get(),b.get());
}
