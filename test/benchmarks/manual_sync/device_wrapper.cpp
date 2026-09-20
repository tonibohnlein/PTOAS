// ARM_SOURCE is either original upstream C++ or one already-synchronized C++ arm.
#include ARM_SOURCE
#ifndef ORIGINAL_ARM
__global__ AICORE void KERNEL_ENTRY(__gm__ float* c, __gm__ half* a, __gm__ half* b) {
    reference_gemm(c, a, b, static_cast<int32_t>(block_idx));
}
#endif
extern "C" void LaunchReference(void* c, void* a, void* b, void* stream) {
#ifdef ORIGINAL_ARM
    GemmPerformance<float, BENCH_CORES, BENCH_M, BENCH_K, BENCH_N,
        BENCH_CM, BENCH_K, BENCH_CN, 128,64,256,1,4,4,1>
        <<<BENCH_CORES,nullptr,stream>>>(static_cast<__gm__ uint8_t*>(c),
            static_cast<__gm__ uint8_t*>(a),static_cast<__gm__ uint8_t*>(b));
#else
    KERNEL_ENTRY<<<BENCH_CORES,nullptr,stream>>>(static_cast<__gm__ float*>(c),
        static_cast<__gm__ half*>(a),static_cast<__gm__ half*>(b));
#endif
}
