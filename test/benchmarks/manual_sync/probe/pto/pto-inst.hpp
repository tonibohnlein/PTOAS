// CPU trace scaffolding only. Never use this header to compile a device kernel.
#pragma once
#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>
#define AICORE
#define __gm__
#define __global__
using half = uint16_t;
enum pipe_t { PIPE_MTE1, PIPE_MTE2, PIPE_M, PIPE_FIX };
using event_t = unsigned;
inline unsigned probe_core;
inline unsigned get_block_idx() { return probe_core; }
inline void probe_sync(const char* kind, pipe_t src, pipe_t dst, event_t key) {
    const char* names[] = {"MTE1", "MTE2", "M", "FIX"};
    std::cout << "{\"sync\":[\"" << kind << "\",\"" << names[src] << "\",\""
              << names[dst] << "\"," << key << "]}\n";
}
inline void set_flag(pipe_t s, pipe_t d, event_t e) { probe_sync("set_flag",s,d,e); }
inline void wait_flag(pipe_t s, pipe_t d, event_t e) { probe_sync("wait_flag",s,d,e); }
namespace pto {
enum class TileType { Mat, Left, Right, Acc };
enum class BLayout { ColMajor, RowMajor };
enum class SLayout { RowMajor, ColMajor };
enum class Layout { ND, DN };
template<TileType Space, typename T, int R, int C, BLayout BL, int VR, int VC, SLayout SL>
struct Tile {
    uintptr_t addr = 0;
    void print() const {
        const char* spaces[] = {"mat", "left", "right", "acc"};
        std::cout << "[\"" << spaces[static_cast<int>(Space)] << "\"," << addr
                  << ',' << R << ',' << C << ',' << 8 * sizeof(T) << ','
                  << static_cast<int>(BL) << ',' << static_cast<int>(SL) << ']';
    }
};
template<typename T, int R, int C, int VR, int VC>
using TileLeft = Tile<TileType::Left,T,R,C,BLayout::RowMajor,VR,VC,SLayout::RowMajor>;
template<typename T, int R, int C, int VR, int VC>
using TileRight = Tile<TileType::Right,T,R,C,BLayout::RowMajor,VR,VC,SLayout::ColMajor>;
template<typename T, int R, int C, int VR, int VC>
using TileAcc = Tile<TileType::Acc,T,R,C,BLayout::ColMajor,VR,VC,SLayout::RowMajor>;
template<typename T, int R, int C, Layout L> struct TileShape2D { static constexpr int rows=R, cols=C; };
template<typename T, int R, int C, Layout L> using BaseShape2D = TileShape2D<T,R,C,L>;
inline half *probe_a, *probe_b;
inline float *probe_c;
template<typename T, typename Shape, typename Whole, Layout L> struct GlobalTensor {
    T* ptr;
    explicit GlobalTensor(T* p): ptr(p) {}
    void print() const {
        // The benchmark has three disjoint arrays; no numerical memory access.
        if constexpr (std::is_same_v<T,float>) {
            std::cout << "[\"out\"," << (ptr-probe_c)*sizeof(T);
        } else if constexpr (L == Layout::DN) {
            std::cout << "[\"b\"," << (ptr-probe_b)*sizeof(T);
        } else {
            std::cout << "[\"a\"," << (ptr-probe_a)*sizeof(T);
        }
        std::cout << ',' << Shape::rows << ',' << Shape::cols << ','
                  << (L==Layout::ND ? Whole::cols : 1)*sizeof(T) << ','
                  << (L==Layout::ND ? 1 : Whole::rows)*sizeof(T) << ']';
    }
};
template<typename T> void TASSIGN(T& t, uintptr_t a) { t.addr=a; }
template<typename T, typename G> void TLOAD(T t, G g) {
    std::cout << "{\"op\":\"tload\",\"tiles\":["; t.print();
    std::cout << "],\"gm\":"; g.print(); std::cout << "}\n";
}
template<typename G, typename T> void TSTORE(G g, T t) {
    std::cout << "{\"op\":\"tstore\",\"tiles\":["; t.print();
    std::cout << "],\"gm\":"; g.print(); std::cout << "}\n";
}
template<typename D, typename S> void TEXTRACT(D d, S s, int r, int c) {
    std::cout << "{\"op\":\"textract\",\"tiles\":["; s.print();
    std::cout << ','; d.print(); std::cout << "],\"slice\":[" << r << ',' << c << "]}\n";
}
template<typename D, typename A, typename B> void TMATMUL(D d, A a, B b) {
    std::cout << "{\"op\":\"tmatmul\",\"tiles\":["; a.print(); std::cout << ','; b.print();
    std::cout << ','; d.print(); std::cout << "]}\n";
}
template<typename D, typename C, typename A, typename B> void TMATMUL_ACC(D d, C c, A a, B b) {
    std::cout << "{\"op\":\"tmatmul.acc\",\"tiles\":["; c.print(); std::cout << ',';
    a.print(); std::cout << ','; b.print(); std::cout << ','; d.print(); std::cout << "]}\n";
}
} // namespace pto
