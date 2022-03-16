#include <omp.h>
#include "vecscatter.h"
#include "brick.h"


// $START naive
__global__ void f3d_naive(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0], bElem (*c)[8][8]) {
    const size_t radius = $SIZE;
    unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
    unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;
    unsigned k = OFF2 + (blockIdx.z) * TILE2 + threadIdx.z;

    bElem base = 0;
    #pragma unroll
    for (int i_diff = -radius; i_diff <= radius; i_diff++) {
        #pragma unroll
        for (int j_diff = -radius; j_diff <= radius; j_diff++) {
            #pragma unroll
            for (int k_diff = -radius; k_diff <= radius; k_diff++) {
                base += (in[i + i_diff][j + j_diff][k + k_diff] * c[i_diff + radius][j_diff + radius][k_diff + radius]);
            }
        }
    }
    out[i][j][k] = base;
}
// $END naive

// $START naive-bricks
__global__ void f3d_naive_bricks(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem (*c)[8][8]) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;
    unsigned k = threadIdx.z;

    bElem base = 0;
    #pragma unroll
    for (int i_diff = -radius; i_diff <= radius; i_diff++) {
        #pragma unroll
        for (int j_diff = -radius; j_diff <= radius; j_diff++) {
            #pragma unroll
            for (int k_diff = -radius; k_diff <= radius; k_diff++) {
                base += (bIn[b][i + i_diff][j + j_diff][k + k_diff] * c[i_diff + radius][j_diff + radius][k_diff + radius]);
            }
        }
    }
    bOut[b][i][j][k] = base;
}
// $END naive-brick

// $START codegen
#define bIn(a, b, c) arr_in[c][b][a]
#define bOut(a, b, c) arr_out[c][b][a]

__global__ void f3d_codegen(bElem (*arr_in)[STRIDE1][STRIDE0], bElem (*arr_out)[STRIDE1][STRIDE0], bElem (*c)[8][8]) {
    long k = OFF2 + (blockIdx.z * TILE2);
    long j = OFF1 + (blockIdx.y * TILE1);
    long i = OFF0 + (blockIdx.x * VECSIZE);
    tile("$PYTHON", VSVEC, (TILE2, TILE1, VECSIZE), ("k", "j", "i"), (1, 1, VECSIZE));
}

#undef bIn
#undef bOut
// $END codegen

// $START codegen-bricks
__global__ void f3d_codegen_bricks(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem (*c)[8][8]) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    brick("$PYTHON", VSVEC, (TILE2, TILE1, TILE0), (FOLD), b);
}
// $END codegen-bricks
