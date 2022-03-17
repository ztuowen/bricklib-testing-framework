#include <omp.h>
#include "vecscatter.h"
#include "brick.h"


// $START naive
__global__ void f2d_naive(bElem (*in)[STRIDE0], bElem (*out)[STRIDE0], bElem (*c)[8]) {
    const size_t radius = $SIZE;
    unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
    unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;

    bElem base = 0;
    #pragma unroll
    for (int i_diff = -radius; i_diff <= radius; i_diff++) {
        #pragma unroll
        for (int j_diff = -radius; j_diff <= radius; j_diff++) {
            base += (in[i + i_diff][j + j_diff] * c[i_diff + radius][j_diff + radius]);
        }
    }
    out[i][j] = base;
}
// $END naive

// $START naive-bricks
__global__ void f2d_naive_bricks(unsigned (*grid)[NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem (*c)[8]) {
    unsigned b = grid[blockIdx.y + GB1][blockIdx.x + GB0];
    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;

    bElem base = 0;
    #pragma unroll
    for (int i_diff = -radius; i_diff <= radius; i_diff++) {
        #pragma unroll
        for (int j_diff = -radius; j_diff <= radius; j_diff++) {
            base += (bIn[b][i + i_diff][j + j_diff] * c[i_diff + radius][j_diff + radius]);
        }
    }
    bOut[b][i][j] = base;
}
// $END naive-brick

// $START codegen
#define bIn(a, b) arr_in[b][a]
#define bOut(a, b) arr_out[b][a]

__global__ void f2d_codegen(bElem (*arr_in)[STRIDE0], bElem (*arr_out)[STRIDE0], bElem (*c)[8]) {
    long j = OFF1 + (blockIdx.y * TILE1);
    long i = OFF0 + (blockIdx.x * VECSIZE);
    tile("$PYTHON", VSVEC, (TILE1, VECSIZE), ("j", "i"), (1, VECSIZE));
}

#undef bIn
#undef bOut
// $END codegen

// $START codegen-bricks
__global__ void f32_codegen_bricks(unsigned (*grid)[NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem (*c)[8]) {
    unsigned b = grid[blockIdx.y + GB1][blockIdx.x + GB0];
    brick("$PYTHON", VSVEC, (TILE1, TILE0), (FOLD), b);
}
// $END codegen-bricks
