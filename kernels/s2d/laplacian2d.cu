
#include <omp.h>
#include "vecscatter.h"
#include "brick.h"

// #include "../out/laplacian-stencils.h"

// $START naive
__global__ void laplacian2d_naive(bElem (*in)[STRIDE0], bElem (*out)[STRIDE0], bElem* dev_coeff) {
    const size_t radius = $SIZE;
    unsigned i = OFF0 + (blockIdx_x) * TILE0 + threadIdx_x;
    unsigned j = OFF1 + (blockIdx_y) * TILE1 + threadIdx_y;

    bElem temp = dev_coeff[0] * in[j][i];
    #pragma unroll
    for (int a = 1; a <= radius; a++) {
        temp += dev_coeff[a] * (
            in[j][i + a] + in[j + a][i] +
            in[j][i - a] + in[j - a][i]);
    }
    out[j][i] = temp;
}
// $END naive

// $START naive-bricks
__global__ void laplacian2d_naive_bricks(unsigned (*grid)[NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem *dev_coeff) {
    unsigned b = grid[blockIdx_y + GB1][blockIdx_x + GB0];
    unsigned i = threadIdx_x;
    unsigned j = threadIdx_y;
    bOut[b][j][i] = dev_coeff[0] * bIn[b][j][i];

    const size_t radius = $SIZE;
    #pragma unroll
    for (int a = 1; a <= radius; a++) {
        bOut[b][j][i] += dev_coeff[a] * (
            bIn[b][j][i + a] + bIn[b][j + a][i] +
            bIn[b][j][i - a] + bIn[b][j - a][i]
        );
    }
}
// $END naive-bricks

// $START codegen-bricks
__global__ void laplacian2d_codegen_bricks(unsigned (*grid)[NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem *dev_coeff) {
    unsigned b = grid[blockIdx_y + GB1][blockIdx_x + GB0];
    brick("$PYTHON", VSVEC, (TILE1, TILE0), (FOLD), b);
}
// $END codegen-bricks


// $START codegen
#define bIn(a, b) arr_in[c][b]
#define bOut(a, b) arr_out[c][b]

__global__ void laplacian_codegen(bElem (*arr_in)[STRIDE0], bElem (*arr_out)[STRIDE0], bElem *dev_coeff) {
    long j = OFF1 + (blockIdx_y * TILE1);
    long i = OFF0 + (blockIdx_x * VECSIZE);
    tile("$PYTHON", VSVEC, (TILE1, VECSIZE), ("j", "i"), (1, VECSIZE));
}

#undef bIn
#undef bOut
// $END codegen
