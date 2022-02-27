
#include <omp.h>
#include "vecscatter.h"
#include "brick.h"

// #include "../out/laplacian-stencils.h"

// $START naive
__global__ void laplacian_naive(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0], bElem* dev_coeff) {
    const size_t radius = $SIZE;
    unsigned i = OFF0 + (blockIdx_x) * TILE0 + threadIdx_x;
    unsigned j = OFF1 + (blockIdx_y) * TILE1 + threadIdx_y;
    unsigned k = OFF2 + (blockIdx_z) * TILE2 + threadIdx_z;

    bElem temp = dev_coeff[0] * in[k][j][i];
    #pragma unroll
    for (int a = 1; a <= radius; a++) {
        temp += dev_coeff[a] * (
            in[k][j][i + a] + in[k][j + a][i] + in[k + a][j][i] +
            in[k][j][i - a] + in[k][j - a][i] + in[k - a][j][i]);
    }
    out[k][j][i] = temp;
}
// $END naive

// $START naive-bricks
__global__ void laplacian_naive_bricks(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem *dev_coeff) {
    unsigned b = grid[blockIdx_z + GB2][blockIdx_y + GB1][blockIdx_x + GB0];
    unsigned i = threadIdx_x;
    unsigned j = threadIdx_y;
    unsigned k = threadIdx_z;
    bOut[b][k][j][i] = dev_coeff[0] * bIn[b][k][j][i];

    const size_t radius = $SIZE;
    #pragma unroll
    for (int a = 1; a <= radius; a++) {
        bOut[b][k][j][i] += dev_coeff[a] * (
            bIn[b][k][j][i + a] + bIn[b][k][j + a][i] + bIn[b][k + a][j][i] +
            bIn[b][k][j][i - a] + bIn[b][k][j - a][i] + bIn[b][k - a][j][i]
        );
    }
}
// $END naive-bricks

// $START codegen-bricks
__global__ void laplacian_codegen_bricks(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem *dev_coeff) {
    unsigned b = grid[blockIdx_z + GB2][blockIdx_y + GB1][blockIdx_x + GB0];
    brick("$PYTHON", VSVEC, (TILE2, TILE1, TILE0), (FOLD), b);
}
// $END codegen-bricks


// $START codegen
#define bIn(a, b, c) arr_in[c][b][a]
#define bOut(a, b, c) arr_out[c][b][a]

__global__ void laplacian_codegen(bElem (*arr_in)[STRIDE1][STRIDE0], bElem (*arr_out)[STRIDE1][STRIDE0], bElem *dev_coeff) {
    long k = OFF2 + (blockIdx_z * TILE2);
    long j = OFF1 + (blockIdx_y * TILE1);
    long i = OFF0 + (blockIdx_x * VECSIZE);
    tile("$PYTHON", VSVEC, (TILE2, TILE1, VECSIZE), ("k", "j", "i"), (1, 1, VECSIZE));
}

#undef bIn
#undef bOut
// $END codegen
