#include <omp.h>
#include "vecscatter.h"
#include "brick.h"

// $START naive
__global__ void poisson_naive(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0]) {
    unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
    unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;
    unsigned k = OFF2 + (blockIdx.z) * TILE2 + threadIdx.z;

    out[k][j][i] = 2.666 * in[k][j][i] - 
        (0.166 * (in[k - 1][j][i] + in[k + 1][j][i] +
                in[k][j - 1][i] + in[k][j + 1][i] +
                in[k][j][i - 1] + in[k][j][i - 1])) -
        (0.0833 * (in[k - 1][j - 1][i] + in[k + 1][j - 1][i] +
                in[k - 1][j + 1][i] + in[k + 1][j + 1][i] +
                in[k - 1][j][i - 1] + in[k + 1][j][i - 1] +
                in[k][j - 1][i - 1] + in[k][j + 1][i - 1] +
                in[k - 1][j][i + 1] + in[k + 1][j][i + 1] +
                in[k][j - 1][i + 1] + in[k][j + 1][i + 1]));
}
// $END naive

// $START codegen
#define in(a, b, c) in_arr[c][b][a]
#define out(a, b, c) out_arr[c][b][a]
__global__ void poisson_codegen(bElem (*in_arr)[STRIDE1][STRIDE0], bElem (*out_arr)[STRIDE1][STRIDE0]) {
    long k = OFF2 + (blockIdx.z * TILE2);
    long j = OFF1 + (blockIdx.y * TILE1);
    long i = OFF0 + (blockIdx.x * TILE0);
    tile("$PYTHON", VSVEC, (TILE2, TILE1, VECSIZE), ("k", "j", "i"), (1, 1, VECSIZE));
}
#undef in
#undef out
// $END codegen

// $START naive-bricks
__global__ void poisson_naive_bricks(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType in, BType out) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;
    unsigned k = threadIdx.z;

    out[b][k][j][i] = 2.666 * in[b][k][j][i] - 
        (0.166 * (in[b][k - 1][j][i] + in[b][k + 1][j][i] +
                in[b][k][j - 1][i] + in[b][k][j + 1][i] +
                in[b][k][j][i - 1] + in[b][k][j][i - 1])) -
        (0.0833 * (in[b][k - 1][j - 1][i] + in[b][k + 1][j - 1][i] +
                in[b][k - 1][j + 1][i] + in[b][k + 1][j + 1][i] +
                in[b][k - 1][j][i - 1] + in[b][k + 1][j][i - 1] +
                in[b][k][j - 1][i - 1] + in[b][k][j + 1][i - 1] +
                in[b][k - 1][j][i + 1] + in[b][k + 1][j][i + 1] +
                in[b][k][j - 1][i + 1] + in[b][k][j + 1][i + 1]));
}
// $END naive-bricks

// $START codegen-bricks
__global__ void poisson_codegen_bricks(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType in, BType out) {
  unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
  brick("$PYTHON", VSVEC, (TILE2, TILE1, TILE0), (FOLD), b);
}
// $END codegen-bricks
