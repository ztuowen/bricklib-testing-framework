#include <omp.h>
#include "vecscatter.h"
#include "brick.h"

// $START naive
__global__ void helmholtz2_naive(bElem (*x)[STRIDE1][STRIDE0], bElem (*alpha)[STRIDE1][STRIDE0], bElem (*beta_i)[STRIDE1][STRIDE0], bElem (*beta_j)[STRIDE1][STRIDE0], bElem (*beta_k)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0], bElem *c) {
    unsigned i = OFF0 + (blockIdx_x) * TILE0 + threadIdx_x;
    unsigned j = OFF1 + (blockIdx_y) * TILE1 + threadIdx_y;
    unsigned k = OFF2 + (blockIdx_z) * TILE2 + threadIdx_z;

    bElem c1 = c[0];
    bElem c2 = c[1];
    bElem h2inv = c[2];

    out[k][j][i] = c1 * alpha[k][j][i] * x[k][j][i] - 
        c2 * h2inv * (
            beta_i[k][j][i + 1] * (x[k][j][i + 1] - x[k][j][i]) + 
            beta_i[k][j][i]     * (x[k][j][i - 1] - x[k][j][i]) +
            beta_j[k][j + 1][i] * (x[k][j + 1][i] - x[k][j][i]) +
            beta_j[k][j - 1][i] * (x[k][j - 1][i] - x[k][j][i]) +
            beta_k[k + 1][j][i] * (x[k + 1][j][i] - x[k][j][i]) +
            beta_k[k - 1][j][i] * (x[k - 1][j][i] - x[k][j][i])
        );
}
// $END naive

// $START codegen
#define x(a, b, c) x_arr[c][b][a]
#define alpha(a, b, c) alpha_arr[c][b][a]
#define beta_i(a, b, c) beta_i_arr[c][b][a]
#define beta_j(a, b, c) beta_j_arr[c][b][a]
#define beta_k(a, b, c) beta_k_arr[c][b][a] 
#define out(a, b, c) out_arr[c][b][a]

__global__ void helmholtz2_codegen(bElem (*x_arr)[STRIDE1][STRIDE0], bElem (*alpha_arr)[STRIDE1][STRIDE0], bElem (*beta_i_arr)[STRIDE1][STRIDE0], bElem (*beta_j_arr)[STRIDE1][STRIDE0], bElem (*beta_k_arr)[STRIDE1][STRIDE0], bElem (*out_arr)[STRIDE1][STRIDE0], bElem *c) {
    unsigned i = OFF0 + (blockIdx_x) * TILE0 + threadIdx_x;
    unsigned j = OFF1 + (blockIdx_y) * TILE1 + threadIdx_y;
    unsigned k = OFF2 + (blockIdx_z) * TILE2 + threadIdx_z;

    tile("$PYTHON", VSVEC, (TILE2, TILE1, VECSIZE), ("k", "j", "i"), (1, 1, VECSIZE));
}

#undef x
#undef alpha
#undef beta_i
#undef beta_j
#undef beta_k
#undef out
// $END codegen

// $START naive-bricks
__global__ void helmholtz2_naive_bricks(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType x, BType alpha, BType beta_i, BType beta_j, BType beta_k, BType out, bElem *c) {
    unsigned b = grid[blockIdx_z + GB2][blockIdx_y + GB1][blockIdx_x + GB0];
    unsigned i = threadIdx_x;
    unsigned j = threadIdx_y;
    unsigned k = threadIdx_z;

    bElem c1 = c[0];
    bElem c2 = c[1];
    bElem h2inv = c[2];

    out[b][k][j][i] = c1 * alpha[b][k][j][i] * x[b][k][j][i] - 
        c2 * h2inv * (
            beta_i[b][k][j][i + 1] * (x[b][k][j][i + 1] - x[b][k][j][i]) + 
            beta_i[b][k][j][i]     * (x[b][k][j][i - 1] - x[b][k][j][i]) +
            beta_j[b][k][j + 1][i] * (x[b][k][j + 1][i] - x[b][k][j][i]) +
            beta_j[b][k][j - 1][i] * (x[b][k][j - 1][i] - x[b][k][j][i]) +
            beta_k[b][k + 1][j][i] * (x[b][k + 1][j][i] - x[b][k][j][i]) +
            beta_k[b][k - 1][j][i] * (x[b][k - 1][j][i] - x[b][k][j][i])
        );
}
// $END naive-bricks

// $START codegen-bricks
__global__ void helmholtz2_codegen_bricks(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType x, BType alpha, BType beta_i, BType beta_j, BType beta_k, BType out, bElem *c) {
  unsigned b = grid[blockIdx_z + GB2][blockIdx_y + GB1][blockIdx_x + GB0];
  brick("$PYTHON", VSVEC, (TILE2, TILE1, TILE0), (FOLD), b);
}
// $END codegen-bricks
