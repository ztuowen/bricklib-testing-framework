#include <omp.h>
#include "vecscatter.h"
#include "brick.h"

// $START naive
__global__ void chebyshev_naive(bElem (*Ac)[STRIDE1][STRIDE0], bElem (*Ap)[STRIDE1][STRIDE0], bElem (*Dinv)[STRIDE1][STRIDE0], bElem (*RHS)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0], bElem *c, bElem (*coeff)[4][4]) {
  const size_t radius = $SIZE;
  unsigned i = OFF0 + (blockIdx_x) * TILE0 + threadIdx_x;
  unsigned j = OFF1 + (blockIdx_y) * TILE1 + threadIdx_y;
  unsigned k = OFF2 + (blockIdx_z) * TILE2 + threadIdx_z;

  bElem base = 0;
  #pragma unroll
  for (int i_diff = -radius; i_diff <= radius; i_diff++) {
    #pragma unroll
    for (int j_diff = -radius; j_diff <= radius; j_diff++) {
      #pragma unroll
      for (int k_diff = -radius; k_diff <= radius; k_diff++) {
	base += coeff[i_diff + radius][j_diff + radius][k_diff + radius] * (Ac[i + i_diff][j + j_diff][k + k_diff]);
      }
    }
  }
  out[i][j][k] = Ac[i][j][k] + c[0] * (Ac[i][j][k] + Ap[i][j][k]) + c[1] * Dinv[i][j][k] *
    (RHS[i][j][k] + (Ac[i][j][k] + c[2] * base));
}
// $END naive

// $START naive-bricks
__global__ void chebyshev_naive_bricks(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType Ac, BType Ap, BType Dinv, BType RHS, BType out, bElem *c, bElem (*coeff)[4][4]) {
  const size_t radius = $SIZE;
  unsigned b = grid[blockIdx_z + GB2][blockIdx_y + GB1][blockIdx_x + GB0];
  unsigned i = threadIdx_x;
  unsigned j = threadIdx_y;
  unsigned k = threadIdx_z;

  bElem base = 0;
  #pragma unroll
  for (int i_diff = -radius; i_diff <= radius; i_diff++) {
    #pragma unroll
    for (int j_diff = -radius; j_diff <= radius; j_diff++) {
      #pragma unroll
      for (int k_diff = -radius; k_diff <= radius; k_diff++) {
	base += coeff[i_diff + radius][j_diff + radius][k_diff + radius] * (Ac[b][i + i_diff][j + j_diff][k + k_diff]);
      }
    }
  }
  out[b][i][j][k] = Ac[b][i][j][k] + c[0] * (Ac[b][i][j][k] + Ap[b][i][j][k]) + c[1] * Dinv[b][i][j][k] * (RHS[b][i][j][k] + (Ac[b][i][j][k] + c[2] * base));
}
// $END naive-bricks

// $START codegen-bricks
__global__ void chebyshev_codegen_bricks(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType Ac, BType Ap, BType Dinv, BType RHS, BType out, bElem *c, bElem (*coeff)[4][4]) {
  unsigned b = grid[blockIdx_z + GB2][blockIdx_y + GB1][blockIdx_x + GB0];
  brick("$PYTHON", VSVEC, (TILE2, TILE1, TILE0), (FOLD), b);
}
// $END codegen-bricks

// $START codegen
#define Ac(a, b, c) Ac_arr[c][b][a]
#define Ap(a, b, c) Ap_arr[c][b][a]
#define Dinv(a, b, c) Dinv_arr[c][b][a]
#define RHS(a, b, c) RHS_arr[c][b][a]
#define out(a, b, c) out_arr[c][b][a]

__global__ void chebyshev_codegen(bElem (*Ac_arr)[STRIDE1][STRIDE0], bElem (*Ap_arr)[STRIDE1][STRIDE0], bElem (*Dinv_arr)[STRIDE1][STRIDE0], bElem (*RHS_arr)[STRIDE1][STRIDE0], bElem (*out_arr)[STRIDE1][STRIDE0], bElem *c, bElem (*coeff)[4][4]) {
  long k = OFF2 + (blockIdx_z * TILE2);
  long j = OFF1 + (blockIdx_y * TILE1);
  long i = OFF0 + (blockIdx_x * TILE0);
  tile("$PYTHON", VSVEC, (TILE2, TILE1, VECSIZE), ("k", "j", "i"), (1, 1, VECSIZE));
}

#undef Ac
#undef Ap
#undef Dinv
#undef RHS
#undef out
// $END codegen

