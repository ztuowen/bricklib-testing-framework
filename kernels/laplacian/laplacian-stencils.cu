
#include <omp.h>
#include "vecscatter.h"
#include "brick.h"

#include "../out/laplacian-stencils.h"

__constant__ bElem dev_coeff[10];

void laplacian_setup() {
    bElem coeff[] = {1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1};
    gpuMemcpyToSymbol(dev_coeff, coeff, sizeof(coeff));
}

__device__ void laplacian_expected_x(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0], size_t radius) {
    bElem(*out_sized)[STRIDE1][STRIDE0] = (bElem (*)[STRIDE1][STRIDE0]) out;
    bElem(*in_sized)[STRIDE1][STRIDE0] = (bElem (*)[STRIDE1][STRIDE0]) in;
    for (int k = OFF2; k < N2 + OFF2; k++) {
        for (int j = OFF1; j < N1 + OFF1; j++) {
            for (int i = OFF0; i < N0 + OFF0; i++) {
                out_sized[k][j][i] = dev_coeff[0] * in_sized[k][j][i];
                #pragma unroll
                for (int a = 1; a <= radius; a++) {
                    out_sized[k][j][i] += dev_coeff[a] * (
                        in_sized[k][j][i + a] + in_sized[k][j + a][i] + in_sized[k + a][j][i] +
                        in_sized[k][j][i - a] + in_sized[k][j - a][i] + in_sized[k - a][j][i]
                    );
                }
            }
        }
    }
}

__global__ void laplacian_expected_13(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0]) {
    return laplacian_expected_x(in, out, 2);
}
__global__ void laplacian_expected_31(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0]) {
    return laplacian_expected_x(in, out, 5);
}
__global__ void laplacian_expected_49(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0]) {
    return laplacian_expected_x(in, out, 9);
}

__device__ void naive_xpt_sum(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0], const size_t radius) {
    unsigned i = OFF0 + (blockIdx.x) * TILE0 + threadIdx.x;
    unsigned j = OFF1 + (blockIdx.y) * TILE1 + threadIdx.y;
    unsigned k = OFF2 + (blockIdx.z) * TILE2 + threadIdx.z;

    bElem temp = dev_coeff[0] * in[k][j][i];
    #pragma unroll
    for (int a = 1; a <= radius; a++) {
        temp += dev_coeff[a] * (
            in[k][j][i + a] + in[k][j + a][i] + in[k + a][j][i] +
            in[k][j][i - a] + in[k][j - a][i] + in[k - a][j][i]);
    }
    out[k][j][i] = temp;
}

__global__ void laplacian_naive_13(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0]) {
    return naive_xpt_sum(in, out, 2);
}

__global__ void laplacian_naive_31(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0]) {
    return naive_xpt_sum(in, out, 5);
}

__global__ void laplacian_naive_49(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0]) {
    return naive_xpt_sum(in, out, 8);
}


__device__ void naive_brick_xpt(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType &bIn, BType &bOut, size_t radius) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;
    unsigned k = threadIdx.z;
    bOut[b][k][j][i] = dev_coeff[0] * bIn[b][k][j][i];
    
    #pragma unroll
    for (int a = 1; a <= radius; a++) {
        bOut[b][k][j][i] += dev_coeff[a] * (
            bIn[b][k][j][i + a] + bIn[b][k][j + a][i] + bIn[b][k + a][j][i] +
            bIn[b][k][j][i - a] + bIn[b][k][j - a][i] + bIn[b][k - a][j][i]
        );
    }
}

__global__ void laplacian_naive_bricks_13(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut) {
    naive_brick_xpt(grid, bIn, bOut, 2);
}

__global__ void laplacian_naive_bricks_31(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut) {
    naive_brick_xpt(grid, bIn, bOut, 5);
}

__global__ void laplacian_naive_bricks_49(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut) {
    naive_brick_xpt(grid, bIn, bOut, 8);
}

__global__ void laplacian_codegen_bricks_13(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    brick("13pt.py", VSVEC, (TILE2, TILE1, TILE0), (FOLD), b);
}

__global__ void laplacian_codegen_bricks_31(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    brick("31pt.py", VSVEC, (TILE2, TILE1, TILE0), (FOLD), b);
}

__global__ void laplacian_codegen_bricks_49(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut) {
    unsigned b = grid[blockIdx.z + GB2][blockIdx.y + GB1][blockIdx.x + GB0];
    brick("49pt.py", VSVEC, (TILE2, TILE1, TILE0), (FOLD), b);
}

#define bIn(a, b, c) arr_in[c][b][a]
#define bOut(a, b, c) arr_out[c][b][a]

__global__ void laplacian_codegen_13(bElem (*arr_in)[STRIDE1][STRIDE0], bElem (*arr_out)[STRIDE1][STRIDE0]) {
    long k = OFF2 + (blockIdx.z * TILE2);
    long j = OFF1 + (blockIdx.y * TILE1);
    long i = OFF0 + (blockIdx.x * VECSIZE);
    tile("13pt.py", VSVEC, (TILE2, TILE1, VECSIZE), ("k", "j", "i"), (1, 1, VECSIZE));
}

__global__ void laplacian_codegen_31(bElem (*arr_in)[STRIDE1][STRIDE0], bElem (*arr_out)[STRIDE1][STRIDE0]) {
    long k = OFF2 + (blockIdx.z * TILE2);
    long j = OFF1 + (blockIdx.y * TILE1);
    long i = OFF0 + (blockIdx.x * VECSIZE);
    tile("31pt.py", VSVEC, (TILE2, TILE1, VECSIZE), ("k", "j", "i"), (1, 1, VECSIZE));
}

__global__ void laplacian_codegen_49(bElem (*arr_in)[STRIDE1][STRIDE0], bElem (*arr_out)[STRIDE1][STRIDE0]) {
    long k = OFF2 + (blockIdx.z * TILE2);
    long j = OFF1 + (blockIdx.y * TILE1);
    long i = OFF0 + (blockIdx.x * VECSIZE);
    tile("49pt.py", VSVEC, (TILE2, TILE1, VECSIZE), ("k", "j", "i"), (1, 1, VECSIZE));
}

#undef bIn
#undef bOut