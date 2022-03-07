__global__ void f3d_naive(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0], bElem (*c)[8][8]);
__global__ void f3d_naive_bricks(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem (*c)[8][8]);

__global__ void f3d_codegen(bElem (*arr_in)[STRIDE1][STRIDE0], bElem (*arr_out)[STRIDE1][STRIDE0], bElem (*c)[8][8]);
__global__ void f3d_codegen_bricks(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem (*c)[8][8]);
