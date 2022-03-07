__global__ void f3d_naive(bElem (*in)[STRIDE0], bElem (*out)[STRIDE0], bElem (*c)[8]);
__global__ void f3d_naive_bricks(unsigned (*grid)[NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem (*c)[8]);

__global__ void f3d_codegen(bElem (*arr_in)[STRIDE0], bElem (*arr_out)[STRIDE0], bElem (*c)[8]);
__global__ void f3d_codegen_bricks(unsigned (*grid)[NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem (*c)[8]);
