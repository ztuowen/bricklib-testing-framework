__global__ void poisson_naive(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0]);
__global__ void poisson_codegen(bElem (*in_arr)[STRIDE1][STRIDE0], bElem (*out_arr)[STRIDE1][STRIDE0]);
__global__ void poisson_naive_bricks(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType in, BType out);
__global__ void poisson_codegen_bricks(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType in, BType out);
