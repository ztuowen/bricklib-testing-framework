
__global__ void laplacian_naive(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0], bElem* dev_coeff);
__global__ void laplacian_naive_bricks(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType &bIn, BType &bOut, bElem *dev_coeff);
// __global__ void laplacian_codegen_bricks(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem *dev_coeff);
// __global__ void laplacian_codegen(bElem (*arr_in)[STRIDE1][STRIDE0], bElem (*arr_out)[STRIDE1][STRIDE0], bElem *dev_coeff);
