__global__ void laplacian2d_naive(bElem (*in)[STRIDE0], bElem (*out)[STRIDE0], bElem* dev_coeff);
__global__ void laplacian2d_naive_bricks(unsigned (*grid)[NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem *dev_coeff);
__global__ void laplacian2d_codegen_bricks(unsigned (*grid)[NAIVE_BSTRIDE0], BType bIn, BType bOut, bElem *dev_coeff);
__global__ void laplacian_codegen(bElem (*arr_in)[STRIDE0], bElem (*arr_out)[STRIDE0], bElem *dev_coeff);