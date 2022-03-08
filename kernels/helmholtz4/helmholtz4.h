__global__ void helmholtz4_naive(bElem (*x)[STRIDE1][STRIDE0], bElem (*alpha)[STRIDE1][STRIDE0], 
  bElem (*beta_i)[STRIDE1][STRIDE0], bElem (*beta_j)[STRIDE1][STRIDE0], bElem (*beta_k)[STRIDE1][STRIDE0], 
  bElem (*out)[STRIDE1][STRIDE0], bElem *c);
__global__ void helmholtz4_codegen(bElem (*x_arr)[STRIDE1][STRIDE0], bElem (*alpha_arr)[STRIDE1][STRIDE0], 
  bElem (*beta_i_arr)[STRIDE1][STRIDE0], bElem (*beta_j_arr)[STRIDE1][STRIDE0], bElem (*beta_k_arr)[STRIDE1][STRIDE0], 
  bElem (*out_arr)[STRIDE1][STRIDE0], bElem *c);
__global__ void helmholtz4_naive_bricks(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType x, BType alpha,
  BType beta_i, BType beta_j, BType beta_k,
  BType out, bElem *c);
__global__ void helmholtz4_codegen_bricks(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType x, BType alpha,
  BType beta_i, BType beta_j, BType beta_k,
  BType out, bElem *c);
  