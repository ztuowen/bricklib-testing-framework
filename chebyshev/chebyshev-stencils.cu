#include "brick.h"
#include "vecscatter.h"

#define Ac(i, j, k) Ac_arr[k][j][i]
#define Ap(i, j, k) Ap_arr[k][j][i]
#define Dinv(i, j, k) Dinv_arr[k][j][i]
#define RHS(i, j, k) RHS_arr[k][j][i]
#define out(i, j, k) out_arr[k][j][i]
#define out(i, j, k) out_arr[k][j][i]
#define c1 c[0]
#define c2 c[1]
#define h2inv c[2]

#include "../out/chebyshev-stencils.h"

