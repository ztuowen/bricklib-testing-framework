#include <iostream>

#include "bricksetup.h"
#include "multiarray.h"
#include "brickcompare.h"
#include <omp.h>
#include <cmath>
#include <cassert>
#include <string.h>

#include "brick.h"

#include "./gen/script_vars.h"
#include "./gen/consts.h"
#include "./gen/incls.h"

#define gpuSynchronizeAssert() assert(gpuDeviceSynchronize() == gpuSuccess)

const char *get_atype_kernel_name(AFunc ptr) {
    for (int i = 0; i < afunc_count + bfunc_count; i++) {
        if (ptr == func_name_lut[i].a) {
            return func_name_lut[i].funcName;
        }
    }
    return "unknown";
}
const char *get_btype_kernel_name(BFunc ptr) {
    for (int i = 0; i < afunc_count + bfunc_count; i++) {
        if (ptr == func_name_lut[i].b) {
            return func_name_lut[i].funcName;
        }
    }
    return "unknown";
}


// #define THRESH 1e-4
// __host__ void check_gpu_answer(bElem (*expected)[STRIDE1][STRIDE0], bElem *dev_solution, const char *error_message) {
//     auto solution = (bElem (*)[STRIDE1][STRIDE1]) malloc(STRIDE2 * STRIDE1 * STRIDE0 * sizeof(bElem));
//     gpuMemcpy((bElem *) solution, dev_solution, STRIDE2 * STRIDE1 * STRIDE0 * sizeof(bElem), gpuMemcpyDeviceToHost);

//     for (int i = OFF2; i < N2 + OFF2; i++) {
//         for (int j = OFF1; j < N1 + OFF1; j++) {
//             for (int k = OFF0; k < N0 + OFF0; k++) {
//                 if (abs(solution[i][j][k] - expected[i][j][k]) > THRESH) {
//                     fprintf(stderr, "Got %f, but expected %f at %d %d %d\n", solution[i][j][k], expected[i][j][k], i, j, k);
//                     fflush(stderr);
//                     throw std::runtime_error(error_message);
//                 }
//             }
//         }
//     }

//     free(solution);
// }

// __host__ void check_device_brick(bElem (*expected)[STRIDE1][STRIDE0], BrickStorage device_bstorage, BrickInfo<3> *binfo, unsigned brick_size, unsigned *bgrid, const char *error_message) {
//     auto brick_storage = movBrickStorage(device_bstorage, gpuMemcpyDeviceToHost);
//     BType bOut(binfo, brick_storage, brick_size);
//     if (!compareBrick<3>({N0, N1, N2}, {PADDING0, PADDING1, PADDING2}, {GZ0, GZ1, GZ2}, (bElem *) expected, bgrid, bOut)) {
//         throw std::runtime_error(error_message);
//     }
// }

int main(int argc, char* argv[]) {
    bool verify = true;
    for (int i = 0; i < argc; i++) {
        verify = verify && (strcmp(argv[i], "--nocheck") != 0);
        if (!verify) {
            printf("Not verifying answers!\n");
        }
    }

    // ---- CREATING BASIC ARRAYS ----
    bElem *arr_a = randomArray({STRIDE2, STRIDE1, STRIDE0});
    bElem *arr_b = zeroArray({STRIDE2, STRIDE1, STRIDE0});
    bElem *dev_a;
    bElem *dev_b;
    {
        unsigned size = STRIDE2 * STRIDE1 * STRIDE0 * sizeof(bElem);
        gpuMalloc(&dev_b, size);
        gpuMalloc(&dev_a, size);
        gpuMemcpy(dev_a, arr_a, size, gpuMemcpyHostToDevice);
    }
    // ---- DONE WITH BASIC ARRAYS ----


    // ---- BRICK SETUP ----
    unsigned *bgrid;
    auto binfo = init_grid<3>(bgrid, {NAIVE_BSTRIDE2, NAIVE_BSTRIDE1, NAIVE_BSTRIDE0});
    unsigned *device_bgrid;
    {
        unsigned grid_size = (NAIVE_BSTRIDE0 * NAIVE_BSTRIDE1 * NAIVE_BSTRIDE2) * sizeof(unsigned);
        gpuMalloc(&device_bgrid, grid_size);
        gpuMemcpy(device_bgrid, bgrid, grid_size, gpuMemcpyHostToDevice);
    }
    
    BrickInfo<3> _binfo = movBrickInfo(binfo, gpuMemcpyHostToDevice);
    BrickInfo<3> *device_binfo;
    {
        unsigned binfo_size = sizeof(BrickInfo<3>);
        gpuMalloc(&device_binfo, binfo_size);
        gpuMemcpy(device_binfo, &_binfo, binfo_size, gpuMemcpyHostToDevice);
    }
    auto brick_size = cal_size<BRICK_SIZE>::value;
    // double number of bricks for a and b
    auto brick_storage = BrickStorage::allocate(binfo.nbricks, brick_size * 2);

    BType bIn(&binfo, brick_storage, 0);
    BType bOut(&binfo, brick_storage, brick_size);
    copyToBrick<3>({N0 + 2 * GZ0, N1 + 2 * GZ1, N2 + 2 * GZ2}, {PADDING0, PADDING1, PADDING2}, {0, 0, 0}, arr_a, bgrid, bIn);

    BrickStorage device_bstorage = movBrickStorage(brick_storage, gpuMemcpyHostToDevice);
    bIn = BType(device_binfo, device_bstorage, 0);
    bOut = BType(device_binfo, device_bstorage, brick_size);
    // ---- DONE WITH BRICK SETUP ----

    // ---- RUN TESTS ----
    gpuDeviceSetCacheConfig(gpuFuncCachePreferL1);

    dim3 blocks(BLOCK0, BLOCK1, BLOCK2); 
    dim3 threads(TILE0, TILE1, TILE2);

    // auto expected = (bElem (*)[STRIDE1][STRIDE0]) malloc(STRIDE2 * STRIDE1 * STRIDE0 * sizeof(bElem));
    // {
    //     printf("Generating expected\n");
    //     bElem *dev_gpu_b;
    //     gpuMalloc(&dev_gpu_b, STRIDE2 * STRIDE1 * STRIDE0 * sizeof(bElem));
    //     gpuExecKernel(laplacian_expected_13, 1, 1, (bElem (*)[STRIDE1][STRIDE0]) dev_a, (bElem (*)[STRIDE1][STRIDE0]) dev_gpu_b);
    //     gpuDeviceSynchronize();

    //     gpuMemcpy((bElem *) expected, dev_gpu_b, STRIDE2 * STRIDE1 * STRIDE0 * sizeof(bElem), gpuMemcpyDeviceToHost);
    //     gpuFree(dev_gpu_b);
    // }

    for (int i = 0; i < sfunc_count; i++) {
        setup_funcs[i]();
    }
    for (int i = 0; i < afunc_count; i++) {
        printf("Executing: %s\n", get_atype_kernel_name(array_funcs[i]));
        gpuExecKernel(array_funcs[i], blocks, threads, (bElem (*)[STRIDE1][STRIDE0]) dev_a, (bElem (*)[STRIDE1][STRIDE0]) dev_b);
        gpuSynchronizeAssert();
        // check_gpu_answer(expected, dev_b, "Mismatch!");
        gpuMemcpy(dev_b, arr_b, STRIDE2 * STRIDE1 * STRIDE0 * sizeof(bElem), gpuMemcpyHostToDevice);
    }
    for (int i = 0; i < bfunc_count; i++) {
        printf("Executing: %s\n", get_btype_kernel_name(brick_funcs[i]));
        gpuExecKernel(brick_funcs[i], blocks, VECSIZE, (unsigned (*)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0]) device_bgrid, bIn, bOut);
        gpuSynchronizeAssert();
        // check_device_brick(expected, device_bstorage, &binfo, brick_size, bgrid, "Brick solution mismatch");
        gpuMemcpy(device_bstorage.dat.get(), brick_storage.dat.get(), brick_storage.step * brick_storage.chunks * sizeof(bElem), gpuMemcpyHostToDevice);
    }
    // ---- DONE RUNNING TESTS ----


    // ---- CLEANUP ----
    // free(expected);
    free(arr_a);
    free(arr_b);
    
    free(bgrid);
    free(binfo.adj);
    gpuFree(device_binfo);
    gpuFree(device_bgrid);
    gpuFree(dev_a);
    gpuFree(dev_b);
    return 0;
}