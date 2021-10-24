python $1/codegen/vecscatter laplacian-stencils.cu ../out/laplacian-stencils-out.cu -c cpp -- \
-DBACKEND=1 -fopenmp -O2 -I../../../bricklib/include -D__HIP_PLATFORM_HCC__= -I/sw/spock/spack-envs/views/rocm-4.1.0/hip/include/ \
-I/sw/spock/spack-envs/views/rocm-4.1.0/hip/include -I/sw/spock/spack-envs/views/rocm-4.1.0/llvm/bin/../lib/clang/12.0.0 \
-I/sw/spock/spack-envs/views/rocm-4.1.0/include -D__HIP_ROCclr__
