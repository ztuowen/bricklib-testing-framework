hipcc tester.cu kernels/out/laplacian-stencils-out.cu -DTILE=8 -DPADDING=32 -DTILEx=16 -DPADDINGx=32 -DBACKEND=1 -I ../bricklib/include -L ../bricklib/build/src -l brickhelper -O2 -fopenmp -o ./out/tester
