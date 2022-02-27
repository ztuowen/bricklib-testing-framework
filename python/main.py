from generate_constants import gen_consts_file
from kernel_config_application import KernelConfigApplier
from argparse import ArgumentParser
import os
import json
import subprocess

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--backend", type=str, required=True, choices=['hip', 'cuda'])

    args = parser.parse_args()
    cwd = os.getcwd()
    config_path = os.path.join(cwd, args.config)
    with open(config_path, 'r') as f:
        decoded = json.load(f)
    
    gen_consts_file(args.backend, args.config)
    kernels = decoded["kernels"]
    kernel_objects = map(
        lambda k: KernelConfigApplier(k, kernels[k]["versions"], kernels[k]["sizes"], args.backend),
        kernels.keys()
    )

    vecscatter_path = os.path.join(decoded["bricklib-path"], "codegen", "vecscatter")
    to_call = []
    for k in kernel_objects:
        k.generate_intermediate_code().run_codegen_vecscatter(vecscatter_path, python="python3", extras=decoded[args.backend]["codegen-flags"])
        to_call.extend(k.wrap_functions())

    compile_files=[]
    with open(os.path.join(cwd, "main.cu"), "w") as f:
        f.write('#include "./gen/consts.h"\n')
        f.write(f'#include <brick-{args.backend}.h>\n')
        f.write("""#include <iostream>
#include \"bricksetup.h\"
#include \"multiarray.h\"
#include \"brickcompare.h\"
#include <omp.h>
#include <cmath>
#include <cassert>
#include <string.h>
#include \"brick.h\"
#include \"vecscatter.h\"
""")
        for kernel_name in kernels.keys():
            f.write(f'#include "./kernels/{kernel_name}/gen/{kernel_name}.h"\n')
            compile_files.append(f"./kernels/{kernel_name}/gen/{kernel_name}.cu")
            
        f.write("typedef void (*kernel)();\n")
        f.write("int main(void) {\n")
        f.write("\t kernel kernels[] = {")
        f.write(",".join(to_call))
        f.write("};\n")

        
        f.write(f"\tfor (int i = 0; i < {len(to_call)}; i++) " + "{\n")
        f.write("\t\tkernels[i]();\n")
        f.write("\t}\n")

        f.write("}\n")

    command = [decoded[args.backend]["compiler"], "main.cu", *compile_files, "-I", f'{decoded["bricklib-path"]}/include', "-L", f'{decoded["bricklib-path"]}/build/src', '-l', 'brickhelper', '-o', 'main', *(decoded[args.backend]["compiler-flags"])]    
    print(" ".join(command))
    subprocess.run(command)
