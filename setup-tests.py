import json
import os
from argparse import ArgumentParser
import subprocess
import re
import itertools

CONST_KEYS = ["dimensions", "brick-dimensions", "brick-padding", "vecsize", "fold"]

CONST_KEY_TO_DEF = {
    "dimensions": "N",
    "brick-dimensions": "TILE",
    "brick-padding": "PADDING",
    "vecsize": "VECSIZE",
    "fold": "FOLD"
}

DERIVATIVES = {
    "GZ": "($brick-dimensions)",
    "OFF": "(GZ$n + $brick-padding)",
    "STRIDE": "($dimensions + 2 * (OFF$n))",
    "GB": "(GZ$n / ($brick-dimensions))",
    "BLOCK": "($dimensions / $brick-dimensions)",
    "NAIVE_BSTRIDE": "(($dimensions + 2 * GZ$n) / ($brick-dimensions))"
}

def collect_constants(conf, backend):
    consts = { k:conf[k] for k in CONST_KEYS if k in conf }

    if backend in conf:
        for k in conf[backend]:
            if k in CONST_KEYS:
                consts[k] = conf[backend][k]
    return consts

def gen_consts_file(consts, backend):
    cwd = os.getcwd()
    if "gen" not in os.listdir(cwd):
        os.mkdir(os.path.join(cwd, "gen"))

    with open("gen/consts.h", 'w') as f:
        f.write("// This is a generated file and should not be edited\n")
        f.write(f'#define VSVEC "{backend.upper()}"\n')
        clines = map(lambda c: value_to_define(c, consts[c]), consts)
        f.writelines(clines)

        derivative_lines = []
        for d in DERIVATIVES:
            keys = re.findall("\$[a-zA-Z-]{2,}", DERIVATIVES[d])
            if len(keys) == 0:
                continue
            lens = len(consts[keys[0][1:]])
            match_len = all(
                lambda e: len(consts[e[1:]]) == lens
                for e in keys
            )
            if not match_len:
                raise RuntimeError(f"Differing value lengths for derivative {d}")
            
            lines = []
            for i in range(0, lens):
                lines.append(f"#define {d}{i} {DERIVATIVES[d]}")

            for key in keys:
                k = key[1:]
                if type(consts[k]) is list:
                    for i in range(0, lens):
                        lines[i] = lines[i].replace(key, str(consts[k][i])).replace("$n", str(i))
                else:
                    lines[0] = lines[0].replace(key, str(consts[k])).replace("$n", str(i))
            derivative_lines.extend(lines)
        f.write("\n".join(derivative_lines) + "\n")

        brick_size = "#define BRICK_SIZE"
        dims = len(consts["brick-dimensions"])
        for i in range(0, dims):
            brick_size = f"{brick_size} TILE{dims - i - 1}{',' if i < dims -1 else ''}"
        f.write(brick_size + "\n")
        f.write("#define BType Brick<Dim<BRICK_SIZE>, Dim<FOLD>>\n")

def value_to_define(key, val):
    d = CONST_KEY_TO_DEF[key]
    if type(val) is list:
        lines = []
        for i, v in enumerate(val):
            lines.append(f"#define {d}{i} {v}")
        return "\n".join(lines) + "\n"
    else:
        return f"#define {d} {val}\n"
    
def gen_script_vars_file(kernel_dir, kernels, setup_funcs, run_funcs):
    cwd = os.getcwd()
    with open(os.path.join(cwd, "gen", "script_vars.h"), 'w') as f:
        f.write("// This is a generated file and should not be edited\n")
        for k in kernels:
            f.write(f'#include "{kernel_dir}/out/{k}-stencils.h"\n')
        f.write("typedef void(*SFunc)();\n")
        f.write("typedef void(*AFunc)(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0]);\n")
        f.write("typedef void(*BFunc)(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut);\n")

        sfuncs = len(setup_funcs)
        f.write(f"#define sfunc_count {str(sfuncs)}\n")
        f.write("SFunc setup_funcs[] {")
        for fun in setup_funcs:
            f.write(fun + ",")
        f.write("};\n")

        afuncs = len(run_funcs['naive']) + len(run_funcs['codegen'])
        f.write(f"#define afunc_count {str(afuncs)}\n")
        f.write("AFunc array_funcs[] {")
        for fun in itertools.chain(run_funcs["naive"], run_funcs["codegen"]):
            f.write(fun + ",")
        f.write("};\n")

        bfuncs = len(run_funcs['naive-bricks']) + len(run_funcs['codegen-bricks'])
        f.write(f"#define bfunc_count {str(bfuncs)}\n")
        f.write("BFunc brick_funcs[]{")
        for fun in itertools.chain(run_funcs["naive-bricks"], run_funcs["codegen-bricks"]):
            f.write(fun + ",")
        f.write("};\n")

        f.write("""typedef struct functionMETA {
    AFunc a;
    BFunc b;
    const char * funcName;
} functionMETA;
""")
        f.write("struct functionMETA func_name_lut[] = {")
        for fun in itertools.chain(run_funcs['naive'], run_funcs['codegen']):
            f.write("{" + fun + ', nullptr, "' + fun + '"},')
        for fun in itertools.chain(run_funcs['naive-bricks'], run_funcs['codegen-bricks']):
            f.write("{nullptr," + fun + ', "' + fun + '"},')
        f.write("};\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--backend", required=True, choices=['hip', 'cuda'])

    args = parser.parse_args()

    cwd = os.getcwd()
    config_path = os.path.join(cwd, args.config)
    with open(config_path, 'r') as j:
        decoded = json.load(j)

    consts = collect_constants(decoded, args.backend)
    gen_consts_file(consts, args.backend)

    with open(os.path.join(cwd, "gen/incls.h"), 'w') as f:
        f.write("// This is a generated file and should not be edited\n")
        f.write(f"#include <brick-{args.backend}.h>\n")
    
    # Collect all the kernels
    kernels = decoded["kernels"]
    kernels_dir = os.path.join(cwd, decoded["kernel-directory"])
    if not os.path.isdir(kernels_dir):
        raise RuntimeError(f"Kernel directory {decoded['kernel-directory']} does not exist")
    
    setup_funcs = []
    run_funcs = {
        "naive": [],
        "codegen": [],
        "naive-bricks": [],
        "codegen-bricks": []
    }
    for kernel_name in kernels:
        setup_funcs.append(f"{kernel_name}_setup")
        vers = kernels[kernel_name]["versions"] if "versions" in kernels[kernel_name] else ["naive", "codegen", "naive-bricks", "codegen-bricks"]
        sizes = kernels[kernel_name]["sizes"]
        for version in vers:
            for size in sizes:
                run_funcs[version].append(f"{kernel_name}_{version.replace('-', '_')}_{size}")

        kernel_dir = os.path.join(kernels_dir, kernel_name)
        if not os.path.isdir(kernel_dir):
            raise RuntimeError(f"Kernel {kernel_name} does not exist")

        with open(f"{kernel_dir}/{kernel_name}-stencils.h", 'r') as f:
            header_data = f.readlines()
        with open(f"{kernels_dir}/out/{kernel_name}-stencils.h", "w") as f:
            f.write(f'#include "{cwd}/gen/consts.h"\n')
            f.write(f'#include "{cwd}/gen/incls.h"\n')
            f.writelines(header_data)

        # Run codegeneration on the kernel
        codegen_script = os.path.join(kernel_dir, f"codegen-{args.backend}.sh")
        if not os.path.isfile(codegen_script):
            raise RuntimeError(f"Could not find codegen script for {kernel_name} with {args.backend} backend")
        subprocess.run(['/bin/bash', codegen_script, decoded["bricklib-path"]], cwd=kernel_dir)
    gen_script_vars_file(kernels_dir, kernels.keys(), setup_funcs, run_funcs)

    all_kernel_paths = map(lambda k: os.path.join(kernels_dir, "out", f"{k}-stencils-out.cu"), kernels)
    compile_command = [decoded[args.backend]["compiler"], "tester.cu", " ".join(all_kernel_paths), "-I", f'{decoded["bricklib-path"]}/include', "-L", f'{decoded["bricklib-path"]}/build/src', '-l', 'brickhelper',]
    if "compiler-flags" in decoded[args.backend]:
        compile_command.extend(decoded[args.backend]["compiler-flags"])
    
    subprocess.run(compile_command, cwd=cwd)
    
