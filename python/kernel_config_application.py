import json
from re import template
from typing import List
from os import getcwd, makedirs, path
import subprocess

ALWAYS_INCLUDED_DATA = [
    "kernel_name",
    "c_template",
    "h_template",
    "python_template"
]
EXECUTION_TYPES = ["naive", "naive-bricks", "codegen", "codegen-bricks"]

def array_generation(array_name: str, details, device_array_name: str = None):
    if details["generator"] == "zeros":
        code = f"{details['type']} *{array_name} = zeroArray(" + "{" + ",".join([str(e) for e in details["dimensions"]]) + "});\n"
    elif details["generator"] == "random":
        code = f"{details['type']} *{array_name} = randomArray(" + "{" + ",".join([str(e) for e in details["dimensions"]]) + "});\n"
    elif details["generator"] == "sequence":
        brackets = "[" + "][".join([str(e) for e in details["dimensions"]]) + "]"
        code = f"{details['type']} {array_name}{brackets} = {details['sequence']};\n"
    else:
        raise RuntimeError(f"Unknown generator: {details['generator']}")
    
    if device_array_name is not None:
        code += f"{details['type']} *{device_array_name};\n"
        code += "{\n"
        code += f"\tunsigned size = {'*'.join([str(e) for e in details['dimensions']])} * sizeof({details['type']});\n"
        code += f"\tgpuMalloc(&{device_array_name}, size);\n"
        code += f"\tgpuMemcpy({device_array_name}, {array_name}, size, gpuMemcpyHostToDevice);\n"
        code += "}\n"
    return code

def check_sizes(SIZES, rules):
    for rule in rules:
        exec(f"assert({rule})")

class KernelConfigApplier:
    def __init__(self, kernel_name: str, execution_types: List[str], sizes: List[int], backend):
        self.execution_types = execution_types
        self.sizes = sizes
        if sizes is None:
            self.sizes = [0]
        self.kernel_name = kernel_name
        self.kernel_path = path.join(getcwd(), 'kernels', kernel_name)
        self.backend = backend
        for t in execution_types:
            if not t in EXECUTION_TYPES:
                raise RuntimeError("Valid execution types are: " + ", ".join(EXECUTION_TYPES))

        p = path.join(self.kernel_path, 'config.json')
        with open(p, "r") as f:
            self.config = {k:v for (k,v) in json.load(f).items() if k in ALWAYS_INCLUDED_DATA or k in execution_types}
        if "rules" in self.config:
            check_sizes(sizes, self.config["rules"])

    def wrap_functions(self):
        wrapped_codes = []
        for t in self.execution_types:
            if "brick" in t:
                call_func = self._wrap_brick_function
            else:
                call_func = self._wrap_array_function
            for s in self.sizes:
                wrapped_codes.append(call_func(t, s))
        return wrapped_codes


    def _wrap_brick_function(self, t: str, size: int):
        ### SETUP
        wrapped = "[]() {\n"
        brick_grid_params = next(e['array'] for e in self.config[t]["arguments"] if e["type"] == "brick-grid")
        num_dimensions = len(brick_grid_params['dimensions'])
        wrapped += f"{brick_grid_params['type']} *bgrid;\n"
        wrapped += f"auto binfo = init_grid<{num_dimensions}>(bgrid, " + "{" + ",".join([str(e) for e in brick_grid_params["dimensions"]]) + "}); \n"
        wrapped += f"{brick_grid_params['type']} *device_bgrid;\n"
        wrapped += f"unsigned grid_size = ({' * '.join([str(e) for e in brick_grid_params['dimensions']])}) * sizeof({brick_grid_params['type']}); \n"
        wrapped += "gpuMalloc(&device_bgrid, grid_size);\n"
        wrapped += "gpuMemcpy(device_bgrid, bgrid, grid_size, gpuMemcpyHostToDevice);\n"
        wrapped += f"BrickInfo<{num_dimensions}> _binfo = movBrickInfo(binfo, gpuMemcpyHostToDevice);\n"
        wrapped += f"BrickInfo<{num_dimensions}> *device_binfo;\n"
        wrapped += f"unsigned binfo_size = sizeof(BrickInfo<{num_dimensions}>);\n"
        wrapped += "gpuMalloc(&device_binfo, binfo_size);\n"
        wrapped += "gpuMemcpy(device_binfo, &_binfo, binfo_size, gpuMemcpyHostToDevice);\n"
        wrapped += "auto brick_size = cal_size<BRICK_SIZE>::value;\n"
        brick_args = [e for e in self.config[t]["arguments"] if e["type"] == "brick-pointer"]
        wrapped += f"auto brick_storage = BrickStorage::allocate(binfo.nbricks, brick_size * {len(brick_args)});\n"

        brick_input_args = []
        for i,brick_params in enumerate(brick_args):
            code = f"BType brick{i}(&binfo, brick_storage, brick_size * {i});\n"
            if brick_params["array"]["generator"] != "zeros":
                code += array_generation(f"temp_arr{i}", brick_params["array"])
                code += f"copyToBrick<{num_dimensions}>(" + "{"
                for j in range(0, num_dimensions):
                    code += f"N{j} + 2 * GZ{j}"
                    if j != (num_dimensions - 1):
                        code += ", "
                code += "}, {"
                for j in range(0, num_dimensions):
                    code += f"PADDING{j}"
                    if j != (num_dimensions - 1):
                        code += ", "
                code += "}, {0,0,0}, "
                code += f"temp_arr{i}, bgrid, brick{i});\n"
                # code += f"free(temp_arr{i});\n"

            brick_input_args.append(code)
        wrapped += "\n".join(brick_input_args)

        wrapped += "BrickStorage device_bstorage = movBrickStorage(brick_storage, gpuMemcpyHostToDevice);\n"
        for i in range(0, len(brick_args)):
            wrapped += f"brick{i} = BType(device_binfo, device_bstorage, brick_size * {i});\n"
        
        other_args = [e for e in self.config[t]["arguments"] if e["type"] == "array"]
        for i,a in enumerate(other_args):
            wrapped += array_generation(f"local_arr{i}", a["array"], f"device_arr{i}")

        ### EXECUTE
        wrapped += f"printf(\"Executing {t} {self.kernel_name}, size {size}\\n\");\n"
        if "codegen" in t:
            threads = "VECSIZE"
        else:
            threads = "dim3(TILE0, TILE1, TILE2)"
        wrapped += "for (int iter = 0; iter < 100; ++iter)\n"
        wrapped += f"gpuExecKernel({self.kernel_name}_{t.replace('-','_')}{size}, dim3(BLOCK0, BLOCK1, BLOCK2), {threads},"
        if len(brick_grid_params["dimensions"]) > 1:
            wrapped += "(" + brick_grid_params["type"] + "(*)[" + "][".join(brick_grid_params["dimensions"][1:]) + "])"
        else:
            wrapped += "(" + brick_grid_params["type"] + "*)"
        wrapped += "device_bgrid"
        for i in range(0, len(brick_args)):
            wrapped += f", brick{i}"
        for i in range(0, len(other_args)):
            if len(other_args[i]["array"]["dimensions"]) > 1:
                dimStr = [str(d) for d in other_args[i]["array"]["dimensions"]]
                type_coalesce = other_args[i]["array"]["type"] + "(*)[" + "][".join(dimStr[::-1][1:]) + "]"
            else:
                type_coalesce = f"{other_args[i]['array']['type']} *"
            wrapped += f", ({type_coalesce}) device_arr{i}"
        wrapped += ");\n"
        wrapped += "gpuDeviceSynchronize();\n"

        ### CLEANUP
        wrapped += "free(bgrid);\n"
        wrapped += "free(binfo.adj);\n"
        wrapped += "gpuFree(device_binfo);\n"

        wrapped = wrapped.replace("\n", "\n\t")

        wrapped += "gpuFree(device_bgrid);\n"
        wrapped += "}"
        return wrapped

    def _wrap_array_function(self, t: str, size: int):
        wrapped = "[]() {\n"
        arguments = [e for e in self.config[t]["arguments"] if e["type"] == "array"]

        ### SETUP
        for i,arg in enumerate(arguments):
            wrapped += array_generation(f"arg{i}", arg["array"], f"dev_arg{i}")
        
        ### EXECUTE
        wrapped += f"printf(\"Executing {t} {self.kernel_name}, size {size}\\n\");\n"
        if "codegen" in t:
            blocks = "dim3(N0/VECSIZE,BLOCK1,BLOCK2)"
            threads = "VECSIZE"
        else:
            blocks = "dim3(BLOCK0,BLOCK1,BLOCK2)"
            threads = "dim3(TILE0,TILE1,TILE2)"
        wrapped += "for (int iter = 0; iter < 100; ++iter)\n"
        wrapped += f"gpuExecKernel({self.kernel_name}_{t.replace('-','_')}{size}, {blocks}, {threads}, "
        for i in range(0, len(arguments)):
            if len(arguments[i]["array"]["dimensions"]) > 1:
                dimStr = [str(d) for d in arguments[i]["array"]["dimensions"]]
                type_coalesce = arguments[i]["array"]["type"] + "(*)[" + "][".join(dimStr[::-1][1:]) + "]"
            else:
                type_coalesce = f"{arguments[i]['array']['type']}*"
            wrapped += f"({type_coalesce}) dev_arg{i}"
            if i != len(arguments) - 1:
                wrapped += ", "
        wrapped += ");\n"
        wrapped += "gpuDeviceSynchronize();\n"

        ### CLEANUP
        for i in range(0, len(arguments)):
            # wrapped += f"free(arg{i});\n"
            wrapped += f"gpuFree(dev_arg{i});\n"

        wrapped = wrapped.replace("\n", "\n\t")[:-1]
        wrapped += "}"
        return wrapped

    def run_codegen_vecscatter(self, vecscatter_path: str, extras: List[str], python:str = "python"):
        out_path = path.join("./kernels", self.kernel_name, "gen")
        if not path.isdir(out_path):
            makedirs(out_path)

        subprocess.run(["cp", path.join(self.kernel_path, "intermediate_gen", self.kernel_name + ".h"), path.join(self.kernel_path, "gen", self.kernel_name + ".h")])
        subprocess.run([python, vecscatter_path, \
            path.join("./kernels", self.kernel_name, "intermediate_gen", self.kernel_name + ".cu"), \
            path.join("./kernels", self.kernel_name, "gen", self.kernel_name + ".cu"),
            *extras])
        return self

    def _collect_c_template_code(self):
        fp = path.join(self.kernel_path, self.config["c_template"])
        codes = {}
        header = ""
        with open(fp, "r") as f:
            header_scanning = True
            start_scanning = True
            current_type = ""
            for line in f.readlines():
                if line.isspace():
                    continue

                if header_scanning:
                    if line.strip().startswith("//") and "$START" in line:
                        header_scanning = False
                    else:
                        header += line

                if start_scanning:
                    if line.strip().startswith("//") and "$START" in line:
                        t = line.split()[-1]
                        if t not in self.config.keys():
                            continue
                        current_type = t
                        codes[current_type] = ""
                        start_scanning = False
                elif line.strip().startswith("//") and "$END" in line:
                    if line.split()[-1] == current_type:
                        current_type = ""
                        start_scanning = True
                    else:
                        raise RuntimeError("Unmatched end")
                else:
                    codes[current_type] += line
        return (header, codes)

    def generate_intermediate_code(self):
        (header, codes) = self._collect_c_template_code()
        gen_dir = path.join(self.kernel_path, "intermediate_gen")
        if not path.isdir(gen_dir):
            makedirs(gen_dir)
        # generate the python
        temp_name = self.config["python_template"].split(".")[0]
        with open(path.join(self.kernel_path, self.config["python_template"]), "r") as pfi:
            temp_contents = pfi.read()
            for size in self.sizes:
                with open(path.join(gen_dir, temp_name + str(size) + ".py"), "w") as pfo:
                    pfo.write(temp_contents.replace("$SIZE", str(size)))
        
        gen_file_path = path.join(gen_dir, self.kernel_name + ".cu")
        default_written = []
        with open(path.join(self.kernel_path, "intermediate_gen", self.kernel_name + ".h"), "w") as h:
            with open(path.join(self.kernel_path, self.config["h_template"]), "r") as ht:
                h.write(ht.read())

            with open(gen_file_path, "w") as f:
                f.write(header)
            
                f.write('#include "../../../gen/consts.h"\n')
                f.write('#include "./' + self.kernel_name + '.h"\n')
                f.write(f"#include <brick-{self.backend}.h>\n\n")
                for size in self.sizes:
                    for type in codes.keys():
                        code = codes[type]
                        new_kernel_name = f"{self.kernel_name}_{type.replace('-','_')}{size}"
                        code = code \
                            .replace("$PYTHON", path.join(gen_dir, temp_name + str(size) + ".py")) \
                            .replace("$SIZE", str(size)) \
                            .replace(self.config[type]["function"], new_kernel_name, 1)
                        # find the function signture and write into header file
                        for line in code.splitlines():
                            if new_kernel_name in line:
                                h.write(line.rstrip().rstrip("{").rstrip())
                                h.write(";\n")
                                break
                        f.write(code)
                        continue
                        
                        # if "codegen" in type:
                        #     code = code \
                        #         .replace("$PYTHON", path.join(gen_dir, temp_name + str(size) + ".py")) \
                        #         .replace(self.config[type]["function"], f"{self.kernel_name}_{type.replace('-','_')}{size}", 1)
                        #     for line in code.splitlines():
                        #         if f"{self.kernel_name}_{type.replace('-','_')}{size}" in line:
                        #             h.write(line.rstrip("{").rstrip())
                        #             h.write(";\n")
                        #             break
                            
                        #     f.write(code)
                        # else:
                        #     args = self.config[type]["arguments"]
                        #     size_replace_index = -1
                        #     for (i, arg) in enumerate(args):
                        #         if "generator" in arg and arg["generator"] == "size":
                        #             size_replace_index = i
                        #             break
                        #     if size_replace_index == -1:
                        #         f.write(code)
                        #     else:
                        #         if type not in default_written:
                        #             f.write(code)
                        #             default_written.append(type)
                        #         h.write(f"#define {self.kernel_name}_{type.replace('-', '_')}{size}(")
                        #         for i in range(0, len(args)):
                        #             if i != size_replace_index:
                        #                 h.write(chr(i+97))
                        #                 if i != (len(args) - 2):
                        #                     h.write(", ")
                        #         h.write(f") {self.config[type]['function']}(")
                        #         for i in range(0, len(args)):
                        #             if i != size_replace_index:
                        #                 h.write(chr(i + 97))
                        #             else:
                        #                 h.write(str(size))
                        #             if i != (len(args) - 1):
                        #                 h.write(", ")
                        #         h.write(")\n")
        return self



if __name__ == "__main__":
    c = KernelConfigApplier("laplacian", ["naive", "naive-bricks", "codegen"], [2,3,5]).generate_intermediate_code().run_codegen_vecscatter("/ccs/home/shirsch/bricklib/codegen/vecscatter", python="python3", extras=["-c", "cpp", "--", "-DBACKEND=1", "-fopenmp", "-O2", "-I../bricklib/include", "-D__HIP_PLATFORM_HCC__=", "-I/sw/spock/spack-envs/views/rocm-4.1.0/llvm/bin/../lib/clang/12.0.0", "-I/sw/spock/spack-envs/views/rocm-4.1.0/include", "-D__HIP_ROCclr__"]).wrap_functions()
    print(c)
