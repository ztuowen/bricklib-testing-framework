from genericpath import isdir
import json
from re import template
from typing import List
from os import getcwd, makedirs, path
import subprocess

ALWAYS_INCLUDED_DATA = [
    "kernel_name",
    "c_template",
    "python_template"
]
EXECUTION_TYPES = ["naive", "naive-bricks", "codegen", "codegen-bricks"]
class KernelConfigApplier:
    def __init__(self, kernel_name: str, execution_types: List[str]):
        self.execution_types = execution_types
        self.kernel_name = kernel_name
        self.kernel_path = path.join(getcwd(), 'kernels', kernel_name)
        p = path.join(self.kernel_path, 'config.json')
        with open(p, "r") as f:
            self.config = {k:v for (k,v) in json.load(f).items() if k in ALWAYS_INCLUDED_DATA or k in execution_types}

    def run_codegen_vecscatter(self, vecscatter_path: str, extras: List[str]):
        subprocess.run(["python", vecscatter_path, \
            path.join(self.kernel_path, "intermediate_gen", self.kernel_name + ".cu"), \
            path.join(self.kernel_path, "gen", self.kernel_name + ".cu"),
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
                if header_scanning:
                    if line.strip().startswith("//") and "$START" in line:
                        header_scanning = False
                        break
                    header += line

                if start_scanning:
                    if line.strip().startswith("//") and "$START" in line:
                        t = line.split()[-1]
                        if t not in self.config.keys():
                            continue
                        current_type = t
                        codes[current_type] = ""
                        start_scanning = False
                elif line.strip().startswith("//") and "$END" in line and line.split()[-1] == current_type:
                    current_type = ""
                    start_scanning = True
                else:
                    codes[current_type] += line
        return (header, codes)

    def generate_code_for_sizes(self, sizes: List[int]):
        (header, codes) = self._collect_c_template_code()
        gen_dir = path.join(self.kernel_path, "intermediate_gen")
        if not isdir(gen_dir):
            makedirs(gen_dir)
        # generate the python
        temp_name = self.config["python_template"].split(".")[0]
        with open(path.join(self.kernel_path, self.config["python_template"]), "r") as pfi:
            temp_contents = pfi.read()
            for size in sizes:
                with open(path.join(gen_dir, temp_name + str(size) + ".py"), "w") as pfo:
                    pfo.write(temp_contents.replace("$SIZE", str(size)))
        
        gen_file_path = path.join(gen_dir, self.kernel_name + ".cu")
        default_written = []
        with open(gen_file_path, "w") as f:
            f.write(header)
            for size in sizes:
                for type in codes.keys():
                    code = codes[type]
                    if "codegen" in type:
                        code = code \
                            .replace("$PYTHON", path.join(gen_dir, temp_name + str(size) + ".py")) \
                            .replace(self.config[type]["function"], f"{self.config[type]['function']}{size}", 1)
                        f.write(code)
                    else:
                        args = self.config[type]["arguments"]
                        size_replace_index = -1
                        for (i, arg) in enumerate(args):
                            if arg["generator"] == "size":
                                size_replace_index = i
                                break
                        if size_replace_index == -1:
                            f.write(code)
                        else:
                            if type not in default_written:
                                f.write(code)
                                default_written.append(type)
                            f.write(f"#define {type}{size}(")
                            for i in range(0, len(args)):
                                if i != size_replace_index:
                                    f.write(chr(i+97))
                                    if i != (len(args) - 2):
                                        f.write(", ")
                            f.write(f") {self.config[type]['function']}(")
                            for i in range(0, len(args)):
                                if i != size_replace_index:
                                    f.write(chr(i + 97))
                                else:
                                    f.write(str(size))
                                if i != (len(args) - 1):
                                    f.write(", ")
                            f.write(")\n")
        return self



if __name__ == "__main__":
    c = KernelConfigApplier("laplacian", ["naive", "codegen"]).generate_code_for_sizes([13,23])
