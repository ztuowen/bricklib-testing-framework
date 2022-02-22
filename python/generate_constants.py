import os
import json
import re

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

def value_to_define(key, val):
    d = CONST_KEY_TO_DEF[key]
    if type(val) is list:
        lines = []
        for i, v in enumerate(val):
            lines.append(f"#define {d}{i} {v}")
        return "\n".join(lines) + "\n"
    else:
        return f"#define {d} {val}\n"

def gen_consts_file(backend, config):
    cwd = os.getcwd()
    config_path = os.path.join(cwd, config)
    with open(config_path, 'r') as j:
        decoded = json.load(j)

    consts = collect_constants(decoded, backend)
    gen_consts_file(consts, backend)

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
