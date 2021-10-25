# bricklib-testing-framework
This project is intended to house and setup tests of different kernels for different systems.
The python script `setup-tests.py` creates all of the needed C++ files to run the tests specified in the `config.json` file.

## Dependencies
- Python 3.x
- A DSL compiler (hipcc, nvcc)
- bricklib

## Config File Format
The tests are configured with a JSON format file. The default config file is `config.json` but it can be overwritten when running `setup-tests.py` with the script flag `--config [path to config file]`

```jsonc
{
	"bricklib-path": // absolute path to the root bricklib directory
	"dimensions": // an array of dimensions of the input array in order [x, y, z]
	"brick-dimensions": // an array that specifies the subset size for bricks (in order [x, y, z])
	"brick-padding": // an array specifying the padding around bricks (in order [x, y, z])
	"vecsize": // the vector size to use
	"fold": // brick fold size. Should be a string (ex. "4,8")
	"compiler": // what compiler to use
	"compiler-flags": // what flags to use when compiling
	"hip": {
		// can override any other settings when using the "hip" backend
	},
	"cuda": {
		// can override any other settings when usin the "cuda" backend
	},
	"kernel-directory": // path to the parent directory that contains the directories for the kernels under test
	"kernels": {
		// what kernels to test
		// each kernel must match the name of a directory in the "kernel-directory" directory
		[kernel name]: {
			"sizes": // an array of what sizes of the kernel to use (ex. specifying the radius of a stencil)
			"versions": // an array containing "naive", "codegen", "naive-bricks", and/or "codegen-bricks"
		}
	}
}
```

## Generating and Compiling the Tests
Once you have specified the testing configuration file, you can use the core python script to generate the required files.
Run the script with `python setup-tests.py --backend [hip or cuda] --config [optional path to config file]`
The output file name and location should be specified via the "compiler-flags" option (ex. -o out/test)

## Adding New Kernels
The file generation makes significant use of a known file structure in order to locate needed files.
In order to add new kernels, use the following steps:
1. add a directory with the kernel name under the "kernels" directory.
1. add a file called [kernel name]-stencils.cu under the new directory.
	1. write the kernel code into this file.
	1. each kernel function name should follow the format [kernel name]_[version]_[size]
	1. "version" is one of naive, codegen, naive_bricks, or codegen_bricks
	1. the naive and codegen function versions must take arguments `(bElem (*in)[STRIDE1][STRIDE0], bElem (*out)[STRIDE1][STRIDE0])`
	1. the naive_bricks and codegen_bricks must take the arguments `(unsigned (*grid)[NAIVE_BSTRIDE1][NAIVE_BSTRIDE0], BType bIn, BType bOut)`
1. add a file called [kernel name]-stencils.h under the new directory.
	- this file should contain the function header declarations for all the functions defined in [kernel name]-stencils.cu.
	- the file generations script uses [kernel name]-stencils.h as a starting point for generating the kernel header file.
1. be sure that [kernel name]-stencils.cu contains the line `#include "./laplacian-stencils.h`.
1. add codegenerations script files to the kernel directory. 
	- these script files should follow the fomat codegen-[backend].sh where backend is one of hip, or cuda.
	- the script files should contain the bash command that will be used to run bricklib code generation on the kernels.



