# bricklib-testing-framework
This project is intended to house and setup tests of different kernels for different systems.
The python script `setup-tests.py` creates all of the needed C++ files to run the tests specified in the `config.json` file.

## Dependencies
- Python 3.x
- A DSL compiler (hipcc, nvcc)
- bricklib

## Currently Supported Kernels

- Laplacian
	- Supports only 3D arrays
	- `size` refers to the radius of the stencil
	- Supports `size` of 1-8

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
Run the script with `python python/main.py --backend [hip or cuda] --config [optional path to config file]`

You can specify and output executable in the `compiler-flags` section of the config (ex. `-o out`)
Otherwise, the default executable will be output as `main`

## Profiling Kernels
This code is *not* self-profiling, it requires the use of an external profiling method such as nsight-compute. My suggestions for profiling tools are below:

### NVIDIA Systems
On NVIDIA machines, I used the Nsight compute CLI tool to profile.

I created a custom data sectioning description that makes this profiling very quick and simple. The code is provided in this repo and can be invoked with the command below:

`ncu --section=DataSection --section-folder-recursive=nsight-sections --section=MemoryWorkloadAnalysis --section=LaunchStats  --kernel-name=regex:'.*' --print-units=base [executable path]`

Note that you can change the `--kernel-name` argument to match a subset of kernels run. However, I suggest removing these kernels from the `config.json` file instead so that they do not waste time in compilation and runtime. 

Example output:

```
  brick_gen31(unsigned int (*) [66][18], Brick<Dim<8u, 8u, 32u>, Dim<4u, 8u> >, Brick<Dim<8u, 8u, 32u>, Dim<4u, 8u> >), 2021-Aug-05 17:15:34, Context 1, Stream 7
    Section: Memory Workload Analysis
    ---------------------------------------------------------------------- --------------- ------------------------------
    Memory Throughput                                                          byte/second             543,217,437,931.05
    Mem Busy                                                                             %                          41.87
    Max Bandwidth                                                                        %                          60.95
    L1/TEX Hit Rate                                                                      %                           4.08
    L2 Hit Rate                                                                          %                          56.81
    Mem Pipes Busy                                                                       %                          35.36
    ---------------------------------------------------------------------- --------------- ------------------------------

    Section: Launch Statistics
    ---------------------------------------------------------------------- --------------- ------------------------------
    Block Size                                                                                                         32
    Function Cache Configuration                                                                  cudaFuncCachePreferNone
    Grid Size                                                                                                      65,536
    Registers Per Thread                                                   register/thread                            176
    Shared Memory Configuration Size                                                  byte                              0
    Driver Shared Memory Per Block                                              byte/block                              0
    Dynamic Shared Memory Per Block                                             byte/block                              0
    Static Shared Memory Per Block                                              byte/block                              0
    Threads                                                                         thread                      2,097,152
    Waves Per SM                                                                                                   102.40
    ---------------------------------------------------------------------- --------------- ------------------------------

    Section: Collected Data
    ---------------------------------------------------------------------- --------------- ------------------------------
    DRAM Bytes Read                                                                   byte                  2,576,346,624
    DRAM Bytes Written                                                                byte                  1,070,719,648
    L1 Reads Hit                                                                                              254,987,328
    L1 Reads Miss                                                                                           4,864,389,344
    L1 Reads Requested                                                                                      5,119,376,672
    L1 Writes Hit                                                                                                       0
    L1 Writes Miss                                                                                          1,073,741,824
    L1 Writes Requested                                                                                     1,073,741,824
    L2 Bytes Written                                                                  byte                  1,073,741,824
    L2 Bytes Read Into L1                                                             byte                  4,862,569,120
    Bytes Requested                                                                   byte                  6,190,959,520
    L1 Bytes Cache Hit                                                                byte                    252,828,352
    L1 Bytes Cache Miss                                                               byte                  5,938,131,168
    Double Precision Adds                                                             inst                              0
    Double Precision Add Mults                                                        inst                  4,160,749,568
    Double Precision Mults                                                            inst                              0
    Single Precision Adds                                                             inst                              0
    Single Precision Add Mults                                                        inst                              0
    Single Precision Mults                                                            inst                              0
    Double Precision Ops                                                              inst                  4,160,749,568
    Double Precision FLOPS / sec                                               inst/second             619,728,722,111.27
    Single Precision Ops                                                              inst                              0
    FLOPS / sec                                                                inst/second                              0
    ---------------------------------------------------------------------- --------------- ------------------------------
```

### AMD Systems
On AMD machines, I used rocprof to retrieve GPU performance counter metrics

`rocprof --timestamp on --stats -i [rocprof config file] -o [output file path] [executable path]`

I used the following configuration as my rocprof config:

```
pmc: TCC_EA_RDREQ_32B_sum TCC_EA_RDREQ_sum TCC_EA_WRREQ_sum TCC_EA_WRREQ_64B_sum
pmc: TCC_HIT_sum TCC_MISS_sum

kernel: [space separated list of kernel names]
```

- Note that memory transactions can be 32B or 64B. To calculate total number of bytes read you will do (TCC_EA_RDREQ_32B_sum * 32) + (TCC_EA_RDREQ_sum - TCC_EA_RDREQ_32B_sum) * 64.
	- Likewise, bytes written can be calculated as (TCC_EA_WRREQ_sum - TCC_EA_WRREQ_64B_sum) * 32 + (TCC_EA_WRREQ_64B_sum) * 64
- This library generates kernels with a known pattern so you can include them in your config file for profiling: `[kernel name]_[kernel type: (naive, naive_bricks, codegen, or codegen_bricks)][size]`
	- Ex. `laplacian_naive_bricks2`

## Adding New Kernels
C code generation makes use of a known file structure and code templates.
In order to add a new kernel, use the following steps:
1. add a directory in the "kernels" directory.
1. create a `.cu` file with the required c code for the 4 kernel versions (naive, codegen, naive-bricks, codegen-bricks)
	1. you can use the token `$SIZE` which will be replaced with the current kernel size when generating code.
	1. instead of a python file name, use `$PYTHON` in any calls to `tile` or `brick`
1. create a `.h` file with declarations for all implemented kernel versions in the `.cu` file
1. create a `.py` file with the python code for 
1. create a `config.json` file. This file structure is defined in detail in the following section.

## Per-kernel configuration
Each kernel has a `config.json` that helps the python code to generate the needed C code. See `kernels/laplacian/config.json` for a full example

```jsonc
{
	"kernel_name": // name to refer to this kernel in logs
	"c_template": // string name of the file with c code
	"h_template": // string name of the file with the c header definitions
	"python_template": // string name of the file with python code for code gen
	"rules": [
		// string representations of simple python expressions returning a boolean
		// use SIZES variable
		// ex.         ["len(SIZES) == 3", "all(s <= 8 for s in SIZES)"] 
	],
	[kernel-type]: {
		"function": // function name for this kernel type definition
		"arguments": [
			{
				"type": // array, brick-grid, or brick-pointer
				"array": {
					"generator": // random, zeros, or sequence
					// Note that generator is NOT used when type is brick-grid
					"dimensions": // list of data dimensions
					"type": // c type of data in array
					"sequence": // string with c-style static array definition
					// Note that sequence is ONLY used when type is sequence
				}
			}
		]
	}
}
```
