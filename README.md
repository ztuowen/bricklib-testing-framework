# bricklib-testing-framework
This project is intended to house and setup tests of different kernels for different systems.
The python script `setup-tests.py` creates all of the needed C++ files to run the tests specified in the `config.json` file.

## Dependencies
- Python 3.x
- A DSL compiler (hipcc, nvcc)
- bricklib

## Config File Format
The tests are configured with a JSON format file. The default config file is `config.json` but it can be overwritten when running `setup-tests.py` with the script flag `--config [path to config file]`

```json
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
