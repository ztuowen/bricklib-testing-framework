# bricklib-testing-framework
This project is intended to house and setup tests of different kernels for different systems.
The python script `setup-tests.py` creates all of the needed C++ files to run the tests specified in the `config.json` file.

## Dependencies
- Python 3.x
- A DSL compiler (hipcc, nvcc)

## Config File Format
The tests are configured with a JSON format file. The default config file is `config.json` but it can be overwritten when running `setup-tests.py` with the script flag `--config [path to config file]`
