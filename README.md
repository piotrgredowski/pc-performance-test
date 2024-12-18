[![Run performance test](https://github.com/piotrgredowski/pc-performance-test/actions/workflows/run-performance-test.yaml/badge.svg)](https://github.com/piotrgredowski/pc-performance-test/actions/workflows/run-performance-test.yaml)

# Performance Test for Git, File, CPU, and Memory Operations

This project is designed to measure the performance of Git, File, CPU, and Memory operations. It provides a comprehensive set of tests to evaluate the efficiency of these operations in different scenarios.

## Features

- **Git Operations**: Clone a repository, commit changes, and perform other Git-related tasks.
- **File Operations**: Read, write, and delete files in various ways.
- **CPU Operations**: Measure CPU usage and performance.
- **Memory Operations**: Monitor memory usage and performance.

## Usage

It doesn't require any dependencies, just run the script.

You can clone the repository and run the script:

```bash
git clone https://github.com/piotrgredowski/pc-performance-test.git
cd pc-performance-test
python main.py -i "your additional information about your computer"
```

Or you can recreate `main.py` file on your local disk and then just run it.

To run the tests, use the following command:

```bash
python main.py -i "My 17 years old computer, Intel Celeron N3350, 256MB RAM"
```
