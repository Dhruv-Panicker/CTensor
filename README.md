# ctensor

A tensor library written from scratch in C with no external dependencies.

It supports N-dimensional tensors with stride-based indexing, element-wise operations,
matrix multiplication, reshaping and transposing through views, and the core operations
needed for a neural network forward pass (activations and a linear layer).

## Benchmark

Two matrix multiply implementations with identical output, on a 512x512 multiply
(compiled with -O2, Apple Silicon):

| Loop order           | Time     |
|----------------------|----------|
| ijk (naive)          | 0.1198 s |
| ikj (cache-friendly) | 0.0072 s |
| speedup              | 16.75x   |

Same computation, same result. Reordering the inner loops so the innermost one walks
memory with stride 1 replaces near constant cache misses with sequential access the
compiler can vectorize. Reproduce with `make bench`.

## Build

```bash
make test       # build and run the test suite
make memcheck   # run the tests with AddressSanitizer and UBSan
make bench      # benchmark naive vs cache-friendly matmul
make clean      # remove build artifacts
make            # build the demo (demo/mlp)
```

To check for leaks on macOS:

```bash
leaks --atExit -- ./tests/test_tensor
```

## Layout

```
tensor.h        public API
src/tensor.c    create, free, zeros, rand, get, set, print
src/ops.c       element-wise ops, matmul, reshape, transpose, contiguous
src/nn.c        relu, sigmoid, softmax, linear layer
demo/mlp.c      MLP forward pass demo
tests/          test programs
```

## Notes

Tensors are stored in row-major order. Each tensor holds its own shape and strides
arrays, and indexing is a dot product between the index vector and the strides.

Any function that returns a `Tensor *` allocates a new tensor, and the caller is
responsible for freeing it with `tensor_free`. Functions whose names end in `_inplace`
modify their argument directly and allocate nothing.

Reshape and transpose are O(1) and share the original data buffer. `contiguous` is the
one operation that copies, used when a non contiguous tensor needs a fresh row-major
layout.

## Features

* Core tensor ops (create, free, zeros, rand, get, set, print)
* Element-wise ops (add, mul, scale)
* Matrix multiply (naive ijk and cache-friendly ikj)
* Views (reshape, transpose, contiguous)
* Neural network ops (relu, sigmoid, softmax, linear layer)
* MLP forward pass demo
