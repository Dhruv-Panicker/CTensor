# ctensor

A tensor library written from scratch in C with no external dependencies.

It supports N-dimensional tensors with stride-based indexing, element-wise operations,
matrix multiplication, reshaping and transposing through views, and the core operations
needed for a neural network forward pass (activations and a linear layer).

## Build

```bash
make test       # build and run the test suite
make memcheck   # run the tests with AddressSanitizer and UBSan
make clean      # remove build artifacts
make            # build the demo (examples/mlp)
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
examples/mlp.c  MLP forward pass demo
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

## Status

This is a work in progress, built in stages.

Done:
* Scaffold (Makefile, header)
* Core tensor ops (create, free, zeros, rand, get, set, print)

Next:
* Element-wise ops
* Matrix multiply (naive vs cache-friendly)
* Views (reshape, transpose, contiguous)
* Neural network ops (activations, linear layer)
* MLP demo
