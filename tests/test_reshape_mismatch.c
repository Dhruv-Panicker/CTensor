#include "tensor.h"

// Verifies reshape rejects a shape whose element count differs.
// t is [3,4] (12 elements); reshaping to [2,5] (10) is invalid, so
// tensor_reshape should print an error and exit(1).
// The Makefile runs this and checks for a non-zero exit code.

int main(void) {
    int shape[2] = {3, 4};   // 12 elements
    int bad[2] = {2, 5};     // 10 elements -> mismatch

    Tensor *t = tensor_create(shape, 2);

    tensor_reshape(t, bad, 2);   // expected: print error + exit(1)

    // Unreachable if the count check fires.
    printf("ERROR: tensor_reshape did not exit on element-count mismatch\n");
    tensor_free(t);
    return 0;
}
