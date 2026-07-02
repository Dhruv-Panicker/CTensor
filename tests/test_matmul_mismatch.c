#include "tensor.h"

// Verifies matmul rejects mismatched inner dimensions.
// A is [2,3] (inner = 3), B is [4,2] (inner = 4); 3 != 4, so
// tensor_matmul should print an error and exit(1).
// The Makefile runs this and checks for a non-zero exit code.

int main(void) {
    int as[2] = {2, 3};
    int bs[2] = {4, 2};   // inner dim 4 != a's inner dim 3

    Tensor *a = tensor_create(as, 2);
    Tensor *b = tensor_create(bs, 2);

    tensor_matmul(a, b);   // expected: print error + exit(1)

    // Unreachable if the dimension check fires.
    printf("ERROR: tensor_matmul did not exit on inner-dimension mismatch\n");
    tensor_free(a);
    tensor_free(b);
    return 0;
}
