#include "tensor.h"

// Verifies that adding mismatched shapes triggers the error path.
// tensor_add prints to stderr and calls exit(1) on a shape mismatch,
int main(void) {
    int s1[2] = {2, 3};
    int s2[2] = {3, 2};   // same size, different shape -> must be rejected

    Tensor *a = tensor_create(s1, 2);
    Tensor *b = tensor_create(s2, 2);

    tensor_add(a, b);   // expected: print error + exit(1)

    // If we get here, the shape check did not fire.
    printf("ERROR: tensor_add did not exit on shape mismatch\n");
    tensor_free(a);
    tensor_free(b);
    return 0;   
}
