#include "tensor.h"
#include <assert.h>

// Benchmark: naive ijk matmul vs cache-friendly ikj on a 512x512 multiply.

int main(void) {
    srand(42);

    int shape[2] = {512, 512};
    Tensor *a = tensor_rand(shape, 2);
    Tensor *b = tensor_rand(shape, 2);

    printf("=== Matmul benchmark: [512,512] x [512,512] ===\n\n");

    // ---- ijk (naive): inner loop walks b down a column -> cache misses ----
    clock_t t0 = clock();
    Tensor *c_ijk = tensor_matmul(a, b);
    clock_t t1 = clock();
    double ijk_time = (double)(t1 - t0) / CLOCKS_PER_SEC;

    // ---- ikj (cache-friendly): inner loop walks b along a row -> stride 1 ----
    clock_t t2 = clock();
    Tensor *c_ikj = tensor_matmul_ikj(a, b);
    clock_t t3 = clock();
    double ikj_time = (double)(t3 - t2) / CLOCKS_PER_SEC;

    // both must produce the identical result ----
    // Each c[i][j] is accumulated over k in the same order in both compute kernels,
    for (int i = 0; i < c_ijk->size; i++) {
        assert(c_ijk->data[i] == c_ikj->data[i]);
    }

    printf("ijk (naive):          %.4f s\n", ijk_time);
    printf("ikj (cache-friendly): %.4f s\n", ikj_time);
    printf("speedup:              %.2fx\n", ijk_time / ikj_time);
    printf("results identical:    yes\n");

    tensor_free(a);
    tensor_free(b);
    tensor_free(c_ijk);
    tensor_free(c_ikj);
    return 0;
}
