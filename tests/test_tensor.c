#include "tensor.h"
#include <assert.h>

//create / set / get / zeros / rand / free.
// Build with sanitizers and run under `leaks` to verify no memory errors.

int main(void) {
    srand(time(NULL));   // seed once, so tensor_rand differs each run

    /* ---- 1. tensor_create [3,4]: shape and strides ---- */
    int shape[2] = {3, 4};
    Tensor *t = tensor_create(shape, 2);
    assert(t != NULL);

    printf("=== tensor_create [3,4] ===\n");
    tensor_print(t);

    assert(t->ndim == 2);
    assert(t->size == 12);
    assert(t->shape[0] == 3 && t->shape[1] == 4);     // shape [3,4]
    assert(t->strides[0] == 4 && t->strides[1] == 1); // strides [4,1]
    printf("[PASS] shape [3,4], strides [4,1], size 12\n\n");

    /* ---- 2. set [1,2] = 99.0f, get it back ---- */
    int idx[2] = {1, 2};
    tensor_set(t, idx, 99.0f);
    float got = tensor_get(t, idx);
    assert(got == 99.0f);
    printf("[PASS] set/get [1,2] == %.1f\n\n", got);

    tensor_free(t);

    /* ---- 3. tensor_zeros: every element is 0.0f ---- */
    Tensor *z = tensor_zeros(shape, 2);
    assert(z != NULL);
    for (int i = 0; i < z->size; i++) {
        assert(z->data[i] == 0.0f);
    }
    printf("[PASS] tensor_zeros: all %d elements == 0.0f\n\n", z->size);
    tensor_free(z);

    /* ---- 4. tensor_rand: all elements in [0, 1) ---- */
    Tensor *r = tensor_rand(shape, 2);
    assert(r != NULL);
    printf("=== tensor_rand [3,4] ===\n");
    tensor_print(r);
    for (int i = 0; i < r->size; i++) {
        assert(r->data[i] >= 0.0f && r->data[i] < 1.0f);
    }
    printf("[PASS] tensor_rand: all elements in [0, 1)\n\n");
    tensor_free(r);

    printf("ALL MEMORY AND INTIALIZATION CHECKS PASSED\n");
    return 0;
}
