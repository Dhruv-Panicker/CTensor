#include "tensor.h"
#include <assert.h>

//  element-wise ops (add, mul, add_inplace, scale).
// The shape-mismatch case is tested separately in test_mismatch.c,
int main(void) {

    /* ---- tensor_add: two [2,3] tensors ---- */
    {
        int shape[2] = {2, 3};
        Tensor *a = tensor_create(shape, 2);
        Tensor *b = tensor_create(shape, 2);
        for (int i = 0; i < a->size; i++) {
            a->data[i] = (float)i;        // 0,1,2,3,4,5
            b->data[i] = (float)(i * 2);  // 0,2,4,6,8,10
        }
        Tensor *c = tensor_add(a, b);
        for (int i = 0; i < c->size; i++) {
            assert(c->data[i] == a->data[i] + b->data[i]);
        }
        printf("[PASS] tensor_add: element-wise sum correct\n");
        tensor_free(a);
        tensor_free(b);
        tensor_free(c);   
    }

    /* ---- tensor_mul: element-wise product ---- */
    {
        int shape[2] = {2, 3};
        Tensor *a = tensor_create(shape, 2);
        Tensor *b = tensor_create(shape, 2);
        for (int i = 0; i < a->size; i++) {
            a->data[i] = (float)(i + 1);  // 1..6
            b->data[i] = 3.0f;
        }
        Tensor *c = tensor_mul(a, b);
        for (int i = 0; i < c->size; i++) {
            assert(c->data[i] == a->data[i] * b->data[i]);
        }
        printf("[PASS] tensor_mul: element-wise product correct\n");
        tensor_free(a);
        tensor_free(b);
        tensor_free(c);
    }

    /* ---- tensor_add_inplace: a changes, b stays the same ---- */
    {
        int shape[2] = {2, 3};
        Tensor *a = tensor_create(shape, 2);
        Tensor *b = tensor_create(shape, 2);
        for (int i = 0; i < a->size; i++) {
            a->data[i] = (float)i;          // 0..5
            b->data[i] = (float)(100 + i);  // 100..105
        }
        tensor_add_inplace(a, b);
        for (int i = 0; i < a->size; i++) {
            assert(a->data[i] == (float)i + (float)(100 + i));  // a updated
            assert(b->data[i] == (float)(100 + i));             // b untouched
        }
        printf("[PASS] tensor_add_inplace: a updated, b unchanged\n");
        tensor_free(a);
        tensor_free(b);
    }

    /* ---- tensor_scale: [3,3] doubled ---- */
    {
        int shape[2] = {3, 3};
        Tensor *a = tensor_create(shape, 2);
        for (int i = 0; i < a->size; i++) {
            a->data[i] = (float)(i + 1);  // 1..9
        }
        tensor_scale(a, 2.0f);
        for (int i = 0; i < a->size; i++) {
            assert(a->data[i] == (float)(i + 1) * 2.0f);
        }
        printf("[PASS] tensor_scale: all elements doubled\n");
        tensor_free(a);
    }

    printf("ALL ELEMENT-WISE OP CHECKS PASSED\n");
    return 0;
}
