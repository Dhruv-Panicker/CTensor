#include "tensor.h"
#include <assert.h>

// Stage 5 gate: views (reshape, transpose, contiguous) and the owns_data flag.
// The reshape element-count-mismatch case is tested separately in
// test_reshape_mismatch.c, since that path calls exit(1).

int main(void) {

    /* ---- owns_data: a freshly created tensor owns its buffer ---- */
    {
        int shape[2] = {3, 4};
        Tensor *t = tensor_create(shape, 2);
        assert(t->owns_data == 1);
        printf("[PASS] tensor_create sets owns_data == 1\n");
        tensor_free(t);
    }

    /* ---- tensor_reshape [3,4] -> [2,6]: shares data, correct access ---- */
    {
        int shape[2] = {3, 4};
        Tensor *t = tensor_create(shape, 2);
        for (int i = 0; i < t->size; i++) t->data[i] = (float)i;  // 0..11

        int ns[2] = {2, 6};
        Tensor *r = tensor_reshape(t, ns, 2);

        assert(r->data == t->data);        // same buffer, no copy
        assert(r->owns_data == 0);         // view does not own the data
        assert(r->size == t->size);
        assert(r->shape[0] == 2 && r->shape[1] == 6);
        assert(r->strides[0] == 6 && r->strides[1] == 1);  // fresh row-major

        // element access: r[1][2] -> flat 1*6+2 = 8 -> value 8
        int idx[2] = {1, 2};
        assert(tensor_get(r, idx) == 8.0f);
        printf("[PASS] tensor_reshape: shared data, correct shape/strides/access\n");

        tensor_free(r);                    // frees struct only (owns_data == 0)
        assert(t->data[0] == 0.0f);        // t's data survived r's free
        tensor_free(t);
    }

    /* ---- tensor_transpose [3,4] -> [4,3]: get(tr,{i,j}) == get(t,{j,i}) ---- */
    {
        int shape[2] = {3, 4};
        Tensor *t = tensor_create(shape, 2);
        for (int i = 0; i < t->size; i++) t->data[i] = (float)i;

        Tensor *tr = tensor_transpose(t);

        assert(tr->data == t->data);       // view shares data
        assert(tr->owns_data == 0);
        assert(tr->ndim == 2);
        assert(tr->size == t->size);
        assert(tr->shape[0] == 4 && tr->shape[1] == 3);

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
                int a[2] = {i, j};
                int b[2] = {j, i};
                assert(tensor_get(tr, a) == tensor_get(t, b));
            }
        }
        printf("[PASS] tensor_transpose: get(tr,{i,j}) == get(t,{j,i})\n");

        tensor_free(tr);
        tensor_free(t);
    }

    /* ---- tensor_contiguous on a transposed view: fresh row-major copy ---- */
    {
        int shape[2] = {3, 4};
        Tensor *t = tensor_create(shape, 2);
        for (int i = 0; i < t->size; i++) t->data[i] = (float)i;

        Tensor *tr = tensor_transpose(t);      // [4,3], non-contiguous
        Tensor *c = tensor_contiguous(tr);     // fresh [4,3], row-major

        assert(c->data != t->data);            // new allocation, not shared
        assert(c->owns_data == 1);             // owns its own buffer
        assert(c->shape[0] == 4 && c->shape[1] == 3);
        assert(c->strides[0] == 3 && c->strides[1] == 1);

        // same logical values as the view it copied
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
                int idx[2] = {i, j};
                assert(tensor_get(c, idx) == tensor_get(tr, idx));
            }
        }
        printf("[PASS] tensor_contiguous: fresh row-major copy, values match\n");

        tensor_free(c);    // frees its own data (owns_data == 1)
        tensor_free(tr);   // struct only (owns_data == 0)
        tensor_free(t);
    }

    printf("ALL VIEW CHECKS PASSED\n");
    return 0;
}
