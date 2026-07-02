#include "tensor.h"
#include <assert.h>

// Stage 4 gate: matrix multiply.
// The inner-dimension mismatch case is tested separately in
// test_matmul_mismatch.c, since that path calls exit(1).

int main(void) {

    /* ---- [2,3] x [3,2] against a hand-computed result ----
     *
     *   A = | 1 2 3 |     B = |  7  8 |
     *       | 4 5 6 |         |  9 10 |
     *                         | 11 12 |
     *
     *   C = A*B = | 1*7+2*9+3*11  1*8+2*10+3*12 | = |  58  64 |
     *             | 4*7+5*9+6*11  4*8+5*10+6*12 |   | 139 154 |
     */
    {
        int as[2] = {2, 3};
        int bs[2] = {3, 2};
        Tensor *a = tensor_create(as, 2);
        Tensor *b = tensor_create(bs, 2);

        float av[6] = {1, 2, 3, 4, 5, 6};
        float bv[6] = {7, 8, 9, 10, 11, 12};
        for (int i = 0; i < 6; i++) { a->data[i] = av[i]; b->data[i] = bv[i]; }

        Tensor *c = tensor_matmul(a, b);

        assert(c->ndim == 2);
        assert(c->shape[0] == 2 && c->shape[1] == 2);
        assert(c->size == 4);

        assert(c->data[0] == 58.0f);   // c[0][0]
        assert(c->data[1] == 64.0f);   // c[0][1]
        assert(c->data[2] == 139.0f);  // c[1][0]
        assert(c->data[3] == 154.0f);  // c[1][1]

        printf("[PASS] tensor_matmul: [2,3] x [3,2] matches hand computation\n");

        tensor_free(a);
        tensor_free(b);
        tensor_free(c);   // matmul allocated this, we own it
    }

    /* ---- multiplying by the identity returns the original matrix ---- */
    {
        int shape[2] = {2, 2};
        Tensor *m = tensor_create(shape, 2);
        Tensor *id = tensor_create(shape, 2);

        float mv[4] = {1, 2, 3, 4};
        float iv[4] = {1, 0, 0, 1};   // 2x2 identity
        for (int i = 0; i < 4; i++) { m->data[i] = mv[i]; id->data[i] = iv[i]; }

        Tensor *c = tensor_matmul(m, id);
        for (int i = 0; i < 4; i++) {
            assert(c->data[i] == m->data[i]);
        }
        printf("[PASS] tensor_matmul: M x I == M\n");

        tensor_free(m);
        tensor_free(id);
        tensor_free(c);
    }

    printf("ALL MATMUL CHECKS PASSED\n");
    return 0;
}
