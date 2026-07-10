#include "tensor.h"
#include <assert.h>

// Stage 6 gate: activation functions and the linear layer.

int main(void) {

    /* ---- relu: negatives -> 0, zero -> 0, positives unchanged ---- */
    {
        int shape[2] = {1, 5};
        Tensor *t = tensor_create(shape, 2);
        float vals[5] = {-3.0f, -0.5f, 0.0f, 0.5f, 2.0f};
        for (int i = 0; i < 5; i++) t->data[i] = vals[i];

        relu(t);
        assert(t->data[0] == 0.0f);   // -3.0 -> 0
        assert(t->data[1] == 0.0f);   // -0.5 -> 0
        assert(t->data[2] == 0.0f);   //  0.0 -> 0
        assert(t->data[3] == 0.5f);   //  unchanged
        assert(t->data[4] == 2.0f);   //  unchanged
        printf("[PASS] relu: negatives and zero -> 0, positives unchanged\n");
        tensor_free(t);
    }

    /* ---- sigmoid: output in (0,1), sigmoid(0) ~= 0.5 ---- */
    {
        int shape[2] = {1, 4};
        Tensor *t = tensor_create(shape, 2);
        float vals[4] = {-10.0f, 0.0f, 10.0f, 1.0f};
        for (int i = 0; i < 4; i++) t->data[i] = vals[i];

        sigmoid_inplace(t);
        for (int i = 0; i < 4; i++) {
            assert(t->data[i] > 0.0f && t->data[i] < 1.0f);
        }
        assert(fabsf(t->data[1] - 0.5f) < 1e-6f);   // sigmoid(0) = 0.5
        printf("[PASS] sigmoid: all outputs in (0,1), sigmoid(0) == 0.5\n");
        tensor_free(t);
    }

    /* ---- softmax: outputs sum to 1.0, each in (0,1) ---- */
    {
        int shape[2] = {1, 5};
        Tensor *t = tensor_create(shape, 2);
        float vals[5] = {2.0f, 1.0f, 0.1f, 3.0f, 0.5f};
        for (int i = 0; i < 5; i++) t->data[i] = vals[i];

        Tensor *s = softmax(t);
        float sum = 0.0f;
        for (int i = 0; i < s->size; i++) {
            assert(s->data[i] > 0.0f && s->data[i] < 1.0f);
            sum += s->data[i];
        }
        printf("softmax sum = %f\n", sum);
        assert(fabsf(sum - 1.0f) < 1e-5f);
        printf("[PASS] softmax: outputs in (0,1) and sum to 1.0\n");

        tensor_free(s);   // softmax returned a new tensor, we own it
        tensor_free(t);
    }

    /* ---- softmax: huge inputs must not overflow to inf/nan ---- */
    {
        int shape[2] = {1, 3};
        Tensor *t = tensor_create(shape, 2);
        t->data[0] = 1000.0f;
        t->data[1] = 1000.0f;
        t->data[2] =  999.0f;

        Tensor *s = softmax(t);
        float sum = 0.0f;
        for (int i = 0; i < s->size; i++) {
            assert(!isinf(s->data[i]) && !isnan(s->data[i]));
            sum += s->data[i];
        }
        assert(fabsf(sum - 1.0f) < 1e-5f);
        printf("[PASS] softmax: large inputs (1000.0f) stay finite, sum to 1.0\n");

        tensor_free(s);
        tensor_free(t);
    }

    /* ---- linear_forward: [1,4] x W[4,8] + b[1,8] -> [1,8] ---- */
    {
        int xs[2] = {1, 4};
        int ws[2] = {4, 8};
        int bs[2] = {1, 8};
        Tensor *x = tensor_create(xs, 2);
        Tensor *W = tensor_create(ws, 2);
        Tensor *b = tensor_create(bs, 2);

        for (int i = 0; i < x->size; i++) x->data[i] = 1.0f;      // all ones
        for (int i = 0; i < W->size; i++) W->data[i] = 1.0f;      // all ones
        for (int i = 0; i < b->size; i++) b->data[i] = (float)i;  // bias = j

        Tensor *y = linear_forward(x, W, b);

        assert(y->ndim == 2);
        assert(y->shape[0] == 1 && y->shape[1] == 8);   // output shape [1,8]

        // y[0][j] = sum_k (1 * 1) over 4 inputs, + b[j] = 4 + j
        for (int j = 0; j < 8; j++) {
            assert(y->data[j] == 4.0f + (float)j);
        }
        printf("[PASS] linear_forward: output shape [1,8], matmul + bias correct\n");

        tensor_free(x);
        tensor_free(W);
        tensor_free(b);
        tensor_free(y);   // linear_forward returned a new tensor, we own it
    }

    printf("ALL NN CHECKS PASSED\n");
    return 0;
}
