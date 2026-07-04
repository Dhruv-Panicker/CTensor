#include "tensor.h" 


//Helper function to check if the tensors in the operations match 
static int shape_match(Tensor *a, Tensor *b) {
	if (a->ndim != b->ndim) return 0; 
	for(int i = 0; i < a->ndim; i++) { 
		if(a->shape[i] != b->shape[i]) return 0; 
	} 
	return 1; 
}

Tensor *tensor_add(Tensor *a, Tensor *b) {
	//Null check and assert shapes match 
	if(a == NULL || b == NULL || !shape_match(a, b)) {
		fprintf(stderr, "tensor_add:shape mismatch\n"); 
		exit(1); 
	} 
	Tensor *t = tensor_create(a->shape, a->ndim); 
	if( t == NULL) return NULL; 


	//Iterate over elements 
	for(int i = 0; i < t->size; i++) {
		t->data[i] = a->data[i] + b->data[i]; 
	}

	return t; 
}

Tensor *tensor_mul(Tensor *a, Tensor *b) {
	//Null check and assert shapes match 
	if(a == NULL || b == NULL || !shape_match(a, b)) {
		fprintf(stderr, "tensor_mul:shape mismatch\n"); 
		exit(1); 
	} 
	Tensor *t = tensor_create(a->shape, a->ndim); 
	if( t == NULL) return NULL; 


	//Iterate over elements 
	for(int i = 0; i < t->size; i++) {
		t->data[i] = a->data[i] * b->data[i]; 
	}

	return t; 
}



void tensor_add_inplace(Tensor *a, Tensor *b) { 
	//Null check and assert shapes match 
	if(a == NULL || b == NULL || !shape_match(a, b)) {
		fprintf(stderr, "tensor_add_in_place:shape mismatch\n"); 
		exit(1); 
	} 
	//Iterate over elements 
	for(int i = 0; i < a->size; i++) {
		a->data[i] += b->data[i]; 
	}
} 


void tensor_scale(Tensor *a, float scalar) { 
	//Null check and assert shapes match 
	if(a == NULL) {
		fprintf(stderr, "tensor_scale:not a valid tensor\n"); 
		exit(1); 
	} 
	//Iterate over elements 
	for(int i = 0; i < a->size; i++) {
		a->data[i] *= scalar; 
	}
}


Tensor *tensor_matmul(Tensor *a, Tensor *b) {
	if(a->ndim != 2 || b->ndim!= 2) {
		fprintf(stderr, "tensor_matmul inputs must be 2D\n"); 
		exit(1); 
	} 
	if(a->shape[1] != b->shape[0]) {
		fprintf(stderr, "tensor_matmul inner dimensions must match\n"); 
		exit(1); 
	} 

	//shape dimensions 
	int M = a->shape[0]; 
	int K = a->shape[1]; 
	int N = b->shape[1]; 

	int out_shape[2] = {M, N}; 
	
	//New Tensor 
	Tensor *c = tensor_zeros(out_shape, a->ndim); 

	//operation 
	for (int i = 0; i < M; i++)
		for(int j = 0; j < N; j++)
			for(int k = 0; k < K; k++)
				c->data[i*N + j] += a->data[i*K + k] * b->data[k*N + j]; 
	return c; 

}

Tensor *tensor_reshape(Tensor *t, int *new_shape, int new_ndim) {


	int new_size = 1; 
	for (int i = 0; i < new_ndim; i++) new_size *= new_shape[i]; 
	if(t->size != new_size) {
		fprintf(stderr, "reshape: element count mismatch\n"); 
		exit(1); 
	}

	Tensor *c = malloc(sizeof(Tensor));
	
	if (c == NULL)  return NULL;

	c->data = t->data;
	c->shape = malloc(new_ndim * sizeof(int));

	if(c->shape == NULL) {
		free(c);
		return NULL;
	}
	for(int i = 0; i < new_ndim; i++) {
		c->shape[i] = new_shape[i]; 
	}


	c->strides = malloc(new_ndim * sizeof(int));
	if(c->strides == NULL) {
		free(c->shape);
		free(c);
		return NULL;
	}


	//Starting from last index calculate the number of strides neeed to index in tensor 
	c->strides[new_ndim - 1] = 1;
	for(int i = new_ndim - 2; i >= 0; i--) {
		c->strides[i] = c->strides[i + 1] * new_shape[i + 1];
	}


	c->ndim = new_ndim;
	c->size = t->size;
	c->owns_data = 0; // view: shares t->data, does not own it

	return c; 
}


Tensor *tensor_transpose(Tensor *t) {

	//check the dimensions match 
	if(t->ndim != 2){
		fprintf(stderr, "transpose: mismatching number of dimensions\n"); 
		exit(1); 
	} 

	Tensor *c = malloc(sizeof(Tensor)); 

	if (c == NULL) return NULL; 
	c->data = t->data;
	c->shape = malloc(2 * sizeof(int)); 
	if(c->shape == NULL) {
		free(c); 
		return NULL; 
	} 
	c->strides = malloc(2 * sizeof(int)); 
	if(c->strides == NULL) {
		free(c); 
		return NULL; 
	} 

	//swap to transpose 
	c->shape[0] = t->shape[1]; 
	c->shape[1] = t->shape[0]; 

	c->strides[0] = t->strides[1]; 
	c->strides[1] = t->strides[0]; 

	c->owns_data = 0; 
	c->ndim = 2; 
	c->size = t->size; 

	return c; 

}



Tensor *tensor_contiguous(Tensor *t) {
	
	Tensor *c = tensor_create(t->shape, t->ndim); 
	if (c == NULL) return NULL; 

	int *indices = malloc(t->ndim * sizeof(int));
	if (indices == NULL) { tensor_free(c); return NULL; }


	for (int flat = 0; flat < t->size; flat++) {
    		// convert flat → indices[]
    		int remaining = flat;
    		for (int d = 0; d < t->ndim; d++) {
        		int count = 1;
			//Looking to the right to see dimesnion sizes 
        		for (int dd = d + 1; dd < t->ndim; dd++) count *= t->shape[dd];
        		indices[d] = remaining / count;
        		remaining  = remaining % count;
    		}
    		c->data[flat] = tensor_get(t, indices);
	}
	free(indices); 
	return c; 

}
