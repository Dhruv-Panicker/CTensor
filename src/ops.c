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
