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
