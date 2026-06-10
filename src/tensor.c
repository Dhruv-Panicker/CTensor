#include "tensor.h"


//Function to allocate and return a new tensor 
Tensor *tensor_create(int *shape, int ndim) {
	if (shape == NULL || ndim <= 0) { 
		return NULL; 
	} 
	Tensor *t = malloc(sizeof(Tensor)); 

	if (t == NULL)  return NULL;

	int size = 1; 
	for (int i = 0; i < ndim; i++)  size *= shape[i];

	t->data = malloc(size * sizeof(float));
	if(t->data == NULL) {
		free(t);
		return NULL;
	}

	t->shape = malloc(ndim * sizeof(int));
	if(t->shape == NULL) {
		free(t->data);
		free(t);
		return NULL;
	}

	//copying elements from shape into tensor
	for (int i = 0; i < ndim; i++) {
		t->shape[i] = shape[i];
	}

	t->strides = malloc(ndim * sizeof(int));
	if(t->strides == NULL) {
		free(t->shape);
		free(t->data);
		free(t);
		return NULL;
	}

	//Starting from last index calculate the number of strides neeed to index in tensor 
	t->strides[ndim - 1] = 1;
	for(int i = ndim - 2; i >= 0; i--) {
		t->strides[i] = t->strides[i + 1] * shape[i + 1];
	}

	t->ndim = ndim;
	t->size = size;

	return t;
}

//Deallocate memory for tensor 
void tensor_free(Tensor *t) {

	if (t == NULL) return ; 

	free(t->data); 
	free(t->shape); 
	free(t->strides); 
	free(t); 
}


Tensor *tensor_zeros(int *shape, int ndim) {

	Tensor *t = tensor_create(shape, ndim); 

	if (t == NULL) return NULL; 

	//zero out all data in the tensor
	memset(t->data, 0, t->size * sizeof(float)); 

	return t; 

}


Tensor *tensor_rand(int *shape, int ndim) {
	
	Tensor *t = tensor_create(shape, ndim); 

	if(t == NULL) return NULL; 

	//fill the data in the tensor with rand numbers 
	for (int i = 0; i < t->size; i++) {
		t->data[i] = (float)rand() / ((float)RAND_MAX + 1.0f); 
	}

	return t; 
}



float tensor_get(Tensor *t, int *indicies) { 

	if (t == NULL || indicies == NULL) return NAN; 

	//calulate the indicie being accessed through strides 
	int flat = 0; 
	for (int i = 0; i < t->ndim; i++) {
		flat += indicies[i] * t->strides[i]; 
	}

	return t->data[flat]; 
} 


void tensor_set(Tensor *t, int *indicies, float elem) {

	if(t == NULL || indicies == NULL) return; 

	//calulate the indicie being accessed through strides 
	int flat = 0; 
	for (int i = 0; i < t->ndim; i++) {
		flat += indicies[i] * t->strides[i]; 
	}

	t->data[flat] = elem; 


} 


void tensor_print(Tensor *t) {

	if(t == NULL) return; 

	//Print the shape 
	printf("Shape: ["); 
	for(int i = 0; i < t->ndim; i++) {
		printf("%d", t->shape[i]); 
		if(i < t->ndim - 1) printf(", "); 
	}
	printf("]\n"); 
	

	//Print the strides 
	printf("Strides: ["); 
	for(int i = 0; i < t->ndim; i++) {
		printf("%d", t->strides[i]); 
		if(i < t->ndim - 1) printf(", "); 
	}
	printf("]\n"); 
	

	//Print the data formatted by dimensions	
	printf("Data:\n"); 
	int last_dim = t->shape[t->ndim - 1]; 
	for(int i = 0; i < t->size; i++) {
		printf("%.4f ", t->data[i]); 
		if((i + 1) % last_dim == 0) printf("\n"); //new line after each row
	} 
	
}
