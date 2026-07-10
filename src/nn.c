#include "tensor.h" 



void relu(Tensor *t) { 
	if(t == NULL) { exit(1); }

	for(int i = 0; i < t->size; i++) {
		if(t->data[i] < 0.0f) t->data[i] = 0.0f; 
	}
}

void sigmoid_inplace(Tensor *t) { 

	if(t == NULL) { exit(1); } 
	
	for(int i = 0; i < t->size; i++) {
		t->data[i] = 1.0f / (1.0f + expf(-t->data[i])); 
	}
} 

Tensor *softmax(Tensor *t) { 
	if(t == NULL) { exit(1); } 

	Tensor *c = tensor_create(t->shape, t->ndim); 
	if(c == NULL) { 
		free(c); 
		exit(1); 
	} 

	//Find the max value 
	float max_val = t->data[0]; 
	for(int i = 0; i < t->size; i++) {
		if (t->data[i] > max_val) max_val = t->data[i];
     	}

	//subtract by max, exponentiate and accumulate sum 
	float sum = 0.0f; 
	for(int i = 0; i < t->size; i++) {
		c->data[i] = expf(t->data[i] - max_val); 
		sum += c->data[i];  
	}

	//Divide by the sum 
	for(int i = 0; i < t->size; i++) {
		c->data[i] /= sum; 
	}

	return c; 

} 

Tensor *linear_forward(Tensor *x, Tensor *W, Tensor *b) {
	if( x == NULL || W == NULL || b == NULL) { 
		fprintf(stderr, "linear_forward_failed: NULL inputs given \n"); 
		exit(1); 
	} 

	Tensor *c = tensor_matmul(x, W); 
	
	//Add the bias 
	for(int i = 0; i < c->shape[0]; i++)
		for(int j = 0; j < c->shape[1]; j++)
			c->data[i * c->shape[1] + j] += b->data[j]; 
	
	return c; 
}
