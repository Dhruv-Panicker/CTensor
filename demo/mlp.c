#include "tensor.h" 

int main(void) { 


	//intialize the seed 
	srand(42); 
	
	//intialize weights and biases tensors 
	int w1_shape[2] = {4, 8}; 
	int b1_shape[2] = {1, 8}; 
	int w2_shape[2] = {8, 3};
	int b2_shape[2] = {1, 3}; 

	Tensor *w1 = tensor_rand(w1_shape, 2); 
	Tensor *b1 = tensor_rand(b1_shape, 2); 
	Tensor *w2 = tensor_rand(w2_shape, 2); 
	Tensor *b2 = tensor_rand(b2_shape, 2); 

	tensor_scale(w1, 0.1f); 
	tensor_scale(w2, 0.1f); 

	//create input 
	int x_shape[2]  = {1, 4}; 
	Tensor *x = tensor_rand(x_shape, 2); 

	//Start a clock cycle 
	clock_t start = clock(); 

	//Forward pass
	Tensor *h_pre = linear_forward(x, w1, b1); 
	relu(h_pre); 
	Tensor *logits = linear_forward(h_pre, w2, b2); 
	Tensor *probs = softmax(logits); 
	

	printf("=== MLP Forward Pass ===\n\n");
	printf("Input:\n");        tensor_print(x);
	printf("Hidden (post-ReLU):\n"); tensor_print(h_pre);
	printf("Logits:\n");       tensor_print(logits);
	printf("Output probs:\n"); tensor_print(probs);

	//check probs sum to 1 
	float sum = 0.0f; 
	for(int i = 0; i < probs->size; i++) sum += probs->data[i]; 
	printf("Probability sum: %.6f (should be 1.0)\n", sum);

	//calculate elapsed time 
	double elapsed = (double) (clock() - start) / CLOCKS_PER_SEC; 
	printf("Forward pass time: %.6f seconds\n", elapsed);

	//free memory 
	tensor_free(w1); tensor_free(b1);
	tensor_free(w2); tensor_free(b2);
	tensor_free(x);
	tensor_free(h_pre);
	tensor_free(logits);
	tensor_free(probs);

	return 0; 
} 



