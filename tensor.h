#ifndef TENSOR_H
#define TENSOR_H


#include <stddef.h>   
#include <stdio.h>    
#include <stdlib.h>  
#include <string.h>   
#include <math.h>     
#include <time.h>

typedef struct {
	float *data; //heap allocated memory for all elements 
	int *shape;  //dynamically allocated 
	int *strides; //dynmaically allocated based on ndim 
	int ndim;
	int size;
	int owns_data; // 1 = owns data buffer, 0 = view (shares another tensor's data)
} Tensor;

//Create a tensor, data is unintialized
Tensor *tensor_create(int *shape, int ndim);

//Free memory from tensor
void tensor_free(Tensor *t);

//Create a tensor of zeros
Tensor *tensor_zeros(int *shape, int ndim);

//Create a tensor with rand elements
Tensor *tensor_rand(int *shape, int ndim);

//Get a tensor, indicies array should match ndim
float tensor_get(Tensor *t, int *indicies);

//Set a value in a tensor
void tensor_set(Tensor *t, int *indicies, float elem);

//Print tensor
void tensor_print(Tensor *t);

/* element wise operations */

//Add two tensors return a new tensor  
Tensor *tensor_add(Tensor *a, Tensor *b); 

//Multiply two tensors return a new tensor 
Tensor *tensor_mul(Tensor *a, Tensor *b); 

//Add in place no allocation 
void tensor_add_inplace(Tensor *a, Tensor *b); 

//Multiply by scalar 
void tensor_scale(Tensor *a, float scalar); 

/*     operations     */ 

//Matrix multiply a tensor both must be greater than 2D and contiguous
Tensor *tensor_matmul(Tensor *a, Tensor *b);

//Matrix multipley with optmized memory computation 
Tensor *tensor_matmul_ikj(Tensor *a, Tensor *b); 
/*     views and layouts  */

//Reshape the view and returns a new tesnor, caller must free the returning struct but not its data
Tensor *tensor_reshape(Tensor *t, int *new_shape, int new_ndim); 


//Returns a new transpose tensor with dims swapped 
Tensor *tensor_transpose(Tensor *t); 

//Return a new copy of a tensor with contigous data
Tensor *tensor_contiguous(Tensor *t); 

/*    activation functions    */ 

//ReLu function sets negative value to 0 
void relu(Tensor *t); 

//Applies sigmoid activation to every element 
void sigmoid_inplace(Tensor *t); 

//Returns a new tensor with softmax applied to the last dimension
Tensor *softmax(Tensor *t); 

//Linear layer forward pass does matmul + add bias 
Tensor *linear_forward(Tensor *x, Tensor *W, Tensor *b); 


#endif
