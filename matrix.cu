%%writefile matrix.cu
#include <iostream>
#include <cuda.h>
#include <curand_kernel.h>
#include <string.h>
#include "util.h"
#include "matrix.h"


/** MEMORY management **/
void initMatrix(Matrix **mat, int rows, int cols) {
  Matrix temp;
  temp.rows = rows;
  temp.cols = cols;

  CERROR( cudaMalloc(&(temp.data), rows * cols * sizeof(float)) );
  CERROR( cudaMalloc(mat, sizeof(Matrix)) );
  CERROR( cudaMemcpy(*mat, &temp, sizeof(Matrix), cudaMemcpyHostToDevice) );
}

void freeMatrix(Matrix **mat) {
  Matrix temp;
  CERROR( cudaMemcpy(&temp, *mat, sizeof(Matrix), cudaMemcpyDeviceToHost) );
  CERROR( cudaFree(temp.data) );
  CERROR( cudaFree(*mat) );
}

__global__ void initRandomData(Matrix *mat, float range) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < mat->rows * mat->cols) {
    curandState_t state;
    curand_init(1234, i, 0, &state);
    mat->data[i] = (float)(range + -((range * 2) * curand_uniform(&state)));
  }
}

void initRandomMatrix(Matrix **mat, int rows, int cols) {
  initMatrix(mat, rows, cols);
  initRandomData<<<(rows*cols + 511) / 512, 512>>>(*mat, 1.0f);
}

__global__ void initZerosData(Matrix *mat) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < mat->rows * mat->cols)
    mat->data[i] = 0;
}

void initZerosMatrix(Matrix **mat, int rows, int cols) {
  initMatrix(mat, rows, cols);
  initZerosData<<<(rows*cols + 511) / 512, 512>>>(*mat);
}

void getDeviceMatrixData(float *dest, Matrix *source, int n) {
  Matrix temp;
  CERROR( cudaMemcpy(&temp, source, sizeof(Matrix), cudaMemcpyDeviceToHost) );
  CERROR( cudaMemcpy(dest, temp.data, n * sizeof(float), cudaMemcpyDeviceToHost) );
}

void setDeviceMatrixData(Matrix *dest, float *source, int n) {
  Matrix temp;
  CERROR( cudaMemcpy(&temp, dest, sizeof(Matrix), cudaMemcpyDeviceToHost) );
  CERROR( cudaMemcpy(temp.data, source, n * sizeof(float), cudaMemcpyHostToDevice) );
}



/** HELPER **/
__device__ int size(Matrix *mat) {
  return mat->rows * mat->cols;
}



/** MATH **/
__global__ void matrixMult(Matrix *a, Matrix *b, Matrix *ab) {
  // calculate the row & col index of the element
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= a->rows * b->cols) return;

  int row = i / b->cols;
  int col = i % b->cols;
  float result = 0;
  // do dot product between row of a and col of b
  for(int k = 0; k < a->cols; ++k)
    result += a->data[row*(a->cols)+k] * b->data[k*(b->cols)+col];
  ab->data[row * b->cols + col] = result;   // (n,m) * (m,p) = (n,p)
}

__global__ void matrixAdd(Matrix *a, Matrix *b, Matrix *c, int negate) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size(c))
    c->data[i] = a->data[i] + (b->data[i] * negate);
}

__global__ void matrixScale(Matrix *a, float scale, Matrix *b) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size(b))
    b->data[i] = a->data[i] * scale;
}

__global__ void hadamardProd(Matrix *a, Matrix *b, Matrix *c) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size(c))
    c->data[i] = a->data[i] * b->data[i];
}

__global__ void sigmoid(Matrix *a, Matrix *b) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size(b))
    b->data[i] =  1/(1+exp(a->data[i] * -1));
}
__global__ void sigmoidOutputDerivative(Matrix *a, Matrix *b) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size(b)) {
    int x = a->data[i];
    b->data[i] = x * (1 - x);
  }
}


/** TRANSPOSE **/
void matrixTranpose(Matrix *a, Matrix **b, int arows, int acols) {
  initMatrix(b, acols, arows); // Create matrix with switched rows/cols
  transpose<<<(arows*acols + 511) / 512, 512>>>(a, *b);
}

__global__ void transpose(Matrix *a, Matrix *b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int rows = a->rows, cols = a->cols;
  if (i >= rows * cols) return;

  int new_i = (i % cols) * rows + (i / cols);  // (curr_col, curr_row)
  b->data[new_i] = a->data[i];
}