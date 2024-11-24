#include <iostream>
#include <cuda.h>
#include <curand_kernel.h>
#include <string.h>
#include "util.h"
#include "matrix.h"
#define BLOCKDIM 512
#define MIN(a, b) (a < b ? a : b)


/** MEMORY management **/
void initMatrix(Matrix **mat, int rows, int cols) {
  Matrix temp;
  temp.rows = rows;
  temp.cols = cols;

  CERROR( cudaMalloc(&(temp.data), rows * cols * sizeof(float)) );
  CERROR( cudaMalloc(mat, sizeof(Matrix)) );
  CERROR( cudaMemcpy(*mat, &temp, sizeof(Matrix), cudaMemcpyHostToDevice) );
}

void freeMatrix(Matrix *mat) {
  Matrix temp;
  CERROR( cudaMemcpy(&temp, mat, sizeof(Matrix), cudaMemcpyDeviceToHost) );
  CERROR( cudaFree(temp.data) );
  CERROR( cudaFree(mat) );
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
void deviceMatrixMult(Matrix *a, Matrix *b, Matrix *ab, int N) {
  matrixMult<<<BLOCKS(N, BLOCKDIM), BLOCKDIM>>>(a, b, ab);
  cudaDeviceSynchronize();
  checkError("Matrix mult");
}

__global__ void matrixAdd(Matrix *a, Matrix *b, Matrix *c, int negate) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size(c))
    c->data[i] = a->data[i] + (b->data[i] * negate);
}
void deviceMatrixAdd(Matrix *a, Matrix *b, Matrix *c, int N) {
  matrixAdd<<<BLOCKS(N, BLOCKDIM), BLOCKDIM>>>(a, b, c, 1);
  cudaDeviceSynchronize();
  checkError("Matrix add");
}
void deviceMatrixSub(Matrix *a, Matrix *b, Matrix *c, int N) {
  matrixAdd<<<BLOCKS(N, BLOCKDIM), BLOCKDIM>>>(a, b, c, -1);
  cudaDeviceSynchronize();
  checkError("Matrix sub");
}
__global__ void matrixAddVec(Matrix *a, Matrix *b, Matrix *c) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size(c)) {
    int row = i / a->cols;
    int col = i % a->cols;
    c->data[row * a->cols + col] = a->data[row * a->cols + col] + b->data[col];
  }
}
void deviceMatrixAddVec(Matrix *a, Matrix *b, Matrix *c, int N) {
  matrixAddVec<<<BLOCKS(N, BLOCKDIM), BLOCKDIM>>>(a, b, c);
  cudaDeviceSynchronize();
  checkError("Matrix add vector");
}

__global__ void reduceRows(Matrix *x, Matrix *y) {
  int row = threadIdx.x;
  int col = blockIdx.x;
  if (col >= x->cols) return;

  extern __shared__ float shared[];

  float result = 0.0f;
  for (int i = row; i < x->rows; i += blockDim.x) {
    result += x->data[i * x->cols + col];
  }
  shared[row] = result;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s /= 2) {
    if (row < s && (row + s) < blockDim.x) {
      shared[row] += shared[row + s];
    }
    __syncthreads();
  }

  if (row == 0) {
    y->data[col] = shared[0];
  }
}
void deviceMatrixReduceRows(Matrix *x, Matrix *y, int rows, int cols) {
  int blockSize = MIN(rows / 2, 1024);
  int blockNum = cols;
  reduceRows<<<blockNum, blockSize, blockSize>>>(x, y);
}

__global__ void matrixScale(Matrix *a, float scale, Matrix *b) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size(b))
    b->data[i] = a->data[i] * scale;
}
void deviceMatrixScale(Matrix *a, float scale, Matrix *b, int N) {
  matrixScale<<<BLOCKS(N, BLOCKDIM), BLOCKDIM>>>(a, scale, b);
  cudaDeviceSynchronize();
  checkError("Matrix scale");
}

__global__ void hadamardProd(Matrix *a, Matrix *b, Matrix *c) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size(c))
    c->data[i] = a->data[i] * b->data[i];
}
void deviceHadamardProd(Matrix *a, Matrix *b, Matrix *c, int N) {
  hadamardProd<<<BLOCKS(N, BLOCKDIM), BLOCKDIM>>>(a, b, c);
  cudaDeviceSynchronize();
  checkError("Hadamard");
}

__global__ void sigmoid(Matrix *a, Matrix *b) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size(b))
    b->data[i] =  1/(1+exp(a->data[i] * -1));
}
void deviceSigmoid(Matrix *a, Matrix *b, int N) {
  sigmoid<<<BLOCKS(N, BLOCKDIM), BLOCKDIM>>>(a, b);
  cudaDeviceSynchronize();
  checkError("Sigmoid");
}

__global__ void sigmoidOutputDerivative(Matrix *a, Matrix *b) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size(b)) {
    float x = a->data[i];
    b->data[i] = x * (1 - x);
  }
}
void deviceSigmoidOutputDerivative(Matrix *a, Matrix *b, int N) {
  sigmoidOutputDerivative<<<BLOCKS(N, BLOCKDIM), BLOCKDIM>>>(a, b);
  cudaDeviceSynchronize();
  checkError("Derivative");
}


/** TRANSPOSE **/
void matrixTranpose(Matrix *a, Matrix **b, int arows, int acols) {
  initMatrix(b, acols, arows); // Create matrix with switched rows/cols
  transpose<<<(arows*acols + 511) / 512, 512>>>(a, *b);
  cudaDeviceSynchronize();
  checkError("Transpose");
}
__global__ void transpose(Matrix *a, Matrix *b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int rows = a->rows, cols = a->cols;
  if (i >= rows * cols) return;

  int new_i = (i % cols) * rows + (i / cols);  // (curr_col, curr_row)
  b->data[new_i] = a->data[i];
}