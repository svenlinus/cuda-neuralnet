#ifndef MATRIX_H
#define MATRIX_H

/** STRUCTS **/
typedef struct {
  float *data;
  int rows;
  int cols;
} Matrix;

/** MEMORY management **/
void initMatrix(Matrix **mat, int rows, int cols);
void freeMatrix(Matrix *mat);
__global__ void initRandomData(Matrix *mat, float range);
__global__ void initZerosData(Matrix *mat);
void initRandomMatrix(Matrix **mat, int rows, int cols);
void initZerosMatrix(Matrix **mat, int rows, int cols);
void getDeviceMatrixData(float *dest, Matrix *source, int n);
void setDeviceMatrixData(Matrix *dest, float *source, int n);

/** HELPER **/
__device__ int size(Matrix *mat);

/** MATH **/
__global__ void matrixMult(Matrix *a, Matrix *b, Matrix *ab);
__global__ void matrixAdd(Matrix *a, Matrix *b, Matrix *c, int negate);
__global__ void matrixScale(Matrix *a, float scale, Matrix *b);
__global__ void hadamardProd(Matrix *a, Matrix *b, Matrix *c);
__global__ void sigmoid(Matrix *a, Matrix *b);
__global__ void sigmoidOutputDerivative(Matrix *a, Matrix *b);
void deviceMatrixMult(Matrix *a, Matrix *b, Matrix *ab, int N);
void deviceMatrixAdd(Matrix *a, Matrix *b, Matrix *c, int N);
void deviceMatrixSub(Matrix *a, Matrix *b, Matrix *c, int N);
void deviceMatrixScale(Matrix *a, float scale, Matrix *b, int N);
void deviceHadamardProd(Matrix *a, Matrix *b, Matrix *c, int N);
void deviceSigmoid(Matrix *a, Matrix *b, int N);
void deviceSigmoidOutputDerivative(Matrix *a, Matrix *b, int N);

/** TRANSPOSE **/
void matrixTranpose(Matrix *a, Matrix **b, int arows, int acols);
__global__ void transpose(Matrix *input, Matrix *output);

#endif