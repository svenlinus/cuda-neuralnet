%%writefile test_matrix.cu
#include "string.h"
#include "util.h"
#include "matrix.h"
#include <stdio.h>
#include <string.h>

void test_matrixMult() {
  // (4,2)*(2,3) = (4,3)
  float a[8] = {-2,-1,
                 0, 1,
                 2, 3,
                 4, 5};
  float b[6] = {-1,0,1,
                 0,1,0};
  Matrix *A, *B, *C;
  initMatrix(&A, 4, 2);
  initMatrix(&B, 2, 3);
  initMatrix(&C, 4, 3);
  setDeviceMatrixData(A, a, 8);
  setDeviceMatrixData(B, b, 6);

  matrixMult<<<1, 512>>>(A, B, C);
  cudaDeviceSynchronize();
  checkError("Matrix mult");

  float c[12];
  getDeviceMatrixData(c, C, 12);

  char result[64];
  char expected[64] = "2 -1 -2 0 1 0 -2 3 2 -4 5 4";
  int offset = 0;
  for (int i = 0; i < 12; ++i) {
    offset += snprintf(result + offset, sizeof(result) - offset, "%d ", (int)c[i]);
  }
  printf("Testing matrix mult\n");
  printf("Result: %s\n", result);
  printf("Expect: %s\n", expected);
  if (strncmp(result, expected, 27) != 0) {
    printf("FAILED\n");
    exit(EXIT_FAILURE);
  }
  printf("\nPASSED\n\n");

  freeMatrix(&A);
  freeMatrix(&B);
  freeMatrix(&C);
}

void test_matrixElementWise() {
  Matrix *A, *B, *C;
  float data[6] = {0,1,2,3,4,5};

  initMatrix(&A, 2, 3);
  initMatrix(&B, 2, 3);
  initMatrix(&C, 2, 3);
  setDeviceMatrixData(A, data, 6);
  setDeviceMatrixData(B, data, 6);

  matrixAdd<<<1, 512>>>(A, B, C, 1);
  cudaDeviceSynchronize();
  checkError("Matrix add");

  float c[6];
  getDeviceMatrixData(c, C, 6);
  
  char result[64];
  int offset = 0;
  for (int i = 0; i < 6; ++i) {
    offset += snprintf(result + offset, sizeof(result) - offset, "%d ", (int)c[i]);
  }
  printf("Testing matrix add\n");
  printf("Result: %s\n", result);
  printf("Expect: 0 2 4 6 8 10\n");
  if (strncmp(result, "0 2 4 6 8 10", 12) != 0) {
    printf("FAILED\n");
    exit(EXIT_FAILURE);
  }

  matrixAdd<<<1, 512>>>(A, B, C, -1);
  cudaDeviceSynchronize();
  checkError("Matrix sub");
  getDeviceMatrixData(c, C, 6);
  
  offset = 0;
  for (int i = 0; i < 6; ++i) {
    offset += snprintf(result + offset, sizeof(result) - offset, "%d ", (int)c[i]);
  }
  printf("Testing matrix sub\n");
  printf("Result: %s\n", result);
  printf("Expect: 0 0 0 0 0 0\n");
  if (strncmp(result, "0 0 0 0 0 0", 11) != 0) {
    printf("FAILED\n");
    exit(EXIT_FAILURE);
  }

  hadamardProd<<<1, 512>>>(A, B, C);
  cudaDeviceSynchronize();
  checkError("Matrix hadamard");
  getDeviceMatrixData(c, C, 6);
  
  offset = 0;
  for (int i = 0; i < 6; ++i) {
    offset += snprintf(result + offset, sizeof(result) - offset, "%d ", (int)c[i]);
  }
  printf("Testing hadamardProd \n");
  printf("Result: %s\n", result);
  printf("Expect: 0 1 4 9 16 25\n");
  if (strncmp(result, "0 1 4 9 16 25", 13) != 0) {
    printf("FAILED\n");
    exit(EXIT_FAILURE);
  }

  sigmoid<<<1, 512>>>(C, C);
  cudaDeviceSynchronize();
  checkError("Matrix sigmoid");
  getDeviceMatrixData(c, C, 6);
  
  offset = 0;
  for (int i = 0; i < 6; ++i) {
    offset += snprintf(result + offset, sizeof(result) - offset, "%.2f ", c[i]);
  }
  printf("Testing sigmoid \n");
  printf("Result: %s\n", result);
  printf("Expect: 0.50 0.73 0.98 1.00 1.00 1.00\n");
  if (strncmp(result, "0.50 0.73 0.98 1.00 1.00 1.00", 28) != 0) {
    printf("FAILED\n");
    exit(EXIT_FAILURE);
  }
  printf("\nPASSED\n\n");

  freeMatrix(&A);
  freeMatrix(&B);
  freeMatrix(&C);
}

void test_transpose() {
  Matrix *A, *tA, *result;
  float a[8] = {
    0,1,2,3,
    4,5,6,7 
  };
  initMatrix(&A, 2, 4);     // A (2,4)
  setDeviceMatrixData(A, a, 8);
  matrixTranpose(A, &tA);   // tA (4,2)
  cudaDeviceSynchronize();
  checkError("Transpose");

  initMatrix(&result, 2, 2);
  matrixMult<<<1, 512>>>(A, tA, result);  // (2,4)(4,2) = (2,2)
  cudaDeviceSynchronize();
  checkError("Matrix mult");

  float c[12];
  getDeviceMatrixData(c, result, 4);

  char result[32];
  char expected[32] = "14 38 38 126";
  int offset = 0;
  for (int i = 0; i < 12; ++i) {
    offset += snprintf(result + offset, sizeof(result) - offset, "%d ", (int)c[i]);
  }
  printf("Testing matrix mult\n");
  printf("Result: %s\n", result);
  printf("Expect: %s\n", expected);
  if (strncmp(result, expected, 12) != 0) {
    printf("FAILED\n");
    exit(EXIT_FAILURE);
  }
  printf("\nPASSED\n\n");

  freeMatrix(&A);
  freeMatrix(&tA);
  freeMatrix(&result);
}

int main() {
  
  test_matrixMult();
  test_matrixElementWise();
  test_transpose();

  return 0;
}