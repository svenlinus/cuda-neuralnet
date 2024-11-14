#include <iostream>

void checkError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      std::cerr << msg << ": " << cudaGetErrorString(err)
                << " in file " << __FILE__
                << " at line " << __LINE__ << std::endl;
      exit(EXIT_FAILURE);
  }
}