#ifndef UTIL_H
#define UTIL_H
#include <iostream>

#define CERROR(e) {                                     \
  if (e != cudaSuccess) {                                      \
    std::cerr << "CUDA error: " << cudaGetErrorString(e)     \
              << " in file " << __FILE__                     \
              << " at line " << __LINE__ << std::endl;       \
    exit(EXIT_FAILURE);                                       \
  }                                                            \
}

#define BLOCKS(N, blockSize) (((N) + (blockSize) - 1) / (blockSize))

#define checkError(msg) {                                   \
  cudaError_t err = cudaGetLastError();                     \
  if (err != cudaSuccess) {                                 \
      std::cerr << msg << ": " << cudaGetErrorString(err)   \
                << " in file " << __FILE__                  \
                << " at line " << __LINE__ << std::endl;    \
      exit(EXIT_FAILURE);                                   \
  }                                                         \
}                                                           

#endif