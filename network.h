#ifndef NETWORK_H
#define NETWORK_H
#include "matrix.h"

// Matices are allocated on the GPU so cannot be accessed on the host device
typedef struct {
  Matrix *input;
  Matrix *wxh;
  Matrix *bh;
  Matrix *hidden;
  Matrix *why;
  Matrix *by;
  Matrix *output;
} Network;

// Model is allocated on the host device
typedef struct {
  Network *network;
  int input;
  int hidden;
  int output;
  float learningRate;
  int batchSize;
} Model;

Model *deviceInitNetwork(int inputSize, int hiddenSize, int outputSize);
void forward(Model *model, float *input);
float backward(Model *model, float *target);

#endif