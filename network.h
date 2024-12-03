#ifndef NETWORK_H
#define NETWORK_H
#include "matrix.h"

typedef struct Layer {
  int size;
  struct Layer *prev;
  Matrix *weights;
  Matrix *bias;
  Matrix *neurons;
  Matrix *delta;
  Matrix *gradient;
  Matrix *error;
  struct Layer *next;
} Layer;

typedef struct {
  Layer *input;
  Layer *layers;
  Layer *output;
  int numLayers;
} Network;

typedef struct {
  Network *network;
  int input;
  int hidden;
  int output;
  float learningRate;
  int batchSize;
} Model;

Model *initModel(int batchSize, float learningRate);
void addInputLayer(Model *model, int size);
void addDenseLayer(Model *model, int size);
void forward(Model *model, float *input);
float backward(Model *model, float *target);
void compileModel(Model *model);

#endif