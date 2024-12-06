#include "network.h"
#include "util.h"
#include <stdio.h>
#include <stdint.h>

void addDenseLayer(Model *model, int size) {
  Layer *prev = model->network->output;
  Layer *layer = (Layer *)calloc(1, sizeof(Layer));
  prev->next = layer;
  layer->prev = prev;
  layer->size = size;
  initMatrix(&layer->neurons, model->batchSize, size);
  initZerosMatrix(&layer->bias, 1, size);
  initRandomMatrix(&layer->weights, prev->size, size);
  model->network->numLayers ++;
  model->network->output = layer;
  model->output = size;
}

void addInputLayer(Model *model, int size) {
  model->input = size;
  Layer *layer = (Layer *)calloc(1, sizeof(Layer));
  initMatrix(&layer->neurons, model->batchSize, size);
  model->network->layers = layer;
  model->network->output = layer;
  layer->size = size;
}

Model *initModel(int batchSize, float learningRate) {
  Model *model = (Model *)malloc(sizeof(Model));
  if (!model) { perror("malloc"); exit(1); }

  Network *nn = (Network *)malloc(sizeof(Network));
  if (!nn) { perror("malloc"); exit(1); }

  model->network = nn;
  model->learningRate = learningRate;
  model->batchSize = batchSize;

  checkError("Init network");
  return model;
}

void layerForward(Layer *layer, int batchSize) {
  int size = batchSize * layer->size;
  deviceMatrixMult(layer->prev->neurons, layer->weights, layer->neurons, size);
  deviceMatrixAddVec(layer->neurons, layer->bias, layer->neurons, size);
  deviceSigmoid(layer->neurons, layer->neurons, size);
}

void forward(Model *model, float *input) {
  Network net = *(model->network);
  // y = s(x⋅wxh + bh)⋅why + by
  int inputBatch = model->batchSize * model->input;
  setDeviceMatrixData(net.layers->neurons, input, inputBatch);

  Layer *curr = net.layers->next;
  for (int i = 0; i < net.numLayers; ++i) {
    if (!curr) break;
    layerForward(curr, model->batchSize);
    curr = curr->next;
  }
}

float loss(Matrix *error, int n) {
  float *buff = (float *)malloc(sizeof(float) * n);
  getDeviceMatrixData(buff, error, n);
  float sum = 0;
  for (int i = 0; i < n; i++)
    sum += buff[i] * buff[i];
  free(buff);
  return sum;
}

void initLayerGradients(Layer *layer, int batchSize) {
  initMatrix(&layer->gradient, batchSize, layer->size);      // (b,n)
  initMatrix(&layer->delta, layer->prev->size, layer->size); // (p,n)
  if (layer->prev)
    initMatrix(&layer->error, batchSize, layer->size);       // (b,n)
}

void compileModel(Model *model) {
  Layer *curr = model->network->layers->next;
  for (int i = 0; i < model->network->numLayers; ++i) {
    if (!curr) break;
    initLayerGradients(curr, model->batchSize);
    curr = curr->next;
  }
}

void layerBackward(Layer *layer, Model *model) {
  Layer *prev = layer->prev;
  Matrix *tPrev, *tWeights;
  int batchSize = model->batchSize;
  // Calculate gradient
  int matSize = batchSize * layer->size;
  deviceSigmoidOutputDerivative(layer->neurons, layer->gradient, matSize);          // sod(y)
  deviceHadamardProd(layer->gradient, layer->error, layer->gradient, matSize);      // sod(y) ⊙ error
  deviceMatrixScale(layer->gradient, model->learningRate, layer->gradient, matSize);// (sod(y) ⊙ error) ⋅ lr
  // Calculate delta weights
  int weightSize = prev->size*layer->size;
  matrixTranpose(prev->neurons, &tPrev, batchSize, prev->size);        // (p,b)
  deviceMatrixMult(tPrev, layer->gradient, layer->delta, weightSize);  // (p,b)(b,n)=(p,n)
  deviceMatrixScale(layer->delta, 1.0f/batchSize, layer->delta, weightSize);
  freeMatrix(tPrev);

  if (prev->prev) {
    // Calculate previous layer error
    int prevSize = batchSize * prev->size;
    matrixTranpose(layer->weights, &tWeights, prev->size, layer->size); // (n,p)
    deviceMatrixMult(layer->error, tWeights, prev->error, prevSize);    // (b,n)(n,p)=(b,p)
    freeMatrix(tWeights);
  }
}

void layerUpdate(Layer *layer, int batchSize) {
  // Update bias
  Matrix *reducedGrad;
  initMatrix(&reducedGrad, 1, layer->size);
  deviceMatrixReduceRows(layer->gradient, reducedGrad, batchSize, layer->size);
  deviceMatrixAdd(layer->bias, reducedGrad, layer->bias, layer->size);
  freeMatrix(reducedGrad);
  // Update wieghts
  deviceMatrixAdd(layer->weights, layer->delta, layer->weights, layer->prev->size*layer->size);
}

float backward(Model *model, float *target) {
  Network net = *(model->network);

  int batchSize = model->batchSize;
  int outputSize = batchSize * model->output;
  Layer *curr = net.output;
  setDeviceMatrixData(curr->error, target, outputSize);
  deviceMatrixSub(curr->error, curr->neurons, curr->error, outputSize);
  float _loss;
  squareLoss(curr->error, &_loss, batchSize, model->output);

  for (int i = 0; i < net.numLayers; ++i) {
    if (!curr->prev) break;
    layerBackward(curr, model);
    curr = curr->prev;
  }
  curr = net.output;
  for (int i = 0; i < net.numLayers; ++i) {
    if (!curr->prev) break;
    layerUpdate(curr, batchSize);
    curr = curr->prev;
  }

  return _loss;
}

int modelAccuracy(Model *model, float **images, uint8_t *labels) {
  forward(model, images[0]);
  float output[model->output * model->batchSize];
  getDeviceMatrixData(output, model->network->output->neurons, 10 * model->batchSize);

  int numCorrect = 0;
  for (int i = 0; i < model->batchSize; ++i) {
    int maxIdx = 0;
    for (int j = 1; j < model->output; ++j){
      if (output[i*model->output + j] > output[i*model->output + maxIdx])
        maxIdx = j;
    }
    if (labels[i] == maxIdx)
      numCorrect ++;
  }

  return numCorrect;
}