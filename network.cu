#include "network.h"
#include "util.h"
#include <stdio.h>

Model *deviceInitNetwork(int inputSize, int hiddenSize, int outputSize, int batchSize) {
  Model *model = (Model *)malloc(sizeof(Model));
  if (!model) { perror("malloc"); exit(1); }

  Network *nn = (Network *)malloc(sizeof(Network));
  if (!nn) { perror("malloc"); exit(1); }

  model->network = nn;
  model->input = inputSize;
  model->hidden = hiddenSize;
  model->output = outputSize;
  model->learningRate = 0.01;
  model->batchSize = batchSize;

  initMatrix(&(nn->input), batchSize, inputSize);      // (bs,i)
  initRandomMatrix(&(nn->wxh), inputSize, hiddenSize); // (bs,i)*(i,h) = (bs,h)
  initZerosMatrix(&(nn->bh), 1, hiddenSize);           // (bs,h)
  initMatrix(&(nn->hidden), batchSize, hiddenSize);    // (bs,h)
  initRandomMatrix(&(nn->why), hiddenSize, outputSize);// (bs,h)*(h,o) = (bs,o)
  initZerosMatrix(&(nn->by), 1, outputSize);           // (1,o)
  initMatrix(&(nn->output), batchSize, outputSize);    // (bs,o)

  checkError("Init network");
  return model;
}

void forward(Model *model, float *input) {
  Network net = *(model->network);
  // y = s(x⋅wxh + bh)⋅why + by
  // Load input
  int inputBatch = model->batchSize * model->input;
  setDeviceMatrixData(net.input, input, inputBatch);
  // Calc hidden layer
  int hiddenBatch = model->batchSize * model->hidden;
  deviceMatrixMult(net.input, net.wxh, net.hidden, hiddenBatch);
  deviceMatrixAddVec(net.hidden, net.bh, net.hidden, hiddenBatch);
  deviceSigmoid(net.hidden, net.hidden, hiddenBatch);
  // Calc output layer
  int outputBatch = model->batchSize * model->output;
  deviceMatrixMult(net.hidden, net.why, net.output, outputBatch);
  deviceMatrixAddVec(net.output, net.by, net.output, outputBatch);
  deviceSigmoid(net.output, net.output, outputBatch);
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

float backward(Model *model, float *target) {
  Network net = *(model->network);
  /** Calulate the derivate of the Cost function
   ** Math:
   * dC/dw = tH ⋅ ((sod(y) ⊙ error) ⋅ lr)
   ** Pseudo code:
   * dC/dw = matrixMult(
   *   transposedHidden,
   *   matrixScale(
   *     hadamardProd(
   *       sigmoidOutputDerivatie(output)
   *       error, 
   *     ), 
   *     learningRate
   *   )
   * );
  */

  // Copy target data to GPU
  int batchSize = model->batchSize;
  int outputBatch = batchSize * model->output;
  Matrix *error;
  initMatrix(&error, batchSize, model->output);     // (b,o)
  setDeviceMatrixData(error, target, outputBatch);
  // Calculate error
  deviceMatrixSub(error, net.output, error, outputBatch);               // error
  float _loss = loss(error, outputBatch);
  // Calculate gradient
  Matrix *gradient;
  initMatrix(&gradient, batchSize, model->output);    // (b,o)
  deviceSigmoidOutputDerivative(net.output, gradient, outputBatch);         // sod(y)
  deviceHadamardProd(gradient, error, gradient, outputBatch);               // sod(y) ⊙ error
  deviceMatrixScale(gradient, model->learningRate, gradient, outputBatch);  // (sod(y) ⊙ error) ⋅ lr
  // Update bias
  Matrix *reducedGrad;
  if (batchSize > 1) {
    initMatrix(&reducedGrad, 1, model->output);       // (1,o)
    deviceMatrixReduceRows(gradient, reducedGrad, batchSize, model->output);
  } else {
    reducedGrad = gradient;
  }
  deviceMatrixAdd(net.by, reducedGrad, net.by, model->output);
  // Calculate delta why weights
  Matrix *tHidden;
  matrixTranpose(net.hidden, &tHidden, batchSize, model->hidden); // (h,b)
  Matrix *deltaWhy;
  initMatrix(&deltaWhy, model->hidden, model->output);  // (h,o)
  deviceMatrixMult(tHidden, gradient, deltaWhy, model->hidden*model->output); // tH ⋅ ((sod(y) ⊙ error) ⋅ lr)
  freeMatrix(gradient);
  freeMatrix(tHidden);
  if (batchSize > 1) {
    deviceMatrixScale(deltaWhy, 1.0f/batchSize, deltaWhy, model->hidden*model->output);
    freeMatrix(reducedGrad);
  }

  // Calculate hidden layer error
  int hiddenBatch = batchSize * model->hidden;
  Matrix *hiddenError;
  initMatrix(&hiddenError, batchSize, model->hidden);           // (b,h)
  Matrix *tWhy;
  matrixTranpose(net.why, &tWhy, model->hidden, model->output); // (o,h)
  deviceMatrixMult(error, tWhy, hiddenError, hiddenBatch);      // (b,o)(o,h) = (b,h)
  freeMatrix(tWhy);
  // Hidden layer gradient
  initMatrix(&gradient, batchSize, model->hidden);              // (b,h)
  deviceSigmoidOutputDerivative(net.hidden, gradient, hiddenBatch);
  deviceHadamardProd(gradient, hiddenError, gradient, hiddenBatch);
  deviceMatrixScale(gradient, model->learningRate, gradient, hiddenBatch);
  // Update bias
  if (batchSize > 1) {
    initMatrix(&reducedGrad, 1, model->hidden);       // (1,h)
    deviceMatrixReduceRows(gradient, reducedGrad, batchSize, model->hidden);
  } else {
    reducedGrad = gradient;
  }
  deviceMatrixAdd(net.bh, reducedGrad, net.bh, model->hidden);
  // Calculate delta wxh weights
  freeMatrix(hiddenError);
  Matrix *tInput;
  matrixTranpose(net.input, &tInput, batchSize, model->input); // (i,b)
  Matrix *deltaWxh;
  initMatrix(&deltaWxh, model->input, model->hidden);  // (i,h)
  deviceMatrixMult(tInput, gradient, deltaWxh, model->input*model->hidden);  // (i,b)(b,h) = (i,h)
  if (batchSize > 1) {
    deviceMatrixScale(deltaWhy, 1.0f/batchSize, deltaWhy, model->hidden*model->output);
    freeMatrix(reducedGrad);
  }

  // Update weights
  deviceMatrixAdd(net.why, deltaWhy, net.why, model->hidden*model->output);
  deviceMatrixAdd(net.wxh, deltaWxh, net.wxh, model->input*model->hidden);
  freeMatrix(gradient);
  freeMatrix(tInput);
  freeMatrix(deltaWhy);
  freeMatrix(deltaWxh);

  return _loss;
}