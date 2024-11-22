#include "network.h"
#include "util.h"
#include <stdio.h>

Model *deviceInitNetwork(int inputSize, int hiddenSize, int outputSize) {
  Model *model = (Model *)malloc(sizeof(Model));
  if (!model) { perror("malloc"); exit(1); }

  Network *nn = (Network *)malloc(sizeof(Network));
  if (!nn) { perror("malloc"); exit(1); }

  model->network = nn;
  model->input = inputSize;
  model->hidden = hiddenSize;
  model->output = outputSize;
  model->learningRate = 0.01;
  int batch = 100;
  model->batchSize = batch;

  initMatrix(&(nn->input), batch, inputSize);          // (bs,i)
  initRandomMatrix(&(nn->wxh), inputSize, hiddenSize); // (bs,i)*(i,h) = (bs,h)
  initZerosMatrix(&(nn->bh), 1, hiddenSize);           // (1,h)
  initMatrix(&(nn->hidden), batch, hiddenSize);        // (bs,h)
  initRandomMatrix(&(nn->why), hiddenSize, outputSize);// (1,h)*(h,o) = (bs,o)
  initZerosMatrix(&(nn->by), 1, outputSize);           // (1,o)
  initMatrix(&(nn->output), batch, outputSize);        // (bs,o)

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
  int outputBatch = model->batchSize * model->input;
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
  Matrix *error;
  initMatrix(&error, 1, model->output);
  setDeviceMatrixData(error, target, model->output);
  // Calculate error
  deviceMatrixSub(error, net.output, error, model->output);               // error
  float _loss = loss(error, model->output);
  // Calculate gradient
  Matrix *gradient;
  initMatrix(&gradient, 1, model->output);
  deviceSigmoidOutputDerivative(net.output, gradient, model->output);         // sod(y)
  deviceHadamardProd(gradient, error, gradient, model->output);               // sod(y) ⊙ error
  deviceMatrixScale(gradient, model->learningRate, gradient, model->output);  // (sod(y) ⊙ error) ⋅ lr
  // Update bias
  deviceMatrixAdd(net.by, gradient, net.by, model->output);
  // Calculate delta why weights
  Matrix *tHidden;
  matrixTranpose(net.hidden, &tHidden, 1, model->hidden); // (h,1)
  Matrix *deltaWhy;
  initMatrix(&deltaWhy, model->hidden, model->output);  // (h,o)
  deviceMatrixMult(tHidden, gradient, deltaWhy, model->hidden*model->output); // tH ⋅ ((sod(y) ⊙ error) ⋅ lr)
  freeMatrix(gradient);
  freeMatrix(tHidden);

  // Calculate hidden layer error
  Matrix *hiddenError;
  initMatrix(&hiddenError, 1, model->hidden);
  Matrix *tWhy;
  matrixTranpose(net.why, &tWhy, model->hidden, model->output); // (o,h)
  deviceMatrixMult(error, tWhy, hiddenError, model->hidden);    // (1,o)(0,h) = (1,h)
  freeMatrix(error);
  freeMatrix(tWhy);
  // Hidden layer gradient
  initMatrix(&gradient, 1, model->hidden);
  deviceSigmoidOutputDerivative(net.hidden, gradient, model->hidden);
  deviceHadamardProd(gradient, hiddenError, gradient, model->hidden);
  deviceMatrixScale(gradient, model->learningRate, gradient, model->hidden);
  // Update bias
  deviceMatrixAdd(net.bh, gradient, net.bh, model->hidden);
  // Calculate delta wxh weights
  freeMatrix(hiddenError);
  Matrix *tInput;
  matrixTranpose(net.input, &tInput, 1, model->input); // (i,1)
  Matrix *deltaWxh;
  initMatrix(&deltaWxh, model->input, model->hidden);  // (i,h)
  deviceMatrixMult(tInput, gradient, deltaWxh, model->input*model->hidden);  // (i,1)(1,h) = (i,h)

  // Update weights
  deviceMatrixAdd(net.why, deltaWhy, net.why, model->hidden*model->output);
  deviceMatrixAdd(net.wxh, deltaWxh, net.wxh, model->input*model->hidden);
  freeMatrix(gradient);
  freeMatrix(tInput);
  freeMatrix(deltaWhy);
  freeMatrix(deltaWxh);

  return _loss;
}