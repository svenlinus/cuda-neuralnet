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

  initMatrix(&(nn->input), 1, inputSize);              // (1,i)
  initRandomMatrix(&(nn->wxh), inputSize, hiddenSize); // (1,i)*(i,h) = (1,h)
  initZerosMatrix(&(nn->bh), 1, hiddenSize);           // (1,h)
  initMatrix(&(nn->hidden), 1, hiddenSize);            // (1,h)
  initRandomMatrix(&(nn->why), hiddenSize, outputSize);// (1,h)*(h,o) = (1,o)
  initZerosMatrix(&(nn->by), 1, outputSize);           // (1,o)
  initMatrix(&(nn->output), 1, outputSize);            // (1,o)

  checkError("Init network");
  return model;
}

void forward(Model *model, float *input) {
  Network net = *(model->network);

  int blockSize = 512;
  int blockNum = BLOCKS(model->hidden, blockSize);

  // y = s(x⋅wxh + bh)⋅why + by
  // Load input
  setDeviceMatrixData(net.input, input, model->input);
  // Calc hidden layer
  matrixMult<<<blockNum, blockSize>>>(net.input, net.wxh, net.hidden);
  cudaDeviceSynchronize();
  matrixAdd<<<blockNum, blockSize>>>(net.hidden, net.bh, net.hidden, 1);
  cudaDeviceSynchronize();
  sigmoid<<<blockNum, blockSize>>>(net.hidden, net.hidden);
  cudaDeviceSynchronize();
  // Calc output layer
  blockNum = BLOCKS(model->output, blockSize);
  matrixMult<<<blockNum, blockSize>>>(net.hidden, net.why, net.output);
  cudaDeviceSynchronize();
  matrixAdd<<<blockNum, blockSize>>>(net.output, net.by, net.output, 1);
  cudaDeviceSynchronize();
}

void backward(Model *model, float *target) {
  Network net = *(model->network);
  int blockSize = 512;
  int blockNum =  BLOCKS(model->output, blockSize);

  /** Calulate the derivate of the Cost function
   * dC/dw = tH ⋅ ((sod(y) ⊙ error) ⋅ lr)
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
  matrixAdd<<<blockNum, blockSize>>>(error, net.output, error, -1);   // error
  cudaDeviceSynchronize();
  checkError("Matrix sub");
  // Calculate gradient
  Matrix *gradient;
  initMatrix(&gradient, 1, model->output);
  sigmoidOutputDerivative<<<blockNum, blockSize>>>(net.output, gradient);   // sod(y)
  cudaDeviceSynchronize();
  checkError("Derivative");
  hadamardProd<<<blockNum, blockSize>>>(gradient, error, gradient);   // sod(y) ⊙ error
  cudaDeviceSynchronize();
  checkError("Hadamard");
  matrixScale<<<blockNum, blockSize>>>(gradient, model->learningRate, gradient);  // (sod(y) ⊙ error) ⋅ lr
  cudaDeviceSynchronize();
  checkError("Scale");
  // Calculate delta weights
  Matrix *tHidden;
  matrixTranpose(net.hidden, &tHidden, 1, model->hidden); // (h,1)
  cudaDeviceSynchronize();
  checkError("Transpose");
  Matrix *deltaWhy;
  initMatrix(&deltaWhy, model->hidden, model->output);  // (h,o)
  matrixMult<<<BLOCKS((model->hidden*model->output), blockSize), blockSize>>>
    (tHidden, gradient, gradient);  // (h,1)(1,o) = (h,o)                       tH ⋅ ((sod(y) ⊙ error) ⋅ lr)
  
}