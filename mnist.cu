#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "network.h"

#define MNIST_TRAIN_IMAGES "train-images-idx3-ubyte"
#define MNIST_TRAIN_LABELS "train-labels-idx1-ubyte"
#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define NUM_TRAIN_IMAGES 60000
#define NUM_TEST_IMAGES 10000

uint32_t reverseInt(uint32_t n) {
  return ((n >> 24) & 0xFF) | 
         ((n >> 8)  & 0xFF00) | 
         ((n << 8)  & 0xFF0000) | 
         ((n << 24) & 0xFF000000);
}

float** loadMNISTImages(const char* filename, int numImages, int batchSize) {
  FILE* file = fopen(filename, "rb");
  if (!file) {
    perror("Cannot open image file");
    exit(1);
  }

  // Read header
  uint32_t magic, numImages_file, rows, cols;
  fread(&magic, 4, 1, file);
  fread(&numImages_file, 4, 1, file);
  fread(&rows, 4, 1, file);
  fread(&cols, 4, 1, file);

  magic = reverseInt(magic);
  numImages_file = reverseInt(numImages_file);
  rows = reverseInt(rows);
  cols = reverseInt(cols);

  // Allocate memory for images
  int numBatches = numImages / batchSize;
  float **batches = (float **)malloc(numBatches * sizeof(float*));
  for (int i = 0; i < numBatches; i++) {
    batches[i] = (float *)malloc(batchSize * rows * cols * sizeof(float));
  }

  // Read images and normalize to [0, 1]
  unsigned char pixel;
  int imgSize = rows * cols;
  for (int batch = 0; batch < numBatches; batch++) {
    for (int img = 0; img < batchSize; img++) {
      for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
          fread(&pixel, 1, 1, file);
          int imgIndex = img * imgSize;
          int pixelIndex = r * cols + c;
          batches[batch][imgIndex + pixelIndex] = pixel / 255.0f;
        }
      }
    }
  }

  fclose(file);
  return batches;
}

void convertToOneHot(uint8_t label, float* oneHot) {
  for (int i = 0; i < 10; i++) {
    oneHot[i] = (i == label) ? 1.0f : 0.0f;
  }
}

float* loadMNISTLabels(const char* filename, int numLabels, uint8_t *compressedLabels) {
  FILE* file = fopen(filename, "rb");
  if (!file) {
    perror("Cannot open label file");
    exit(1);
  }

  // Read header
  uint32_t magic, numLabels_file;
  fread(&magic, 4, 1, file);
  fread(&numLabels_file, 4, 1, file);

  magic = reverseInt(magic);
  numLabels_file = reverseInt(numLabels_file);

  float *labels = (float *)malloc(numLabels * 10 * sizeof(float));
  for (int i = 0; i < numLabels; ++i) {
    uint8_t label;
    fread(&label, 1, 1, file);
    compressedLabels[i] = label;
    convertToOneHot(label, labels + (i * 10));
  }

  fclose(file);
  return labels;
}

void printArray(float *a, int n) {
  printf("[ ");
  for (int i = 0; i < n; ++i)
    printf("%.2f ", a[i]);
  printf("]\n");
}

int main() {
  // Hyperparameters
  const int inputSize = IMAGE_WIDTH * IMAGE_HEIGHT;  // 784
  const int hiddenSize = 128;
  const int outputSize = 10;
  const int epochs = 5;
  const int batchSize = 100;
  const float learningRate = 0.01;

  // Initialize model
  Model* model = initModel(batchSize, learningRate);
  addInputLayer(model, inputSize);
  addDenseLayer(model, hiddenSize);
  addDenseLayer(model, outputSize);
  compileModel(model);

  // Load MNIST data
  uint8_t *compressedLabels = (uint8_t *)malloc(NUM_TRAIN_IMAGES * sizeof(uint8_t));
  float **trainImages = loadMNISTImages(MNIST_TRAIN_IMAGES, NUM_TRAIN_IMAGES, batchSize);
  printf("Loaded %d images into %d batches of size %d\n", NUM_TRAIN_IMAGES, NUM_TRAIN_IMAGES / batchSize, batchSize);
  float *trainLabels = loadMNISTLabels(MNIST_TRAIN_LABELS, NUM_TRAIN_IMAGES, compressedLabels);
  printf("Loaded %d labels\n", NUM_TRAIN_IMAGES);

  // Training loop
  for (int epoch = 0; epoch < epochs; epoch++) {
    float totalLoss = 0.0f;

    for (int i = 0; i < NUM_TRAIN_IMAGES / batchSize; i++) {
      forward(model, trainImages[i]);
      totalLoss += backward(model, trainLabels + (i * 10 * batchSize));
    }
    printf("Epoch %d completed, loss %f\n", epoch + 1, totalLoss);
  }

  const int testIndex = 0;
  printf("\nTesting accuracy: \n");
  int numCorrect = 0;
  int numBatches = 600;
  for (int i = 0; i < numBatches; ++i)
    numCorrect += modelAccuracy(model, trainImages + i, compressedLabels + i * batchSize);
  printf("Accuracy %.3f\n\n", (float)(numCorrect) / (batchSize * numBatches));

  forward(model, trainImages[testIndex]);
  float output[10 * batchSize];
  getDeviceMatrixData(output, model->network->output->neurons, 10 * batchSize);

  for (int i = 0; i < batchSize; ++i) {
    printf("Image %d \nLabel:  ", testIndex + i);
    printArray(trainLabels + (i*10), 10);
    printf("Output: ");
    printArray(output + (i*10), 10);
  }

  for (int i = 0; i < NUM_TRAIN_IMAGES / batchSize; i++) {
    free(trainImages[i]);
  }
  free(trainImages);
  free(trainLabels);

  return 0;
}