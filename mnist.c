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

float** loadMNISTImages(const char* filename, int numImages) {
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
  float **images = (float **)malloc(numImages * sizeof(float*));
  for (int i = 0; i < numImages; i++) {
    images[i] = (float *)malloc(rows * cols * sizeof(float));
  }

  // Read images and normalize to [0, 1]
  unsigned char pixel;
  for (int img = 0; img < numImages; img++) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        fread(&pixel, 1, 1, file);
        images[img][r * cols + c] = pixel / 255.0f;
      }
    }
  }

  fclose(file);
  return images;
}

uint8_t* loadMNISTLabels(const char* filename, int numLabels) {
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

  // Allocate memory for labels
  uint8_t *labels = (uint8_t *)malloc(numLabels * sizeof(uint8_t));
  fread(labels, 1, numLabels, file);

  fclose(file);
  return labels;
}

void convertToOneHot(uint8_t label, float* oneHot) {
  for (int i = 0; i < 10; i++) {
    oneHot[i] = (i == label) ? 1.0f : 0.0f;
  }
}

int main() {
  // Hyperparameters
  int inputSize = IMAGE_WIDTH * IMAGE_HEIGHT;  // 784
  int hiddenSize = 128;
  int outputSize = 10;
  int epochs = 5;

  // Initialize model
  Model* model = deviceInitNetwork(inputSize, hiddenSize, outputSize);

  // Load MNIST data
  float** trainImages = loadMNISTImages(MNIST_TRAIN_IMAGES, NUM_TRAIN_IMAGES);
  uint8_t* trainLabels = loadMNISTLabels(MNIST_TRAIN_LABELS, NUM_TRAIN_IMAGES);

  // Training loop
  for (int epoch = 0; epoch < epochs; epoch++) {
    float totalLoss = 0.0f;

    for (int i = 0; i < NUM_TRAIN_IMAGES; i++) {
      forward(model, trainImages[i]);
      float target[10] = {0};
      convertToOneHot(trainLabels[i], target);
      totalLoss += backward(model, target);
    }
    printf("Epoch %d completed, loss %f\n", epoch + 1, totalLoss);
  }

  for (int i = 0; i < NUM_TRAIN_IMAGES; i++) {
    free(trainImages[i]);
  }
  free(trainImages);
  free(trainLabels);

  return 0;
}