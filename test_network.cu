#include "util.h"
#include "network.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#define TWO_DEC(n) (((int)(n * 100)) / 100)

void test_networkForward(Model **m) {
  Model *model = initModel(1, 0.5f);
  addInputLayer(model, 2);
  addDenseLayer(model, 2);
  addDenseLayer(model, 1);
  Network net = *(model->network);

  float input[] = {1, 2};
  float wxh[] = {-1, 1,
                 -1, 1}; // xm = {-3, 3}
  float bh[] = {1, -1};  // xm + b = {-2, 2}
                         // h = s(xm + b) = {0.12, 0.88}
  float why[] = {2, 
                 2};     // hm = {2}
  float by[] = {1.5};    // y = s(hm + b) = {0.97}

  setDeviceMatrixData(net.layers->next->weights, wxh, 4);
  setDeviceMatrixData(net.layers->next->bias, bh, 2);
  setDeviceMatrixData(net.layers->next->next->weights, why, 2);
  setDeviceMatrixData(net.layers->next->next->bias, by, 1);

  forward(model, input);

  float output;
  getDeviceMatrixData(&output, net.output->neurons, 1);

  printf("Testing network forward\n");
  char expected[] = "0.97";
  char result[10];
  snprintf(result, sizeof(result), "%.2f", output);
  printf("Result: %.2f\n", output);
  printf("Expect: %s\n", expected);
  if (strncmp(result, expected, 4) != 0) {
    printf("FAILED\n");
    exit(EXIT_FAILURE);
  }
  printf("\nPASSED\n\n");

  *m = model;
}

void test_networkBackward(Model *model) {
  Network net = *(model->network);
                 // x = {1, 2}
  float wxh[4];  // wxh = {-1, 1, -1, 1}
  float bh[2];   // bh = {1, -1}
                 // h = {0.12, 0.88}
  float why[2];  // why = {2,
                 //        2}
  float by;      // by = {1.5}
                 // y = {0.97}
  float target[] = {-0.03};
  compileModel(model);
  printf("Testing network backward\n");
  float loss = backward(model, target);
  printf("Loss %.2f %.2f\n", loss, 1.0f);
  assert(TWO_DEC(1) == TWO_DEC(loss));
  Layer *output = net.output;
  Layer *hidden = output->prev;
  // error = target - y = -0.03 - 0.97 = -1
  // (sod(y) ⊙ error) ⋅ lr = 0.0291 * -1 * 0.5 = -0.01455
  // by = by + (-0.01455) = 1.5 - 0.01455 = 1.48545
  getDeviceMatrixData(&by, output->bias, 1);
  printf("Bias %.2f %.2f\n", by, 1.48545);
  assert(TWO_DEC(by) == TWO_DEC(1.48545));
  // tH * grad = 0.12 * -0.01455 = -0.001
  //             0.88              -0.012
  // why = why + (tH * grad) = {1.99, 1.98}
  getDeviceMatrixData(why, output->weights, 2);
  printf("Why1 %.2f %.2f\n", why[0], 1.998254);
  printf("Why2 %.2f %.2f\n", why[1], 1.987196);
  // error = error * tWhy = [-1] * [2, 2] = [-2, -2]
  // sod(h) = sod([0.12, 0.88]) = [0.1056, 0.1056]
  // (sod(h) ⊙ error) ⋅ lr = [0.1056, 0.1056] ⊙ [-2, -2] ⋅ 0.5 = [-0.1056, -0.1056]
  // bh = bh + [-0.1056, -0.1056] = [0.8944, -1.1056]
  getDeviceMatrixData(bh, hidden->bias, 2);
  printf("Bias1 %.2f %.2f\n", bh[0], 0.8944);
  assert(TWO_DEC(bh[0]) == TWO_DEC(0.8944));
  printf("Bias2 %.2f %.2f\n", bh[1], -1.1056);
  assert(TWO_DEC(bh[1]) == TWO_DEC(-1.1056));
  // tO * grad = 1 * [-0.1056, -0.1056] = -0.11 -0.11
  //             2                        -0.21 -0.21
  // wxh = wxh + (tO * grad) = -1.11, 0.89
  //                           -1.21  0.79
  getDeviceMatrixData(wxh, hidden->weights, 4);
  printf("Wxh1 %.2f %.2f\n", wxh[0], -1.11);
  printf("Wxh2 %.2f %.2f\n", wxh[1], 0.89);
  printf("Wxh3 %.2f %.2f\n", wxh[2], -1.21);
  printf("Wxh4 %.2f %.2f\n", wxh[3], 0.79);

  printf("\nPASSED\n\n");
}

int main() {
  Model *model;
  test_networkForward(&model);
  test_networkBackward(model);

  return 0;
}