#include "string.h"
#include "util.h"
#include "network.h"

void test_networkForward() {
  Model *model = deviceInitNetwork(2, 2, 1);
  Network net = *(model->network);

  float input[] = {1, 2};
  float wxh[] = {-1, 1,
                 -1, 1}; // xm = {-3, 3}
  float bh[] = {1, -1};  // xm + b = {-2, 2}
                         // h = s(xm + b) = {0.12, 0.88}
  float why[] = {2, 2};  // hm = {2}
  float by[] = {1.5};    // y = hm + b = {3.5}

  setDeviceMatrixData(net.wxh, wxh, 4);
  setDeviceMatrixData(net.bh, bh, 2);
  setDeviceMatrixData(net.why, why, 2);
  setDeviceMatrixData(net.by, by, 1);

  forward(model, input);

  float output;
  getDeviceMatrixData(&output, net.output, 1);

  char expected[] = "3.5";
  char result[10];
  snprintf(result, sizeof(result), "%.1f", output);
  printf("Result: %.1f\n", result);
  printf("Expect: %s\n", expected);
  if (strncmp(result, expected, 3) != 0) {
    printf("FAILED\n");
    exit(EXIT_FAILURE);
  }
  printf("\nPASSED\n\n");
}

int main() {
  test_networkForward();

  return 0;
}