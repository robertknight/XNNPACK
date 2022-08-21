#include <iostream>

#include <xnnpack.h>

bool check_status(xnn_status status, const char* description = 0) {
  if (status != xnn_status_success && description) {
      std::cout << description << " failed: " << status << std::endl;
      return false;
  }
  return true;
}

int main(int, char**) {
  auto status = xnn_initialize(nullptr /* allocator */);

  if (!check_status(status, "xnn_initialize")) {
    return 1;
  }

  xnn_operator_t max_pool_op;
  status = xnn_create_max_pooling2d_nhwc_s8(
    // Padding top, right, bottom, left
    0, 0, 0, 0,
    // Pooling height, width
    2, 2,
    // Stride height, width
    2, 2,
    // Dilation height, width
    1, 1,
    1, // Channels
    1, // Input pixel stride
    1, // Output pixel stride
    0, // Output min
    5, // Outut max
    0, // Flags
    &max_pool_op);

  if (!check_status(status, "xnn_create_max_pooling2d_nhwc_s8")) {
    return 1;
  }

  const int8_t input[4] = {3, 2, 5, 0};
  int8_t output[4] = {0, 0, 0, 0};

  status = xnn_setup_max_pooling2d_nhwc_s8(max_pool_op,
    1 /* batch size */, 2 /* input_height */, 2 /* input_height */, input, output, nullptr /* threadpool */);

  if (!check_status(status, "xnn_setup_max_pooling2d_nhwc_s8")) {
    return 1;
  }

  status = xnn_run_operator(max_pool_op, nullptr /* threadpool */);
  if (!check_status(status, "xnn_run_operator")) {
    return 1;
  }

  for (int i=0; i < 4; i++) {
    std::cout << "result" << i << ": " << static_cast<int>(output[i]) << std::endl;
  }

  status = xnn_delete_operator(max_pool_op);
  if (!check_status(status, "xnn_delete_operator")) {
    return 1;
  }

  status = xnn_deinitialize();
  if (!check_status(status, "xnn_deinitialize")) {
    return 1;
  }

  std::cout << "XNNPack test finished" << std::endl;
}
