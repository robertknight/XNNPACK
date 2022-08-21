#include <iostream>

#include <xnnpack.h>

/* enum xnn_status xnn_create_max_pooling2d_nhwc_s8( */
/*   uint32_t input_padding_top, */
/*   uint32_t input_padding_right, */
/*   uint32_t input_padding_bottom, */
/*   uint32_t input_padding_left, */
/*   uint32_t pooling_height, */
/*   uint32_t pooling_width, */
/*   uint32_t stride_height, */
/*   uint32_t stride_width, */
/*   uint32_t dilation_height, */
/*   uint32_t dilation_width, */
/*   size_t channels, */
/*   size_t input_pixel_stride, */
/*   size_t output_pixel_stride, */
/*   int8_t output_min, */
/*   int8_t output_max, */
/*   uint32_t flags, */
/*   xnn_operator_t* max_pooling_op_out); */

/* enum xnn_status xnn_setup_max_pooling2d_nhwc_s8( */
/*   xnn_operator_t max_pooling_op, */
/*   size_t batch_size, */
/*   size_t input_height, */
/*   size_t input_width, */
/*   const int8_t* input, */
/*   int8_t* output, */
/*   pthreadpool_t threadpool); */

int main(int, char**) {
  auto status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    std::cout << "xnn_initialize failed" << std::endl;
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

  if (status != xnn_status_success) {
    std::cout << "xnn_create_max_pooling2d_nhwc_s8 failed" << status << std::endl;
    return 1;
  }

  const int8_t input[4] = {3, 2, 5, 0};
  int8_t output[4] = {0, 0, 0, 0};

  status = xnn_setup_max_pooling2d_nhwc_s8(max_pool_op,
    1 /* batch size */, 2 /* input_height */, 2 /* input_height */, input, output, nullptr /* threadpool */);

  if (status != xnn_status_success) {
    std::cout << "xnn_setup_max_pooling2d_nhwc_s8 failed" << status << std::endl;
    return 1;
  }

  status = xnn_run_operator(max_pool_op, nullptr /* threadpool */);

  if (status != xnn_status_success) {
    std::cout << "xnn_run_operator failed" << status << std::endl;
    return 1;
  }

  for (int i=0; i < 4; i++) {
    std::cout << "result" << i << ": " << static_cast<int>(output[i]) << std::endl;
  }

  status = xnn_delete_operator(max_pool_op);
  if (status != xnn_status_success) {
    std::cout << "xnn_delete_operator failed" << status << std::endl;
    return 1;
  }

  status = xnn_deinitialize();
  if (status != xnn_status_success) {
    std::cout << "xnn_deinitialize failed" << status << std::endl;
    return 1;
  }

  std::cout << "XNNPack test finished" << std::endl;
}
