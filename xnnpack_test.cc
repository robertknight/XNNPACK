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

  xnn_subgraph_t subgraph;
  status = xnn_create_subgraph(2 /* external_value_ids */, 0 /* flags */, &subgraph);
  if (!check_status(status, "xnn_create_subgraph")) {
    return 1;
  }

  uint32_t input_id = 0;
  uint32_t output_id = 1;

  uint32_t unused;

  // Define input as NHWC
  size_t in_dims[4] = {1, 2, 2, 1};
  status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 4 /* dims */, in_dims, nullptr /* data */, input_id, XNN_VALUE_FLAG_EXTERNAL_INPUT /* flags */, &unused /* id_out */);

  // Define output as NHWC
  size_t out_dims[4] = {1, 2, 2, 1};
  status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 4 /* dims */, out_dims, nullptr /* data */, output_id, XNN_VALUE_FLAG_EXTERNAL_OUTPUT /* flags */, &unused /* id_out */);

  status = xnn_define_max_pooling_2d(
    subgraph,
    // Padding top, right, bottom, left
    0, 0, 0, 0,
    // Pooling height, width
    2, 2,
    // Stride height, width
    2, 2,
    // Dilation height, width
    1, 1,
    0, // Output min
    5, // Outut max
    input_id,
    output_id,
    0 // Flags
    );

  if (!check_status(status, "xnn_define_max_pooling_2d")) {
    return 1;
  }

  xnn_runtime_t runtime;
  status = xnn_create_runtime(subgraph, &runtime);
  if (!check_status(status, "xnn_create_runtime")) {
    return 1;
  }

  float input[4] = {3.f, 2.f, 5.f, 0.f};
  float output[4] = {0.f, 0.f, 0.f, 0.f};

  xnn_external_value in_val = { .id = input_id, .data = input };
  xnn_external_value out_val = { .id = output_id, .data = output };
  xnn_external_value external_values[2] = {in_val, out_val};
  status = xnn_setup_runtime(runtime, 2 /* num_external_values */, external_values);
  if (!check_status(status, "xnn_setup_runtime")) {
    return 1;
  }

  status = xnn_invoke_runtime(runtime);
  if (!check_status(status, "xnn_invoke_runtime")) {
    return 1;
  }

  for (int i=0; i < 4; i++) {
    std::cout << "graph output " << i << ": " << static_cast<int>(output[i]) << std::endl;
  }

  // TODO: Cleanup

  std::cout << "XNNPack test finished" << std::endl;
}
