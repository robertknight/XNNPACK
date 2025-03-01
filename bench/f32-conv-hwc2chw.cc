// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <cpuinfo.h>

#include <benchmark/benchmark.h>
#include "bench/dconv.h"
#include "bench/utils.h"

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/conv.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/pack.h>


static void DConvHWC2CHW3X3S2P1Benchmark(benchmark::State& state,
  xnn_f32_conv_hwc2chw_ukernel_function conv,
  uint32_t output_channels_tile,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }

  const size_t input_height = state.range(0);
  const size_t input_width = state.range(1);
  const size_t output_channels = state.range(2);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), std::ref(rng));

  const size_t input_channels = 3;
  const size_t kernel_size = 3;
  const size_t padding = 1;
  const size_t subsampling = 2;

  const size_t output_height = (input_height + 2 * padding - kernel_size) / subsampling + 1;
  const size_t output_width = (input_width + 2 * padding - kernel_size) / subsampling + 1;

  std::vector<float> input(input_height * input_width * input_channels + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::vector<float> kernel(output_channels * kernel_size * kernel_size * input_channels);
  std::generate(kernel.begin(), kernel.end(), std::ref(f32rng));
  std::vector<float> bias(output_channels);
  std::generate(bias.begin(), bias.end(), std::ref(f32rng));

  std::vector<float, AlignedAllocator<float, 64>> zero(input_channels * input_width + XNN_EXTRA_BYTES / sizeof(float));

  const size_t weights_elements = (kernel_size * kernel_size * input_channels + 1) *
    benchmark::utils::RoundUp<size_t>(output_channels, output_channels_tile);
  const size_t output_elements = output_height * output_width * output_channels;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (weights_elements + output_elements));

  std::vector<float, AlignedAllocator<float, 64>> packed_weights(weights_elements * num_buffers);
  std::fill(packed_weights.begin(), packed_weights.end(), 0.0f);
  xnn_pack_f32_dconv_oki_w(
    output_channels, input_channels, output_channels_tile,
    kernel_size /* kernel height */, kernel_size /* kernel width */,
    kernel.data(), bias.data(), packed_weights.data(), NULL);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(packed_weights.cbegin(),
      packed_weights.cbegin() + weights_elements,
      packed_weights.begin() + n * weights_elements);
  }

  std::vector<float> output(output_elements * num_buffers);
  std::fill(output.begin(), output.end(), std::nanf(""));

  xnn_f32_minmax_params params;
  xnn_init_f32_minmax_params(&params, -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(input.data(), input.size() * sizeof(float));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    conv(
      input_height, input_width,
      0 /* output_y_start */, output_height /* output_y_end */,
      input.data(), zero.data(),
      packed_weights.data() + buffer_index * weights_elements,
      output.data() + buffer_index * output_elements,
      padding, output_channels,
      output_channels * output_width * sizeof(float),
      output_channels * sizeof(float),
      &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 *
      output_height * output_width *
      input_channels * output_channels *
      kernel_size * kernel_size,
    benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM64
  static void f32_conv_hwc2chw_3x3s2p1c3x4__neonfma_2x2(benchmark::State& state, const char* net) {
    DConvHWC2CHW3X3S2P1Benchmark(state, xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__neonfma_2x2, 4, benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_DCONV(f32_conv_hwc2chw_3x3s2p1c3x4__neonfma_2x2);
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_conv_hwc2chw_3x3s2p1c3x4__sse_1x1(benchmark::State& state, const char* net) {
    DConvHWC2CHW3X3S2P1Benchmark(state, xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_1x1, 4);
  }

  BENCHMARK_DCONV(f32_conv_hwc2chw_3x3s2p1c3x4__sse_1x1);

  static void f32_conv_hwc2chw_3x3s2p1c3x4__sse_2x2(benchmark::State& state, const char* net) {
    DConvHWC2CHW3X3S2P1Benchmark(state, xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2, 4);
  }

  BENCHMARK_DCONV(f32_conv_hwc2chw_3x3s2p1c3x4__sse_2x2);
#endif

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_conv_hwc2chw_3x3s2p1c3x4__wasmsimd_2x2(benchmark::State& state, const char* net) {
    DConvHWC2CHW3X3S2P1Benchmark(state, xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2, 4);
  }

  BENCHMARK_DCONV(f32_conv_hwc2chw_3x3s2p1c3x4__wasmsimd_2x2);
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

static void f32_conv_hwc2chw_3x3s2p1c3x4__scalar_1x1(benchmark::State& state, const char* net) {
  DConvHWC2CHW3X3S2P1Benchmark(state, xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1, 4);
}

BENCHMARK_DCONV(f32_conv_hwc2chw_3x3s2p1c3x4__scalar_1x1);

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
