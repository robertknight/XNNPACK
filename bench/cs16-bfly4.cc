// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#include "bench/utils.h"
#include <benchmark/benchmark.h>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/fft.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>


void bfly4(
    benchmark::State& state,
    xnn_cs16_bfly4_ukernel_function bfly4,
    benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }
  const size_t fft_size = state.range(0);
  const size_t samples = state.range(1);
  const size_t stride = state.range(2);

  assert(fft_size == samples * stride * 4);  // 4 for bfly4.

  std::vector<int16_t, AlignedAllocator<int16_t, 64>> output(
      fft_size * 2 + XNN_EXTRA_BYTES / sizeof(int16_t));
  std::vector<int16_t, AlignedAllocator<int16_t, 64>> twiddle(
      fft_size * 2 + XNN_EXTRA_BYTES / sizeof(int16_t));

  std::iota(output.begin(), output.end(), 0);
  std::iota(twiddle.begin(), twiddle.end(), 0);

  for (auto _ : state) {
    bfly4(samples, output.data(), stride, twiddle.data());
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

static void BenchmarkKernelSize(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"fft_size", "samples", "stride"});
  b->Args({256, 64, 1});
  b->Args({256, 16, 4});
  b->Args({256, 4, 16});
  b->Args({256, 1, 64});
  b->Args({1024, 256, 1});
  b->Args({1024, 64,  4});
  b->Args({1024, 16, 16});
  b->Args({1024, 4,  64});
  b->Args({1024, 1, 256});
}

static void BenchmarkM1KernelSize(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"fft_size", "samples", "stride"});
  b->Args({256, 1, 64});
  b->Args({1024, 1, 256});
}

BENCHMARK_CAPTURE(bfly4, cs16_scalar_m1_x1, xnn_cs16_bfly4m1_ukernel__scalar_x1)->Apply(BenchmarkM1KernelSize)->UseRealTime();
BENCHMARK_CAPTURE(bfly4, cs16_scalar_x1, xnn_cs16_bfly4_ukernel__scalar_x1)->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(bfly4, cs16_scalar_x2, xnn_cs16_bfly4_ukernel__scalar_x2)->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(bfly4, cs16_scalar_x3, xnn_cs16_bfly4_ukernel__scalar_x3)->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(bfly4, cs16_scalar_x4, xnn_cs16_bfly4_ukernel__scalar_x4)->Apply(BenchmarkKernelSize)->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
