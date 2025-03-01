// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/math.h>
#include <xnnpack/params.h>

// twiddle table for bfly4 for fft size 256 (complex numbers)
// Even numbers are numpy.floor(0.5 + 32767 * numpy.cos(-2*pi*numpy.linspace(0, 255, num=256) / 256)).astype(numpy.int16).tolist()
// Odd numbers are numpy.floor(0.5 + 32767 * numpy.sin(-2*pi*numpy.linspace(0, 255, num=256) / 256)).astype(numpy.int16).tolist()

static const int16_t xnn_reference_table_fft256_twiddle[512] = {
   32767,     0,  32757,  -804,  32728, -1608,  32678, -2410,
   32609, -3212,  32521, -4011,  32412, -4808,  32285, -5602,
   32137, -6393,  31971, -7179,  31785, -7962,  31580, -8739,
   31356, -9512,  31113,-10278,  30852,-11039,  30571,-11793,
   30273,-12539,  29956,-13279,  29621,-14010,  29268,-14732,
   28898,-15446,  28510,-16151,  28105,-16846,  27683,-17530,
   27245,-18204,  26790,-18868,  26319,-19519,  25832,-20159,
   25329,-20787,  24811,-21403,  24279,-22005,  23731,-22594,
   23170,-23170,  22594,-23731,  22005,-24279,  21403,-24811,
   20787,-25329,  20159,-25832,  19519,-26319,  18868,-26790,
   18204,-27245,  17530,-27683,  16846,-28105,  16151,-28510,
   15446,-28898,  14732,-29268,  14010,-29621,  13279,-29956,
   12539,-30273,  11793,-30571,  11039,-30852,  10278,-31113,
    9512,-31356,   8739,-31580,   7962,-31785,   7179,-31971,
    6393,-32137,   5602,-32285,   4808,-32412,   4011,-32521,
    3212,-32609,   2410,-32678,   1608,-32728,    804,-32757,
       0,-32767,   -804,-32757,  -1608,-32728,  -2410,-32678,
   -3212,-32609,  -4011,-32521,  -4808,-32412,  -5602,-32285,
   -6393,-32137,  -7179,-31971,  -7962,-31785,  -8739,-31580,
   -9512,-31356, -10278,-31113, -11039,-30852, -11793,-30571,
  -12539,-30273, -13279,-29956, -14010,-29621, -14732,-29268,
  -15446,-28898, -16151,-28510, -16846,-28105, -17530,-27683,
  -18204,-27245, -18868,-26790, -19519,-26319, -20159,-25832,
  -20787,-25329, -21403,-24811, -22005,-24279, -22594,-23731,
  -23170,-23170, -23731,-22594, -24279,-22005, -24811,-21403,
  -25329,-20787, -25832,-20159, -26319,-19519, -26790,-18868,
  -27245,-18204, -27683,-17530, -28105,-16846, -28510,-16151,
  -28898,-15446, -29268,-14732, -29621,-14010, -29956,-13279,
  -30273,-12539, -30571,-11793, -30852,-11039, -31113,-10278,
  -31356, -9512, -31580, -8739, -31785, -7962, -31971, -7179,
  -32137, -6393, -32285, -5602, -32412, -4808, -32521, -4011,
  -32609, -3212, -32678, -2410, -32728, -1608, -32757,  -804,
  -32767,     0, -32757,   804, -32728,  1608, -32678,  2410,
  -32609,  3212, -32521,  4011, -32412,  4808, -32285,  5602,
  -32137,  6393, -31971,  7179, -31785,  7962, -31580,  8739,
  -31356,  9512, -31113, 10278, -30852, 11039, -30571, 11793,
  -30273, 12539, -29956, 13279, -29621, 14010, -29268, 14732,
  -28898, 15446, -28510, 16151, -28105, 16846, -27683, 17530,
  -27245, 18204, -26790, 18868, -26319, 19519, -25832, 20159,
  -25329, 20787, -24811, 21403, -24279, 22005, -23731, 22594,
  -23170, 23170, -22594, 23731, -22005, 24279, -21403, 24811,
  -20787, 25329, -20159, 25832, -19519, 26319, -18868, 26790,
  -18204, 27245, -17530, 27683, -16846, 28105, -16151, 28510,
  -15446, 28898, -14732, 29268, -14010, 29621, -13279, 29956,
  -12539, 30273, -11793, 30571, -11039, 30852, -10278, 31113,
   -9512, 31356,  -8739, 31580,  -7962, 31785,  -7179, 31971,
   -6393, 32137,  -5602, 32285,  -4808, 32412,  -4011, 32521,
   -3212, 32609,  -2410, 32678,  -1608, 32728,   -804, 32757,
       0, 32767,    804, 32757,   1608, 32728,   2410, 32678,
    3212, 32609,   4011, 32521,   4808, 32412,   5602, 32285,
    6393, 32137,   7179, 31971,   7962, 31785,   8739, 31580,
    9512, 31356,  10278, 31113,  11039, 30852,  11793, 30571,
   12539, 30273,  13279, 29956,  14010, 29621,  14732, 29268,
   15446, 28898,  16151, 28510,  16846, 28105,  17530, 27683,
   18204, 27245,  18868, 26790,  19519, 26319,  20159, 25832,
   20787, 25329,  21403, 24811,  22005, 24279,  22594, 23731,
   23170, 23170,  23731, 22594,  24279, 22005,  24811, 21403,
   25329, 20787,  25832, 20159,  26319, 19519,  26790, 18868,
   27245, 18204,  27683, 17530,  28105, 16846,  28510, 16151,
   28898, 15446,  29268, 14732,  29621, 14010,  29956, 13279,
   30273, 12539,  30571, 11793,  30852, 11039,  31113, 10278,
   31356,  9512,  31580,  8739,  31785,  7962,  31971,  7179,
   32137,  6393,  32285,  5602,  32412,  4808,  32521,  4011,
   32609,  3212,  32678,  2410,  32728,  1608,  32757,   804
};

void xnn_cs16_bfly4_reference(
    size_t samples,
    int16_t* data,
    const size_t stride,
    const int16_t* twiddle) {

  const int16_t* tw1 = twiddle;
  const int16_t* tw2 = tw1;
  const int16_t* tw3 = tw1;
  int16_t* out0 = data;
  int16_t* out1 = data + samples * 2;
  int16_t* out2 = data + samples * 4;
  int16_t* out3 = data + samples * 6;

  assert(samples != 0);
  assert(stride != 0);
  assert(twiddle != NULL);
  assert(data != NULL);

  do {
      int32_t vout0_r = (int32_t) out0[0];
      int32_t vout0_i = (int32_t) out0[1];
      int32_t vout1_r = (int32_t) out1[0];
      int32_t vout1_i = (int32_t) out1[1];
      int32_t vout2_r = (int32_t) out2[0];
      int32_t vout2_i = (int32_t) out2[1];
      int32_t vout3_r = (int32_t) out3[0];
      int32_t vout3_i = (int32_t) out3[1];

      const int32_t tw1_r = (const int32_t) tw1[0];
      const int32_t tw1_i = (const int32_t) tw1[1];
      const int32_t tw2_r = (const int32_t) tw2[0];
      const int32_t tw2_i = (const int32_t) tw2[1];
      const int32_t tw3_r = (const int32_t) tw3[0];
      const int32_t tw3_i = (const int32_t) tw3[1];

      // Note 32767 / 4 = 8191.  Should be 8192.
      vout0_r = (vout0_r * 8191 + 16384) >> 15;
      vout0_i = (vout0_i * 8191 + 16384) >> 15;
      vout1_r = (vout1_r * 8191 + 16384) >> 15;
      vout1_i = (vout1_i * 8191 + 16384) >> 15;
      vout2_r = (vout2_r * 8191 + 16384) >> 15;
      vout2_i = (vout2_i * 8191 + 16384) >> 15;
      vout3_r = (vout3_r * 8191 + 16384) >> 15;
      vout3_i = (vout3_i * 8191 + 16384) >> 15;

      const int32_t vtmp0_r = math_asr_s32(vout1_r * tw1_r - vout1_i * tw1_i + 16384, 15);
      const int32_t vtmp0_i = math_asr_s32(vout1_r * tw1_i + vout1_i * tw1_r + 16384, 15);
      const int32_t vtmp1_r = math_asr_s32(vout2_r * tw2_r - vout2_i * tw2_i + 16384, 15);
      const int32_t vtmp1_i = math_asr_s32(vout2_r * tw2_i + vout2_i * tw2_r + 16384, 15);
      const int32_t vtmp2_r = math_asr_s32(vout3_r * tw3_r - vout3_i * tw3_i + 16384, 15);
      const int32_t vtmp2_i = math_asr_s32(vout3_r * tw3_i + vout3_i * tw3_r + 16384, 15);

      const int32_t vtmp5_r = vout0_r - vtmp1_r;
      const int32_t vtmp5_i = vout0_i - vtmp1_i;
      vout0_r  += vtmp1_r;
      vout0_i  += vtmp1_i;
      const int32_t vtmp3_r = vtmp0_r + vtmp2_r;
      const int32_t vtmp3_i = vtmp0_i + vtmp2_i;
      const int32_t vtmp4_r = vtmp0_r - vtmp2_r;
      const int32_t vtmp4_i = vtmp0_i - vtmp2_i;
      vout2_r = vout0_r - vtmp3_r;
      vout2_i = vout0_i - vtmp3_i;

      tw1 += stride * 2;
      tw2 += stride * 4;
      tw3 += stride * 6;
      vout0_r += vtmp3_r;
      vout0_i += vtmp3_i;

      vout1_r = vtmp5_r + vtmp4_i;
      vout1_i = vtmp5_i - vtmp4_r;
      vout3_r = vtmp5_r - vtmp4_i;
      vout3_i = vtmp5_i + vtmp4_r;

      out0[0] = (int16_t) vout0_r;
      out0[1] = (int16_t) vout0_i;
      out1[0] = (int16_t) vout1_r;
      out1[1] = (int16_t) vout1_i;
      out2[0] = (int16_t) vout2_r;
      out2[1] = (int16_t) vout2_i;
      out3[0] = (int16_t) vout3_r;
      out3[1] = (int16_t) vout3_i;
      out0 += 2;
      out1 += 2;
      out2 += 2;
      out3 += 2;
  } while(--samples != 0);
}

class BFly4MicrokernelTester {
 public:
  inline BFly4MicrokernelTester& samples(size_t samples) {
    assert(samples != 0);
    this->samples_ = samples;
    return *this;
  }

  inline size_t samples() const {
    return this->samples_;
  }

  inline BFly4MicrokernelTester& stride(uint32_t stride) {
    this->stride_ = stride;
    return *this;
  }

  inline uint32_t stride() const {
    return this->stride_;
  }

  inline BFly4MicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_cs16_bfly4_ukernel_function bfly4) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto i16rng = std::bind(std::uniform_int_distribution<int16_t>(), std::ref(rng));
    const size_t fft_size = samples() * stride() * 4;  // 4 for bfly4.

    // 256 complex numbers = fft_size * 2 = 512
    std::vector<int16_t> y(fft_size * 2 + XNN_EXTRA_BYTES / sizeof(int16_t));
    std::vector<int16_t> y_ref(fft_size * 2 + XNN_EXTRA_BYTES / sizeof(int16_t));

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(y.begin(), y.end(), std::ref(i16rng));
      y_ref = y;

      // Compute reference results.
      xnn_cs16_bfly4_reference(samples(), y_ref.data(), stride(), xnn_reference_table_fft256_twiddle);

      // Call optimized micro-kernel.
      bfly4(samples(), y.data(), stride(), xnn_reference_table_fft256_twiddle);

      // Verify results.
      for (size_t n = 0; n < fft_size * 2; n++) {
        ASSERT_EQ(y[n], y_ref[n])
            << "at sample " << n << " / " << fft_size
            << "\nsamples " << samples()
            << "\nstride " << stride();
      }
    }
  }

 private:
  size_t samples_{1};
  uint32_t stride_{1};
  size_t iterations_{15};
};
