// Auto-generated file. Do not edit!
//   Template: src/s16-window/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include <xnnpack/math.h>
#include <xnnpack/window.h>


void xnn_s16_window_shift12_ukernel__neon_x8(
    size_t rows,
    size_t batch_size,
    const int16_t* input,
    const int16_t* weights,
    uint32_t shift,
    int16_t* output) {

  assert(rows != 0);
  assert(batch_size != 0);
  assert(input != NULL);
  assert(weights != NULL);
  assert(shift == 12);
  assert(output != NULL);


  do {
    const int16_t* w = weights;
    size_t n = batch_size * sizeof(int16_t);

    // Remainder of full vectors
    for (; n >= 8 * sizeof(int16_t); n -= 8 * sizeof(int16_t)) {
      const int16x8_t vi = vld1q_s16(input); input += 8;
      const int16x8_t vw = vld1q_s16(w); w += 8;
      int32x4_t vacc_lo = vmull_s16(vget_low_s16(vi), vget_low_s16(vw));
      int32x4_t vacc_hi = vmull_s16(vget_high_s16(vi), vget_high_s16(vw));
      const int16x4_t vshift_lo = vqshrn_n_s32(vacc_lo, 12);
      const int16x4_t vshift_hi = vqshrn_n_s32(vacc_hi, 12);
      const int16x8_t vout = vcombine_s16(vshift_lo, vshift_hi);
      vst1q_s16(output, vout); output += 8;
    }

    assert(n % 2 == 0);
    // Remainder of 1 to 7 batch_size
    if XNN_UNLIKELY(n != 0) {
      const int16x8_t vi = vld1q_s16(input); input = (const int16_t*) ((uintptr_t) input + n);
      const int16x8_t vw = vld1q_s16(w);
      int32x4_t vacc = vmull_s16(vget_low_s16(vi), vget_low_s16(vw));
      int16x4_t vout = vqshrn_n_s32(vacc, 12);
      if (n & (4 * sizeof(int16_t))) {
        vst1_s16(output, vout); output += 4;
        vacc = vmull_s16(vget_high_s16(vi), vget_high_s16(vw));
        vout = vqshrn_n_s32(vacc, 12);
      }
      if (n & (2 * sizeof(int16_t))) {
        vst1_lane_u32((void*) output, vreinterpret_u32_s16(vout), 0); output += 2;
        vout = vext_s16(vout, vout, 2);
      }
      if (n & (1 * sizeof(int16_t))) {
        vst1_lane_s16(output, vout, 0); output += 1;
      }
    }

  } while (--rows != 0);
}
