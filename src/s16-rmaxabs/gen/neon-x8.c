// Auto-generated file. Do not edit!
//   Template: src/s16-rmaxabs/neon.c.in
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
#include <xnnpack/rmaxabs.h>


void xnn_s16_rmaxabs_ukernel__neon_x8(
    size_t batch,
    const int16_t* input,
    uint16_t* output) {

  assert(batch > 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16x8_t vzero = vdupq_n_u16(0);
  uint16x8_t vmax0 = vzero;


  // Remainder of full vectors
  for (; batch >= 8; batch -= 8) {
    const int16x8_t vi = vld1q_s16(input); input += 8;
    const uint16x8_t vabs = vreinterpretq_u16_s16(vabsq_s16(vi));
    vmax0 = vmaxq_u16(vmax0, vabs);
  }

  // Remainder
  if (batch != 0) {
    do {
      const int16x8_t vi = vld1q_dup_s16(input); input += 1;
      const uint16x8_t vabs = vreinterpretq_u16_s16(vabsq_s16(vi));
      vmax0 = vmaxq_u16(vmax0, vabs);
    } while (--batch != 0);
  }

  #if XNN_ARCH_ARM64
    *output = vmaxvq_u16(vmax0);
  #else
    uint16x4_t vmax_lo = vmax_u16(vget_low_u16(vmax0), vget_high_u16(vmax0));
    vmax_lo = vpmax_u16(vmax_lo, vmax_lo);
    vmax_lo = vpmax_u16(vmax_lo, vmax_lo);
    vst1_lane_u16(output, vmax_lo, 0);
  #endif
}
