// Auto-generated file. Do not edit!
//   Template: src/cs16-fftr/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found il the
// LICENSE file il the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack/math.h>
#include <xnnpack/fft.h>


void xnn_cs16_fftr_ukernel__scalar_x3(
    size_t samples,
    const int16_t* input,
    int16_t* output,
    const int16_t* twiddle) {

  assert(samples >= 2);
  assert(samples % 2 == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(twiddle != NULL);

  const int16_t* il = input;
  const int16_t* ir = input + samples * 2;
  int32_t vdcr = (int32_t) il[0];
  int32_t vdci = (int32_t) il[1];
  il += 2;
  vdcr = math_asr_s32(vdcr * 16383 + 16384, 15);
  vdci = math_asr_s32(vdci * 16383 + 16384, 15);

  int16_t* ol  = output;
  int16_t* or = output + samples * 2;
  ol[0] = vdcr + vdci;
  ol[1] = 0;
  ol += 2;
  or[0] = vdcr - vdci;
  or[1] = 0;

  samples >>= 1;

  for (; samples >= 3; samples -= 3) {
    int32_t vilr0 = il[0];
    int32_t vili0 = il[1];
    int32_t vilr1 = il[2];
    int32_t vili1 = il[3];
    int32_t vilr2 = il[4];
    int32_t vili2 = il[5];
    il += 3 * 2;
    ir -= 3 * 2;
    int32_t virr0 =  (int32_t) ir[4];
    int32_t viri0 = -(int32_t) ir[5];
    int32_t virr1 =  (int32_t) ir[2];
    int32_t viri1 = -(int32_t) ir[3];
    int32_t virr2 =  (int32_t) ir[0];
    int32_t viri2 = -(int32_t) ir[1];
    const int32_t vtwr0 = twiddle[0];
    const int32_t vtwi0 = twiddle[1];
    const int32_t vtwr1 = twiddle[2];
    const int32_t vtwi1 = twiddle[3];
    const int32_t vtwr2 = twiddle[4];
    const int32_t vtwi2 = twiddle[5];
    twiddle += 3 * 2;

    vilr0 = math_asr_s32(vilr0 * 16383 + 16384, 15);
    virr0 = math_asr_s32(virr0 * 16383 + 16384, 15);
    vilr1 = math_asr_s32(vilr1 * 16383 + 16384, 15);
    virr1 = math_asr_s32(virr1 * 16383 + 16384, 15);
    vilr2 = math_asr_s32(vilr2 * 16383 + 16384, 15);
    virr2 = math_asr_s32(virr2 * 16383 + 16384, 15);
    vili0 = math_asr_s32(vili0 * 16383 + 16384, 15);
    viri0 = math_asr_s32(viri0 * 16383 + 16384, 15);
    vili1 = math_asr_s32(vili1 * 16383 + 16384, 15);
    viri1 = math_asr_s32(viri1 * 16383 + 16384, 15);
    vili2 = math_asr_s32(vili2 * 16383 + 16384, 15);
    viri2 = math_asr_s32(viri2 * 16383 + 16384, 15);
    const int32_t vacc1r0 = vilr0 + virr0;
    const int32_t vacc2r0 = vilr0 - virr0;
    const int32_t vacc1r1 = vilr1 + virr1;
    const int32_t vacc2r1 = vilr1 - virr1;
    const int32_t vacc1r2 = vilr2 + virr2;
    const int32_t vacc2r2 = vilr2 - virr2;
    const int32_t vacc1i0 = vili0 + viri0;
    const int32_t vacc2i0 = vili0 - viri0;
    const int32_t vacc1i1 = vili1 + viri1;
    const int32_t vacc2i1 = vili1 - viri1;
    const int32_t vacc1i2 = vili2 + viri2;
    const int32_t vacc2i2 = vili2 - viri2;

    const int32_t twr0 = math_asr_s32(vacc2r0 * vtwr0 - vacc2i0 * vtwi0 + 16384, 15);
    const int32_t twr1 = math_asr_s32(vacc2r1 * vtwr1 - vacc2i1 * vtwi1 + 16384, 15);
    const int32_t twr2 = math_asr_s32(vacc2r2 * vtwr2 - vacc2i2 * vtwi2 + 16384, 15);
    const int32_t twi0 = math_asr_s32(vacc2r0 * vtwi0 + vacc2i0 * vtwr0 + 16384, 15);
    const int32_t twi1 = math_asr_s32(vacc2r1 * vtwi1 + vacc2i1 * vtwr1 + 16384, 15);
    const int32_t twi2 = math_asr_s32(vacc2r2 * vtwi2 + vacc2i2 * vtwr2 + 16384, 15);

    ol[0] = math_asr_s32(vacc1r0 + twr0, 1);
    ol[1] = math_asr_s32(vacc1i0 + twi0, 1);
    ol[2] = math_asr_s32(vacc1r1 + twr1, 1);
    ol[3] = math_asr_s32(vacc1i1 + twi1, 1);
    ol[4] = math_asr_s32(vacc1r2 + twr2, 1);
    ol[5] = math_asr_s32(vacc1i2 + twi2, 1);
    ol += 3 * 2;
    or -= 3 * 2;
    or[4] = math_asr_s32(vacc1r0 - twr0, 1);
    or[5] = math_asr_s32(twi0 - vacc1i0, 1);
    or[2] = math_asr_s32(vacc1r1 - twr1, 1);
    or[3] = math_asr_s32(twi1 - vacc1i1, 1);
    or[0] = math_asr_s32(vacc1r2 - twr2, 1);
    or[1] = math_asr_s32(twi2 - vacc1i2, 1);
  }

  if XNN_UNLIKELY(samples != 0) {
    do {
      int32_t vilr = il[0];
      int32_t vili = il[1];
      il += 2;
      ir -= 2;
      int32_t virr =  (int32_t) ir[0];
      int32_t viri = -(int32_t) ir[1];
      const int32_t vtwr = twiddle[0];
      const int32_t vtwi = twiddle[1];
      twiddle += 2;

      vilr =  math_asr_s32(vilr * 16383 + 16384, 15);
      vili =  math_asr_s32(vili * 16383 + 16384, 15);
      virr = math_asr_s32(virr * 16383 + 16384, 15);
      viri = math_asr_s32(viri * 16383 + 16384, 15);
      const int32_t vacc1r = vilr + virr;
      const int32_t vacc1i = vili + viri;
      const int32_t vacc2r = vilr - virr;
      const int32_t vacc2i = vili - viri;

      const int32_t twr = math_asr_s32(vacc2r * vtwr - vacc2i * vtwi + 16384, 15);
      const int32_t twi = math_asr_s32(vacc2r * vtwi + vacc2i * vtwr + 16384, 15);

      ol[0] = math_asr_s32(vacc1r + twr, 1);
      ol[1] = math_asr_s32(vacc1i + twi, 1);
      ol += 2;
      or -= 2;
      or[0] = math_asr_s32(vacc1r - twr, 1);
      or[1] = math_asr_s32(twi - vacc1i, 1);

    } while (--samples != 0);
  }
}
