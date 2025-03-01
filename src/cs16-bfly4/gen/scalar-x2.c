// Auto-generated file. Do not edit!
//   Template: src/cs16-bfly4/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack/math.h>
#include <xnnpack/fft.h>


void xnn_cs16_bfly4_ukernel__scalar_x2(
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
  assert(data != NULL);
  assert(stride != 0);
  assert(twiddle != NULL);

  for (; samples >= 2; samples -= 2) {
    int32_t vout0r0 = (int32_t) out0[0];
    int32_t vout0i0 = (int32_t) out0[1];
    int32_t vout0r1 = (int32_t) out0[2];
    int32_t vout0i1 = (int32_t) out0[3];
    int32_t vout1r0 = (int32_t) out1[0];
    int32_t vout1i0 = (int32_t) out1[1];
    int32_t vout1r1 = (int32_t) out1[2];
    int32_t vout1i1 = (int32_t) out1[3];
    int32_t vout2r0 = (int32_t) out2[0];
    int32_t vout2i0 = (int32_t) out2[1];
    int32_t vout2r1 = (int32_t) out2[2];
    int32_t vout2i1 = (int32_t) out2[3];
    int32_t vout3r0 = (int32_t) out3[0];
    int32_t vout3i0 = (int32_t) out3[1];
    int32_t vout3r1 = (int32_t) out3[2];
    int32_t vout3i1 = (int32_t) out3[3];

    const int32_t vtw1r0 = (const int32_t) tw1[0];
    const int32_t vtw1i0 = (const int32_t) tw1[1];
    tw1 += stride * 2;
    const int32_t vtw1r1 = (const int32_t) tw1[0];
    const int32_t vtw1i1 = (const int32_t) tw1[1];
    tw1 += stride * 2;
    const int32_t vtw2r0 = (const int32_t) tw2[0];
    const int32_t vtw2i0 = (const int32_t) tw2[1];
    tw2 += stride * 4;
    const int32_t vtw2r1 = (const int32_t) tw2[0];
    const int32_t vtw2i1 = (const int32_t) tw2[1];
    tw2 += stride * 4;
    const int32_t vtw3r0 = (const int32_t) tw3[0];
    const int32_t vtw3i0 = (const int32_t) tw3[1];
    tw3 += stride * 6;
    const int32_t vtw3r1 = (const int32_t) tw3[0];
    const int32_t vtw3i1 = (const int32_t) tw3[1];
    tw3 += stride * 6;

    // Note 32767 / 4 = 8191.  Should be 8192.
    vout0r0 = math_asr_s32(vout0r0 * 8191 + 16384, 15);
    vout0r1 = math_asr_s32(vout0r1 * 8191 + 16384, 15);
    vout0i0 = math_asr_s32(vout0i0 * 8191 + 16384, 15);
    vout0i1 = math_asr_s32(vout0i1 * 8191 + 16384, 15);
    vout1r0 = math_asr_s32(vout1r0 * 8191 + 16384, 15);
    vout1r1 = math_asr_s32(vout1r1 * 8191 + 16384, 15);
    vout1i0 = math_asr_s32(vout1i0 * 8191 + 16384, 15);
    vout1i1 = math_asr_s32(vout1i1 * 8191 + 16384, 15);
    vout2r0 = math_asr_s32(vout2r0 * 8191 + 16384, 15);
    vout2r1 = math_asr_s32(vout2r1 * 8191 + 16384, 15);
    vout2i0 = math_asr_s32(vout2i0 * 8191 + 16384, 15);
    vout2i1 = math_asr_s32(vout2i1 * 8191 + 16384, 15);
    vout3r0 = math_asr_s32(vout3r0 * 8191 + 16384, 15);
    vout3r1 = math_asr_s32(vout3r1 * 8191 + 16384, 15);
    vout3i0 = math_asr_s32(vout3i0 * 8191 + 16384, 15);
    vout3i1 = math_asr_s32(vout3i1 * 8191 + 16384, 15);

    const int32_t vtmp0r0 = math_asr_s32(vout1r0 * vtw1r0 - vout1i0 * vtw1i0 + 16384, 15);
    const int32_t vtmp0r1 = math_asr_s32(vout1r1 * vtw1r1 - vout1i1 * vtw1i1 + 16384, 15);
    const int32_t vtmp0i0 = math_asr_s32(vout1r0 * vtw1i0 + vout1i0 * vtw1r0 + 16384, 15);
    const int32_t vtmp0i1 = math_asr_s32(vout1r1 * vtw1i1 + vout1i1 * vtw1r1 + 16384, 15);
    const int32_t vtmp1r0 = math_asr_s32(vout2r0 * vtw2r0 - vout2i0 * vtw2i0 + 16384, 15);
    const int32_t vtmp1r1 = math_asr_s32(vout2r1 * vtw2r1 - vout2i1 * vtw2i1 + 16384, 15);
    const int32_t vtmp1i0 = math_asr_s32(vout2r0 * vtw2i0 + vout2i0 * vtw2r0 + 16384, 15);
    const int32_t vtmp1i1 = math_asr_s32(vout2r1 * vtw2i1 + vout2i1 * vtw2r1 + 16384, 15);
    const int32_t vtmp2r0 = math_asr_s32(vout3r0 * vtw3r0 - vout3i0 * vtw3i0 + 16384, 15);
    const int32_t vtmp2r1 = math_asr_s32(vout3r1 * vtw3r1 - vout3i1 * vtw3i1 + 16384, 15);
    const int32_t vtmp2i0 = math_asr_s32(vout3r0 * vtw3i0 + vout3i0 * vtw3r0 + 16384, 15);
    const int32_t vtmp2i1 = math_asr_s32(vout3r1 * vtw3i1 + vout3i1 * vtw3r1 + 16384, 15);

    const int32_t vtmp5r0 = vout0r0 - vtmp1r0;
    const int32_t vtmp5r1 = vout0r1 - vtmp1r1;
    const int32_t vtmp5i0 = vout0i0 - vtmp1i0;
    const int32_t vtmp5i1 = vout0i1 - vtmp1i1;
    vout0r0  += vtmp1r0;
    vout0r1  += vtmp1r1;
    vout0i0  += vtmp1i0;
    vout0i1  += vtmp1i1;
    const int32_t vtmp3r0 = vtmp0r0 + vtmp2r0;
    const int32_t vtmp3r1 = vtmp0r1 + vtmp2r1;
    const int32_t vtmp3i0 = vtmp0i0 + vtmp2i0;
    const int32_t vtmp3i1 = vtmp0i1 + vtmp2i1;
    const int32_t vtmp4r0 = vtmp0r0 - vtmp2r0;
    const int32_t vtmp4r1 = vtmp0r1 - vtmp2r1;
    const int32_t vtmp4i0 = vtmp0i0 - vtmp2i0;
    const int32_t vtmp4i1 = vtmp0i1 - vtmp2i1;
    vout2r0 = vout0r0 - vtmp3r0;
    vout2r1 = vout0r1 - vtmp3r1;
    vout2i0 = vout0i0 - vtmp3i0;
    vout2i1 = vout0i1 - vtmp3i1;
    vout0r0 += vtmp3r0;
    vout0r1 += vtmp3r1;
    vout0i0 += vtmp3i0;
    vout0i1 += vtmp3i1;
    vout1r0 = vtmp5r0 + vtmp4i0;
    vout1r1 = vtmp5r1 + vtmp4i1;
    vout1i0 = vtmp5i0 - vtmp4r0;
    vout1i1 = vtmp5i1 - vtmp4r1;
    vout3r0 = vtmp5r0 - vtmp4i0;
    vout3r1 = vtmp5r1 - vtmp4i1;
    vout3i0 = vtmp5i0 + vtmp4r0;
    vout3i1 = vtmp5i1 + vtmp4r1;

    out0[0] = (int16_t) vout0r0;
    out0[1] = (int16_t) vout0i0;
    out0[2] = (int16_t) vout0r1;
    out0[3] = (int16_t) vout0i1;
    out0 += 2 * 2;
    out1[0] = (int16_t) vout1r0;
    out1[1] = (int16_t) vout1i0;
    out1[2] = (int16_t) vout1r1;
    out1[3] = (int16_t) vout1i1;
    out1 += 2 * 2;
    out2[0] = (int16_t) vout2r0;
    out2[1] = (int16_t) vout2i0;
    out2[2] = (int16_t) vout2r1;
    out2[3] = (int16_t) vout2i1;
    out2 += 2 * 2;
    out3[0] = (int16_t) vout3r0;
    out3[1] = (int16_t) vout3i0;
    out3[2] = (int16_t) vout3r1;
    out3[3] = (int16_t) vout3i1;
    out3 += 2 * 2;
  }

  if XNN_UNLIKELY(samples != 0) {
    do {
      int32_t vout0r = (int32_t) out0[0];
      int32_t vout0i = (int32_t) out0[1];
      int32_t vout1r = (int32_t) out1[0];
      int32_t vout1i = (int32_t) out1[1];
      int32_t vout2r = (int32_t) out2[0];
      int32_t vout2i = (int32_t) out2[1];
      int32_t vout3r = (int32_t) out3[0];
      int32_t vout3i = (int32_t) out3[1];

      const int32_t vtw1r = (const int32_t) tw1[0];
      const int32_t vtw1i = (const int32_t) tw1[1];
      const int32_t vtw2r = (const int32_t) tw2[0];
      const int32_t vtw2i = (const int32_t) tw2[1];
      const int32_t vtw3r = (const int32_t) tw3[0];
      const int32_t vtw3i = (const int32_t) tw3[1];
      tw1 += stride * 2;
      tw2 += stride * 4;
      tw3 += stride * 6;

      // Note 32767 / 4 = 8191.  Should be 8192.
      vout0r = math_asr_s32(vout0r * 8191 + 16384, 15);
      vout0i = math_asr_s32(vout0i * 8191 + 16384, 15);
      vout1r = math_asr_s32(vout1r * 8191 + 16384, 15);
      vout1i = math_asr_s32(vout1i * 8191 + 16384, 15);
      vout2r = math_asr_s32(vout2r * 8191 + 16384, 15);
      vout2i = math_asr_s32(vout2i * 8191 + 16384, 15);
      vout3r = math_asr_s32(vout3r * 8191 + 16384, 15);
      vout3i = math_asr_s32(vout3i * 8191 + 16384, 15);

      const int32_t vtmp0r = math_asr_s32(vout1r * vtw1r - vout1i * vtw1i + 16384, 15);
      const int32_t vtmp0i = math_asr_s32(vout1r * vtw1i + vout1i * vtw1r + 16384, 15);
      const int32_t vtmp1r = math_asr_s32(vout2r * vtw2r - vout2i * vtw2i + 16384, 15);
      const int32_t vtmp1i = math_asr_s32(vout2r * vtw2i + vout2i * vtw2r + 16384, 15);
      const int32_t vtmp2r = math_asr_s32(vout3r * vtw3r - vout3i * vtw3i + 16384, 15);
      const int32_t vtmp2i = math_asr_s32(vout3r * vtw3i + vout3i * vtw3r + 16384, 15);

      const int32_t vtmp5r = vout0r - vtmp1r;
      const int32_t vtmp5i = vout0i - vtmp1i;
      vout0r  += vtmp1r;
      vout0i  += vtmp1i;
      const int32_t vtmp3r = vtmp0r + vtmp2r;
      const int32_t vtmp3i = vtmp0i + vtmp2i;
      const int32_t vtmp4r = vtmp0r - vtmp2r;
      const int32_t vtmp4i = vtmp0i - vtmp2i;
      vout2r = vout0r - vtmp3r;
      vout2i = vout0i - vtmp3i;

      vout0r += vtmp3r;
      vout0i += vtmp3i;

      vout1r = vtmp5r + vtmp4i;
      vout1i = vtmp5i - vtmp4r;
      vout3r = vtmp5r - vtmp4i;
      vout3i = vtmp5i + vtmp4r;

      out0[0] = (int16_t) vout0r;
      out0[1] = (int16_t) vout0i;
      out1[0] = (int16_t) vout1r;
      out1[1] = (int16_t) vout1i;
      out2[0] = (int16_t) vout2r;
      out2[1] = (int16_t) vout2i;
      out3[0] = (int16_t) vout3r;
      out3[1] = (int16_t) vout3i;
      out0 += 2;
      out1 += 2;
      out2 += 2;
      out3 += 2;
    } while(--samples != 0);
  }
}
