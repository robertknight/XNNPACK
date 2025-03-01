// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/igemm.h>
#include <xnnpack/math.h>


void xnn_qc8_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qc8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc3x0123 = vacc0x0123;
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      do {
        v128_t vxa0 = wasm_i16x8_load8x8(a0);
        a0 += 8;
        v128_t vxa1 = wasm_i16x8_load8x8(a1);
        a1 += 8;
        v128_t vxa2 = wasm_i16x8_load8x8(a2);
        a2 += 8;
        v128_t vxa3 = wasm_i16x8_load8x8(a3);
        a3 += 8;

        const v128_t vxb0 = wasm_i16x8_load8x8(w);

        vacc0x0123 = wasm_i32x4_add(vacc0x0123, wasm_i32x4_dot_i16x8(vxa0, vxb0));
        vxa0 = wasm_v32x4_shuffle(vxa0, vxa0, 1, 2, 3, 4);
        vacc1x0123 = wasm_i32x4_add(vacc1x0123, wasm_i32x4_dot_i16x8(vxa1, vxb0));
        vxa1 = wasm_v32x4_shuffle(vxa1, vxa1, 1, 2, 3, 4);
        vacc2x0123 = wasm_i32x4_add(vacc2x0123, wasm_i32x4_dot_i16x8(vxa2, vxb0));
        vxa2 = wasm_v32x4_shuffle(vxa2, vxa2, 1, 2, 3, 4);
        vacc3x0123 = wasm_i32x4_add(vacc3x0123, wasm_i32x4_dot_i16x8(vxa3, vxb0));
        vxa3 = wasm_v32x4_shuffle(vxa3, vxa3, 1, 2, 3, 4);
        const v128_t vxb1 = wasm_i16x8_load8x8((const int8_t*) w + 8);

        vacc0x0123 = wasm_i32x4_add(vacc0x0123, wasm_i32x4_dot_i16x8(vxa0, vxb1));
        vxa0 = wasm_v32x4_shuffle(vxa0, vxa0, 1, 2, 3, 4);
        vacc1x0123 = wasm_i32x4_add(vacc1x0123, wasm_i32x4_dot_i16x8(vxa1, vxb1));
        vxa1 = wasm_v32x4_shuffle(vxa1, vxa1, 1, 2, 3, 4);
        vacc2x0123 = wasm_i32x4_add(vacc2x0123, wasm_i32x4_dot_i16x8(vxa2, vxb1));
        vxa2 = wasm_v32x4_shuffle(vxa2, vxa2, 1, 2, 3, 4);
        vacc3x0123 = wasm_i32x4_add(vacc3x0123, wasm_i32x4_dot_i16x8(vxa3, vxb1));
        vxa3 = wasm_v32x4_shuffle(vxa3, vxa3, 1, 2, 3, 4);
        const v128_t vxb2 = wasm_i16x8_load8x8((const int8_t*) w + 16);

        vacc0x0123 = wasm_i32x4_add(vacc0x0123, wasm_i32x4_dot_i16x8(vxa0, vxb2));
        vxa0 = wasm_v32x4_shuffle(vxa0, vxa0, 1, 2, 3, 4);
        vacc1x0123 = wasm_i32x4_add(vacc1x0123, wasm_i32x4_dot_i16x8(vxa1, vxb2));
        vxa1 = wasm_v32x4_shuffle(vxa1, vxa1, 1, 2, 3, 4);
        vacc2x0123 = wasm_i32x4_add(vacc2x0123, wasm_i32x4_dot_i16x8(vxa2, vxb2));
        vxa2 = wasm_v32x4_shuffle(vxa2, vxa2, 1, 2, 3, 4);
        vacc3x0123 = wasm_i32x4_add(vacc3x0123, wasm_i32x4_dot_i16x8(vxa3, vxb2));
        vxa3 = wasm_v32x4_shuffle(vxa3, vxa3, 1, 2, 3, 4);
        const v128_t vxb3 = wasm_i16x8_load8x8((const int8_t*) w + 24);

        vacc0x0123 = wasm_i32x4_add(vacc0x0123, wasm_i32x4_dot_i16x8(vxa0, vxb3));
        vacc1x0123 = wasm_i32x4_add(vacc1x0123, wasm_i32x4_dot_i16x8(vxa1, vxb3));
        vacc2x0123 = wasm_i32x4_add(vacc2x0123, wasm_i32x4_dot_i16x8(vxa2, vxb3));
        vacc3x0123 = wasm_i32x4_add(vacc3x0123, wasm_i32x4_dot_i16x8(vxa3, vxb3));

        w = (const int8_t*) w + 32;
        k -= 8 * sizeof(int8_t);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);
    vacc1x0123 = wasm_f32x4_convert_i32x4(vacc1x0123);
    vacc2x0123 = wasm_f32x4_convert_i32x4(vacc2x0123);
    vacc3x0123 = wasm_f32x4_convert_i32x4(vacc3x0123);

    const v128_t vscale0123 = wasm_v128_load(w);
    w = (const void*) ((const float*) w + 4);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vscale0123);
    vacc2x0123 = wasm_f32x4_mul(vacc2x0123, vscale0123);
    vacc3x0123 = wasm_f32x4_mul(vacc3x0123, vscale0123);

    const v128_t vmagic_bias = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias);
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vmagic_bias);
    vacc1x0123 = wasm_f32x4_add(vacc1x0123, vmagic_bias);
    vacc2x0123 = wasm_f32x4_add(vacc2x0123, vmagic_bias);
    vacc3x0123 = wasm_f32x4_add(vacc3x0123, vmagic_bias);

    const v128_t vmagic_min = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_min);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vmagic_min);
    vacc1x0123 = wasm_i32x4_max(vacc1x0123, vmagic_min);
    vacc2x0123 = wasm_i32x4_max(vacc2x0123, vmagic_min);
    vacc3x0123 = wasm_i32x4_max(vacc3x0123, vmagic_min);

    const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias_less_output_zero_point);
    vacc0x0123 = wasm_i32x4_sub(vacc0x0123, vmagic_bias_less_output_zero_point);
    vacc1x0123 = wasm_i32x4_sub(vacc1x0123, vmagic_bias_less_output_zero_point);
    vacc2x0123 = wasm_i32x4_sub(vacc2x0123, vmagic_bias_less_output_zero_point);
    vacc3x0123 = wasm_i32x4_sub(vacc3x0123, vmagic_bias_less_output_zero_point);

    v128_t vacc01x0123 = wasm_i16x8_narrow_i32x4(vacc0x0123, vacc1x0123);
    v128_t vacc23x0123 = wasm_i16x8_narrow_i32x4(vacc2x0123, vacc3x0123);

    v128_t vout = wasm_i8x16_narrow_i16x8(vacc01x0123, vacc23x0123);

    const v128_t voutput_max = wasm_v128_load64_splat(params->fp32_wasmsimd.output_max);
    vout = wasm_i8x16_min(vout, voutput_max);

    if (nc >= 4) {
      *((float*) c3) = (float) wasm_f32x4_extract_lane(vout, 3);
      *((float*) c2) = (float) wasm_f32x4_extract_lane(vout, 2);
      *((float*) c1) = (float) wasm_f32x4_extract_lane(vout, 1);
      *((float*) c0) = (float) wasm_f32x4_extract_lane(vout, 0);

      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      uint32_t vout3 = wasm_i32x4_extract_lane(vout, 3);
      uint32_t vout2 = wasm_i32x4_extract_lane(vout, 2);
      uint32_t vout1 = wasm_i32x4_extract_lane(vout, 1);
      uint32_t vout0 = wasm_i32x4_extract_lane(vout, 0);
      if (nc & 2) {
        *((uint16_t*) c3) = (uint16_t) vout3;
        vout3 >>= 16;
        c3 += 2;
        *((uint16_t*) c2) = (uint16_t) vout2;
        vout2 >>= 16;
        c2 += 2;
        *((uint16_t*) c1) = (uint16_t) vout1;
        vout1 >>= 16;
        c1 += 2;
        *((uint16_t*) c0) = (uint16_t) vout0;
        vout0 >>= 16;
        c0 += 2;
      }
      if (nc & 1) {
        *c3 = (int8_t) vout3;
        *c2 = (int8_t) vout2;
        *c1 = (int8_t) vout1;
        *c0 = (int8_t) vout0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
