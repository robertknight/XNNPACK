// Auto-generated file. Do not edit!
//   Template: src/qs8-vlrelu/wasmsimd-x86.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/common.h>
#include <xnnpack/vcvt.h>


void xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_x86_x32(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const union xnn_qu8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(uint8_t) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const v128_t vinput_zero_point = wasm_v128_load64_splat(params->wasmsimd_x86.input_zero_point);
  const v128_t vmultiplier_diff = wasm_v128_load64_splat(params->wasmsimd_x86.multiplier_diff);
  const v128_t vmultiplier_base = wasm_v128_load64_splat(params->wasmsimd_x86.multiplier_base);
  const v128_t voutput_zero_point = wasm_v128_load64_splat(params->wasmsimd_x86.output_zero_point);
  for (; n >= 32 * sizeof(uint8_t); n -= 32 * sizeof(uint8_t)) {
    v128_t vacc0 = wasm_u16x8_load8x8(x);
    v128_t vacc1 = wasm_u16x8_load8x8(x + 8);
    v128_t vacc2 = wasm_u16x8_load8x8(x + 16);
    v128_t vacc3 = wasm_u16x8_load8x8(x + 24);
    x += 32;

    v128_t vmultiplier0 = wasm_i16x8_gt(vacc0, vinput_zero_point);
    vacc0 = wasm_i16x8_sub(vinput_zero_point, vacc0);
    v128_t vmultiplier1 = wasm_i16x8_gt(vacc1, vinput_zero_point);
    vacc1 = wasm_i16x8_sub(vinput_zero_point, vacc1);
    v128_t vmultiplier2 = wasm_i16x8_gt(vacc2, vinput_zero_point);
    vacc2 = wasm_i16x8_sub(vinput_zero_point, vacc2);
    v128_t vmultiplier3 = wasm_i16x8_gt(vacc3, vinput_zero_point);
    vacc3 = wasm_i16x8_sub(vinput_zero_point, vacc3);

    vmultiplier0 = wasm_v128_and(vmultiplier0, vmultiplier_diff);
    vacc0 = wasm_i16x8_shl(vacc0, 7);
    vmultiplier0 = wasm_v128_xor(vmultiplier0, vmultiplier_base);
    vmultiplier1 = wasm_v128_and(vmultiplier1, vmultiplier_diff);
    vacc1 = wasm_i16x8_shl(vacc1, 7);
    vmultiplier1 = wasm_v128_xor(vmultiplier1, vmultiplier_base);
    vmultiplier2 = wasm_v128_and(vmultiplier2, vmultiplier_diff);
    vacc2 = wasm_i16x8_shl(vacc2, 7);
    vmultiplier2 = wasm_v128_xor(vmultiplier2, vmultiplier_base);
    vmultiplier3 = wasm_v128_and(vmultiplier3, vmultiplier_diff);
    vacc3 = wasm_i16x8_shl(vacc3, 7);
    vmultiplier3 = wasm_v128_xor(vmultiplier3, vmultiplier_base);

    vacc0 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc0, vmultiplier0);
    vacc1 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc1, vmultiplier1);
    vacc2 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc2, vmultiplier2);
    vacc3 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc3, vmultiplier3);

    vacc0 = wasm_i16x8_add_sat(vacc0, voutput_zero_point);
    vacc1 = wasm_i16x8_add_sat(vacc1, voutput_zero_point);
    vacc2 = wasm_i16x8_add_sat(vacc2, voutput_zero_point);
    vacc3 = wasm_i16x8_add_sat(vacc3, voutput_zero_point);

    const v128_t vy0 = wasm_u8x16_narrow_i16x8(vacc0, vacc1);
    const v128_t vy1 = wasm_u8x16_narrow_i16x8(vacc2, vacc3);

    wasm_v128_store(y, vy0);
    wasm_v128_store((y + 16), vy1);
    y += 32;
  }
  for (; n >= 8 * sizeof(uint8_t); n -= 8 * sizeof(uint8_t)) {
    v128_t vacc = wasm_u16x8_load8x8(x);
    v128_t vmultiplier = wasm_i16x8_gt(vacc, vinput_zero_point);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vmultiplier = wasm_v128_and(vmultiplier, vmultiplier_diff);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = wasm_v128_xor(vmultiplier, vmultiplier_base);
    vacc = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);
    x += 8;

    const v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);
    wasm_v128_store64_lane(y, vy, 0);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(uint8_t));
    assert(n <= 7 * sizeof(uint8_t));

    v128_t vacc = wasm_u16x8_load8x8(x);
    v128_t vmultiplier = wasm_i16x8_gt(vacc, vinput_zero_point);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vmultiplier = wasm_v128_and(vmultiplier, vmultiplier_diff);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = wasm_v128_xor(vmultiplier, vmultiplier_base);
    vacc = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);

    v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);
    if (n & (4 * sizeof(uint8_t))) {
      wasm_v128_store32_lane(y, vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      y += 4;
    }
    if (n & (2 * sizeof(uint8_t))) {
      wasm_v128_store16_lane(y, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      y += 2;
    }
    if (n & (1 * sizeof(uint8_t))) {
      wasm_v128_store8_lane(y, vy, 0);
    }
  }
}
