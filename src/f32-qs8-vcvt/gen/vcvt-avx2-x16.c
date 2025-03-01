// Auto-generated file. Do not edit!
//   Template: src/f32-qs8-vcvt/avx2.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vcvt.h>


void xnn_f32_qs8_vcvt_ukernel__avx2_x16(
    size_t n,
    const float* x,
    int8_t* y,
    const union xnn_f32_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const __m256 vscale = _mm256_load_ps(params->avx2.scale);
  const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->avx2.output_max_less_zero_point);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->avx2.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->avx2.output_min);

  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    __m256 vx01 = _mm256_loadu_ps(x);
    __m256 vx23 = _mm256_loadu_ps(x + 8);
    x += 16;

    vx01 = _mm256_mul_ps(vx01, vscale);
    vx23 = _mm256_mul_ps(vx23, vscale);

    vx01 = _mm256_min_ps(vx01, voutput_max_less_zero_point);
    vx23 = _mm256_min_ps(vx23, voutput_max_less_zero_point);

    const __m256i vacc01 = _mm256_cvtps_epi32(vx01);
    const __m256i vacc23 = _mm256_cvtps_epi32(vx23);

    __m256i vacc0213 = _mm256_packs_epi32(vacc01, vacc23);

    vacc0213 = _mm256_adds_epi16(vacc0213, voutput_zero_point);

    const __m128i vy0213 = _mm_packs_epi16(_mm256_castsi256_si128(vacc0213), _mm256_extracti128_si256(vacc0213, 1));

    __m128i vy0123 = _mm_shuffle_epi32(vy0213, _MM_SHUFFLE(3, 1, 2, 0));

    vy0123 = _mm_max_epi8(vy0123, voutput_min);

    _mm_storeu_si128((__m128i*) y, vy0123);
    y += 16;
  }
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    __m256 vx = _mm256_loadu_ps(x);
    vx = _mm256_mul_ps(vx, vscale);
    vx = _mm256_min_ps(vx, voutput_max_less_zero_point);
    x += 8;

    const __m256i vacc = _mm256_cvtps_epi32(vx);

    __m128i vy = _mm_packs_epi32(_mm256_castsi256_si128(vacc), _mm256_extracti128_si256(vacc, 1));
    vy = _mm_adds_epi16(vy, _mm256_castsi256_si128(voutput_zero_point));
    vy = _mm_packs_epi16(vy, vy);
    vy = _mm_max_epi8(vy, voutput_min);

    _mm_storel_epi64((__m128i*) y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx2.mask_table[7] - n));

    __m256 vx = _mm256_maskload_ps(x, vmask);
    vx = _mm256_mul_ps(vx, vscale);
    vx = _mm256_min_ps(vx, voutput_max_less_zero_point);

    const __m256i vacc = _mm256_cvtps_epi32(vx);

    __m128i vy = _mm_packs_epi32(_mm256_castsi256_si128(vacc), _mm256_extracti128_si256(vacc, 1));
    vy = _mm_adds_epi16(vy, _mm256_castsi256_si128(voutput_zero_point));
    vy = _mm_packs_epi16(vy, vy);
    vy = _mm_max_epi8(vy, voutput_min);

    if (n & (4 * sizeof(float))) {
      _mm_storeu_si32(y, vy);
      y += 4;
      vy = _mm_srli_epi64(vy, 32);
    }
    if (n & (2 * sizeof(float))) {
      _mm_storeu_si16(y, vy);
      y += 2;
      vy = _mm_srli_epi32(vy, 16);
    }
    if (n & (1 * sizeof(float))) {
      *y = (int8_t) _mm_extract_epi8(vy, 0);
    }
  }
}
