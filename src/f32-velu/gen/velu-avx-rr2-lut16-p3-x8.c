// Auto-generated file. Do not edit!
//   Template: src/f32-velu/avx-rr2-lut16-p3.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vunary.h>


extern XNN_INTERNAL const int xnn_table_exp2minus_k_over_16[16];

void xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x8(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n % sizeof(float) == 0);

  const __m256 vprescale = _mm256_load_ps(params->avx_rr2_lut16_p3.prescale);
  const __m256 valpha = _mm256_load_ps(params->avx_rr2_lut16_p3.alpha);
  const __m256 vbeta = _mm256_load_ps(params->avx_rr2_lut16_p3.beta);
  const __m256 vsat_cutoff = _mm256_load_ps(params->avx_rr2_lut16_p3.sat_cutoff);
  const __m256 vmagic_bias = _mm256_load_ps(params->avx_rr2_lut16_p3.magic_bias);
  const __m256 vlog2e = _mm256_load_ps(params->avx_rr2_lut16_p3.log2e);
  const __m256 vindex_mask = _mm256_load_ps((const float*) params->avx_rr2_lut16_p3.index_mask);
  const __m256 vminus_ln2_hi = _mm256_load_ps(params->avx_rr2_lut16_p3.minus_ln2_hi);
  const __m256 vminus_ln2_lo = _mm256_load_ps(params->avx_rr2_lut16_p3.minus_ln2_lo);
  const __m256 vc3 = _mm256_load_ps(params->avx_rr2_lut16_p3.c3);
  const __m256 vc2 = _mm256_load_ps(params->avx_rr2_lut16_p3.c2);
  const __m256 vone = _mm256_load_ps(params->avx_rr2_lut16_p3.one);

  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    __m256 vx = _mm256_loadu_ps(x);
    x += 8;

    const __m256 vz = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx, vprescale));

    __m256 vn = _mm256_add_ps(_mm256_mul_ps(vz, vlog2e), vmagic_bias);

    const __m256 vidx = _mm256_and_ps(vn, vindex_mask);

    const __m128i vidx_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vidx)), 2);
    const __m128i vidx_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vidx, 1)), 2);
    #if XNN_ARCH_X86_64
      const uint64_t vidx_ll = (uint64_t) _mm_cvtsi128_si64(vidx_lo);
      const uint64_t vidx_lh = (uint64_t) _mm_extract_epi64(vidx_lo, 1);
      const uint64_t vidx_hl = (uint64_t) _mm_cvtsi128_si64(vidx_hi);
      const uint64_t vidx_hh = (uint64_t) _mm_extract_epi64(vidx_hi, 1);
      __m128i vl_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_ll)));
      __m128i vl_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_lh)));
      __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hl)));
      __m128i vl_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hh)));
      vl_ll = _mm_insert_epi32(vl_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_ll >> 32))), 1);
      vl_lh = _mm_insert_epi32(vl_lh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_lh >> 32))), 1);
      vl_hl = _mm_insert_epi32(vl_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hl >> 32))), 1);
      vl_hh = _mm_insert_epi32(vl_hh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hh >> 32))), 1);
    #else
      __m128i vl_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx_lo))));
      __m128i vl_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_lo, 2))));
      __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx_hi))));
      __m128i vl_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_hi, 2))));
      vl_ll = _mm_insert_epi32(vl_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_lo, 1))), 1);
      vl_lh = _mm_insert_epi32(vl_lh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_lo, 3))), 1);
      vl_hl = _mm_insert_epi32(vl_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_hi, 1))), 1);
      vl_hh = _mm_insert_epi32(vl_hh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_hi, 3))), 1);
    #endif
    const __m128i ven_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn)), 19);
    const __m128i ven_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn, 1)), 19);

    const __m128i vl_lo = _mm_unpacklo_epi64(vl_ll, vl_lh);
    const __m128i vl_hi = _mm_unpacklo_epi64(vl_hl, vl_hh);

    vn = _mm256_sub_ps(vn, vmagic_bias);
    const __m128 vs_lo = _mm_castsi128_ps(_mm_add_epi32(vl_lo, ven_lo));
    const __m128 vs_hi = _mm_castsi128_ps(_mm_add_epi32(vl_hi, ven_hi));

    __m256 vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_hi), vz);
    vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_lo), vt);
    __m256 vs = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo), vs_hi, 1);

    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc3, vt), vc2);
    vp = _mm256_mul_ps(vp, vt);

    vt = _mm256_mul_ps(vt, vs);
    vs = _mm256_sub_ps(vs, vone);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vt);

    const __m256 ve = _mm256_mul_ps(_mm256_add_ps(vp, vs), valpha);
    vx = _mm256_mul_ps(vx, vbeta);
    const __m256 vy = _mm256_blendv_ps(vx, ve, vx);

    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx_rr2_lut16_p3.mask_table[7] - n));

    __m256 vx = _mm256_maskload_ps(x, vmask);

    const __m256 vz = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx, vprescale));

    __m256 vn = _mm256_add_ps(_mm256_mul_ps(vz, vlog2e), vmagic_bias);

    const __m256 vidx = _mm256_and_ps(vn, vindex_mask);

    const __m128i vidx_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vidx)), 2);
    const __m128i vidx_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vidx, 1)), 2);
    #if XNN_ARCH_X86_64
      const uint64_t vidx_ll = (uint64_t) _mm_cvtsi128_si64(vidx_lo);
      const uint64_t vidx_lh = (uint64_t) _mm_extract_epi64(vidx_lo, 1);
      const uint64_t vidx_hl = (uint64_t) _mm_cvtsi128_si64(vidx_hi);
      const uint64_t vidx_hh = (uint64_t) _mm_extract_epi64(vidx_hi, 1);
      __m128i vl_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_ll)));
      __m128i vl_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_lh)));
      __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hl)));
      __m128i vl_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hh)));
      vl_ll = _mm_insert_epi32(vl_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_ll >> 32))), 1);
      vl_lh = _mm_insert_epi32(vl_lh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_lh >> 32))), 1);
      vl_hl = _mm_insert_epi32(vl_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hl >> 32))), 1);
      vl_hh = _mm_insert_epi32(vl_hh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hh >> 32))), 1);
    #else
      __m128i vl_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx_lo))));
      __m128i vl_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_lo, 2))));
      __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx_hi))));
      __m128i vl_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_hi, 2))));
      vl_ll = _mm_insert_epi32(vl_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_lo, 1))), 1);
      vl_lh = _mm_insert_epi32(vl_lh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_lo, 3))), 1);
      vl_hl = _mm_insert_epi32(vl_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_hi, 1))), 1);
      vl_hh = _mm_insert_epi32(vl_hh, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi32(vidx_hi, 3))), 1);
    #endif
    const __m128i ven_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn)), 19);
    const __m128i ven_hi = _mm_slli_epi32(_mm_castps_si128(_mm256_extractf128_ps(vn, 1)), 19);

    const __m128i vl_lo = _mm_unpacklo_epi64(vl_ll, vl_lh);
    const __m128i vl_hi = _mm_unpacklo_epi64(vl_hl, vl_hh);

    vn = _mm256_sub_ps(vn, vmagic_bias);
    const __m128 vs_lo = _mm_castsi128_ps(_mm_add_epi32(vl_lo, ven_lo));
    const __m128 vs_hi = _mm_castsi128_ps(_mm_add_epi32(vl_hi, ven_hi));

    __m256 vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_hi), vz);
    vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2_lo), vt);
    __m256 vs = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo), vs_hi, 1);

    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc3, vt), vc2);
    vp = _mm256_mul_ps(vp, vt);

    vt = _mm256_mul_ps(vt, vs);
    vs = _mm256_sub_ps(vs, vone);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vt);

    const __m256 ve = _mm256_mul_ps(_mm256_add_ps(vp, vs), valpha);
    vx = _mm256_mul_ps(vx, vbeta);
    const __m256 vy = _mm256_blendv_ps(vx, ve, vx);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (n & (4 * sizeof(float))) {
      _mm_storeu_ps(y, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      y += 4;
    }
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy_lo);
    }
  }
}
