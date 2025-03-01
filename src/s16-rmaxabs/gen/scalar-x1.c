// Auto-generated file. Do not edit!
//   Template: src/s16-rmaxabs/scalar.c.in
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
#include <xnnpack/rmaxabs.h>


void xnn_s16_rmaxabs_ukernel__scalar_x1(
    size_t batch,
    const int16_t* input,
    uint16_t* output) {

  assert(batch > 0);
  assert(input != NULL);
  assert(output != NULL);

  int32_t vmax0 = 0;


  if (batch != 0) {
    do {
      const int32_t vi = (int32_t) *input++;
      const int32_t vabs = vi >= 0 ? vi : -vi;
      vmax0 = math_max_s32(vmax0, vabs);
    } while (--batch != 0);
  }
  *output = (uint16_t) vmax0;
}
