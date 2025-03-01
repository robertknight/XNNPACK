// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <xnnpack/common.h>

#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_CS16_BFLY4_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                         \
    size_t samples,                                  \
    int16_t* data,                                   \
    const size_t stride,                             \
    const int16_t* twiddle);

DECLARE_CS16_BFLY4_UKERNEL_FUNCTION(xnn_cs16_bfly4_ukernel__scalar_x1)
DECLARE_CS16_BFLY4_UKERNEL_FUNCTION(xnn_cs16_bfly4_ukernel__scalar_x2)
DECLARE_CS16_BFLY4_UKERNEL_FUNCTION(xnn_cs16_bfly4_ukernel__scalar_x3)
DECLARE_CS16_BFLY4_UKERNEL_FUNCTION(xnn_cs16_bfly4_ukernel__scalar_x4)
DECLARE_CS16_BFLY4_UKERNEL_FUNCTION(xnn_cs16_bfly4m1_ukernel__scalar_x1)

#define DECLARE_CS16_FFTR_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                        \
    size_t samples,                                 \
    const int16_t* input,                           \
    int16_t* output,                                \
    const int16_t* twiddle);

DECLARE_CS16_FFTR_UKERNEL_FUNCTION(xnn_cs16_fftr_ukernel__scalar_x1)
DECLARE_CS16_FFTR_UKERNEL_FUNCTION(xnn_cs16_fftr_ukernel__scalar_x2)
DECLARE_CS16_FFTR_UKERNEL_FUNCTION(xnn_cs16_fftr_ukernel__scalar_x3)
DECLARE_CS16_FFTR_UKERNEL_FUNCTION(xnn_cs16_fftr_ukernel__scalar_x4)

#ifdef __cplusplus
}  // extern "C"
#endif
