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


#define DECLARE_S16_WINDOW_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                         \
    size_t rows,                                     \
    size_t batch_size,                               \
    const int16_t* input,                            \
    const int16_t* weights,                          \
    uint32_t shift,                                  \
    int16_t* output);


DECLARE_S16_WINDOW_UKERNEL_FUNCTION(xnn_s16_window_ukernel__neon_x8)
DECLARE_S16_WINDOW_UKERNEL_FUNCTION(xnn_s16_window_ukernel__neon_x16)
DECLARE_S16_WINDOW_UKERNEL_FUNCTION(xnn_s16_window_ukernel__neon_x24)
DECLARE_S16_WINDOW_UKERNEL_FUNCTION(xnn_s16_window_ukernel__neon_x32)

DECLARE_S16_WINDOW_UKERNEL_FUNCTION(xnn_s16_window_shift12_ukernel__neon_x8)
DECLARE_S16_WINDOW_UKERNEL_FUNCTION(xnn_s16_window_shift12_ukernel__neon_x16)
DECLARE_S16_WINDOW_UKERNEL_FUNCTION(xnn_s16_window_shift12_ukernel__neon_x24)
DECLARE_S16_WINDOW_UKERNEL_FUNCTION(xnn_s16_window_shift12_ukernel__neon_x32)

DECLARE_S16_WINDOW_UKERNEL_FUNCTION(xnn_s16_window_shift15_ukernel__neon_x8)
DECLARE_S16_WINDOW_UKERNEL_FUNCTION(xnn_s16_window_shift15_ukernel__neon_x16)
DECLARE_S16_WINDOW_UKERNEL_FUNCTION(xnn_s16_window_shift15_ukernel__neon_x24)
DECLARE_S16_WINDOW_UKERNEL_FUNCTION(xnn_s16_window_shift15_ukernel__neon_x32)

DECLARE_S16_WINDOW_UKERNEL_FUNCTION(xnn_s16_window_ukernel__scalar_x1)
DECLARE_S16_WINDOW_UKERNEL_FUNCTION(xnn_s16_window_ukernel__scalar_x2)
DECLARE_S16_WINDOW_UKERNEL_FUNCTION(xnn_s16_window_ukernel__scalar_x3)
DECLARE_S16_WINDOW_UKERNEL_FUNCTION(xnn_s16_window_ukernel__scalar_x4)

#ifdef __cplusplus
}  // extern "C"
#endif
