#!/bin/sh

set -eu

# Build with `-c opt` for release build with minimal assert checks. Build without
# for verbose logging.

# bazel build '//:xnnpack-wasm-test' --cpu=wasm --crosstool_top=@emsdk//emscripten_toolchain:everything -c opt
bazel build '//:xnnpack-wasm-test' --cpu=wasm --crosstool_top=@emsdk//emscripten_toolchain:everything
tar -xf bazel-bin/xnnpack-wasm-test
node xnnpack-wasm-test.js


