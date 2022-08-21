#!/bin/sh

bazel build '//:xnnpack-wasm-test' --cpu=wasm --crosstool_top=@emsdk//emscripten_toolchain:everything -c opt
tar -xf bazel-bin/xnnpack-wasm-test
node xnnpack-wasm-test.js


