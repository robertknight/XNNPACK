workspace(name = "xnnpack")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Load emsdk. See https://github.com/emscripten-core/emsdk/tree/26a0dea0d3bbf616fa7f0a908e5b08aab406f7c4/bazel#setup-instructions
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "emsdk",
    sha256 = "6479c60710bfb1d146a8bdd8619b693699e73185c850a6eb79ef2bd7e2a8e411",
    strip_prefix = "emsdk-3.1.18/bazel",
    url = "https://github.com/emscripten-core/emsdk/archive/refs/tags/3.1.18.tar.gz"
)

load("@emsdk//:deps.bzl", emsdk_deps = "deps")
emsdk_deps()

load("@emsdk//:emscripten_deps.bzl", emsdk_emscripten_deps = "emscripten_deps")
emsdk_emscripten_deps(emscripten_version = "3.1.18")

# Bazel rule definitions
http_archive(
    name = "rules_cc",
    strip_prefix = "rules_cc-main",
    urls = ["https://github.com/bazelbuild/rules_cc/archive/main.zip"],
)

# Bazel Skylib.
http_archive(
    name = "bazel_skylib",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.2.1/bazel-skylib-1.2.1.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.2.1/bazel-skylib-1.2.1.tar.gz",
    ],
    sha256 = "f7be3474d42aae265405a592bb7da8e171919d74c16f082a5457840f06054728",
)

# Google Test framework, used by most unit-tests.
http_archive(
    name = "com_google_googletest",
    strip_prefix = "googletest-master",
    urls = ["https://github.com/google/googletest/archive/master.zip"],
)

# Google Benchmark library, used in micro-benchmarks.
http_archive(
    name = "com_google_benchmark",
    strip_prefix = "benchmark-master",
    urls = ["https://github.com/google/benchmark/archive/master.zip"],
)

# FP16 library, used for half-precision conversions
http_archive(
    name = "FP16",
    strip_prefix = "FP16-0a92994d729ff76a58f692d3028ca1b64b145d91",
    sha256 = "e66e65515fa09927b348d3d584c68be4215cfe664100d01c9dbc7655a5716d70",
    urls = [
        "https://github.com/Maratyszcza/FP16/archive/0a92994d729ff76a58f692d3028ca1b64b145d91.zip",
    ],
    build_file = "@//third_party:FP16.BUILD",
)

# FXdiv library, used for repeated integer division by the same factor
http_archive(
    name = "FXdiv",
    strip_prefix = "FXdiv-b408327ac2a15ec3e43352421954f5b1967701d1",
    sha256 = "ab7dfb08829bee33dca38405d647868fb214ac685e379ec7ef2bebcd234cd44d",
    urls = ["https://github.com/Maratyszcza/FXdiv/archive/b408327ac2a15ec3e43352421954f5b1967701d1.zip"],
)

# pthreadpool library, used for parallelization
http_archive(
    name = "pthreadpool",
    strip_prefix = "pthreadpool-b8374f80e42010941bda6c85b0e3f1a1bd77a1e0",
    sha256 = "b96413b10dd8edaa4f6c0a60c6cf5ef55eebeef78164d5d69294c8173457f0ec",
    urls = ["https://github.com/Maratyszcza/pthreadpool/archive/b8374f80e42010941bda6c85b0e3f1a1bd77a1e0.zip"],
)

# clog library, used for logging
http_archive(
    name = "clog",
    strip_prefix = "cpuinfo-d5e37adf1406cf899d7d9ec1d317c47506ccb970",
    sha256 = "3f2dc1970f397a0e59db72f9fca6ff144b216895c1d606f6c94a507c1e53a025",
    urls = [
        "https://github.com/pytorch/cpuinfo/archive/d5e37adf1406cf899d7d9ec1d317c47506ccb970.tar.gz",
    ],
    build_file = "@//third_party:clog.BUILD",
)


# cpuinfo library, used for detecting processor characteristics
http_archive(
    name = "cpuinfo",
    strip_prefix = "cpuinfo-ed8b86a253800bafdb7b25c5c399f91bff9cb1f3",
    sha256 = "a7f9a188148a1660149878f737f42783e72f33a4f842f3e362fee2c981613e53",
    urls = [
        "https://github.com/pytorch/cpuinfo/archive/ed8b86a253800bafdb7b25c5c399f91bff9cb1f3.zip",
    ],
    build_file = "@//third_party:cpuinfo.BUILD",
    patches = ["@//third_party:cpuinfo.patch"],
)

# Ruy library, used to benchmark against
http_archive(
   name = "ruy",
   strip_prefix = "ruy-9f53ba413e6fc879236dcaa3e008915973d67a4f",
   sha256 = "fe8345f521bb378745ebdd0f8c5937414849936851d2ec2609774eb2d7098e54",
   urls = [
       "https://github.com/google/ruy/archive/9f53ba413e6fc879236dcaa3e008915973d67a4f.zip",
   ],
)

# # Android NDK location and version is auto-detected from $ANDROID_NDK_HOME environment variable
# android_ndk_repository(name = "androidndk")

# # Android SDK location and API is auto-detected from $ANDROID_HOME environment variable
# android_sdk_repository(name = "androidsdk")
