#!/usr/bin/env python
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import codecs
import math
import os
import re
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primes import next_prime
import xngen
import xnncommon


parser = argparse.ArgumentParser(description='BFly4 microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  m = 0
  match = re.fullmatch(r"xnn_cs16_bfly4(m(\d+))?_ukernel__(.+)_x(\d+)", name)
  assert match is not None
  if match.group(2):
    m = int(match.group(2))
  samples_tile = int(match.group(4))

  arch, isa = xnncommon.parse_target_name(target_name=match.group(3))
  return m, samples_tile, arch, isa


BFLY4_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, samples_eq_1) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  BFly4MicrokernelTester()
    .samples(1)
    .stride(64)
    .Test(${", ".join(TEST_ARGS)});
}

$if M == 0:
  TEST(${TEST_NAME}, samples_eq_4) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    BFly4MicrokernelTester()
      .samples(4)
      .stride(16)
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, samples_eq_16) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    BFly4MicrokernelTester()
      .samples(16)
      .stride(4)
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, samples_eq_64) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    BFly4MicrokernelTester()
      .samples(64)
      .stride(1)
      .Test(${", ".join(TEST_ARGS)});
  }

"""


def generate_test_cases(ukernel, m, samples_tile, isa):
  """Generates all tests cases for a BFly4 micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    m: fixed number of samples for specialized m1 microkernel.
    samples_tile: Number of samples processed per one iteration of the inner
                  loop of the micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  return xngen.preprocess(BFLY4_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": [ukernel],
      "DATATYPE": datatype,
      "M": m,
      "SAMPLE_TILE": samples_tile,
      "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
      "next_prime": next_prime,
    })


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of micro-kernels in the spec")

    tests = """\
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/fft.h>
#include "bfly4-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      m, samples_tile, arch, isa = split_ukernel_name(name)

      # specification can override architecture
      arch = ukernel_spec.get("arch", arch)

      test_case = generate_test_cases(name, m, samples_tile, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    txt_changed = True
    if os.path.exists(options.output):
      with codecs.open(options.output, "r", encoding="utf-8") as output_file:
        txt_changed = output_file.read() != tests

    if txt_changed:
      with codecs.open(options.output, "w", encoding="utf-8") as output_file:
        output_file.write(tests)


if __name__ == "__main__":
  main(sys.argv[1:])
