#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### SCALAR ###################################
tools/xngen src/cs16-bfly4/scalar.c.in -D SAMPLE_TILE=1 -D M=0 -o src/cs16-bfly4/gen/scalar-x1.c &
tools/xngen src/cs16-bfly4/scalar.c.in -D SAMPLE_TILE=2 -D M=0 -o src/cs16-bfly4/gen/scalar-x2.c &
tools/xngen src/cs16-bfly4/scalar.c.in -D SAMPLE_TILE=3 -D M=0 -o src/cs16-bfly4/gen/scalar-x3.c &
tools/xngen src/cs16-bfly4/scalar.c.in -D SAMPLE_TILE=4 -D M=0 -o src/cs16-bfly4/gen/scalar-x4.c &

tools/xngen src/cs16-bfly4/scalar.c.in -D SAMPLE_TILE=1 -D M=1 -o src/cs16-bfly4/gen/scalar-m1-x1.c &


################################## Unit tests #################################
tools/generate-bfly4-test.py --spec test/cs16-bfly4.yaml --output test/cs16-bfly4.cc &

wait
