#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

: "${CXX:=icpx}"
: "${ESIMD_DEVICE:=ptl-h}"
echo "Building ESIMD SDP shared library for $ESIMD_DEVICE with $CXX..."
"$CXX" sdp_kernels.cpp -shared -fPIC -o libesimd.unify.lgrf.so \
    -DBUILD_ESIMD_KERNEL_LIB \
    -fsycl -fsycl-targets=spir64_gen \
    -Xs "-device $ESIMD_DEVICE -options -doubleGRF" \
    -Wl,-soname,libesimd.unify.lgrf.so \
    -O3
echo "Built: $script_dir/libesimd.unify.lgrf.so"
