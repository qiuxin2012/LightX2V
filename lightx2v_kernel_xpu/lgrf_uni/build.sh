#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

: "${CXX:=icpx}"
target=${XPU_TARGET:-bmg}
case "$target" in
    bmg) arch_define=OMNI_XPU_ARCH_BMG ;;
    ptl-h) arch_define=OMNI_XPU_ARCH_PTL_H ;;
    *) echo "ERROR: XPU_TARGET must be bmg or ptl-h (got '$target')" >&2; exit 1 ;;
esac
echo "Building ESIMD SDP shared library for $target with $CXX..."
"$CXX" sdp_kernels.cpp -shared -fPIC -o libesimd.unify.lgrf.so \
    -DBUILD_ESIMD_KERNEL_LIB \
    -D"$arch_define" \
    -fsycl -fsycl-targets=spir64_gen \
    -Xs "-device $target -options -doubleGRF" \
    -Wl,-soname,libesimd.unify.lgrf.so \
    -O3
echo "Built: $script_dir/libesimd.unify.lgrf.so"
