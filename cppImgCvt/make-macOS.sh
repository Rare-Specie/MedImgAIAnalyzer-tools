#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

OPENCV_CFLAGS="$(pkg-config --cflags opencv4)"
OPENCV_LIBS="$(pkg-config --libs opencv4)"

clang++ \
  -std=c++17 \
  -stdlib=libc++ \
  -O2 \
  -I"${ROOT_DIR}" \
  ${OPENCV_CFLAGS} \
  "${SCRIPT_DIR}/main.cpp" \
  "${ROOT_DIR}/cnpy.cpp" \
  -o "${SCRIPT_DIR}/imgcvt" \
  ${OPENCV_LIBS} \
  -lz

echo "✅ 编译成功: ${SCRIPT_DIR}/imgcvt"
