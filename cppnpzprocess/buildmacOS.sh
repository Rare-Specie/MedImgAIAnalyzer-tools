#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

OPENCV_CFLAGS="$(pkg-config --cflags opencv4)"
OPENCV_LIBS="$(pkg-config --libs opencv4)"

clang++ \
  -std=c++17 \
  -stdlib=libc++ \
  -O2 \
  ${OPENCV_CFLAGS} \
  -I"${SCRIPT_DIR}" \
  "${SCRIPT_DIR}/main.cpp" \
  -o "${SCRIPT_DIR}/npz_image_processor" \
  ${OPENCV_LIBS} \
  -lz

echo "编译成功: ${SCRIPT_DIR}/npz_image_processor"