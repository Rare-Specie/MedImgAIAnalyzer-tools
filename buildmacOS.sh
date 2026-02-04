#!/bin/bash

# 检查是否传入了待编译的 .cpp 文件参数
if [ $# -ne 1 ]; then
    echo "使用方法: ./build.sh [你的C++源文件.cpp]"
    echo "示例: ./build.sh main.cpp"
    exit 1
fi

# 定义变量，对应 VS Code 中的内置变量
SOURCE_FILE="$1"  # 传入的第一个参数 = ${file}
FILE_DIR=$(dirname "$SOURCE_FILE")  # 文件所在目录 = ${fileDirname}
FILE_NAME=$(basename "$SOURCE_FILE")  # 带后缀的文件名
FILE_NAME_NO_EXT=${FILE_NAME%.*}  # 去掉后缀的文件名 = ${fileBasenameNoExtension}
OUTPUT_FILE="${FILE_DIR}/${FILE_NAME_NO_EXT}"  # 输出可执行文件路径

# 通过 pkg-config 获取 OpenCV 编译/链接参数（若可用）
OPENCV_CFLAGS=$(pkg-config --cflags opencv4 2>/dev/null)
OPENCV_LIBS=$(pkg-config --libs opencv4 2>/dev/null)

# 原 VS Code 任务中的 clang++ 编译命令（补全链接库与 cnpy 源码）
clang++ \
-std=c++17 \
-stdlib=libc++ \
-I/opt/homebrew/include \
-I/opt/homebrew/include/opencv4 \
-I/usr/local/include \
${OPENCV_CFLAGS} \
-g \
"${SOURCE_FILE}" \
"${FILE_DIR}/cnpy.cpp" \
-o "${OUTPUT_FILE}" \
${OPENCV_LIBS} \
-lz

# 检查编译是否成功
if [ $? -eq 0 ]; then
    echo "✅ 编译成功！"
    echo "📁 可执行文件路径: ${OUTPUT_FILE}"
else
    echo "❌ 编译失败，请检查代码或编译参数！"
    exit 1
fi