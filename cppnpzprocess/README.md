# cppnpzprocess

这个目录提供了一个 C++ 版的 NPZ 图像处理程序，用来改写 pynpzprocess/npz_image_processor.py 的主要功能。

程序名：npz_image_processor

## 功能

- 对 NPZ 中的 image 键执行缩放、旋转、裁切、对比度调整、伽马调整。
- 当存在 label 键时，缩放、旋转、裁切会同步作用于 label，以保证 image 和 label 的空间对应关系不变。
- 对比度调整和伽马调整只作用于 image，不修改 label。
- 输出时保留原有 NPZ 键名结构，未修改的条目会继续写回输出文件。

## 编译

依赖：

- clang++
- OpenCV 4
- zlib

macOS 编译命令：

```bash
bash cppnpzprocess/buildmacOS.sh
```

编译完成后会生成：

```bash
cppnpzprocess/npz_image_processor
```

## 参数说明

| 参数 | 是否必填 | 说明 |
| --- | --- | --- |
| --input <path> | 是 | 输入 NPZ 文件路径 |
| --output <path> | 否 | 输出 NPZ 文件路径；不传时默认输出为原文件同目录下的 *_processed.npz |
| --scale-x <float> | 否 | X 方向缩放倍率，默认 1.0 |
| --scale-y <float> | 否 | Y 方向缩放倍率，默认 1.0 |
| --rotate <float> | 否 | 逆时针旋转角度，单位为度，默认 0 |
| --crop X Y W H | 否 | 裁切矩形，格式为左上角坐标加宽高 |
| --contrast <float> | 否 | 对比度倍率，默认 1.0 |
| --gamma <float> | 否 | 伽马值，默认 1.0，必须大于 0 |
| --preserve-resolution | 否 | 处理后尝试恢复为输入原始分辨率 |
| --help / -h | 否 | 显示用法 |

## 用法示例

只做旋转：

```bash
./cppnpzprocess/npz_image_processor \
  --input "/path/to/case.npz" \
  --output "/path/to/case_rotate.npz" \
  --rotate 15
```

几何处理加图像增强组合：

```bash
./cppnpzprocess/npz_image_processor \
  --input "/path/to/case.npz" \
  --output "/path/to/case_combo.npz" \
  --scale-x 1.2 \
  --scale-y 0.8 \
  --rotate 15 \
  --crop 10 20 256 256 \
  --contrast 1.15 \
  --gamma 0.9
```

## 行为说明

- 如果输入文件不存在，程序会直接报错退出。
- 如果输入 NPZ 中没有 image 键，程序会直接报错退出。
- 当前实现支持 2D 图像，以及常见 3D 布局：HWC、CHW、N×H×W。
- 样例数据中 image.npy 和 label.npy 为 Fortran-order 存储，程序已兼容该布局读取。