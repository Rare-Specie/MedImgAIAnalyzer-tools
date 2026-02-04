#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "cnpy.h"

struct NpzImageLabel {
  std::vector<double> image;  // H*W
  std::optional<std::vector<double>> label;  // H*W
  int height = 0;
  int width = 0;
};

inline NpzImageLabel LoadNpzImageLabel(const std::string &npz_path) {
  if (!std::filesystem::exists(npz_path)) {
    throw std::runtime_error("未找到输入文件: " + npz_path);
  }

  cnpy::npz_t npz = cnpy::npz_load(npz_path);
  if (npz.find("image") == npz.end()) {
    throw std::runtime_error("npz中未找到image键");
  }

  const cnpy::NpyArray &img_arr = npz.at("image");
  if (img_arr.shape.size() != 2) {
    throw std::runtime_error("image应为2D数组");
  }

  const size_t h = img_arr.shape[0];
  const size_t w = img_arr.shape[1];
  const size_t numel = h * w;

  std::vector<double> img(numel, 0.0);
  if (img_arr.word_size == sizeof(double)) {
    const double *src = img_arr.data<double>();
    std::copy(src, src + numel, img.begin());
  } else if (img_arr.word_size == sizeof(float)) {
    const float *src = img_arr.data<float>();
    for (size_t i = 0; i < numel; ++i) {
      img[i] = static_cast<double>(src[i]);
    }
  } else if (img_arr.word_size == sizeof(uint8_t)) {
    const uint8_t *src = img_arr.data<uint8_t>();
    for (size_t i = 0; i < numel; ++i) {
      img[i] = static_cast<double>(src[i]);
    }
  } else if (img_arr.word_size == sizeof(int16_t)) {
    const int16_t *src = img_arr.data<int16_t>();
    for (size_t i = 0; i < numel; ++i) {
      img[i] = static_cast<double>(src[i]);
    }
  } else if (img_arr.word_size == sizeof(int32_t)) {
    const int32_t *src = img_arr.data<int32_t>();
    for (size_t i = 0; i < numel; ++i) {
      img[i] = static_cast<double>(src[i]);
    }
  } else if (img_arr.word_size == sizeof(uint16_t)) {
    const uint16_t *src = img_arr.data<uint16_t>();
    for (size_t i = 0; i < numel; ++i) {
      img[i] = static_cast<double>(src[i]);
    }
  } else {
    throw std::runtime_error("不支持的image数据类型");
  }

  std::optional<std::vector<double>> label;
  if (npz.find("label") != npz.end()) {
    const cnpy::NpyArray &lab_arr = npz.at("label");
    if (lab_arr.shape.size() != 2) {
      throw std::runtime_error("label应为2D数组");
    }
    const size_t lnum = lab_arr.shape[0] * lab_arr.shape[1];
    std::vector<double> lab(lnum, 0.0);
    if (lab_arr.word_size == sizeof(double)) {
      const double *src = lab_arr.data<double>();
      std::copy(src, src + lnum, lab.begin());
    } else if (lab_arr.word_size == sizeof(float)) {
      const float *src = lab_arr.data<float>();
      for (size_t i = 0; i < lnum; ++i) {
        lab[i] = static_cast<double>(src[i]);
      }
    } else if (lab_arr.word_size == sizeof(uint8_t)) {
      const uint8_t *src = lab_arr.data<uint8_t>();
      for (size_t i = 0; i < lnum; ++i) {
        lab[i] = static_cast<double>(src[i]);
      }
    } else if (lab_arr.word_size == sizeof(int16_t)) {
      const int16_t *src = lab_arr.data<int16_t>();
      for (size_t i = 0; i < lnum; ++i) {
        lab[i] = static_cast<double>(src[i]);
      }
    } else if (lab_arr.word_size == sizeof(int32_t)) {
      const int32_t *src = lab_arr.data<int32_t>();
      for (size_t i = 0; i < lnum; ++i) {
        lab[i] = static_cast<double>(src[i]);
      }
    } else if (lab_arr.word_size == sizeof(uint16_t)) {
      const uint16_t *src = lab_arr.data<uint16_t>();
      for (size_t i = 0; i < lnum; ++i) {
        lab[i] = static_cast<double>(src[i]);
      }
    } else {
      throw std::runtime_error("不支持的label数据类型");
    }
    label = std::move(lab);
  }

  if (img_arr.fortran_order) {
    std::vector<double> reordered(numel, 0.0);
    for (size_t y = 0; y < h; ++y) {
      for (size_t x = 0; x < w; ++x) {
        reordered[y * w + x] = img[x * h + y];
      }
    }
    img.swap(reordered);
  }

  if (label.has_value() && npz.at("label").fortran_order) {
    const cnpy::NpyArray &lab_arr = npz.at("label");
    const size_t lh = lab_arr.shape[0];
    const size_t lw = lab_arr.shape[1];
    const size_t lnum = lh * lw;
    std::vector<double> reordered(lnum, 0.0);
    auto &lab = *label;
    for (size_t y = 0; y < lh; ++y) {
      for (size_t x = 0; x < lw; ++x) {
        reordered[y * lw + x] = lab[x * lh + y];
      }
    }
    label = std::move(reordered);
  }

  return {std::move(img), std::move(label), static_cast<int>(h), static_cast<int>(w)};
}

inline void NormalizeImage(std::vector<double> &image) {
  double max_val = 0.0;
  for (double v : image) {
    if (v > max_val) max_val = v;
  }
  if (max_val > 1.0) {
    for (double &v : image) {
      v /= 255.0;
    }
  }
}

inline std::vector<double> ResizeImage(const std::vector<double> &image, int h, int w, int size) {
  cv::Mat src(h, w, CV_64FC1, const_cast<double *>(image.data()));
  cv::Mat dst;
  cv::resize(src, dst, cv::Size(size, size), 0, 0, cv::INTER_LINEAR);
  std::vector<double> out(size * size);
  std::memcpy(out.data(), dst.ptr<double>(), out.size() * sizeof(double));
  return out;
}

inline std::vector<double> ResizeMaskNearest(const std::vector<double> &mask, int h, int w, int size) {
  cv::Mat src(h, w, CV_64FC1, const_cast<double *>(mask.data()));
  cv::Mat dst;
  cv::resize(src, dst, cv::Size(size, size), 0, 0, cv::INTER_NEAREST);
  std::vector<double> out(size * size);
  std::memcpy(out.data(), dst.ptr<double>(), out.size() * sizeof(double));
  return out;
}

inline std::vector<int64_t> ResizeMaskNearestFromInt(const std::vector<int64_t> &mask, int h, int w, int size) {
  std::vector<float> tmp(h * w, 0.0f);
  for (int i = 0; i < h * w; ++i) {
    tmp[i] = static_cast<float>(mask[i]);
  }
  cv::Mat src(h, w, CV_32FC1, tmp.data());
  cv::Mat dst;
  cv::resize(src, dst, cv::Size(size, size), 0, 0, cv::INTER_NEAREST);
  std::vector<int64_t> out(static_cast<size_t>(size) * size);
  const float *p = dst.ptr<float>();
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<int64_t>(std::lround(p[i]));
  }
  return out;
}

inline void SavePredMaskNpz(const std::string &out_npz, const std::vector<int64_t> &mask, int size) {
  std::vector<size_t> shape = {static_cast<size_t>(size), static_cast<size_t>(size)};
  cnpy::npz_save(out_npz, "pred_mask", mask.data(), shape, "w");
}

inline std::pair<double, double> EvalDiceIou(const std::vector<int64_t> &pred,
                                             const std::vector<int64_t> &label) {
  if (pred.size() != label.size()) {
    throw std::runtime_error("pred与label大小不一致");
  }
  double inter = 0.0;
  double uni = 0.0;
  double pred_sum = 0.0;
  double label_sum = 0.0;

  for (size_t i = 0; i < pred.size(); ++i) {
    bool p = pred[i] > 0;
    bool l = label[i] > 0;
    if (p) pred_sum += 1.0;
    if (l) label_sum += 1.0;
    if (p && l) inter += 1.0;
    if (p || l) uni += 1.0;
  }

  double dice = (2.0 * inter) / (pred_sum + label_sum + 1e-6);
  double iou = inter / (uni + 1e-6);
  return {dice, iou};
}

inline std::vector<float> MakeInputTensorCHW(const std::vector<double> &image,
                                             int size,
                                             int channels = 3) {
  std::vector<float> input(static_cast<size_t>(channels) * size * size, 0.0f);
  const size_t hw = static_cast<size_t>(size) * size;
  for (int c = 0; c < channels; ++c) {
    for (size_t i = 0; i < hw; ++i) {
      input[static_cast<size_t>(c) * hw + i] = static_cast<float>(image[i]);
    }
  }
  return input;
}