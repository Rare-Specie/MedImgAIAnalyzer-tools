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

inline bool IsValidCrop(int xL, int xR, int yL, int yR, int width, int height) {
  return xL >= 0 && yL >= 0 && xR > xL && yR > yL && xR <= width && yR <= height;
}

template <typename T>
inline std::vector<T> Crop2D(const T *src,
                             int height,
                             int width,
                             int xL,
                             int xR,
                             int yL,
                             int yR,
                             bool fortran_order) {
  int out_w = xR - xL;
  int out_h = yR - yL;
  std::vector<T> out(static_cast<size_t>(out_h) * out_w);
  for (int y = 0; y < out_h; ++y) {
    for (int x = 0; x < out_w; ++x) {
      int src_x = xL + x;
      int src_y = yL + y;
      size_t idx = fortran_order ? static_cast<size_t>(src_x) * height + src_y
                                  : static_cast<size_t>(src_y) * width + src_x;
      out[static_cast<size_t>(y) * out_w + x] = src[idx];
    }
  }
  return out;
}

inline void SaveNpzWithSameKeys(const std::string &src_npz,
                                const std::string &out_npz,
                                const std::vector<int64_t> &pred,
                                int pred_h,
                                int pred_w,
                                const std::string &label_key = "label",
                                int crop_xL = -1,
                                int crop_xR = -1,
                                int crop_yL = -1,
                                int crop_yR = -1) {
  cnpy::npz_t npz = cnpy::npz_load(src_npz);
  bool first = true;
  bool has_valid_crop = false;
  int crop_w = pred_w;
  int crop_h = pred_h;

  auto resolve_crop = [&](int width, int height) {
    if (IsValidCrop(crop_xL, crop_xR, crop_yL, crop_yR, width, height)) {
      has_valid_crop = true;
      crop_w = crop_xR - crop_xL;
      crop_h = crop_yR - crop_yL;
    }
  };
  for (const auto &kv : npz) {
    const std::string &key = kv.first;
    const cnpy::NpyArray &arr = kv.second;
    const std::string mode = first ? "w" : "a";
    first = false;

    if (key == label_key) {
      if (arr.shape.size() != 2) {
        throw std::runtime_error("label应为2D数组");
      }
      resolve_crop(static_cast<int>(arr.shape[1]), static_cast<int>(arr.shape[0]));
      const size_t expected = arr.shape[0] * arr.shape[1];
      if (pred.size() != expected && pred.size() != static_cast<size_t>(pred_h * pred_w)) {
        throw std::runtime_error("pred与label大小不一致");
      }
      std::vector<size_t> shape = {static_cast<size_t>(crop_h), static_cast<size_t>(crop_w)};
      switch (arr.word_size) {
        case sizeof(double): {
          std::vector<double> out;
          if (has_valid_crop) {
            std::vector<double> pred_d(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) pred_d[i] = static_cast<double>(pred[i]);
            out = Crop2D(pred_d.data(), pred_h, pred_w, crop_xL, crop_xR, crop_yL, crop_yR, false);
          } else {
            out.resize(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) out[i] = static_cast<double>(pred[i]);
          }
          cnpy::npz_save(out_npz, key, out.data(), shape, mode);
          break;
        }
        case sizeof(float): {
          std::vector<float> out;
          if (has_valid_crop) {
            std::vector<float> pred_f(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) pred_f[i] = static_cast<float>(pred[i]);
            out = Crop2D(pred_f.data(), pred_h, pred_w, crop_xL, crop_xR, crop_yL, crop_yR, false);
          } else {
            out.resize(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) out[i] = static_cast<float>(pred[i]);
          }
          cnpy::npz_save(out_npz, key, out.data(), shape, mode);
          break;
        }
        case sizeof(uint16_t): {
          std::vector<uint16_t> out;
          if (has_valid_crop) {
            std::vector<uint16_t> pred_u16(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) pred_u16[i] = static_cast<uint16_t>(pred[i]);
            out = Crop2D(pred_u16.data(), pred_h, pred_w, crop_xL, crop_xR, crop_yL, crop_yR, false);
          } else {
            out.resize(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) out[i] = static_cast<uint16_t>(pred[i]);
          }
          cnpy::npz_save(out_npz, key, out.data(), shape, mode);
          break;
        }
        case sizeof(uint8_t): {
          std::vector<uint8_t> out;
          if (has_valid_crop) {
            std::vector<uint8_t> pred_u8(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) pred_u8[i] = static_cast<uint8_t>(pred[i]);
            out = Crop2D(pred_u8.data(), pred_h, pred_w, crop_xL, crop_xR, crop_yL, crop_yR, false);
          } else {
            out.resize(pred.size());
            for (size_t i = 0; i < pred.size(); ++i) out[i] = static_cast<uint8_t>(pred[i]);
          }
          cnpy::npz_save(out_npz, key, out.data(), shape, mode);
          break;
        }
        default:
          throw std::runtime_error("不支持的label数据类型");
      }
      continue;
    }

    if (arr.shape.size() == 2) {
      resolve_crop(static_cast<int>(arr.shape[1]), static_cast<int>(arr.shape[0]));
      int height = static_cast<int>(arr.shape[0]);
      int width = static_cast<int>(arr.shape[1]);
      int out_h = has_valid_crop ? crop_h : height;
      int out_w = has_valid_crop ? crop_w : width;
      std::vector<size_t> shape = {static_cast<size_t>(out_h), static_cast<size_t>(out_w)};
      switch (arr.word_size) {
        case sizeof(double): {
          std::vector<double> out = has_valid_crop
                                        ? Crop2D(arr.data<double>(), height, width, crop_xL, crop_xR, crop_yL, crop_yR,
                                                 arr.fortran_order)
                                        : std::vector<double>(arr.data<double>(), arr.data<double>() + arr.num_vals);
          cnpy::npz_save(out_npz, key, out.data(), shape, mode);
          break;
        }
        case sizeof(float): {
          std::vector<float> out = has_valid_crop
                                       ? Crop2D(arr.data<float>(), height, width, crop_xL, crop_xR, crop_yL, crop_yR,
                                                arr.fortran_order)
                                       : std::vector<float>(arr.data<float>(), arr.data<float>() + arr.num_vals);
          cnpy::npz_save(out_npz, key, out.data(), shape, mode);
          break;
        }
        case sizeof(uint16_t): {
          std::vector<uint16_t> out = has_valid_crop
                                          ? Crop2D(arr.data<uint16_t>(), height, width, crop_xL, crop_xR, crop_yL, crop_yR,
                                                   arr.fortran_order)
                                          : std::vector<uint16_t>(arr.data<uint16_t>(), arr.data<uint16_t>() + arr.num_vals);
          cnpy::npz_save(out_npz, key, out.data(), shape, mode);
          break;
        }
        case sizeof(uint8_t): {
          std::vector<uint8_t> out = has_valid_crop
                                         ? Crop2D(arr.data<uint8_t>(), height, width, crop_xL, crop_xR, crop_yL, crop_yR,
                                                  arr.fortran_order)
                                         : std::vector<uint8_t>(arr.data<uint8_t>(), arr.data<uint8_t>() + arr.num_vals);
          cnpy::npz_save(out_npz, key, out.data(), shape, mode);
          break;
        }
        default:
          throw std::runtime_error("不支持的npz数据类型");
      }
    } else {
      switch (arr.word_size) {
        case sizeof(double):
          if (arr.fortran_order) {
            cnpy::npz_save_fortran(out_npz, key, arr.data<double>(), arr.shape, mode);
          } else {
            cnpy::npz_save(out_npz, key, arr.data<double>(), arr.shape, mode);
          }
          break;
        case sizeof(float):
          if (arr.fortran_order) {
            cnpy::npz_save_fortran(out_npz, key, arr.data<float>(), arr.shape, mode);
          } else {
            cnpy::npz_save(out_npz, key, arr.data<float>(), arr.shape, mode);
          }
          break;
        case sizeof(uint16_t):
          if (arr.fortran_order) {
            cnpy::npz_save_fortran(out_npz, key, arr.data<uint16_t>(), arr.shape, mode);
          } else {
            cnpy::npz_save(out_npz, key, arr.data<uint16_t>(), arr.shape, mode);
          }
          break;
        case sizeof(uint8_t):
          if (arr.fortran_order) {
            cnpy::npz_save_fortran(out_npz, key, arr.data<uint8_t>(), arr.shape, mode);
          } else {
            cnpy::npz_save(out_npz, key, arr.data<uint8_t>(), arr.shape, mode);
          }
          break;
        default:
          throw std::runtime_error("不支持的npz数据类型");
      }
    }
  }

  if (npz.find(label_key) == npz.end()) {
    int out_h = pred_h;
    int out_w = pred_w;
    std::vector<int64_t> out = pred;
    if (IsValidCrop(crop_xL, crop_xR, crop_yL, crop_yR, pred_w, pred_h)) {
      out_h = crop_yR - crop_yL;
      out_w = crop_xR - crop_xL;
      out = Crop2D(pred.data(), pred_h, pred_w, crop_xL, crop_xR, crop_yL, crop_yR, false);
    }
    std::vector<size_t> shape = {static_cast<size_t>(out_h), static_cast<size_t>(out_w)};
    cnpy::npz_save(out_npz, label_key, out.data(), shape, first ? "w" : "a");
  }
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