#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <onnxruntime/onnxruntime_cxx_api.h>

#include "npz_infer_utils.h"

struct Args {
  std::string input;
  std::string onnx = "/Users/rarespecies/Documents/folder/MedImgAIAnalyzer-tools/U-SAM/u_sam.onnx";
  std::string output;
  int img_size = 224;
  int out_size = 512;
};

static Args ParseArgs(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto require_value = [&](const std::string &opt) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("参数缺少值: " + opt);
      }
      return std::string(argv[++i]);
    };

    if (key == "--input") {
      args.input = require_value(key);
    } else if (key == "--onnx") {
      args.onnx = require_value(key);
    } else if (key == "--output") {
      args.output = require_value(key);
    } else if (key == "--img-size") {
      args.img_size = std::stoi(require_value(key));
    } else if (key == "--out-size") {
      args.out_size = std::stoi(require_value(key));
    } else if (key == "--help" || key == "-h") {
      std::cout
          << "用法: inference_onnx --input <npz> [--onnx <model>] [--output <dir>] "
             "[--img-size 224] [--out-size 512]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("未知参数: " + key);
    }
  }

  if (args.input.empty()) {
    throw std::runtime_error("必须提供--input参数");
  }
  return args;
}

int main(int argc, char **argv) {
  try {
    Args args = ParseArgs(argc, argv);

    if (!std::filesystem::exists(args.onnx)) {
      throw std::runtime_error("未找到ONNX模型: " + args.onnx);
    }

    if (args.img_size != 224) {
      std::cout << "警告: 当前权重仅支持img-size=224，已自动重置为224\n";
      args.img_size = 224;
    }

    std::string out_dir = args.output.empty()
                              ? std::filesystem::path(args.input).parent_path().string()
                              : args.output;
    std::filesystem::create_directories(out_dir);

    NpzImageLabel data = LoadNpzImageLabel(args.input);
    NormalizeImage(data.image);
    std::vector<double> resized = ResizeImage(data.image, data.height, data.width, args.img_size);
    std::vector<float> input_chw = MakeInputTensorCHW(resized, args.img_size, 3);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "u_sam_infer");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::Session session(env, args.onnx.c_str(), opts);

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);

    std::vector<int64_t> input_shape = {1, 3, args.img_size, args.img_size};
    size_t input_tensor_size = input_chw.size();

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_chw.data(), input_tensor_size, input_shape.data(), input_shape.size());

    const char *input_name_cstr = input_name.get();
    size_t output_count = session.GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> output_names;
    std::vector<const char *> output_name_cstrs;
    output_names.reserve(output_count);
    output_name_cstrs.reserve(output_count);
    for (size_t i = 0; i < output_count; ++i) {
      output_names.emplace_back(session.GetOutputNameAllocated(i, allocator));
      output_name_cstrs.push_back(output_names.back().get());
    }

    std::vector<Ort::Value> outputs = session.Run(Ort::RunOptions{nullptr},
                                                  &input_name_cstr,
                                                  &input_tensor,
                                                  1,
                                                  output_name_cstrs.data(),
                                                  output_count);

    if (outputs.empty()) {
      throw std::runtime_error("ONNX输出为空");
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
      auto info = outputs[i].GetTensorTypeAndShapeInfo();
      auto shape = info.GetShape();
      std::cout << "输出" << i << "(" << output_name_cstrs[i] << ") shape=[";
      for (size_t j = 0; j < shape.size(); ++j) {
        std::cout << shape[j] << (j + 1 < shape.size() ? "," : "]\n");
      }
    }

    Ort::Value &out = outputs[0];
    auto out_info = out.GetTensorTypeAndShapeInfo();
    auto out_shape = out_info.GetShape();
    if (out_shape.size() != 4) {
      throw std::runtime_error("ONNX输出维度不符合预期");
    }

    int64_t out_n = out_shape[0];
    int64_t out_c = out_shape[1];
    int64_t out_h = out_shape[2];
    int64_t out_w = out_shape[3];
    if (out_n != 1) {
      throw std::runtime_error("ONNX输出batch不为1");
    }

    auto half_to_float = [](uint16_t h) -> float {
      uint32_t sign = (h & 0x8000u) << 16;
      uint32_t exp = (h & 0x7C00u) >> 10;
      uint32_t mant = (h & 0x03FFu);
      uint32_t f;
      if (exp == 0) {
        if (mant == 0) {
          f = sign;
        } else {
          exp = 1;
          while ((mant & 0x0400u) == 0) {
            mant <<= 1;
            exp--;
          }
          mant &= 0x03FFu;
          exp = exp + (127 - 15);
          f = sign | (exp << 23) | (mant << 13);
        }
      } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (mant << 13);
      } else {
        exp = exp + (127 - 15);
        f = sign | (exp << 23) | (mant << 13);
      }
      float out_f;
      std::memcpy(&out_f, &f, sizeof(float));
      return out_f;
    };

    std::vector<float> out_fallback;
    const float *out_data = nullptr;
    auto out_type = out_info.GetElementType();
    if (out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      out_data = out.GetTensorData<float>();
    } else if (out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      const uint16_t *src = out.GetTensorData<uint16_t>();
      size_t total = static_cast<size_t>(out_n * out_c * out_h * out_w);
      out_fallback.resize(total);
      for (size_t i = 0; i < total; ++i) {
        out_fallback[i] = half_to_float(src[i]);
      }
      out_data = out_fallback.data();
    } else {
      throw std::runtime_error("不支持的ONNX输出数据类型");
    }
    std::vector<int64_t> pred(out_h * out_w, 0);
    float out_min = out_data[0];
    float out_max = out_data[0];
    size_t total_vals = static_cast<size_t>(out_n * out_c * out_h * out_w);
    for (size_t i = 1; i < total_vals; ++i) {
      float v = out_data[i];
      if (v < out_min) out_min = v;
      if (v > out_max) out_max = v;
    }
    std::cout << "ONNX输出范围: min=" << out_min << ", max=" << out_max
              << ", shape=[" << out_n << "," << out_c << "," << out_h << "," << out_w << "]\n";

    if (out_c == 1) {
      bool already_prob = (out_min >= 0.0f && out_max <= 1.0f);
      for (int64_t y = 0; y < out_h; ++y) {
        for (int64_t x = 0; x < out_w; ++x) {
          float v = out_data[y * out_w + x];
          float prob = already_prob ? v : (1.0f / (1.0f + std::exp(-v)));
          pred[y * out_w + x] = (prob >= 0.5f) ? 1 : 0;
        }
      }
    } else {
      for (int64_t y = 0; y < out_h; ++y) {
        for (int64_t x = 0; x < out_w; ++x) {
          int64_t best_c = 0;
          float best_v = out_data[(0 * out_c + 0) * out_h * out_w + y * out_w + x];
          for (int64_t c = 1; c < out_c; ++c) {
            float v = out_data[(0 * out_c + c) * out_h * out_w + y * out_w + x];
            if (v > best_v) {
              best_v = v;
              best_c = c;
            }
          }
          pred[y * out_w + x] = best_c;
        }
      }
    }

    size_t fg_count = 0;
    for (int64_t v : pred) {
      if (v > 0) ++fg_count;
    }
    std::cout << "前景像素数: " << fg_count << "/" << pred.size() << "\n";

    std::vector<int64_t> pred_up =
      ResizeMaskNearestFromInt(pred, static_cast<int>(out_h), static_cast<int>(out_w), args.out_size);

    std::string base = std::filesystem::path(args.input).stem().string();
    std::string out_npz = (std::filesystem::path(out_dir) / ("inference_" + base + "_onnxcpp.npz")).string();
    SaveNpzWithSameKeys(args.input, out_npz, pred_up, args.out_size, args.out_size);

    if (data.label.has_value()) {
      std::vector<double> lab = std::move(*data.label);
      std::vector<double> lab_up = ResizeMaskNearest(lab, data.height, data.width, args.out_size);
      std::vector<int64_t> lab_up_i64(lab_up.size(), 0);
      for (size_t i = 0; i < lab_up.size(); ++i) {
        lab_up_i64[i] = static_cast<int64_t>(std::lround(lab_up[i]));
      }
      auto [dice, iou] = EvalDiceIou(pred_up, lab_up_i64);
      std::cout << "简单评估: dice=" << dice << ", iou=" << iou << "\n";
    }

    std::cout << "推理完成，输出已保存到: " << out_npz << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "错误: " << e.what() << "\n";
    return 1;
  }
}