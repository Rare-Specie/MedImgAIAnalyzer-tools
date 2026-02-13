#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../cnpy.h"

namespace imgcvt {

inline constexpr const char* kNpzEmbedMagic = "NPZ_ROUNDTRIP_V1\0";
inline constexpr size_t kNpzEmbedMagicSize = 17;
inline constexpr const char* kPngSidecarSuffix = ".npzembed";

inline std::vector<uint8_t> read_file_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("无法打开文件: " + path.string());
    }
    input.seekg(0, std::ios::end);
    const auto size = static_cast<size_t>(input.tellg());
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(size);
    if (size > 0) {
        input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
        if (!input) {
            throw std::runtime_error("读取文件失败: " + path.string());
        }
    }
    return data;
}

inline void write_file_bytes(const std::filesystem::path& path, const std::vector<uint8_t>& data) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("无法写入文件: " + path.string());
    }
    if (!data.empty()) {
        output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        if (!output) {
            throw std::runtime_error("写入文件失败: " + path.string());
        }
    }
}

inline std::string base64_encode(const std::vector<uint8_t>& input) {
    static constexpr char kTable[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((input.size() + 2) / 3) * 4);

    size_t i = 0;
    while (i + 3 <= input.size()) {
        const uint32_t v = (static_cast<uint32_t>(input[i]) << 16) |
                           (static_cast<uint32_t>(input[i + 1]) << 8) |
                           static_cast<uint32_t>(input[i + 2]);
        out.push_back(kTable[(v >> 18) & 0x3F]);
        out.push_back(kTable[(v >> 12) & 0x3F]);
        out.push_back(kTable[(v >> 6) & 0x3F]);
        out.push_back(kTable[v & 0x3F]);
        i += 3;
    }

    if (i < input.size()) {
        uint32_t v = static_cast<uint32_t>(input[i]) << 16;
        out.push_back(kTable[(v >> 18) & 0x3F]);
        if (i + 1 < input.size()) {
            v |= static_cast<uint32_t>(input[i + 1]) << 8;
            out.push_back(kTable[(v >> 12) & 0x3F]);
            out.push_back(kTable[(v >> 6) & 0x3F]);
            out.push_back('=');
        } else {
            out.push_back(kTable[(v >> 12) & 0x3F]);
            out.push_back('=');
            out.push_back('=');
        }
    }

    return out;
}

inline std::vector<uint8_t> base64_decode(const std::string& input) {
    static std::array<int, 256> decode_table = [] {
        std::array<int, 256> table{};
        table.fill(-1);
        const std::string chars =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (size_t i = 0; i < chars.size(); ++i) {
            table[static_cast<uint8_t>(chars[i])] = static_cast<int>(i);
        }
        return table;
    }();

    std::vector<uint8_t> out;
    uint32_t accum = 0;
    int bits = 0;

    for (char c : input) {
        if (std::isspace(static_cast<unsigned char>(c)) != 0) {
            continue;
        }
        if (c == '=') {
            break;
        }
        const int value = decode_table[static_cast<uint8_t>(c)];
        if (value < 0) {
            throw std::runtime_error("base64 解码失败: 非法字符");
        }
        accum = (accum << 6) | static_cast<uint32_t>(value);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((accum >> bits) & 0xFF));
        }
    }

    return out;
}

inline std::vector<uint8_t> pack_embedded_npz(const std::vector<uint8_t>& npz_bytes) {
    const std::string b64 = base64_encode(npz_bytes);
    std::vector<uint8_t> out;
    out.reserve(kNpzEmbedMagicSize + 8 + b64.size());
    out.insert(out.end(), kNpzEmbedMagic, kNpzEmbedMagic + kNpzEmbedMagicSize);

    uint64_t len = static_cast<uint64_t>(b64.size());
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<uint8_t>((len >> (8 * i)) & 0xFF));
    }
    out.insert(out.end(), b64.begin(), b64.end());
    return out;
}

inline bool unpack_embedded_npz(const std::vector<uint8_t>& payload, std::vector<uint8_t>* npz_bytes) {
    if (payload.size() < kNpzEmbedMagicSize + 8) {
        return false;
    }
    if (std::memcmp(payload.data(), kNpzEmbedMagic, kNpzEmbedMagicSize) != 0) {
        return false;
    }
    uint64_t len = 0;
    for (int i = 0; i < 8; ++i) {
        len |= (static_cast<uint64_t>(payload[kNpzEmbedMagicSize + i]) << (8 * i));
    }
    const size_t start = kNpzEmbedMagicSize + 8;
    if (payload.size() < start + static_cast<size_t>(len)) {
        return false;
    }
    const std::string b64(reinterpret_cast<const char*>(payload.data() + start),
                          static_cast<size_t>(len));
    *npz_bytes = base64_decode(b64);
    return true;
}

inline bool try_extract_embedded_npz_from_bytes(const std::vector<uint8_t>& all_bytes,
                                                std::vector<uint8_t>* npz_bytes) {
    for (size_t i = 0; i + kNpzEmbedMagicSize + 8 <= all_bytes.size(); ++i) {
        if (std::memcmp(all_bytes.data() + i, kNpzEmbedMagic, kNpzEmbedMagicSize) == 0) {
            std::vector<uint8_t> slice(all_bytes.begin() + static_cast<long>(i), all_bytes.end());
            if (unpack_embedded_npz(slice, npz_bytes)) {
                return true;
            }
        }
    }
    return false;
}

inline std::vector<size_t> require_shape_2d(const cnpy::NpyArray& arr) {
    if (arr.shape.size() != 2) {
        throw std::runtime_error("仅支持二维数组，当前维度=" + std::to_string(arr.shape.size()));
    }
    return arr.shape;
}

inline std::vector<double> to_double_2d(const cnpy::NpyArray& arr) {
    const auto shape = require_shape_2d(arr);
    const size_t n = shape[0] * shape[1];
    std::vector<double> out(n, 0.0);

    if (arr.word_size == sizeof(double)) {
        const auto* p = arr.data<double>();
        std::copy(p, p + n, out.begin());
    } else if (arr.word_size == sizeof(float)) {
        const auto* p = arr.data<float>();
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<double>(p[i]);
    } else if (arr.word_size == sizeof(uint16_t)) {
        const auto* p = arr.data<uint16_t>();
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<double>(p[i]);
    } else if (arr.word_size == sizeof(int16_t)) {
        const auto* p = arr.data<int16_t>();
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<double>(p[i]);
    } else if (arr.word_size == sizeof(uint8_t)) {
        const auto* p = arr.data<uint8_t>();
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<double>(p[i]);
    } else if (arr.word_size == sizeof(int32_t)) {
        const auto* p = arr.data<int32_t>();
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<double>(p[i]);
    } else {
        throw std::runtime_error("不支持的 image 数据类型字节宽度: " + std::to_string(arr.word_size));
    }

    return out;
}

inline std::vector<uint16_t> to_uint16_clipped(const std::vector<double>& input) {
    std::vector<uint16_t> out(input.size(), 0);
    for (size_t i = 0; i < input.size(); ++i) {
        double v = input[i];
        if (!std::isfinite(v)) {
            v = 0.0;
        }
        v = std::clamp(v, 0.0, 65535.0);
        out[i] = static_cast<uint16_t>(std::llround(v));
    }
    return out;
}

inline std::vector<float> to_float32(const std::vector<double>& input) {
    std::vector<float> out(input.size(), 0.0F);
    for (size_t i = 0; i < input.size(); ++i) {
        out[i] = static_cast<float>(input[i]);
    }
    return out;
}

inline void save_onnx_compatible_npz(const std::filesystem::path& out_path,
                                     const std::vector<size_t>& image_shape,
                                     const std::vector<double>& image_data) {
    if (image_shape.size() != 2) {
        throw std::runtime_error("save_onnx_compatible_npz 仅支持 2D image");
    }
    const size_t n = image_shape[0] * image_shape[1];
    if (image_data.size() != n) {
        throw std::runtime_error("image 数据长度与 shape 不匹配");
    }

    std::vector<uint8_t> label(n, 0);
    cnpy::npz_save(out_path.string(), "image", image_data.data(), image_shape, "w");
    cnpy::npz_save(out_path.string(), "label", label.data(), image_shape, "a");
}

inline cv::Mat grayscale_u8_from_image_2d(const cnpy::NpyArray& arr) {
    const auto shape = require_shape_2d(arr);
    const int rows = static_cast<int>(shape[0]);
    const int cols = static_cast<int>(shape[1]);
    const auto values = to_double_2d(arr);

    cv::Mat out(rows, cols, CV_8UC1);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const double v = std::clamp(values[static_cast<size_t>(r) * shape[1] + static_cast<size_t>(c)],
                                        0.0, 1.0);
            out.at<uint8_t>(r, c) = static_cast<uint8_t>(std::lround(v * 255.0));
        }
    }
    return out;
}

inline std::vector<double> image_from_gray_u8(const cv::Mat& gray) {
    if (gray.type() != CV_8UC1) {
        throw std::runtime_error("png 需要为单通道 8 位图像");
    }
    const int rows = gray.rows;
    const int cols = gray.cols;
    std::vector<double> out(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            out[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)] =
                static_cast<double>(gray.at<uint8_t>(r, c)) / 255.0;
        }
    }
    return out;
}

inline std::string uid_like() {
    static std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<uint64_t> dist(1000000ULL, 999999999ULL);
    return "2.25." + std::to_string(dist(rng));
}

inline void append_u16_le(std::vector<uint8_t>& out, uint16_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}

inline void append_u32_le(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}

inline void append_tag(std::vector<uint8_t>& out,
                       uint16_t group,
                       uint16_t element,
                       const std::string& vr,
                       const std::vector<uint8_t>& value) {
    append_u16_le(out, group);
    append_u16_le(out, element);
    out.push_back(static_cast<uint8_t>(vr[0]));
    out.push_back(static_cast<uint8_t>(vr[1]));

    const bool long_vr = (vr == "OB" || vr == "OW" || vr == "OF" || vr == "SQ" || vr == "UT" || vr == "UN");
    std::vector<uint8_t> val = value;
    if (val.size() % 2 != 0) {
        const uint8_t pad = (vr == "UI" || vr == "LO" || vr == "PN" || vr == "CS" || vr == "DA" || vr == "TM")
                                ? static_cast<uint8_t>(' ')
                                : static_cast<uint8_t>(0);
        val.push_back(pad);
    }

    if (long_vr) {
        out.push_back(0);
        out.push_back(0);
        append_u32_le(out, static_cast<uint32_t>(val.size()));
    } else {
        append_u16_le(out, static_cast<uint16_t>(val.size()));
    }
    out.insert(out.end(), val.begin(), val.end());
}

inline std::vector<uint8_t> str_bytes(const std::string& s) {
    return std::vector<uint8_t>(s.begin(), s.end());
}

inline std::vector<uint8_t> u16_bytes(uint16_t v) {
    std::vector<uint8_t> out;
    append_u16_le(out, v);
    return out;
}

#pragma pack(push, 1)
struct Nifti1Header {
    int32_t sizeof_hdr;
    char data_type[10];
    char db_name[18];
    int32_t extents;
    int16_t session_error;
    char regular;
    char dim_info;
    int16_t dim[8];
    float intent_p1;
    float intent_p2;
    float intent_p3;
    int16_t intent_code;
    int16_t datatype;
    int16_t bitpix;
    int16_t slice_start;
    float pixdim[8];
    float vox_offset;
    float scl_slope;
    float scl_inter;
    int16_t slice_end;
    char slice_code;
    char xyzt_units;
    float cal_max;
    float cal_min;
    float slice_duration;
    float toffset;
    int32_t glmax;
    int32_t glmin;
    char descrip[80];
    char aux_file[24];
    int16_t qform_code;
    int16_t sform_code;
    float quatern_b;
    float quatern_c;
    float quatern_d;
    float qoffset_x;
    float qoffset_y;
    float qoffset_z;
    float srow_x[4];
    float srow_y[4];
    float srow_z[4];
    char intent_name[16];
    char magic[4];
};
#pragma pack(pop)

static_assert(sizeof(Nifti1Header) == 348, "Nifti1Header 大小必须为 348 字节");

inline size_t element_count(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (size_t d : shape) n *= d;
    return n;
}

}  // namespace imgcvt
