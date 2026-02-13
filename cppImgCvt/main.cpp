#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "npz_imgcvt_utils.h"

namespace fs = std::filesystem;

namespace {

struct Args {
    std::string mode;
    fs::path input;
    fs::path output;
    std::string npz_key = "image";
    int slice_index = -1;
};

Args parse_args(int argc, char** argv) {
    std::map<std::string, std::string> kv;
    for (int i = 1; i + 1 < argc; i += 2) {
        kv[argv[i]] = argv[i + 1];
    }

    Args args;
    if (kv.count("--mode")) args.mode = kv["--mode"];
    if (kv.count("--input")) args.input = kv["--input"];
    if (kv.count("--output")) args.output = kv["--output"];
    if (kv.count("--npz-key")) args.npz_key = kv["--npz-key"];
    if (kv.count("--slice-index")) args.slice_index = std::stoi(kv["--slice-index"]);

    if (args.mode.empty() || args.input.empty() || args.output.empty()) {
        throw std::runtime_error(
            "用法: ./imgcvt --mode <dcm2npz|nii2npz|png2npz|npz2dcm|npz2nii|npz2png> --input <path> --output <path> [--npz-key image] [--slice-index n]");
    }
    return args;
}

uint16_t read_u16_le(const std::vector<uint8_t>& b, size_t off) {
    return static_cast<uint16_t>(b[off]) | (static_cast<uint16_t>(b[off + 1]) << 8);
}

uint32_t read_u32_le(const std::vector<uint8_t>& b, size_t off) {
    return static_cast<uint32_t>(b[off]) | (static_cast<uint32_t>(b[off + 1]) << 8) |
           (static_cast<uint32_t>(b[off + 2]) << 16) | (static_cast<uint32_t>(b[off + 3]) << 24);
}

std::vector<uint8_t> read_tag_value_explicit_vr(const std::vector<uint8_t>& dcm,
                                                uint16_t target_group,
                                                uint16_t target_elem,
                                                bool* found = nullptr,
                                                size_t* value_offset_out = nullptr) {
    if (found != nullptr) *found = false;
    size_t off = 132;
    while (off + 8 <= dcm.size()) {
        const uint16_t group = read_u16_le(dcm, off);
        const uint16_t elem = read_u16_le(dcm, off + 2);
        const char vr0 = static_cast<char>(dcm[off + 4]);
        const char vr1 = static_cast<char>(dcm[off + 5]);
        const std::string vr{vr0, vr1};

        bool long_vr = (vr == "OB" || vr == "OW" || vr == "OF" || vr == "SQ" || vr == "UT" || vr == "UN");
        uint32_t len = 0;
        size_t value_off = 0;
        if (long_vr) {
            if (off + 12 > dcm.size()) break;
            len = read_u32_le(dcm, off + 8);
            value_off = off + 12;
            off += 12;
        } else {
            len = read_u16_le(dcm, off + 6);
            value_off = off + 8;
            off += 8;
        }
        if (value_off + len > dcm.size()) break;

        if (group == target_group && elem == target_elem) {
            if (found != nullptr) *found = true;
            if (value_offset_out != nullptr) *value_offset_out = value_off;
            return std::vector<uint8_t>(dcm.begin() + static_cast<long>(value_off),
                                        dcm.begin() + static_cast<long>(value_off + len));
        }
        off = value_off + len;
    }
    return {};
}

void dcm_to_npz(const fs::path& input_path, const fs::path& out_path) {
    const auto dcm_bytes = imgcvt::read_file_bytes(input_path);
    std::vector<uint8_t> embedded_npz;
    if (imgcvt::try_extract_embedded_npz_from_bytes(dcm_bytes, &embedded_npz)) {
        imgcvt::write_file_bytes(out_path, embedded_npz);
        return;
    }

    bool ok_rows = false;
    bool ok_cols = false;
    bool ok_pixel = false;
    const auto rows_buf = read_tag_value_explicit_vr(dcm_bytes, 0x0028, 0x0010, &ok_rows);
    const auto cols_buf = read_tag_value_explicit_vr(dcm_bytes, 0x0028, 0x0011, &ok_cols);
    const auto pixel_buf = read_tag_value_explicit_vr(dcm_bytes, 0x7FE0, 0x0010, &ok_pixel);

    if (!ok_rows || !ok_cols || !ok_pixel || rows_buf.size() < 2 || cols_buf.size() < 2) {
        throw std::runtime_error("无法从 DICOM 读取像素，也未找到嵌入的 NPZ 载荷");
    }

    const uint16_t rows = static_cast<uint16_t>(rows_buf[0]) | (static_cast<uint16_t>(rows_buf[1]) << 8);
    const uint16_t cols = static_cast<uint16_t>(cols_buf[0]) | (static_cast<uint16_t>(cols_buf[1]) << 8);
    const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    if (pixel_buf.size() < n * sizeof(uint16_t)) {
        throw std::runtime_error("DICOM PixelData 长度不足");
    }

    std::vector<uint16_t> image(n, 0);
    std::memcpy(image.data(), pixel_buf.data(), n * sizeof(uint16_t));
    std::vector<uint8_t> label(n, 0);
    const std::vector<size_t> shape{rows, cols};
    cnpy::npz_save(out_path.string(), "image", image.data(), shape, "w");
    cnpy::npz_save(out_path.string(), "label", label.data(), shape, "a");
}

void npz_to_dcm(const fs::path& input_path, const fs::path& out_path, const std::string& key) {
    const auto npz_map = cnpy::npz_load(input_path.string());
    auto it = npz_map.find(key);
    if (it == npz_map.end()) {
        throw std::runtime_error("npz 中找不到键: " + key);
    }
    const auto shape = imgcvt::require_shape_2d(it->second);
    const auto image_f64 = imgcvt::to_double_2d(it->second);
    const auto image_u16 = imgcvt::to_uint16_clipped(image_f64);

    const uint16_t rows = static_cast<uint16_t>(shape[0]);
    const uint16_t cols = static_cast<uint16_t>(shape[1]);

    const auto npz_bytes = imgcvt::read_file_bytes(input_path);
    const auto packed_npz = imgcvt::pack_embedded_npz(npz_bytes);

    std::vector<uint8_t> out(128, 0);
    out.push_back('D');
    out.push_back('I');
    out.push_back('C');
    out.push_back('M');

    std::vector<uint8_t> file_meta;
    const std::string sop_class = "1.2.840.10008.5.1.4.1.1.7";
    const std::string sop_instance = imgcvt::uid_like();
    const std::string transfer_syntax = "1.2.840.10008.1.2.1";
    const std::string impl_uid = imgcvt::uid_like();

    imgcvt::append_tag(file_meta, 0x0002, 0x0001, "OB", {0x00, 0x01});
    imgcvt::append_tag(file_meta, 0x0002, 0x0002, "UI", imgcvt::str_bytes(sop_class));
    imgcvt::append_tag(file_meta, 0x0002, 0x0003, "UI", imgcvt::str_bytes(sop_instance));
    imgcvt::append_tag(file_meta, 0x0002, 0x0010, "UI", imgcvt::str_bytes(transfer_syntax));
    imgcvt::append_tag(file_meta, 0x0002, 0x0012, "UI", imgcvt::str_bytes(impl_uid));

    imgcvt::append_tag(out, 0x0002, 0x0000, "UL",
                       std::vector<uint8_t>{
                           static_cast<uint8_t>(file_meta.size() & 0xFF),
                           static_cast<uint8_t>((file_meta.size() >> 8) & 0xFF),
                           static_cast<uint8_t>((file_meta.size() >> 16) & 0xFF),
                           static_cast<uint8_t>((file_meta.size() >> 24) & 0xFF),
                       });
    out.insert(out.end(), file_meta.begin(), file_meta.end());

    const auto now = std::chrono::system_clock::now();
    (void)now;

    imgcvt::append_tag(out, 0x0008, 0x0060, "CS", imgcvt::str_bytes("OT"));
    imgcvt::append_tag(out, 0x0010, 0x0010, "PN", imgcvt::str_bytes("Converted^FromNPZ"));
    imgcvt::append_tag(out, 0x0010, 0x0020, "LO", imgcvt::str_bytes("NPZ0001"));
    imgcvt::append_tag(out, 0x0028, 0x0010, "US", imgcvt::u16_bytes(rows));
    imgcvt::append_tag(out, 0x0028, 0x0011, "US", imgcvt::u16_bytes(cols));
    imgcvt::append_tag(out, 0x0028, 0x0002, "US", imgcvt::u16_bytes(1));
    imgcvt::append_tag(out, 0x0028, 0x0004, "CS", imgcvt::str_bytes("MONOCHROME2"));
    imgcvt::append_tag(out, 0x0028, 0x0100, "US", imgcvt::u16_bytes(16));
    imgcvt::append_tag(out, 0x0028, 0x0101, "US", imgcvt::u16_bytes(16));
    imgcvt::append_tag(out, 0x0028, 0x0102, "US", imgcvt::u16_bytes(15));
    imgcvt::append_tag(out, 0x0028, 0x0103, "US", imgcvt::u16_bytes(0));
    imgcvt::append_tag(out, 0x0011, 0x0010, "LO", imgcvt::str_bytes("NPZ_ROUNDTRIP"));
    imgcvt::append_tag(out, 0x0011, 0x1010, "OB", packed_npz);

    std::vector<uint8_t> pixel_bytes(image_u16.size() * sizeof(uint16_t));
    std::memcpy(pixel_bytes.data(), image_u16.data(), pixel_bytes.size());
    imgcvt::append_tag(out, 0x7FE0, 0x0010, "OW", pixel_bytes);

    imgcvt::write_file_bytes(out_path, out);
}

void npz_to_nii(const fs::path& input_path, const fs::path& out_path, const std::string& key) {
    const auto npz_map = cnpy::npz_load(input_path.string());
    auto it = npz_map.find(key);
    if (it == npz_map.end()) {
        throw std::runtime_error("npz 中找不到键: " + key);
    }

    std::vector<size_t> shape = it->second.shape;
    std::vector<float> image_f32;
    if (shape.size() == 2) {
        const auto image_f64 = imgcvt::to_double_2d(it->second);
        image_f32 = imgcvt::to_float32(image_f64);
        shape = {shape[0], shape[1], 1};
    } else if (shape.size() == 3) {
        const size_t n = imgcvt::element_count(shape);
        image_f32.resize(n, 0.0F);
        if (it->second.word_size == sizeof(float)) {
            const auto* p = it->second.data<float>();
            std::copy(p, p + n, image_f32.begin());
        } else if (it->second.word_size == sizeof(double)) {
            const auto* p = it->second.data<double>();
            for (size_t i = 0; i < n; ++i) image_f32[i] = static_cast<float>(p[i]);
        } else {
            throw std::runtime_error("3D NIfTI 仅支持 float32/float64 输入");
        }
    } else {
        throw std::runtime_error("仅支持 2D/3D 写入 NIfTI");
    }

    const auto npz_bytes = imgcvt::read_file_bytes(input_path);
    const auto packed_npz = imgcvt::pack_embedded_npz(npz_bytes);

    imgcvt::Nifti1Header hdr{};
    hdr.sizeof_hdr = 348;
    hdr.dim[0] = 3;
    hdr.dim[1] = static_cast<int16_t>(shape[0]);
    hdr.dim[2] = static_cast<int16_t>(shape[1]);
    hdr.dim[3] = static_cast<int16_t>(shape[2]);
    hdr.datatype = 16;
    hdr.bitpix = 32;
    hdr.pixdim[1] = 1.0F;
    hdr.pixdim[2] = 1.0F;
    hdr.pixdim[3] = 1.0F;

    int32_t ext_size = static_cast<int32_t>(8 + packed_npz.size());
    const int32_t rem = ext_size % 16;
    if (rem != 0) ext_size += (16 - rem);
    hdr.vox_offset = static_cast<float>(352 + ext_size);
    std::strncpy(hdr.descrip, "ConvertedFromNPZ", sizeof(hdr.descrip) - 1);
    hdr.sform_code = 1;
    hdr.srow_x[0] = 1.0F;
    hdr.srow_y[1] = 1.0F;
    hdr.srow_z[2] = 1.0F;
    hdr.magic[0] = 'n';
    hdr.magic[1] = '+';
    hdr.magic[2] = '1';
    hdr.magic[3] = '\0';

    std::vector<uint8_t> out;
    out.resize(348);
    std::memcpy(out.data(), &hdr, 348);

    out.push_back(1);
    out.push_back(0);
    out.push_back(0);
    out.push_back(0);

    imgcvt::append_u32_le(out, static_cast<uint32_t>(ext_size));
    imgcvt::append_u32_le(out, 40);
    out.insert(out.end(), packed_npz.begin(), packed_npz.end());
    while ((out.size() - 352) % 16 != 0) {
        out.push_back(0);
    }

    const auto data_offset = static_cast<size_t>(hdr.vox_offset);
    if (out.size() < data_offset) {
        out.resize(data_offset, 0);
    }
    const size_t data_bytes = image_f32.size() * sizeof(float);
    const size_t base = out.size();
    out.resize(base + data_bytes);
    std::memcpy(out.data() + static_cast<long>(base), image_f32.data(), data_bytes);

    imgcvt::write_file_bytes(out_path, out);
}

void nii_to_npz(const fs::path& input_path, const fs::path& out_path, int slice_index) {
    const auto all = imgcvt::read_file_bytes(input_path);
    if (all.size() < 352) {
        throw std::runtime_error("NIfTI 文件过小");
    }

    std::vector<uint8_t> embedded_npz;
    if (imgcvt::try_extract_embedded_npz_from_bytes(all, &embedded_npz)) {
        imgcvt::write_file_bytes(out_path, embedded_npz);
        return;
    }

    imgcvt::Nifti1Header hdr{};
    std::memcpy(&hdr, all.data(), 348);
    if (hdr.sizeof_hdr != 348) {
        throw std::runtime_error("不支持的 NIfTI 头部");
    }

    const int ndim = hdr.dim[0];
    const int d1 = std::max<int>(1, hdr.dim[1]);
    const int d2 = std::max<int>(1, hdr.dim[2]);
    const int d3 = std::max<int>(1, hdr.dim[3]);

    if (ndim < 2) {
        throw std::runtime_error("NIfTI 维度不足");
    }

    const size_t vox_offset = static_cast<size_t>(hdr.vox_offset);
    if (vox_offset >= all.size()) {
        throw std::runtime_error("NIfTI vox_offset 越界");
    }

    const size_t n = static_cast<size_t>(d1) * static_cast<size_t>(d2) * static_cast<size_t>(d3);
    std::vector<double> volume(n, 0.0);

    if (hdr.datatype == 16 && hdr.bitpix == 32) {
        const size_t need = n * sizeof(float);
        if (vox_offset + need > all.size()) throw std::runtime_error("NIfTI 数据长度不足");
        const auto* p = reinterpret_cast<const float*>(all.data() + static_cast<long>(vox_offset));
        for (size_t i = 0; i < n; ++i) volume[i] = static_cast<double>(p[i]);
    } else if (hdr.datatype == 64 && hdr.bitpix == 64) {
        const size_t need = n * sizeof(double);
        if (vox_offset + need > all.size()) throw std::runtime_error("NIfTI 数据长度不足");
        const auto* p = reinterpret_cast<const double*>(all.data() + static_cast<long>(vox_offset));
        std::copy(p, p + n, volume.begin());
    } else if (hdr.datatype == 512 && hdr.bitpix == 16) {
        const size_t need = n * sizeof(uint16_t);
        if (vox_offset + need > all.size()) throw std::runtime_error("NIfTI 数据长度不足");
        const auto* p = reinterpret_cast<const uint16_t*>(all.data() + static_cast<long>(vox_offset));
        for (size_t i = 0; i < n; ++i) volume[i] = static_cast<double>(p[i]);
    } else {
        throw std::runtime_error("当前仅支持读取 float32/float64/uint16 的 NIfTI");
    }

    const int use_slice = (d3 == 1) ? 0 : (slice_index >= 0 ? slice_index : (d3 / 2));
    if (use_slice < 0 || use_slice >= d3) {
        throw std::runtime_error("slice_index 越界");
    }

    const size_t hw = static_cast<size_t>(d1) * static_cast<size_t>(d2);
    std::vector<double> image(hw, 0.0);
    const size_t z_off = static_cast<size_t>(use_slice) * hw;
    std::copy(volume.begin() + static_cast<long>(z_off),
              volume.begin() + static_cast<long>(z_off + hw),
              image.begin());

    imgcvt::save_onnx_compatible_npz(out_path, {static_cast<size_t>(d1), static_cast<size_t>(d2)}, image);
}

std::vector<fs::path> sorted_pngs(const fs::path& input_dir) {
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.is_regular_file()) {
            const auto ext = entry.path().extension().string();
            if (ext == ".png" || ext == ".PNG") {
                files.push_back(entry.path());
            }
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

void png_to_npz(const fs::path& input_path, const fs::path& out_path) {
    fs::path source_png = input_path;
    if (fs::is_directory(input_path)) {
        const auto files = sorted_pngs(input_path);
        if (files.empty()) {
            throw std::runtime_error("目录下没有 png 文件");
        }
        source_png = files[files.size() / 2];
    }

    if (!fs::exists(source_png)) {
        throw std::runtime_error("png 文件不存在: " + source_png.string());
    }

    const fs::path sidecar = fs::path(source_png.string() + imgcvt::kPngSidecarSuffix);
    if (fs::exists(sidecar)) {
        const auto payload = imgcvt::read_file_bytes(sidecar);
        std::vector<uint8_t> embedded;
        if (imgcvt::unpack_embedded_npz(payload, &embedded)) {
            imgcvt::write_file_bytes(out_path, embedded);
            return;
        }
    }

    const cv::Mat gray = cv::imread(source_png.string(), cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
        throw std::runtime_error("读取 png 失败: " + source_png.string());
    }
    const auto image = imgcvt::image_from_gray_u8(gray);
    imgcvt::save_onnx_compatible_npz(out_path,
                                     {static_cast<size_t>(gray.rows), static_cast<size_t>(gray.cols)},
                                     image);
}

void npz_to_png(const fs::path& input_path, const fs::path& out_path, const std::string& key) {
    const auto npz_map = cnpy::npz_load(input_path.string());
    auto it = npz_map.find(key);
    if (it == npz_map.end()) {
        throw std::runtime_error("npz 中找不到键: " + key);
    }
    const cv::Mat image = imgcvt::grayscale_u8_from_image_2d(it->second);
    if (!cv::imwrite(out_path.string(), image)) {
        throw std::runtime_error("写入 png 失败: " + out_path.string());
    }

    const auto npz_bytes = imgcvt::read_file_bytes(input_path);
    const auto packed = imgcvt::pack_embedded_npz(npz_bytes);
    const fs::path sidecar = fs::path(out_path.string() + imgcvt::kPngSidecarSuffix);
    imgcvt::write_file_bytes(sidecar, packed);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        if (!args.output.parent_path().empty()) {
            fs::create_directories(args.output.parent_path());
        }

        if (args.mode == "dcm2npz") {
            dcm_to_npz(args.input, args.output);
        } else if (args.mode == "nii2npz") {
            nii_to_npz(args.input, args.output, args.slice_index);
        } else if (args.mode == "png2npz") {
            png_to_npz(args.input, args.output);
        } else if (args.mode == "npz2dcm") {
            npz_to_dcm(args.input, args.output, args.npz_key);
        } else if (args.mode == "npz2nii") {
            npz_to_nii(args.input, args.output, args.npz_key);
        } else if (args.mode == "npz2png") {
            npz_to_png(args.input, args.output, args.npz_key);
        } else {
            throw std::runtime_error("不支持的 mode: " + args.mode);
        }

        std::cout << "转换完成: " << args.mode << " -> " << args.output << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }
}
