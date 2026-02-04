#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "cnpy.h"

static cv::Mat npyToMat(const cnpy::NpyArray& arr) {
    if (arr.word_size != sizeof(double) || arr.shape.size() != 2) {
        throw std::runtime_error("Only 2D float64 supported");
    }
    const size_t h = arr.shape[0];
    const size_t w = arr.shape[1];
    const double* data = arr.data<double>();

    cv::Mat img(static_cast<int>(h), static_cast<int>(w), CV_8UC1);

    // Fortran order means column-major
    if (arr.fortran_order) {
        for (size_t r = 0; r < h; ++r) {
            for (size_t c = 0; c < w; ++c) {
                double v = data[c * h + r];
                v = std::min(1.0, std::max(0.0, v));
                img.at<uchar>(static_cast<int>(r), static_cast<int>(c)) =
                    static_cast<uchar>(v * 255.0 + 0.5);
            }
        }
    } else {
        for (size_t r = 0; r < h; ++r) {
            for (size_t c = 0; c < w; ++c) {
                double v = data[r * w + c];
                v = std::min(1.0, std::max(0.0, v));
                img.at<uchar>(static_cast<int>(r), static_cast<int>(c)) =
                    static_cast<uchar>(v * 255.0 + 0.5);
            }
        }
    }
    return img;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: npz2png <file.npz>\n";
        return 1;
    }

    const std::string npzPath = argv[1];
    std::cerr << "Loading npz: " << npzPath << "\n";
    std::filesystem::path p(npzPath);
    std::string stem = p.stem().string(); // filename without extension
    std::string dir = p.parent_path().string();

    try {
        cnpy::npz_t npz = cnpy::npz_load(npzPath);
        std::cerr << "Loaded npz. Arrays: " << npz.size() << "\n";
        if (npz.empty()) {
            std::cerr << "No arrays in npz\n";
            return 1;
        }

        for (const auto& kv : npz) {
            const std::string& name = kv.first; // npy name
            const cnpy::NpyArray& arr = kv.second;
            std::cerr << "Converting: " << name << "\n";

            cv::Mat img = npyToMat(arr);

            std::string outName = stem + "_" + name + ".png";
            std::filesystem::path outPath = dir.empty()
                ? std::filesystem::path(outName)
                : (std::filesystem::path(dir) / outName);
            if (!cv::imwrite(outPath.string(), img)) {
                std::cerr << "Failed to write: " << outPath << "\n";
                return 1;
            }
            std::cout << "Wrote: " << outPath << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}