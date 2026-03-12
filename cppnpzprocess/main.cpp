#include <iostream>
#include <stdexcept>

#include "npz_process_utils.h"

int main(int argc, char** argv) {
    try {
        const auto args = npzproc::parse_args(argc, argv);
        npzproc::process_npz_file(args);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << std::endl;
        return 1;
    }
}