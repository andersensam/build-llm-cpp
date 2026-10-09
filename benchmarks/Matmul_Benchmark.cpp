/* Large Language Model training framework in C++
 * Copyright © 2026 Samuel Andersen
 *  ________   ___   __    ______   ______   ______    ______   ______   ___   __    ______   ________   ___ __ __     
 * /_______/\ /__/\ /__/\ /_____/\ /_____/\ /_____/\  /_____/\ /_____/\ /__/\ /__/\ /_____/\ /_______/\ /__//_//_/\    
 * \::: _  \ \\::\_\\  \ \\:::_ \ \\::::_\/_\:::_ \ \ \::::_\/_\::::_\/_\::\_\\  \ \\::::_\/_\::: _  \ \\::\| \| \ \   
 *  \::(_)  \ \\:. `-\  \ \\:\ \ \ \\:\/___/\\:(_) ) )_\:\/___/\\:\/___/\\:. `-\  \ \\:\/___/\\::(_)  \ \\:.      \ \  
 *   \:: __  \ \\:. _    \ \\:\ \ \ \\::___\/_\: __ `\ \\_::._\:\\::___\/_\:. _    \ \\_::._\:\\:: __  \ \\:.\-/\  \ \ 
 *    \:.\ \  \ \\. \`-\  \ \\:\/.:| |\:\____/\\ \ `\ \ \ /____\:\\:\____/\\. \`-\  \ \ /____\:\\:.\ \  \ \\. \  \  \ \
 *     \__\/\__\/ \__\/ \__\/ \____/_/ \_____\/ \_\/ \_\/ \_____\/ \_____\/ \__\/ \__\/ \_____\/ \__\/\__\/ \__\/ \__\/    
 *                                                                                                               
 * Project: Large Language Model in C++
 * @author : Samuel Andersen
 * @version: 2026-10-07
 *
 * Notes:
 * Matmul_Benchmark.cpp: Used for benchmarking matmul across different data types and generations of
 * matmul implementations.
 *
 * TODO: Continue adding functionality 
 */

/* Standard dependencies */
#include <cstdint>
#include <cstddef>
#include <exception>
#include <format>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

/* Local dependencies */
#include "ListTensorSlice.hpp"
#include "Log.hpp"
#include "SliceConfig.hpp"
#include "Tensor.hpp"
#include "TensorMatmul.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers, cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
int main() {

    /* Use logging */
    using Log_NS::Log_Priority;
    using Log_NS::log_message;

    /* Use Tensor */
    using Tensor_NS::Tensor;

    /* Use matmul */
    using TensorMatmul_NS::matmul;

    /* Use ListTensorSlice and required utils */
    using SliceConfig_NS::MatrixSliceConfig;
    using SliceConfig_NS::IndexType;
    using ListTensorSlice_NS::ListTensorSlice;

    try {
        log_message(Log_Priority::INFO, "main", "Initializing matmul benchmark");

        // Test matmul with int16_t
        Tensor<int16_t> int16_t0({256, 512});
        int16_t0.random(-10, 10);
        Tensor<int16_t> int16_t1({512, 256});
        int16_t1.fill(2);
        Tensor<int16_t> int16_mm_result({256, 256});
        log_message(Log_Priority::INFO, "main", "Starting int16_t matmul [256, 512] @ [512, 256] --> [256, 256]");
        TensorMatmul_NS::matmul(int16_t0, int16_t1, int16_mm_result);
        log_message(Log_Priority::INFO, "main", "Finished int16_t matmul");

        // Test matmul with int64_t
        Tensor<int64_t> int64_t0({256, 512});
        int64_t0.random(-256, 256);
        Tensor<int64_t> int64_t1({512, 256});
        int64_t1.fill(4);
        Tensor<int64_t> int64_mm_result({256, 256});
        log_message(Log_Priority::INFO, "main", "Starting int64_t matmul [256, 512] @ [512, 256] --> [256, 256]");
        TensorMatmul_NS::matmul(int64_t0, int64_t1, int64_mm_result);
        log_message(Log_Priority::INFO, "main", "Finished int64_t matmul");

        // Test matmul with uint16_t
        Tensor<uint16_t> uint16_t0({256, 512});
        uint16_t0.random(0, 4);
        Tensor<uint16_t> uint16_t1({512, 256});
        uint16_t1.fill(2);
        Tensor<uint16_t> uint16_mm_result({256, 256});
        log_message(Log_Priority::INFO, "main", "Starting uint16_t matmul [256, 512] @ [512, 256] --> [256, 256]");
        TensorMatmul_NS::matmul(uint16_t0, uint16_t1, uint16_mm_result);
        log_message(Log_Priority::INFO, "main", "Finished uint16_t matmul");

        // Test matmul with float
        Tensor<float> float_t0({256, 512});
        float_t0.random(-10, 10);
        Tensor<float> float_t1({512, 256});
        float_t1.random(-2, 2);
        Tensor<float> float_mm_result({256, 256});
        log_message(Log_Priority::INFO, "main", "Starting float matmul [256, 512] @ [512, 256] --> [256, 256]");
        TensorMatmul_NS::matmul(float_t0, float_t1, float_mm_result);
        log_message(Log_Priority::INFO, "main", "Finished float matmul");

        // Test matmul with self + transpose
        log_message(Log_Priority::INFO, "main", "Starting transposed float matmul [256, 512] @ T --> [256, 256]");
        TensorMatmul_NS::matmul(float_t0, 0, 1, float_t0, 1, 0, float_mm_result);
        log_message(Log_Priority::INFO, "main", "Finished transposed float matmul");

        // Test using ListTensorSlice
        auto f_ptr = std::make_shared<Tensor<float>>(float_t0.shape());
        std::vector<size_t> idx0(128);
        std::vector<size_t> idx1(64);
        for (size_t i = 0; i < 128; ++i) {
            idx0[i] = i;
        }
        for (size_t i = 0; i < 64; ++i) {
            idx1[i] = i;
        }
        // Creates [128, 512]
        MatrixSliceConfig msc0(0, IndexType::LIST, idx0, 1, {}, {});
        // Creates [64, 512]
        MatrixSliceConfig msc1(0, IndexType::LIST, idx1, 1, {}, {});
        ListTensorSlice<float> lts0(f_ptr, msc0);
        ListTensorSlice<float> lts1(f_ptr, msc1);
        Tensor<float> lts_mm_result({128, 64});
        log_message(Log_Priority::INFO, "main", "Starting ListTensorSlice<float> matmul [128, 512] @ [64, 512]^T --> [128, 64]");
        TensorMatmul_NS::matmul(lts0, 0, 1, lts1, 1, 0, lts_mm_result);
        log_message(Log_Priority::INFO, "main", "Finished ListTensorSlice<float> float matmul");

        // Test using a normal Tensor slice
        Tensor<double> double_t0({128, 256});
        double_t0.random(-100, 100);
        Tensor<double> double_mm_result({256, 256});
        log_message(Log_Priority::INFO, "main", "Starting double matmul [128, 256]^T @ [128, 256] --> [256, 256]");
        TensorMatmul_NS::matmul(double_t0, 1, 0, double_t0, 0, 1, double_mm_result);
        log_message(Log_Priority::INFO, "main", "Finished double matmul");

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return -1;
    }

    return 0;
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers, cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
