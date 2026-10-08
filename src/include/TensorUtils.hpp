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
 *
 * TensorUtils.hpp: Lesser-used utilities for Tensor operations, such as those used
 * by TensorMatmul, etc., that may now have utility for other libraries, but reduce
 * boilerplate code.
 * 
 */

#ifndef TENSOR_UTILS_HPP
#define TENSOR_UTILS_HPP

/* Standard dependencies */
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

/* Local dependencies */
#include "AbstractTensor.hpp"
#include "Tensor.hpp"

namespace TensorUtils_NS {

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;

/* Use Tensor */
using Tensor_NS::Tensor;

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, bugprone-easily-swappable-parameters)
/**
 * Copy the contents of a high-rank AbstractTensor into a Tensor with dim1 stide == 1
 * @param src Const ref to the source AbstractTensor
 * @param src_dim0 First dim from src to copy
 * @param src_dim1 Second dim from src to copy
 * @param src_coordinates Coordinates of other dims in src to handle, passed by value since this is called from
 * the _naive_matmul_impl_v2 dispatch, where we need to preserve the coordinates for later
 * @param dest Reference to Tensor, with dim1 stride == 1
 * @param dest_dim0 First dim in dest
 * @param dest_dim1 Second dim in dest
 * @param dest_offset Base offset in dest
 */
template <typename T>
requires std::is_arithmetic_v<T>
void copy_high_rank_abstract_to_tensor(const AbstractTensor<T>& src, size_t src_dim0, size_t src_dim1, std::vector<size_t> src_coordinates,
                                       Tensor<T>& dest, size_t dest_dim0, size_t dest_dim1, size_t dest_offset) {
    // Ensure dest has stride == 1 in dim1
    if (dest.dim_stride(dest_dim1) != 1) {
        throw std::invalid_argument("TensorUtils::copy_high_rank_abstract_to_tensor: Destination Tensor does NOT have stide == 1 in dim1.\n");
    }
    // Ensure the shapes are the same
    if (src.extent(src_dim0) != dest.extent(dest_dim0) || src.extent(src_dim1) != dest.extent(dest_dim1)) {
        throw std::invalid_argument("TensorUtils::copy_high_rank_abstract_to_tensor: Incompatible dims.\n");
    }
    // Prepare the coordinates for src_dim0 and src_dim1
    size_t& dim0_c = src_coordinates.at(src_dim0);
    size_t& dim1_c = src_coordinates.at(src_dim1);
    // Get the stride for dest_dim0
    const size_t dest_dim0_stride = dest.dim_stride(dest_dim0);
    // Get the data pointer for dest
    T* dest_data = dest._data();
    // Iterate over dim0 and dim1
    for (size_t i = 0; i < src.extent(src_dim0); ++i) {
        // Calculate the row for dest
        const size_t dest_row = (dest_dim0_stride * i) + dest_offset;
        // Update dim0_c to i
        dim0_c = i;
        for (size_t j = 0; j < src.extent(src_dim1); ++j) {
            // Update dim1_c to j
            dim1_c = j;
            // Copy into dest
            dest_data[dest_row + j] = src.at(src_coordinates);
        }
    }
}
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, bugprone-easily-swappable-parameters)

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, bugprone-easily-swappable-parameters)
/**
 * Copy the contents of a Tensor with stride == 1 in dim1 to another high-rank Tensor
 * @param src Const ref to the source Tensor
 * @param src_dim0 First dim from src to copy
 * @param src_dim1 Second dim from src to copy
 * @param src_offset Base offset in src
 * @param dest Reference to destination Tensor
 * @param dest_dim0 First dim in dest
 * @param dest_dim1 Second dim in dest
 * @param dest_coordinates Coordinates of other dims in dest to handle, passed by value since this is called from
 * the _naive_matmul_impl_v2 dispatch, where we need to preserve the coordinates for later
 */
template <typename T>
requires std::is_arithmetic_v<T>
void copy_tensor_to_high_rank_tensor(const Tensor<T>& src, size_t src_dim0, size_t src_dim1, size_t src_offset,
                                     Tensor<T>& dest, size_t dest_dim0, size_t dest_dim1, std::vector<size_t> dest_coordinates) {
    // Ensure dest has stride == 1 in dim1
    if (src.dim_stride(src_dim1) != 1) {
        throw std::invalid_argument("TensorUtils::copy_tensor_to_high_rank_tensor: Source Tensor does NOT have stide == 1 in dim1.\n");
    }
    // Ensure the shapes are the same
    if (src.extent(src_dim0) != dest.extent(dest_dim0) || src.extent(src_dim1) != dest.extent(dest_dim1)) {
        throw std::invalid_argument("TensorUtils::copy_tensor_to_high_rank_tensor: Incompatible dims.\n");
    }
    // Prepare the coordinates for dest_dim0 and dest_dim1
    size_t& dim0_c = dest_coordinates.at(dest_dim0);
    size_t& dim1_c = dest_coordinates.at(dest_dim1);
    // Get the stride for src_dim0
    const size_t src_dim0_stride = src.dim_stride(src_dim0);
    // Get the data pointer for src
    const T* src_data = src._data();
    // Iterate over dim0 and dim1
    for (size_t i = 0; i < dest.extent(dest_dim0); ++i) {
        // Calculate the row for src
        const size_t src_row = (src_dim0_stride * i) + src_offset;
        // Update dim0_c to i
        dim0_c = i;
        for (size_t j = 0; j < dest.extent(dest_dim1); ++j) {
            // Update dim1_c to j
            dim1_c = j;
            // Copy into dest
            dest.at(dest_coordinates) = src_data[src_row + j];
        }
    }
}
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, bugprone-easily-swappable-parameters)

}; // namespace TensorUtils_NS

#endif
