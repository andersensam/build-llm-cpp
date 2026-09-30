/*  ________   ___   __    ______   ______   ______    ______   ______   ___   __    ______   ________   ___ __ __     
 * /_______/\ /__/\ /__/\ /_____/\ /_____/\ /_____/\  /_____/\ /_____/\ /__/\ /__/\ /_____/\ /_______/\ /__//_//_/\    
 * \::: _  \ \\::\_\\  \ \\:::_ \ \\::::_\/_\:::_ \ \ \::::_\/_\::::_\/_\::\_\\  \ \\::::_\/_\::: _  \ \\::\| \| \ \   
 *  \::(_)  \ \\:. `-\  \ \\:\ \ \ \\:\/___/\\:(_) ) )_\:\/___/\\:\/___/\\:. `-\  \ \\:\/___/\\::(_)  \ \\:.      \ \  
 *   \:: __  \ \\:. _    \ \\:\ \ \ \\::___\/_\: __ `\ \\_::._\:\\::___\/_\:. _    \ \\_::._\:\\:: __  \ \\:.\-/\  \ \ 
 *    \:.\ \  \ \\. \`-\  \ \\:\/.:| |\:\____/\\ \ `\ \ \ /____\:\\:\____/\\. \`-\  \ \ /____\:\\:.\ \  \ \\. \  \  \ \
 *     \__\/\__\/ \__\/ \__\/ \____/_/ \_____\/ \_\/ \_\/ \_____\/ \_____\/ \__\/ \__\/ \_____\/ \__\/\__\/ \__\/ \__\/    
 *                                                                                                               
 * Project: Large Language Model in C++
 * @author : Samuel Andersen
 * @version: 2026-09-30
 *
 * Notes:
 * TensorMatmul.hpp contains kernels for running matrix multiplications on Tensors and TensorSlices
 *
 * TODO: Continue adding functionality 
 */

#ifndef TENSOR_MATMUL_HPP
#define TENSOR_MATMUL_HPP

/* Standard dependencies */
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <stdexcept>
#include <type_traits>

/* Local dependencies */
#include "AbstractTensor.hpp"
#include "Tensor.hpp"
#include "TensorMath.hpp"

namespace Tensor_Matmul_NS {

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;
using Tensor_NS::Tensor;

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
/**
 * Naive matmul implementation
 * @param lhs Lefthand Tensor to matmul
 * @param lhs_dim0 The first dim of lhs for the matmul
 * @param lhs_dim1 The second dim of lhs for the matmul
 * @param lhs_coordinates Reference to a coordinate vector to handle dims of high-rank Tensors
 * @param rhs Righthand Tensor to matmul
 * @param rhs_dim0 The first dim of the rhs for the matmul
 * @param rhs_dim1 The second dim of the rhs for the matmul
 * @param rhs_coordinates Reference to a coordinate vector to handle dims of high-rank Tensors
 * @param destination Reference to the Tensor to put the output into
 * NOTE: We assume this will never be called directly so we skip the additional
 * safety checks you would otherwise need to do (handled in Tensor_Matmul::matmul, etc.)
 */
template <typename T> 
requires std::is_arithmetic_v<T>
void _naive_matmul_impl_v1(const AbstractTensor<T>& lhs, size_t lhs_dim0, size_t lhs_dim1, std::vector<size_t>& lhs_coordinates,
                           const AbstractTensor<T>& rhs, size_t rhs_dim0, size_t rhs_dim1, std::vector<size_t>& rhs_coordinates,
                           Tensor<T>& destination) {
    // Per the NOTE above, skip the regular safety checks and perform the operation
    // Handle overflow detection for floating point types first
    if constexpr (std::is_floating_point_v<T>) {
        bool overflow = false;
        for (size_t i = 0; i < destination.extent(0); ++i) {
            // Update the coordinate for i in lhs
            lhs_coordinates.at(lhs_dim0) = i;
            for (size_t j = 0; j < destination.extent(1); ++j) {
                // Update the coordinate for j in rhs
                rhs_coordinates.at(rhs_dim1) = j;
                // Get the destination reference
                T& dest_ref = destination.at({i, j});
                for (size_t k = 0; k < lhs.extent(lhs_dim1); ++k) {
                    // Update the coordinates for k in lhs and rhs
                    lhs_coordinates.at(lhs_dim1) = k;
                    rhs_coordinates.at(rhs_dim0) = k;
                    // Multiply [i, k] * [k, j]
                    dest_ref += lhs.at(lhs_coordinates) * rhs.at(rhs_coordinates);
                    overflow |= !std::isfinite(dest_ref);
                }
            }
        }
        if (overflow) {
            throw std::overflow_error("Tensor_Matmul::_naive_matmul_impl: Overflow / underflow detected in matmul.\n");
        }
        return;
    }
    else if constexpr (std::is_signed_v<T>) {
        bool overflow = false;
        for (size_t i = 0; i < destination.extent(0); ++i) {
            // Update the coordinate for i in lhs
            lhs_coordinates.at(lhs_dim0) = i;
            for (size_t j = 0; j < destination.extent(1); ++j) {
                // Update the coordinate for j in rhs
                rhs_coordinates.at(rhs_dim1) = j;
                // Get the destination reference
                T& dest_ref = destination.at({i, j});
                T mul_result = 0;
                for (size_t k = 0; k < lhs.extent(lhs_dim1); ++k) {
                    // Update the coordinates for k in lhs and rhs
                    lhs_coordinates.at(lhs_dim1) = k;
                    rhs_coordinates.at(rhs_dim0) = k;
                    // Multiply [i, k] * [k, j]
                    overflow |= TensorMath_NS::_mul_overflow_signed(lhs.at(lhs_coordinates), rhs.at(rhs_coordinates), &mul_result);
                    overflow |= TensorMath_NS::_add_overflow_signed(dest_ref, mul_result, &dest_ref);
                }
            }
        }
        if (overflow) {
            throw std::overflow_error("Tensor_Matmul::_naive_matmul_impl: Overflow / underflow detected in matmul.\n");
        }
        return;
    }
    else {
        bool overflow = false;
        for (size_t i = 0; i < destination.extent(0); ++i) {
            // Update the coordinate for i in lhs
            lhs_coordinates.at(lhs_dim0) = i;
            for (size_t j = 0; j < destination.extent(1); ++j) {
                // Update the coordinate for j in rhs
                rhs_coordinates.at(rhs_dim1) = j;
                // Get the destination reference
                T& dest_ref = destination.at({i, j});
                T mul_result = 0;
                for (size_t k = 0; k < lhs.extent(lhs_dim1); ++k) {
                    // Update the coordinates for k in lhs and rhs
                    lhs_coordinates.at(lhs_dim1) = k;
                    rhs_coordinates.at(rhs_dim0) = k;
                    // Multiply [i, k] * [k, j]
                    overflow |= TensorMath_NS::_mul_overflow_unsigned(lhs.at(lhs_coordinates), rhs.at(rhs_coordinates), &mul_result);
                    overflow |= TensorMath_NS::_add_overflow_unsigned(dest_ref, mul_result, &dest_ref);
                }
            }
        }
        if (overflow) {
            throw std::overflow_error("Tensor_Matmul::_naive_matmul_impl: Overflow / underflow detected in matmul.\n");
        }
        return;
    }
}

/**
 * Naive matmul implementation for Tensors of rank == 2 only
 * @param lhs Lefthand Tensor to matmul
 * @param rhs Righthand Tensor to matmul
 * @param destination Reference to the Tensor to put the output into
 * NOTE: We assume this will never be called directly so we skip the additional
 * safety checks you would otherwise need to do (handled in Tensor_Matmul::matmul, etc.)
 */
template <typename T> 
requires std::is_arithmetic_v<T>
void _naive_matmul_impl_v1(const AbstractTensor<T>& lhs,
                           const AbstractTensor<T>& rhs,
                           Tensor<T>& destination) {
    // Per the NOTE above, skip the regular safety checks and perform the operation
    if constexpr (std::is_floating_point_v<T>) {
        bool overflow = false;
        for (size_t i = 0; i < destination.extent(0); ++i) {
            for (size_t j = 0; j < destination.extent(1); ++j) {
                // Get a pointer to the result's [i, j]
                T* result_i_j = &(destination.at({i, j}));
                for (size_t k = 0; k < lhs.extent(1); ++k) {
                    *result_i_j += lhs.at({i, k}) * rhs.at({k, j});
                    overflow |= !std::isfinite(*result_i_j);
                }
            }
        }
        if (overflow) {
            throw std::overflow_error("Tensor_Matmul::_naive_matmul_impl: Overflow / underflow detected in matmul.\n");
        }
    }
    else if constexpr (std::is_signed_v<T>) {
        // Create buffer for overflow / underflow checking
        T mul_result = 0;
        bool overflow = false;
        for (size_t i = 0; i < destination.extent(0); ++i) {
            for (size_t j = 0; j < destination.extent(1); ++j) {
                // Get a pointer to the result's [i, j]
                T* result_i_j = &(destination.at({i, j}));
                for (size_t k = 0; k < lhs.extent(1); ++k) {
                    // First multiply [i, k] * [k, j]
                    overflow |= TensorMath_NS::_mul_overflow_signed(lhs.at({i, k}), rhs.at({k, j}), &mul_result);
                    overflow |= TensorMath_NS::_add_overflow_signed(*result_i_j, mul_result, result_i_j);
                }
            }
        }
        if (overflow) {
            throw std::overflow_error("Tensor_Matmul::_naive_matmul_impl: Overflow / underflow detected in matmul.\n");
        }
    }
    else {
        // Create buffer for overflow / underflow checking
        T mul_result = 0;
        bool overflow = false;
        for (size_t i = 0; i < destination.extent(0); ++i) {
            for (size_t j = 0; j < destination.extent(1); ++j) {
                // Get a pointer to the result's [i, j]
                T* result_i_j = &(destination.at({i, j}));
                for (size_t k = 0; k < lhs.extent(1); ++k) {
                    // First multiply [i, k] * [k, j]
                    overflow |= TensorMath_NS::_mul_overflow_unsigned(lhs.at({i, k}), rhs.at({k, j}), &mul_result);
                    overflow |= TensorMath_NS::_add_overflow_unsigned(*result_i_j, mul_result, result_i_j);
                }
            }
        }
        if (overflow) {
            throw std::overflow_error("Tensor_Matmul::_naive_matmul_impl: Overflow / underflow detected in matmul.\n");
        }
    }
}

/**
 * Perform a matmul across two Tensors, with their target dims specified
 * @param lhs Lefthand Tensor to matmul
 * @param lhs_dim0 The first dim of lhs for the matmul
 * @param lhs_dim1 The second dim of lhs for the matmul
 * @param lhs_coordinates Base coordinates if we don't want to use 0
 * @param rhs Righthand Tensor to matmul
 * @param rhs_dim0 The first dim of the rhs for the matmul
 * @param rhs_dim1 The second dim of the rhs for the matmul
 * @param rhs_coordinates Base coordinates if we don't want to use 0
 * @param destination Tensor reference to write the result to
 */
template <typename T> 
requires std::is_arithmetic_v<T>
Tensor<T>& matmul(const AbstractTensor<T>& lhs, size_t lhs_dim0, size_t lhs_dim1, std::vector<size_t>& lhs_coordinates,
                  const AbstractTensor<T>& rhs, size_t rhs_dim0, size_t rhs_dim1, std::vector<size_t>& rhs_coordinates,
                  Tensor<T>& destination) {
    // Use the checker function instead of writing the check manually several times
    auto compat = lhs._can_matmul(lhs_dim0, lhs_dim1, rhs, rhs_dim0, rhs_dim1);
    if (!compat.has_value()) {
        throw std::invalid_argument(std::format("Tensor_Matmul::matmul: Unable to matmul. Error: {}.", compat.error()));
    }
    // Ensure lhs_coordinates and rhs_coordinates have the correct size
    if (lhs_coordinates.size() != lhs.rank() || rhs_coordinates.size() != rhs.rank()) {
        throw std::invalid_argument("Tensor_Matmul::matmul: Invalid coordinate vector(s) provided.\n");
    }
    // Ensure the destination Tensor has the right dimensions
    if (destination.extent(0) != lhs.extent(lhs_dim0) || destination.extent(1) != rhs.extent(rhs_dim1)) {
        throw std::invalid_argument(
            std::format(
                "Tensor_Matmul::matmul: Destination Tensor has shape [{}, {}] but should be [{}, {}].\n",
                destination.extent(0), destination.extent(1), lhs.extent(lhs_dim0), rhs.extent(rhs_dim1)
            )
        );
    }
    // Zero out the content of the destination
    destination.fill(0);
    // Use the desired matmul impl to execute the operation
    _naive_matmul_impl_v1(lhs, lhs_dim0, lhs_dim1, lhs_coordinates, rhs, rhs_dim0, rhs_dim1, rhs_coordinates, destination);
    return destination;
}

/**
 * Perform a matmul across two Tensors, with their target dims specified
 * @param lhs Lefthand Tensor to matmul
 * @param lhs_dim0 The first dim of lhs for the matmul
 * @param lhs_dim1 The second dim of lhs for the matmul
 * @param lhs_coordinates Base coordinates if we don't want to use 0
 * @param rhs Righthand Tensor to matmul
 * @param rhs_dim0 The first dim of the rhs for the matmul
 * @param rhs_dim1 The second dim of the rhs for the matmul
 * @param rhs_coordinates Base coordinates if we don't want to use 0
 */
template <typename T> 
requires std::is_arithmetic_v<T>
Tensor<T> matmul(const AbstractTensor<T>& lhs, size_t lhs_dim0, size_t lhs_dim1, std::vector<size_t>& lhs_coordinates,
                 const AbstractTensor<T>& rhs, size_t rhs_dim0, size_t rhs_dim1, std::vector<size_t>& rhs_coordinates) {
    // Use the checker function instead of writing the check manually several times
    auto compat = lhs._can_matmul(lhs_dim0, lhs_dim1, rhs, rhs_dim0, rhs_dim1);
    if (!compat.has_value()) {
        throw std::invalid_argument(std::format("Tensor_Matmul::matmul: Unable to matmul. Error: {}.", compat.error()));
    }
    // Ensure lhs_coordinates and rhs_coordinates have the correct size
    if (lhs_coordinates.size() != lhs.rank() || rhs_coordinates.size() != rhs.rank()) {
        throw std::invalid_argument("Tensor_Matmul::matmul: Invalid coordinate vector(s) provided.\n");
    }
    // Create a Tensor to store the result
    Tensor<T> result({lhs.extent(lhs_dim0), rhs.extent(rhs_dim1)});
    // Use the desired matmul impl to execute the operation
    _naive_matmul_impl_v1(lhs, lhs_dim0, lhs_dim1, lhs_coordinates, rhs, rhs_dim0, rhs_dim1, rhs_coordinates, result);
    // Use RVO to return the result without copying
    return result;
}

/**
 * Perform a matmul across two Tensors, with their target dims specified
 * @param lhs Lefthand Tensor to matmul
 * @param lhs_dim0 The first dim of lhs for the matmul
 * @param lhs_dim1 The second dim of lhs for the matmul
 * @param rhs Righthand Tensor to matmul
 * @param rhs_dim0 The first dim of the rhs for the matmul
 * @param rhs_dim1 The second dim of the rhs for the matmul
 * @param destination Tensor reference to write the result to
 */
template <typename T> 
requires std::is_arithmetic_v<T>
Tensor<T>& matmul(const AbstractTensor<T>& lhs, size_t lhs_dim0, size_t lhs_dim1,
                  const AbstractTensor<T>& rhs, size_t rhs_dim0, size_t rhs_dim1,
                  Tensor<T>& destination) {
    // Use the checker function instead of writing the check manually several times
    auto compat = lhs._can_matmul(lhs_dim0, lhs_dim1, rhs, rhs_dim0, rhs_dim1);
    if (!compat.has_value()) {
        throw std::invalid_argument(std::format("Tensor_Matmul::matmul: Unable to matmul. Error: {}.", compat.error()));
    }
    // Create reusable std::vector to use with Tensor.at
    std::vector<size_t> lhs_coordinates(lhs.rank(), 0);
    std::vector<size_t> rhs_coordinates(rhs.rank(), 0);
    // Ensure the destination Tensor has the right dimensions
    if (destination.extent(0) != lhs.extent(lhs_dim0) || destination.extent(1) != rhs.extent(rhs_dim1)) {
        throw std::invalid_argument(
            std::format(
                "Tensor_Matmul::matmul: Destination Tensor has shape [{}, {}] but should be [{}, {}].\n",
                destination.extent(0), destination.extent(1), lhs.extent(lhs_dim0), rhs.extent(rhs_dim1)
            )
        );
    }
    // Zero out the content of the destination
    destination.fill(0);
    // Use the desired matmul impl to execute the operation
    _naive_matmul_impl_v1(lhs, lhs_dim0, lhs_dim1, lhs_coordinates, rhs, rhs_dim0, rhs_dim1, rhs_coordinates, destination);
    return destination;
}

/**
 * Perform a matmul across two Tensors, with their target dims specified
 * @param lhs Lefthand Tensor to matmul
 * @param lhs_dim0 The first dim of lhs for the matmul
 * @param lhs_dim1 The second dim of lhs for the matmul
 * @param rhs Righthand Tensor to matmul
 * @param rhs_dim0 The first dim of the rhs for the matmul
 * @param rhs_dim1 The second dim of the rhs for the matmul
 */
template <typename T> 
requires std::is_arithmetic_v<T>
Tensor<T> matmul(const AbstractTensor<T>& lhs, size_t lhs_dim0, size_t lhs_dim1,
                 const AbstractTensor<T>& rhs, size_t rhs_dim0, size_t rhs_dim1) {
    // Use the checker function instead of writing the check manually several times
    auto compat = lhs._can_matmul(lhs_dim0, lhs_dim1, rhs, rhs_dim0, rhs_dim1);
    if (!compat.has_value()) {
        throw std::invalid_argument(std::format("Tensor_Matmul::matmul: Unable to matmul. Error: {}.", compat.error()));
    }
    // Create reusable std::vector to use with Tensor.at
    std::vector<size_t> lhs_coordinates(lhs.rank(), 0);
    std::vector<size_t> rhs_coordinates(rhs.rank(), 0);
    // Create a Tensor to store the result
    Tensor<T> result({lhs.extent(lhs_dim0), rhs.extent(rhs_dim1)});
    // Use the desired matmul impl to execute the operation
    _naive_matmul_impl_v1(lhs, lhs_dim0, lhs_dim1, lhs_coordinates, rhs, rhs_dim0, rhs_dim1, rhs_coordinates, result);
    // Use RVO to return the result without copying
    return result;
}

/**
 * Perform a matmul across two Tensors, with their target dims specified
 * @param lhs Lefthand Tensor to matmul
 * @param rhs Righthand Tensor to matmul
 * @param destination Tensor reference to write the result to
 */
template <typename T> 
requires std::is_arithmetic_v<T>
Tensor<T>& matmul(const AbstractTensor<T>& lhs,
                  const AbstractTensor<T>& rhs,
                  Tensor<T>& destination) {
    // Assume we want dims 0 and 1 from lhs and rhs and that their rank must == 2
    if (lhs.rank() != 2 || rhs.rank() != 2) {
        throw std::invalid_argument(
            std::format(
                "Tensor_Matmul::matmul: Invalid Tensor rank for matmul(lhs, rhs). lhs.rank == {}. rhs.rank == {}.",
                    lhs.rank(), rhs.rank()));
    }
    auto compat = lhs._can_matmul(0, 1, rhs, 0, 1);
    if (!compat.has_value()) {
        throw std::invalid_argument(std::format("Tensor_Matmul::matmul: Unable to matmul. Error: {}.", compat.error()));
    }
    // Ensure the destination Tensor has the right dimensions
    if (destination.extent(0) != lhs.extent(0) || destination.extent(1) != rhs.extent(1)) {
        throw std::invalid_argument(
            std::format(
                "Tensor_Matmul::matmul: Destination Tensor has shape [{}, {}] but should be [{}, {}].\n",
                destination.extent(0), destination.extent(1), lhs.extent(0), rhs.extent(1)
            )
        );
    }
    // Zero out the content of the destination
    destination.fill(0);
    // Use the desired matmul impl to execute the operation
    _naive_matmul_impl_v1(lhs, rhs, destination);
    return destination;
}

/**
 * Perform a matmul across two Tensors, with their target dims specified
 * @param lhs Lefthand Tensor to matmul
 * @param rhs Righthand Tensor to matmul
 */
template <typename T> 
requires std::is_arithmetic_v<T>
Tensor<T> matmul(const AbstractTensor<T>& lhs,
                 const AbstractTensor<T>& rhs) {
    // Assume we want dims 0 and 1 from lhs and rhs and that their rank must == 2
    if (lhs.rank() != 2 || rhs.rank() != 2) {
        throw std::invalid_argument(
            std::format(
                "Tensor_Matmul::matmul: Invalid Tensor rank for matmul(lhs, rhs). lhs.rank == {}. rhs.rank == {}.",
                    lhs.rank(), rhs.rank()));
    }
    auto compat = lhs._can_matmul(0, 1, rhs, 0, 1);
    if (!compat.has_value()) {
        throw std::invalid_argument(std::format("Tensor_Matmul::matmul: Unable to matmul. Error: {}.", compat.error()));
    }
    // Create a Tensor to store the result
    Tensor<T> result({lhs.extent(0), rhs.extent(1)});
    // Use the desired matmul impl to execute the operation
    _naive_matmul_impl_v1(lhs, rhs, result);
    // Use RVO to return the result without copying
    return result;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

}; // namespace Tensor_Matmul_NS

#endif
