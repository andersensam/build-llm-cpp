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
#include "Log.hpp"
#include "Numerics.hpp"
#include "Tensor.hpp"
#include "TensorMath.hpp"
#include "TensorSlice.hpp"
#include "TensorUtils.hpp"

namespace TensorMatmul_NS {

/* Control whether or not we want to enable the naive matmul v2 path */
inline constexpr bool TENSORMATMUL_ENABLE_NAIVE_MATMUL_V2 = true;

/* Control whether to log warnings about falling back to naive matmul v1 */
inline constexpr bool TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING = true;

/* Control whether or not we prefer to make AbstractTensors that are not contiguous into Tensors to speed up matmuls */
inline constexpr bool TENSORMATMUL_ENABLE_ABSTRACT_TENSOR_CONVERSION = true;

/* Control whether or not to prefer copying input Tensors to resolve stride issues instead of falling back to v1 */
inline constexpr bool TENSORMATMUL_NAIVE_MATMUL_V2_PREFER_TEMP_INPUT_COPY = true;

/* Control whether or not to allow creating a temporary destination Tensor if is_unique() is false, or throw an exception */
inline constexpr bool TENSORMATMUL_PREFER_TEMP_DESTINATION_OVER_EXCEPTION = false;

/* Control whether or not to log the creation of temporary Tensors (see above)*/
inline constexpr bool TENSORMATMUL_NAIVE_MATMUL_V2_LOG_TEMP = true;

/* Control whether or not to allow running .contiguous() on the destination Tensor for matmuls */
inline constexpr bool TENSORMATMUL_ENABLE_DESTINATION_TENSOR_CONTIGUOUS = true;

/**
 * Enum class to determine errors that prevent usage of _naive_matmul_impl_v2
 */
enum class NaiveMatmulV2Error : uint8_t {
    OK,
    DEST_STRIDE,
    CONVERT_ABSTRACT,
    INPUT_STRIDE,
};

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;

/* Use logging */
using Log_NS::log_message;
using Log_NS::Log_Priority;

/* Use Tensor */
using Tensor_NS::Tensor;

/* Use ListTensorSlice */
using TensorSlice_NS::ListTensorSlice;

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
 * safety checks you would otherwise need to do (handled in TensorMatmul::matmul, etc.)
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
            throw std::overflow_error("TensorMatmul::_naive_matmul_impl: Overflow / underflow detected in matmul.\n");
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
            throw std::overflow_error("TensorMatmul::_naive_matmul_impl: Overflow / underflow detected in matmul.\n");
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
            throw std::overflow_error("TensorMatmul::_naive_matmul_impl: Overflow / underflow detected in matmul.\n");
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
 * safety checks you would otherwise need to do (handled in TensorMatmul::matmul, etc.)
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
            throw std::overflow_error("TensorMatmul::_naive_matmul_impl: Overflow / underflow detected in matmul.\n");
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
            throw std::overflow_error("TensorMatmul::_naive_matmul_impl: Overflow / underflow detected in matmul.\n");
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
            throw std::overflow_error("TensorMatmul::_naive_matmul_impl: Overflow / underflow detected in matmul.\n");
        }
    }
}

/**
 * Helper function to determine compatibility with _naive_matmul_impl_v2
 * @param lhs Const ref to an AbstractTensor to be lhs
 * @param lhs_dim1 Second dim in lhs
 * @param rhs Const ref to an AbstractTensor to be rhs
 * @param rhs_dim1 Second dim in rhs
 * @param dest Cost ref to the destination Tensor
 * @param dest_dim1 Second dim in dest
 * @returns Returns a code specified in the NaiveMatmulV2Error enum
 */
template <typename T> 
requires std::is_arithmetic_v<T>
NaiveMatmulV2Error can_use_naive_matmul_impl_v2(const AbstractTensor<T>& lhs, size_t lhs_dim1,
                                                const AbstractTensor<T>& rhs, size_t rhs_dim1,
                                                const Tensor<T>& dest, size_t dest_dim1) {
    // Don't check for uniqueness since all matmuls require this to be true
    // Verify that the destination stride in dim1 == 1
    if (dest.dim_stride(dest_dim1) != 1) {
        return NaiveMatmulV2Error::DEST_STRIDE;
    }
    // Check to see if we might need to convert lhs and rhs --> Tensors
    if (!lhs.is_contiguous() || !rhs.is_contiguous()) {
        const Tensor<T>* lhs_ptr = dynamic_cast<const Tensor<T>*>(&lhs);
        const Tensor<T>* rhs_ptr = dynamic_cast<const Tensor<T>*>(&rhs);
        // If either of the conversions fail, we know we must be dealing with ListTensorSlices
        if (lhs_ptr == nullptr || rhs_ptr == nullptr) {
            return NaiveMatmulV2Error::CONVERT_ABSTRACT;
        }
        else {
            // If we can convert a non-contiguous Tensor, check the stride in dim1
            if (lhs_ptr->dim_stride(lhs_dim1) != 1 || rhs_ptr->dim_stride(rhs_dim1) != 1) {
                return NaiveMatmulV2Error::INPUT_STRIDE;
            }
        }
    }
    // If the underlying lhs and rhs are contiguous, but their dim1 might not have stride == 1
    if (lhs.stride().at(lhs_dim1) != 1 || rhs.stride().at(rhs_dim1) != 1) {
        return NaiveMatmulV2Error::INPUT_STRIDE;
    }
    // Return OK if no errors come up
    return NaiveMatmulV2Error::OK;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters, cppcoreguidelines-pro-bounds-pointer-arithmetic)
/**
 * Vectorized naive matmul implementation
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param lhs_dim0_stride Stride of the lhs outer dim
 * @param rhs_ptr Pointer to the raw data underlying the Tensor
 * @param rhs_offset Offset to the beginning of the data block
 * @param rhs_dim0_stride Stride of the rhs outer dim
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param result_dim0_stride Stride of the result outer dim
 * @param dim0_extent Number of rows in the destination Tensor
 * @param dim1_extent Number of columns in the destination Tensor
 * @param inner_dim_extent Inner dim of lhs and rhs
 * NOTE: v2 can ONLY be used when result_ptr is unique (passes is_unique() with lhs and rhs).
 * Since lhs and rhs are never being written to, we don't really care if they overlap or not
 */
template <typename T>
requires std::is_arithmetic_v<T>
void _naive_matmul_impl_v2(const T* lhs_ptr, size_t lhs_offset, size_t lhs_dim0_stride,
                           const T* rhs_ptr, size_t rhs_offset, size_t rhs_dim0_stride,
                           T* __restrict__ result_ptr, size_t result_offset, size_t result_dim0_stride,
                           size_t dim0_extent, size_t dim1_extent, size_t inner_dim_extent) {
    // Zero out the result Tensor for both dims
    for (size_t i = 0; i < dim0_extent; ++i) {
        const size_t result_row = result_offset + (result_dim0_stride * i);
        for (size_t j = 0; j < dim1_extent; ++j) {
            result_ptr[result_row + j] = 0;
        }
    }
    // Handle floating point types first
    if constexpr (std::is_floating_point_v<T>) {
        // Loop over dim0
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
            // Loop over the inner dim to prevent cache misses
            for (size_t k = 0; k < inner_dim_extent; ++k) {
                const size_t rhs_row = rhs_offset + (rhs_dim0_stride * k);
                // Broadcast lhs[i, k] for the operation
                const T lhs_val = lhs_ptr[lhs_row + k];
                // If lhs_val == 0, we can skip the entire inner dim1 loop
                if (lhs_val == 0) { continue; }
                // Loop over dim1 and perform the core op
                for (size_t j = 0; j < dim1_extent; ++j) {
                    result_ptr[result_row + j] += lhs_val * rhs_ptr[rhs_row + j];
                }
            }
        }
        // Check for INF and NaN in the result Tensor
        bool overflow = false;
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                T result = result_ptr[result_row + j];
                overflow |= !std::isfinite(result);
            }
        }
        if (overflow) {
            throw std::overflow_error("TensorMatmul::_naive_matmul_impl_v2: Overflow / underflow detected.\n");
        }
    }
    // Handle the integer types, checking for accumulator availability
    else {
        // Track overflow
        bool overflow = false;
        // Check for accumulator avaibility
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            // Use an overflow mask as a target for SIMD
            // We might run into issues with small types (int8_t or int16_t) being
            // promoted to int32_t, which can cause issues with |= operation on
            // a bool
            uint32_t overflow_mask = 0;
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            for (size_t i = 0; i < dim0_extent; ++i) {
                const size_t result_row = result_offset + (result_dim0_stride * i);
                const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                // Loop over the inner dim to prevent cache misses
                for (size_t k = 0; k < inner_dim_extent; ++k) {
                    const size_t rhs_row = rhs_offset + (rhs_dim0_stride * k);
                    // Broadcast lhs[i, k] for the operation
                    const accumulator_t lhs_val = static_cast<accumulator_t>(lhs_ptr[lhs_row + k]);
                    // If lhs_val == 0, we can skip the entire inner dim1 loop
                    if (lhs_val == 0) { continue; }
                    // Loop over dim1 and perform the core op
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        accumulator_t result = lhs_val * static_cast<accumulator_t>(rhs_ptr[rhs_row + j]);
                        accumulator_t result_i_j = static_cast<accumulator_t>(result_ptr[result_row + j]) + result;
                        overflow_mask |= static_cast<uint32_t>((result_i_j > MAX_VAL | result_i_j < MIN_VAL));
                        result_ptr[result_row + j] = static_cast<T>(result_i_j);
                    }
                }
            }
            overflow |= (overflow_mask != 0);
        }
        // Handle signed types without accumulators
        else if constexpr (std::is_signed_v<T>) {
            // Get the corresponding unsigned type
            using U = std::make_unsigned_t<T>;
            // Use an overflow mask as a target for SIMD
            // We might run into issues with small types (int8_t or int16_t) being
            // promoted to int32_t, which can cause issues with |= operation on
            // a bool
            uint32_t overflow_mask = 0;
            // Get the min and max values for the type
            constexpr T MIN_VAL = std::numeric_limits<T>::min();
            constexpr T MAX_VAL = std::numeric_limits<T>::max();
            // Iterate over the outer dim
            for (size_t i = 0; i < dim0_extent; ++i) {
                const size_t result_row = result_offset + (result_dim0_stride * i);
                const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                // Loop over the inner dim to prevent cache misses
                for (size_t k = 0; k < inner_dim_extent; ++k) {
                    const size_t rhs_row = rhs_offset + (rhs_dim0_stride * k);
                    // Broadcast lhs[i, k] for the operation
                    const T lhs_val = lhs_ptr[lhs_row + k];
                    // If lhs_val == 0, we can skip the entire inner dim1 loop
                    if (lhs_val == 0) { continue; }
                    // Calcuate the valid range of values for rhs once outside the inner loop
                    /**
                     * DISCLOSURE: AI tools were used to analyze why this branch was not easily
                     * vectorizable -- new learnings: integer promotion for int8_t and int16_t
                     * and the boolean accumulator issues. I do find it interesting
                     * that the unsigned path doesn't have this issue and was vectorizable
                     * from the beginning.
                     */
                    T rhs_v_min = 0, rhs_v_max = 0;
                    if (lhs_val > 0) {
                        rhs_v_min = MIN_VAL / lhs_val;
                        rhs_v_max = MAX_VAL / lhs_val;
                    }
                    else if (lhs_val == -1) {
                        rhs_v_min = -MAX_VAL;
                        rhs_v_max = MAX_VAL;
                    }
                    else {
                        rhs_v_min = MAX_VAL / lhs_val;
                        rhs_v_max = MIN_VAL / lhs_val;
                    }
                    // Loop over dim1 and perform the core op
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        // Get the vals for rhs and result
                        const T rhs_val = rhs_ptr[rhs_row + j];
                        const T result_val = result_ptr[result_row + j];
                        // Use unsigned ops to perform the multiplication of lhs_val * rhs_val
                        const T prod = static_cast<T>(static_cast<U>(lhs_val) * static_cast<U>(rhs_val));
                        const T sum = static_cast<T>(static_cast<U>(result_val) + static_cast<U>(prod));
                        // Detect multiplcation and addition overflows by validating against the rhs range
                        // The rhs checks could occur earlier; however, vectorization prevents early exits,
                        // so just calculate with the addition overflow for readability
                        const bool mul_overflow = (rhs_val < rhs_v_min) | (rhs_val > rhs_v_max);
                        // If result_val and sum have different signs AND prod and sum have different signs
                        // then we must have overflowed on the addition
                        // XOR checks the MSB (see TensorMath.hpp)
                        const T add_check = static_cast<T>((result_val ^ sum) & (prod ^ sum));
                        const bool add_overflow = add_check < 0;
                        // Write to the overflow_mask
                        overflow_mask |= static_cast<uint32_t>((mul_overflow | add_overflow));
                        // Persist the value
                        result_ptr[result_row + j] = sum;
                    }
                }
            }
            overflow |= (overflow_mask != 0);
        }
        // Handle unsigned types
        else {
            for (size_t i = 0; i < dim0_extent; ++i) {
                const size_t result_row = result_offset + (result_dim0_stride * i);
                const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                // Loop over the inner dim to prevent cache misses
                for (size_t k = 0; k < inner_dim_extent; ++k) {
                    const size_t rhs_row = rhs_offset + (rhs_dim0_stride * k);
                    // Broadcast lhs[i, k] for the operation
                    const T lhs_val = lhs_ptr[lhs_row + k];
                    // If lhs_val == 0, we can skip the entire inner dim1 loop
                    if (lhs_val == 0) { continue; }
                    // Loop over dim1 and perform the core op
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        T partial = 0;
                        overflow |= TensorMath_NS::_mul_overflow_unsigned(lhs_val, rhs_ptr[rhs_row + j], &partial);
                        overflow |= TensorMath_NS::_add_overflow_unsigned(result_ptr[result_row + j], partial, &(result_ptr[result_row + j]));
                    }
                }
            }
        }
        if (overflow) {
            throw std::overflow_error("TensorMatmul::_naive_matmul_impl_v2: Overflow / underflow detected.\n");
        }
    }
}
// NOLINTEND(bugprone-easily-swappable-parameters, cppcoreguidelines-pro-bounds-pointer-arithmetic)

// NOLINTBEGIN(bugprone-easily-swappable-parameters
/**
 * Dispatch function for _naive_matmul_impl_v2, creating a unified interface that can be recycled
 * through the various matmul(...) functions, instead of maintaining various branches / code paths
 * @param lhs Const ref to an AbstractTensor to matmul
 * @param lhs_dim0 First dim of lhs
 * @param lhs_dim1 Second dim of lhs
 * @param lhs_coordinates If lhs.rank() > 2, provide coordinates for the other dims
 * @param rhs Const ref to an AbstractTensor to matmul
 * @param rhs_dim0 First dim of rhs
 * @param rhs_dim1 Second dim of rhs
 * @param rhs_coordinates If rhs.rank() > 2, provide coordinates for the other dims
 * @param dest Reference to a Tensor to store the result of the matmul in
 * @param dest_dim0 First dim of dest
 * @param dest_dim1 Second dim of dest
 * @param dest_coordinates If dest.rank() > 2, provide coordinates for the other dims
 * @returns True if dispatch is successful, false if an error occurred
 */
template <typename T> 
requires std::is_arithmetic_v<T>
bool naive_matmul_v2_dispatch(const AbstractTensor<T>& lhs, size_t lhs_dim0, size_t lhs_dim1, std::vector<size_t>& lhs_coordinates,
                              const AbstractTensor<T>& rhs, size_t rhs_dim0, size_t rhs_dim1, std::vector<size_t>& rhs_coordinates,
                              Tensor<T>& dest, size_t dest_dim0, size_t dest_dim1, std::vector<size_t>& dest_coordinates) {
    // Handle the case where lhs, rhs, and dest are all rank-2 Tensors
    if (lhs.rank() == 2 && rhs.rank() == 2 && dest.rank() == 2) {
        // If all are of rank == 2, we can ignore the coordinate vectors
        NaiveMatmulV2Error check = can_use_naive_matmul_impl_v2(lhs, lhs_dim1,
                                                                rhs, rhs_dim1,
                                                                dest, dest_dim1);
        // Switch the potential error codes (or OK)
        switch (check) {
            // Handle the OK case and execute immediately
            case NaiveMatmulV2Error::OK:
                _naive_matmul_impl_v2(lhs._data(), lhs.offset(), lhs.stride().at(lhs_dim0),
                                      rhs._data(), rhs.offset(), rhs.stride().at(rhs_dim0),
                                      dest._data(), dest.offset(), dest.dim_stride(dest_dim0),
                                      dest.extent(dest_dim0), dest.extent(dest_dim1), lhs.extent(lhs_dim1));
                // Return true to indicate the dispatch was successful
                return true;
            // Handle the case where the destination Tensor's stride in dest_dim1 != 1
            case NaiveMatmulV2Error::DEST_STRIDE: {
                // Check to see if we are allowed to run .contiguous() on destination
                if constexpr (TENSORMATMUL_ENABLE_DESTINATION_TENSOR_CONTIGUOUS) {
                    // If dest.dim_stride(dest_dim0) == 1, running .contiguous() won't have any effect,
                    // so we will need to create a temp destination and copy back later
                    if (dest.dim_stride(dest_dim0) == 1) {
                        if constexpr (TENSORMATMUL_PREFER_TEMP_DESTINATION_OVER_EXCEPTION) {
                            if constexpr (TENSORMATMUL_NAIVE_MATMUL_V2_LOG_TEMP) {
                                log_message(Log_Priority::DEBUG, "TensorMatmul::naive_matmul_v2_dispatch",
                                    "Creating temporary destination Tensor"
                                );
                            }
                            // The new temp_dest is guaranteed to be contiguous and have stride == 1 in dest_dim1
                            Tensor<T> temp_dest({dest.extent(dest_dim0), dest.extent(dest_dim1)});
                            // Dispatch again and return the result
                            if (naive_matmul_v2_dispatch(lhs, lhs_dim0, lhs_dim1, lhs_coordinates,
                                                         rhs, rhs_dim0, rhs_dim1, rhs_coordinates,
                                                         temp_dest, 0, 1, dest_coordinates)) {
                                // If we are successful, copy the data back into dest
                                dest.copy_from(temp_dest);
                                return true;
                            }
                            if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
                                log_message(Log_Priority::WARNING, "TensorMatmul::naive_matmul_v2_dispatch",
                                    "Error with recursive call to dispatch after DEST_STRIDE error -- falling back to v1.");
                            }
                            return false;
                        }
                        else {
                            log_message(Log_Priority::WARNING, "TensorMatmul::naive_matmul_v2_dispatch",
                                "Creating temp destination Tensors is disabled, falling back to v1 after DEST_STRIDE error.");
                            return false;
                        }
                    }
                    else {
                        // If neither dim in dest has stride == 1, try to run .contiguous() if dest_dim1 is the outer dim
                        if (dest.shape().at(1) == dest_dim1) {
                            dest.contiguous();
                            // Do another sanity check to ensure the .contiguous() call worked
                            if (dest.dim_stride(dest_dim1) == 1) {
                                // Dispatch again and return the result
                                return naive_matmul_v2_dispatch(lhs, lhs_dim0, lhs_dim1, lhs_coordinates,
                                                                rhs, rhs_dim0, rhs_dim1, rhs_coordinates,
                                                                dest, dest_dim0, dest_dim1, dest_coordinates);
                            }
                            else {
                                // If the .continguous() call failed to resolve the dim stride issue, fall back
                                if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
                                    log_message(Log_Priority::WARNING, "TensorMatmul::naive_matmul_v2_dispatch",
                                        "Calling .contiguous() did not resolve the DEST_STRIDE error. Falling back to v1.");
                                }
                                return false;
                            }
                        }
                        else {
                            // If dest's outer dim != dest_dim1, we need to create a temporary destination
                            // and copy the result back
                            if constexpr (TENSORMATMUL_PREFER_TEMP_DESTINATION_OVER_EXCEPTION) {
                                if constexpr (TENSORMATMUL_NAIVE_MATMUL_V2_LOG_TEMP) {
                                    log_message(Log_Priority::DEBUG, "TensorMatmul::naive_matmul_v2_dispatch",
                                        "Creating temporary destination Tensor"
                                    );
                                }
                                // The new temp_dest is guaranteed to be contiguous and have stride == 1 in dest_dim1
                                Tensor<T> temp_dest({dest.extent(dest_dim0), dest.extent(dest_dim1)});
                                // Dispatch again and return the result
                                if (naive_matmul_v2_dispatch(lhs, lhs_dim0, lhs_dim1, lhs_coordinates,
                                                             rhs, rhs_dim0, rhs_dim1, rhs_coordinates,
                                                             temp_dest, 0, 1, dest_coordinates)) {
                                    // If we are successful, copy the data back into dest
                                    dest.copy_from(temp_dest);
                                    return true;
                                }
                                if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
                                    log_message(Log_Priority::WARNING, "TensorMatmul::naive_matmul_v2_dispatch",
                                        "Error with recursive call to dispatch after DEST_STRIDE error -- falling back to v1.");
                                }
                                return false;
                            }
                            else {
                                if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
                                    log_message(Log_Priority::WARNING, "TensorMatmul::naive_matmul_v2_dispatch",
                                        "Creating temp destination Tensors is disabled, falling back to v1 after DEST_STRIDE error.");
                                }
                                return false;
                            }
                        }
                    }
                }
                else if constexpr (TENSORMATMUL_PREFER_TEMP_DESTINATION_OVER_EXCEPTION) {
                    if constexpr (TENSORMATMUL_NAIVE_MATMUL_V2_LOG_TEMP) {
                        log_message(Log_Priority::DEBUG, "TensorMatmul::naive_matmul_v2_dispatch",
                            "Creating temporary destination Tensor"
                        );
                    }
                    Tensor<T> temp_dest({dest.extent(dest_dim0), dest.extent(dest_dim1)});
                    if (naive_matmul_v2_dispatch(lhs, lhs_dim0, lhs_dim1, lhs_coordinates,
                                                 rhs, rhs_dim0, rhs_dim1, rhs_coordinates,
                                                 temp_dest, 0, 1, dest_coordinates)) {
                        // Copy the result back into dest
                        dest.copy_from(temp_dest);
                        return true;
                    }
                    if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
                        log_message(Log_Priority::WARNING, "TensorMatmul::naive_matmul_v2_dispatch",
                            "Unknown error when correcting DEST_STRIDE. Falling back to v1.");
                    }
                    return false;
                }
                else {
                    if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
                        log_message(Log_Priority::WARNING, "TensorMatmul::naive_matmul_v2_dispatch",
                            "Error correcting DEST_STRIDE: .contiguous() and creating temp destinations are not enabled.");
                    }
                    return false;
                }
            }
            // Handle the case where we need to convert the input AbstractTensors --> Tensors
            case NaiveMatmulV2Error::CONVERT_ABSTRACT: {
                // Check to see if we are allowed to convert AbstractTensor
                if constexpr (TENSORMATMUL_ENABLE_ABSTRACT_TENSOR_CONVERSION) {
                    if constexpr (TENSORMATMUL_NAIVE_MATMUL_V2_LOG_TEMP) {
                        log_message(Log_Priority::DEBUG, "TensorMatmul::naive_matmul_v2_dispatch",
                            "Creating temporary lhs and/or rhs via dynamic_cast --> to_tensor()"
                        );
                    }
                    // Attempt to dynamic cast lhs and rhs to Tensors
                    const Tensor<T>* lhs_ptr = dynamic_cast<const Tensor<T>*>(&lhs);
                    const Tensor<T>* rhs_ptr = dynamic_cast<const Tensor<T>*>(&rhs);
                    // Convert whichever (or both) failed to cast successfully
                    const Tensor<T> lhs_c = (lhs_ptr == nullptr) ? dynamic_cast<const ListTensorSlice<T>*>(&lhs)->to_tensor() : *lhs_ptr;
                    const Tensor<T> rhs_c = (rhs_ptr == nullptr) ? dynamic_cast<const ListTensorSlice<T>*>(&rhs)->to_tensor() : *rhs_ptr;
                    // Take the converted Tensors and dispatch again
                    return naive_matmul_v2_dispatch(lhs_c, lhs_dim0, lhs_dim1, lhs_coordinates,
                                                    rhs_c, rhs_dim0, rhs_dim1, rhs_coordinates,
                                                    dest, dest_dim0, dest_dim1, dest_coordinates);
                }
                else {
                    if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
                        log_message(Log_Priority::WARNING, "TensorMatmul::naive_matmul_v2_dispatch",
                            "AbstractTensor conversion is disabled. Falling back to v1.");
                    }
                    return false;
                }
            }
            // Handle the case where the input stride in dim1 != 1
            case NaiveMatmulV2Error::INPUT_STRIDE: {
                // Check to see if we are allowed to create temporary input copies
                if constexpr (TENSORMATMUL_NAIVE_MATMUL_V2_PREFER_TEMP_INPUT_COPY) {
                    // Since potential CONVERT_ABSTRACT issues are already taken care of we should be able to
                    // cast to Tensor without issue
                    // Attempt to dynamic cast lhs and rhs to Tensors
                    const Tensor<T>* lhs_ptr = dynamic_cast<const Tensor<T>*>(&lhs);
                    const Tensor<T>* rhs_ptr = dynamic_cast<const Tensor<T>*>(&rhs);
                    // Do a sanity check to make sure we aren't running into any unexpected casting issues
                    if (lhs_ptr == nullptr || rhs_ptr == nullptr) {
                        if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
                            log_message(Log_Priority::WARNING, "TensorMatmul::naive_matmul_v2_dispatch",
                                "Unexpected error when using dynamic_cast on input AbstractTensor. Falling back to v1.");
                        }
                        return false;
                    }
                    // Handle both Tensors requiring copies
                    if (lhs_ptr->dim_stride(lhs_dim1) != 1 && rhs_ptr->dim_stride(rhs_dim1) != 1) {
                        if constexpr (TENSORMATMUL_NAIVE_MATMUL_V2_LOG_TEMP) {
                            log_message(Log_Priority::DEBUG, "TensorMatmul::naive_matmul_v2_dispatch",
                                "Creating temporary lhs and rhs Tensors"
                            );
                        }
                        // Both of these Tensors are contiguous with dim_stride(dim1) == 1 by definition
                        Tensor<T> temp_lhs({lhs_ptr->extent(lhs_dim0), lhs_ptr->extent(lhs_dim1)});
                        // Use the utility function because it's possible that lhs == rhs with a transpose applied
                        TensorUtils_NS::copy_high_rank_abstract_to_tensor(*lhs_ptr, lhs_dim0, lhs_dim1, std::vector<size_t>{0, 0},
                                                                          temp_lhs, 0, 1, temp_lhs.offset());
                        Tensor<T> temp_rhs({rhs_ptr->extent(rhs_dim0), rhs_ptr->extent(rhs_dim1)});
                        TensorUtils_NS::copy_high_rank_abstract_to_tensor(*rhs_ptr, rhs_dim0, rhs_dim1, std::vector<size_t>{0, 0},
                                                                          temp_rhs, 0, 1, temp_rhs.offset());
                        // Since destination hasn't been rewritten, we should be able to return the result directly
                        return naive_matmul_v2_dispatch(temp_lhs, 0, 1, lhs_coordinates,
                                                        temp_rhs, 0, 1, rhs_coordinates,
                                                        dest, dest_dim0, dest_dim1, dest_coordinates);
                    }
                    // If only lhs needs to be copied
                    else if (lhs_ptr->dim_stride(lhs_dim1) != 1) {
                        if constexpr (TENSORMATMUL_NAIVE_MATMUL_V2_LOG_TEMP) {
                            log_message(Log_Priority::DEBUG, "TensorMatmul::naive_matmul_v2_dispatch",
                                "Creating temporary lhs Tensor"
                            );
                        }
                        Tensor<T> temp_lhs({lhs_ptr->extent(lhs_dim0), lhs_ptr->extent(lhs_dim1)});
                        TensorUtils_NS::copy_high_rank_abstract_to_tensor(*lhs_ptr, lhs_dim0, lhs_dim1, std::vector<size_t>{0, 0},
                                                                          temp_lhs, 0, 1, temp_lhs.offset());
                        // Since destination hasn't been rewritten, we should be able to return the result directly
                        return naive_matmul_v2_dispatch(temp_lhs, 0, 1, lhs_coordinates,
                                                        rhs, rhs_dim0, rhs_dim1, rhs_coordinates,
                                                        dest, dest_dim0, dest_dim1, dest_coordinates);
                    }
                    // If only rhs needs to be copied
                    else {
                        if constexpr (TENSORMATMUL_NAIVE_MATMUL_V2_LOG_TEMP) {
                            log_message(Log_Priority::DEBUG, "TensorMatmul::naive_matmul_v2_dispatch",
                                "Creating temporary rhs Tensor"
                            );
                        }
                        Tensor<T> temp_rhs({rhs_ptr->extent(rhs_dim0), rhs_ptr->extent(rhs_dim1)});
                        TensorUtils_NS::copy_high_rank_abstract_to_tensor(*rhs_ptr, rhs_dim0, rhs_dim1, std::vector<size_t>{0, 0},
                                                                          temp_rhs, 0, 1, temp_rhs.offset());
                        // Since destination hasn't been rewritten, we should be able to return the result directly
                        return naive_matmul_v2_dispatch(lhs, lhs_dim0, lhs_dim1, lhs_coordinates,
                                                        temp_rhs, 0, 1, rhs_coordinates,
                                                        dest, dest_dim0, dest_dim1, dest_coordinates);
                    }
                }
                else {
                    if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
                        log_message(Log_Priority::WARNING, "TensorMatmul::naive_matmul_v2_dispatch",
                            "Creating temporary input copies is disabled. Falling back to v1.");
                    }
                    return false;
                }
            }
            // Handle any unexpected / unimplemented errors
            default: {
                if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
                    log_message(Log_Priority::WARNING, "TensorMatmul::naive_matmul_v2_dispatch",
                        "Unknown error occurred in dispatch routing. Falling back to v1.");
                }
                return false;
            }
        }
    }
    // Handle higher rank Tensors, which may require slicing to resolve issues
    else {
        NaiveMatmulV2Error check = can_use_naive_matmul_impl_v2(lhs, lhs_dim1,
                                                                rhs, rhs_dim1,
                                                                dest, dest_dim1);
        // Switch the potential error codes (or OK)
        switch (check) {
            // Handle the OK case and execute immediately
            case NaiveMatmulV2Error::OK: {
                // Grab the offset from _get_offset before sending to _naive_matmul_impl_v2,
                // which requires casting to Tensor first. This op is guaranteed to work
                // since we dont have CONVERT_ABSTRACT
                const Tensor<T>* lhs_ptr = dynamic_cast<const Tensor<T>*>(&lhs);
                const Tensor<T>* rhs_ptr = dynamic_cast<const Tensor<T>*>(&rhs);
                if (lhs_ptr == nullptr || rhs_ptr == nullptr) {
                    if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
                        log_message(Log_Priority::WARNING, "TensorMatmul::naive_matmul_v2_dispatch",
                            "Unknown error occurred in dynamic_cast. Falling back to v1.");
                    }
                    return false;
                }
                if (!dest_coordinates.empty()) {
                    _naive_matmul_impl_v2(lhs_ptr->_data(), lhs_ptr->_get_offset(lhs_coordinates), lhs_ptr->dim_stride(lhs_dim0),
                                          rhs_ptr->_data(), rhs_ptr->_get_offset(rhs_coordinates), rhs_ptr->dim_stride(rhs_dim0),
                                          dest._data(), dest._get_offset(dest_coordinates), dest.dim_stride(dest_dim0),
                                          dest.extent(dest_dim0), dest.extent(dest_dim1), lhs_ptr->extent(lhs_dim1));
                }
                else {
                    _naive_matmul_impl_v2(lhs_ptr->_data(), lhs_ptr->_get_offset(lhs_coordinates), lhs_ptr->dim_stride(lhs_dim0),
                                          rhs_ptr->_data(), rhs_ptr->_get_offset(rhs_coordinates), rhs_ptr->dim_stride(rhs_dim0),
                                          dest._data(), dest.offset(), dest.dim_stride(dest_dim0),
                                          dest.extent(dest_dim0), dest.extent(dest_dim1), lhs_ptr->extent(lhs_dim1));
                }
                return true;
            }
            // Handle the case where the destination Tensor's stride in dest_dim1 != 1
            case NaiveMatmulV2Error::DEST_STRIDE: {
                if constexpr (TENSORMATMUL_ENABLE_DESTINATION_TENSOR_CONTIGUOUS) {
                    // Check to see if dest_dim1 is the outer dim
                    const auto& dest_dims = dest.shape();
                    if (dest_dims.at(dest_dims.size()) - 1 == dest_dim1) {
                        // If the outer dim is indeed dest_dim1, try running .contiguous()
                        dest.contiguous();
                        // Check to see that our call worked
                        if (dest.dim_stride(dest_dim1) == 1) {
                            // Dispatch again with the updated dest
                            return naive_matmul_v2_dispatch(lhs, lhs_dim0, lhs_dim1, lhs_coordinates,
                                                            rhs, rhs_dim0, rhs_dim1, rhs_coordinates,
                                                            dest, dest_dim0, dest_dim1, dest_coordinates);
                        }
                        else {
                            // We ran into some kind of unexpected error
                            if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
                                log_message(Log_Priority::WARNING, "TensorMatmul::naive_matmul_v2_dispatch",
                                    "Unexpected error when running dest.contiguous(). Falling back to v1.");
                            }
                            return false;
                        }
                    }
                    // If dest_dim1 is not the outer dim
                    else {
                        // We need to create a new destination Tensor to resolve
                        if constexpr (TENSORMATMUL_PREFER_TEMP_DESTINATION_OVER_EXCEPTION) {
                            if constexpr (TENSORMATMUL_NAIVE_MATMUL_V2_LOG_TEMP) {
                                log_message(Log_Priority::DEBUG, "TensorMatmul::naive_matmul_v2_dispatch",
                                    "Creating temporary destination Tensor"
                                );
                            }
                            // Create a temp destination
                            Tensor<T> temp_dest({dest.extent(dest_dim0), dest.extent(dest_dim1)});
                            // Dispatch again and check the result
                            if (naive_matmul_v2_dispatch(lhs, lhs_dim0, lhs_dim1, lhs_coordinates,
                                                         rhs, rhs_dim0, rhs_dim1, rhs_coordinates,
                                                         temp_dest, 0, 1, dest_coordinates)) {
                                // Check the original rank of dest and copy back
                                if (dest.rank() == 2) {
                                    // Copy the result back into dest
                                    dest.copy_from(temp_dest);
                                    return true;
                                }
                                // If rank > 2, copy back using dest_coordinates
                                else {
                                    TensorUtils_NS::copy_tensor_to_high_rank_tensor(temp_dest, 0, 1, temp_dest.offset(),
                                                                                    dest, dest_dim0, dest_dim1, dest_coordinates);
                                    return true;
                                }
                            }
                            else {
                                if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
                                    log_message(Log_Priority::WARNING, "TensorMatmul::naive_matmul_v2_dispatch",
                                        "Unexpected error when running creating a temporary destination Tensor. Falling back to v1.");
                                }
                                return false;
                            }
                        }
                        else {
                            if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
                                log_message(Log_Priority::WARNING, "TensorMatmul::naive_matmul_v2_dispatch",
                                    "Creating temporary destination Tensors is disabled. Falling back to v1.");
                            }
                            return false;
                        }
                    }
                }
                else if constexpr (TENSORMATMUL_PREFER_TEMP_DESTINATION_OVER_EXCEPTION) {
                    if constexpr (TENSORMATMUL_NAIVE_MATMUL_V2_LOG_TEMP) {
                        log_message(Log_Priority::DEBUG, "TensorMatmul::naive_matmul_v2_dispatch",
                            "Creating temporary destination Tensor"
                        );
                    }
                    // Create a temp destination
                    Tensor<T> temp_dest({dest.extent(dest_dim0), dest.extent(dest_dim1)});
                    // Dispatch again and check the result
                    if (naive_matmul_v2_dispatch(lhs, lhs_dim0, lhs_dim1, lhs_coordinates,
                                                 rhs, rhs_dim0, rhs_dim1, rhs_coordinates,
                                                 temp_dest, 0, 1, dest_coordinates)) {
                        // Check the original rank of dest and copy back
                        if (dest.rank() == 2) {
                            // Copy the result back into dest
                            dest.copy_from(temp_dest);
                            return true;
                        }
                        else {
                            // Use the helper function in TensorUtils to copy the data back into dest
                            TensorUtils_NS::copy_tensor_to_high_rank_tensor(temp_dest, 0, 1, temp_dest.offset(),
                                                                            dest, dest_dim0, dest_dim1, dest_coordinates);
                            return true;
                        }
                    }
                    else {
                        if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
                            log_message(Log_Priority::WARNING, "TensorMatmul::naive_matmul_v2_dispatch",
                                "Unexpected error when running creating a temporary destination Tensor. Falling back to v1.");
                        }
                        return false;
                    }
                }
                else {
                    if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
                        log_message(Log_Priority::WARNING, "TensorMatmul::naive_matmul_v2_dispatch",
                            "Executing .contiguous() and creating temporary destination Tensors are disabled. Falling back to v1.");
                    }
                    return false;
                }
            }
                // Handle the case where we need to convert the input AbstractTensors --> Tensors
            case NaiveMatmulV2Error::CONVERT_ABSTRACT: {
                // Check to see if we are allowed to convert AbstractTensor
                if constexpr (TENSORMATMUL_ENABLE_ABSTRACT_TENSOR_CONVERSION) {
                    if constexpr (TENSORMATMUL_NAIVE_MATMUL_V2_LOG_TEMP) {
                        log_message(Log_Priority::DEBUG, "TensorMatmul::naive_matmul_v2_dispatch",
                            "Creating temporary lhs and/or rhs via dynamic_cast --> to_tensor()"
                        );
                    }
                    // Attempt to dynamic cast lhs and rhs to Tensors
                    const Tensor<T>* lhs_ptr = dynamic_cast<const Tensor<T>*>(&lhs);
                    const Tensor<T>* rhs_ptr = dynamic_cast<const Tensor<T>*>(&rhs);
                    // Convert whichever (or both) failed to cast successfully
                    const Tensor<T> lhs_c = (lhs_ptr == nullptr) ? dynamic_cast<const ListTensorSlice<T>*>(&lhs)->to_tensor() : *lhs_ptr;
                    const Tensor<T> rhs_c = (rhs_ptr == nullptr) ? dynamic_cast<const ListTensorSlice<T>*>(&rhs)->to_tensor() : *rhs_ptr;
                    // Take the converted Tensors and dispatch again
                    return naive_matmul_v2_dispatch(lhs_c, lhs_dim0, lhs_dim1, lhs_coordinates,
                                                    rhs_c, rhs_dim0, rhs_dim1, rhs_coordinates,
                                                    dest, dest_dim0, dest_dim1, dest_coordinates);
                }
                else {
                    if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
                        log_message(Log_Priority::WARNING, "TensorMatmul::naive_matmul_v2_dispatch",
                            "AbstractTensor conversion is disabled. Falling back to v1.");
                    }
                    return false;
                }
            }
            // Handle the case where the input stride in dim1 != 1
            case NaiveMatmulV2Error::INPUT_STRIDE: {
                // Check to see if we are allowed to create temporary input copies
                if constexpr (TENSORMATMUL_NAIVE_MATMUL_V2_PREFER_TEMP_INPUT_COPY) {
                    // Since potential CONVERT_ABSTRACT issues are already taken care of we should be able to
                    // cast to Tensor without issue
                    // Attempt to dynamic cast lhs and rhs to Tensors
                    const Tensor<T>* lhs_ptr = dynamic_cast<const Tensor<T>*>(&lhs);
                    const Tensor<T>* rhs_ptr = dynamic_cast<const Tensor<T>*>(&rhs);
                    // Do a sanity check to make sure we aren't running into any unexpected casting issues
                    if (lhs_ptr == nullptr || rhs_ptr == nullptr) {
                        if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
                            log_message(Log_Priority::WARNING, "TensorMatmul::naive_matmul_v2_dispatch",
                                "Unexpected error when using dynamic_cast on input AbstractTensor. Falling back to v1.");
                        }
                        return false;
                    }
                    // Handle both Tensors requiring copies
                    if (lhs_ptr->dim_stride(lhs_dim1) != 1 && rhs_ptr->dim_stride(rhs_dim1) != 1) {
                        if constexpr (TENSORMATMUL_NAIVE_MATMUL_V2_LOG_TEMP) {
                            log_message(Log_Priority::DEBUG, "TensorMatmul::naive_matmul_v2_dispatch",
                                "Creating temporary lhs and rhs Tensors"
                            );
                        }
                        // Both of these Tensors are contiguous with dim_stride(dim1) == 1 by definition
                        Tensor<T> temp_lhs({lhs_ptr->extent(lhs_dim0), lhs_ptr->extent(lhs_dim1)});
                        // Use the helper from TensorUtils
                        TensorUtils_NS::copy_high_rank_abstract_to_tensor(lhs, lhs_dim0, lhs_dim1, lhs_coordinates,
                                                                          temp_lhs, 0, 1, temp_lhs.offset());
                        Tensor<T> temp_rhs({rhs_ptr->extent(rhs_dim0), rhs_ptr->extent(rhs_dim1)});
                        TensorUtils_NS::copy_high_rank_abstract_to_tensor(rhs, rhs_dim0, rhs_dim1, rhs_coordinates,
                                                                          temp_rhs, 0, 1, temp_rhs.offset());
                        // Since destination hasn't been rewritten, we should be able to return the result directly
                        return naive_matmul_v2_dispatch(temp_lhs, 0, 1, lhs_coordinates,
                                                        temp_rhs, 0, 1, rhs_coordinates,
                                                        dest, dest_dim0, dest_dim1, dest_coordinates);
                    }
                    // If only lhs needs to be copied
                    else if (lhs_ptr->dim_stride(lhs_dim1) != 1) {
                        if constexpr (TENSORMATMUL_NAIVE_MATMUL_V2_LOG_TEMP) {
                            log_message(Log_Priority::DEBUG, "TensorMatmul::naive_matmul_v2_dispatch",
                                "Creating temporary lhs Tensor"
                            );
                        }
                        Tensor<T> temp_lhs({lhs_ptr->extent(lhs_dim0), lhs_ptr->extent(lhs_dim1)});
                        TensorUtils_NS::copy_high_rank_abstract_to_tensor(lhs, lhs_dim0, lhs_dim1, lhs_coordinates,
                                                                          temp_lhs, 0, 1, temp_lhs.offset());
                        // Since destination hasn't been rewritten, we should be able to return the result directly
                        return naive_matmul_v2_dispatch(temp_lhs, 0, 1, lhs_coordinates,
                                                        rhs, rhs_dim0, rhs_dim1, rhs_coordinates,
                                                        dest, dest_dim0, dest_dim1, dest_coordinates);
                    }
                    // If only rhs needs to be copied
                    else {
                        if constexpr (TENSORMATMUL_NAIVE_MATMUL_V2_LOG_TEMP) {
                            log_message(Log_Priority::DEBUG, "TensorMatmul::naive_matmul_v2_dispatch",
                                "Creating temporary rhs Tensor"
                            );
                        }
                        Tensor<T> temp_rhs({rhs_ptr->extent(rhs_dim0), rhs_ptr->extent(rhs_dim1)});
                        TensorUtils_NS::copy_high_rank_abstract_to_tensor(rhs, rhs_dim0, rhs_dim1, rhs_coordinates,
                                                                          temp_rhs, 0, 1, temp_rhs.offset());
                        // Since destination hasn't been rewritten, we should be able to return the result directly
                        return naive_matmul_v2_dispatch(lhs, lhs_dim0, lhs_dim1, lhs_coordinates,
                                                        temp_rhs, 0, 1, rhs_coordinates,
                                                        dest, dest_dim0, dest_dim1, dest_coordinates);
                    }
                }
                else {
                    if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
                        log_message(Log_Priority::WARNING, "TensorMatmul::naive_matmul_v2_dispatch",
                            "Creating temporary input copies is disabled. Falling back to v1.");
                    }
                    return false;
                }
            }
            // Handle any unexpected / unimplemented errors
            default: {
                if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
                    log_message(Log_Priority::WARNING, "TensorMatmul::naive_matmul_v2_dispatch",
                        "Unknown error occurred in dispatch routing. Falling back to v1.");
                }
                return false;
            }
        }
    }
}
// NOLINTEND(bugprone-easily-swappable-parameters

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
 * @param destination_dim0 The first dim of the destination
 * @param destination_dim1 The second dim of the destination
 * @param destination_coordinates Base coordinates if we don't want to use 0
 */
template <typename T> 
requires std::is_arithmetic_v<T>
Tensor<T>& matmul(const AbstractTensor<T>& lhs, size_t lhs_dim0, size_t lhs_dim1, std::vector<size_t>& lhs_coordinates,
                  const AbstractTensor<T>& rhs, size_t rhs_dim0, size_t rhs_dim1, std::vector<size_t>& rhs_coordinates,
                  Tensor<T>& destination, size_t destination_dim0, size_t destination_dim1, std::vector<size_t>& destination_coordinates) {
    // Use the checker function instead of writing the check manually several times
    auto compat = lhs._can_matmul(lhs_dim0, lhs_dim1, rhs, rhs_dim0, rhs_dim1);
    if (!compat.has_value()) {
        throw std::invalid_argument(std::format("TensorMatmul::matmul: Unable to matmul. Error: {}.", compat.error()));
    }
    // Ensure lhs_coordinates and rhs_coordinates have the correct size
    if (lhs_coordinates.size() != lhs.rank() || rhs_coordinates.size() != rhs.rank()) {
        throw std::invalid_argument("TensorMatmul::matmul: Invalid coordinate vector(s) provided.\n");
    }
    // Ensure the destination Tensor has the right dimensions
    if (destination.extent(0) != lhs.extent(lhs_dim0) || destination.extent(1) != rhs.extent(rhs_dim1)) {
        throw std::invalid_argument(
            std::format(
                "TensorMatmul::matmul: Destination Tensor has shape [{}, {}] but should be [{}, {}].\n",
                destination.extent(0), destination.extent(1), lhs.extent(lhs_dim0), rhs.extent(rhs_dim1)
            )
        );
    }
    // Check to see if we want make the destination contiguous
    if constexpr (TENSORMATMUL_ENABLE_DESTINATION_TENSOR_CONTIGUOUS) {
        // Only run .contiguous() if destination_dim1 is the outer dim
        if ((destination.shape().at(destination.rank() - 1) == destination_dim1) && (destination.dim_stride(destination_dim1) != 1)) {
            destination.contiguous();
            // Do a sanity check to ensure the op was successful
            if (destination.dim_stride(1) != 1) {
                throw std::runtime_error("TensorMatmul::matmul: Unknown error when trying to call destination.contiguous().\n");
            }
        }
    }
    // Ensure that destination is unique before zeroing out its contents
    if (!destination.is_unique(lhs) || !destination.is_unique(rhs)) {
        // Ensure we are allowed to create temp destinations
        if constexpr (TENSORMATMUL_PREFER_TEMP_DESTINATION_OVER_EXCEPTION) {
            if constexpr (TENSORMATMUL_NAIVE_MATMUL_V2_LOG_TEMP) {
                log_message(Log_Priority::DEBUG, "TensorMatmul::matmul",
                    "Creating temporary destination Tensor"
                );
            }
            // temp_destination is contiguous with stride == 1 in dim1 by definition
            Tensor<T> temp_destination({destination.extent(destination_dim0), destination.extent(destination_dim1)});
            // Create some dummy coordinates
            std::vector<size_t> dummy_coordinates = std::vector<size_t>();
            matmul(lhs, lhs_dim0, lhs_dim1, lhs_coordinates,
                   rhs, rhs_dim0, rhs_dim1, rhs_coordinates,
                   temp_destination, 0, 1, dummy_coordinates);
            if (destination.rank() == 2) {
                return destination.copy_from(temp_destination);
            }
            else {
                TensorUtils_NS::copy_tensor_to_high_rank_tensor(temp_destination, 0, 1, temp_destination.offset(),
                                                                destination, destination_dim0, destination_dim1, destination_coordinates);
                return destination;
            }
        }
        else {
            throw std::invalid_argument("TensorMatmul::matmul: Destination Tensor failed is_unique() check.\n");
        }
    }
    // Perform additional checks if using naive matmul v2
    if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V2) {
        // Use the v2 dispatch and check for errors
        if (naive_matmul_v2_dispatch(lhs, lhs_dim0, lhs_dim1, lhs_coordinates,
                                     rhs, rhs_dim0, rhs_dim1, rhs_coordinates,
                                     destination, destination_dim0, destination_dim1, destination_coordinates)) {
            // If we are successful, just return destination
            return destination;
        }
        // Fall back to v1
        if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
            log_message(Log_Priority::WARNING, "TensorMatmul::matmul", "Error in v2 dispatch. Falling back to naive matmul v1.");
        }
        if (destination.rank() != 2) {
            throw std::invalid_argument("TensorMatmul::matmul: Destination Tensor rank != 2. Unable to safely fall back to v1.\n");
        }
        // Zero out the destination Tensor
        destination.fill(0);
        // Use the desired matmul impl to execute the operation
        _naive_matmul_impl_v1(lhs, lhs_dim0, lhs_dim1, lhs_coordinates,
                              rhs, rhs_dim0, rhs_dim1, rhs_coordinates,
                              destination);
        return destination;
    }
    else {
        if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V2 && TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
            log_message(Log_Priority::WARNING, "TensorMatmul::matmul", "Falling back to naive matmul v1.");
        }
        if (destination.rank() != 2) {
            throw std::invalid_argument("TensorMatmul::matmul: Destination Tensor rank != 2. Unable to safely fall back to v1.\n");
        }
        // Zero out the destination Tensor
        destination.fill(0);
        // Use the desired matmul impl to execute the operation
        _naive_matmul_impl_v1(lhs, lhs_dim0, lhs_dim1, lhs_coordinates,
                              rhs, rhs_dim0, rhs_dim1, rhs_coordinates,
                              destination);
        return destination;
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
 */
template <typename T> 
requires std::is_arithmetic_v<T>
Tensor<T> matmul(const AbstractTensor<T>& lhs, size_t lhs_dim0, size_t lhs_dim1, std::vector<size_t>& lhs_coordinates,
                 const AbstractTensor<T>& rhs, size_t rhs_dim0, size_t rhs_dim1, std::vector<size_t>& rhs_coordinates) {
    // Use the checker function instead of writing the check manually several times
    auto compat = lhs._can_matmul(lhs_dim0, lhs_dim1, rhs, rhs_dim0, rhs_dim1);
    if (!compat.has_value()) {
        throw std::invalid_argument(std::format("TensorMatmul::matmul: Unable to matmul. Error: {}.", compat.error()));
    }
    // Ensure lhs_coordinates and rhs_coordinates have the correct size
    if (lhs_coordinates.size() != lhs.rank() || rhs_coordinates.size() != rhs.rank()) {
        throw std::invalid_argument("TensorMatmul::matmul: Invalid coordinate vector(s) provided.\n");
    }
    // Create a Tensor to store the result
    Tensor<T> result({lhs.extent(lhs_dim0), rhs.extent(rhs_dim1)});
    // Create dummy coordinates for result, since we know the actual offset must be zero
    std::vector<size_t> dummy_coordinates = std::vector<size_t>();
    // Use the desired matmul impl to execute the operation
    matmul(lhs, lhs_dim0, lhs_dim1, lhs_coordinates,
           rhs, rhs_dim0, rhs_dim1, rhs_coordinates,
           result, 0, 1, dummy_coordinates);
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
 * NOTE: If lhs, rhs, or dest is rank > 2, we use all zeroes for other dim coordinates
 */
template <typename T> 
requires std::is_arithmetic_v<T>
Tensor<T>& matmul(const AbstractTensor<T>& lhs, size_t lhs_dim0, size_t lhs_dim1,
                  const AbstractTensor<T>& rhs, size_t rhs_dim0, size_t rhs_dim1,
                  Tensor<T>& destination) {
    // Use the checker function instead of writing the check manually several times
    auto compat = lhs._can_matmul(lhs_dim0, lhs_dim1, rhs, rhs_dim0, rhs_dim1);
    if (!compat.has_value()) {
        throw std::invalid_argument(std::format("TensorMatmul::matmul: Unable to matmul. Error: {}.", compat.error()));
    }
    // Ensure the destination Tensor has the right dimensions
    if (destination.extent(0) != lhs.extent(lhs_dim0) || destination.extent(1) != rhs.extent(rhs_dim1)) {
        throw std::invalid_argument(
            std::format(
                "TensorMatmul::matmul: Destination Tensor has shape [{}, {}] but should be [{}, {}].\n",
                destination.extent(0), destination.extent(1), lhs.extent(lhs_dim0), rhs.extent(rhs_dim1)
            )
        );
    }
    // Since we don't specify dims for destination, it must be rank == 2
    if (destination.rank() != 2) {
        throw std::invalid_argument(
            std::format(
                "TensorMatmul::matmul: Destination Tensor's rank != 2; got {}.", destination.rank()
            )
        );
    }
    // Check to see if we want make the destination contiguous
    if constexpr (TENSORMATMUL_ENABLE_DESTINATION_TENSOR_CONTIGUOUS) {
        if (destination.dim_stride(1) != 1) {
            destination.contiguous();
            // Do a sanity check to ensure the op was successful
            if (destination.dim_stride(1) != 1) {
                throw std::runtime_error("TensorMatmul::matmul: Unknown error when trying to call destination.contiguous().\n");
            }
        }
    }
    // Ensure that destination is unique before zeroing out its contents
    if (!destination.is_unique(lhs) || !destination.is_unique(rhs)) {
        // Ensure we are allowed to create temp destinations
        if constexpr (TENSORMATMUL_PREFER_TEMP_DESTINATION_OVER_EXCEPTION) {
            if constexpr (TENSORMATMUL_NAIVE_MATMUL_V2_LOG_TEMP) {
                log_message(Log_Priority::DEBUG, "TensorMatmul::matmul",
                    "Creating temporary destination Tensor"
                );
            }
            // temp_destination is contiguous with stride == 1 in dim1 by definition
            Tensor<T> temp_destination = destination.clone();
            matmul(lhs, lhs_dim0, lhs_dim1,
                   rhs, rhs_dim0, rhs_dim1,
                   temp_destination);
            return destination.copy_from(temp_destination);
        }
        else {
            throw std::invalid_argument("TensorMatmul::matmul: Destination Tensor failed is_unique() check.\n");
        }
    }
    // Perform additional checks if using naive matmul v2
    if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V2) {
        // Dispatch requires coordinate vectors, so ensure lhs and rhs have them
        std::vector<size_t> lhs_coordinates(lhs.rank(), 0);
        std::vector<size_t> rhs_coordinates(rhs.rank(), 0);
        // Destination must have rank == 2 due to the function signature
        std::vector<size_t> dummy_coordinates = std::vector<size_t>();
        // Use the v2 dispatch and check for errors
        if (naive_matmul_v2_dispatch(lhs, lhs_dim0, lhs_dim1, lhs_coordinates,
                                     rhs, rhs_dim0, rhs_dim1, rhs_coordinates,
                                     destination, 0, 1, dummy_coordinates)) {
            // If we are successful, just return destination
            return destination;
        }
        // Fall back to v1
        if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
            log_message(Log_Priority::WARNING, "TensorMatmul::matmul", "Error in v2 dispatch. Falling back to naive matmul v1.");
        }
        // Zero out the destination Tensor
        destination.fill(0);
        // Use the desired matmul impl to execute the operation
        _naive_matmul_impl_v1(lhs, lhs_dim0, lhs_dim1, lhs_coordinates,
                              rhs, rhs_dim0, rhs_dim1, rhs_coordinates,
                              destination);
        return destination;
    }
    else {
        if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V2 && TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
            log_message(Log_Priority::WARNING, "TensorMatmul::matmul", "Falling back to naive matmul v1.");
        }
        // Create reusable std::vector to use with Tensor.at
        std::vector<size_t> lhs_coordinates(lhs.rank(), 0);
        std::vector<size_t> rhs_coordinates(rhs.rank(), 0);
        // Zero out the destination Tensor
        destination.fill(0);
        // Use the desired matmul impl to execute the operation
        _naive_matmul_impl_v1(lhs, lhs_dim0, lhs_dim1, lhs_coordinates,
                              rhs, rhs_dim0, rhs_dim1, rhs_coordinates,
                              destination);
        return destination;
    }
}

/**
 * Perform a matmul across two Tensors, with their target dims specified
 * @param lhs Lefthand Tensor to matmul
 * @param lhs_dim0 The first dim of lhs for the matmul
 * @param lhs_dim1 The second dim of lhs for the matmul
 * @param rhs Righthand Tensor to matmul
 * @param rhs_dim0 The first dim of the rhs for the matmul
 * @param rhs_dim1 The second dim of the rhs for the matmul
 * NOTE: If lhs or rhs is rank > 2, we use all zeroes for other dim coordinates
 */
template <typename T> 
requires std::is_arithmetic_v<T>
Tensor<T> matmul(const AbstractTensor<T>& lhs, size_t lhs_dim0, size_t lhs_dim1,
                 const AbstractTensor<T>& rhs, size_t rhs_dim0, size_t rhs_dim1) {
    // Use the checker function instead of writing the check manually several times
    auto compat = lhs._can_matmul(lhs_dim0, lhs_dim1, rhs, rhs_dim0, rhs_dim1);
    if (!compat.has_value()) {
        throw std::invalid_argument(std::format("TensorMatmul::matmul: Unable to matmul. Error: {}.", compat.error()));
    }
    // Create a Tensor to store the result
    Tensor<T> result({lhs.extent(lhs_dim0), rhs.extent(rhs_dim1)});
    // Use the desired matmul impl to execute the operation
    matmul(lhs, lhs_dim0, lhs_dim1,
           rhs, rhs_dim0, rhs_dim1,
           result);
    // Use RVO to return the result without copying
    return result;
}

/**
 * Perform a matmul across two Tensors, with their target dims specified
 * @param lhs Lefthand Tensor to matmul
 * @param rhs Righthand Tensor to matmul
 * @param destination Tensor reference to write the result to
 * NOTE: This requires lhs, rhs, and destination to be rank == 2
 */
template <typename T> 
requires std::is_arithmetic_v<T>
Tensor<T>& matmul(const AbstractTensor<T>& lhs,
                  const AbstractTensor<T>& rhs,
                  Tensor<T>& destination) {
    // Assume we want dims 0 and 1 from lhs and rhs and that their rank must == 2
    if (lhs.rank() != 2 || rhs.rank() != 2 || destination.rank() != 2) {
        throw std::invalid_argument(
            std::format(
                "TensorMatmul::matmul: Invalid Tensor rank for matmul(lhs, rhs, destination). lhs.rank == {}. rhs.rank == {}. destination.rank == {}",
                    lhs.rank(), rhs.rank(), destination.rank()));
    }
    auto compat = lhs._can_matmul(0, 1, rhs, 0, 1);
    if (!compat.has_value()) {
        throw std::invalid_argument(std::format("TensorMatmul::matmul: Unable to matmul. Error: {}.", compat.error()));
    }
    // Ensure the destination Tensor has the right dimensions
    if (destination.extent(0) != lhs.extent(0) || destination.extent(1) != rhs.extent(1)) {
        throw std::invalid_argument(
            std::format(
                "TensorMatmul::matmul: Destination Tensor has shape [{}, {}] but should be [{}, {}].\n",
                destination.extent(0), destination.extent(1), lhs.extent(0), rhs.extent(1)
            )
        );
    }
    // Check to see if we want make the destination contiguous
    if constexpr (TENSORMATMUL_ENABLE_DESTINATION_TENSOR_CONTIGUOUS) {
        if (destination.dim_stride(1) != 1) {
            destination.contiguous();
            // Do a sanity check to ensure the op was successful
            if (destination.dim_stride(1) != 1) {
                throw std::runtime_error("TensorMatmul::matmul: Unknown error when trying to call destination.contiguous().\n");
            }
        }
    }
    // Ensure that destination is unique before zeroing out its contents
    if (!destination.is_unique(lhs) || !destination.is_unique(rhs)) {
        // Ensure we are allowed to create temp destinations
        if constexpr (TENSORMATMUL_PREFER_TEMP_DESTINATION_OVER_EXCEPTION) {
            if constexpr (TENSORMATMUL_NAIVE_MATMUL_V2_LOG_TEMP) {
                log_message(Log_Priority::DEBUG, "TensorMatmul::matmul",
                    "Creating temporary destination Tensor"
                );
            }
            // temp_destination is contiguous with stride == 1 in dim1 by definition
            Tensor<T> temp_destination = destination.clone();
            matmul(lhs, rhs, temp_destination);
            return destination.copy_from(temp_destination);
        }
        else {
            throw std::invalid_argument("TensorMatmul::matmul: Destination Tensor failed is_unique() check.\n");
        }
    }
    // Perform additional checks if using naive matmul v2
    if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V2) {
        // Dispatch requires coordinate vectors. Since this variant of matmul requires all Tensors to be
        // rank == 2, we can create a blank vector and pass the same ref multiple times
        std::vector<size_t> dummy_coordinates = std::vector<size_t>();
        // Use the v2 dispatch and check for errors
        if (naive_matmul_v2_dispatch(lhs, 0, 1, dummy_coordinates,
                                     rhs, 0, 1, dummy_coordinates,
                                     destination, 0, 1, dummy_coordinates)) {
            // If we are successful, just return destination
            return destination;
        }
        // Fall back to v1
        if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
            log_message(Log_Priority::WARNING, "TensorMatmul::matmul", "Error in v2 dispatch. Falling back to naive matmul v1.");
        }
        // Zero out the content of the destination
        destination.fill(0);
        // Use the desired matmul impl to execute the operation
        _naive_matmul_impl_v1(lhs, rhs, destination);
    }
    else {
        if constexpr (TENSORMATMUL_ENABLE_NAIVE_MATMUL_V2 && TENSORMATMUL_ENABLE_NAIVE_MATMUL_V1_FALLBACK_WARNING) {
            log_message(Log_Priority::WARNING, "TensorMatmul::matmul", "Falling back to naive matmul v1.");
        }
        // Zero out the content of the destination
        destination.fill(0);
        // Use the desired matmul impl to execute the operation
        _naive_matmul_impl_v1(lhs, rhs, destination);
    }
    return destination;
}

/**
 * Perform a matmul across two Tensors, with their target dims specified
 * @param lhs Lefthand Tensor to matmul
 * @param rhs Righthand Tensor to matmul
 * NOTE: This requires lhs and rhs to be rank == 2
 */
template <typename T> 
requires std::is_arithmetic_v<T>
Tensor<T> matmul(const AbstractTensor<T>& lhs,
                 const AbstractTensor<T>& rhs) {
    // Assume we want dims 0 and 1 from lhs and rhs and that their rank must == 2
    if (lhs.rank() != 2 || rhs.rank() != 2) {
        throw std::invalid_argument(
            std::format(
                "TensorMatmul::matmul: Invalid Tensor rank for matmul(lhs, rhs). lhs.rank == {}. rhs.rank == {}.",
                    lhs.rank(), rhs.rank()));
    }
    auto compat = lhs._can_matmul(0, 1, rhs, 0, 1);
    if (!compat.has_value()) {
        throw std::invalid_argument(std::format("TensorMatmul::matmul: Unable to matmul. Error: {}.", compat.error()));
    }
    // Create a Tensor to store the result
    Tensor<T> result({lhs.extent(0), rhs.extent(1)});
    // Use the desired matmul impl to execute the operation
    matmul(lhs, rhs, result);
    // Use RVO to return the result without copying
    return result;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

}; // namespace TensorMatmul_NS

#endif
