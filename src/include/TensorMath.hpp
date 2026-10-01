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
 * @version: 2026-09-28
 *
 * Notes:
 * This file is full of boilerplate code; however, given the variance on __restrict__ and the ability
 * to add contiguous blocks of memory, strided arrays, or simply add a scalar value to a Tensor, it seemed
 * like the best way to handle was to be explicit. Splitting this out from Tensor.hpp made the most sense to me.
 * 
 * Also note that this file has AI tool usage with the _(add|sub|mul)_overflow functions. I've called that out
 * in each function signature.
 * 
 * Due to the high amount of boilerplate, I've only documented each function type once, trying
 * to explain what each path does and why it was designed that way.
 *
 * TODO: Continue adding functionality 
 */

#ifndef TENSOR_MATH_HPP
#define TENSOR_MATH_HPP

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
#include "Numerics.hpp"

namespace TensorMath_NS {

/**
 * Compiler-independent check for addition overflow / underflow
 * @param a First value
 * @param b Second value
 * @param result Pointer to store the result in
 * @returns Returns true if the operation would cause overflow / underflow
 */
template <typename T>
requires std::is_integral_v<T> && std::is_unsigned_v<T>
[[nodiscard]] inline bool _add_overflow_unsigned(T a, T b, T* result) {
    *result = a + b;
    // Wraparound check
    return *result < a;
}

/**
 * Compiler-independent check for addition overflow / underflow
 * @param a First value
 * @param b Second value
 * @param result Pointer to store the result in
 * @returns Returns true if the operation would cause overflow / underflow
 * DISCLAIMER: AI tools were used to refine the overflow / underflow logic. 
 * Testing revealed that signed int UB can be optimized out and I wasn't
 * able to figure this one out on my own.
 */
template <typename T> 
requires std::is_integral_v<T> && std::is_signed_v<T>
[[nodiscard]] inline bool _add_overflow_signed(T a, T b, T* result) {
    // Get the corresponding unsigned type
    using U = std::make_unsigned_t<T>;
    U res_u = static_cast<U>(a) + static_cast<U>(b);
    *result = static_cast<T>(res_u);
    // Overflow occurs if and only if a and b have the same sign, 
    // but the result of a + b has a different sign
    return ((a ^ *result) & (b ^ *result)) < 0;
}

/**
 * Compiler-independent check for subtraction overflow / underflow
 * @param a First value
 * @param b Second value
 * @param result Pointer to store the result in
 * @returns Returns true if the operation would cause overflow / underflow
 */
template <typename T>
requires std::is_integral_v<T> && std::is_unsigned_v<T>
[[nodiscard]] inline bool _sub_overflow_unsigned(T a, T b, T* result) {
    *result = a - b;
    // By nature of being unsigned, if a is less than b, we will underflow
    return a < b;
}

/**
 * Compiler-independent check for subtraction overflow / underflow
 * @param a First value
 * @param b Second value
 * @param result Pointer to store the result in
 * @returns Returns true if the operation would cause overflow / underflow
 * DISCLAIMER: AI tools were used to refine the overflow / underflow logic. 
 * Testing revealed that signed int UB can be optimized out and I wasn't
 * able to figure this one out on my own.
 */
template <typename T> 
requires std::is_integral_v<T> && std::is_signed_v<T>
[[nodiscard]] inline bool _sub_overflow_signed(T a, T b, T* result) {
    // Get the corresponding unsigned type
    using U = std::make_unsigned_t<T>;
    U res_u = static_cast<U>(a) - static_cast<U>(b);
    *result = static_cast<T>(res_u);
    // Check if a and b have different signs
    // Check if a and result have different signs
    return ((a ^ b) & (a ^ *result)) < 0;
}

/**
 * Compiler-independent check for multiplication overflow / underflow
 * @param a First value
 * @param b Second value
 * @param result Pointer to store the result in
 * @returns Returns true if the operation would cause overflow / underflow
 * DISCLAIMER: AI tools were used to refine the overflow / underflow logic. 
 * I wasn't able to figure this one out on my own.
 */
template <typename T> 
requires std::is_integral_v<T> && std::is_unsigned_v<T>
[[nodiscard]] inline bool _mul_overflow_unsigned(T a, T b, T* result) {
    // Calculate the bit boundaries for type T and split in half
    constexpr size_t HALF_BITS = (sizeof(T) * 8) / 2;
    constexpr T HALF_MASK = (static_cast<T>(1) << HALF_BITS) - 1;

    // Split operands into high and low halves
    T a_hi = a >> HALF_BITS;
    T a_lo = a & HALF_MASK;
    T b_hi = b >> HALF_BITS;
    T b_lo = b & HALF_MASK;

    // Compute partial products (all fit safely within type T)
    T p0 = a_lo * b_lo;
    T p1 = a_hi * b_lo;
    T p2 = a_lo * b_hi;
    T p3 = a_hi * b_hi;

    T p1_lo = p1 & HALF_MASK;
    T p1_hi = p1 >> HALF_BITS;
    T p2_lo = p2 & HALF_MASK;
    T p2_hi = p2 >> HALF_BITS;

    // Accumulate carries into the upper half
    T mid_sum = (p0 >> HALF_BITS) + p1_lo + p2_lo;
    T carry_to_hi = (mid_sum >> HALF_BITS) + p1_hi + p2_hi + p3;

    // Unconditional store for SIMD vectorization
    *result = a * b;

    // Overflow occurs if any bits carry beyond the B-bit boundary
    return carry_to_hi != 0;
}

/**
 * Compiler-independent check for multiplication overflow / underflow
 * @param a First value
 * @param b Second value
 * @param result Pointer to store the result in
 * @returns Returns true if the operation would cause overflow / underflow
 * DISCLAIMER: AI tools were used to refine the overflow / underflow logic. 
 * I wasn't able to figure this one out on my own.
 */
template <typename T> 
requires std::is_integral_v<T> && std::is_signed_v<T>
[[nodiscard]] inline bool _mul_overflow_signed(T a, T b, T* result) {
    // Get the corresponding unsigned type
    using U = std::make_unsigned_t<T>;

    // Unconditional store of wrapped product
    U res_u = static_cast<U>(a) * static_cast<U>(b);
    *result = static_cast<T>(res_u);

    // Absolute values in unsigned space
    U ua = (a < 0) ? (0U - static_cast<U>(a)) : static_cast<U>(a);
    U ub = (b < 0) ? (0U - static_cast<U>(b)) : static_cast<U>(b);

    // Use the unsigned overflow detection
    U u_prod = 0;
    bool u_ovf = _mul_overflow_unsigned(ua, ub, &u_prod);

    // Same sign: max allowed magnitude is MAX (e.g. 127 for int8_t)
    // Diff sign: max allowed magnitude is |MIN| = MAX + 1 (e.g. 128 for int8_t)
    bool same_sign = (a ^ b) >= 0;
    U max_allowed = same_sign ? static_cast<U>(std::numeric_limits<T>::max())
                              : (static_cast<U>(std::numeric_limits<T>::max()) + 1U);

    return u_ovf | (u_prod > max_allowed);
}

/**
 * Helper check for signed integer division overflow (e.g., INT_MIN / -1).
 * @param lhs Numerator value
 * @param rhs Denominator value
 * @returns True if dividing lhs by rhs will cause signed integer overflow
 */
template <typename T>
requires std::is_integral_v<T> && std::is_signed_v<T>
[[nodiscard]] inline bool _is_signed_div_overflow(T lhs, T rhs) {
    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        return rhs == -1 && lhs == std::numeric_limits<T>::min();
    }
    return false;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters, cppcoreguidelines-pro-bounds-pointer-arithmetic)
template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Add the contents of two Tensors together, storing in the data block of another Tensor
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param lhs_dim0_stride Stride of the lhs outer dim
 * @param rhs_ptr Pointer to the raw data underlying the Tensor
 * @param rhs_offset Offset to the beginning of the data block
 * @param rhs_dim0_stride Stride of the rhs outer dim
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param result_dim0_stride Stride of the result outer dim
 * @param dim0_extent Outer dim size for all three Tensors
 * @param dim1_extent Inner dim size for all three Tensors
 * NOTE: This version of add should only be used when we are sure that lhs_ptr, rhs_ptr, and
 * result_ptr absolutely do NOT overlap. Inner dim stride must equal 1.
 * This method should be used when we are adding two TensorSlices with variable strides.
 */
inline void _safe_2d_tensor_add_contiguous_v(const T* __restrict__ lhs_ptr, size_t lhs_offset, size_t lhs_dim0_stride,
                                             const T* __restrict__ rhs_ptr, size_t rhs_offset, size_t rhs_dim0_stride,
                                             T* __restrict__ result_ptr, size_t result_offset, size_t result_dim0_stride,
                                             size_t dim0_extent, size_t dim1_extent) {
    // Handle the floating point types first, which have defined behaviors for overflow / underflow.
    // We perform the operation and then scan for NaN or inf afterwards
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
            const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                result_ptr[result_row + j] = lhs_ptr[lhs_row + j] + rhs_ptr[rhs_row + j];
            }
        }
        bool overflow = false;
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                T result = result_ptr[result_row + j];
                overflow |= !std::isfinite(result);
            }
        }
        if (overflow) {
            throw std::overflow_error("TensorMath::_safe_2d_tensor_add_contiguous_v: Overflow / underflow detected.\n");
        }
    }
    // For integer types, try to discover a double width type and do the operation inside that
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < dim0_extent; ++i) {
                const size_t result_row = result_offset + (result_dim0_stride * i);
                const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
                for (size_t j = 0; j < dim1_extent; ++j) {
                    accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_row + j]) + static_cast<accumulator_t>(rhs_ptr[rhs_row + j]);
                    overflow |= (result > MAX_VAL || result < MIN_VAL);
                    result_ptr[result_row + j] = static_cast<T>(result);
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_add_contiguous_v: Overflow / underflow detected.\n");
            }
        }
        // If there isn't a double-width type, then use the safe functions for the operation instead
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _add_overflow_unsigned(lhs_ptr[lhs_row + j], rhs_ptr[rhs_row + j], &(result_ptr[result_row + j]));
                    }
                }
            }
            else {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _add_overflow_signed(lhs_ptr[lhs_row + j], rhs_ptr[rhs_row + j], &(result_ptr[result_row + j]));
                    }
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_add_contiguous_v: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Add a scalar value to each element in a Tensor, storing in the data block of another Tensor
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param lhs_dim0_stride Stride of the lhs outer dim
 * @param rhs Scalar value to add to each value in lhs
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param result_dim0_stride Stride of the result outer dim
 * @param dim0_extent Outer dim size for all three Tensors
 * @param dim1_extent Inner dim size for all three Tensors
 * NOTE: This version of add should only be used when we are sure that lhs_ptr and result_ptr
 * absolutely do NOT overlap. Inner dim stride must equal 1.
 * This method should be used when adding a scalar to a TensorSlice with variable strides.
 */
inline void _safe_2d_tensor_add_contiguous_v(const T* __restrict__ lhs_ptr, size_t lhs_offset, size_t lhs_dim0_stride,
                                             const T rhs,
                                             T* __restrict__ result_ptr, size_t result_offset, size_t result_dim0_stride,
                                             size_t dim0_extent, size_t dim1_extent) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                result_ptr[result_row + j] = lhs_ptr[lhs_row + j] + rhs;
            }
        }
        bool overflow = false;
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                T result = result_ptr[result_row + j];
                overflow |= !std::isfinite(result);
            }
        }
        if (overflow) {
            throw std::overflow_error("TensorMath::_safe_2d_tensor_add_contiguous_v: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < dim0_extent; ++i) {
                const size_t result_row = result_offset + (result_dim0_stride * i);
                const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                for (size_t j = 0; j < dim1_extent; ++j) {
                    accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_row + j]) + static_cast<accumulator_t>(rhs);
                    overflow |= (result > MAX_VAL || result < MIN_VAL);
                    result_ptr[result_row + j] = static_cast<T>(result);
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_add_contiguous_v: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _add_overflow_unsigned(lhs_ptr[lhs_row + j], rhs, &(result_ptr[result_row + j]));
                    }
                }
            }
            else {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _add_overflow_signed(lhs_ptr[lhs_row + j], rhs, &(result_ptr[result_row + j]));
                    }
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_add_contiguous_v: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Add the contents of two Tensors together, storing in the data block of another Tensor
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param lhs_dim0_stride Stride of the lhs outer dim
 * @param rhs_ptr Pointer to the raw data underlying the Tensor
 * @param rhs_offset Offset to the beginning of the data block
 * @param rhs_dim0_stride Stride of the rhs outer dim
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param result_dim0_stride Stride of the result outer dim
 * @param dim0_extent Outer dim size for all three Tensors
 * @param dim1_extent Inner dim size for all three Tensors
 * NOTE: There is flexibility in different strides for each Tensor; however, the inner dim stride must == 1.
 * Memory blocks may overlap safely.
 */
inline void _safe_2d_tensor_add_contiguous(const T* lhs_ptr, size_t lhs_offset, size_t lhs_dim0_stride,
                                           const T* rhs_ptr, size_t rhs_offset, size_t rhs_dim0_stride,
                                           T* result_ptr, size_t result_offset, size_t result_dim0_stride,
                                           size_t dim0_extent, size_t dim1_extent) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
            const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                result_ptr[result_row + j] = lhs_ptr[lhs_row + j] + rhs_ptr[rhs_row + j];
            }
        }
        bool overflow = false;
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                T result = result_ptr[result_row + j];
                overflow |= !std::isfinite(result);
            }
        }
        if (overflow) {
            throw std::overflow_error("TensorMath::_safe_2d_tensor_add_contiguous: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < dim0_extent; ++i) {
                const size_t result_row = result_offset + (result_dim0_stride * i);
                const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
                for (size_t j = 0; j < dim1_extent; ++j) {
                    accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_row + j]) + static_cast<accumulator_t>(rhs_ptr[rhs_row + j]);
                    overflow |= (result > MAX_VAL || result < MIN_VAL);
                    result_ptr[result_row + j] = static_cast<T>(result);
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_add_contiguous: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _add_overflow_unsigned(lhs_ptr[lhs_row + j], rhs_ptr[rhs_row + j], &(result_ptr[result_row + j]));
                    }
                }
            }
            else {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _add_overflow_signed(lhs_ptr[lhs_row + j], rhs_ptr[rhs_row + j], &(result_ptr[result_row + j]));
                    }
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_add_contiguous: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Add a scalar value to each element in a Tensor, storing in the data block of another Tensor
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param lhs_dim0_stride Stride of the lhs outer dim
 * @param rhs Scalar value to add to each value in lhs
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param result_dim0_stride Stride of the result outer dim
 * @param dim0_extent Outer dim size for all three Tensors
 * @param dim1_extent Inner dim size for all three Tensors
 * NOTE: There is flexibility in different strides for each Tensor; however, the inner dim stride must == 1.
 * Memory blocks may overlap safely.
 */
inline void _safe_2d_tensor_add_contiguous(const T* lhs_ptr, size_t lhs_offset, size_t lhs_dim0_stride,
                                           const T rhs,
                                           T* result_ptr, size_t result_offset, size_t result_dim0_stride,
                                           size_t dim0_extent, size_t dim1_extent) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                result_ptr[result_row + j] = lhs_ptr[lhs_row + j] + rhs;
            }
        }
        bool overflow = false;
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                T result = result_ptr[result_row + j];
                overflow |= !std::isfinite(result);
            }
        }
        if (overflow) {
            throw std::overflow_error("TensorMath::_safe_2d_tensor_add_contiguous: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < dim0_extent; ++i) {
                const size_t result_row = result_offset + (result_dim0_stride * i);
                const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                for (size_t j = 0; j < dim1_extent; ++j) {
                    accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_row + j]) + static_cast<accumulator_t>(rhs);
                    overflow |= (result > MAX_VAL || result < MIN_VAL);
                    result_ptr[result_row + j] = static_cast<T>(result);
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_add_contiguous: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _add_overflow_unsigned(lhs_ptr[lhs_row + j], rhs, &(result_ptr[result_row + j]));
                    }
                }
            }
            else {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _add_overflow_signed(lhs_ptr[lhs_row + j], rhs, &(result_ptr[result_row + j]));
                    }
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_add_contiguous: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Add the contents of two 1D Tensors together, storing in the data block of another Tensor
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param rhs_ptr Pointer to the raw data underlying the Tensor
 * @param rhs_offset Offset to the beginning of the data block
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param elements Number of elements to traverse
 * NOTE: This can only be used with contiguous memory blocks, so the caller must ensure that
 * if we are using a TensorSlice, that its memory is contiguous. Also note that this version
 * of add should only be used when we are sure that lhs_ptr, rhs_ptr, and result_ptr
 * absolutely do NOT overlap.
 */
inline void _safe_tensor_add_contiguous_v(const T* __restrict__ lhs_ptr, size_t lhs_offset, 
                                          const T* __restrict__ rhs_ptr, size_t rhs_offset,
                                          T* __restrict__ result_ptr, size_t result_offset, size_t elements) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < elements; ++i) {
            // When we have fully contiguous blocks of memory, traverse linearly with a single loop
            result_ptr[result_offset + i] = lhs_ptr[lhs_offset + i] + rhs_ptr[rhs_offset + i];
        }
        // Use the any_of to traverse the data and check for NaN and inf
        if (std::any_of(result_ptr + result_offset, result_ptr + result_offset + elements, [](T result) {
            return !std::isfinite(result);
        })) {
            throw std::overflow_error("TensorMath::_safe_tensor_add_contiguous_v: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < elements; ++i) {
                accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_offset + i]) + static_cast<accumulator_t>(rhs_ptr[rhs_offset + i]);
                overflow |= (result > MAX_VAL || result < MIN_VAL);
                result_ptr[result_offset + i] = static_cast<T>(result);
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_add_contiguous_v: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _add_overflow_unsigned(lhs_ptr[lhs_offset + i], rhs_ptr[rhs_offset + i], &(result_ptr[result_offset + i]));
                }
            }
            else {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _add_overflow_signed(lhs_ptr[lhs_offset + i], rhs_ptr[rhs_offset + i], &(result_ptr[result_offset + i]));
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_add_contiguous_v: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Add a scalar value to a 1D Tensor
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param rhs Value to add to each index
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param elements Number of elements to traverse
 * NOTE: This can only be used with contiguous memory blocks, so the caller must ensure that
 * if we are using a TensorSlice, that its memory is contiguous. This version
 * should only be used when lhs_ptr and result_ptr do NOT overlap.
 */
inline void _safe_tensor_add_contiguous_v(const T* __restrict__ lhs_ptr, size_t lhs_offset,
                                          const T rhs,
                                          T* __restrict__ result_ptr, size_t result_offset, size_t elements) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < elements; ++i) {
            result_ptr[result_offset + i] = lhs_ptr[lhs_offset + i] + rhs;
        }
        if (std::any_of(result_ptr + result_offset, result_ptr + result_offset + elements, [](T result) {
            return !std::isfinite(result);
        })) {
            throw std::overflow_error("TensorMath::_safe_tensor_add_contiguous_v: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < elements; ++i) {
                accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_offset + i]) + static_cast<accumulator_t>(rhs);
                overflow |= (result > MAX_VAL || result < MIN_VAL);
                result_ptr[result_offset + i] = static_cast<T>(result);
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_add_contiguous_v: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _add_overflow_unsigned(lhs_ptr[lhs_offset + i], rhs, &(result_ptr[result_offset + i]));
                }
            }
            else {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _add_overflow_signed(lhs_ptr[lhs_offset + i], rhs, &(result_ptr[result_offset + i]));
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_add_contiguous_v: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Add the contents of two 1D Tensors together, storing in the data block of another Tensor
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param rhs_ptr Pointer to the raw data underlying the Tensor
 * @param rhs_offset Offset to the beginning of the data block
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param elements Number of elements to traverse
 * NOTE: This can only be used with contiguous memory blocks, so the caller must ensure that
 * if we are using a TensorSlice, that its memory is contiguous.
 */
inline void _safe_tensor_add_contiguous(const T* lhs_ptr, size_t lhs_offset, 
                                        const T* rhs_ptr, size_t rhs_offset,
                                        T* result_ptr, size_t result_offset, size_t elements) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < elements; ++i) {
            result_ptr[result_offset + i] = lhs_ptr[lhs_offset + i] + rhs_ptr[rhs_offset + i];
        }
        if (std::any_of(result_ptr + result_offset, result_ptr + result_offset + elements, [](T result) {
            return !std::isfinite(result);
        })) {
            throw std::overflow_error("TensorMath::_safe_tensor_add_contiguous: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < elements; ++i) {
                accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_offset + i]) + static_cast<accumulator_t>(rhs_ptr[rhs_offset + i]);
                overflow |= (result > MAX_VAL || result < MIN_VAL);
                result_ptr[result_offset + i] = static_cast<T>(result);
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_add_contiguous: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _add_overflow_unsigned(lhs_ptr[lhs_offset + i], rhs_ptr[rhs_offset + i], &(result_ptr[result_offset + i]));
                }
            }
            else {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _add_overflow_signed(lhs_ptr[lhs_offset + i], rhs_ptr[rhs_offset + i], &(result_ptr[result_offset + i]));
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_add_contiguous: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Add a scalar value to a 1D Tensor
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param rhs Value to add to each index
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param elements Number of elements to traverse
 * NOTE: This can only be used with contiguous memory blocks, so the caller must ensure that
 * if we are using a TensorSlice, that its memory is contiguous.
 */
inline void _safe_tensor_add_contiguous(const T* lhs_ptr, size_t lhs_offset, 
                                        const T rhs,
                                        T* result_ptr, size_t result_offset, size_t elements) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < elements; ++i) {
            result_ptr[result_offset + i] = lhs_ptr[lhs_offset + i] + rhs;
        }
        if (std::any_of(result_ptr + result_offset, result_ptr + result_offset + elements, [](T result) {
            return !std::isfinite(result);
        })) {
            throw std::overflow_error("TensorMath::_safe_tensor_add_contiguous: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < elements; ++i) {
                accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_offset + i]) + static_cast<accumulator_t>(rhs);
                overflow |= (result > MAX_VAL || result < MIN_VAL);
                result_ptr[result_offset + i] = static_cast<T>(result);
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_add_contiguous: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _add_overflow_unsigned(lhs_ptr[lhs_offset + i], rhs, &(result_ptr[result_offset + i]));
                }
            }
            else {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _add_overflow_signed(lhs_ptr[lhs_offset + i], rhs, &(result_ptr[result_offset + i]));
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_add_contiguous: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Subtract the contents of two Tensors, storing in the data block of another Tensor
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param lhs_dim0_stride Stride of the lhs outer dim
 * @param rhs_ptr Pointer to the raw data underlying the Tensor
 * @param rhs_offset Offset to the beginning of the data block
 * @param rhs_dim0_stride Stride of the rhs outer dim
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param result_dim0_stride Stride of the result outer dim
 * @param dim0_extent Outer dim size for all three Tensors
 * @param dim1_extent Inner dim size for all three Tensors
 * NOTE: This version of sub should only be used when we are sure that lhs_ptr, rhs_ptr, and
 * result_ptr absolutely do NOT overlap. Inner dim stride must equal 1.
 * This method should be used when we are subtracting two TensorSlices with variable strides.
 */
inline void _safe_2d_tensor_sub_contiguous_v(const T* __restrict__ lhs_ptr, size_t lhs_offset, size_t lhs_dim0_stride,
                                             const T* __restrict__ rhs_ptr, size_t rhs_offset, size_t rhs_dim0_stride,
                                             T* __restrict__ result_ptr, size_t result_offset, size_t result_dim0_stride,
                                             size_t dim0_extent, size_t dim1_extent) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
            const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                result_ptr[result_row + j] = lhs_ptr[lhs_row + j] - rhs_ptr[rhs_row + j];
            }
        }
        bool overflow = false;
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                T result = result_ptr[result_row + j];
                overflow |= !std::isfinite(result);
            }
        }
        if (overflow) {
            throw std::overflow_error("TensorMath::_safe_2d_tensor_sub_contiguous_v: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < dim0_extent; ++i) {
                const size_t result_row = result_offset + (result_dim0_stride * i);
                const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
                for (size_t j = 0; j < dim1_extent; ++j) {
                    accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_row + j]) - static_cast<accumulator_t>(rhs_ptr[rhs_row + j]);
                    overflow |= (result > MAX_VAL || result < MIN_VAL);
                    result_ptr[result_row + j] = static_cast<T>(result);
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_sub_contiguous_v: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _sub_overflow_unsigned(lhs_ptr[lhs_row + j], rhs_ptr[rhs_row + j], &(result_ptr[result_row + j]));
                    }
                }
            }
            else {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _sub_overflow_signed(lhs_ptr[lhs_row + j], rhs_ptr[rhs_row + j], &(result_ptr[result_row + j]));
                    }
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_sub_contiguous_v: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Subtract a scalar value from each element in a Tensor, storing in the data block of another Tensor
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param lhs_dim0_stride Stride of the lhs outer dim
 * @param rhs Scalar value to subtract from each value in lhs
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param result_dim0_stride Stride of the result outer dim
 * @param dim0_extent Outer dim size for all three Tensors
 * @param dim1_extent Inner dim size for all three Tensors
 * NOTE: This version of sub should only be used when we are sure that lhs_ptr and result_ptr
 * absolutely do NOT overlap. Inner dim stride must equal 1.
 * This method should be used when subtracting a scalar from a TensorSlice with variable strides.
 */
inline void _safe_2d_tensor_sub_contiguous_v(const T* __restrict__ lhs_ptr, size_t lhs_offset, size_t lhs_dim0_stride,
                                             const T rhs,
                                             T* __restrict__ result_ptr, size_t result_offset, size_t result_dim0_stride,
                                             size_t dim0_extent, size_t dim1_extent) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                result_ptr[result_row + j] = lhs_ptr[lhs_row + j] - rhs;
            }
        }
        bool overflow = false;
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                T result = result_ptr[result_row + j];
                overflow |= !std::isfinite(result);
            }
        }
        if (overflow) {
            throw std::overflow_error("TensorMath::_safe_2d_tensor_sub_contiguous_v: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < dim0_extent; ++i) {
                const size_t result_row = result_offset + (result_dim0_stride * i);
                const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                for (size_t j = 0; j < dim1_extent; ++j) {
                    accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_row + j]) - static_cast<accumulator_t>(rhs);
                    overflow |= (result > MAX_VAL || result < MIN_VAL);
                    result_ptr[result_row + j] = static_cast<T>(result);
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_sub_contiguous_v: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _sub_overflow_unsigned(lhs_ptr[lhs_row + j], rhs, &(result_ptr[result_row + j]));
                    }
                }
            }
            else {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _sub_overflow_signed(lhs_ptr[lhs_row + j], rhs, &(result_ptr[result_row + j]));
                    }
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_sub_contiguous_v: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Subtract the contents of two Tensors, storing in the data block of another Tensor
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param lhs_dim0_stride Stride of the lhs outer dim
 * @param rhs_ptr Pointer to the raw data underlying the Tensor
 * @param rhs_offset Offset to the beginning of the data block
 * @param rhs_dim0_stride Stride of the rhs outer dim
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param result_dim0_stride Stride of the result outer dim
 * @param dim0_extent Outer dim size for all three Tensors
 * @param dim1_extent Inner dim size for all three Tensors
 * NOTE: There is flexibility in different strides for each Tensor; however, the inner dim stride must == 1.
 * Memory blocks may overlap safely.
 */
inline void _safe_2d_tensor_sub_contiguous(const T* lhs_ptr, size_t lhs_offset, size_t lhs_dim0_stride,
                                           const T* rhs_ptr, size_t rhs_offset, size_t rhs_dim0_stride,
                                           T* result_ptr, size_t result_offset, size_t result_dim0_stride,
                                           size_t dim0_extent, size_t dim1_extent) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
            const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                result_ptr[result_row + j] = lhs_ptr[lhs_row + j] - rhs_ptr[rhs_row + j];
            }
        }
        bool overflow = false;
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                T result = result_ptr[result_row + j];
                overflow |= !std::isfinite(result);
            }
        }
        if (overflow) {
            throw std::overflow_error("TensorMath::_safe_2d_tensor_sub_contiguous: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < dim0_extent; ++i) {
                const size_t result_row = result_offset + (result_dim0_stride * i);
                const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
                for (size_t j = 0; j < dim1_extent; ++j) {
                    accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_row + j]) - static_cast<accumulator_t>(rhs_ptr[rhs_row + j]);
                    overflow |= (result > MAX_VAL || result < MIN_VAL);
                    result_ptr[result_row + j] = static_cast<T>(result);
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_sub_contiguous: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _sub_overflow_unsigned(lhs_ptr[lhs_row + j], rhs_ptr[rhs_row + j], &(result_ptr[result_row + j]));
                    }
                }
            }
            else {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _sub_overflow_signed(lhs_ptr[lhs_row + j], rhs_ptr[rhs_row + j], &(result_ptr[result_row + j]));
                    }
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_sub_contiguous: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Subtract a scalar value from each element in a Tensor, storing in the data block of another Tensor
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param lhs_dim0_stride Stride of the lhs outer dim
 * @param rhs Scalar value to subtract from each value in lhs
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param result_dim0_stride Stride of the result outer dim
 * @param dim0_extent Outer dim size for all three Tensors
 * @param dim1_extent Inner dim size for all three Tensors
 * NOTE: There is flexibility in different strides for each Tensor; however, the inner dim stride must == 1.
 * Memory blocks may overlap safely.
 */
inline void _safe_2d_tensor_sub_contiguous(const T* lhs_ptr, size_t lhs_offset, size_t lhs_dim0_stride,
                                           const T rhs,
                                           T* result_ptr, size_t result_offset, size_t result_dim0_stride,
                                           size_t dim0_extent, size_t dim1_extent) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                result_ptr[result_row + j] = lhs_ptr[lhs_row + j] - rhs;
            }
        }
        bool overflow = false;
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                T result = result_ptr[result_row + j];
                overflow |= !std::isfinite(result);
            }
        }
        if (overflow) {
            throw std::overflow_error("TensorMath::_safe_2d_tensor_sub_contiguous: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < dim0_extent; ++i) {
                const size_t result_row = result_offset + (result_dim0_stride * i);
                const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                for (size_t j = 0; j < dim1_extent; ++j) {
                    accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_row + j]) - static_cast<accumulator_t>(rhs);
                    overflow |= (result > MAX_VAL || result < MIN_VAL);
                    result_ptr[result_row + j] = static_cast<T>(result);
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_sub_contiguous: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _sub_overflow_unsigned(lhs_ptr[lhs_row + j], rhs, &(result_ptr[result_row + j]));
                    }
                }
            }
            else {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _sub_overflow_signed(lhs_ptr[lhs_row + j], rhs, &(result_ptr[result_row + j]));
                    }
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_sub_contiguous: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Subtract the contents of two 1D Tensors, storing in the data block of another Tensor
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param rhs_ptr Pointer to the raw data underlying the Tensor
 * @param rhs_offset Offset to the beginning of the data block
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param elements Number of elements to traverse
 * NOTE: This can only be used with contiguous memory blocks, so the caller must ensure that
 * if we are using a TensorSlice, that its memory is contiguous. Also note that this version
 * of sub should only be used when we are sure that lhs_ptr, rhs_ptr, and result_ptr
 * absolutely do NOT overlap.
 */
inline void _safe_tensor_sub_contiguous_v(const T* __restrict__ lhs_ptr, size_t lhs_offset, 
                                          const T* __restrict__ rhs_ptr, size_t rhs_offset,
                                          T* __restrict__ result_ptr, size_t result_offset, size_t elements) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < elements; ++i) {
            result_ptr[result_offset + i] = lhs_ptr[lhs_offset + i] - rhs_ptr[rhs_offset + i];
        }
        if (std::any_of(result_ptr + result_offset, result_ptr + result_offset + elements, [](T result) {
            return !std::isfinite(result);
        })) {
            throw std::overflow_error("TensorMath::_safe_tensor_sub_contiguous_v: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < elements; ++i) {
                accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_offset + i]) - static_cast<accumulator_t>(rhs_ptr[rhs_offset + i]);
                overflow |= (result > MAX_VAL || result < MIN_VAL);
                result_ptr[result_offset + i] = static_cast<T>(result);
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_sub_contiguous_v: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _sub_overflow_unsigned(lhs_ptr[lhs_offset + i], rhs_ptr[rhs_offset + i], &(result_ptr[result_offset + i]));
                }
            }
            else {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _sub_overflow_signed(lhs_ptr[lhs_offset + i], rhs_ptr[rhs_offset + i], &(result_ptr[result_offset + i]));
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_sub_contiguous_v: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Subtract a scalar value from a 1D Tensor
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param rhs Scalar value to subtract from each value in lhs
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param elements Number of elements to traverse
 * NOTE: This can only be used with contiguous memory blocks, so the caller must ensure that
 * if we are using a TensorSlice, that its memory is contiguous. This version
 * should only be used when lhs_ptr and result_ptr do NOT overlap.
 */
inline void _safe_tensor_sub_contiguous_v(const T* __restrict__ lhs_ptr, size_t lhs_offset,
                                          const T rhs,
                                          T* __restrict__ result_ptr, size_t result_offset, size_t elements) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < elements; ++i) {
            result_ptr[result_offset + i] = lhs_ptr[lhs_offset + i] - rhs;
        }
        if (std::any_of(result_ptr + result_offset, result_ptr + result_offset + elements, [](T result) {
            return !std::isfinite(result);
        })) {
            throw std::overflow_error("TensorMath::_safe_tensor_sub_contiguous_v: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < elements; ++i) {
                accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_offset + i]) - static_cast<accumulator_t>(rhs);
                overflow |= (result > MAX_VAL || result < MIN_VAL);
                result_ptr[result_offset + i] = static_cast<T>(result);
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_sub_contiguous_v: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _sub_overflow_unsigned(lhs_ptr[lhs_offset + i], rhs, &(result_ptr[result_offset + i]));
                }
            }
            else {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _sub_overflow_signed(lhs_ptr[lhs_offset + i], rhs, &(result_ptr[result_offset + i]));
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_sub_contiguous_v: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Subtract the contents of two 1D Tensors, storing in the data block of another Tensor
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param rhs_ptr Pointer to the raw data underlying the Tensor
 * @param rhs_offset Offset to the beginning of the data block
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param elements Number of elements to traverse
 * NOTE: This can only be used with contiguous memory blocks, so the caller must ensure that
 * if we are using a TensorSlice, that its memory is contiguous.
 */
inline void _safe_tensor_sub_contiguous(const T* lhs_ptr, size_t lhs_offset, 
                                        const T* rhs_ptr, size_t rhs_offset,
                                        T* result_ptr, size_t result_offset, size_t elements) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < elements; ++i) {
            result_ptr[result_offset + i] = lhs_ptr[lhs_offset + i] - rhs_ptr[rhs_offset + i];
        }
        if (std::any_of(result_ptr + result_offset, result_ptr + result_offset + elements, [](T result) {
            return !std::isfinite(result);
        })) {
            throw std::overflow_error("TensorMath::_safe_tensor_sub_contiguous: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < elements; ++i) {
                accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_offset + i]) - static_cast<accumulator_t>(rhs_ptr[rhs_offset + i]);
                overflow |= (result > MAX_VAL || result < MIN_VAL);
                result_ptr[result_offset + i] = static_cast<T>(result);
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_sub_contiguous: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _sub_overflow_unsigned(lhs_ptr[lhs_offset + i], rhs_ptr[rhs_offset + i], &(result_ptr[result_offset + i]));
                }
            }
            else {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _sub_overflow_signed(lhs_ptr[lhs_offset + i], rhs_ptr[rhs_offset + i], &(result_ptr[result_offset + i]));
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_sub_contiguous: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Subtract a scalar value from a 1D Tensor
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param rhs Scalar value to subtract from each value in lhs
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param elements Number of elements to traverse
 * NOTE: This can only be used with contiguous memory blocks, so the caller must ensure that
 * if we are using a TensorSlice, that its memory is contiguous.
 */
inline void _safe_tensor_sub_contiguous(const T* lhs_ptr, size_t lhs_offset, 
                                        const T rhs,
                                        T* result_ptr, size_t result_offset, size_t elements) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < elements; ++i) {
            result_ptr[result_offset + i] = lhs_ptr[lhs_offset + i] - rhs;
        }
        if (std::any_of(result_ptr + result_offset, result_ptr + result_offset + elements, [](T result) {
            return !std::isfinite(result);
        })) {
            throw std::overflow_error("TensorMath::_safe_tensor_sub_contiguous: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < elements; ++i) {
                accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_offset + i]) - static_cast<accumulator_t>(rhs);
                overflow |= (result > MAX_VAL || result < MIN_VAL);
                result_ptr[result_offset + i] = static_cast<T>(result);
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_sub_contiguous: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _sub_overflow_unsigned(lhs_ptr[lhs_offset + i], rhs, &(result_ptr[result_offset + i]));
                }
            }
            else {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _sub_overflow_signed(lhs_ptr[lhs_offset + i], rhs, &(result_ptr[result_offset + i]));
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_sub_contiguous: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Multiply the contents of two Tensors together, storing in the data block of another Tensor
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param lhs_dim0_stride Stride of the lhs outer dim
 * @param rhs_ptr Pointer to the raw data underlying the Tensor
 * @param rhs_offset Offset to the beginning of the data block
 * @param rhs_dim0_stride Stride of the rhs outer dim
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param result_dim0_stride Stride of the result outer dim
 * @param dim0_extent Outer dim size for all three Tensors
 * @param dim1_extent Inner dim size for all three Tensors
 * NOTE: This version of mul should only be used when we are sure that lhs_ptr, rhs_ptr, and
 * result_ptr absolutely do NOT overlap. Inner dim stride must equal 1.
 * This method should be used when we are multiplying two TensorSlices with variable strides.
 */
inline void _safe_2d_tensor_mul_contiguous_v(const T* __restrict__ lhs_ptr, size_t lhs_offset, size_t lhs_dim0_stride,
                                             const T* __restrict__ rhs_ptr, size_t rhs_offset, size_t rhs_dim0_stride,
                                             T* __restrict__ result_ptr, size_t result_offset, size_t result_dim0_stride,
                                             size_t dim0_extent, size_t dim1_extent) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
            const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                result_ptr[result_row + j] = lhs_ptr[lhs_row + j] * rhs_ptr[rhs_row + j];
            }
        }
        bool overflow = false;
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                T result = result_ptr[result_row + j];
                overflow |= !std::isfinite(result);
            }
        }
        if (overflow) {
            throw std::overflow_error("TensorMath::_safe_2d_tensor_mul_contiguous_v: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < dim0_extent; ++i) {
                const size_t result_row = result_offset + (result_dim0_stride * i);
                const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
                for (size_t j = 0; j < dim1_extent; ++j) {
                    accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_row + j]) * static_cast<accumulator_t>(rhs_ptr[rhs_row + j]);
                    overflow |= (result > MAX_VAL || result < MIN_VAL);
                    result_ptr[result_row + j] = static_cast<T>(result);
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_mul_contiguous_v: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _mul_overflow_unsigned(lhs_ptr[lhs_row + j], rhs_ptr[rhs_row + j], &(result_ptr[result_row + j]));
                    }
                }
            }
            else {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _mul_overflow_signed(lhs_ptr[lhs_row + j], rhs_ptr[rhs_row + j], &(result_ptr[result_row + j]));
                    }
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_mul_contiguous_v: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Multiply each element in a Tensor by a scalar value, storing in the data block of another Tensor
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param lhs_dim0_stride Stride of the lhs outer dim
 * @param rhs Scalar value to multiply with each value in lhs
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param result_dim0_stride Stride of the result outer dim
 * @param dim0_extent Outer dim size for all three Tensors
 * @param dim1_extent Inner dim size for all three Tensors
 * NOTE: This version of mul should only be used when we are sure that lhs_ptr and result_ptr
 * absolutely do NOT overlap. Inner dim stride must equal 1.
 * This method should be used when multiplying a TensorSlice by a scalar value with variable strides.
 */
inline void _safe_2d_tensor_mul_contiguous_v(const T* __restrict__ lhs_ptr, size_t lhs_offset, size_t lhs_dim0_stride,
                                             const T rhs,
                                             T* __restrict__ result_ptr, size_t result_offset, size_t result_dim0_stride,
                                             size_t dim0_extent, size_t dim1_extent) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                result_ptr[result_row + j] = lhs_ptr[lhs_row + j] * rhs;
            }
        }
        bool overflow = false;
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                T result = result_ptr[result_row + j];
                overflow |= !std::isfinite(result);
            }
        }
        if (overflow) {
            throw std::overflow_error("TensorMath::_safe_2d_tensor_mul_contiguous_v: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < dim0_extent; ++i) {
                const size_t result_row = result_offset + (result_dim0_stride * i);
                const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                for (size_t j = 0; j < dim1_extent; ++j) {
                    accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_row + j]) * static_cast<accumulator_t>(rhs);
                    overflow |= (result > MAX_VAL || result < MIN_VAL);
                    result_ptr[result_row + j] = static_cast<T>(result);
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_mul_contiguous_v: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _mul_overflow_unsigned(lhs_ptr[lhs_row + j], rhs, &(result_ptr[result_row + j]));
                    }
                }
            }
            else {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _mul_overflow_signed(lhs_ptr[lhs_row + j], rhs, &(result_ptr[result_row + j]));
                    }
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_mul_contiguous_v: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Multiply the contents of two Tensors together, storing in the data block of another Tensor
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param lhs_dim0_stride Stride of the lhs outer dim
 * @param rhs_ptr Pointer to the raw data underlying the Tensor
 * @param rhs_offset Offset to the beginning of the data block
 * @param rhs_dim0_stride Stride of the rhs outer dim
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param result_dim0_stride Stride of the result outer dim
 * @param dim0_extent Outer dim size for all three Tensors
 * @param dim1_extent Inner dim size for all three Tensors
 * NOTE: There is flexibility in different strides for each Tensor; however, the inner dim stride must == 1.
 * Memory blocks may overlap safely.
 */
inline void _safe_2d_tensor_mul_contiguous(const T* lhs_ptr, size_t lhs_offset, size_t lhs_dim0_stride,
                                           const T* rhs_ptr, size_t rhs_offset, size_t rhs_dim0_stride,
                                           T* result_ptr, size_t result_offset, size_t result_dim0_stride,
                                           size_t dim0_extent, size_t dim1_extent) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
            const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                result_ptr[result_row + j] = lhs_ptr[lhs_row + j] * rhs_ptr[rhs_row + j];
            }
        }
        bool overflow = false;
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                T result = result_ptr[result_row + j];
                overflow |= !std::isfinite(result);
            }
        }
        if (overflow) {
            throw std::overflow_error("TensorMath::_safe_2d_tensor_mul_contiguous: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < dim0_extent; ++i) {
                const size_t result_row = result_offset + (result_dim0_stride * i);
                const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
                for (size_t j = 0; j < dim1_extent; ++j) {
                    accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_row + j]) * static_cast<accumulator_t>(rhs_ptr[rhs_row + j]);
                    overflow |= (result > MAX_VAL || result < MIN_VAL);
                    result_ptr[result_row + j] = static_cast<T>(result);
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_mul_contiguous: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _mul_overflow_unsigned(lhs_ptr[lhs_row + j], rhs_ptr[rhs_row + j], &(result_ptr[result_row + j]));
                    }
                }
            }
            else {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _mul_overflow_signed(lhs_ptr[lhs_row + j], rhs_ptr[rhs_row + j], &(result_ptr[result_row + j]));
                    }
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_mul_contiguous: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Multiply each element in a Tensor by a scalar value, storing in the data block of another Tensor
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param lhs_dim0_stride Stride of the lhs outer dim
 * @param rhs Scalar value to multiply with each value in lhs
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param result_dim0_stride Stride of the result outer dim
 * @param dim0_extent Outer dim size for all three Tensors
 * @param dim1_extent Inner dim size for all three Tensors
 * NOTE: There is flexibility in different strides for each Tensor; however, the inner dim stride must == 1.
 * Memory blocks may overlap safely.
 */
inline void _safe_2d_tensor_mul_contiguous(const T* lhs_ptr, size_t lhs_offset, size_t lhs_dim0_stride,
                                           const T rhs,
                                           T* result_ptr, size_t result_offset, size_t result_dim0_stride,
                                           size_t dim0_extent, size_t dim1_extent) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                result_ptr[result_row + j] = lhs_ptr[lhs_row + j] * rhs;
            }
        }
        bool overflow = false;
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                T result = result_ptr[result_row + j];
                overflow |= !std::isfinite(result);
            }
        }
        if (overflow) {
            throw std::overflow_error("TensorMath::_safe_2d_tensor_mul_contiguous: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < dim0_extent; ++i) {
                const size_t result_row = result_offset + (result_dim0_stride * i);
                const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                for (size_t j = 0; j < dim1_extent; ++j) {
                    accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_row + j]) * static_cast<accumulator_t>(rhs);
                    overflow |= (result > MAX_VAL || result < MIN_VAL);
                    result_ptr[result_row + j] = static_cast<T>(result);
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_mul_contiguous: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _mul_overflow_unsigned(lhs_ptr[lhs_row + j], rhs, &(result_ptr[result_row + j]));
                    }
                }
            }
            else {
                for (size_t i = 0; i < dim0_extent; ++i) {
                    const size_t result_row = result_offset + (result_dim0_stride * i);
                    const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
                    for (size_t j = 0; j < dim1_extent; ++j) {
                        overflow |= _mul_overflow_signed(lhs_ptr[lhs_row + j], rhs, &(result_ptr[result_row + j]));
                    }
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_2d_tensor_mul_contiguous: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Multiply the contents of two 1D Tensors together, storing in the data block of another Tensor
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param rhs_ptr Pointer to the raw data underlying the Tensor
 * @param rhs_offset Offset to the beginning of the data block
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param elements Number of elements to traverse
 * NOTE: This can only be used with contiguous memory blocks, so the caller must ensure that
 * if we are using a TensorSlice, that its memory is contiguous. Also note that this version
 * of mul should only be used when we are sure that lhs_ptr, rhs_ptr, and result_ptr
 * absolutely do NOT overlap.
 */
inline void _safe_tensor_mul_contiguous_v(const T* __restrict__ lhs_ptr, size_t lhs_offset, 
                                          const T* __restrict__ rhs_ptr, size_t rhs_offset,
                                          T* __restrict__ result_ptr, size_t result_offset, size_t elements) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < elements; ++i) {
            result_ptr[result_offset + i] = lhs_ptr[lhs_offset + i] * rhs_ptr[rhs_offset + i];
        }
        if (std::any_of(result_ptr + result_offset, result_ptr + result_offset + elements, [](T result) {
            return !std::isfinite(result);
        })) {
            throw std::overflow_error("TensorMath::_safe_tensor_mul_contiguous_v: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < elements; ++i) {
                accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_offset + i]) * static_cast<accumulator_t>(rhs_ptr[rhs_offset + i]);
                overflow |= (result > MAX_VAL || result < MIN_VAL);
                result_ptr[result_offset + i] = static_cast<T>(result);
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_mul_contiguous_v: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _mul_overflow_unsigned(lhs_ptr[lhs_offset + i], rhs_ptr[rhs_offset + i], &(result_ptr[result_offset + i]));
                }
            }
            else {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _mul_overflow_signed(lhs_ptr[lhs_offset + i], rhs_ptr[rhs_offset + i], &(result_ptr[result_offset + i]));
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_mul_contiguous_v: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Multiply each element in a 1D Tensor by a scalar value
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param rhs Scalar value to multiply with each value in lhs
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param elements Number of elements to traverse
 * NOTE: This can only be used with contiguous memory blocks, so the caller must ensure that
 * if we are using a TensorSlice, that its memory is contiguous. This version
 * should only be used when lhs_ptr and result_ptr do NOT overlap.
 */
inline void _safe_tensor_mul_contiguous_v(const T* __restrict__ lhs_ptr, size_t lhs_offset,
                                          const T rhs,
                                          T* __restrict__ result_ptr, size_t result_offset, size_t elements) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < elements; ++i) {
            result_ptr[result_offset + i] = lhs_ptr[lhs_offset + i] * rhs;
        }
        if (std::any_of(result_ptr + result_offset, result_ptr + result_offset + elements, [](T result) {
            return !std::isfinite(result);
        })) {
            throw std::overflow_error("TensorMath::_safe_tensor_mul_contiguous_v: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < elements; ++i) {
                accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_offset + i]) * static_cast<accumulator_t>(rhs);
                overflow |= (result > MAX_VAL || result < MIN_VAL);
                result_ptr[result_offset + i] = static_cast<T>(result);
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_mul_contiguous_v: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _mul_overflow_unsigned(lhs_ptr[lhs_offset + i], rhs, &(result_ptr[result_offset + i]));
                }
            }
            else {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _mul_overflow_signed(lhs_ptr[lhs_offset + i], rhs, &(result_ptr[result_offset + i]));
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_mul_contiguous_v: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Multiply the contents of two 1D Tensors together, storing in the data block of another Tensor
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param rhs_ptr Pointer to the raw data underlying the Tensor
 * @param rhs_offset Offset to the beginning of the data block
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param elements Number of elements to traverse
 * NOTE: This can only be used with contiguous memory blocks, so the caller must ensure that
 * if we are using a TensorSlice, that its memory is contiguous.
 */
inline void _safe_tensor_mul_contiguous(const T* lhs_ptr, size_t lhs_offset, 
                                        const T* rhs_ptr, size_t rhs_offset,
                                        T* result_ptr, size_t result_offset, size_t elements) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < elements; ++i) {
            result_ptr[result_offset + i] = lhs_ptr[lhs_offset + i] * rhs_ptr[rhs_offset + i];
        }
        if (std::any_of(result_ptr + result_offset, result_ptr + result_offset + elements, [](T result) {
            return !std::isfinite(result);
        })) {
            throw std::overflow_error("TensorMath::_safe_tensor_mul_contiguous: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < elements; ++i) {
                accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_offset + i]) * static_cast<accumulator_t>(rhs_ptr[rhs_offset + i]);
                overflow |= (result > MAX_VAL || result < MIN_VAL);
                result_ptr[result_offset + i] = static_cast<T>(result);
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_mul_contiguous: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _mul_overflow_unsigned(lhs_ptr[lhs_offset + i], rhs_ptr[rhs_offset + i], &(result_ptr[result_offset + i]));
                }
            }
            else {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _mul_overflow_signed(lhs_ptr[lhs_offset + i], rhs_ptr[rhs_offset + i], &(result_ptr[result_offset + i]));
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_mul_contiguous: Overflow / underflow detected.\n");
            }
        }
    }
}

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Multiply each element in a 1D Tensor by a scalar value
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using TensorSlices
 * @param rhs Scalar value to multiply with each value in lhs
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param elements Number of elements to traverse
 * NOTE: This can only be used with contiguous memory blocks, so the caller must ensure that
 * if we are using a TensorSlice, that its memory is contiguous.
 */
inline void _safe_tensor_mul_contiguous(const T* lhs_ptr, size_t lhs_offset, 
                                        const T rhs,
                                        T* result_ptr, size_t result_offset, size_t elements) {
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < elements; ++i) {
            result_ptr[result_offset + i] = lhs_ptr[lhs_offset + i] * rhs;
        }
        if (std::any_of(result_ptr + result_offset, result_ptr + result_offset + elements, [](T result) {
            return !std::isfinite(result);
        })) {
            throw std::overflow_error("TensorMath::_safe_tensor_mul_contiguous: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            bool overflow = false;
            for (size_t i = 0; i < elements; ++i) {
                accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_offset + i]) * static_cast<accumulator_t>(rhs);
                overflow |= (result > MAX_VAL || result < MIN_VAL);
                result_ptr[result_offset + i] = static_cast<T>(result);
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_mul_contiguous: Overflow / underflow detected.\n");
            }
        }
        else {
            bool overflow = false;
            if constexpr (std::is_unsigned_v<T>) {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _mul_overflow_unsigned(lhs_ptr[lhs_offset + i], rhs, &(result_ptr[result_offset + i]));
                }
            }
            else {
                for (size_t i = 0; i < elements; ++i) {
                    overflow |= _mul_overflow_signed(lhs_ptr[lhs_offset + i], rhs, &(result_ptr[result_offset + i]));
                }
            }
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_mul_contiguous: Overflow / underflow detected.\n");
            }
        }
    }
}

/**
 * Divide the contents of two 2D Tensors element-wise, storing in the data block of another Tensor.
 * @param lhs_ptr Pointer to raw data underlying the numerator Tensor
 * @param lhs_offset Offset to the beginning of the data block for the numerator Tensor
 * @param lhs_dim0_stride Stride of the numerator outer dim
 * @param rhs_ptr Pointer to raw data underlying the denominator Tensor
 * @param rhs_offset Offset to the beginning of the data block for the denominator Tensor
 * @param rhs_dim0_stride Stride of the denominator outer dim
 * @param result_ptr Pointer to raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block for the result Tensor
 * @param result_dim0_stride Stride of the result outer dim
 * @param dim0_extent Outer dim size for all three Tensors
 * @param dim1_extent Inner dim size for all three Tensors
 * NOTE: This version of div should only be used when we are sure that lhs_ptr, rhs_ptr, and
 * result_ptr absolutely do NOT overlap. Inner dim stride must equal 1.
 */
template <typename T>
requires std::is_arithmetic_v<T>
inline void _safe_2d_tensor_div_contiguous_v(const T* __restrict__ lhs_ptr, size_t lhs_offset, size_t lhs_dim0_stride,
                                             const T* __restrict__ rhs_ptr, size_t rhs_offset, size_t rhs_dim0_stride,
                                             T* __restrict__ result_ptr, size_t result_offset, size_t result_dim0_stride,
                                             size_t dim0_extent, size_t dim1_extent) {
    for (size_t i = 0; i < dim0_extent; ++i) {
        const size_t result_row = result_offset + (result_dim0_stride * i);
        const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
        const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);

        if (std::any_of(rhs_ptr + rhs_row, rhs_ptr + rhs_row + dim1_extent, [](T val) { return val == 0; })) {
            throw std::runtime_error("TensorMath::_safe_2d_tensor_div_contiguous_v: Divide by zero detected.\n");
        }

        if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
            for (size_t j = 0; j < dim1_extent; ++j) {
                if (_is_signed_div_overflow(lhs_ptr[lhs_row + j], rhs_ptr[rhs_row + j])) {
                    throw std::overflow_error("TensorMath::_safe_2d_tensor_div_contiguous_v: Signed integer division overflow detected.\n");
                }
            }
        }

        for (size_t j = 0; j < dim1_extent; ++j) {
            result_ptr[result_row + j] = lhs_ptr[lhs_row + j] / rhs_ptr[rhs_row + j];
        }
    }

    if constexpr (std::is_floating_point_v<T>) {
        bool overflow = false;
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                overflow |= !std::isfinite(result_ptr[result_row + j]);
            }
        }
        if (overflow) {
            throw std::overflow_error("TensorMath::_safe_2d_tensor_div_contiguous_v: Overflow / underflow detected.\n");
        }
    }
}

/**
 * Divide a 2D Tensor element-wise by a scalar value, storing in the data block of another Tensor.
 * @param lhs_ptr Pointer to raw data underlying the numerator Tensor
 * @param lhs_offset Offset to the beginning of the data block for the numerator Tensor
 * @param lhs_dim0_stride Stride of the numerator outer dim
 * @param rhs Scalar denominator value to divide each element in lhs by
 * @param result_ptr Pointer to raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block for the result Tensor
 * @param result_dim0_stride Stride of the result outer dim
 * @param dim0_extent Outer dim size for both Tensors
 * @param dim1_extent Inner dim size for both Tensors
 * NOTE: This version of div should only be used when we are sure that lhs_ptr and
 * result_ptr absolutely do NOT overlap. Inner dim stride must equal 1.
 */
template <typename T>
requires std::is_arithmetic_v<T>
inline void _safe_2d_tensor_div_contiguous_v(const T* __restrict__ lhs_ptr, size_t lhs_offset, size_t lhs_dim0_stride,
                                             const T rhs,
                                             T* __restrict__ result_ptr, size_t result_offset, size_t result_dim0_stride,
                                             size_t dim0_extent, size_t dim1_extent) {
    if (rhs == 0) {
        throw std::runtime_error("TensorMath::_safe_2d_tensor_div_contiguous_v: Divide by zero detected.\n");
    }

    for (size_t i = 0; i < dim0_extent; ++i) {
        const size_t result_row = result_offset + (result_dim0_stride * i);
        const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);

        if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
            if (rhs == -1) {
                for (size_t j = 0; j < dim1_extent; ++j) {
                    if (lhs_ptr[lhs_row + j] == std::numeric_limits<T>::min()) {
                        throw std::overflow_error("TensorMath::_safe_2d_tensor_div_contiguous_v: Signed integer division overflow detected.\n");
                    }
                }
            }
        }

        for (size_t j = 0; j < dim1_extent; ++j) {
            result_ptr[result_row + j] = lhs_ptr[lhs_row + j] / rhs;
        }
    }

    if constexpr (std::is_floating_point_v<T>) {
        bool overflow = false;
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                overflow |= !std::isfinite(result_ptr[result_row + j]);
            }
        }
        if (overflow) {
            throw std::overflow_error("TensorMath::_safe_2d_tensor_div_contiguous_v: Overflow / underflow detected.\n");
        }
    }
}

/**
 * Divide the contents of two 2D Tensors element-wise, storing in the data block of another Tensor.
 * @param lhs_ptr Pointer to raw data underlying the numerator Tensor
 * @param lhs_offset Offset to the beginning of the data block for the numerator Tensor
 * @param lhs_dim0_stride Stride of the numerator outer dim
 * @param rhs_ptr Pointer to raw data underlying the denominator Tensor
 * @param rhs_offset Offset to the beginning of the data block for the denominator Tensor
 * @param rhs_dim0_stride Stride of the denominator outer dim
 * @param result_ptr Pointer to raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block for the result Tensor
 * @param result_dim0_stride Stride of the result outer dim
 * @param dim0_extent Outer dim size for all three Tensors
 * @param dim1_extent Inner dim size for all three Tensors
 * NOTE: Inner dim stride must equal 1. Memory blocks may overlap safely.
 */
template <typename T>
requires std::is_arithmetic_v<T>
inline void _safe_2d_tensor_div_contiguous(const T* lhs_ptr, size_t lhs_offset, size_t lhs_dim0_stride,
                                           const T* rhs_ptr, size_t rhs_offset, size_t rhs_dim0_stride,
                                           T* result_ptr, size_t result_offset, size_t result_dim0_stride,
                                           size_t dim0_extent, size_t dim1_extent) {
    for (size_t i = 0; i < dim0_extent; ++i) {
        const size_t result_row = result_offset + (result_dim0_stride * i);
        const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);
        const size_t rhs_row = rhs_offset + (rhs_dim0_stride * i);

        if (std::any_of(rhs_ptr + rhs_row, rhs_ptr + rhs_row + dim1_extent, [](T val) { return val == 0; })) {
            throw std::runtime_error("TensorMath::_safe_2d_tensor_div_contiguous: Divide by zero detected.\n");
        }

        if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
            for (size_t j = 0; j < dim1_extent; ++j) {
                if (_is_signed_div_overflow(lhs_ptr[lhs_row + j], rhs_ptr[rhs_row + j])) {
                    throw std::overflow_error("TensorMath::_safe_2d_tensor_div_contiguous: Signed integer division overflow detected.\n");
                }
            }
        }

        for (size_t j = 0; j < dim1_extent; ++j) {
            result_ptr[result_row + j] = lhs_ptr[lhs_row + j] / rhs_ptr[rhs_row + j];
        }
    }

    if constexpr (std::is_floating_point_v<T>) {
        bool overflow = false;
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                overflow |= !std::isfinite(result_ptr[result_row + j]);
            }
        }
        if (overflow) {
            throw std::overflow_error("TensorMath::_safe_2d_tensor_div_contiguous: Overflow / underflow detected.\n");
        }
    }
}

/**
 * Divide a 2D Tensor element-wise by a scalar value, storing in the data block of another Tensor.
 * @param lhs_ptr Pointer to raw data underlying the numerator Tensor
 * @param lhs_offset Offset to the beginning of the data block for the numerator Tensor
 * @param lhs_dim0_stride Stride of the numerator outer dim
 * @param rhs Scalar denominator value to divide each element in lhs by
 * @param result_ptr Pointer to raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block for the result Tensor
 * @param result_dim0_stride Stride of the result outer dim
 * @param dim0_extent Outer dim size for both Tensors
 * @param dim1_extent Inner dim size for both Tensors
 * NOTE: Inner dim stride must equal 1. Memory blocks may overlap safely.
 */
template <typename T>
requires std::is_arithmetic_v<T>
inline void _safe_2d_tensor_div_contiguous(const T* lhs_ptr, size_t lhs_offset, size_t lhs_dim0_stride,
                                           const T rhs,
                                           T* result_ptr, size_t result_offset, size_t result_dim0_stride,
                                           size_t dim0_extent, size_t dim1_extent) {
    if (rhs == 0) {
        throw std::runtime_error("TensorMath::_safe_2d_tensor_div_contiguous: Divide by zero detected.\n");
    }

    for (size_t i = 0; i < dim0_extent; ++i) {
        const size_t result_row = result_offset + (result_dim0_stride * i);
        const size_t lhs_row = lhs_offset + (lhs_dim0_stride * i);

        if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
            if (rhs == -1) {
                for (size_t j = 0; j < dim1_extent; ++j) {
                    if (lhs_ptr[lhs_row + j] == std::numeric_limits<T>::min()) {
                        throw std::overflow_error("TensorMath::_safe_2d_tensor_div_contiguous: Signed integer division overflow detected.\n");
                    }
                }
            }
        }

        for (size_t j = 0; j < dim1_extent; ++j) {
            result_ptr[result_row + j] = lhs_ptr[lhs_row + j] / rhs;
        }
    }

    if constexpr (std::is_floating_point_v<T>) {
        bool overflow = false;
        for (size_t i = 0; i < dim0_extent; ++i) {
            const size_t result_row = result_offset + (result_dim0_stride * i);
            for (size_t j = 0; j < dim1_extent; ++j) {
                overflow |= !std::isfinite(result_ptr[result_row + j]);
            }
        }
        if (overflow) {
            throw std::overflow_error("TensorMath::_safe_2d_tensor_div_contiguous: Overflow / underflow detected.\n");
        }
    }
}

/**
 * Divide the contents of two 1D Tensors element-wise, storing in the data block of another Tensor.
 * @param lhs_ptr Pointer to raw data underlying the numerator Tensor
 * @param lhs_offset Offset to the beginning of the data block for the numerator Tensor
 * @param rhs_ptr Pointer to raw data underlying the denominator Tensor
 * @param rhs_offset Offset to the beginning of the data block for the denominator Tensor
 * @param result_ptr Pointer to raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block for the result Tensor
 * @param elements Number of elements to traverse
 * NOTE: This can only be used with contiguous memory blocks. This version of div should
 * only be used when lhs_ptr, rhs_ptr, and result_ptr absolutely do NOT overlap.
 */
template <typename T>
requires std::is_arithmetic_v<T>
inline void _safe_tensor_div_contiguous_v(const T* __restrict__ lhs_ptr, size_t lhs_offset, 
                                          const T* __restrict__ rhs_ptr, size_t rhs_offset,
                                          T* __restrict__ result_ptr, size_t result_offset, size_t elements) {
    if (std::any_of(rhs_ptr + rhs_offset, rhs_ptr + rhs_offset + elements, [](T val) { return val == 0; })) {
        throw std::runtime_error("TensorMath::_safe_tensor_div_contiguous_v: Divide by zero detected.\n");
    }

    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        for (size_t i = 0; i < elements; ++i) {
            if (_is_signed_div_overflow(lhs_ptr[lhs_offset + i], rhs_ptr[rhs_offset + i])) {
                throw std::overflow_error("TensorMath::_safe_tensor_div_contiguous_v: Signed integer division overflow detected.\n");
            }
        }
    }

    for (size_t i = 0; i < elements; ++i) {
        result_ptr[result_offset + i] = lhs_ptr[lhs_offset + i] / rhs_ptr[rhs_offset + i];
    }

    if constexpr (std::is_floating_point_v<T>) {
        if (std::any_of(result_ptr + result_offset, result_ptr + result_offset + elements, [](T res) { return !std::isfinite(res); })) {
            throw std::overflow_error("TensorMath::_safe_tensor_div_contiguous_v: Overflow / underflow detected.\n");
        }
    }
}

/**
 * Divide a 1D Tensor element-wise by a scalar value, storing in the data block of another Tensor.
 * @param lhs_ptr Pointer to raw data underlying the numerator Tensor
 * @param lhs_offset Offset to the beginning of the data block for the numerator Tensor
 * @param rhs Scalar denominator value to divide each element in lhs by
 * @param result_ptr Pointer to raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block for the result Tensor
 * @param elements Number of elements to traverse
 * NOTE: This can only be used with contiguous memory blocks. This version of div should
 * only be used when lhs_ptr and result_ptr absolutely do NOT overlap.
 */
template <typename T>
requires std::is_arithmetic_v<T>
inline void _safe_tensor_div_contiguous_v(const T* __restrict__ lhs_ptr, size_t lhs_offset,
                                          const T rhs,
                                          T* __restrict__ result_ptr, size_t result_offset, size_t elements) {
    if (rhs == 0) {
        throw std::runtime_error("TensorMath::_safe_tensor_div_contiguous_v: Divide by zero detected.\n");
    }

    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        if (rhs == -1) {
            for (size_t i = 0; i < elements; ++i) {
                if (lhs_ptr[lhs_offset + i] == std::numeric_limits<T>::min()) {
                    throw std::overflow_error("TensorMath::_safe_tensor_div_contiguous_v: Signed integer division overflow detected.\n");
                }
            }
        }
    }

    for (size_t i = 0; i < elements; ++i) {
        result_ptr[result_offset + i] = lhs_ptr[lhs_offset + i] / rhs;
    }

    if constexpr (std::is_floating_point_v<T>) {
        if (std::any_of(result_ptr + result_offset, result_ptr + result_offset + elements, [](T res) { return !std::isfinite(res); })) {
            throw std::overflow_error("TensorMath::_safe_tensor_div_contiguous_v: Overflow / underflow detected.\n");
        }
    }
}

/**
 * Divide the contents of two 1D Tensors element-wise, storing in the data block of another Tensor.
 * @param lhs_ptr Pointer to raw data underlying the numerator Tensor
 * @param lhs_offset Offset to the beginning of the data block for the numerator Tensor
 * @param rhs_ptr Pointer to raw data underlying the denominator Tensor
 * @param rhs_offset Offset to the beginning of the data block for the denominator Tensor
 * @param result_ptr Pointer to raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block for the result Tensor
 * @param elements Number of elements to traverse
 * NOTE: This can only be used with contiguous memory blocks. Memory blocks may overlap safely.
 */
template <typename T>
requires std::is_arithmetic_v<T>
inline void _safe_tensor_div_contiguous(const T* lhs_ptr, size_t lhs_offset, 
                                        const T* rhs_ptr, size_t rhs_offset,
                                        T* result_ptr, size_t result_offset, size_t elements) {
    if (std::any_of(rhs_ptr + rhs_offset, rhs_ptr + rhs_offset + elements, [](T val) { return val == 0; })) {
        throw std::runtime_error("TensorMath::_safe_tensor_div_contiguous: Divide by zero detected.\n");
    }

    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        for (size_t i = 0; i < elements; ++i) {
            if (_is_signed_div_overflow(lhs_ptr[lhs_offset + i], rhs_ptr[rhs_offset + i])) {
                throw std::overflow_error("TensorMath::_safe_tensor_div_contiguous: Signed integer division overflow detected.\n");
            }
        }
    }

    for (size_t i = 0; i < elements; ++i) {
        result_ptr[result_offset + i] = lhs_ptr[lhs_offset + i] / rhs_ptr[rhs_offset + i];
    }

    if constexpr (std::is_floating_point_v<T>) {
        if (std::any_of(result_ptr + result_offset, result_ptr + result_offset + elements, [](T res) { return !std::isfinite(res); })) {
            throw std::overflow_error("TensorMath::_safe_tensor_div_contiguous: Overflow / underflow detected.\n");
        }
    }
}

/**
 * Divide a 1D Tensor element-wise by a scalar value, storing in the data block of another Tensor.
 * @param lhs_ptr Pointer to raw data underlying the numerator Tensor
 * @param lhs_offset Offset to the beginning of the data block for the numerator Tensor
 * @param rhs Scalar denominator value to divide each element in lhs by
 * @param result_ptr Pointer to raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block for the result Tensor
 * @param elements Number of elements to traverse
 * NOTE: This can only be used with contiguous memory blocks. Memory blocks may overlap safely.
 */
template <typename T>
requires std::is_arithmetic_v<T>
inline void _safe_tensor_div_contiguous(const T* lhs_ptr, size_t lhs_offset, 
                                        const T rhs,
                                        T* result_ptr, size_t result_offset, size_t elements) {
    if (rhs == 0) {
        throw std::runtime_error("TensorMath::_safe_tensor_div_contiguous: Divide by zero detected.\n");
    }

    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        if (rhs == -1) {
            for (size_t i = 0; i < elements; ++i) {
                if (lhs_ptr[lhs_offset + i] == std::numeric_limits<T>::min()) {
                    throw std::overflow_error("TensorMath::_safe_tensor_div_contiguous: Signed integer division overflow detected.\n");
                }
            }
        }
    }

    for (size_t i = 0; i < elements; ++i) {
        result_ptr[result_offset + i] = lhs_ptr[lhs_offset + i] / rhs;
    }

    if constexpr (std::is_floating_point_v<T>) {
        if (std::any_of(result_ptr + result_offset, result_ptr + result_offset + elements, [](T res) { return !std::isfinite(res); })) {
            throw std::overflow_error("TensorMath::_safe_tensor_div_contiguous: Overflow / underflow detected.\n");
        }
    }
}
// NOLINTEND(bugprone-easily-swappable-parameters, cppcoreguidelines-pro-bounds-pointer-arithmetic)

}; // namespace TensorMath_NS

#endif
