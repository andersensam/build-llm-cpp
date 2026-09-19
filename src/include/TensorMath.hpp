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
 * @version: 2026-09-18
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef TENSOR_MATH_HPP
#define TENSOR_MATH_HPP

/* Standard dependencies */
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
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
 * DISCLAIMER: AI tools were used to refine the overflow / underflow logic. 
 * Testing revealed that signed int UB can be optimized out and I wasn't
 * able to figure this one out on my own.
 */
template <typename T> 
requires std::is_arithmetic_v<T> && std::is_signed_v<T>
[[nodiscard]] inline bool _add_overflow_signed(T a, T b, T* result) {
    using U = std::make_unsigned_t<T>;
    U res_u = static_cast<U>(a) + static_cast<U>(b);
    *result = static_cast<T>(res_u);
    // Overflow occurs if and only if a and b have the same sign, 
    // but the result of a + b has a different sign
    return ((a ^ *result) & (b ^ *result)) < 0;
}

/**
 * Compiler-independent check for addition overflow / underflow
 * @param a First value
 * @param b Second value
 * @param result Pointer to store the result in
 * @returns Returns true if the operation would cause overflow / underflow
 */
template <typename T>
requires std::is_arithmetic_v<T> && std::is_unsigned_v<T>
[[nodiscard]] inline bool _add_overflow_unsigned(T a, T b, T* result) {
    *result = a + b;
    return *result < a; // Wraparound check
}

/**
 * Compiler-independent check for subtraction overflow / underflow
 * @param a First value
 * @param b Second value
 * @param result Pointer to store the result in
 * @returns Returns true if the operation would cause overflow / underflow
 */
template <typename T> 
requires std::is_arithmetic_v<T>
[[nodiscard]] inline bool _sub_overflow(T a, T b, T* result) {
    // We always check to see if the type is eligible for overflow before calling
    // this function, so no need to check again
    if constexpr (std::numeric_limits<T>::is_signed) {
        if (((b > 0) && (a < (std::numeric_limits<T>::min() + b))) || 
            ((b < 0) && (a > (std::numeric_limits<T>::max() - b)))) {
            return true;
        }
    }
    else {
        if (b > a) {
            return true;
        }
    }
    *result = a - b;
    return false;
}

/**
 * Compiler-independent check for multiplication overflow / underflow
 * @param a First value
 * @param b Second value
 * @param result Pointer to store the result in
 * @returns Returns true if the operation would cause overflow / underflow
 */
template <typename T> 
requires std::is_arithmetic_v<T>
[[nodiscard]] inline bool _mul_overflow(T a, T b, T* result) {
    // We always check to see if the type is eligible for overflow before calling
    // this function, so no need to check again
    if (a == 0 || b == 0) {
        *result = 0;
        return false;
    }
    // Get the min and max values for the type T
    T min_val = std::numeric_limits<T>::min();
    T max_val = std::numeric_limits<T>::max();
    // Validate the multiplication
    if (a > 0) {
        if (b > 0) {
            if (a > (max_val / b)) {
                return true;
            }
            else {
                *result = a * b;
                return false;
            }
        }
        else {
            if (b < (min_val / a)) {
                return true;
            }
            else {
                *result = a * b;
                return false;
            }
        }
    }
    else {
        if (b > 0) {
            if (a < (min_val / b)) {
                return true;
            }
            else {
                *result = a * b;
                return false;
            }
        }
        else {
            if (a < (max_val) / b) {
                return true;
            }
            else {
                *result = a * b;
                return false;
            }
        }
    }
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters, cppcoreguidelines-pro-bounds-pointer-arithmetic)
template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Add the contents of two Tensors together
 * @param lhs_ptr Pointer to raw data underlying the Tensor
 * @param lhs_offset Offset to the beginning of the data block for the Tensor, needed when using
 * TensorSlices
 * @param rhs_ptr Pointer to the raw data underlying the Tensor
 * @param rhs_offset Offset to the beginning of the data block
 * @param result_ptr Pointer to the raw data block underlying the result Tensor
 * @param result_offset Offset to the beginning of the data block
 * @param elements Number of elements to traverse
 * NOTE: The can only be used with contiguous memory blocks, so the caller must ensure that
 * if we are using a TensorSlice, that its memory is contiguous
 */
inline void _safe_tensor_add_contiguous(const T* lhs_ptr, size_t lhs_offset, const T* rhs_ptr, size_t rhs_offset,
                                        T* result_ptr, size_t result_offset, size_t elements) {
    // Handle the case where T represents a float type -- just scan for inf or NaN after the operation
    if constexpr (std::is_floating_point_v<T>) {
        for (size_t i = 0; i < elements; ++i) {
            result_ptr[result_offset + i] = lhs_ptr[lhs_offset + i] + rhs_ptr[rhs_offset + i];
        }
        // Check for NaN and inf
        if (std::any_of(result_ptr + result_offset, result_ptr + result_offset + elements, [](T val) {
            return std::isnan(val) || std::isinf(val);
        })) {
            throw std::overflow_error("TensorMath::_safe_tensor_add_contiguous: Overflow / underflow detected.\n");
        }
    }
    else if constexpr (std::is_integral_v<T>) {
        // Create the Accumulator type binding, though we will use direct casts in this use case
        using accumulator_t = typename Numerics_NS::Accumulator<T>::type;
        // Handle signed integer types, with clearly defined MIN and MAX values
        if constexpr (!std::is_same_v<T, accumulator_t> && std::is_signed_v<T>) {
            // Define the min and max values for the underlying dtype
            constexpr accumulator_t MIN_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::min());
            constexpr accumulator_t MAX_VAL = static_cast<accumulator_t>(std::numeric_limits<T>::max());
            // Track whether or not we've had an overflow
            bool overflow = false;
            // Iterate over the elements, casting to accumulator_t
            for (size_t i = 0; i < elements; ++i) {
                // Use the wider type, perform the addition
                accumulator_t result = static_cast<accumulator_t>(lhs_ptr[lhs_offset + i]) + static_cast<accumulator_t>(rhs_ptr[rhs_offset + i]);
                // Use the |= operator to accumulate whether or not an overflow occurred 
                overflow |= (result > MAX_VAL || result < MIN_VAL);
                // Cast back to the normal value
                result_ptr[result_offset + i] = static_cast<T>(result);
            }
            // Once the loop is done, check for overflow
            if (overflow) {
                throw std::overflow_error("TensorMath::_safe_tensor_add_contiguous: Overflow / underflow detected.\n");
            }
        }
        // If we either are dealing with an unsigned type or an accumulator that is the same, i.e.
        // for int64_t, there is no double width, so we need to take this into account
        // We also need special treatment for unsigned integer types, since we can't compare
        // to a minimum value easily.
        else {
            // Track whether or not we get an overflow
            bool overflow = false;
            // Use the unsigned overflow function if we have is_unsigned_v
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
    else {
        throw std::logic_error("TensorMath::_safe_tensor_add_contiguous: Invalid type (not _is_floating_point_v or _is_integral_v).\n");
    }
}
// NOLINTEND(bugprone-easily-swappable-parameters, cppcoreguidelines-pro-bounds-pointer-arithmetic)

}; // namespace TensorMath_NS

#endif
