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
 * @version: 2026-10-22
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef NUMERICS_HPP
#define NUMERICS_HPP

/* Standard dependencies */
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <type_traits>

namespace Numerics_NS {

/**
 * Templates for accumulation, which generally is the double-width version of the
 * underlying data type. We accumulate in this type, check for overflow / underflow
 * in the corresponding type, then cast back, preventing branching in the matmul
 * and allowing for better vectorization
 */
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template <Arithmetic T>
struct Accumulator;

// Handle the integer (integral) types first
template <> struct Accumulator<int8_t> { 
    using type = int16_t;
    using serialize_as = int8_t;
    static constexpr uint32_t _base_type_mn = 0x88000000;
    static constexpr size_t _base_type_size = sizeof(int8_t);
    static constexpr size_t _accumulator_size = sizeof(int16_t);
    static constexpr size_t _s_size = sizeof(serialize_as);
};

template <> struct Accumulator<uint8_t> {
    using type = uint8_t;
    using serialize_as = uint8_t;
    static constexpr uint32_t _base_type_mn = 0x88000001;
    static constexpr size_t _base_type_size = sizeof(uint8_t);
    static constexpr size_t _accumulator_size = sizeof(uint8_t);
    static constexpr size_t _s_size = sizeof(serialize_as);
};

template <> struct Accumulator<int16_t> {
    using type = int32_t;
    using serialize_as = int16_t;
    static constexpr uint32_t _base_type_mn = 0x88000002;
    static constexpr size_t _base_type_size = sizeof(int16_t);
    static constexpr size_t _accumulator_size = sizeof(int32_t);
    static constexpr size_t _s_size = sizeof(serialize_as);
};

template <> struct Accumulator<uint16_t> {
    using type = uint16_t;
    using serialize_as = uint16_t;
    static constexpr uint32_t _base_type_mn = 0x88000003;
    static constexpr size_t _base_type_size = sizeof(uint16_t);
    static constexpr size_t _accumulator_size = sizeof(uint16_t);
    static constexpr size_t _s_size = sizeof(serialize_as);
};

template <> struct Accumulator<int32_t> {
    using type = int64_t;
    using serialize_as = int32_t;
    static constexpr uint32_t _base_type_mn = 0x88000004;
    static constexpr size_t _base_type_size = sizeof(int32_t);
    static constexpr size_t _accumulator_size = sizeof(int64_t);
    static constexpr size_t _s_size = sizeof(serialize_as);
};

template <> struct Accumulator<uint32_t> {
    using type = uint32_t;
    using serialize_as = uint32_t;
    static constexpr uint32_t _base_type_mn = 0x88000005;
    static constexpr size_t _base_type_size = sizeof(uint32_t);
    static constexpr size_t _accumulator_size = sizeof(uint32_t);
    static constexpr size_t _s_size = sizeof(serialize_as);
};

template <> struct Accumulator<int64_t> {
    using type = int64_t;
    using serialize_as = int64_t;
    static constexpr uint32_t _base_type_mn = 0x88000006;
    static constexpr size_t _base_type_size = sizeof(int64_t);
    static constexpr size_t _accumulator_size = sizeof(int64_t);
    static constexpr size_t _s_size = sizeof(serialize_as);
};

template <> struct Accumulator<uint64_t> {
    using type = uint64_t;
    using serialize_as = uint64_t;
    static constexpr uint32_t _base_type_mn = 0x88000007;
    static constexpr size_t _base_type_size = sizeof(uint64_t);
    static constexpr size_t _accumulator_size = sizeof(uint64_t);
    static constexpr size_t _s_size = sizeof(serialize_as);
};

// Special handling for size_t as its implementation is different on
// Linux and macOS
template <Arithmetic T>
requires (std::is_same_v<T, size_t> && !std::is_same_v<T, uint64_t> && !std::is_same_v<T, uint32_t>)
struct Accumulator<T> {
    using type = size_t;
    using serialize_as = uint64_t;
    // Use the same magic number as uint64_t to avoid compatibility issues between Linux and macOS
    static constexpr uint32_t _base_type_mn = 0x88000007;
    static constexpr size_t _base_type_size = sizeof(size_t);
    static constexpr size_t _accumulator_size = sizeof(size_t);
    static constexpr size_t _s_size = sizeof(serialize_as);
};

// Handle half precision float types, disabled for now
//template <> struct Accumulator<float16_t> { using type = float; };
//template <> struct Accumulator<bfloat16_t> { using type = float; };

// Ensure full precision floats use the same type
template <> struct Accumulator<float> {
    using type = float;
    using serialize_as = uint32_t;
    static constexpr uint32_t _base_type_mn = 0x88000009;
    static constexpr size_t _base_type_size = sizeof(float);
    static constexpr size_t _accumulator_size = sizeof(float);
    static constexpr size_t _s_size = sizeof(serialize_as);
};

template <> struct Accumulator<double> {
    using type = double;
    using serialize_as = uint64_t;
    static constexpr uint32_t _base_type_mn = 0x8800000A;
    static constexpr size_t _base_type_size = sizeof(double);
    static constexpr size_t _accumulator_size = sizeof(double);
    static constexpr size_t _s_size = sizeof(serialize_as);
};

/**
 * Allow conversion of big endian --> little endian
 * @param val Value to convert
 * @returns Returns the std::byteswap if a conversion is needed
 */
template <typename T>
requires std::is_integral_v<T>
typename Accumulator<T>::serialize_as to_little_endian(T val) noexcept {
    // Get the serialization type
    using S = typename Accumulator<T>::serialize_as;
    S s_val = static_cast<S>(val);
    // Check the system's endian type
    if constexpr (std::endian::native == std::endian::little) {
        return s_val;
    }
    else {
        return std::byteswap(s_val);
    }
}

/**
 * Allow conversion of little endian --> big endian
 * @param val Value to convert
 * @returns Returns the std::byteswap if a conversion is needed
 */
template <typename T>
requires std::is_integral_v<T>
T to_big_endian(T val) noexcept {
    // Check the system's endian type
    if constexpr (std::endian::native == std::endian::big) {
        return val;
    }
    else {
        return std::byteswap(val);
    }
}

/**
 * Convert a value into an array of unsigned char
 * @param val Value to split
 * @returns Returns an array of sizeof(T) elements
 */
template <typename T>
requires std::is_arithmetic_v<T>
std::array<char, Accumulator<T>::_s_size> bytes(T val) noexcept {
    if constexpr (std::is_floating_point_v<T>) {
        return bytes(std::bit_cast<typename Accumulator<T>::serialize_as, T>(val));
    }
    else {
        using S = typename Accumulator<T>::serialize_as;
        return std::bit_cast<std::array<char, Accumulator<T>::_s_size>>(static_cast<S>(val));
    }
}

/**
 * Convert a floating point type to its serializable format
 * @param val Value to convert
 * @returns Returns uint32_t or uint64_t
 */
template <typename T>
requires std::is_floating_point_v<T>
typename Accumulator<T>::serialize_as float_to_int(T val) noexcept {
    return std::bit_cast<typename Accumulator<T>::serialize_as, T>(val);
}

/**
 * Convert an integer type into its corresponding floating point type
 * @param val Value to convert
 * @returns Returns either float or double
 */
template <typename T>
requires std::is_floating_point_v<T>
T int_to_float(typename Accumulator<T>::serialize_as val) noexcept {
    return std::bit_cast<T, typename Accumulator<T>::serialize_as>(val);
}

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
/**
 * Convert raw, serialized bytes into a target type
 * @param p Pointer to char
 * @param index Index to start reading from, updated on return
 * @returns Returns the converted value
 */
template <typename T>
requires std::is_arithmetic_v<T>
T from_bytes(const char* p, size_t& index) {
    // Ensure we have a valid pointer
    if (p == nullptr) {
        throw std::invalid_argument("Numerics::from_bytes: Invalid pointer provided.\n");
    }
    // Special handling for size_t
    if constexpr (std::is_same_v<T, size_t>) {
        // Create a buffer for size_t
        uint64_t temp_val = 0;
        std::memcpy(&temp_val, static_cast<const void*>(p + index), sizeof(uint64_t));
        // We know that our bytes are always in little endian, so check if we need to use
        // big endian on the host
        if constexpr (std::endian::native == std::endian::big) {
            temp_val = to_big_endian(temp_val);
        }
        // Ensure that temp_val can fit inside size_t
        if (temp_val > std::numeric_limits<size_t>::max()) {
            throw std::overflow_error("Numerics::from_bytes: Overflow detected during uint64_t --> size_t conversion.\n");
        }
        index += Accumulator<T>::_s_size;
        return static_cast<size_t>(temp_val);
    }
    else {
        // Create a destination
        T val = 0;
        // Use memcpy to read into T
        std::memcpy(&val, static_cast<const void*>(p + index), Accumulator<T>::_s_size);
        if constexpr (std::endian::native == std::endian::big) {
            val = to_big_endian(val);
        }
        index += Accumulator<T>::_s_size;
        return val;
    }
}

/**
 * Convert raw, serialized bytes into a target type
 * @param p Pointer to char
 * @returns Returns the converted value
 */
template <typename T>
requires std::is_arithmetic_v<T>
T from_bytes(const char* p) {
    // Ensure we have a valid pointer
    if (p == nullptr) {
        throw std::invalid_argument("Numerics::from_bytes: Invalid pointer provided.\n");
    }
    // Special handling for size_t
    if constexpr (std::is_same_v<T, size_t>) {
        // Create a buffer for size_t
        uint64_t temp_val = 0;
        std::memcpy(&temp_val, static_cast<const void*>(p), sizeof(uint64_t));
        // We know that our bytes are always in little endian, so check if we need to use
        // big endian on the host
        if constexpr (std::endian::native == std::endian::big) {
            temp_val = to_big_endian(temp_val);
        }
        // Ensure that temp_val can fit inside size_t
        if (temp_val > std::numeric_limits<size_t>::max()) {
            throw std::overflow_error("Numerics::from_bytes: Overflow detected during uint64_t --> size_t conversion.\n");
        }
        return static_cast<size_t>(temp_val);
    }
    else {
        // Create a destination
        T val = 0;
        // Use memcpy to read into T
        std::memcpy(&val, static_cast<const void*>(p), Accumulator<T>::_s_size);
        if constexpr (std::endian::native == std::endian::big) {
            val = to_big_endian(val);
        }
        return val;
    }
}
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)


}; // namespace Numerics_NS

#endif
