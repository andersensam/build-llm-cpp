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
 * @version: 2026-10-20
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef NUMERICS_HPP
#define NUMERICS_HPP

/* Standard dependencies */
#include <cstddef>
#include <cstdint>
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
    static constexpr uint32_t _base_type_mn = 0x880000;
    static constexpr size_t _base_type_size = sizeof(int8_t);
    static constexpr size_t _accumulator_size = sizeof(int16_t); 
};
template <> struct Accumulator<uint8_t> {
    using type = uint8_t;
    static constexpr uint32_t _base_type_mn = 0x880001;
    static constexpr size_t _base_type_size = sizeof(uint8_t);
    static constexpr size_t _accumulator_size = sizeof(uint8_t); 
};
template <> struct Accumulator<int16_t> {
    using type = int32_t;
    static constexpr uint32_t _base_type_mn = 0x880002;
    static constexpr size_t _base_type_size = sizeof(int16_t);
    static constexpr size_t _accumulator_size = sizeof(int32_t); 
};
template <> struct Accumulator<uint16_t> {
    using type = uint16_t;
    static constexpr uint32_t _base_type_mn = 0x880003;
    static constexpr size_t _base_type_size = sizeof(uint16_t);
    static constexpr size_t _accumulator_size = sizeof(uint16_t);
};
template <> struct Accumulator<int32_t> {
    using type = int64_t;
    static constexpr uint32_t _base_type_mn = 0x880004;
    static constexpr size_t _base_type_size = sizeof(int32_t);
    static constexpr size_t _accumulator_size = sizeof(int64_t);
};
template <> struct Accumulator<uint32_t> {
    using type = uint32_t;
    static constexpr uint32_t _base_type_mn = 0x880005;
    static constexpr size_t _base_type_size = sizeof(uint32_t);
    static constexpr size_t _accumulator_size = sizeof(uint32_t);
};
template <> struct Accumulator<int64_t> {
    using type = int64_t;
    static constexpr uint32_t _base_type_mn = 0x880006;
    static constexpr size_t _base_type_size = sizeof(int64_t);
    static constexpr size_t _accumulator_size = sizeof(int64_t);
};
template <> struct Accumulator<uint64_t> {
    using type = uint64_t;
    static constexpr uint32_t _base_type_mn = 0x880007;
    static constexpr size_t _base_type_size = sizeof(uint64_t);
    static constexpr size_t _accumulator_size = sizeof(uint64_t);
};

// Special handling for size_t as its implementation is different on
// Linux and macOS
template <Arithmetic T>
requires (std::is_same_v<T, size_t> && !std::is_same_v<T, uint64_t> && !std::is_same_v<T, uint32_t>)
struct Accumulator<T> {
    using type = size_t;
    static constexpr uint32_t _base_type_mn = 0x880008;
    static constexpr size_t _base_type_size = sizeof(size_t);
    static constexpr size_t _accumulator_size = sizeof(size_t);
};

// Handle half precision float types, disabled for now
//template <> struct Accumulator<float16_t> { using type = float; };
//template <> struct Accumulator<bfloat16_t> { using type = float; };

// Ensure full precision floats use the same type
template <> struct Accumulator<float> {
    using type = float;
    static constexpr uint32_t _base_type_mn = 0x880009;
    static constexpr size_t _base_type_size = sizeof(float);
    static constexpr size_t _accumulator_size = sizeof(float);
};
template <> struct Accumulator<double> {
    using type = double;
    static constexpr uint32_t _base_type_mn = 0x88000A;
    static constexpr size_t _base_type_size = sizeof(double);
    static constexpr size_t _accumulator_size = sizeof(double);
};

}; // namespace Numerics_NS

#endif
