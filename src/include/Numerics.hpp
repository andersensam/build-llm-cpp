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
requires std::is_arithmetic_v<T>
struct Accumulator;

// Handle the integer (integral) types first
template <> struct Accumulator<int8_t> { using type = int16_t; };
template <> struct Accumulator<uint8_t> { using type = uint8_t; };
template <> struct Accumulator<int16_t> { using type = int32_t; };
template <> struct Accumulator<uint16_t> { using type = uint16_t; };
template <> struct Accumulator<int32_t> { using type = int64_t; };
template <> struct Accumulator<uint32_t> { using type = uint32_t; };
template <> struct Accumulator<int64_t> { using type = int64_t; };
template <> struct Accumulator<uint64_t> { using type = uint64_t; };
template <> struct Accumulator<size_t> { using type = size_t; };

// Handle half precision float types, disabled for now
//template <> struct Accumulator<float16_t> { using type = float; };
//template <> struct Accumulator<bfloat16_t> { using type = float; };

// Ensure full precision floats use the same type
template <> struct Accumulator<float> { using type = float; };
template <> struct Accumulator<double> { using type = double; };

}; // namespace Numerics_NS

#endif
