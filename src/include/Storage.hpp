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
 * @version: 2026-09-30
 *
 * General Notes:
 *
 * Storage.hpp implements a lightweight wrapper around a heap-allocated memory block, consumed
 * by Tensor and TensorSlice
 * 
 */

#ifndef STORAGE_HPP
#define STORAGE_HPP

/* Standard dependencies */
#include <cstddef>
#include <cstring>
#include <memory>
#include <type_traits>

namespace Storage_NS {

// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays)
/**
 * Storage class, containing the raw data pointer used by Tensor and TensorSlice
 */
template <typename T>
requires std::is_arithmetic_v<T>
struct Storage {
    /**
     * Number of elements in the memory block
     */
    size_t c_elements = 0;

    /**
     * Smart pointer to a block of memory
     */
    std::unique_ptr<T[]> m_data = nullptr;

    /**
     * Constructor for Storage, taking in the desired number of elements
     * @param elements Number of elements desired in the memory block
     */
    explicit Storage(size_t elements) : c_elements(elements), m_data(std::make_unique<T[]>(c_elements)) {
        // Clear out any possible garbage and set the block to zero
        std::memset(m_data.get(), 0, sizeof(T) * c_elements);
    }
};
// NOLINTEND(cppcoreguidelines-avoid-c-arrays)

}; // namespace Storage_NS

#endif
