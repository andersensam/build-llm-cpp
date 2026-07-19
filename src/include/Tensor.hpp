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
 * @version: 2026-07-19
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef TENSOR_HPP
#define TENSOR_HPP

/* Standard dependencies */
#include <algorithm>
#include <cstring>
#include <expected>
#include <format>
#include <functional>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <stdexcept>
#include <vector>

/* Local dependencies */
#include "Log.hpp"

namespace Tensor_NS {

// Control verbosity of Tensor logging
//constexpr bool TENSOR_DEBUG = true;

/**
 * Template for a generic Tensor, where T can be any numeric type, like a variety of int,
 * double, float, etc.
 */
template <typename T> class Tensor {
// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays, cppcoreguidelines-pro-bounds-pointer-arithmetic)
/* Private data elements */
private:
    /**
     * Smart pointer representing the memory block allocated to the Tensor. We use a smart pointer here
     * to aid in lifecycle management of Matrix
     */
    std::unique_ptr<T[]> m_data = nullptr;

    /**
     * Total number of elements stored in the Tensor
     */
    size_t c_elements = 1;

    /**
     * Tensor rank, i.e. how many dimensions a Tensor represents, anywhere from 0..N, where rank 0 represents
     * a single scalar value, rank 1 is a vector of values, rank 2 is a matrix, and so on
     */
    size_t c_rank = 0;

    /**
     * Tensor stride, i.e. how many values we need to jump forward between rows, columns, etc. across total
     * tensor rank / dimension
     */
    std::vector<size_t> m_stride = std::vector<size_t>();

    /**
     * Tensor dimensions, sequenced from outer to inner dim
     */
    std::vector<size_t> m_dims = std::vector<size_t>();

/* Private methods */
    /**
     * Calculate the offset from a set of input cooridnates
     * @param c Coordinates to calculate the offset from
     * @returns Returns a size_t representing the offset we want
     */
    std::expected<size_t, std::string> _get_offset(const std::initializer_list<size_t>& c) const {

        // Do some pointer arithmetic to calculate the exact address to retrieve from m_data
        size_t target_offset = 0;
        // Ensure the coordinates are valid for our Tensor
        for (size_t i = 0; i < c_rank; ++i) {
            if (c.begin()[i] >= m_dims.at(i)) {
                return std::unexpected(std::format("Index {} exceeds dim ({}). Max possible index is {}.", 
                                                    c.begin()[i], m_dims.at(i), m_dims.at(i) - 1));
            }
            target_offset += c.begin()[i] * m_stride.at(i);
            //std::cout << "Current offset: " << target_offset << "\n";
        }
        return target_offset;
    }

/* Public methods */
public:
    /**
     * Default constructor for Tensor, taking in a reference to a vector containing the desired dimensions
     * for the resulting Tensor
     * @param dims Const reference to std::vector<size_t> containing the desired dimensions
     */
    Tensor(const std::initializer_list<size_t>& dims) : c_rank(dims.size()), m_stride(dims.size()), m_dims(dims) {

        // Handle the case where we have a rank-0 tensor (scalar value). Allocate space for the singular
        // element and then return immediately
        if (c_rank == 0) {
            m_data = std::make_unique<T[]>(1);
            std::memset(m_data.get(), 0, sizeof(T));
            return;
        }
        // Store the target 1-D representation of the desired size of our Tensor
        for (const size_t& ds : dims) {
            if (ds <= 1) {
                throw std::invalid_argument(std::format("Tensor.Tensor: invalid dim ({}) provided to Tensor. Dims must be >= 1 or should be omitted.", ds));
            }
            c_elements *= ds;
        }
        // Allocate the block of memory for the tensor
        m_data = std::make_unique<T[]>(c_elements);
        std::memset(m_data.get(), 0, sizeof(T) * c_elements);
        // Calculate the stides needed to get between dims
        m_stride.at(0) = 1;
        for (size_t i = 1; i < c_rank; ++i) {
            m_stride.at(i) = m_stride.at(i - 1) * m_dims.at(i);
        }
    }

    /**
     * Get or set a value at a specific coordinate inside the Tensor
     * @param target The coordinate we want to fetch from the Tensor
     * @returns Returns a reference to the value that can be updated
     */
    T& at(const std::initializer_list<size_t>& target) {

        // Handle the case where we have a rank-0 tensor
        if (c_rank == 0) {
            return m_data.get()[0];
        }
        // Validate we have the correct number of elements in our coordinates
        if (target.size() != c_rank) {
            throw std::invalid_argument(std::format("Invalid number of coordinates provided to at, expected {} but got {}.\n", c_rank, target.size()));
        }
        // Calculate the offset we want, or get the error back
        auto target_offset = _get_offset(target);
        if (target_offset) {
            //std::cout << "Final offset.. attempting to get value @ " << target_offset.value() << "\n";
            return m_data.get()[target_offset.value()];
        }
        throw std::out_of_range(std::format("Tensor.at: Out of range: {}", target_offset.error()));
    }

    /**
     * Get a value at a specific coordinate inside the Tensor
     * @param target The coordinate we want to fetch from the tensor
     * @returns Returns the value at the coordinate
     */
    const T& at(const std::initializer_list<size_t>& target) const {
        // We use the same logic as the mutable case, just casting to a const reference
        return at(target);
    }
// NOLINTEND(cppcoreguidelines-avoid-c-arrays, cppcoreguidelines-pro-bounds-pointer-arithmetic)
};

}; // namespace Tensor_NS

#endif
