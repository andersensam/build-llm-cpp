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
 * @version: 2026-10-14
 *
 * General Notes:
 *
 * AbstractEmbedding.hpp: Interface for embeddings
 */

#ifndef ABSTRACT_EMBEDDING_HPP
#define ABSTRACT_EMBEDDING_HPP

/* Standard dependencies */
#include <cstddef>
#include <vector>

/* Local dependencies */
#include "ListTensorSlice.hpp"
#include "Tensor.hpp"

namespace AbstractEmbedding_NS {

/* Use ListTensorSlice */
using ListTensorSlice_NS::ListTensorSlice;

/* Use Tensor */
using Tensor_NS::Tensor;

// NOLINTBEGIN(cppcoreguidelines-special-member-functions)
/**
 * Define a common interface for embeddings
 */
template <typename T>
requires std::is_arithmetic_v<T>
class AbstractEmbedding {
/* Public functions */
public:
    /**
     * Virtual destructor for AbstractEmbedding
     */
    virtual ~AbstractEmbedding() = default;

    /**
     * Get the embeddings for an input, returning a ListTensorSlice from the underlying weight Tensor
     * @param input Const ref to a vector containing the token ids
     * @returns Returns a zero-copy ListTensorSlice with the token embeddings
     */
    virtual const ListTensorSlice<T> get_embeddings(const std::vector<size_t>& input) const = 0;

    /**
     * Get the embeddings for an input, returning a ListTensorSlice from the underlying weight Tensor
     * @param input Const ref to a Tensor containing the token ids
     * @returns Returns a zero-copy ListTensorSlice with the token embeddings
     */
    virtual const ListTensorSlice<T> get_embeddings(const Tensor<size_t>& input) const = 0;

    /**
     * Calculate the size for serialization, including the raw data and any metadata
     * @returns Returns a size_t with total serialized size
     * NOTE: Due to inconsistencies with size_t representation across systems,
     * we use Numerics.hpp to calculate the conversion to fixed-width types, thus
     * the size returned by size() may not reflect what is being used in memory
     */
    virtual size_t s_size() const = 0;

    /**
     * Serialize an object
     * @returns Returns a vector of bytes / char containing the layer
     */
    virtual std::vector<char> serialize() const = 0;
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace AbstractEmbedding_NS

#endif