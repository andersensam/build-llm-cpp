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
 * AbsolutePositionalEmbedding.hpp: Get the absolute positional embedding of a sequence of token ids
 */

#ifndef ABSOLUTE_POSITIONAL_EMBEDDING_HPP
#define ABSOLUTE_POSITIONAL_EMBEDDING_HPP

/* Standard dependencies */
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

/* Local dependencies */
#include "AbstractEmbedding.hpp"
#include "ListTensorSlice.hpp"
#include "Numerics.hpp"
#include "SliceConfig.hpp"
#include "Tensor.hpp"

namespace AbsolutePositionalEmbedding_NS {

/* Magic numbers for serialization */
inline constexpr uint32_t ABSOLUTE_POSITIONAL_EMBEDDING_START_MAGIC_NUMBER = 0x88800020;
inline constexpr uint32_t ABSOLUTE_POSITIONAL_EMBEDDING_END_MAGIC_NUMBER = 0x88800021;

/* Use the AbstractEmbedding interface */
using AbstractEmbedding_NS::AbstractEmbedding;

/* Use ListTensorSlice */
using ListTensorSlice_NS::ListTensorSlice;

/* Use SliceConfig */
using SliceConfig_NS::IndexType;
using SliceConfig_NS::MatrixSliceConfig;

/* Use Tensor */
using Tensor_NS::Tensor;

// NOLINTBEGIN(cppcoreguidelines-special-member-functions)
/**
 * AbsolutePositionalEmbedding class
 */
template <typename T>
requires std::is_arithmetic_v<T>
class AbsolutePositionalEmbedding final : public AbstractEmbedding<T> {
/* Private data elements */
private:
    /**
     * Max sequence length for the model
     */
    size_t c_max_seq_len = 0;

    /**
     * Embedding dimension
     */
    size_t c_emb_dim = 0;

    /**
     * Shared pointer to a Tensor containing the positional embedding weights
     */
    std::shared_ptr<Tensor<T>> m_w_pos_emb = nullptr;

/* Public functions */
public:
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    /**
     * Default constructor for AbsolutePositionalEmbedding
     * @param max_seq_len Max sequence length for the model
     * @param emb_dim Embedding dimension
     */
    explicit AbsolutePositionalEmbedding(size_t max_seq_len, size_t emb_dim) : c_max_seq_len(max_seq_len), c_emb_dim(emb_dim) {
        // Ensure we have valid max_seq_len and emb_dim
        if (c_max_seq_len == 0 || c_emb_dim == 0) {
            throw std::invalid_argument("AbsolutePositionalEmbedding.AbsolutePositionalEmbedding: Invalid max seq len or embedding dim provided.\n");
        }
        // Create the positional embedding weight Tensor
        m_w_pos_emb = std::make_shared<Tensor<T>>(std::initializer_list{c_max_seq_len, c_emb_dim});
        // If using a floating point dtype, create random weights with positive and negative values
        if constexpr (std::is_floating_point_v<T>) {
            m_w_pos_emb->random(-2, 2);
        }
        else {
            m_w_pos_emb->random(0, 2);
        }
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    /**
     * Constructor for AbsolutePositionalEmbedding, accepting a pointer to char (bytes) and reading in the data
     * @param s Pointer to the a serialized AbsolutePositionalEmbedding (char)
     */
    explicit AbsolutePositionalEmbedding(const char* s) {
        // Ensure we have a valid pointer to serialized data
        if (s == nullptr) {
            throw std::invalid_argument("AbsolutePositionalEmbedding.AbsolutePositionalEmbedding: Invalid pointer provided.\n");
        }
        size_t index = 0;
        // Check that the magic number is correct
        uint32_t u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != ABSOLUTE_POSITIONAL_EMBEDDING_START_MAGIC_NUMBER) {
            throw std::invalid_argument("AbsolutePositionalEmbedding.AbsolutePositionalEmbedding: Invalid start magic number.\n");
        }
        // Get the target size of the LinearLayer
        size_t total_size = Numerics_NS::from_bytes<size_t>(s, index);
        // Validate the dtype matches the template parameter
        u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != Numerics_NS::Accumulator<T>::_base_type_mn) {
            throw std::runtime_error("AbsolutePositionalEmbedding.AbsolutePositionalEmbedding: Template parameter does not match serialized dtype.\n");
        }
        // Get the max_seq_len
        if (index >= total_size) { throw std::runtime_error("AbsolutePositionalEmbedding.AbsolutePositionalEmbedding: Invalid index or total size for deserializing.\n"); }
        c_max_seq_len = Numerics_NS::from_bytes<size_t>(s, index);
        // Get the emb_dim
        if (index >= total_size) { throw std::runtime_error("AbsolutePositionalEmbedding.AbsolutePositionalEmbedding: Invalid index or total size for deserializing.\n"); }
        c_emb_dim = Numerics_NS::from_bytes<size_t>(s, index);
        // Initialize the m_w_pos_emb Tensor
        if (index >= total_size) { throw std::runtime_error("AbsolutePositionalEmbedding.AbsolutePositionalEmbedding: Invalid index or total size for deserializing.\n"); }
        m_w_pos_emb = std::make_shared<Tensor<T>>(s + index);
        index += m_w_pos_emb->s_size();
        // Ensure the end magic number is valid
        if (index >= total_size) { throw std::runtime_error("AbsolutePositionalEmbedding.AbsolutePositionalEmbedding: Invalid index or total size for deserializing.\n"); }
        u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != ABSOLUTE_POSITIONAL_EMBEDDING_END_MAGIC_NUMBER) {
            throw std::runtime_error("AbsolutePositionalEmbedding.AbsolutePositionalEmbedding: Invalid end magic number.\n");   
        }
        if (index != total_size) {
            throw std::runtime_error("AbsolutePositionalEmbedding.AbsolutePositionalEmbedding: Reached end of object but total_size indicates there is more data to be read.\n");
        }
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

    /**
     * Default destructor for AbsolutePositionalEmbedding
     */
    ~AbsolutePositionalEmbedding() override {
        // Do nothing
    }

    /**
     * Lookup the positional embeddings for a sequence of tokens
     * @param input Reference to a vector with the positional ids of the tokens
     * @returns Returns a ListTensorSlice containing the positional embeddings of the tokens
     */
    const ListTensorSlice<T> get_embeddings(const std::vector<size_t>& input) const override {
        if (input.empty()) {
            throw std::invalid_argument("AbsolutePositionalEmbedding.get_embeddings: Input cannot be empty.\n");
        }
        if (input.size() > c_max_seq_len) {
            throw std::invalid_argument("AbsolutePositionalEmbedding.get_embeddings: Too many elements in input vector.\n");
        }
        // Check the input to ensure don't have a positional id larger than max_seq_len
        auto max_pos_id = std::max_element(input.begin(), input.end());
        if (max_pos_id == input.end()) {
            // If it is input.end() we ran into some kind of issue
            throw std::invalid_argument("AbsolutePositionalEmbedding.get_embeddings: Error calculating max token id in input.\n");
        }
        if (*max_pos_id >= c_max_seq_len) {
            throw std::invalid_argument("AbsolutePositionalEmbedding.get_embeddings: Invalid position id provided.\n");
        }
        // Create a MatrixSliceConfig to lookup the tokens in m_w_pos_emb, filter on dim0, with the
        // input as a list of embeddings to pull -- apply no filters
        MatrixSliceConfig msc(0, IndexType::LIST, input, 1, {}, {});
        // Instantiate the ListTensorSlice
        ListTensorSlice<T> result(m_w_pos_emb, msc);
        return result;
    }

    /**
     * Get the embeddings for an input, returning a ListTensorSlice from the underlying weight Tensor
     * @param input Const ref to a Tensor containing the token ids
     * @returns Returns a zero-copy ListTensorSlice with the token embeddings
     */
    const ListTensorSlice<T> get_embeddings(const Tensor<size_t>& input) const override {
        // Ensure that we have a valid input
        if (input.rank() == 0 || input.rank() > 2) {
            throw std::invalid_argument("AbsolutePositionalEmbedding.get_embeddings: Input Tensor must have rank 1 or 2.\n");
        }
        // If we have a rank-2 input, ensure we have shape [1, seq_len]
        if (input.rank() == 2) {
            if (input.extent(0) != 1) {
                throw std::invalid_argument("AbsolutePositionalEmbedding.get_embeddings: A rank-2 input Tensor must have shape [1, seq_len].\n");
            }
        }
        // Ensure we don't get more elements than c_max_seq_len
        if (input.elements() > c_max_seq_len) {
            throw std::invalid_argument("AbsolutePositionalEmbedding.get_embeddings: Too many elements in input Tensor.\n");
        }
        // Check the input to ensure don't have a positional id larger than max_seq_len
        size_t max_pos_id = input.max();
        if (max_pos_id >= c_max_seq_len) {
            throw std::invalid_argument("AbsolutePositionalEmbedding.get_embeddings: Invalid position id provided.\n");
        }
        // Create a vector to lookup the tokens via MatrixSliceConfig
        std::vector<size_t> token_idx = input.to_vector();
        // Create a MatrixSliceConfig to lookup the tokens in m_w_emb, filter on dim0, with the
        // input as a list of embeddings to pull -- apply no filters
        MatrixSliceConfig msc(0, IndexType::LIST, token_idx, 1, {}, {});
        // Instantiate the ListTensorSlice
        ListTensorSlice<T> result(m_w_pos_emb, msc);
        return result;
    }

    /**
     * Calculate the size for serialization, including the raw data and any metadata
     * @returns Returns a size_t with total serialized size
     * NOTE: Due to inconsistencies with size_t representation across systems,
     * we use Numerics.hpp to calculate the conversion to fixed-width types, thus
     * the size returned by size() may not reflect what is being used in memory
     */
    size_t s_size() const override {
        // Start with the magic numbers and dtype
        size_t result = 3 * Numerics_NS::Accumulator<uint32_t>::_s_size;
        // Add the max seq len, emb dim, and total size
        result += 3 * Numerics_NS::Accumulator<size_t>::_s_size;
        // Add the weights Tensor
        result += m_w_pos_emb->s_size();
        return result;
    }

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    /**
     * Serialize AbsolutePositionalEmbedding
     * @returns Returns a vector of bytes / char containing the layer
     */
    std::vector<char> serialize() const override {
        // Persist the target size since we will do bounds checking
        size_t target_size = s_size();
        // Allocate memory using unsigned char (same as std::byte)
        std::vector<char> result(target_size);
        // Grab a pointer to the underlying data
        char* data = result.data();
        // Keep track of the index
        size_t index = 0;
        // As an extra safety check, ensure that index will never exceed s_size
        // Write the start magic number
        if (index + Numerics_NS::Accumulator<uint32_t>::_s_size >= target_size) {
            throw std::runtime_error("AbsolutePositionalEmbedding.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(ABSOLUTE_POSITIONAL_EMBEDDING_START_MAGIC_NUMBER))) {
            data[index] = b;
            ++index;
        }
        // Write the serialized size size
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("AbsolutePositionalEmbedding.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(target_size))) {
            data[index] = b;
            ++index;
        }
        // Write the dtype
        if (index + Numerics_NS::Accumulator<uint32_t>::_s_size >= target_size) {
            throw std::runtime_error("AbsolutePositionalEmbedding.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(Numerics_NS::Accumulator<T>::_base_type_mn))) {
            data[index] = b;
            ++index;
        }
        // Write the max_seq_len
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("AbsolutePositionalEmbedding.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(c_max_seq_len))) {
            data[index] = b;
            ++index;
        }
        // Write the emb_dim
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("AbsolutePositionalEmbedding.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(c_emb_dim))) {
            data[index] = b;
            ++index;
        }
        // Write the embedding weight Tensor
        if (index + m_w_pos_emb->s_size() >= target_size) {
            throw std::runtime_error("AbsolutePositionalEmbedding.serialize: Error serializing -- invalid index or s_size.\n");
        }
        std::vector<char> ts = m_w_pos_emb->serialize();
        std::memcpy(data + index, ts.data(), ts.size());
        index += ts.size();
        // Write the end magic number
        if (index + Numerics_NS::Accumulator<uint32_t>::_s_size > target_size) {
            throw std::runtime_error("AbsolutePositionalEmbedding.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(ABSOLUTE_POSITIONAL_EMBEDDING_END_MAGIC_NUMBER))) {
            data[index] = b;
            ++index;
        }
        return result;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace AbsolutePositionalEmbedding_NS

#endif
