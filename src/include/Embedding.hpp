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
 * @version: 2026-10-23
 *
 * General Notes:
 *
 * Embedding.hpp: Take in a sequence of tokens and return their embeddings
 */

#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP

/* Standard dependencies */
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

/* Local dependencies */
#include "AbstractEmbedding.hpp"
#include "ListTensorSlice.hpp"
#include "Numerics.hpp"
#include "SliceConfig.hpp"
#include "Tensor.hpp"

namespace Embedding_NS {

/* Magic numbers for serialization */
inline constexpr uint32_t EMBEDDING_START_MAGIC_NUMBER = 0x88800030;
inline constexpr uint32_t EMBEDDING_END_MAGIC_NUMBER = 0x88800031;

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
 * Embedding class, used for calculating the embeddings for a sequence of tokens
 */
template <typename T>
requires std::is_arithmetic_v<T>
class Embedding final : public AbstractEmbedding<T> {
/* Private data elements */
private:
    /**
     * The vocab size of the model
     */
    size_t c_vocab_size = 0;

    /**
     * The embedding dimension
     */
    size_t c_emb_dim = 0;

    /**
     * Shared pointer to a Tensor containing the embedding weights 
     */
    std::shared_ptr<Tensor<T>> m_w_emb = nullptr;

/* Public functions */
public:
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    /**
     * Default constructor for Embedding
     * @param vocab_size The model's vocab size
     * @param emb_dim Embedding dimension
     */
    explicit Embedding(size_t vocab_size, size_t emb_dim) : c_vocab_size(vocab_size), c_emb_dim(emb_dim) {
        // Validate the vocab size and embedding dim
        if (c_vocab_size == 0 || c_emb_dim == 0) {
            throw std::invalid_argument("Embedding.Embedding: Vocab size and embedding dim must be > 0.\n");
        }
        // Create the embedding weight Tensor using std::make_shared to avoid copies
        m_w_emb = std::make_shared<Tensor<T>>(std::initializer_list<size_t>{c_vocab_size, c_emb_dim});
        // If using a floating point dtype, create random weights with positive and negative values
        if constexpr (std::is_floating_point_v<T>) {
            m_w_emb->random(-2, 2);
        }
        else {
            m_w_emb->random(0, 2);
        }
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    /**
     * Constructor for Embedding, accepting a pointer to char (bytes) and reading in the data
     * @param s Pointer to the a serialized Embedding (char)
     */
    explicit Embedding(const char* s) {
        // Ensure we have a valid pointer to serialized data
        if (s == nullptr) {
            throw std::invalid_argument("Embedding.Embedding: Invalid pointer provided.\n");
        }
        size_t index = 0;
        // Check that the magic number is correct
        uint32_t u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != EMBEDDING_START_MAGIC_NUMBER) {
            throw std::invalid_argument("Embedding.Embedding: Invalid start magic number.\n");
        }
        // Get the target size of the LinearLayer
        size_t total_size = Numerics_NS::from_bytes<size_t>(s, index);
        // Validate the dtype matches the template parameter
        u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != Numerics_NS::Accumulator<T>::_base_type_mn) {
            throw std::runtime_error("Embedding.Embedding: Template parameter does not match serialized dtype.\n");
        }
        // Get the vocab_size
        if (index >= total_size) { throw std::runtime_error("Embedding.Embedding: Invalid index or total size for deserializing.\n"); }
        c_vocab_size = Numerics_NS::from_bytes<size_t>(s, index);
        // Get the emb_dim
        if (index >= total_size) { throw std::runtime_error("Embedding.Embedding: Invalid index or total size for deserializing.\n"); }
        c_emb_dim = Numerics_NS::from_bytes<size_t>(s, index);
        if (c_vocab_size == 0 || c_emb_dim == 0) {
            throw std::invalid_argument("Embedding.Embedding: Vocab size and embedding dim must be > 0.\n");
        }
        // Initialize the m_w_emb Tensor
        if (index >= total_size) { throw std::runtime_error("Embedding.Embedding: Invalid index or total size for deserializing.\n"); }
        m_w_emb = std::make_shared<Tensor<T>>(s + index);
        index += m_w_emb->s_size();
        // Validate m_w_emb Tensor has the correct properties
        if (m_w_emb->rank() != 2 || m_w_emb->extent(0) != c_vocab_size || m_w_emb->extent(1) != c_emb_dim) {
            throw std::runtime_error("Embedding.Embedding: Invalid Tensor instantiated from serialized config.\n");
        }
        // Ensure the end magic number is valid
        if (index >= total_size) { throw std::runtime_error("Embedding.Embedding: Invalid index or total size for deserializing.\n"); }
        u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != EMBEDDING_END_MAGIC_NUMBER) {
            throw std::runtime_error("Embedding.Embedding: Invalid end magic number.\n");   
        }
        if (index != total_size) {
            throw std::runtime_error("Embedding.Embedding: Reached end of object but total_size indicates there is more data to be read.\n");
        }
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

    /**
     * Default destructor for Embedding
     */
    ~Embedding() override {
        // Do nothing
    }

    /**
     * Lookup the embeddings for a sequence of tokens
     * @param input Const ref to a vector of token ids
     * @returns Returns a ListTensorSlice containing the embeddings for the tokens
     */
    const ListTensorSlice<T> get_embeddings(const std::vector<size_t>& input) const override {
        if (input.empty()) {
            throw std::invalid_argument("Embedding.get_embeddings: Input cannot be empty.\n");
        }
        // Check the input to ensure we don't have any token ids >= c_vocab_size
        auto max_token_id = std::max_element(input.begin(), input.end());
        if (max_token_id == input.end()) {
            // If it is input.end() we ran into some kind of issue
            throw std::invalid_argument("Embedding.get_embeddings: Error calculating max token id in input.\n");
        }
        if (*max_token_id >= c_vocab_size) {
            throw std::invalid_argument("Embedding.get_embeddings: Unknown token id present in input.\n");
        }
        // Create a MatrixSliceConfig to lookup the tokens in m_w_emb, filter on dim0, with the
        // input as a list of embeddings to pull -- apply no filters
        MatrixSliceConfig msc(0, IndexType::LIST, input, 1, {}, {});
        // Instantiate the ListTensorSlice
        ListTensorSlice<T> result(m_w_emb, msc);
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
            throw std::invalid_argument("Embedding.get_embeddings: Input Tensor must have rank 1 or 2.\n");
        }
        // If we have a rank-2 input, ensure we have shape [1, seq_len]
        if (input.rank() == 2) {
            if (input.extent(0) != 1) {
                throw std::invalid_argument("Embedding.get_embeddings: A rank-2 input Tensor must have shape [1, seq_len].\n");
            }
        }
        // Check the input to ensure we don't have any token ids >= c_vocab_size
        size_t max_token_id = input.max();
        if (max_token_id >= c_vocab_size) {
            throw std::invalid_argument("Embedding.get_embeddings: Unknown token id present in input.\n");
        }
        // Create a vector to lookup the tokens via MatrixSliceConfig
        std::vector<size_t> token_idx = input.to_vector();
        // Create a MatrixSliceConfig to lookup the tokens in m_w_emb, filter on dim0, with the
        // input as a list of embeddings to pull -- apply no filters
        MatrixSliceConfig msc(0, IndexType::LIST, token_idx, 1, {}, {});
        // Instantiate the ListTensorSlice
        ListTensorSlice<T> result(m_w_emb, msc);
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
        result += m_w_emb->s_size();
        return result;
    }

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    /**
     * Serialize Embedding
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
            throw std::runtime_error("Embedding.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(EMBEDDING_START_MAGIC_NUMBER))) {
            data[index] = b;
            ++index;
        }
        // Write the serialized size size
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("Embedding.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(target_size))) {
            data[index] = b;
            ++index;
        }
        // Write the dtype
        if (index + Numerics_NS::Accumulator<uint32_t>::_s_size >= target_size) {
            throw std::runtime_error("Embedding.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(Numerics_NS::Accumulator<T>::_base_type_mn))) {
            data[index] = b;
            ++index;
        }
        // Write the vocab_size
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("Embedding.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(c_vocab_size))) {
            data[index] = b;
            ++index;
        }
        // Write the emb_dim
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("Embedding.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(c_emb_dim))) {
            data[index] = b;
            ++index;
        }
        // Write the embedding weight Tensor
        if (index + m_w_emb->s_size() >= target_size) {
            throw std::runtime_error("Embedding.serialize: Error serializing -- invalid index or s_size.\n");
        }
        std::vector<char> ts = m_w_emb->serialize();
        std::memcpy(data + index, ts.data(), ts.size());
        index += ts.size();
        // Write the end magic number
        if (index + Numerics_NS::Accumulator<uint32_t>::_s_size > target_size) {
            throw std::runtime_error("Embedding.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(EMBEDDING_END_MAGIC_NUMBER))) {
            data[index] = b;
            ++index;
        }
        return result;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace Embedding_NS

#endif
