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
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

/* Local dependencies */
#include "AbstractEmbedding.hpp"
#include "ListTensorSlice.hpp"
#include "SliceConfig.hpp"
#include "Tensor.hpp"

namespace AbsolutePositionalEmbedding_NS {

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
    std::shared_ptr<Tensor<T>> m_w_pos_emb;

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
            throw std::invalid_argument("AbsolutePositionalEmbedding.get_embeddings: Invalid position id provided.\n");
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
        if (input.elements() >= c_max_seq_len) {
            throw std::invalid_argument("AbsolutePositionalEmbedding.get_embeddings: Invalid position id provided.\n");
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
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace AbsolutePositionalEmbedding_NS

#endif
