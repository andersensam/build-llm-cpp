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
 * @version: 2026-10-12
 *
 * General Notes:
 *
 * Embedding.hpp: Take in a sequence of tokens and return their embeddings
 */

#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP

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

namespace Embedding_NS {

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
    std::shared_ptr<Tensor<T>> m_w_emb;

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
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace Embedding_NS

#endif
