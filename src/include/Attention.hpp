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
 * @version: 2026-09-08
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef ATTENTION_HPP
#define ATTENTION_HPP

/* Standard dependencies */
#include <cmath>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

/* Local dependencies */
#include "Log.hpp"
#include "Tensor.hpp"
#include "TensorSlice.hpp"

namespace Attention_NS {

using Tensor_NS::Tensor;
using Tensor_NS::CausalMaskType;
using TensorSlice_NS::TensorSlice;
using TensorSlice_NS::MatrixSliceConfig;
using TensorSlice_NS::IndexType;

// NOLINTBEGIN(cppcoreguidelines-special-member-functions)
/**
 * Abstract base class for all attention implementations
 */
template <typename T> 
requires std::is_arithmetic_v<T>
class Attention {
/* Public functions */
public:
    /**
     * Destructor, to be implemented in the individual Attention impls
     */
    virtual ~Attention() = default;

    /**
     * Run the forward pass of attention, returning a context Tensor for the inputs
     * @param input Const ref to the input Tensor
     * @returns Returns a new Tensor with the calculated context
     */
    virtual Tensor<T> forward(const Tensor<T>& input) const = 0;
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

/**
 * Causal Attention implementation
 */
template <typename T> 
requires std::is_arithmetic_v<T>
class CausalAttention : public Attention<T> {
/* Private data elemenets */
private:
    /**
     * Embedding dimension
     */
    size_t c_emb_dim = 0;

    /**
     * Output dimension for the attention head
     */
    size_t c_output_dim = 0;

    /**
     * Dropout percentage (> 0, < 1)
     */
    float c_dropout = 0.0;

    /**
     * Query weights
     */
    Tensor<T> m_w_query;

    /**
     * Key weights
     */
    Tensor<T> m_w_key;

    /**
     * Value weights
     */
    Tensor<T> m_w_value;

/* Public functions */
public:
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    /**
     * Constructor for CausalAttention, taking in the embedding dim, output dim, and whether or not to use dropout
     * @param emb_dim Embedding (input) dimension
     * @param output_dim Output dimension
     * @param dropout Float >= 0, < 1 (set to 0 to disable)
     */
    CausalAttention(size_t emb_dim, size_t output_dim, float dropout) : c_emb_dim(emb_dim), c_output_dim(emb_dim), c_dropout(dropout),
                                                                        m_w_query({emb_dim, output_dim}), m_w_key({emb_dim, output_dim}),
                                                                        m_w_value({emb_dim, output_dim}) {
        // Ensure we get a dropout within the acceptable range
        if (dropout < 0 || dropout >= 1) {
            throw std::invalid_argument("CausalAttention.CausalAttention: Dropout must be >= 0, < 1.\n");
        }
        // Initialize the three matrices with random values
        m_w_query.random(-2, 2);
        m_w_key.random(-2, 2);
        m_w_value.random(-2, 2);
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    /**
     * Calculate the context for a given input
     * @param input Const ref to an input Tensor
     * @returns Returns a new Tensor with the calculated context
     */
    Tensor<T> forward(const Tensor<T>& input) const override {
        // Get the QKV for the given input
        Tensor<T> queries = Tensor_NS::matmul(input, m_w_query);
        Tensor<T> keys = Tensor_NS::matmul(input, m_w_key);
        Tensor<T> values = Tensor_NS::matmul(input, m_w_value);
        // Transpose keys for calculating attention scores
        keys.transpose();
        Tensor<T> attn_scores = Tensor_NS::matmul(queries, keys);
        // Apply masking to the attention scores
        attn_scores.apply_mask(CausalMaskType::UPPER);
        // Apply scaling
        attn_scores /= static_cast<T>(std::sqrt(c_emb_dim));
        // Apply softmax
        attn_scores.softmax(0);
        // Apply dropout if enabled
        if (c_dropout > 0) {
            attn_scores.apply_dropout(c_dropout);
        }
        // Return the context
        return Tensor_NS::matmul(attn_scores, values);
    }

};

/**
 * Class for storing the TensorSlices associated with each head in
 * Multi-Head Attention
 */
template <typename T>
requires std::is_arithmetic_v<T>
class AttentionHead {
/* Private data elements */
private:
    /**
     * The id of the Attention Head
     */
    size_t c_id = 0;

    /**
     * The MatrixSliceConfig used to build the Attention Head
     */
    MatrixSliceConfig c_msc;

    /**
     * Query Matrix TensorSlice
     */
    TensorSlice<T> m_w_query;

    /**
     * Key Matrix TensorSlice
     */
    TensorSlice<T> m_w_key;

    /**
     * Value Matrix TensorSlice
     */
    TensorSlice<T> m_w_value;

/* Public functions */
public:
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    /**
     * Constructor for AttentionHead
     * @param id Id to assign to the attention head
     * @param start_dim Dimension to start sharding
     * @param end_dim Dimension to stop sharding
     * @param query_ptr std::shared_ptr<Tensor<T>> for the query Matrix
     * @param key_ptr std::shared_ptr<Tensor<T>> for the key Matrix
     * @param value_ptr std::shared_ptr<Tensor<T>> for the value Matrix
     */
    AttentionHead(size_t id, size_t start_dim, size_t end_dim, std::shared_ptr<Tensor<T>> query_ptr, 
                  std::shared_ptr<Tensor<T>> key_ptr, std::shared_ptr<Tensor<T>> value_ptr) :
                  c_id(id), c_msc(1, IndexType::RANGE, std::vector<size_t>{start_dim, end_dim}, 0, {}, {}),
                  m_w_query(std::move(query_ptr), c_msc), m_w_key(std::move(key_ptr), c_msc),
                  m_w_value(std::move(value_ptr), c_msc) {
        // Left blank since everything is initialized above
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    /**
     * Get the id for an attention head
     * @returns Returns the id
     */
    size_t id() const {
        return c_id;
    }

    /**
     * Get the MatrixSliceConfig used to create the Attention Head, used for calculating sharding
     * for the QKV after the first matmul step
     * @returns Returns a reference to the MatrixSliceConfig
     */
    const MatrixSliceConfig& slice_config() const {
        return c_msc;
    }

    /**
     * Get the query TensorSlice
     * @returns Returns a reference to the TensorSlice
     */
    const TensorSlice<T>& query() const {
        return m_w_query;
    }

    /**
     * Get the key TensorSlice
     * @returns Returns a reference to the TensorSlice
     */
    const TensorSlice<T>& key() const {
        return m_w_key;
    }

    /**
     * Get the value TensorSlice
     * @returns Returns a reference to the TensorSlice
     */
    const TensorSlice<T>& value() const {
        return m_w_value;
    }
};

/**
 * Multi-Head Attention implementation
 */
template <typename T> 
requires std::is_arithmetic_v<T>
class MultiHeadAttention : public Attention<T> {
/* Private data elements */
private:
    /**
     * Embedding dimension
     */
    size_t c_emb_dim = 0;

    /**
     * Output dimension for the attention head
     */
    size_t c_output_dim = 0;

    /**
     * Dropout percentage (>= 0, < 1)
     */
    float c_dropout = 0.0;

    /**
     * Context length for inputs
     */
    size_t c_context_len = 0;

    /**
     * Number of heads
     */
    size_t c_num_heads = 0;

    /**
     * Head dimension
     */
    size_t c_head_dim = 0;

    /**
     * Query weights
     */
    std::shared_ptr<Tensor<T>> m_w_query;

    /**
     * Key weights
     */
    std::shared_ptr<Tensor<T>> m_w_key;

    /**
     * Value weights
     */
    std::shared_ptr<Tensor<T>> m_w_value;

    /**
     * Vector of TensorSlices, containing query, key, and value for each attention head
     */
    std::vector<AttentionHead<T>> c_heads = std::vector<AttentionHead<T>>();

/* Public functions */
public:
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    /**
     * Constructor for MultiHeadAttention
     * @param emb_dim Embedding dimension
     * @param output_dim Output dimension
     * @param context_len Context length
     * @param dropout Float >= 0, < 1 (set to 0 to disable)
     * @param num_heads Number of heads
     */
    MultiHeadAttention(size_t emb_dim, size_t output_dim, size_t context_len, float dropout, size_t num_heads) :
                       c_emb_dim(emb_dim), c_output_dim(output_dim), c_dropout(dropout), c_context_len(context_len),
                       c_num_heads(num_heads), c_head_dim(c_output_dim / c_num_heads) {
        // Ensure we get a dropout within the acceptable range
        if (dropout < 0 || dropout >= 1) {
            throw std::invalid_argument("MultiHeadAttention.MultiHeadAttention: Dropout must be >= 0, < 1.\n");
        }
        // Ensure the output dimension is divisible by the number of heads
        if (c_output_dim % c_num_heads != 0) {
            throw std::invalid_argument("MultiHeadAttention.MultiHeadAttention: Output dim must be divisible by number of heads.\n");
        }
        // Initialize the query, key, and value matrices
        m_w_query = std::make_shared<Tensor<T>>(std::initializer_list<size_t>{c_emb_dim, c_output_dim});
        m_w_key = std::make_shared<Tensor<T>>(std::initializer_list<size_t>{c_emb_dim, c_output_dim});
        m_w_value = std::make_shared<Tensor<T>>(std::initializer_list<size_t>{c_emb_dim, c_output_dim});
        // Initialize the three matrices with random values
        m_w_query->random(-2, 2);
        m_w_key->random(-2, 2);
        m_w_value->random(-2, 2);
        // Reserve space for the number of attention heads
        c_heads.reserve(c_num_heads);
        // Shard query, key, and value over the number of attention heads
        for (size_t i = 0; i < c_num_heads; ++i) {
            // id = i
            // start_dim = i * c_head_dim
            // end_dim = (i + 1) * c_head_him
            c_heads.emplace_back(i, i * c_head_dim, (i + 1) * c_head_dim, m_w_query, m_w_key, m_w_value);
        }
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    /**
     * Calculate the context for a given input
     * @param input Const ref to an input Matrix
     * @returns Returns a new Matrix with the calculated context
     */
    Tensor<T> forward(const Tensor<T>& input) const override {
        // Get the QKV for the given input
        Tensor<T> queries = Tensor_NS::matmul(input, *m_w_query);
        Tensor<T> keys = Tensor_NS::matmul(input, *m_w_key);
        Tensor<T> values = Tensor_NS::matmul(input, *m_w_value);
        // Convert the QKV to std::shared_ptr without copying
        auto q_ptr = std::make_shared<Tensor<T>>(std::move(queries));
        auto k_ptr = std::make_shared<Tensor<T>>(std::move(keys));
        auto v_ptr = std::make_shared<Tensor<T>>(std::move(values));
        // Iterate over the Attention Heads
        for (const AttentionHead<T>& head : c_heads) {
            // Create TensorSlices for QKV based on the sharding inside the AttentionHead
            const MatrixSliceConfig& msc = head.slice_config();
            TensorSlice<T> q_slice(q_ptr, msc);
            TensorSlice<T> k_slice(k_ptr, msc);
            TensorSlice<T> v_slice(v_ptr, msc);
            // 
        }
        // Return a fake value for now and implement later
        return Tensor<T>({0});
    }
};

}; // namespace Attention_NS

#endif
