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
 * @version: 2026-10-07
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef ATTENTION_HPP
#define ATTENTION_HPP

/* Standard dependencies */
#include <cmath>
#include <cstddef>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

/* Local dependencies */
#include "Log.hpp"
#include "Tensor.hpp"
#include "TensorMatmul.hpp"
#include "TensorSlice.hpp"

namespace Attention_NS {

using Tensor_NS::Tensor;
using Tensor_NS::CausalMaskType;
using TensorMatmul_NS::matmul;

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
    CausalAttention(size_t emb_dim, size_t output_dim, float dropout) : c_emb_dim(emb_dim), c_output_dim(output_dim), c_dropout(dropout),
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
        Tensor<T> queries = matmul(input, m_w_query);
        Tensor<T> keys = matmul(input, m_w_key);
        Tensor<T> values = matmul(input, m_w_value);
        // Transpose keys for calculating attention scores
        keys.transpose();
        Tensor<T> attn_scores = matmul(queries, keys);
        // Handle masking, softmax, etc. if we are using a float dtype
        if constexpr (std::is_floating_point_v<T>) {
            // Apply masking
            attn_scores.ninf_tri(CausalMaskType::UPPER);
            // Divide each value by the embedding dim, using unsafe_uniform_op to still vectorize
            // without checking for INF of NaN
            Tensor_NS::unsafe_uniform_op(attn_scores, std::sqrtf(static_cast<T>(c_output_dim)), attn_scores, Tensor_NS::SqueezedOpType::DIV);
            // Apply softmax on dim 0
            attn_scores.softmax(0);
        }
        else {
            // Apply masking
            attn_scores.apply_mask(CausalMaskType::UPPER);
            // TODO: Implement non-floating point activation function
        }
        // Apply dropout if enabled
        if (c_dropout > 0) {
            attn_scores.apply_dropout(c_dropout);
        }
        // Return the context
        return matmul(attn_scores, values);
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
    Tensor<T> m_w_query;

    /**
     * Key weights
     */
    Tensor<T> m_w_key;

    /**
     * Value weights
     */
    Tensor<T> m_w_value;

    /**
     * Start and stop dims for each attention head
     */
    std::vector<std::pair<size_t, size_t>> c_heads = std::vector<std::pair<size_t, size_t>>();

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
                       c_num_heads(num_heads), c_head_dim(c_output_dim / c_num_heads), m_w_query({c_emb_dim, c_output_dim}),
                       m_w_key({c_emb_dim, c_output_dim}), m_w_value({c_emb_dim, c_output_dim}) {
        // Ensure we get a dropout within the acceptable range
        if (dropout < 0 || dropout >= 1) {
            throw std::invalid_argument("MultiHeadAttention.MultiHeadAttention: Dropout must be >= 0, < 1.\n");
        }
        // Ensure the output dimension is divisible by the number of heads
        if (c_output_dim % c_num_heads != 0) {
            throw std::invalid_argument("MultiHeadAttention.MultiHeadAttention: Output dim must be divisible by number of heads.\n");
        }
        // Initialize the three matrices with random values
        m_w_query.random(-2, 2);
        m_w_key.random(-2, 2);
        m_w_value.random(-2, 2);
        // Reserve space for the number of attention heads
        c_heads.reserve(c_num_heads);
        // Shard query, key, and value over the number of attention heads
        for (size_t i = 0; i < c_num_heads; ++i) {
            // start_dim = i * c_head_dim
            // end_dim = (i + 1) * c_head_him
            c_heads.emplace_back(i * c_head_dim, (i + 1) * c_head_dim);
        }
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    /**
     * Calculate the context for a given input
     * @param input Const ref to an input Matrix
     * @returns Returns a new Matrix with the calculated context
     */
    Tensor<T> forward(const Tensor<T>& input) const override {
        // Get the QKV for the given input
        Tensor<T> queries = matmul(input, m_w_query);
        Tensor<T> keys = matmul(input, m_w_key);
        Tensor<T> values = matmul(input, m_w_value);
        // Create an output Tensor
        Tensor<T> output({input.extent(0), input.extent(0)});
        // Create another Tensor for attn_weights @ values
        Tensor<T> context_vec({input.extent(0), c_head_dim});
        // Create a Tensor for the final, concatenated context vector
        Tensor<T> result({input.extent(0), c_output_dim});
        // Iterate over the Attention Heads, unpacking their start and stops
        for (size_t head_id = 0 ; head_id < c_heads.size(); ++head_id) {
            // Get the start and stop for the head
            const auto [start, end] = c_heads[head_id];
            // Create Tensor slices for QKV based on the sharding inside the AttentionHead
            Tensor<T> q_slice = queries.slice({0, 1}, {{0, 0}, {start, end}}, {});
            Tensor<T> k_slice = keys.slice({0, 1}, {{0, 0}, {start, end}}, {});
            Tensor<T> v_slice = values.slice({0, 1}, {{0, 0}, {start, end}}, {});
            // Calculate the attention scores, Q @ K.transpose
            matmul(q_slice, 0, 1, k_slice, 1, 0, output);
            // Handle masking, softmax, etc. if we are using a float dtype
            if constexpr (std::is_floating_point_v<T>) {
                // Apply masking
                output.ninf_tri(CausalMaskType::UPPER);
                // Divide each value by the embedding dim, using unsafe_uniform_op to still vectorize
                // without checking for INF of NaN
                Tensor_NS::unsafe_uniform_op(output, std::sqrtf(static_cast<T>(c_head_dim)), output, Tensor_NS::SqueezedOpType::DIV);
                // Apply softmax on dim 0
                output.softmax(0);
            }
            else {
                // Apply masking
                output.apply_mask(CausalMaskType::UPPER);
                // TODO: Implement non-floating point activation function
            }
            // Apply dropout
            if (c_dropout > 0) {
                output.apply_dropout(c_dropout);
            }
            // Calculate the context vector via attn_weights (output) @ values (v_slice)
            matmul(output, v_slice, context_vec);
            // Copy the values into the final result Tensor
            for (size_t i = 0; i < input.extent(0); ++i) {
                // Copy the second dim from the slice
                for (size_t j = head_id * c_head_dim; j < (head_id + 1) * c_head_dim; ++j) {
                    result.at({i, j}) = context_vec.at({i, j - (head_id * c_head_dim)});
                }
            }
        }
        // Return the concatenated context vector
        return result;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
};

}; // namespace Attention_NS

#endif
