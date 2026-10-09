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
 * @version: 2026-10-08
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef CAUSAL_ATTENTION_HPP
#define CAUSAL_ATTENTION_HPP

/* Standard dependencies */
#include <cmath>
#include <cstddef>
#include <stdexcept>

/* Local dependencies */
#include "AbstractTensor.hpp"
#include "Attention.hpp"
#include "Tensor.hpp"
#include "TensorMatmul.hpp"

namespace CausalAttention_NS {

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;

/* Use the Attention interface */
using Attention_NS::Attention;
using Attention_NS::ATTENTION_PREFER_TEMP_DESTINATION_OVER_EXCEPTION;

/* Use Tensor */
using Tensor_NS::Tensor;
using Tensor_NS::CausalMaskType;

/* Use matmul */
using TensorMatmul_NS::matmul;

// NOLINTBEGIN(cppcoreguidelines-special-member-functions)
/**
 * Causal Attention implementation
 */
template <typename T> 
requires std::is_floating_point_v<T>
class CausalAttention final : public Attention<T> {
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
     * Destructor for CausalAttention
     */
    ~CausalAttention() override {
        // Do nothing
    }

    /**
     * Run the forward pass of attention, returning a context Tensor for the inputs
     * @param input Const ref to the input AbstractTensor
     * @returns Returns a new Tensor with the calculated context
     */
    Tensor<T> forward(const AbstractTensor<T>& input) const override {
        // Ensure we have a rank 2 Tensor
        if (input.rank() != 2) {
            throw std::invalid_argument("CausalAttention.forward: Input Tensor must have rank == 2.\n");
        }
        Tensor<T> result({input.extent(0), c_output_dim});
        forward(input, result);
        return result;
    }

    /**
     * Run the forward pass of attention, returning a context Tensor for the inputs
     * @param input Const ref to the input AbstractTensor
     * @param dest Reference to a Tensor to store the output
     * @returns Returns a reference to the dest Tensor with the calculated context
     */
    Tensor<T>& forward(const AbstractTensor<T>& input, Tensor<T>& dest) const override {
        // Ensure we have a rank 2 Tensor
        if (input.rank() != 2 || dest.rank() != 2) {
            throw std::invalid_argument("CausalAttention.forward: Input and destination Tensors must have rank == 2.\n");
        }
        // Ensure destination has the right shape
        if (dest.extent(0) != input.extent(0) || dest.extent(1) != c_output_dim) {
            throw std::invalid_argument("CausalAttention.forward: Incorrect shape for destination Tensor.\n");
        }
        if (!dest.is_unique(input)) {
            if constexpr (ATTENTION_PREFER_TEMP_DESTINATION_OVER_EXCEPTION) {
                Tensor<T> temp_dest = dest.clone();
                forward(input, temp_dest);
                return dest.copy_from(temp_dest);
            }
            throw std::invalid_argument("CausalAttention.forward: Input and destination Tensors must be unique.\n");
        }
        // Get the QKV for the given input
        Tensor<T> queries = matmul(input, m_w_query);
        Tensor<T> keys = matmul(input, m_w_key);
        Tensor<T> values = matmul(input, m_w_value);
        // Transpose keys for calculating attention scores
        keys.transpose();
        Tensor<T> attn_scores = matmul(queries, keys);
        // Handle masking, softmax, etc. if we are using a float dtype
        // Apply masking
        attn_scores.ninf_tri(CausalMaskType::UPPER);
        // Divide each value by the embedding dim, using unsafe_uniform_op to still vectorize
        // without checking for INF of NaN
        Tensor_NS::unsafe_uniform_op(attn_scores, std::sqrtf(static_cast<T>(c_output_dim)), attn_scores, Tensor_NS::SqueezedOpType::DIV);
        // Apply softmax on dim 0
        attn_scores.softmax(0);
        // Apply dropout if enabled
        if (c_dropout > 0) {
            attn_scores.apply_dropout(c_dropout);
        }
        // Return the context
        return matmul(attn_scores, values, dest);
    }
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace CausalAttention_NS

#endif
