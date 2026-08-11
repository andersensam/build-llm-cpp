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
 * @version: 2026-08-11
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
#include <stdexcept>
#include <string>

/* Local dependencies */
#include "Log.hpp"
#include "Tensor.hpp"
#include "TensorSlice.hpp"

namespace Attention_NS {

using Tensor_NS::Tensor;
using Tensor_NS::Matrix;
using Tensor_NS::CausalMaskType;

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
     * Run the forward pass of attention, returning a context matrix for the inputs
     * @param input Const ref to the input Matrix
     * @returns Returns a new Matrix with the calculated context
     */
    virtual Matrix<T> forward(const Matrix<T>& input) const = 0;
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
    Matrix<T> m_w_query;

    /**
     * Key weights
     */
    Matrix<T> m_w_key;

    /**
     * Value weights
     */
    Matrix<T> m_w_value;

/* Public functions */
public:
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    /**
     * Constructor for CausalAttention, taking in the embedding dim, output dim, and whether or not to use dropout
     * @param emb_dim Embedding (input) dimension
     * @param output_dim Output dimension
     * @param use_dropout True to enable dropout, false to disable
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
     * @param input Const ref to an input Matrix
     * @returns Returns a new Matrix with the calculated context
     */
    Matrix<T> forward(const Matrix<T>& input) const override {
        // Get the QKV for the given input
        Matrix<T> queries = input.matmul(m_w_query);
        Matrix<T> keys = input.matmul(m_w_key);
        Matrix<T> values = input.matmul(m_w_value);
        // Transpose keys for calculating attention scores
        keys.transpose();
        Matrix<T> attn_scores = queries.matmul(keys);
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
        return attn_scores.matmul(values);
    }

};

}; // namespace Attention_NS

#endif
