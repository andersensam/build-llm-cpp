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
 * Notes: TransformerBlock.hpp
 *
 */

#ifndef TRANSFORMER_BLOCK_HPP
#define TRANSFORMER_BLOCK_HPP

/* Standard dependencies */
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

/* Local dependencies */
#include "AbstractTensor.hpp"
#include "DropoutLayer.hpp"
#include "FeedForward.hpp"
#include "TransformerBlock.hpp"
#include "Layer.hpp"
#include "MultiHeadAttention.hpp"
#include "NormalizationLayer.hpp"
#include "Tensor.hpp"

namespace TransformerBlock_NS {

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;

/* Use Dropout */
using DropoutLayer_NS::DropoutLayer;

/* Use the FeedForward network */
using FeedForward_NS::FeedForward;

/* Use the Layer interface */
using Layer_NS::Layer;
using Layer_NS::LAYER_PREFER_TEMP_DESTINATION_OVER_EXCEPTION;

/* Use MultiHeadAttention */
using MultiHeadAttention_NS::MultiHeadAttention;

/* Use Normalization */
using NormalizationLayer_NS::NormalizationLayer;

/* Use Tensor */
using Tensor_NS::Tensor;

/**
 * The Transformer Block backbone of the LLM
 */
template <typename T>
requires std::is_floating_point_v<T>
class TransformerBlock final : public Layer<T> {
/* Private data elements */
private:
    /**
     * Embedding dimension
     */
    size_t c_emb_dim = 0;

    /**
     * Context length
     */
    size_t c_context_len = 0;

    /**
     * Dropout
     */
    float c_dropout = 0;

    /**
     * Attention Heads
     */
    size_t c_attn_heads = 0;

    /**
     * Whether a shortcut connection comes after this layer
     */
    bool c_shortcut = false;

    /**
     * Vector of layers, including normalization, attention, and feed-forward
     */
    std::vector<std::unique_ptr<Layer<T>>> m_layers;
/* Public functions */
public:
    // NOLINTBEGIN(bugprone-easily-swappable-parameters, cppcoreguidelines-avoid-magic-numbers)
    /**
     * Default constructor for TransformerBlock, accepting in_dim and out_dim
     * @param in_dim Input dimension
     * @param out_dim Output dimension
     */
    TransformerBlock(size_t emb_dim, size_t context_len, float dropout,
                     size_t attn_heads, bool shortcut) : c_emb_dim(emb_dim), c_context_len(context_len),
                                                         c_dropout(dropout), c_attn_heads(attn_heads),
                                                         c_shortcut(shortcut) {
        // Validate the params
        if (emb_dim == 0 || context_len == 0 || attn_heads == 0) {
            throw std::invalid_argument("TransformerBlock.TransformerBlock: emb_dim, context_len, and attn_head cannot be zero.\n");
        }
        // Ensure the output dimension is divisible by the number of heads
        if (c_emb_dim % c_attn_heads != 0) {
            throw std::invalid_argument("TransformerBlock.TransformerBlock: Output dim must be divisible by number of heads.\n");
        }
        // Ensure we get a dropout within the acceptable range
        if (dropout < 0 || dropout >= 1) {
            throw std::invalid_argument("TransformerBlock.TransformerBlock: Dropout must be >= 0, < 1.\n");
        }
        // Reserve 6 layers per Transformer Block
        m_layers.reserve(6);
        // Start with a normalization layer
        m_layers.push_back(std::make_unique<NormalizationLayer<T>>(false));
        // Add MultiHeadAttention
        m_layers.push_back(std::make_unique<MultiHeadAttention<T>>(c_emb_dim, c_emb_dim, c_context_len,
                                                                c_dropout, c_attn_heads, false));
        // Add the first dropout with a shortcut after it
        m_layers.push_back(std::make_unique<DropoutLayer<T>>(c_dropout, true));
        // Create another normalization layer
        m_layers.push_back(std::make_unique<NormalizationLayer<T>>(false));
        // Create a feed forward network (FFN)
        m_layers.push_back(std::make_unique<FeedForward<T>>(c_emb_dim, 4, false));
        // Finish the Transformer Block with a final dropout
        m_layers.push_back(std::make_unique<DropoutLayer<T>>(c_dropout, true));
    }
    // NOLINTEND(bugprone-easily-swappable-parameters, cppcoreguidelines-avoid-magic-numbers)

    /**
     * Forward function, performing the forward pass on the layer and returning a reference
     * to itself with the computation complete
     * @param input Const ref to an AbstractTensor serving as the input
     * @returns Returns a new Tensor with the result
     */
    Tensor<T> forward(const AbstractTensor<T>& input) const override {
        // Ensure we have a rank 2 Tensor
        if (input.rank() != 2) {
            throw std::invalid_argument("TransformerBlock.forward: Input Tensor must have rank == 2.\n");
        }
        // Ensure that the 2nd dim of input == emb_dim
        if (input.extent(1) != c_emb_dim) {
            throw std::invalid_argument("TransformerBlock.forward: Input Tensor's second dim must be emb_dim.\n");
        }
        Tensor<T> result(input.shape());
        forward(input, result);
        return result;
    }

    /**
     * Forward function, performing the forward pass on the layer and writing the result
     * to a defined destination
     * @param input Const ref to an AbstractTensor serving as the input
     * @returns Returns a reference to the destination Tensor provided
     */
    Tensor<T>& forward(const AbstractTensor<T>& input, Tensor<T>& dest) const override {
        // Ensure we have a rank 2 Tensor
        if (input.rank() != 2 || dest.rank() != 2) {
            throw std::invalid_argument("TransformerBlock.forward: Input and dest Tensors must have rank == 2.\n");
        }
        // Ensure that the 2nd dim of input == emb_dim
        if (input.extent(1) != c_emb_dim) {
            throw std::invalid_argument("TransformerBlock.forward: Input Tensor's second dim must be emb_dim.\n");
        }
        // Ensure that the 2nd dim of input == emb_dim
        if (input.shape() != dest.shape()) {
            throw std::invalid_argument("TransformerBlock.forward: Input and dest must have the same shape.\n");
        }
        if (!dest.is_unique(input)) {
            if constexpr (LAYER_PREFER_TEMP_DESTINATION_OVER_EXCEPTION) {
                Tensor<T> temp_dest = dest.clone();
                forward(input, temp_dest);
                return dest.copy_from(temp_dest);
            }
            throw std::invalid_argument("TransformerBlock.forward: Input and destination Tensors must be unique.\n");
        }
        // Iterate over the layers
        const AbstractTensor<T>& layer_input = input;
        // Make a shallow copy of dest before we overwrite it
        Tensor<T> layer_output = dest.clone();
        // Shortcut input
        Tensor<T> shortcut_input(input.shape());
        shortcut_input.copy_from(input);
        // Enable polymorphic behavior
        for (const auto& layer : m_layers) {
            layer->forward(layer_input, layer_output);
            if (layer->has_shortcut()) {
                layer_output += shortcut_input;
                shortcut_input.copy_from(layer_output);
            }
        }
        dest.copy_from(layer_output);
        return dest;
    }

    /**
     * Check to see if a layer has a shortcut
     */
    bool has_shortcut() const override {
        return c_shortcut;
    }
};

}; // namespace TransformerBlock_NS

#endif
