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
 * GPTModel.cpp: A GPT-like model implementation
 */

#include "include/GPTModel.hpp"

using AbstractTensor_NS::AbstractTensor;
using GPTModel_NS::GPTModel;
using Tensor_NS::Tensor;

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
GPTModel::GPTModel(size_t vocab_size, size_t num_layers, size_t emb_dim,
                   size_t attn_heads, size_t context_len, float dropout) : c_vocab_size(vocab_size), c_num_layers(num_layers), c_emb_dim(emb_dim),
                                                                           c_attn_heads(attn_heads), c_context_len(context_len), c_dropout(dropout) {
    // Validate vocab_size, num_layers, emb_dim, attn_heads, and context_len
    if (vocab_size == 0 || num_layers == 0 || emb_dim == 0 || attn_heads == 0 || context_len == 0) {
        throw std::invalid_argument("GPTModel.GPTModel: vocab_size, num_layers, emb_dim, attn_heads, and context_len cannot be zero.\n");
    }
    if (emb_dim % attn_heads != 0) {
        throw std::invalid_argument("GPTModel.GPTModel: Embedding dimension must be divisible by number of attention heads.\n");
    }
    if (dropout < 0 || dropout >= 1) {
        throw std::invalid_argument("GPTModel.GPTModel: Dropout must be >= 0, < 1.\n");
    }
    // Starting with the token embedding layer
    m_tok_emb = std::make_unique<Embedding<float>>(c_vocab_size, c_emb_dim);
    // Positional embedding
    m_pos_emb = std::make_unique<AbsolutePositionalEmbedding<float>>(c_context_len, c_emb_dim);
    // Dropout layer
    m_drop = std::make_unique<DropoutLayer<float>>(c_dropout, false);
    // Reserve space for the layers and then begin to instantiate
    m_layers.reserve(c_num_layers);
    for (size_t i = 0; i < c_num_layers; ++i) {
        m_layers.push_back(std::make_unique<TransformerBlock<float>>(c_emb_dim, c_context_len, c_dropout, c_attn_heads, false));
    }
    // Normalization layer
    m_norm = std::make_unique<NormalizationLayer<float>>(false);
    // Final linear output layer
    m_linear_output = std::make_unique<LinearLayer<float>>(c_emb_dim, c_vocab_size, false);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

GPTModel::~GPTModel() {
    // Do nothing
}

Tensor<float> GPTModel::forward(const AbstractTensor<size_t>& input) const {
    // Ensure the input Tensor has rank == 2
    if (input.rank() != 2) {
        throw std::invalid_argument("GPTModel.forward: Input must have rank == 2.\n");
    }
    // Create the output Tensor with shape [batch_size, seq_len, vocab_size]
    Tensor<float> result({input.extent(0), input.extent(1), c_vocab_size});
    forward(input, result);
    return result;
}

Tensor<float>& GPTModel::forward(const AbstractTensor<size_t>& input, Tensor<float>& dest) const {
    // Ensure the input Tensor has rank == 2
    if (input.rank() != 2) {
        throw std::invalid_argument("GPTModel.forward: Input must have rank == 2.\n");
    }
    // Ensure the destination Tensor has rank == 3
    if (dest.rank() != 3) {
        throw std::invalid_argument("GPTModel.forward: Destination Tensor must have rank == 3.\n");
    }
    // Ensure the destination Tensor has the correct shape ([batch_size, seq_len, vocab_size])
    if (dest.extent(0) != input.extent(0) || dest.extent(1) != input.extent(1) || dest.extent(2) != c_vocab_size) {
        throw std::invalid_argument("GPTModel.forward: Destination Tensor must have shape [batch_size, seq_len, vocab_size].\n");
    }
    // We need to convert input from AbstractTensor --> Tensor if it isn't one already, due to slicing requirements
    const Tensor<size_t>* input_ptr = dynamic_cast<const Tensor<size_t>*>(&input);
    // If we are dealign with ListTensorSlice, convert via to_tensor()
    if (input_ptr == nullptr) {
        const Tensor<size_t> input_c = dynamic_cast<const ListTensorSlice<size_t>*>(&input)->to_tensor();
        return forward(input_c, dest);
    }
    // input_ptr must have been cast successfully, so route to the right function
    return forward(*input_ptr, dest);
}

Tensor<float>& GPTModel::forward(const Tensor<size_t>& input, Tensor<float>& dest) const {
    // Ensure the input Tensor has rank == 2
    if (input.rank() != 2) {
        throw std::invalid_argument("GPTModel.forward: Input must have rank == 2.\n");
    }
    // Ensure the destination Tensor has rank == 3
    if (dest.rank() != 3) {
        throw std::invalid_argument("GPTModel.forward: Destination Tensor must have rank == 3.\n");
    }
    // Ensure the destination Tensor has the correct shape ([batch_size, seq_len, vocab_size])
    if (dest.extent(0) != input.extent(0) || dest.extent(1) != input.extent(1) || dest.extent(2) != c_vocab_size) {
        throw std::invalid_argument("GPTModel.forward: Destination Tensor must have shape [batch_size, seq_len, vocab_size].\n");
    }
    // Create a Tensor of shape [seq_len, emb_dim] to use for the layers
    Tensor<float> layer_output({input.extent(1), c_emb_dim});
    // Create a Tensor of shape [seq_len] to lookup embeddings with
    Tensor<size_t> input_batch({input.extent(1)});
    // Create a Tensor of shape [seq_len] to record the positions of the tokens
    Tensor<size_t> input_pos({input.extent(1)});
    for (size_t i = 0; i < input.extent(1); ++i) {
        input_pos.at(i) = i;
    }
    // Since we are dealing with multiple batches, we will need to slice across each batch and run the passes sequentially
    for (size_t i = 0; i < input.extent(0); ++i) {
        // Use the Tensor util to copy the input batch into our 1-D input_batch
        TensorUtils_NS::copy_2d_tensor_filtered_dim(input, 0, i, input_batch);
        // Get the embeddings for the input batch
        ListTensorSlice<float> tok_emb = m_tok_emb->get_embeddings(input_batch);
        ListTensorSlice<float> pos_emb = m_pos_emb->get_embeddings(input_pos);
        // Copy tok_emb into layer_output
        layer_output.copy_from(tok_emb);
        // Materialize pos_emb and add it to the tok_emb (layer_output)
        // layer_out and pos_emb.to_tensor() are both guaranteed to be contiguous with stride == 1 in the final dim
        // so we wil greatly speed up access and route via Tensor::uniform_op
        layer_output += pos_emb.to_tensor();
        // Apply dropout
        m_drop->forward(layer_output, layer_output);
        // Run the foward pass for all of the TransformerBlocks
        for (const auto& layer : m_layers) {
            layer->forward(layer_output, layer_output);
        }
        // Apply normalization
        m_norm->forward(layer_output, layer_output);
        // Create a slice for the final output
        Tensor<float> final_output = dest.slice({1, 2}, {}, {{0, i}});
        // Project the output into [seq_len, vocab_size]
        m_linear_output->forward(layer_output, final_output);
    }
    return dest;
}

std::string GPTModel::info() const {
    return std::format("{} (dtype={}): Vocab size: {}, num layers: {}, embedding dim: {}, num attn heads: {}, context len: {}, dropout: {}.",
        _model_name, _model_dtype, c_vocab_size, c_num_layers, c_emb_dim, c_attn_heads, c_context_len, c_dropout);
}
