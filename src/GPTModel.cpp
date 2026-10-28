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
 * @version: 2026-10-26
 *
 * GPTModel.cpp: A GPT-like model implementation
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/AbsolutePositionalEmbedding.hpp"
#include "include/AbstractTensor.hpp"
#include "include/DropoutLayer.hpp"
#include "include/Embedding.hpp"
#include "include/GPTModel.hpp"
#include "include/LinearLayer.hpp"
#include "include/ListTensorSlice.hpp"
#include "include/Log.hpp"
#include "include/NormalizationLayer.hpp"
#include "include/Numerics.hpp"
#include "include/Tensor.hpp"
#include "include/TensorUtils.hpp"
#include "include/TransformerBlock.hpp"

/* Use AbsolutePositionalEmbedding */
using AbsolutePositionalEmbedding_NS::AbsolutePositionalEmbedding;

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;

/* Use Dropout */
using DropoutLayer_NS::DropoutLayer;

/* Use Embedding */
using Embedding_NS::Embedding;

/* Use GPTModel */
using GPTModel_NS::GPTModel;

/* Use LinearLayer */
using LinearLayer_NS::LinearLayer;

/* Use ListTensorSlice */
using ListTensorSlice_NS::ListTensorSlice;

/* Use Logging */
using Log_NS::log_message;
using Log_NS::Log_Priority;

/* Use Normalization */
using NormalizationLayer_NS::NormalizationLayer;

/* Use Tensor */
using Tensor_NS::Tensor;

/* Use TransformerBlock */
using TransformerBlock_NS::TransformerBlock;

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

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
GPTModel::GPTModel(const std::string& path) {
    // Validate that we can read in the file
    std::fstream model(path, std::ios::binary | std::ios::in);
    if (!model) {
        throw std::invalid_argument("GPTModel.GPTModel: Invalid model file provided.\n");
    }
    // Create buffers for trivial types
    std::array<char, Numerics_NS::Accumulator<uint32_t>::_s_size> u_buf{0};
    uint32_t u32 = 0;
    std::array<char, Numerics_NS::Accumulator<size_t>::_s_size> s_buf{0};
    // Verify the start magic number
    model.read(u_buf.data(), u_buf.size());
    if (model.fail()) {
        model.close();
        throw std::runtime_error("GPTModel.GPTModel: Unable to read magic number.\n");
    }
    u32 = Numerics_NS::from_bytes<uint32_t>(u_buf.data());
    if (u32 != GPT_MODEL_START_MAGIC_NUMBER) {
        model.close();
        throw std::runtime_error("GPTModel.GPTModel: Invalid start magic number.\n");
    }
    // Read the expected size of the model
    model.read(s_buf.data(), s_buf.size());
    if (model.fail()) {
        model.close();
        throw std::runtime_error("GPTModel.GPTModel: Unable to read model size.\n");
    }
    size_t total_size = Numerics_NS::from_bytes<size_t>(s_buf.data());
    // Allocate a new char* buffer and read in the rest of the model definition, ensuring that
    // we strip out the start magic number and the size
    if (total_size > u_buf.size() + s_buf.size()) {
        total_size -= u_buf.size() + s_buf.size();
    }
    else {
        model.close();
        throw std::runtime_error("GPTModel.GPTModel: Corrupted size in model file.\n");
    }
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
    auto s_ptr = std::make_unique<char[]>(total_size);
    char* s = s_ptr.get();
    // Read in the remainder of the model and switch to using index
    if (total_size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        model.close();
        throw std::runtime_error("GPTModel.GPTModel: Unable to safely cast size_t to streamsize without overflow.\n");
    }
    std::streamsize sz = static_cast<std::streamsize>(total_size);
    model.read(s, sz);
    if (model.fail()) {
        model.close();
        throw std::runtime_error("GPTModel.GPTModel: Unable to read remainder of model into buffer.\n");
    }
    // Close the model since we are done with it at this point
    model.close();
    size_t index = 0;
    // Validate the model dtype
    u32 = Numerics_NS::from_bytes<uint32_t>(s, index);
    if (u32 != Numerics_NS::Accumulator<float>::_base_type_mn) {
        throw std::runtime_error("GPTModel.GPTModel: Model dtype does not match serialized dtype.\n");
    }
    // Get vocab_size
    if (index >= total_size) { throw std::runtime_error("GPTModel.GPTModel: Invalid index or total size for deserializing.\n"); }
    c_vocab_size = Numerics_NS::from_bytes<size_t>(s, index);
    // Get num_layers
    if (index >= total_size) { throw std::runtime_error("GPTModel.GPTModel: Invalid index or total size for deserializing.\n"); }
    c_num_layers = Numerics_NS::from_bytes<size_t>(s, index);
    // Get emb_dim
    if (index >= total_size) { throw std::runtime_error("GPTModel.GPTModel: Invalid index or total size for deserializing.\n"); }
    c_emb_dim = Numerics_NS::from_bytes<size_t>(s, index);
    // Get attn_heads
    if (index >= total_size) { throw std::runtime_error("GPTModel.GPTModel: Invalid index or total size for deserializing.\n"); }
    c_attn_heads = Numerics_NS::from_bytes<size_t>(s, index);
    // Get context_len
    if (index >= total_size) { throw std::runtime_error("GPTModel.GPTModel: Invalid index or total size for deserializing.\n"); }
    c_context_len = Numerics_NS::from_bytes<size_t>(s, index);
    // Get dropout
    if (index >= total_size) { throw std::runtime_error("GPTModel.GPTModel: Invalid index or total size for deserializing.\n"); }
    c_dropout = Numerics_NS::int_to_float<float>(Numerics_NS::from_bytes<typename Numerics_NS::Accumulator<float>::serialize_as>(s, index));
    // Validate vocab_size, num_layers, emb_dim, attn_heads, and context_len
    if (c_vocab_size == 0 || c_num_layers == 0 || c_emb_dim == 0 || c_attn_heads == 0 || c_context_len == 0) {
        throw std::invalid_argument("GPTModel.GPTModel: vocab_size, num_layers, emb_dim, attn_heads, and context_len cannot be zero.\n");
    }
    if (c_emb_dim % c_attn_heads != 0) {
        throw std::invalid_argument("GPTModel.GPTModel: Embedding dimension must be divisible by number of attention heads.\n");
    }
    if (c_dropout < 0 || c_dropout >= 1 || !std::isfinite(c_dropout)) {
        throw std::invalid_argument("GPTModel.GPTModel: Dropout must be >= 0, < 1.\n");
    }
    // Instantiate m_tok_emb
    if (index >= total_size) { throw std::runtime_error("GPTModel.GPTModel: Invalid index or total size for deserializing.\n"); }
    m_tok_emb = std::make_unique<Embedding<float>>(s + index);
    index += m_tok_emb->s_size();
    // Instantiate m_pos_emb
    if (index >= total_size) { throw std::runtime_error("GPTModel.GPTModel: Invalid index or total size for deserializing.\n"); }
    m_pos_emb = std::make_unique<AbsolutePositionalEmbedding<float>>(s + index);
    index += m_pos_emb->s_size();
    // Instantiate m_drop
    if (index >= total_size) { throw std::runtime_error("GPTModel.GPTModel: Invalid index or total size for deserializing.\n"); }
    m_drop = std::make_unique<DropoutLayer<float>>(s + index);
    index += m_drop->s_size();
    // Reserve space and load the layers
    m_layers.reserve(c_num_layers);
    for (size_t i = 0; i < c_num_layers; ++i) {
        if (index >= total_size) { throw std::runtime_error("GPTModel.GPTModel: Invalid index or total size for deserializing.\n"); }
        m_layers.push_back(std::make_unique<TransformerBlock<float>>(s + index));
        index += m_layers.back()->s_size();
    }
    // Instantiate m_norm
    if (index >= total_size) { throw std::runtime_error("GPTModel.GPTModel: Invalid index or total size for deserializing.\n"); }
    m_norm = std::make_unique<NormalizationLayer<float>>(s + index);
    index += m_norm->s_size();
    // Instantiate m_linear_output
    if (index >= total_size) { throw std::runtime_error("GPTModel.GPTModel: Invalid index or total size for deserializing.\n"); }
    m_linear_output = std::make_unique<LinearLayer<float>>(s + index);
    index += m_linear_output->s_size();
    // Ensure the end magic number is valid
    if (index >= total_size) { throw std::runtime_error("GPTModel.GPTModel: Invalid index or total size for deserializing.\n"); }
    u32 = Numerics_NS::from_bytes<uint32_t>(s, index);
    if (u32 != GPT_MODEL_END_MAGIC_NUMBER) {
        throw std::runtime_error("GPTModel.GPTModel: Invalid end magic number.\n");   
    }
    if (index != total_size) {
        throw std::runtime_error("GPTModel.GPTModel: Reached end of object but total_size indicates there is more data to be read.\n");
    }
}
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

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

Tensor<float> GPTModel::train(const AbstractTensor<size_t>& input, const AbstractTensor<size_t>& target) {
    // Ensure the input Tensor has rank == 2
    if (input.rank() != 2 || target.rank() != 2) {
        throw std::invalid_argument("GPTModel.train: Input and target must have rank == 2.\n");
    }
    // Ensure input and target have the same shape
    if (input.shape() != target.shape()) {
        throw std::invalid_argument("GPTModel.train: Input and target must have the same shape.\n");
    }
    // Convert input and target to Tensors
    const Tensor<size_t>* input_ptr = dynamic_cast<const Tensor<size_t>*>(&input);
    const Tensor<size_t>* target_ptr = dynamic_cast<const Tensor<size_t>*>(&target);
    // Handle failure converting input or target (is a ListTensorSlice)
    if (input_ptr == nullptr && target_ptr == nullptr) {
        return train(dynamic_cast<const ListTensorSlice<size_t>*>(&input)->to_tensor(), dynamic_cast<const ListTensorSlice<size_t>*>(&target)->to_tensor());
    }
    if (input_ptr == nullptr) {
        return train(dynamic_cast<const ListTensorSlice<size_t>*>(&input)->to_tensor(), target);
    }
    if (target_ptr == nullptr) {
        return train(input, dynamic_cast<const ListTensorSlice<size_t>*>(&target)->to_tensor());
    }
    // Convert to refs
    const Tensor<size_t>& conv_input = *input_ptr;
    const Tensor<size_t>& conv_target = *target_ptr;
    // Create a result Tensor with the loss per batch
    Tensor<float> result({input.extent(0)});
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
        TensorUtils_NS::copy_2d_tensor_filtered_dim(conv_input, 0, i, input_batch);
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
        m_drop->forward_train(layer_output, layer_output);
        // Run the foward pass for all of the TransformerBlocks
        for (const auto& layer : m_layers) {
            layer->forward_train(layer_output, layer_output);
        }
        // Apply normalization
        m_norm->forward_train(layer_output, layer_output);
        // Project the output into [seq_len, vocab_size]
        Tensor<float> batch_output = m_linear_output->forward_train(layer_output);
        // Run softmax on the batch_output Tensor
        batch_output.softmax(0);
        // Create a Tensor of [seq_len, vocab_size] to compare output against target
        Tensor<float> target_prob(batch_output.shape());
        // Iterate over the correct tokens in the target batch and place a 1 in that index
        for (size_t j = 0; j < conv_target.extent(1); ++j) {
            size_t correct_token_id = conv_target.at({i, j});
            target_prob.at({j, correct_token_id}) = 1;
        }
        // Get the loss for the batch, scoped to force a quick deallocation
        {
            Tensor<float> output_log = batch_output.clone();
            //std::cerr << "max: " << output_log.max() << ". min: " << output_log.min() << '\n';
            output_log.apply(std::log);
            //std::cerr << "max: " << output_log.max() << ". min: " << output_log.min() << '\n';
            //output_log *= target_prob;
            // Use the utility function to set tiny values to zero
            //std::cerr << "max: " << output_log.max() << ". min: " << output_log.min() << '\n';
            Tensor_NS::unsafe_uniform_op(output_log, target_prob, output_log, Tensor_NS::SqueezedOpType::MUL);
            //std::cerr << "max: " << output_log.max() << ". min: " << output_log.min() << '\n';
            //std::cerr << "max: " << output_log.max() << ". min: " << output_log.min() << '\n';
            result.at({i}) = (-1.0f) * output_log.sum();
        }
        // Subtract the correct from the predicted
        batch_output -= target_prob;
        // Begin backprop here...
    }
    return result;
}

std::string GPTModel::info() const {
    return std::format("{} (dtype={}): Vocab size: {}, num layers: {}, embedding dim: {}, num attn heads: {}, context len: {}, dropout: {}.",
        _model_name, _model_dtype, c_vocab_size, c_num_layers, c_emb_dim, c_attn_heads, c_context_len, c_dropout);
}

size_t GPTModel::context_len() const {
    return c_context_len;
}

size_t GPTModel::s_size() const {
    // Start with the magic numbers and dtype
    size_t result = 3 * Numerics_NS::Accumulator<uint32_t>::_s_size;
    // Add the total size, vocab_size, num_layers, emb_dim, attn_heads, and context_len
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    result += 6 * Numerics_NS::Accumulator<size_t>::_s_size;
    // Add dropout
    result += Numerics_NS::Accumulator<float>::_s_size;
    // Add the token embedding layer
    result += m_tok_emb->s_size();
    // Add the positional embedding layer
    result += m_pos_emb->s_size();
    // Add the dropout layer
    result += m_drop->s_size();
    // Add the TransformerBlocks
    for (const auto& layer : m_layers) {
        result += layer->s_size();
    }
    // Add the normalization layer
    result += m_norm->s_size();
    // Add the final output layer
    result += m_linear_output->s_size();
    return result;
}

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
std::vector<char> GPTModel::serialize() const {
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
        throw std::runtime_error("GPTModel.serialize: Error serializing -- invalid index or s_size.\n");
    }
    for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(GPT_MODEL_START_MAGIC_NUMBER))) {
        data[index] = b;
        ++index;
    }
    // Write the serialized size
    if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
        throw std::runtime_error("GPTModel.serialize: Error serializing -- invalid index or s_size.\n");
    }
    for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(target_size))) {
        data[index] = b;
        ++index;
    }
    // Write the dtype
    if (index + Numerics_NS::Accumulator<uint32_t>::_s_size >= target_size) {
        throw std::runtime_error("GPTModel.serialize: Error serializing -- invalid index or s_size.\n");
    }
    for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(Numerics_NS::Accumulator<float>::_base_type_mn))) {
        data[index] = b;
        ++index;
    }
    // Write the vocab_size
    if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
        throw std::runtime_error("GPTModel.serialize: Error serializing -- invalid index or s_size.\n");
    }
    for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(c_vocab_size))) {
        data[index] = b;
        ++index;
    }
    // Write the num_layers
    if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
        throw std::runtime_error("GPTModel.serialize: Error serializing -- invalid index or s_size.\n");
    }
    for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(c_num_layers))) {
        data[index] = b;
        ++index;
    }
    // Write the emb_dim
    if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
        throw std::runtime_error("GPTModel.serialize: Error serializing -- invalid index or s_size.\n");
    }
    for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(c_emb_dim))) {
        data[index] = b;
        ++index;
    }
    // Write the attn_heads
    if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
        throw std::runtime_error("GPTModel.serialize: Error serializing -- invalid index or s_size.\n");
    }
    for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(c_attn_heads))) {
        data[index] = b;
        ++index;
    }
    // Write the context_len
    if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
        throw std::runtime_error("GPTModel.serialize: Error serializing -- invalid index or s_size.\n");
    }
    for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(c_context_len))) {
        data[index] = b;
        ++index;
    }
    // Write the dropout rate
    if (index + Numerics_NS::Accumulator<float>::_s_size >= target_size) {
        throw std::runtime_error("GPTModel.serialize: Error serializing -- invalid index or s_size.\n");
    }
    for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(Numerics_NS::float_to_int(c_dropout)))) {
        data[index] = b;
        ++index;
    }
    // Write the token embedding layer
    if (index + m_tok_emb->s_size() >= target_size) {
        throw std::runtime_error("GPTModel.serialize: Error serializing -- invalid index or s_size.\n");
    }
    std::vector<char> sl = m_tok_emb->serialize();
    std::memcpy(data + index, sl.data(), sl.size());
    index += sl.size();
    sl.clear();
    // Write the positional embedding layer
    if (index + m_pos_emb->s_size() >= target_size) {
        throw std::runtime_error("GPTModel.serialize: Error serializing -- invalid index or s_size.\n");
    }
    sl = m_pos_emb->serialize();
    std::memcpy(data + index, sl.data(), sl.size());
    index += sl.size();
    sl.clear();
    // Write the dropout layer
    if (index + m_drop->s_size() >= target_size) {
        throw std::runtime_error("GPTModel.serialize: Error serializing -- invalid index or s_size.\n");
    }
    sl = m_drop->serialize();
    std::memcpy(data + index, sl.data(), sl.size());
    index += sl.size();
    sl.clear();
    // Write the TransformerBlocks
    for (const auto& layer : m_layers) {
        if (index + layer->s_size() >= target_size) {
            throw std::runtime_error("GPTModel.serialize: Error serializing -- invalid index or s_size.\n");
        }
        sl = layer->serialize();
        std::memcpy(data + index, sl.data(), sl.size());
        index += sl.size();
        sl.clear();
    }
    // Write the normalization layer
    if (index + m_norm->s_size() >= target_size) {
        throw std::runtime_error("GPTModel.serialize: Error serializing -- invalid index or s_size.\n");
    }
    sl = m_norm->serialize();
    std::memcpy(data + index, sl.data(), sl.size());
    index += sl.size();
    sl.clear();
    // Write the final linear output layer
    if (index + m_linear_output->s_size() >= target_size) {
        throw std::runtime_error("GPTModel.serialize: Error serializing -- invalid index or s_size.\n");
    }
    sl = m_linear_output->serialize();
    std::memcpy(data + index, sl.data(), sl.size());
    index += sl.size();
    sl.clear();
    // Write the end magic number
    if (index + Numerics_NS::Accumulator<uint32_t>::_s_size > target_size) {
        throw std::runtime_error("GPTModel.serialize: Error serializing -- invalid index or s_size.\n");
    }
    for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(GPT_MODEL_END_MAGIC_NUMBER))) {
        data[index] = b;
        ++index;
    }
    return result;
}
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

bool GPTModel::save_to_file(const std::string& path) const {
    // Open the file for writing
    std::ofstream model(path, std::ios::binary);
    if (!model) {
        log_message(Log_Priority::ERROR, "GPTModel.save_to_file", "Unable to open model file for writing.");
        return false;
    }
    // Serialize the model
    std::vector<char> s = serialize();
    if (s.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        log_message(Log_Priority::ERROR, "GPTModel.save_to_file", "Unable to safely cast size_t to streamsize without overflow.");
        model.close();
        return false;
    }
    std::streamsize sz = static_cast<std::streamsize>(s.size());
    model.write(s.data(), sz);
    if (model.fail()) {
        log_message(Log_Priority::ERROR, "GPTModel.save_to_file", "Error while writing model file.");
        model.close();
        return false;
    }
    // Explicitly close the file
    model.close();
    return true;
}
