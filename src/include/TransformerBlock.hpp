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
 * Notes: TransformerBlock.hpp
 *
 */

#ifndef TRANSFORMER_BLOCK_HPP
#define TRANSFORMER_BLOCK_HPP

/* Standard dependencies */
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

/* Local dependencies */
#include "AbstractTensor.hpp"
#include "DropoutLayer.hpp"
#include "FeedForward.hpp"
#include "Layer.hpp"
#include "MultiHeadAttention.hpp"
#include "Numerics.hpp"
#include "NormalizationLayer.hpp"
#include "Tensor.hpp"

namespace TransformerBlock_NS {

/* Magic numbers for serialization */
inline constexpr uint32_t TRANSFORMER_BLOCK_START_MAGIC_NUMBER = 0x88800080;
inline constexpr uint32_t TRANSFORMER_BLOCK_END_MAGIC_NUMBER = 0x88800081;

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

// NOLINTBEGIN(cppcoreguidelines-special-member-functions)
/**
 * The Transformer Block backbone of the LLM
 */
template <typename T>
requires std::is_floating_point_v<T>
class TransformerBlock final : public Layer<T> {
/* Private data elements */
private:
    /**
     * Vector of layers, including normalization, attention, and feed-forward
     */
    std::vector<std::unique_ptr<Layer<T>>> m_layers;

    /**
     * Output of the layer, used in training. For Transformer Block, this is initialized
     * to a single-element Tensor, as the output we are interested in really lives inside
     * the final layer of the Transformer block
     */
    std::unique_ptr<Tensor<T>> m_out = std::make_unique<Tensor<T>>(std::initializer_list<size_t>{1});

    /**
     * Embedding dimension
     */
    size_t c_emb_dim = 0;

    /**
     * Context length
     */
    size_t c_context_len = 0;

    /**
     * Attention Heads
     */
    size_t c_attn_heads = 0;

    /**
     * Dropout
     */
    float c_dropout = 0;

    /**
     * Whether a shortcut connection comes after this layer
     */
    bool c_shortcut = false;

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
                                                        c_attn_heads(attn_heads), c_dropout(dropout),
                                                         c_shortcut(shortcut) {
        // Validate the params
        if (emb_dim == 0 || context_len == 0 || attn_heads == 0) {
            throw std::invalid_argument("TransformerBlock.TransformerBlock: emb_dim, context_len, and attn_head cannot be zero.\n");
        }
        // Ensure the output dimension is divisible by the number of heads
        if (c_emb_dim % c_attn_heads != 0) {
            throw std::invalid_argument("TransformerBlock.TransformerBlock: Embedding dim must be divisible by number of heads.\n");
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

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, cppcoreguidelines-avoid-magic-numbers)
    /**
     * Constructor for TransformerBlock, accepting a pointer to char (bytes) and reading in the data
     * @param s Pointer to the a serialized TransformerBlock (char)
     */
    explicit TransformerBlock(const char* s) {
        // Ensure we have a valid pointer to serialized data
        if (s == nullptr) {
            throw std::invalid_argument("TransformerBlock.TransformerBlock: Invalid pointer provided.\n");
        }
        size_t index = 0;
        // Check that the magic number is correct
        uint32_t u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != TRANSFORMER_BLOCK_START_MAGIC_NUMBER) {
            throw std::invalid_argument("TransformerBlock.TransformerBlock: Invalid start magic number.\n");
        }
        // Get the target size
        size_t total_size = Numerics_NS::from_bytes<size_t>(s, index);
        // Validate the dtype matches the template parameter
        if (index >= total_size) { throw std::runtime_error("TransformerBlock.TransformerBlock: Invalid index or total size for deserializing.\n"); }
        u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != Numerics_NS::Accumulator<T>::_base_type_mn) {
            throw std::runtime_error("TransformerBlock.TransformerBlock: Template parameter does not match serialized dtype.\n");
        }
        // Reserve space and load the layers
        m_layers.reserve(6);
        // Start with a normalization layer
        if (index >= total_size) { throw std::runtime_error("TransformerBlock.TransformerBlock: Invalid index or total size for deserializing.\n"); }
        m_layers.push_back(std::make_unique<NormalizationLayer<T>>(s + index));
        index += m_layers.back()->s_size();
        // Add MultiHeadAttention
        if (index >= total_size) { throw std::runtime_error("TransformerBlock.TransformerBlock: Invalid index or total size for deserializing.\n"); }
        m_layers.push_back(std::make_unique<MultiHeadAttention<T>>(s + index));
        index += m_layers.back()->s_size();
        // Add the first dropout
        if (index >= total_size) { throw std::runtime_error("TransformerBlock.TransformerBlock: Invalid index or total size for deserializing.\n"); }
        m_layers.push_back(std::make_unique<DropoutLayer<T>>(s + index));
        index += m_layers.back()->s_size();
        // Create another normalization layer
        if (index >= total_size) { throw std::runtime_error("TransformerBlock.TransformerBlock: Invalid index or total size for deserializing.\n"); }
        m_layers.push_back(std::make_unique<NormalizationLayer<T>>(s + index));
        index += m_layers.back()->s_size();
        // Create a feed forward network (FFN)
        if (index >= total_size) { throw std::runtime_error("TransformerBlock.TransformerBlock: Invalid index or total size for deserializing.\n"); }
        m_layers.push_back(std::make_unique<FeedForward<T>>(s + index));
        index += m_layers.back()->s_size();
        // Finish the Transformer Block with a final dropout
        if (index >= total_size) { throw std::runtime_error("TransformerBlock.TransformerBlock: Invalid index or total size for deserializing.\n"); }
        m_layers.push_back(std::make_unique<DropoutLayer<T>>(s + index));
        index += m_layers.back()->s_size();
        // Get emb_dim
        if (index >= total_size) { throw std::runtime_error("TransformerBlock.TransformerBlock: Invalid index or total size for deserializing.\n"); }
        c_emb_dim = Numerics_NS::from_bytes<size_t>(s, index);
        // Get context_len
        if (index >= total_size) { throw std::runtime_error("TransformerBlock.TransformerBlock: Invalid index or total size for deserializing.\n"); }
        c_context_len = Numerics_NS::from_bytes<size_t>(s, index);
        // Get attn_heads
        if (index >= total_size) { throw std::runtime_error("TransformerBlock.TransformerBlock: Invalid index or total size for deserializing.\n"); }
        c_attn_heads = Numerics_NS::from_bytes<size_t>(s, index);
        // Get dropout
        if (index >= total_size) { throw std::runtime_error("TransformerBlock.TransformerBlock: Invalid index or total size for deserializing.\n"); }
        c_dropout = Numerics_NS::int_to_float<float>(Numerics_NS::from_bytes<typename Numerics_NS::Accumulator<float>::serialize_as>(s, index));
        // Get the shortcut bool
        if (index >= total_size) { throw std::runtime_error("TransformerBlock.TransformerBlock: Invalid index or total size for deserializing.\n"); }
        c_shortcut = Numerics_NS::from_bytes<uint8_t>(s, index) == 1 ? true : false;
        // Validate the configuration before loading the layers
        if (c_emb_dim == 0 || c_context_len == 0 || c_attn_heads == 0) {
            throw std::invalid_argument("TransformerBlock.TransformerBlock: emb_dim, context_len, and attn_head cannot be zero.\n");
        }
        // Ensure the output dimension is divisible by the number of heads
        if (c_emb_dim % c_attn_heads != 0) {
            throw std::invalid_argument("TransformerBlock.TransformerBlock: Embedding dim must be divisible by number of heads.\n");
        }
        // Ensure we get a dropout within the acceptable range
        if (c_dropout < 0 || c_dropout >= 1 || !std::isfinite(c_dropout)) {
            throw std::invalid_argument("TransformerBlock.TransformerBlock: Dropout must be >= 0, < 1.\n");
        }
        // Ensure the end magic number is valid
        if (index >= total_size) { throw std::runtime_error("TransformerBlock.TransformerBlock: Invalid index or total size for deserializing.\n"); }
        u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != TRANSFORMER_BLOCK_END_MAGIC_NUMBER) {
            throw std::runtime_error("TransformerBlock.TransformerBlock: Invalid end magic number.\n");   
        }
        if (index != total_size) {
            throw std::runtime_error("TransformerBlock.TransformerBlock: Reached end of object but total_size indicates there is more data to be read.\n");
        }
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, cppcoreguidelines-avoid-magic-numbers)

    /**
     * Default destructor for TransformerBlock
     */
    ~TransformerBlock() override {
        // Do nothing
    }

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
     * Forward pass, saving the output to m_out in each layer of the Transformer Block
     * @param input Const ref to an AbstractTensor serving as the input
     * @returns Returns a reference to m_out in Transformer Block (which should not be used)
     */
    Tensor<T> forward_train(const AbstractTensor<T>& input) override {
        // Ensure we have a rank 2 Tensor
        if (input.rank() != 2) {
            throw std::invalid_argument("TransformerBlock.forward_train: Input Tensor must have rank == 2.\n");
        }
        // Ensure that the 2nd dim of input == emb_dim
        if (input.extent(1) != c_emb_dim) {
            throw std::invalid_argument("TransformerBlock.forward_train: Input Tensor's second dim must be emb_dim.\n");
        }
        Tensor<T> result(input.shape());
        forward_train(input, result);
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
        // Copy input --> layer_output and then use it as input / dest for layer iterations
        Tensor<T> layer_output(input.shape());
        layer_output.copy_from(input);
        // Shortcut input
        Tensor<T> shortcut_input(input.shape());
        shortcut_input.copy_from(input);
        // Enable polymorphic behavior
        for (const auto& layer : m_layers) {
            layer->forward(layer_output, layer_output);
            if (layer->has_shortcut()) {
                layer_output += shortcut_input;
                shortcut_input.copy_from(layer_output);
            }
        }
        dest.copy_from(layer_output);
        return dest;
    }

    /**
     * Forward function, performing the forward pass on the layer and writing the result
     * to a defined destination, also copying to m_out for later
     * @param input Const ref to an AbstractTensor serving as input
     * @param dest Reference to a Tensor to store the result in
     * @returns Returns a reference to the destination Tensor provided
     */
    Tensor<T>& forward_train(const AbstractTensor<T>& input, Tensor<T>& dest) override {
        // Ensure we have a rank 2 Tensor
        if (input.rank() != 2 || dest.rank() != 2) {
            throw std::invalid_argument("TransformerBlock.forward_train: Input and dest Tensors must have rank == 2.\n");
        }
        // Ensure that the 2nd dim of input == emb_dim
        if (input.extent(1) != c_emb_dim) {
            throw std::invalid_argument("TransformerBlock.forward_train: Input Tensor's second dim must be emb_dim.\n");
        }
        // Ensure that the 2nd dim of input == emb_dim
        if (input.shape() != dest.shape()) {
            throw std::invalid_argument("TransformerBlock.forward_train: Input and dest must have the same shape.\n");
        }
        if (!dest.is_unique(input)) {
            if constexpr (LAYER_PREFER_TEMP_DESTINATION_OVER_EXCEPTION) {
                Tensor<T> temp_dest = dest.clone();
                forward(input, temp_dest);
                return dest.copy_from(temp_dest);
            }
            throw std::invalid_argument("TransformerBlock.forward_train: Input and destination Tensors must be unique.\n");
        }
        // Iterate over the layers
        // Copy input --> layer_output and then use it as input / dest for layer iterations
        Tensor<T> layer_output(input.shape());
        layer_output.copy_from(input);
        // Shortcut input
        Tensor<T> shortcut_input(input.shape());
        shortcut_input.copy_from(input);
        // Enable polymorphic behavior
        for (const auto& layer : m_layers) {
            layer->forward_train(layer_output, layer_output);
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
        // Add the total size, emb_dim, context_len, attn_heads
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
        result += 4 * Numerics_NS::Accumulator<size_t>::_s_size;
        // Add the shortcut bool --> uint8_t
        result += Numerics_NS::Accumulator<uint8_t>::_s_size;
        // Add dropout
        result += Numerics_NS::Accumulator<float>::_s_size;
        // Add the layers
        for (const auto& layer : m_layers) {
            result += layer->s_size();
        }
        return result;
    }

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    /**
     * Serialize TransformerBlock
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
            throw std::runtime_error("TransformerBlock.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(TRANSFORMER_BLOCK_START_MAGIC_NUMBER))) {
            data[index] = b;
            ++index;
        }
        // Write the serialized size size
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("TransformerBlock.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(target_size))) {
            data[index] = b;
            ++index;
        }
        // Write the dtype
        if (index + Numerics_NS::Accumulator<uint32_t>::_s_size >= target_size) {
            throw std::runtime_error("TransformerBlock.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(Numerics_NS::Accumulator<T>::_base_type_mn))) {
            data[index] = b;
            ++index;
        }
        // Write the layers
        for (const auto& layer : m_layers) {
            if (index + layer->s_size() >= target_size) {
                throw std::runtime_error("TransformerBlock.serialize: Error serializing -- invalid index or s_size.\n");
            }
            std::vector<char> sl = layer->serialize();
            std::memcpy(data + index, sl.data(), sl.size());
            index += sl.size();
        }
        // Write the emb_dim
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("TransformerBlock.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(c_emb_dim))) {
            data[index] = b;
            ++index;
        }
        // Write the context_len
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("TransformerBlock.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(c_context_len))) {
            data[index] = b;
            ++index;
        }
        // Write the attn_heads
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("TransformerBlock.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(c_attn_heads))) {
            data[index] = b;
            ++index;
        }
        // Write the dropout rate
        if (index + Numerics_NS::Accumulator<float>::_s_size >= target_size) {
            throw std::runtime_error("TransformerBlock.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(Numerics_NS::float_to_int(c_dropout)))) {
            data[index] = b;
            ++index;
        }
        // Write the shortcut bool
        if (index + Numerics_NS::Accumulator<uint8_t>::_s_size >= target_size) {
            throw std::runtime_error("TransformerBlock.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(static_cast<uint8_t>(c_shortcut ? 1 : 0)))) {
            data[index] = b;
            ++index;
        }
        // Write the end magic number
        if (index + Numerics_NS::Accumulator<uint32_t>::_s_size > target_size) {
            throw std::runtime_error("TransformerBlock.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(TRANSFORMER_BLOCK_END_MAGIC_NUMBER))) {
            data[index] = b;
            ++index;
        }
        return result;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace TransformerBlock_NS

#endif
