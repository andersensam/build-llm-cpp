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
 * @version: 2026-10-23
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef MULTI_HEAD_ATTENTION_HPP
#define MULTI_HEAD_ATTENTION_HPP

/* Standard dependencies */
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

/* Local dependencies */
#include "AbstractTensor.hpp"
#include "Attention.hpp"
#include "Numerics.hpp"
#include "Tensor.hpp"
#include "TensorMatmul.hpp"

namespace MultiHeadAttention_NS {

/* Magic numbers for serialization */
inline constexpr uint32_t MULTI_HEAD_ATTENTION_START_MAGIC_NUMBER = 0x888000A0;
inline constexpr uint32_t MULTI_HEAD_ATTENTION_END_MAGIC_NUMBER = 0x888000A1;

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
 * Multi-Head Attention implementation
 */
template <typename T> 
requires std::is_floating_point_v<T>
class MultiHeadAttention final : public Attention<T> {
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
     * Whether a shortcut connection comes after this layer
     */
    bool c_shortcut = false;

    /**
     * Query weights
     */
    Tensor<T> m_w_query = Tensor<T>({1});

    /**
     * Key weights
     */
    Tensor<T> m_w_key = Tensor<T>({1});

    /**
     * Value weights
     */
    Tensor<T> m_w_value = Tensor<T>({1});

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
    MultiHeadAttention(size_t emb_dim, size_t output_dim, size_t context_len,
                       float dropout, size_t num_heads, bool shortcut) :
                                                                        c_emb_dim(emb_dim), c_output_dim(output_dim), 
                                                                        c_dropout(dropout), c_context_len(context_len),
                                                                        c_num_heads(num_heads), c_shortcut(shortcut),
                                                                        m_w_query({c_emb_dim, c_output_dim}),
                                                                        m_w_key({c_emb_dim, c_output_dim}),
                                                                        m_w_value({c_emb_dim, c_output_dim}) {
        // Ensure emb_dim, output_dim, context_len, and num_heads are not zero
        if (emb_dim == 0 || output_dim == 0 || context_len == 0 || num_heads == 0) {
            throw std::invalid_argument("MultiHeadAttention.MultiHeadAttention: emb_dim, output_dim, context_len, and num_heads cannot be zero.\n");
        }
        // Ensure we get a dropout within the acceptable range
        if (dropout < 0 || dropout >= 1) {
            throw std::invalid_argument("MultiHeadAttention.MultiHeadAttention: Dropout must be >= 0, < 1.\n");
        }
        // Ensure the output dimension is divisible by the number of heads
        if (c_output_dim % c_num_heads != 0) {
            throw std::invalid_argument("MultiHeadAttention.MultiHeadAttention: Output dim must be divisible by number of heads.\n");
        }
        c_head_dim = c_output_dim / c_num_heads;
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

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    /**
     * Constructor for MultiHeadAttention, accepting a pointer to char (bytes) and reading in the data
     * @param s Pointer to the a serialized MultiHeadAttention (char)
     */
    explicit MultiHeadAttention(const char* s) {
        // Ensure we have a valid pointer to serialized data
        if (s == nullptr) {
            throw std::invalid_argument("MultiHeadAttention.MultiHeadAttention: Invalid pointer provided.\n");
        }
        size_t index = 0;
        // Check that the magic number is correct
        uint32_t u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != MULTI_HEAD_ATTENTION_START_MAGIC_NUMBER) {
            throw std::invalid_argument("MultiHeadAttention.MultiHeadAttention: Invalid start magic number.\n");
        }
        // Get the target size
        size_t total_size = Numerics_NS::from_bytes<size_t>(s, index);
        // Validate the dtype matches the template parameter
        if (index >= total_size) { throw std::runtime_error("MultiHeadAttention.MultiHeadAttention: Invalid index or total size for deserializing.\n"); }
        u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != Numerics_NS::Accumulator<T>::_base_type_mn) {
            throw std::runtime_error("MultiHeadAttention.MultiHeadAttention: Template parameter does not match serialized dtype.\n");
        }
        // Get emb_dim
        if (index >= total_size) { throw std::runtime_error("MultiHeadAttention.MultiHeadAttention: Invalid index or total size for deserializing.\n"); }
        c_emb_dim = Numerics_NS::from_bytes<size_t>(s, index);
        // Get output_dim
        if (index >= total_size) { throw std::runtime_error("MultiHeadAttention.MultiHeadAttention: Invalid index or total size for deserializing.\n"); }
        c_output_dim = Numerics_NS::from_bytes<size_t>(s, index);
        // Get dropout
        if (index >= total_size) { throw std::runtime_error("MultiHeadAttention.MultiHeadAttention: Invalid index or total size for deserializing.\n"); }
        c_dropout = Numerics_NS::int_to_float<float>(Numerics_NS::from_bytes<typename Numerics_NS::Accumulator<float>::serialize_as>(s, index));
        // Get context_len
        if (index >= total_size) { throw std::runtime_error("MultiHeadAttention.MultiHeadAttention: Invalid index or total size for deserializing.\n"); }
        c_context_len = Numerics_NS::from_bytes<size_t>(s, index);
        // Get num_heads
        if (index >= total_size) { throw std::runtime_error("MultiHeadAttention.MultiHeadAttention: Invalid index or total size for deserializing.\n"); }
        c_num_heads = Numerics_NS::from_bytes<size_t>(s, index);
        // Get the shortcut bool
        if (index >= total_size) { throw std::runtime_error("MultiHeadAttention.MultiHeadAttention: Invalid index or total size for deserializing.\n"); }
        c_shortcut = Numerics_NS::from_bytes<uint8_t>(s, index) == 1 ? true : false;
        // Validate the configuration before loading the Tensors
        // Ensure emb_dim, output_dim, context_len, and num_heads are not zero
        if (c_emb_dim == 0 || c_output_dim == 0 || c_context_len == 0 || c_num_heads == 0) {
            throw std::invalid_argument("MultiHeadAttention.MultiHeadAttention: emb_dim, output_dim, context_len, and num_heads cannot be zero.\n");
        }
        // Ensure we get a dropout within the acceptable range
        if (c_dropout < 0 || c_dropout >= 1 || !std::isfinite(c_dropout)) {
            throw std::invalid_argument("MultiHeadAttention.MultiHeadAttention: Dropout must be >= 0, < 1.\n");
        }
        // Ensure the output dimension is divisible by the number of heads
        if (c_output_dim % c_num_heads != 0) {
            throw std::invalid_argument("MultiHeadAttention.MultiHeadAttention: Output dim must be divisible by number of heads.\n");
        }
        c_head_dim = c_output_dim / c_num_heads;
        // Reserve space for the number of attention heads
        c_heads.reserve(c_num_heads);
        // Shard query, key, and value over the number of attention heads
        for (size_t i = 0; i < c_num_heads; ++i) {
            // start_dim = i * c_head_dim
            // end_dim = (i + 1) * c_head_him
            c_heads.emplace_back(i * c_head_dim, (i + 1) * c_head_dim);
        }
        // Get m_w_query
        if (index >= total_size) { throw std::runtime_error("MultiHeadAttention.MultiHeadAttention: Invalid index or total size for deserializing.\n"); }
        m_w_query = Tensor<T>(s + index);
        index += m_w_query.s_size();
        // Validate m_w_query
        if (m_w_query.rank() != 2 || m_w_query.extent(0) != c_emb_dim || m_w_query.extent(1) != c_output_dim) {
            throw std::runtime_error("MultiHeadAttention.MultiHeadAttention: Invalid Tensor instantiated from serialized config.\n");
        }
        // Get m_w_key
        if (index >= total_size) { throw std::runtime_error("MultiHeadAttention.MultiHeadAttention: Invalid index or total size for deserializing.\n"); }
        m_w_key = Tensor<T>(s + index);
        index += m_w_key.s_size();
        // Validate m_w_key
        if (m_w_key.rank() != 2 || m_w_key.extent(0) != c_emb_dim || m_w_key.extent(1) != c_output_dim) {
            throw std::runtime_error("MultiHeadAttention.MultiHeadAttention: Invalid Tensor instantiated from serialized config.\n");
        }
        // Get m_w_value
        if (index >= total_size) { throw std::runtime_error("MultiHeadAttention.MultiHeadAttention: Invalid index or total size for deserializing.\n"); }
        m_w_value = Tensor<T>(s + index);
        index += m_w_value.s_size();
        // Validate m_w_value
        if (m_w_value.rank() != 2 || m_w_value.extent(0) != c_emb_dim || m_w_value.extent(1) != c_output_dim) {
            throw std::runtime_error("MultiHeadAttention.MultiHeadAttention: Invalid Tensor instantiated from serialized config.\n");
        }
        // Ensure the end magic number is valid
        if (index >= total_size) { throw std::runtime_error("MultiHeadAttention.MultiHeadAttention: Invalid index or total size for deserializing.\n"); }
        u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != MULTI_HEAD_ATTENTION_END_MAGIC_NUMBER) {
            throw std::runtime_error("MultiHeadAttention.MultiHeadAttention: Invalid end magic number.\n");   
        }
        if (index != total_size) {
            throw std::runtime_error("MultiHeadAttention.MultiHeadAttention: Reached end of object but total_size indicates there is more data to be read.\n");
        }
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

    /**
     * Default destructor for MultiHeadAttention
     */
    ~MultiHeadAttention() override {
        // No nothing
    }

    /**
     * Run the forward pass of attention, returning a context Tensor for the inputs
     * @param input Const ref to the input AbstractTensor
     * @returns Returns a new Tensor with the calculated context
     */
    Tensor<T> forward(const AbstractTensor<T>& input) const override {
        // Ensure we have a rank 2 Tensor
        if (input.rank() != 2) {
            throw std::invalid_argument("MultiHeadAttention.forward: Input Tensor must have rank == 2.\n");
        }
        Tensor<T> result({input.extent(0), c_output_dim});
        forward(input, result);
        return result;
    }

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    /**
     * Run the forward pass of attention, returning a context Tensor for the inputs
     * @param input Const ref to the input AbstractTensor
     * @param dest Reference to a Tensor to store the output
     * @returns Returns a reference to the dest Tensor with the calculated context
     */
    Tensor<T>& forward(const AbstractTensor<T>& input, Tensor<T>& dest) const override {
        // Ensure we have a rank 2 Tensor
        if (input.rank() != 2 || dest.rank() != 2) {
            throw std::invalid_argument("MultiHeadAttention.forward: Input and destination Tensors must have rank == 2.\n");
        }
        // Ensure destination has the right shape
        if (dest.extent(0) != input.extent(0) || dest.extent(1) != c_output_dim) {
            throw std::invalid_argument("MultiHeadAttention.forward: Incorrect shape for destination Tensor.\n");
        }
        if (!dest.is_unique(input)) {
            if constexpr (ATTENTION_PREFER_TEMP_DESTINATION_OVER_EXCEPTION) {
                Tensor<T> temp_dest = dest.clone();
                forward(input, temp_dest);
                return dest.copy_from(temp_dest);
            }
            throw std::invalid_argument("MultiHeadAttention.forward: Input and destination Tensors must be unique.\n");
        }
        // Get the QKV for the given input
        Tensor<T> queries = matmul(input, m_w_query);
        Tensor<T> keys = matmul(input, m_w_key);
        Tensor<T> values = matmul(input, m_w_value);
        // Create an output Tensor
        Tensor<T> output({input.extent(0), input.extent(0)});
        // Create another Tensor for attn_weights @ values
        Tensor<T> context_vec({input.extent(0), c_head_dim});
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
                Tensor_NS::unsafe_uniform_op(output, std::sqrt(static_cast<T>(c_head_dim)), output, Tensor_NS::SqueezedOpType::DIV);
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
                    dest.at({i, j}) = context_vec.at({i, j - (head_id * c_head_dim)});
                }
            }
        }
        // Return the concatenated context vector
        return dest;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

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
        // Add the total size, emb_dim, output_dim, context_len, num_heads
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
        result += 5 * Numerics_NS::Accumulator<size_t>::_s_size;
        // Add the shortcut bool --> uint8_t
        result += Numerics_NS::Accumulator<uint8_t>::_s_size;
        // Add dropout
        result += Numerics_NS::Accumulator<float>::_s_size;
        // Add query, key, and value Tensors, all of which have the same shape and size
        result += 3 * m_w_query.s_size();
        return result;
    }

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    /**
     * Serialize MultiHeadAttention
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
            throw std::runtime_error("MultiHeadAttention.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(MULTI_HEAD_ATTENTION_START_MAGIC_NUMBER))) {
            data[index] = b;
            ++index;
        }
        // Write the serialized size size
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("MultiHeadAttention.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(target_size))) {
            data[index] = b;
            ++index;
        }
        // Write the dtype
        if (index + Numerics_NS::Accumulator<uint32_t>::_s_size >= target_size) {
            throw std::runtime_error("MultiHeadAttention.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(Numerics_NS::Accumulator<T>::_base_type_mn))) {
            data[index] = b;
            ++index;
        }
        // Write the emb_dim
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("MultiHeadAttention.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(c_emb_dim))) {
            data[index] = b;
            ++index;
        }
        // Write the output_dim
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("MultiHeadAttention.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(c_output_dim))) {
            data[index] = b;
            ++index;
        }
        // Write the dropout rate
        if (index + Numerics_NS::Accumulator<float>::_s_size >= target_size) {
            throw std::runtime_error("MultiHeadAttention.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(Numerics_NS::float_to_int(c_dropout)))) {
            data[index] = b;
            ++index;
        }
        // Write the context_len
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("MultiHeadAttention.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(c_context_len))) {
            data[index] = b;
            ++index;
        }
        // Write the num_heads
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("MultiHeadAttention.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(c_num_heads))) {
            data[index] = b;
            ++index;
        }
        // Write the shortcut bool
        if (index + Numerics_NS::Accumulator<uint8_t>::_s_size >= target_size) {
            throw std::runtime_error("MultiHeadAttention.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(static_cast<uint8_t>(c_shortcut ? 1 : 0)))) {
            data[index] = b;
            ++index;
        }
        // Write m_w_query Tensor
        if (index + m_w_query.s_size() >= target_size) {
            throw std::runtime_error("MultiHeadAttention.serialize: Error serializing -- invalid index or s_size.\n");
        }
        std::vector<char> st = m_w_query.serialize();
        std::memcpy(data + index, st.data(), st.size());
        index += st.size();
        st.clear();
        // Write m_w_key Tensor
        if (index + m_w_key.s_size() >= target_size) {
            throw std::runtime_error("MultiHeadAttention.serialize: Error serializing -- invalid index or s_size.\n");
        }
        st = m_w_key.serialize();
        std::memcpy(data + index, st.data(), st.size());
        index += st.size();
        st.clear();
        // Write m_w_value Tensor
        if (index + m_w_value.s_size() >= target_size) {
            throw std::runtime_error("MultiHeadAttention.serialize: Error serializing -- invalid index or s_size.\n");
        }
        st = m_w_value.serialize();
        std::memcpy(data + index, st.data(), st.size());
        index += st.size();
        st.clear();
        // Write the end magic number
        if (index + Numerics_NS::Accumulator<uint32_t>::_s_size > target_size) {
            throw std::runtime_error("MultiHeadAttention.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(MULTI_HEAD_ATTENTION_END_MAGIC_NUMBER))) {
            data[index] = b;
            ++index;
        }
        return result;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace MultiHeadAttention_NS

#endif
