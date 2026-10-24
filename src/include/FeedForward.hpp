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
 * FeedForward.hpp: The main feed-forward layer
 * 
 */

#ifndef FEED_FORWARD_HPP
#define FEED_FORWARD_HPP

/* Standard dependencies */
#include <array>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

/* Local dependencies */
#include "AbstractTensor.hpp"
#include "GELULayer.hpp"
#include "Layer.hpp"
#include "LinearLayer.hpp"
#include "Numerics.hpp"
#include "Tensor.hpp"

namespace FeedForward_NS {

/* Magic numbers for serialization */
inline constexpr uint32_t FEED_FORWARD_START_MAGIC_NUMBER = 0x88800070;
inline constexpr uint32_t FEED_FORWARD_END_MAGIC_NUMBER = 0x88800071;

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;

/* Use GELU */
using GELULayer_NS::GELULayer;

/* Use the Layer interface */
using Layer_NS::Layer;
using Layer_NS::LAYER_PREFER_TEMP_DESTINATION_OVER_EXCEPTION;

/* Use LinearLayer */
using LinearLayer_NS::LinearLayer;

/* Use Tensor */
using Tensor_NS::Tensor;

// NOLINTBEGIN(cppcoreguidelines-special-member-functions)
/**
 * Feed Forward block, which essentially is a layer with multiple
 * sublayers executed in sequence
 */
template <typename T>
requires std::is_floating_point_v<T>
class FeedForward final : public Layer<T> {
/* Private data elements */
private:
    /**
     * Input dimension
     */
    size_t c_emb_dim = 0;

    /**
     * Output dimension
     */
    size_t c_scale_dim = 0;

    /**
     * Whether a shortcut connection comes after this layer
     */
    bool c_shortcut = false;

    /**
     * First Linear Layer, where we scale up the dims input
     */
    std::unique_ptr<LinearLayer<T>> m_linear_0 = nullptr;

    /**
     * GELU Activation Layer
     */
    std::unique_ptr<GELULayer<T>> m_gelu = nullptr;

    /**
     * Second Linear Layer, scaling down the dims back to emb_dim
     */
    std::unique_ptr<LinearLayer<T>> m_linear_1 = nullptr;

/* Public functions */
public:
    /**
     * Default constructor for FeedForward
     */
    FeedForward(size_t emb_dim, size_t scale_dim, bool shortcut) : c_emb_dim(emb_dim), c_scale_dim(scale_dim), c_shortcut(shortcut),
                                                                   m_linear_0(std::make_unique<LinearLayer<T>>(emb_dim, scale_dim * emb_dim, false)),
                                                                   m_gelu(std::make_unique<GELULayer<T>>(false)),
                                                                   m_linear_1(std::make_unique<LinearLayer<T>>(scale_dim * emb_dim, emb_dim, false)) {
        // Ensure we have valid embedding and scale dims
        if (c_emb_dim == 0 || c_scale_dim == 0) {
            throw std::invalid_argument("FeedForward.FeedForward: Invalid embedding or scale dim.\n");
        }
    }

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    /**
     * Constructor for FeedForward, accepting a pointer to char (bytes) and reading in the data
     * @param s Pointer to the a serialized FeedForward (char)
     */
    explicit FeedForward(const char* s) {
        // Ensure we have a valid pointer to serialized data
        if (s == nullptr) {
            throw std::invalid_argument("FeedForward.FeedForward: Invalid pointer provided.\n");
        }
        size_t index = 0;
        // Check that the magic number is correct
        uint32_t u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != FEED_FORWARD_START_MAGIC_NUMBER) {
            throw std::invalid_argument("FeedForward.FeedForward: Invalid start magic number.\n");
        }
        // Get the target size
        size_t total_size = Numerics_NS::from_bytes<size_t>(s, index);
        // Validate the dtype matches the template parameter
        if (index >= total_size) { throw std::runtime_error("FeedForward.FeedForward: Invalid index or total size for deserializing.\n"); }
        u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != Numerics_NS::Accumulator<T>::_base_type_mn) {
            throw std::runtime_error("FeedForward.FeedForward: Template parameter does not match serialized dtype.\n");
        }
        // Get the emb_dim
        if (index >= total_size) { throw std::runtime_error("FeedForward.FeedForward: Invalid index or total size for deserializing.\n"); }
        c_emb_dim = Numerics_NS::from_bytes<size_t>(s, index);
        // Get the scale_dim
        if (index >= total_size) { throw std::runtime_error("FeedForward.FeedForward: Invalid index or total size for deserializing.\n"); }
        c_scale_dim = Numerics_NS::from_bytes<size_t>(s, index);
        // Validate emb_dim and scale_dim
        if (c_emb_dim == 0 || c_scale_dim == 0) {
            throw std::invalid_argument("FeedForward.FeedForward: Invalid embedding or scale dim.\n");
        }
        // Get the shortcut bool
        if (index >= total_size) { throw std::runtime_error("FeedForward.FeedForward: Invalid index or total size for deserializing.\n"); }
        c_shortcut = Numerics_NS::from_bytes<uint8_t>(s, index) == 1 ? true : false;
        // Instantiate m_linear_0
        if (index >= total_size) { throw std::runtime_error("FeedForward.FeedForward: Invalid index or total size for deserializing.\n"); }
        m_linear_0 = std::make_unique<LinearLayer<T>>(s + index);
        index += m_linear_0->s_size();
        // Since LinearLayer doesn't expose anything publicly, we can't do any validation for now
        // Instantiate m_gelu
        if (index >= total_size) { throw std::runtime_error("FeedForward.FeedForward: Invalid index or total size for deserializing.\n"); }
        m_gelu = std::make_unique<GELULayer<T>>(s + index);
        index += m_gelu->s_size();
        // Instantiate m_linear_1
        if (index >= total_size) { throw std::runtime_error("FeedForward.FeedForward: Invalid index or total size for deserializing.\n"); }
        m_linear_1 = std::make_unique<LinearLayer<T>>(s + index);
        index += m_linear_1->s_size();
        // Ensure the end magic number is valid
        if (index >= total_size) { throw std::runtime_error("FeedForward.FeedForward: Invalid index or total size for deserializing.\n"); }
        u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != FEED_FORWARD_END_MAGIC_NUMBER) {
            throw std::runtime_error("FeedForward.FeedForward: Invalid end magic number.\n");   
        }
        if (index != total_size) {
            throw std::runtime_error("FeedForward.FeedForward: Reached end of object but total_size indicates there is more data to be read.\n");
        }
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

    /**
     * FeedForward destructor
     */
    ~FeedForward() override {
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
            throw std::invalid_argument("FeedForward.forward: Input Tensor must have rank == 2.\n");
        }
        // Ensure that the 2nd dim of input == emb_dim
        if (input.extent(1) != c_emb_dim) {
            throw std::invalid_argument("FeedForward.forward: Input Tensor's second dim must be emb_dim.\n");
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
            throw std::invalid_argument("FeedForward.forward: Input and dest Tensors must have rank == 2.\n");
        }
        // Ensure that the 2nd dim of input == emb_dim
        if (input.extent(1) != c_emb_dim) {
            throw std::invalid_argument("FeedForward.forward: Input Tensor's second dim must be emb_dim.\n");
        }
        // Ensure that the 2nd dim of input == emb_dim
        if (input.shape() != dest.shape()) {
            throw std::invalid_argument("FeedForward.forward: Input and dest must have the same shape.\n");
        }
        if (!dest.is_unique(input)) {
            if constexpr (LAYER_PREFER_TEMP_DESTINATION_OVER_EXCEPTION) {
                Tensor<T> temp_dest = dest.clone();
                forward(input, temp_dest);
                return dest.copy_from(temp_dest);
            }
            throw std::invalid_argument("FeedForward.forward: Input and destination Tensors must be unique.\n");
        }
        // Perform the forward operation: m_linear_0 --> m_gelu --> m_linear_1
        Tensor<T> int_res = m_linear_0->forward(input);
        m_gelu->forward(int_res, int_res);
        return m_linear_1->forward(int_res, dest);
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
        // Add the total size, emb_dim, and scale_dim
        result += 3 * Numerics_NS::Accumulator<size_t>::_s_size;
        // Add the shortcut bool --> uint8_t
        result += Numerics_NS::Accumulator<uint8_t>::_s_size;
        // Add m_linear_0
        result += m_linear_0->s_size();
        // Add m_gelu
        result += m_gelu->s_size();
        // Add m_linear_1
        result += m_linear_1->s_size();
        return result;
    }

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    /**
     * Serialize FeedForward
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
            throw std::runtime_error("FeedForward.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(FEED_FORWARD_START_MAGIC_NUMBER))) {
            data[index] = b;
            ++index;
        }
        // Write the serialized size size
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("FeedForward.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(target_size))) {
            data[index] = b;
            ++index;
        }
        // Write the dtype
        if (index + Numerics_NS::Accumulator<uint32_t>::_s_size >= target_size) {
            throw std::runtime_error("FeedForward.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(Numerics_NS::Accumulator<T>::_base_type_mn))) {
            data[index] = b;
            ++index;
        }
        // Write the emb_dim
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("FeedForward.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(c_emb_dim))) {
            data[index] = b;
            ++index;
        }
        // Write the scale_dim
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("FeedForward.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(c_scale_dim))) {
            data[index] = b;
            ++index;
        }
        // Write the shortcut bool
        if (index + Numerics_NS::Accumulator<uint8_t>::_s_size >= target_size) {
            throw std::runtime_error("FeedForward.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(static_cast<uint8_t>(c_shortcut ? 1 : 0)))) {
            data[index] = b;
            ++index;
        }
        // Write m_linear_0
        if (index + m_linear_0->s_size() >= target_size) {
            throw std::runtime_error("FeedForward.serialize: Error serializing -- invalid index or s_size.\n");
        }
        std::vector<char> lin0 = m_linear_0->serialize();
        std::memcpy(data + index, lin0.data(), lin0.size());
        index += lin0.size();
        lin0.clear();
        // Write m_gelu
        if (index + m_gelu->s_size() >= target_size) {
            throw std::runtime_error("FeedForward.serialize: Error serializing -- invalid index or s_size.\n");
        }
        std::vector<char> gelu = m_gelu->serialize();
        std::memcpy(data + index, gelu.data(), gelu.size());
        index += gelu.size();
        gelu.clear();
        // Write m_linear_1
        if (index + m_linear_1->s_size() >= target_size) {
            throw std::runtime_error("FeedForward.serialize: Error serializing -- invalid index or s_size.\n");
        }
        std::vector<char> lin1 = m_linear_1->serialize();
        std::memcpy(data + index, lin1.data(), lin1.size());
        index += lin1.size();
        lin1.clear();
        // Write the end magic number
        if (index + Numerics_NS::Accumulator<uint32_t>::_s_size > target_size) {
            throw std::runtime_error("FeedForward.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(FEED_FORWARD_END_MAGIC_NUMBER))) {
            data[index] = b;
            ++index;
        }
        return result;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace FeedForward_NS

#endif
