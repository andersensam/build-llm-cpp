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
 * @version: 2026-10-22
 *
 * Notes:
 * LinearLayer.hpp defines a standard linear layer for use in the feed forward network (FFN)
 */

#ifndef LINEAR_LAYER_HPP
#define LINEAR_LAYER_HPP

/* Standard dependencies */
#include <expected>
#include <format>
#include <stdexcept>
#include <string>
#include <type_traits>

/* Local dependencies */
#include "AbstractTensor.hpp"
#include "Layer.hpp"
#include "Numerics.hpp"
#include "Tensor.hpp"
#include "TensorMatmul.hpp"

namespace LinearLayer_NS {

/* Magic numbers for serialization */
inline constexpr uint32_t LINEAR_LAYER_START_MAGIC_NUMBER = 0x88800010;
inline constexpr uint32_t LINEAR_LAYER_END_MAGIC_NUMBER = 0x88800020;

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;

/* Use the Layer interface */
using Layer_NS::Layer;
using Layer_NS::LAYER_PREFER_TEMP_DESTINATION_OVER_EXCEPTION;

/* Use Tensor */
using Tensor_NS::Tensor;

// NOLINTBEGIN(cppcoreguidelines-special-member-functions)
/**
 * LinearLayer class (used in FFN)
 */
template <typename T>
requires std::is_arithmetic_v<T>
class LinearLayer final : public Layer<T> {
/* Private data elements */
private:
    /**
     * Input dimension
     */
    size_t c_input_dim = 0;

    /**
     * Output dimension
     */
    size_t c_output_dim = 0;

    /**
     * Whether a shortcut connection comes after this layer
     */
    bool c_shortcut = false;

    /**
     * Weight Tensor
     */
    Tensor<T> m_w;

/* Public functions */
public:
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    /**
     * Default constructor for LinearLayer
     * @param in_dim Input dimension
     * @param out_dim Output dimension
     */
    LinearLayer(size_t in_dim, size_t out_dim, bool shortcut) : c_input_dim(in_dim), c_output_dim(out_dim), c_shortcut(shortcut),
                                                                m_w({c_input_dim, c_output_dim}) {
        // Initialize the Tensor to have random weights
        if constexpr (std::is_unsigned_v<T>) {
            m_w.random(0, 2);
        }
        else {
            m_w.random(-2, 2);
        }
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    /**
     * LinearLayer destructor
     */
    ~LinearLayer() override {
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
            throw std::invalid_argument("LinearLayer.forward: Input Tensor must have rank == 2.\n");
        }
        // Check to see that we can matmul
        auto compatible = input._can_matmul(0, 1, m_w, 0, 1);
        if (!compatible.has_value()) {
            throw std::invalid_argument(
                std::format("LinearLayer.forward: Incompatible Tensor shapes: {}", compatible.error())
            );
        }
        Tensor<T> output({input.extent(0), m_w.extent(1)});
        forward(input, output);
        return output;
    }

    /**
     * Forward function, performing the forward pass on the layer and writing the result
     * to a defined destination
     * @returns Returns a reference to the destination Tensor provided
     */
    Tensor<T>& forward(const AbstractTensor<T>& input, Tensor<T>& dest) const override {
        // Ensure we have a rank 2 Tensor
        if (input.rank() != 2) {
            throw std::invalid_argument("LinearLayer.forward: Input Tensor must have rank == 2.\n");
        }
        // Check to see that we can matmul
        auto compatible = input._can_matmul(0, 1, m_w, 0, 1);
        if (!compatible.has_value()) {
            throw std::invalid_argument(
                std::format("LinearLayer.forward: Incompatible Tensor shapes: {}", compatible.error())
            );
        }
        if (dest.extent(0) != input.extent(0) || dest.extent(1) != m_w.extent(1)) {
            throw std::invalid_argument(
                std::format("LinearLayer.forward: Invalid destination Tensor. Got dims [{}, {}], but expected [{}, {}];",
                    dest.extent(0), dest.extent(1), input.extent(0), m_w.extent(1))
            );
        }
        if (!dest.is_unique(input)) {
            if constexpr (LAYER_PREFER_TEMP_DESTINATION_OVER_EXCEPTION) {
                Tensor<T> temp_dest = dest.clone();
                forward(input, temp_dest);
                return dest.copy_from(temp_dest);
            }
            throw std::invalid_argument("LinearLayer.forward: Input and destination Tensors must be unique.\n");
        }
        // Execute the matmul
        return TensorMatmul_NS::matmul(input, m_w, dest);
    }

    /**
     * Check to see if a layer has a shortcut
     */
    bool has_shortcut() const override {
        return c_shortcut;
    }

    /**
     * Calculate the size of a Tensor for serialization, including the raw data and any metadata
     * @returns Returns a size_t with the Tensor's total size
     * NOTE: Due to inconsistencies with size_t representation across systems,
     * we use Numerics.hpp to calculate the conversion to fixed-width types, thus
     * the size returned by size() may not reflect what is being used in memory
     */
    size_t s_size() const {
        // Store the magic numbers
        size_t result = 2 * Numerics_NS::Accumulator<uint32_t>::_s_size;
        // Store the input and output dims 
        result += 2 * Numerics_NS::Accumulator<size_t>::_s_size;
        // Store c_shortcut
        result += Numerics_NS::Accumulator<uint8_t>::_s_size;
        // Store m_w
        result += m_w.s_size();
        return result;
    }

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    /**
     * Serialize a Linear Layer
     * @returns Returns a vector of bytes / char containing the layer
     */
    std::vector<char> serialize() const {
        // Persist the target size since we will do bounds checking
        size_t target_size = s_size();
        // Allocate memory using unsigned char (same as std::byte)
        std::vector<char> result(target_size);
        // Grab a pointer to the underlying data
        char* data = result.data();
        // Keep track of the index
        size_t index = 0;
        // As an extra safety check, ensure that index will never exceed s_size
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("LinearLayer.serialize: Error serializing LinearLayer -- invalid index or s_size.\n");
        }
        // Write the LinearLayer start magic number
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(LINEAR_LAYER_START_MAGIC_NUMBER))) {
            data[index] = b;
            ++index;
        }
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("LinearLayer.serialize: Error serializing LinearLayer -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(c_input_dim))) {
            data[index] = b;
            ++index;
        }
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("LinearLayer.serialize: Error serializing LinearLayer -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(c_output_dim))) {
            data[index] = b;
            ++index;
        }
        if (index + Numerics_NS::Accumulator<uint8_t>::_s_size >= target_size) {
            throw std::runtime_error("LinearLayer.serialize: Error serializing LinearLayer -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(static_cast<uint8_t>(c_shortcut ? 1 : 0)))) {
            data[index] = b;
            ++index;
        }
        // Add the m_w Tensor
        std::vector<char> m_w_vector = m_w.serialize();
        // Copy the contents into our output
        std::memcpy(data + index, m_w_vector.data(), m_w_vector.size());
        index += m_w_vector.size();
        if (index + Numerics_NS::Accumulator<uint32_t>::_s_size > target_size) {
            throw std::runtime_error("LinearLayer.serialize: Error serializing LinearLayer -- invalid index or s_size.\n");
        }
        // Write the LinearLayer start magic number
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(LINEAR_LAYER_END_MAGIC_NUMBER))) {
            data[index] = b;
            ++index;
        }
        return result;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace LinearLayer_NS

#endif
