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
 * Notes:
 * GELULayer.hpp defines a Gaussian error linear unit activation layer
 */

#ifndef GELU_LAYER_HPP
#define GELU_LAYER_HPP

/* Standard dependencies */
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <numbers>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

/* Local dependencies */
#include "AbstractTensor.hpp"
#include "Layer.hpp"
#include "Numerics.hpp"
#include "Tensor.hpp"

namespace GELULayer_NS {

/* Magic numbers for serialization */
inline constexpr uint32_t GELU_LAYER_START_MAGIC_NUMBER = 0x88800050;
inline constexpr uint32_t GELU_LAYER_END_MAGIC_NUMBER = 0x88800051;

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;

/* Use the Layer interface */
using Layer_NS::Layer;
using Layer_NS::LAYER_PREFER_TEMP_DESTINATION_OVER_EXCEPTION;

/* Use Tensor */
using Tensor_NS::Tensor;

// NOLINTBEGIN(cppcoreguidelines-special-member-functions)
/**
 * GELULayer class (used in FFN)
 */
template <typename T>
requires std::is_floating_point_v<T>
class GELULayer final : public Layer<T> {
/* Private data elements */
private:
    /**
     * Constant used by the GELU activation function
     */
    static constexpr T _gelu_const = 0.044715;

    /**
     * Scaling constant for the input (const) * input * (1 + tanh[...])
     */
    static constexpr T _scale_const = 0.5;

    /**
     * Whether a shortcut connection comes after this layer
     */
    bool c_shortcut = false;

/* Public functions */
public:
    /**
     * Default constructor for GELULayer
     */
    explicit GELULayer(bool shortcut) : c_shortcut(shortcut) {
        // Do nothing
    }

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    /**
     * Constructor for GELULayer, accepting a pointer to char (bytes) and reading in the data
     * @param s Pointer to the a serialized GELULayer (char)
     */
    explicit GELULayer(const char* s) {
        // Ensure we have a valid pointer to serialized data
        if (s == nullptr) {
            throw std::invalid_argument("GELULayer.GELULayer: Invalid pointer provided.\n");
        }
        size_t index = 0;
        // Check that the magic number is correct
        uint32_t u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != GELU_LAYER_START_MAGIC_NUMBER) {
            throw std::invalid_argument("GELULayer.GELULayer: Invalid start magic number.\n");
        }
        // Get the target size
        size_t total_size = Numerics_NS::from_bytes<size_t>(s, index);
        // Validate the dtype matches the template parameter
        if (index >= total_size) { throw std::runtime_error("GELULayer.GELULayer: Invalid index or total size for deserializing.\n"); }
        u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != Numerics_NS::Accumulator<T>::_base_type_mn) {
            throw std::runtime_error("GELULayer.GELULayer: Template parameter does not match serialized dtype.\n");
        }
        // Get the shortcut bool
        if (index >= total_size) { throw std::runtime_error("GELULayer.GELULayer: Invalid index or total size for deserializing.\n"); }
        c_shortcut = Numerics_NS::from_bytes<uint8_t>(s, index) == 1 ? true : false;
        // Ensure the end magic number is valid
        if (index >= total_size) { throw std::runtime_error("GELULayer.GELULayer: Invalid index or total size for deserializing.\n"); }
        u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != GELU_LAYER_END_MAGIC_NUMBER) {
            throw std::runtime_error("GELULayer.GELULayer: Invalid end magic number.\n");   
        }
        if (index != total_size) {
            throw std::runtime_error("GELULayer.GELULayer: Reached end of object but total_size indicates there is more data to be read.\n");
        }
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

    /**
     * GELULayer destructor
     */
    ~GELULayer() override {
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
            throw std::invalid_argument("GELULayer.forward: Input Tensor must have rank == 2.\n");
        }
        Tensor<T> result(input.shape());
        forward(input, result);
        return result;
    }

    /**
     * Forward function, performing the forward pass on the layer and writing the result
     * to a defined destination
     * @returns Returns a reference to the destination Tensor provided
     */
    Tensor<T>& forward(const AbstractTensor<T>& input, Tensor<T>& dest) const override {
        // Ensure we have a rank 2 Tensor
        if (input.rank() != 2) {
            throw std::invalid_argument("GELULayer.forward: Input Tensor must have rank == 2.\n");
        }
        // Since GELU is an in-place operation, we might have input == dest. To ensure we aren't
        // overwriting the input, create a temp destination and then copy the result back
        if (!dest.is_unique(input)) {
            if constexpr (LAYER_PREFER_TEMP_DESTINATION_OVER_EXCEPTION) {
                Tensor<T> temp_dest = dest.clone();
                forward(input, temp_dest);
                return dest.copy_from(temp_dest);
            }
            throw std::invalid_argument("GELULayer.forward: Input and destination Tensors must be unique.\n");
        }
        // Copy the input to dest
        dest.copy_from(input);
        // GELU requires a few copies of the input
        Tensor<T> x = dest.clone();
        // Take input ^ 3
        x.apply(std::pow, 3);
        // Multiply input ^ 3 by the constant 0.044715
        x *= _gelu_const;
        // Add the result to the input
        dest += x;
        // Scale dest by sqrt(2 / pi)
        dest *= std::sqrt(2 / std::numbers::pi_v<T>);
        // Take tanh
        dest.apply(std::tanh);
        // Add one to the result
        dest += 1;
        // Copy the original input back into x
        x.copy_from(input);
        // Scale by 0.5
        x *= _scale_const;
        // Multiply dest * x to get the final result
        dest *= x;
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
        // Add the total size
        result += Numerics_NS::Accumulator<size_t>::_s_size;
        // Add the shortcut bool --> uint8_t
        result += Numerics_NS::Accumulator<uint8_t>::_s_size;
        return result;
    }

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    /**
     * Serialize GELULayer
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
            throw std::runtime_error("GELULayer.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(GELU_LAYER_START_MAGIC_NUMBER))) {
            data[index] = b;
            ++index;
        }
        // Write the serialized size size
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("GELULayer.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(target_size))) {
            data[index] = b;
            ++index;
        }
        // Write the dtype
        if (index + Numerics_NS::Accumulator<uint32_t>::_s_size >= target_size) {
            throw std::runtime_error("GELULayer.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(Numerics_NS::Accumulator<T>::_base_type_mn))) {
            data[index] = b;
            ++index;
        }
        // Write the shortcut bool
        if (index + Numerics_NS::Accumulator<uint8_t>::_s_size >= target_size) {
            throw std::runtime_error("GELULayer.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(static_cast<uint8_t>(c_shortcut ? 1 : 0)))) {
            data[index] = b;
            ++index;
        }
        // Write the end magic number
        if (index + Numerics_NS::Accumulator<uint32_t>::_s_size > target_size) {
            throw std::runtime_error("GELULayer.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(GELU_LAYER_END_MAGIC_NUMBER))) {
            data[index] = b;
            ++index;
        }
        return result;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace GELULayer_NS

#endif
