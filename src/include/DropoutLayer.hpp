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
 * Dropout.hpp defines the dropout layer, zeroing out a percentage of entries in a layer
 */

#ifndef DROPOUT_LAYER_HPP
#define DROPOUT_LAYER_HPP

/* Standard dependencies */
#include <array>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

/* Local dependencies */
#include "AbstractTensor.hpp"
#include "Layer.hpp"
#include "Numerics.hpp"
#include "Tensor.hpp"

namespace DropoutLayer_NS {

/* Magic numbers for serialization */
inline constexpr uint32_t DROPOUT_LAYER_START_MAGIC_NUMBER = 0x88800040;
inline constexpr uint32_t DROPOUT_LAYER_END_MAGIC_NUMBER = 0x88800041;

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;

/* Use the Layer interface */
using Layer_NS::Layer;
using Layer_NS::LAYER_PREFER_TEMP_DESTINATION_OVER_EXCEPTION;

/* Use Tensor */
using Tensor_NS::Tensor;

// NOLINTBEGIN(cppcoreguidelines-special-member-functions)
/**
 * Dropout layer
 */
template <typename T>
requires std::is_floating_point_v<T>
class DropoutLayer final : public Layer<T> {
/* Private data elements */
private:
    /**
     * Dropout percentage
     */
    float c_dropout = 0;

    /**
     * Whether a shortcut connection comes after this layer
     */
    bool c_shortcut = false;

/* Public functions */
public:
    /**
     * Default constructor for DropoutLayer
     */
    DropoutLayer(float dropout, bool shortcut) : c_dropout(dropout), c_shortcut(shortcut) {
        // Ensure we get a dropout within the acceptable range
        if (dropout < 0 || dropout >= 1) {
            throw std::invalid_argument("DropoutLayer.DropoutLayer: Dropout must be >= 0, < 1.\n");
        }
    }

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    /**
     * Constructor for DropoutLayer, accepting a pointer to char (bytes) and reading in the data
     * @param s Pointer to the a serialized DropoutLayer (char)
     */
    explicit DropoutLayer(const char* s) {
        // Ensure we have a valid pointer to serialized data
        if (s == nullptr) {
            throw std::invalid_argument("DropoutLayer.DropoutLayer: Invalid pointer provided.\n");
        }
        size_t index = 0;
        // Check that the magic number is correct
        uint32_t u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != DROPOUT_LAYER_START_MAGIC_NUMBER) {
            throw std::invalid_argument("DropoutLayer.DropoutLayer: Invalid start magic number.\n");
        }
        // Get the target size of the LinearLayer
        size_t total_size = Numerics_NS::from_bytes<size_t>(s, index);
        // Validate the dtype matches the template parameter
        u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != Numerics_NS::Accumulator<T>::_base_type_mn) {
            throw std::runtime_error("DropoutLayer.DropoutLayer: Template parameter does not match serialized dtype.\n");
        }
        // Get the dropout float
        if (index >= total_size) { throw std::runtime_error("DropoutLayer.DropoutLayer: Invalid index or total size for deserializing.\n"); }
        c_dropout = Numerics_NS::int_to_float<float>(Numerics_NS::from_bytes<Numerics_NS::Accumulator<float>::serialize_as>(s, index));
        // Get the shortcut bool
        if (index >= total_size) { throw std::runtime_error("DropoutLayer.DropoutLayer: Invalid index or total size for deserializing.\n"); }
        c_shortcut = Numerics_NS::from_bytes<uint8_t>(s, index) == 1 ? true : false;
        // Ensure the end magic number is valid
        if (index >= total_size) { throw std::runtime_error("DropoutLayer.DropoutLayer: Invalid index or total size for deserializing.\n"); }
        u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != DROPOUT_LAYER_END_MAGIC_NUMBER) {
            throw std::runtime_error("DropoutLayer.DropoutLayer: Invalid end magic number.\n");   
        }
        if (index != total_size) {
            throw std::runtime_error("DropoutLayer.DropoutLayer: Reached end of object but total_size indicates there is more data to be read.\n");
        }
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

    /**
     * DropoutLayer destructor
     */
    ~DropoutLayer() override {
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
            throw std::invalid_argument("DropoutLayer.forward: Input Tensor must have rank == 2.\n");
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
        if (input.rank() != 2 || dest.rank() != 2) {
            throw std::invalid_argument("DropoutLayer.forward: Input and dest Tensors must have rank == 2.\n");
        }
        if (input.shape() != dest.shape()) {
            throw std::invalid_argument("DropoutLayer.forward: Input and dest Tensor shapes must match.\n");
        }
        // Since dropout is an in-place operation, we might have input == dest. To ensure we aren't
        // overwriting the input, create a temp destination and then copy the result back
        if (!dest.is_unique(input)) {
            if constexpr (LAYER_PREFER_TEMP_DESTINATION_OVER_EXCEPTION) {
                Tensor<T> temp_dest = dest.clone();
                forward(input, temp_dest);
                return dest.copy_from(temp_dest);
            }
            throw std::invalid_argument("DropoutLayer.forward: Input and destination Tensors must be unique.\n");
        }
        // Copy the input to dest
        dest.copy_from(input);
        // Apply dropout via Tensor.apply_dropout
        dest.apply_dropout(c_dropout);
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
    size_t s_size() const {
        // Start with the magic numbers and dtype
        size_t result = 3 * Numerics_NS::Accumulator<uint32_t>::_s_size;
        // Total size
        result += Numerics_NS::Accumulator<size_t>::_s_size;
        // Add the dropout float
        result += Numerics_NS::Accumulator<float>::_s_size;
        // Add the shortcut bool (cast to uint8_t)
        result += Numerics_NS::Accumulator<uint8_t>::_s_size;
        return result;
    }

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    /**
     * Serialize DropoutLayer
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
        // Write the start magic number
        if (index + Numerics_NS::Accumulator<uint32_t>::_s_size >= target_size) {
            throw std::runtime_error("DropoutLayer.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(DROPOUT_LAYER_START_MAGIC_NUMBER))) {
            data[index] = b;
            ++index;
        }
        // Write the serialized size size
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("DropoutLayer.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(target_size))) {
            data[index] = b;
            ++index;
        }
        // Write the dtype
        if (index + Numerics_NS::Accumulator<uint32_t>::_s_size >= target_size) {
            throw std::runtime_error("DropoutLayer.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(Numerics_NS::Accumulator<T>::_base_type_mn))) {
            data[index] = b;
            ++index;
        }
        // Write the dropout float
        if (index + Numerics_NS::Accumulator<float>::_s_size >= target_size) {
            throw std::runtime_error("DropoutLayer.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(Numerics_NS::float_to_int(c_dropout)))) {
            data[index] = b;
            ++index;
        }
        // Write the shortcut bool
        if (index + Numerics_NS::Accumulator<uint8_t>::_s_size >= target_size) {
            throw std::runtime_error("DropoutLayer.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(static_cast<uint8_t>(c_shortcut ? 1 : 0)))) {
            data[index] = b;
            ++index;
        }
        // Write the end magic number
        if (index + Numerics_NS::Accumulator<uint32_t>::_s_size > target_size) {
            throw std::runtime_error("DropoutLayer.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(DROPOUT_LAYER_END_MAGIC_NUMBER))) {
            data[index] = b;
            ++index;
        }
        return result;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace DropoutLayer_NS

#endif
