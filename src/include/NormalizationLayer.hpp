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
 * NormalizationLayer.hpp
 */

#ifndef NORMALIZATION_LAYER_HPP
#define NORMALIZATION_LAYER_HPP

/* Standard dependencies */
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

/* Local dependencies */
#include "AbstractTensor.hpp"
#include "Layer.hpp"
#include "ListTensorSlice.hpp"
#include "Numerics.hpp"
#include "Tensor.hpp"

namespace NormalizationLayer_NS {

/* Magic numbers for serialization */
inline constexpr uint32_t NORMALIZATION_LAYER_START_MAGIC_NUMBER = 0x88800060;
inline constexpr uint32_t NORMALIZATION_LAYER_END_MAGIC_NUMBER = 0x88800061;

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;

/* Use the Layer interface */
using Layer_NS::Layer;
using Layer_NS::LAYER_PREFER_TEMP_DESTINATION_OVER_EXCEPTION;

/* Use TensorSlice */
using ListTensorSlice_NS::ListTensorSlice;

/* Use Tensor */
using Tensor_NS::Tensor;
using Tensor_NS::SqueezedOpType;

// NOLINTBEGIN(cppcoreguidelines-special-member-functions)
/**
 * NormalizationLayer class
 */
template <typename T>
/**
 * TODO: Implement normalization for non-floating point types
 */
requires std::is_floating_point_v<T>
class NormalizationLayer final : public Layer<T> {
/* Private data elements */
private:
    /**
     * Epsilon constant to prevent divide by zero (which shouldn't be possible due to extent())
     */
    static constexpr T eps = 0.00001;

    /**
     * Whether a shortcut connection comes after this layer
     */
    bool c_shortcut = false;

    /**
     * Scale scalar, used for scaling the output up or down
     */
    T m_scale = 1;

    /**
     * Shift scalar, used for shifting the output closer to / further from zero
     */
    T m_shift = 0;

/* Public functions */
public:
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    /**
     * Default constructor for NormalizationLayer
     * @param in_dim Input dimension
     * @param out_dim Output dimension
     */
    explicit NormalizationLayer(bool shortcut) : c_shortcut(shortcut) {
        // Do nothing
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    /**
     * Constructor for NormalizationLayer, accepting a pointer to char (bytes) and reading in the data
     * @param s Pointer to the a serialized NormalizationLayer (char)
     */
    explicit NormalizationLayer(const char* s) {
        // Ensure we have a valid pointer to serialized data
        if (s == nullptr) {
            throw std::invalid_argument("NormalizationLayer.NormalizationLayer: Invalid pointer provided.\n");
        }
        size_t index = 0;
        // Check that the magic number is correct
        uint32_t u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != NORMALIZATION_LAYER_START_MAGIC_NUMBER) {
            throw std::invalid_argument("NormalizationLayer.NormalizationLayer: Invalid start magic number.\n");
        }
        // Get the target size
        size_t total_size = Numerics_NS::from_bytes<size_t>(s, index);
        // Validate the dtype matches the template parameter
        if (index >= total_size) { throw std::runtime_error("NormalizationLayer.NormalizationLayer: Invalid index or total size for deserializing.\n"); }
        u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != Numerics_NS::Accumulator<T>::_base_type_mn) {
            throw std::runtime_error("NormalizationLayer.NormalizationLayer: Template parameter does not match serialized dtype.\n");
        }
        // Get the shortcut bool
        if (index >= total_size) { throw std::runtime_error("NormalizationLayer.NormalizationLayer: Invalid index or total size for deserializing.\n"); }
        c_shortcut = Numerics_NS::from_bytes<uint8_t>(s, index) == 1 ? true : false;
        // Get the scale scalar
        if (index >= total_size) { throw std::runtime_error("NormalizationLayer.NormalizationLayer: Invalid index or total size for deserializing.\n"); }
        m_scale = Numerics_NS::int_to_float<T>(Numerics_NS::from_bytes<typename Numerics_NS::Accumulator<T>::serialize_as>(s, index));
        // Get the shift scalar
        if (index >= total_size) { throw std::runtime_error("NormalizationLayer.NormalizationLayer: Invalid index or total size for deserializing.\n"); }
        m_shift = Numerics_NS::int_to_float<T>(Numerics_NS::from_bytes<typename Numerics_NS::Accumulator<T>::serialize_as>(s, index));
        // Ensure the end magic number is valid
        if (index >= total_size) { throw std::runtime_error("NormalizationLayer.NormalizationLayer: Invalid index or total size for deserializing.\n"); }
        u32_b = Numerics_NS::from_bytes<uint32_t>(s, index);
        if (u32_b != NORMALIZATION_LAYER_END_MAGIC_NUMBER) {
            throw std::runtime_error("NormalizationLayer.NormalizationLayer: Invalid end magic number.\n");   
        }
        if (index != total_size) {
            throw std::runtime_error("NormalizationLayer.NormalizationLayer: Reached end of object but total_size indicates there is more data to be read.\n");
        }
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

    /**
     * NormalizationLayer destructor
     */
    ~NormalizationLayer() override {
        // Do nothing
    }

    /**
     * Forward function, performing the forward pass on the layer and returning a reference
     * to itself with the computation complete
     * @returns Returns a new Tensor with the result
     */
    Tensor<T> forward(const AbstractTensor<T>& input) const override {
        // Ensure we have a rank 2 Tensor before attempting normalization
        if (input.rank() != 2) {
            throw std::invalid_argument("NormalizationLayer.forward: Input Tensor must have rank == 2.\n");
        }
        Tensor<T> output(input.shape());
        forward(input, output);
        return output;
    }

    /**
     * Forward function, performing the forward pass on the layer and writing the result
     * to a defined destination
     * @returns Returns a reference to the destination Tensor provided
     */
    Tensor<T>& forward(const AbstractTensor<T>& input, Tensor<T>& dest) const override {
        // Ensure we have a rank 2 Tensor before attempting normalization
        if (input.rank() != 2) {
            throw std::invalid_argument("NormalizationLayer.forward: Input Tensor must have rank == 2.\n");
        }
        if (dest.extent(0) != input.extent(0) || dest.extent(1) != input.extent(1)) {
            throw std::invalid_argument("NormalizationLayer.forward: Input and dest Tensor shapes do not match.\n");
        }
        // We need convert AbstractTensor --> Tensor
        const Tensor<T>* input_ptr = dynamic_cast<const Tensor<T>*>(&input);
        // If we are dealign with ListTensorSlice, convert via to_tensor()
        if (input_ptr == nullptr) {
            const Tensor<T> input_c = dynamic_cast<const ListTensorSlice<T>*>(&input)->to_tensor();
            return forward(input_c, dest);
        }
        // input_ptr must have been cast successfully, so route to the right function
        return forward(*input_ptr, dest);
    }

    /**
     * Forward function, performing the forward pass on the layer and writing the result
     * to a defined destination
     * @param input Const ref to a Tensor serving as the input
     * @returns Returns a reference to the destination Tensor provided
     */
    Tensor<T>& forward(const Tensor<T>& input, Tensor<T>& dest) const {
        // Ensure we have a rank 2 Tensor before attempting normalization
        if (input.rank() != 2) {
            throw std::invalid_argument("NormalizationLayer.forward: Input Tensor must have rank == 2.\n");
        }
        if (dest.extent(0) != input.extent(0) || dest.extent(1) != input.extent(1)) {
            throw std::invalid_argument("NormalizationLayer.forward: Input and dest Tensor shapes do not match.\n");
        }
        // Since normalization is an in-place operation, we might have input == dest. To ensure we aren't
        // overwriting the input, create a temp destination and then copy the result back
        if (!dest.is_unique(input)) {
            if constexpr (LAYER_PREFER_TEMP_DESTINATION_OVER_EXCEPTION) {
                Tensor<T> temp_dest = dest.clone();
                forward(input, temp_dest);
                return dest.copy_from(temp_dest);
            }
            throw std::invalid_argument("NormalizationLayer.forward: Input and destination Tensors must be unique.\n");
        }
        // Calculate the mean for each input row in the Tensor
        Tensor<T> means({input.extent(0)});
        T denom = static_cast<T>(input.extent(1));
        for (size_t i = 0; i < input.extent(0); ++i) {
            // Mean is the sum / number of elements (which is always extent(1))
            means.at({i}) = (input.sum(0, i) / denom);
        }
        // Calculate the variance
        // First copy the raw values into dest
        dest.copy_from(input);
        // Subtract the mean from each value in input
        dest.squeezed_op(0, means, SqueezedOpType::SUB);
        // Square each value
        dest.apply(std::pow, 2);
        // Create a new squeezed Tensor
        Tensor<T> var_result({input.extent(0)});
        for (size_t i = 0; i < input.extent(0); ++i) {
            // Get the sum of the squares and divide by n, then do sqrt in place since we
            // need it for the next step anyways
            var_result.at({i}) = std::sqrt((dest.sum(0, i) / denom) + eps);
        }
        // Start fresh with copying from input and then subtract mean and divide by the square root
        // of the variance
        dest.copy_from(input);
        dest.squeezed_op(0, means, SqueezedOpType::SUB);
        dest.squeezed_op(0, var_result, SqueezedOpType::DIV);
        // Scale and shift the output
        dest *= m_scale;
        dest += m_shift;
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
        // Add the two scalars
        result += 2 * Numerics_NS::Accumulator<T>::_s_size;
        return result;
    }

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    /**
     * Serialize NormalizationLayer
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
            throw std::runtime_error("NormalizationLayer.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(NORMALIZATION_LAYER_START_MAGIC_NUMBER))) {
            data[index] = b;
            ++index;
        }
        // Write the serialized size size
        if (index + Numerics_NS::Accumulator<size_t>::_s_size >= target_size) {
            throw std::runtime_error("NormalizationLayer.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(target_size))) {
            data[index] = b;
            ++index;
        }
        // Write the dtype
        if (index + Numerics_NS::Accumulator<uint32_t>::_s_size >= target_size) {
            throw std::runtime_error("NormalizationLayer.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(Numerics_NS::Accumulator<T>::_base_type_mn))) {
            data[index] = b;
            ++index;
        }
        // Write the shortcut bool
        if (index + Numerics_NS::Accumulator<uint8_t>::_s_size >= target_size) {
            throw std::runtime_error("NormalizationLayer.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(static_cast<uint8_t>(c_shortcut ? 1 : 0)))) {
            data[index] = b;
            ++index;
        }
        // Write the scale scalar
        if (index + Numerics_NS::Accumulator<T>::_s_size >= target_size) {
            throw std::runtime_error("NormalizationLayer.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(Numerics_NS::float_to_int(m_scale)))) {
            data[index] = b;
            ++index;
        }
        // Write the shift scalar
        if (index + Numerics_NS::Accumulator<T>::_s_size >= target_size) {
            throw std::runtime_error("NormalizationLayer.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(Numerics_NS::float_to_int(m_shift)))) {
            data[index] = b;
            ++index;
        }
        // Write the end magic number
        if (index + Numerics_NS::Accumulator<uint32_t>::_s_size > target_size) {
            throw std::runtime_error("NormalizationLayer.serialize: Error serializing -- invalid index or s_size.\n");
        }
        for (auto b : Numerics_NS::bytes(Numerics_NS::to_little_endian(NORMALIZATION_LAYER_END_MAGIC_NUMBER))) {
            data[index] = b;
            ++index;
        }
        return result;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace NormalizationLayer_NS

#endif
