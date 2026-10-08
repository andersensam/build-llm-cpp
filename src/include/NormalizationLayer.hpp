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
 * @version: 2026-10-08
 *
 * Notes:
 * NormalizationLayer.hpp
 */

#ifndef NORMALIZATION_LAYER_HPP
#define NORMALIZATION_LAYER_HPP

/* Standard dependencies */
#include <cmath>
#include <format>
#include <stdexcept>
#include <string>
#include <type_traits>

/* Local dependencies */
#include "AbstractTensor.hpp"
#include "Layer.hpp"
#include "Tensor.hpp"
#include "TensorSlice.hpp"

namespace NormalizationLayer_NS {

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;

/* Use the Layer interface */
using Layer_NS::Layer;

/* Use Tensor */
using Tensor_NS::Tensor;
using Tensor_NS::SqueezedOpType;

/* Use TensorSlice */
using TensorSlice_NS::ListTensorSlice;

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
     * Input dimension
     */
    size_t c_input_dim = 0;

    /**
     * Output dimension
     */
    size_t c_output_dim = 0;

    /**
     * Scale Tensor, used for scaling the output up or down
     */
    Tensor<T> m_scale;

    /**
     * Shift Tensor, used for shifting the output closer to / further from zero
     */
    Tensor<T> m_shift;

/* Public functions */
public:
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    /**
     * Default constructor for NormalizationLayer
     * @param in_dim Input dimension
     * @param out_dim Output dimension
     */
    NormalizationLayer(size_t in_dim, size_t out_dim) : c_input_dim(in_dim), c_output_dim(out_dim),
                                                        m_scale({in_dim, out_dim}), m_shift({in_dim, out_dim}) {
        // Ensure scale is set to 1
        m_scale.fill(1);
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

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
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace NormalizationLayer_NS

#endif
