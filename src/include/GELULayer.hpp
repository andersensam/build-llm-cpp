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
 * GELULayer.hpp defines a Gaussian error linear unit activation layer
 */

#ifndef GELU_LAYER_HPP
#define GELU_LAYER_HPP

/* Standard dependencies */
#include <cmath>
#include <format>
#include <numbers>
#include <stdexcept>
#include <string>
#include <type_traits>

/* Local dependencies */
#include "AbstractTensor.hpp"
#include "Layer.hpp"
#include "Tensor.hpp"

namespace GELULayer_NS {

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;

/* Use the Layer interface */
using Layer_NS::Layer;

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

/* Public functions */
public:
    /**
     * Default constructor for GELULayer
     */
    GELULayer() {
        // Do nothing
    }

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
            Tensor<T> temp_dest = dest.clone();
            forward(input, temp_dest);
            dest.copy_from(temp_dest);
            return dest;
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
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace GELULayer_NS

#endif
