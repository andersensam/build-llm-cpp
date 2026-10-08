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
 * Layer.hpp defines the interface for all layers used throughout the LLM. All normalization, activations, etc.,
 * should inherit from Layer and implement its interface.
 */

#ifndef LAYER_HPP
#define LAYER_HPP

/* Standard dependencies */
#include <stdexcept>
#include <type_traits>

/* Local dependencies */
#include "AbstractTensor.hpp"
#include "Tensor.hpp"
#include "TensorMath.hpp"

namespace Layer_NS {

/* Control whether or not to prefer creating temporary destination Tensors if is_unique() fails */
inline constexpr bool LAYER_PREFER_TEMP_DESTIATION_OVER_EXCEPTION = true;

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;

/* Use Tensor */
using Tensor_NS::Tensor;

// NOLINTBEGIN(cppcoreguidelines-special-member-functions)
/**
 * Abstract Layer class to be implemented
 */
template <typename T>
requires std::is_arithmetic_v<T>
class Layer {
public:
    /**
     * Virtual destructor, to be handled by the implementation
     */
    virtual ~Layer() = default;

    /**
     * Forward function, performing the forward pass on the layer and returning a reference
     * to itself with the computation complete
     * @param input Const ref to an AbstractTensor serving as the input
     * @returns Returns a new Tensor with the result
     */
    virtual Tensor<T> forward(const AbstractTensor<T>& input) const = 0;

    /**
     * Forward function, performing the forward pass on the layer and writing the result
     * to a defined destination
     * @param input Const ref to an AbstractTensor serving as the input
     * @returns Returns a reference to the destination Tensor provided
     */
    virtual Tensor<T>& forward(const AbstractTensor<T>& input, Tensor<T>& dest) const = 0;
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace Layer_NS

#endif
