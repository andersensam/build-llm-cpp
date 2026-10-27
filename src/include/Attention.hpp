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
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef ATTENTION_HPP
#define ATTENTION_HPP

/* Standard dependencies */
#include <cstddef>
#include <type_traits>
#include <vector>

/* Local dependencies */
#include "AbstractTensor.hpp"
#include "Layer.hpp"
#include "Tensor.hpp"

namespace Attention_NS {
    
/* Control whether or not to prefer creating temporary destination Tensors if is_unique() fails */
inline constexpr bool ATTENTION_PREFER_TEMP_DESTINATION_OVER_EXCEPTION = true;

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;

/* Use the Layer interface */
using Layer_NS::Layer;

/* Use Tensor */
using Tensor_NS::Tensor;

// NOLINTBEGIN(cppcoreguidelines-special-member-functions)
/**
 * Abstract base class for all attention implementations
 */
template <typename T> 
requires std::is_arithmetic_v<T>
class Attention : public Layer<T> {
/* Public functions */
public:
    /**
     * Destructor, to be implemented in the individual Attention impls
     */
    ~Attention() override = default;

    /**
     * Run the forward pass of attention, returning a context Tensor for the inputs
     * @param input Const ref to the input AbstractTensor
     * @returns Returns a new Tensor with the calculated context
     */
    Tensor<T> forward(const AbstractTensor<T>& input) const override = 0;

    /**
     * Forward pass, saving the output to inside of each layer
     * @param input Const ref to an AbstractTensor serving as the input
     * @returns Returns a reference to the output of the layer
     */
    Tensor<T> forward_train(const AbstractTensor<T>& input) override = 0;

    /**
     * Run the forward pass of attention, returning a context Tensor for the inputs
     * @param input Const ref to the input AbstractTensor
     * @param dest Reference to a Tensor to store the output
     * @returns Returns a reference to the dest Tensor with the calculated context
     */
    Tensor<T>& forward(const AbstractTensor<T>& input, Tensor<T>& dest) const override = 0;

    /**
     * Forward function, performing the forward pass on the layer and writing the result
     * to a defined destination, also copying to m_out for later
     * @param input Const ref to an AbstractTensor serving as input
     * @param dest Reference to a Tensor to store the result in
     * @returns Returns a reference to the destination Tensor provided
     */
    Tensor<T>& forward_train(const AbstractTensor<T>& input, Tensor<T>& dest) override = 0;

    /**
     * Check to see if a layer has a shortcut
     */
    bool has_shortcut() const override = 0;

    /**
     * Calculate the size for serialization, including the raw data and any metadata
     * @returns Returns a size_t with total serialized size
     * NOTE: Due to inconsistencies with size_t representation across systems,
     * we use Numerics.hpp to calculate the conversion to fixed-width types, thus
     * the size returned by size() may not reflect what is being used in memory
     */
    size_t s_size() const override = 0;

    /**
     * Serialize an object
     * @returns Returns a vector of bytes / char containing the layer
     */
    std::vector<char> serialize() const override = 0;
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace Attention_NS

#endif
