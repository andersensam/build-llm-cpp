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
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef ATTENTION_HPP
#define ATTENTION_HPP

/* Standard dependencies */
#include <type_traits>

/* Local dependencies */
#include "AbstractTensor.hpp"
#include "Tensor.hpp"

namespace Attention_NS {
    
/* Control whether or not to prefer creating temporary destination Tensors if is_unique() fails */
inline constexpr bool ATTENTION_PREFER_TEMP_DESTINATION_OVER_EXCEPTION = true;

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;

/* Use Tensor */
using Tensor_NS::Tensor;

// NOLINTBEGIN(cppcoreguidelines-special-member-functions)
/**
 * Abstract base class for all attention implementations
 */
template <typename T> 
requires std::is_arithmetic_v<T>
class Attention {
/* Public functions */
public:
    /**
     * Destructor, to be implemented in the individual Attention impls
     */
    virtual ~Attention() = default;

    /**
     * Run the forward pass of attention, returning a context Tensor for the inputs
     * @param input Const ref to the input AbstractTensor
     * @returns Returns a new Tensor with the calculated context
     */
    virtual Tensor<T> forward(const AbstractTensor<T>& input) const = 0;

    /**
     * Run the forward pass of attention, returning a context Tensor for the inputs
     * @param input Const ref to the input AbstractTensor
     * @param dest Reference to a Tensor to store the output
     * @returns Returns a reference to the dest Tensor with the calculated context
     */
    virtual Tensor<T>& forward(const AbstractTensor<T>& input, Tensor<T>& dest) const = 0;
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace Attention_NS

#endif
