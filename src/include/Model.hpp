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
 * @version: 2026-10-14
 *
 * Model.hpp: Model interface that all models must implement
 */

#ifndef MODEL_HPP
#define MODEL_HPP

/* Standard dependencies */
#include <cstddef>
#include <string>
#include <type_traits>

/* Local dependencies */
#include "AbstractTensor.hpp"
#include "Tensor.hpp"

namespace Model_NS {

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;

/* Use Tensor */
using Tensor_NS::Tensor;

// NOLINTBEGIN(cppcoreguidelines-special-member-functions)
/**
 * Model base class, to be implemented by downstream models
 */
template <typename T>
requires std::is_arithmetic_v<T>
class Model {
/* Public functions */
public:
    /**
     * Virtual destructor
     */
    virtual ~Model() = default;

    /**
     * Run the forward pass of the model, accepting an AbstractTensor of input tokens
     * and returning a Tensor of logits
     * @param input Const ref to an AbstractTensor of shape [batch_size, seq_len]
     * @returns Returns a new Tensor containing the logits
     */
    virtual Tensor<T> forward(const AbstractTensor<size_t>& input) const = 0;

    /**
     * Run the forward pass of the model, accepting an AbstractTensor of input tokens
     * and returning a reference to the destination Tensor, containing the logits
     * @param input Const ref to an AbstractTensor of shape [batch_size, seq_len]
     * @param dest Reference to a Tensor to write the result to
     * @returns Returns a reference to the dest Tensor
     */
    virtual Tensor<T>& forward(const AbstractTensor<size_t>& input, Tensor<T>& dest) const = 0;

    /**
     * Get information about the model
     * @returns Returns a string containing information about the model
     */
    virtual std::string info() const = 0;
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace Model_NS

#endif
