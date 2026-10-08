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
#include "Tensor.hpp"
#include "TensorMath.hpp"
#include "TensorMatmul.hpp"

namespace LinearLayer_NS {

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;

/* Use the Layer interface */
using Layer_NS::Layer;
using Layer_NS::LAYER_PREFER_TEMP_DESTIATION_OVER_EXCEPTION;

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
    LinearLayer(size_t in_dim, size_t out_dim) : c_input_dim(in_dim), c_output_dim(out_dim), m_w({c_input_dim, c_output_dim}) {
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
                std::format("LinearLayer.foward: Invalid destination Tensor. Got dims [{}, {}], but expected [{}, {}];",
                    dest.extent(0), dest.extent(1), input.extent(0), m_w.extent(1))
            );
        }
        if (!dest.is_unique(input)) {
            if constexpr (LAYER_PREFER_TEMP_DESTIATION_OVER_EXCEPTION) {
                Tensor<T> temp_dest = dest.clone();
                forward(input, temp_dest);
                return dest.copy_from(temp_dest);
            }
            throw std::invalid_argument("LinearLayer.forward: Input and destination Tensors must be unique.\n");
        }
        // Execute the matmul
        return TensorMatmul_NS::matmul(input, m_w, dest);
    }
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace LinearLayer_NS

#endif
