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
 * FeedForward.hpp
 */

#ifndef FEED_FORWARD_HPP
#define FEED_FORWARD_HPP

/* Standard dependencies */
#include <memory>
#include <stdexcept>
#include <type_traits>

/* Local dependencies */
#include "AbstractTensor.hpp"
#include "GELULayer.hpp"
#include "Layer.hpp"
#include "LinearLayer.hpp"
#include "Tensor.hpp"

namespace FeedForward_NS {

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;

/* Use GELU */
using GELULayer_NS::GELULayer;

/* Use the Layer interface */
using Layer_NS::Layer;
using Layer_NS::LAYER_PREFER_TEMP_DESTIATION_OVER_EXCEPTION;

/* Use LinearLayer */
using LinearLayer_NS::LinearLayer;

/* Use Tensor */
using Tensor_NS::Tensor;

// NOLINTBEGIN(cppcoreguidelines-special-member-functions)
/**
 * Feed Forward block, which essentially is a layer with multiple
 * sublayers executed in sequence
 */
template <typename T>
requires std::is_floating_point_v<T>
class FeedForward final : public Layer<T> {
/* Private data elements */
private:
    /**
     * Input dimension
     */
    size_t c_emb_dim = 0;

    /**
     * Output dimension
     */
    size_t c_scale_dim = 0;

    /**
     * First Linear Layer, where we scale up the dims input
     */
    LinearLayer<T> linear_0;

    /**
     * GELU Activation Layer
     */
    GELULayer<T> gelu;

    /**
     * Second Linear Layer, scaling down the dims back to emb_dim
     */
    LinearLayer<T> linear_1;

/* Public functions */
public:
    /**
     * Default constructor for FeedForward
     */
    FeedForward(size_t emb_dim, size_t scale_dim) : c_emb_dim(emb_dim), c_scale_dim(scale_dim), linear_0(emb_dim, scale_dim * emb_dim),
                                                    gelu(), linear_1(scale_dim * emb_dim, emb_dim) {
        // Do nothing
    }

    /**
     * FeedForward destructor
     */
    ~FeedForward() override {
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
            throw std::invalid_argument("FeedForward.forward: Input Tensor must have rank == 2.\n");
        }
        // Ensure that the 2nd dim of input == emb_dim
        if (input.extent(1) != c_emb_dim) {
            throw std::invalid_argument("FeedForward.forward: Input Tensor's second dim must be emb_dim.\n");
        }
        Tensor<T> result(input.shape());
        forward(input, result);
        return result;
    }

    /**
     * Forward function, performing the forward pass on the layer and writing the result
     * to a defined destination
     * @param input Const ref to an AbstractTensor serving as the input
     * @returns Returns a reference to the destination Tensor provided
     */
    Tensor<T>& forward(const AbstractTensor<T>& input, Tensor<T>& dest) const override {
        // Ensure we have a rank 2 Tensor
        if (input.rank() != 2 || dest.rank() != 2) {
            throw std::invalid_argument("FeedForward.forward: Input and dest Tensors must have rank == 2.\n");
        }
        // Ensure that the 2nd dim of input == emb_dim
        if (input.extent(1) != c_emb_dim) {
            throw std::invalid_argument("FeedForward.forward: Input Tensor's second dim must be emb_dim.\n");
        }
        // Ensure that the 2nd dim of input == emb_dim
        if (input.shape() != dest.shape()) {
            throw std::invalid_argument("FeedForward.forward: Input and dest must have the same shape.\n");
        }
        if (!dest.is_unique(input)) {
            if constexpr (LAYER_PREFER_TEMP_DESTIATION_OVER_EXCEPTION) {
                Tensor<T> temp_dest = dest.clone();
                forward(input, temp_dest);
                return dest.copy_from(temp_dest);
            }
            throw std::invalid_argument("FeedForward.forward: Input and destination Tensors must be unique.\n");
        }
        // Perform the forward operation: linear_0 --> gelu --> linear_1
        Tensor<T> int_res = linear_0.forward(input);
        gelu.forward(int_res, int_res);
        return linear_1.forward(int_res, dest);
    }
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace FeedForward_NS

#endif
