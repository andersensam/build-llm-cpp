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
 * GPTModel.hpp: A GPT-like model implementation
 */

#ifndef GPT_MODEL_HPP
#define GPT_MODEL_HPP

/* Standard dependencies */
#include <cstddef>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

/* Local dependencies */
#include "AbsolutePositionalEmbedding.hpp"
#include "AbstractTensor.hpp"
#include "DropoutLayer.hpp"
#include "Embedding.hpp"
#include "LinearLayer.hpp"
#include "ListTensorSlice.hpp"
#include "Model.hpp"
#include "NormalizationLayer.hpp"
#include "Tensor.hpp"
#include "TensorUtils.hpp"
#include "TransformerBlock.hpp"

namespace GPTModel_NS {

/* Use AbsolutePositionalEmbedding */
using AbsolutePositionalEmbedding_NS::AbsolutePositionalEmbedding;

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;

/* Use Dropout */
using DropoutLayer_NS::DropoutLayer;

/* Use Embedding */
using Embedding_NS::Embedding;

/* Use LinearLayer */
using LinearLayer_NS::LinearLayer;

/* Use ListTensorSlice */
using ListTensorSlice_NS::ListTensorSlice;

/* Use the Model interface */
using Model_NS::Model;

/* Use Normalization */
using NormalizationLayer_NS::NormalizationLayer;

/* Use Tensor */
using Tensor_NS::Tensor;

/* Use TransformerBlock */
using TransformerBlock_NS::TransformerBlock;

// NOLINTBEGIN(cppcoreguidelines-special-member-functions)
/**
 * GPT Model implementation, using the Model base class with dtype=float
 */
class GPTModel : public Model<float> {
/* Private data elements */
private:
    /**
     * Vocab size
     */
    size_t c_vocab_size = 0;

    /**
     * Number of layers in the model
     */
    size_t c_num_layers = 0;

    /**
     * Embedding dimension of the model
     */
    size_t c_emb_dim = 0;

    /**
     * Number of attention heads
     */
    size_t c_attn_heads = 0;

    /**
     * Context len (max_seq_len)
     */
    size_t c_context_len = 0;

    /**
     * Dropout percentage
     */
    float c_dropout = 0;

    /**
     * Token embedding layer
     */
    std::unique_ptr<Embedding<float>> m_tok_emb;

    /**
     * AbsolutePositionalEmbedding layer
     */
    std::unique_ptr<AbsolutePositionalEmbedding<float>> m_pos_emb;

    /**
     * Dropout layer
     */
    std::unique_ptr<DropoutLayer<float>> m_drop;

    /**
     * TransformerBlocks, matching num_layers
     */
    std::vector<std::unique_ptr<TransformerBlock<float>>> m_layers;

    /**
     * Normalization layer
     */
    std::unique_ptr<NormalizationLayer<float>> m_norm;

    /**
     * Final linear output layer
     */
    std::unique_ptr<LinearLayer<float>> m_linear_output;

    /**
     * Model name
     */
    static constexpr std::string_view _model_name = "GPTModel";

    /**
     * Dtype used by the model
     */
    static constexpr std::string_view _model_dtype = "float32";

/* Public functions */
public:
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    /**
     * Default constructor for GPTModel
     * @param vocab_size Number of tokens in the model's vocab
     * @param num_layers Number of TransformerBlocks to use
     * @param emb_dim Embedding dimension of the model
     * @param attn_heads Number of attention heads (emb_dim % attn_heads == 0)
     * @param context_len Context size of the model (max_seq_len)
     * @param dropout Inverted dropout percentage (0.0 - 1.0)
     */
    GPTModel(size_t vocab_size, size_t num_layers, size_t emb_dim, size_t attn_heads, size_t context_len, float dropout);
    // NOLINTEND(bugprone-easily-swappable-parameters)
    
        /**
     * Default destructor for GPTModel
     */
    ~GPTModel() override;

    /**
     * Run the forward pass of the model, accepting an AbstractTensor of input tokens
     * and returning a Tensor of logits
     * @param input Const ref to an AbstractTensor of shape [batch_size, seq_len]
     * @returns Returns a new Tensor containing the logits, shape [batch_size, seq_len, vocab_size]
     */
    Tensor<float> forward(const AbstractTensor<size_t>& input) const override;

    /**
     * Run the forward pass of the model, accepting an AbstractTensor of input tokens
     * and returning a reference to the destination Tensor, containing the logits
     * @param input Const ref to an AbstractTensor of shape [batch_size, seq_len]
     * @param dest Reference to a Tensor to write the result to, shape [batch_size, seq_len, vocab_size]
     * @returns Returns a reference to the dest Tensor
     */
    Tensor<float>& forward(const AbstractTensor<size_t>& input, Tensor<float>& dest) const override;

    /**
     * Run the forward pass of the model, accepting a Tensor of input tokens
     * and returning a reference to the destination Tensor, containing the logits
     * @param input Const ref to a Tensor of shape [batch_size, seq_len]
     * @param dest Reference to a Tensor to write the result to, shape [batch_size, seq_len, vocab_size]
     * @returns Returns a reference to the dest Tensor
     */
    Tensor<float>& forward(const Tensor<size_t>& input, Tensor<float>& dest) const;

    /**
     * Get information about the model
     * @returns Returns a string containing information about the model
     */
    std::string info() const override;
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace GPTModel_NS

#endif
