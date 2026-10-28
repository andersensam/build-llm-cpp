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
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

/* Standard dependencies */
#include <cstddef>
#include <exception>
#include <format>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

/* Local dependencies */
#include "include/BytePairEncoding.hpp"
#include "include/DataLoader.hpp"
#include "include/GPTModel.hpp"
#include "include/Log.hpp"
#include "include/Tensor.hpp"
#include "include/TensorUtils.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
int main() {

    using DataLoader_NS::DataLoader;

    using GPTModel_NS::GPTModel;

    using Log_NS::Log_Priority;
    using Log_NS::log_message;

    using Tensor_NS::Tensor;

    try {
        
        log_message(Log_Priority::INFO, "main", "Initializing tokenizer");
        auto tokenizer_ptr = std::make_shared<BytePairEncoding_NS::BytePairEncodingTokenizer>("./models/BPE4.model");
        BytePairEncoding_NS::BytePairEncodingTokenizer& BPET = *tokenizer_ptr;
        log_message(Log_Priority::INFO, "main", std::format("Tokenizer initialized with vocab size {}", BPET.vocab_size()));

        // Define the embedding dimension and vocab size
        size_t vocab_size = BPET.vocab_size();

        // Create a data loader using a shared ptr to the tokenizer
        size_t dataloader_max_len = 8;
        size_t dataloader_batch_size = 2;
        size_t dataloader_stride = 8;
        DataLoader d(tokenizer_ptr, dataloader_max_len, dataloader_batch_size, dataloader_stride);
        d.ingest("./data/the-verdict.txt");
        // Get the first input batch and then target batch (shifted by one token)
        const Tensor<size_t>& input_batch = d.next_input();
        const Tensor<size_t>& target_batch = d.next_target();
        log_message(Log_Priority::INFO, "main", std::format("Input batch info: {}\nTarget batch info: {}", input_batch.to_string(), target_batch.to_string()));

        // Instantiate the GPT Model, disabling dropout since this is forward pass only
        // Use the below if you do not have a presaved model checkpoint
        /*
        size_t gptmodel_num_layers = 12;
        size_t gptmodel_emb_dim = 768;
        size_t gptmodel_attn_heads = 12;
        size_t gptmodel_max_seq_len = 1024;
        float gptmodel_dropout = 0.0f;
        GPTModel model(vocab_size, gptmodel_num_layers, gptmodel_emb_dim, gptmodel_attn_heads, gptmodel_max_seq_len, gptmodel_dropout);
        */
        // Use the below to load a checkpoint
        GPTModel model("./models/GPT2-124M.model");
        log_message(Log_Priority::INFO, "main", std::format("{}", model.info()));
        // Define the number of tokens we want out from the model
        size_t target_num_tokens = 8;
        // Create an output Tensor that is of dim [batch_size, dataloader_max_len + target_num_tokens]
        Tensor<size_t> final_input({dataloader_batch_size, dataloader_max_len + target_num_tokens});
        // Create a slice of size [batch_size, dataloader_max_len] and copy the token ids into it
        Tensor<size_t> model_input = final_input.slice({0, 1}, {{0, 0}, {0, dataloader_max_len}}, {});
        // We know the final ouput from logits will have size [batch_size, dataloader_max_len + target_num_tokens, vocab_size]
        Tensor<float> final_logits({dataloader_batch_size, dataloader_max_len + target_num_tokens, vocab_size});
        model_input.copy_from(input_batch);
        log_message(Log_Priority::INFO, "main", "Starting forward pass.");
        for (size_t i = 0; i < target_num_tokens; ++i) {
            // Handle the case where we hit the max_seq_len
            size_t seq_end = dataloader_max_len + i;
            size_t seq_begin = seq_end > model.context_len() ? (seq_end - model.context_len()) : 0;
            // Create a slice from final_input of shape [batch_size, dataloader_max_len + i (up to target_num_tokens)]
            // when i = 0, this is a no-op
            Tensor<size_t> current_input = final_input.slice({0, 1}, {{0, 0}, {seq_begin, seq_end}}, {});
            // We can slice the final_logits Tensor and use it as a destination to avoid reallocating memory
            Tensor<float> logits = final_logits.slice({0, 1, 2}, {{0, 0}, {seq_begin, seq_end}, {0, 0}}, {});
            model.forward(current_input, logits);
            // Run the forward pass and generate the logits of shape [batch_size, dataloader_max_len + i, vocab_size]
            // Create a slice to grab the last token across all batches with shape [batch_size, vocab_size], indexing on only the final token output
            Tensor<float> logits_slice = logits.slice({0, 2}, {}, {{1, logits.extent(1) - 1}});
            // Calculate the argmax for all batches, shape [batch_size]
            Tensor<size_t> argmax = logits_slice.max_idx(0, true);
            for (size_t j = 0; j < argmax.elements(); ++j) {
                final_input.at({j, dataloader_max_len + i}) = argmax.at(j);
            }
        }
        log_message(Log_Priority::INFO, "main", "Finished forward pass.");
        for (size_t i = 0; i < dataloader_batch_size; ++i) {
            Tensor<size_t> input_slice = input_batch.slice({1}, {}, {{0, i}});
            Tensor<size_t> output_slice = final_input.slice({1}, {}, {{0, i}});
            log_message(Log_Priority::INFO, "main",
                std::format(
                    "[Batch {}]: Input token ids: {}\nDecoded input: {}\nDesired tokens: {}\nOutput token ids: {}\nDecoded output: {}",
                        i, input_slice.to_string(), BPET.detokenize_to_string(input_slice), target_num_tokens,
                        output_slice.to_string(), BPET.detokenize_to_string(output_slice)
                )
            );
        }
        // Start the backward pass
        log_message(Log_Priority::INFO, "main", "Starting backward pass.");
        size_t target_train_steps = 1;
        for (size_t i = 0; i < target_train_steps; ++i) {
            // Get the input and target batches
            const Tensor<size_t>& current_input_batch = d.next_input();
            const Tensor<size_t>& current_target_batch = d.next_target();
            log_message(Log_Priority::INFO, "main", std::format("Input batch info: {}\nTarget batch info: {}", input_batch.to_string(), target_batch.to_string()));
            // NOTE: We are just calculating loss and not actually doing any training yet
            Tensor<float> current_batch_loss = model.train(current_input_batch, current_target_batch);
            for (size_t j = 0; j < current_batch_loss.elements(); ++j) {
                log_message(Log_Priority::INFO, "main", std::format("Training step {}, mini-batch {}: loss={}", i, j, current_batch_loss.at(j)));
            }
        }
        log_message(Log_Priority::INFO, "main", "Backward pass complete.");
        // Save the model
        /*
        if (!model.save_to_file("./models/GPT2-124M.model")) {
            log_message(Log_Priority::ERROR, "main", "Failed to save model.");
        }
        */

    } catch (const std::exception& e) {

        std::cout << "Exception: " << e.what() << "\n";
        return -1;
    }

    return 0;
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
