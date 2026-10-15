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
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

/* Standard dependencies */
#include <cstdint>
#include <cstddef>
#include <exception>
#include <format>
#include <iostream>
#include <map>
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
        auto tokenizer_ptr = std::make_shared<BytePairEncoding_NS::BytePairEncodingTokenizer>("./models/BPE.model");
        BytePairEncoding_NS::BytePairEncodingTokenizer& BPET = *tokenizer_ptr;
        log_message(Log_Priority::INFO, "main", std::format("Tokenizer initialized with vocab size {}", BPET.vocab_size()));

        // Define the embedding dimension and vocab size
        size_t vocab_size = BPET.vocab_size();
        size_t emb_dim = 1536;

        // Create a data loader using a shared ptr to the tokenizer
        DataLoader d(tokenizer_ptr, 8, 4, 8);
        d.ingest("./data/the-verdict.txt");
        // Get the first input batch and then target batch (shifted by one token)
        const Tensor<size_t>& input_batch = d.next_input();
        const Tensor<size_t>& target_batch = d.next_target();
        log_message(Log_Priority::INFO, "main", std::format("Input batch info: {}\nTarget batch info: {}", input_batch.to_string(), target_batch.to_string()));

        // Instantiate the GPT Model, disabling dropout since this is forward pass only
        GPTModel model(vocab_size, 12, emb_dim, 12, 1024, 0);
        log_message(Log_Priority::INFO, "main", std::format("{}", model.info()));
        // Put the input batch from the dataloader into the model, input = [4, 8], output = [4, 8, vocab_size]
        Tensor<float> logits = model.forward(input_batch);
        // Create a Tensor to dump each batch's token ids into
        Tensor<size_t> batch_tokens({8});
        // Iterate over the batches
        for (size_t i = 0; i < logits.extent(0); ++i) {
            // Extract the input batch token ids
            TensorUtils_NS::copy_2d_tensor_filtered_dim(input_batch, 0, i, batch_tokens);
            // Slice the logits Tensor along the batch dim
            Tensor<float> logits_slice = logits.slice({1, 2}, {}, {{0, i}});
            // Run argmax on the logits slice, giving the next token across the batch
            Tensor<size_t> argmax = logits_slice.max_idx(0, true);
            log_message(Log_Priority::INFO, "main",
                std::format("Batch {}:\nToken ids: {}\nDetokenized input: {}\nOutput token ids: {}\nDetokenized output: {}",
                    i, batch_tokens.to_string(), BPET.detokenize_to_string(batch_tokens), argmax.to_string(), BPET.detokenize_to_string(argmax)));
        }

    } catch (const std::exception& e) {

        std::cout << "Exception: " << e.what() << "\n";
        return -1;
    }

    return 0;
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
