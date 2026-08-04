/*  ________   ___   __    ______   ______   ______    ______   ______   ___   __    ______   ________   ___ __ __     
 * /_______/\ /__/\ /__/\ /_____/\ /_____/\ /_____/\  /_____/\ /_____/\ /__/\ /__/\ /_____/\ /_______/\ /__//_//_/\    
 * \::: _  \ \\::\_\\  \ \\:::_ \ \\::::_\/_\:::_ \ \ \::::_\/_\::::_\/_\::\_\\  \ \\::::_\/_\::: _  \ \\::\| \| \ \   
 *  \::(_)  \ \\:. `-\  \ \\:\ \ \ \\:\/___/\\:(_) ) )_\:\/___/\\:\/___/\\:. `-\  \ \\:\/___/\\::(_)  \ \\:.      \ \  
 *   \:: __  \ \\:. _    \ \\:\ \ \ \\::___\/_\: __ `\ \\_::._\:\\::___\/_\:. _    \ \\_::._\:\\:: __  \ \\:.\-/\  \ \ 
 *    \:.\ \  \ \\. \`-\  \ \\:\/.:| |\:\____/\\ \ `\ \ \ /____\:\\:\____/\\. \`-\  \ \ /____\:\\:.\ \  \ \\. \  \  \ \
 *     \__\/\__\/ \__\/ \__\/ \____/_/ \_____\/ \_\/ \_\/ \_____\/ \_____\/ \__\/ \__\/ \_____\/ \__\/\__\/ \__\/ \__\/    
 *                                                                                                               
 * Project: Large Language Model in C++
 * @author : Samuel Andersen
 * @version: 2026-08-01
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

/* Standard dependencies */
#include <exception>
#include <format>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

/* Local dependencies */
#include "include/Log.hpp"
#include "include/BytePairEncoding.hpp"
#include "include/Tensor.hpp"
#include "include/DataLoader.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
int main() {

    using Log::Log_Priority;
    using Log::log_message;
    using Tensor_NS::Tensor;
    using Tensor_NS::Matrix;
    using Tensor_NS::TensorSlice;
    using DataLoader_NS::DataLoader;

    try {
        
        log_message(Log_Priority::INFO, "main", "Initializing tokenizer");
        auto tokenizer_ptr = std::make_shared<BytePairEncoding_NS::BytePairEncodingTokenizer>("./models/BPE.model");
        BytePairEncoding_NS::BytePairEncodingTokenizer& BPET = *tokenizer_ptr;
        //BPET.update_vocabulary_from_file("./data/the-verdict.txt");
        log_message(Log_Priority::INFO, "main", std::format("Tokenizer initialized with vocab size {}", BPET.vocab_size()));

        // Define the embedding dimension and vocab size
        size_t vocab_size = BPET.vocab_size();
        size_t emb_dim = 256;

        // Create the embedding Matrix
        Matrix<float> emb({vocab_size, emb_dim});
        // Fill the Matrix with random values
        emb.random(-2.f, 2.f);

        // Create a data loader using a shared ptr to the tokenizer
        DataLoader d(tokenizer_ptr, 4, 8, 4);
        d.ingest("./data/the-verdict.txt");
        const Matrix<size_t>& input_batch = d.next_input();
        log_message(Log_Priority::INFO, "main", std::format("Input batch info: {}", input_batch.to_string()));
        
        const Matrix<size_t>& target_batch = d.next_target();
        log_message(Log_Priority::INFO, "main", std::format("Target batch info: {}", target_batch.to_string()));

        // Try fetching another batch
        const Matrix<size_t>& input_batch2 = d.next_input();
        const Matrix<size_t>& target_batch2 = d.next_target();
        log_message(Log_Priority::INFO, "main", std::format("Input batch: {}\nTarget batch: {}", input_batch2.to_string(), target_batch2.to_string()));

        // Create a Tensor Slice
        Tensor<size_t> desired_vals({4});
        desired_vals.at({0}) = 1;
        desired_vals.at({1}) = 2;
        desired_vals.at({2}) = 3;
        desired_vals.at({3}) = 4;
        TensorSlice<float> ts(emb, {0}, desired_vals, {0, 1});
        log_message(Log_Priority::INFO, "main", std::format("Slice info: {}", ts.info()));

        // Convert the 1-D Slice into a proper Tensor
        Tensor<float> ct = ts.to_tensor();
        log_message(Log_Priority::INFO, "main", std::format("Converted Tensor: {}", ct.info()));

        // Create some ranges to extract from the larget Matrix
        Tensor<size_t> target_ranges({2, 2});
        target_ranges.at({0, 0}) = 15;
        target_ranges.at({0, 1}) = 20;
        target_ranges.at({1, 0}) = 250;
        target_ranges.at({1, 1}) = 255;

        // Create a 2-D Slice
        TensorSlice<float> tdts(emb, {0, 1}, target_ranges, {});
        log_message(Log_Priority::INFO, "main", std::format("Slice info: {}", tdts.info()));

        // Convert the 2-D Slice into a standalone Matrix
        Matrix<float> cm = tdts.to_matrix();
        log_message(Log_Priority::INFO, "main", std::format("Converted Matrix info: {}", cm.to_string()));

        // Create a TensorSlice looking up the embeddings of an input batch
        // Get the first batch from the input
        // Define a Tensor to fetch the index
        Tensor<size_t> idx({});
        TensorSlice<size_t> iids(input_batch, {0, 1}, idx, {});
        log_message(Log_Priority::INFO, "main", std::format("Slice info: {}", iids.info()));
        // Using the iids Tensor, fetch the embedding for all tokens in the batch
        TensorSlice<float> emb_lookup(emb, {0, 1}, iids, {});
        log_message(Log_Priority::INFO, "main", std::format("Embedding projection info: {}", emb_lookup.info()));
        Matrix<float> emb_output = emb_lookup.to_matrix();
        log_message(Log_Priority::INFO, "main", std::format("Embedding output: {}", emb_output.to_string()));


    } catch (const std::exception& e) {

        std::cout << "Exception: " << e.what() << "\n";
        return -1;
    }

    return 0;
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
