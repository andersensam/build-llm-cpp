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
 * @version: 2026-08-05
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
#include "include/TensorSlice.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
int main() {

    using Log::Log_Priority;
    using Log::log_message;

    using Tensor_NS::Tensor;
    using Tensor_NS::Matrix;

    using TensorSlice_NS::TensorSlice;
    using TensorSlice_NS::VectorSliceConfig;
    using TensorSlice_NS::MatrixSliceConfig;
    using TensorSlice_NS::SliceOrientation;
    using TensorSlice_NS::IndexType;

    using DataLoader_NS::DataLoader;

    try {
        
        log_message(Log_Priority::INFO, "main", "Initializing tokenizer");
        auto tokenizer_ptr = std::make_shared<BytePairEncoding_NS::BytePairEncodingTokenizer>("./models/BPE.model");
        BytePairEncoding_NS::BytePairEncodingTokenizer& BPET = *tokenizer_ptr;
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
        // Get the first input batch and then target batch (shifted by one token)
        const Matrix<size_t>& input_batch = d.next_input();
        const Matrix<size_t>& target_batch = d.next_target();
        log_message(Log_Priority::INFO, "main", std::format("Input batch info: {}\nTarget batch info: {}", input_batch.to_string(), target_batch.to_string()));

        // Create a VectorSliceConfig
        VectorSliceConfig vsc(0, 1, 1, {2, 5}, SliceOrientation::COLUMN, {});
        TensorSlice<float> emb_slice(emb, vsc);
        std::cout << "Value: " << emb_slice.at({0}) << "\n";
        std::cout << "Value in Tensor: " << emb.at({1, 2}) << "\n";


    } catch (const std::exception& e) {

        std::cout << "Exception: " << e.what() << "\n";
        return -1;
    }

    return 0;
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
