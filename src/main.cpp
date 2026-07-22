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
 * @version: 2026-07-21
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

/* Standard dependencies */
#include <vector>
#include <map>
#include <exception>
#include <iostream>
#include <format>
#include <string>

/* Local dependencies */
#include "include/Log.hpp"
#include "include/BytePairEncoding.hpp"
#include "include/Tensor.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
int main() {

    using Log::Log_Priority;
    using Log::log_message;
    using Tensor_NS::Tensor;
    using Tensor_NS::Matrix;

    try {
        
        log_message(Log_Priority::INFO, "main", "Initializing tokenizer");
        BytePairEncoding_NS::BytePairEncodingTokenizer BPET("./data/the-verdict.txt");
        log_message(Log_Priority::INFO, "main", std::format("Tokenizer initialized with vocab size {}", BPET.vocab_size()));

        // Define the embedding dimension and vocab size
        size_t vocab_size = BPET.vocab_size();
        size_t emb_dim = 256;

        // Create the embedding Matrix
        Matrix<float> emb({vocab_size, emb_dim});
        // Fill the Matrix with random values
        emb.random(-2.f, 2.f);

    } catch (const std::exception& e) {

        std::cout << "Exception: " << e.what() << "\n";
        return -1;
    }

    return 0;
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
