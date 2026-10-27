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
 * @version: 2026-10-24
 *
 * General Notes:
 *
 * CreateBPETModel.cpp: Utility to create a new BPE tokenizer model from test data
 */

/* Standard dependencies */
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>

/* Local dependencies */
#include "BytePairEncoding.hpp"
#include "Log.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers, cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
int main() {

    /* Use BPE */
    using BytePairEncoding_NS::BytePairEncodingTokenizer;

    /* Use logging */
    using Log_NS::Log_Priority;
    using Log_NS::log_message;

    try {
        log_message(Log_Priority::INFO, "main", "Initializing new BPE Tokenizer");
        std::string path = "models/BPE3.model";
        BytePairEncodingTokenizer BPET;
        // Ingest new data into the tokenizer
        BPET.update_vocabulary_from_file("./data/the-verdict.txt");
        BPET.update_vocabulary_from_file("./data/en_random_0.txt");
        BPET.update_vocabulary_from_file("./data/zh_random_0.txt");
        // Write the model
        BPET.save_to_file(path);
        log_message(Log_Priority::INFO, "main", std::format("Finished writing BPE model to {}. Vocab size: {}", path, BPET.vocab_size()));

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return -1;
    }

    return 0;
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers, cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)