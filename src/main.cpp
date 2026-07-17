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
 * @version: 2026-07-16
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

int main() {

    try {

        std::cout << "Initializing tokenizer...\n";
        BytePairEncoding_NS::BytePairEncodingTokenizer BPET("./data/the-verdict.txt");

        std::cout << "Tokenizing...\n";
        std::string test_string = "The gentle breeze whispered through the tall, emerald trees while the warm sun illuminated the quiet meadow. Rabbits darted playfully between colorful wildflowers, their tiny ears twitching at every sudden noise. High above, a solitary eagle soared effortlessly against the vast, cloudless blue sky, searching for its next meal. Down below, a crystal-clear stream bubbled joyfully as it carved a winding path through smooth, ancient stones. Nature thrived in perfect harmony, untouched by the chaotic rush of the modern world. Every leaf and petal danced in a silent celebration, welcoming the serene beauty that each passing moment so freely offered.";
        std::vector<size_t> tokenized = BPET.tokenize(test_string);
        std::cout << "Token ids: ";
        for (const auto& tkid : tokenized) {
            std::cout << tkid << " ";
        }
        std::cout << "\n";
        std::cout << std::format("Detokenized string: {}\n", BPET.detokenize_to_string(tokenized));
        std::cout << std::format("Compressed {} bytes into {} tokens\n", test_string.size(), tokenized.size());
        
    } catch (const std::exception& e) {

        std::cout << "Exception: " << e.what() << "\n";
        return -1;
    }

    return 0;
}
