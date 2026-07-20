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
 * @version: 2026-07-19
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

int main() {

    try {

        /*std::cout << "Initializing tokenizer...\n";
        BytePairEncoding_NS::BytePairEncodingTokenizer BPET("./data/the-verdict.txt");

        std::string test_string = "First some numbers: 132497, 234, 2398342, 247474, and of course 88888888. 然後我想吃飯喔。 The gentle breeze whispered through the tall, emerald trees while the warm sun illuminated the quiet meadow. Rabbits darted playfully between colorful wildflowers, their tiny ears twitching at every sudden noise. High above, a solitary eagle soared effortlessly against the vast, cloudless blue sky, searching for its next meal. Down below, a crystal-clear stream bubbled joyfully as it carved a winding path through smooth, ancient stones. Nature thrived in perfect harmony, untouched by the chaotic rush of the modern world. Every leaf and petal danced in a silent celebration, welcoming the serene beauty that each passing moment so freely offered.";
        std::cout << "\nOriginal text: " << test_string << "\n";
        std::cout << "\nTokenizing...\n";
        std::vector<size_t> tokenized = BPET.tokenize(test_string);
        std::cout << "Token ids: ";
        for (const auto& tkid : tokenized) {
            std::cout << tkid << " ";
        }
        std::cout << "\n";
        std::cout << std::format("\nDetokenized string: {}\n", BPET.detokenize_to_string(tokenized));
        std::cout << std::format("\nCompressed {} bytes into {} tokens\n", test_string.size(), tokenized.size());*/

        Tensor_NS::Tensor<int> it({3, 3, 3});
        it.at({0,1,2}) = 33;
        std::cout << sizeof(it) << "\n";
        std::cout << std::format("Value at [0, 0, 0]: {}\n", it.at({0,0,0}));

        auto Q = it;
        Q.at({1,1,1}) = 12;
        Q.at({0,1,2}) = 99;
        std::cout << std::format("Ref in Q: [1,1,1]: {}\n", Q.at({1,1,1}));

        auto Z = it + Q;
        std::cout << std::format("Val for it + Q [0,1,2]: {}\n", Z.at({0,1,2}));
        std::cout << std::format("Val for it [0,1,2]: {}\n", it.at({0,1,2}));
        std::cout << std::format("Val for Q [0,1,2]: {}\n", Q.at({0,1,2}));

        Z -= Q;
        std::cout << std::format("Val for it + Q [0,1,2]: {}\n", Z.at({0,1,2}));

        auto M = Z * Z;
        std::cout << std::format("Val for Z * Z [0,1,2]: {}\n", M.at({0,1,2}));
        
    } catch (const std::exception& e) {

        std::cout << "Exception: " << e.what() << "\n";
        return -1;
    }

    return 0;
}
