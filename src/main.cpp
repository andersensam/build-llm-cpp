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
 * @version: 2026-07-20
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

        Tensor_NS::Tensor<int> a({3, 3, 3});
        a.set({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});

        Tensor_NS::Matrix<int> b({2, 2});
        b.set({1, 3, 5, 7});

        Tensor_NS::Matrix<int> c({2, 2});
        c.fill(5);

        auto qz = b.matmul(c);

        Tensor_NS::Matrix<int> bc({3,3});
        bc.set({1,2,3,4,5,6,7,8,9});

        const auto& dims = bc.dims();
        for (size_t i = 0; i < dims.at(0); ++i) {
            for (size_t j = 0; j < dims.at(1); ++j) {
                std::cout << bc.at({i, j}) << "\t";
            }
            std::cout << "\n";
        }

        bc.transpose();

        const auto& dims2 = bc.dims();
        for (size_t i = 0; i < dims2.at(0); ++i) {
            for (size_t j = 0; j < dims2.at(1); ++j) {
                std::cout << bc.at({i, j}) << "\t";
            }
            std::cout << "\n";
        }

        
    } catch (const std::exception& e) {

        std::cout << "Exception: " << e.what() << "\n";
        return -1;
    }

    return 0;
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
