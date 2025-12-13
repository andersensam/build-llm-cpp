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
 * @version: 2025-12-03
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#include "include/main.hpp"

using Tokenizer_NS::Tokenizer;

int main(int argc, char* argv[]) {

    Tokenizer t = Tokenizer("../data/the-verdict.txt");

    std::vector<std::string> test_tokenizer = Tokenizer_NS::split_input("It's the last he painted, you know, Mrs. Gisburn said with pardonable pride.");
    auto enc = t.tokenize(test_tokenizer);

    std::cout << "Tokenized contents: ";
    for (const auto tk : enc) {
        std::cout << tk << " ";
    }
    std::cout << "\n";

    auto dec = t.detokenize(enc);

    std::cout << "Detokenized contents: ";
    for (const std::string* tk : dec) {
        std::cout << *tk << "";
    }
    std::cout << "\n";

    return 0;
}
