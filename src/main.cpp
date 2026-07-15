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
 * @version: 2026-07-14
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

        BytePairEncoding_NS::BytePairEncodingTokenizer BPE = BytePairEncoding_NS::BytePairEncodingTokenizer("./data/the-verdict.txt");
        const auto& tkid = BPE.get_vocab();

        for (size_t i = BytePairEncoding_NS::MAX_FIRST_CHAR_VAL + 1; i < tkid.size(); ++i) {
            auto [c1, c2] = BytePairEncoding_NS::uint16_t_to_char_pair(tkid.at(i));
            std::cout << std::format("Token id [{}]: {}{}\n", i, c1, c2);
        }

        const auto& tkzd = BPE.tokenize("hug pug pun bun hugs pug pug pug pug hugs hug pug. Pugs are hugs and that's why they b*rk too!");
        std::string tkstr = BytePairEncoding_NS::token_vector_to_string(tkzd);

        std::cout << "Tokenized ids: " << tkstr << "\n";

        std::string dk = BPE.detokenize_to_string(tkzd);

        std::cout << "Detokenized contents: " << dk << "\n";

        size_t num_bp = 0;
        for (size_t token : tkzd) {
            if (token > BytePairEncoding_NS::MAX_FIRST_CHAR_VAL) {
                num_bp++;
            }
        }

        std::cout << std::format("Stats: number of tokens covered by BPE: {}/{}: {}%\n", num_bp, tkzd.size(), 
            (static_cast<float>(num_bp) / static_cast<float>(tkzd.size())) * 100.f);
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << "\n";
        return -1;
    }

    return 0;
}
