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
 * @version: 2026-06-16
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

/* Standard dependencies */
#include <vector>
#include <map>

/* Local dependencies */
#include "include/Log.hpp"
#include "include/BytePairEncoding.hpp"

int main(int argc, char* argv[]) {

    std::string input = BytePairEncoding_NS::text_file_to_string("./data/the-verdict.txt");
    auto bpv = BytePairEncoding_NS::string_to_uint8_t_vector(input);

    std::map<uint8_t, size_t> counts = std::map<uint8_t, size_t>();

    for (uint8_t current_byte : bpv) {
        if (counts.count(current_byte) != 0) {
            counts[current_byte] += 1;
        }
        else {
            counts[current_byte] = 1;
        }
    }

    for (std::map<uint8_t, size_t>::iterator i = counts.begin(); i != counts.end(); ++i) {
        std::cout << "Char: '" << static_cast<char>(i->first) << "'. Count: " << i->second << "\n";
    }

    std::map<uint16_t, size_t> pair_counts = std::map<uint16_t, size_t>();

    for (size_t i = 0; i < bpv.size() - 1; ++i) {

        uint16_t combined_pair = (static_cast<uint16_t>(bpv[i]) << 8) + static_cast<uint16_t>(bpv[i + 1]);

        if (pair_counts.count(combined_pair) != 0) {
            pair_counts[combined_pair] += 1;
        }
        else {
            pair_counts[combined_pair] = 1;
        }

    }

    unsigned mask = (1 << 8) - 1;

    for (std::map<uint16_t, size_t>::iterator i = pair_counts.begin(); i != pair_counts.end(); ++i) {

        char second_char = static_cast<char>(i->first & mask);
        char first_char = static_cast<char>(i->first >> 8);

        std::cout << "BytePair '" << first_char << second_char << "'. Count: " << i->second << "\n";
    }

    return 0;
}
