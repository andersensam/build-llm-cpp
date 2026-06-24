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
 * @version: 2026-06-18
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
#include "include/Tokenizer.hpp"
#include "include/BytePairEncoding.hpp"

int main(int argc, char* argv[]) {

    std::string input = BytePairEncoding_NS::text_file_to_string("./data/the-verdict.txt");
    //auto bpv = BytePairEncoding_NS::string_to_uint8_t_vector("hug pug pun bun hugs");
    //auto ss = Tokenizer_NS::split_input("hug pug pun bun hugs pug pug pug pug hugs hug pug");
    //auto bpv = BytePairEncoding_NS::string_vector_to_uint8_t_vector(ss);
    auto bpv = BytePairEncoding_NS::string_to_uint8_t_vector("hug pug pun bun hugs pug pug pug pug hugs hug pug.");

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

    for (std::map<uint16_t, size_t>::iterator i = pair_counts.begin(); i != pair_counts.end(); ++i) {

        auto [first_char, second_char] = BytePairEncoding_NS::uint16_t_to_char_pair(i->first);

        std::cout << "BytePair '" << first_char << second_char << "'. Count: " << i->second << "\n";
    }

    auto [fc, sc] = BytePairEncoding_NS::uint16_t_to_char_pair(BytePairEncoding_NS::search_top_byte_pair(bpv));
    std::cout << "Most frequent pair: " << fc << sc << "\n";

    BytePairEncoding_NS::BytePositionInfo bpi = BytePairEncoding_NS::get_top_byte_pair(bpv);
    auto [ffc, ssc] = BytePairEncoding_NS::uint16_t_to_char_pair(bpi.get_byte_pair());
    auto pv = bpi.get_positions();

    std::cout << "Per new tool, top byte pair: " << ffc << ssc << " with positions: ";

    for (auto [p1, p2] : pv) {
        std::cout << std::format("[{},{}]", p1, p2) << ", ";
    }

    std::cout << "\n";

    BytePairEncoding_NS::BytePairEncodingTokenizer BPE = BytePairEncoding_NS::BytePairEncodingTokenizer("./data/the-verdict.txt");
    //bool u = BPE.update_vocabulary("hug pug pun bun hugs pug pug pug pug hugs hug pug.");
    auto tkid = BPE.get_vocab();

    //std::cout << "Updated vocab: " << u << "\n";

    for (size_t i = 256; i < tkid.size(); ++i) {
        auto [c1, c2] = BytePairEncoding_NS::uint16_t_to_char_pair(tkid[i]);
        std::cout << std::format("Token id [{}]: {}{}\n", i, c1, c2);
    }

    auto tkzd = BPE.tokenize("hug pug pun bun hugs pug pug pug pug hugs hug pug. Pugs are hugs and that's why they b*rk too!");
    std::string tkstr = BytePairEncoding_NS::token_vector_to_string(tkzd);

    std::cout << "Tokenized ids: " << tkstr << "\n";

    std::string dk = BPE.detokenize_to_string(tkzd);

    std::cout << "Detokenized contents: " << dk << "\n";

    size_t num_bp = 0;
    for (size_t token : tkzd) {
        if (token > 255) {
            num_bp++;
        }
    }

    std::cout << std::format("Stats: number of tokens covered by BPE: {}/{}: {}%\n", num_bp, tkzd.size(), ((float)num_bp / (float)tkzd.size()) * 100.f);

    return 0;
}
