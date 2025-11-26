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
 * @version: 2025-11-25
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#include "include/main.hpp"

std::vector<std::string>&& split_input(const std::string& target_str) {

    std::vector<std::string>* split = new std::vector<std::string>();

    std::regex match("([,.:;?_!\"()']|--|\\s)");
    std::sregex_token_iterator iter(target_str.begin(), target_str.end(), match, -1);
    std::sregex_token_iterator end;

    while (iter != end) {
        split->push_back(*iter);
        ++iter;
    }

    return std::move(*split);
}

uint32_t update_tokenizer(const std::vector<std::string>& input, std::map<std::string, uint32_t>& vocab, uint32_t& current_max_token) {

    for (const auto& token : input) {
        if (!vocab.count(token)) {
            vocab[token] = current_max_token;
            ++current_max_token;
        }
    }

    return current_max_token;
}

void tokenize_input(const std::vector<std::string>& input, const std::map<std::string, uint32_t>& vocab, std::vector<uint32_t>& tk) {

    for (const auto& token : input) {
        tk.push_back(vocab.at(token));
    }
}

std::vector<std::string>&& detokenize(const std::map<std::string, uint32_t>& vocab, const std::vector<uint32_t>& tk) {

    std::vector<std::string>* dk = new std::vector<std::string>();
    dk->reserve(tk.size());

    std::vector<std::string> token_to_vocab;
    token_to_vocab.resize(vocab.size());

    for (const auto& [k, v] : vocab) {
        token_to_vocab[v] = k;
    }

    for (const auto& token : tk) {
        dk->push_back(token_to_vocab[token]);
    }

    return std::move(*dk);
}

int main(int argc, char* argv[]) {

    std::ifstream file = std::ifstream("../data/the-verdict.txt", std::ios::in);
    std::string str;
    uint32_t max_token = 0;
    std::map<std::string, uint32_t> vocab = std::map<std::string, uint32_t>();
    std::vector<uint32_t> tk;

    if (file) {
        while (!file.eof()) {
            std::getline(file, str);
            std::vector<std::string> r = split_input(str);
            max_token = update_tokenizer(r, vocab, max_token);
            tokenize_input(r, vocab, tk);
        }
        file.close();
    }

    std::cout << "Tokenized form: ";
    for (const auto& token : tk) {
        std::cout << token << " ";
    }
    std::cout << "\n";

    std::cout << "map size: " << vocab.size() << ". tokens: " << max_token << "\n"; 

    std::vector<std::string> dk = detokenize(vocab, tk);

    std::cout << "Detokenized form: ";
    for (const auto& token : dk) {
        std::cout << token << " ";
    }
    std::cout << "\n";

    return 0;
}
