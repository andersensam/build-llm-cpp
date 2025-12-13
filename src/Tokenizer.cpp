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
 * @version: 2025-12-12
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#include "include/Tokenizer.hpp"

using Tokenizer_NS::Tokenizer;
using Log::Log_Priority;
using Log::log_message;

bool Tokenizer::maybe_update_tokenizer(const std::vector<std::string>& input) {

    // Track the previous number of tokens to know if we've added additional tokens
    uint32_t prev_max_token = m_max_token;

    // Iterate over the tokens in the input and see if they are known
    for (const auto& token : input) {
        if (!known(token)) {
            // Add a new mapping of string to uint32_t token id and increment the max token id
            m_vocab[token] = m_max_token;
            ++m_max_token;
        }
    }

    if (prev_max_token != m_max_token) {
        return true;
    }
    return false;
}

bool Tokenizer::create_or_update_mapping(void) {

    // Check to see if the existing mapping has the same number of elements
    // as the std::map. If yes, then do nothing
    if (m_vocab.size() == m_tokens.size()) {
        return false;
    }

    // Clear out any existing contents and then resize
    m_tokens.clear();
    m_tokens.resize(m_vocab.size());

    // Use older style iterator to access memory address of std::string
    for (std::map<std::string, uint32_t>::iterator i = m_vocab.begin(); i != m_vocab.end(); ++i) {
        m_tokens[i->second] = &(i->first);
    }

    return true;
}

bool Tokenizer::known(const std::string& str) {

    if (m_vocab.count(str) > 0) {
        return true;
    }
    return false;
}

bool Tokenizer::known(const uint32_t& id) {

    if (id < m_tokens.size()) {
        return true;
    }
    return false;
}

Tokenizer::Tokenizer(const std::string& path) {

    // Ensure we add the unknown token, <|unk|>
    m_vocab["<|unk|>"] = 0;
    // Add the end of text token
    m_vocab["<|endoftext|>"] = 1;

    // Add any new tokens from the provided text file
    add_vocab_from_text_file(path);
}

Tokenizer::~Tokenizer() {

}

bool Tokenizer::add_vocab_from_text_file(const std::string& path) {

    std::ifstream file = std::ifstream(path, std::ios::in);
    bool updated = false;

    if (file) {
        // Create a string to store the incoming line
        std::string str;
        // Iterate over the contents of the file's lines
        while (!file.eof()) {
            std::getline(file, str);
            std::vector<std::string> r = split_input(str);
            updated = maybe_update_tokenizer(r);
        }
        file.close();
    }
    else {
        log_message(Log_Priority::ERROR, "Tokenizer::add_vocab_from_text",
            "Unable to open file for reading");
        return false;
    }

    if (updated) {
        create_or_update_mapping();
    }

    return updated;
}

std::vector<uint32_t>&& Tokenizer::tokenize(const std::vector<std::string>& input) {

    // Allocate a new vector for the results
    std::vector<uint32_t>* tk = new std::vector<uint32_t>();

    // Allocate space all at once to be more efficient than push_back
    tk->resize(input.size());

    for (size_t i = 0; i < input.size(); ++i) {

        if (known(input[i])) {
            // Fetch the corresponding uint32_t representation of the input token
            (*tk)[i] = m_vocab.at(input[i]);
        }
        else {
            // Use the unknown token id, corresponding to <|unk|>
            (*tk)[i] = 0;
        }
    }

    return std::move(*tk);
}

std::vector<const std::string*>&& Tokenizer::detokenize(const std::vector<uint32_t>& input) {

    std::vector<const std::string*>* dk = new std::vector<const std::string*>();

    // Ensure we add the end of text token to the very end
    dk->resize(input.size() + 1);
    (*dk)[input.size()] = m_tokens[1];

    for (size_t i = 0; i < input.size(); ++i) {

        if (known(input[i])) {

            (*dk)[i] = m_tokens[input[i]];
        }
        else {
            // The unknown token lives at index zero
            (*dk)[i] = m_tokens[0];
        }
    }

    return std::move(*dk);
}

std::vector<std::string>&& Tokenizer_NS::split_input(const std::string& target_str) {

    // Allocate a new vector for storing the results
    std::vector<std::string>* split = new std::vector<std::string>();

    // Set up the regex to search for characters that split tokens
    std::regex match("([,.:;?_!\"()'\\s-]+)");

    // Set the regex iterator to get clusters of characters that *do not* match
    std::sregex_token_iterator iter(target_str.begin(), target_str.end(), match, -1);
    // Set another regex iterator to get the split character itself
    std::sregex_token_iterator split_iter(target_str.begin(), target_str.end(), match, 0);
    std::sregex_token_iterator end;

    // Iterate over the matches and add the strings to the result vector
    while (iter != end && split_iter != end) {
        split->push_back(*iter);
        split->push_back(*split_iter);
        ++iter;
        ++split_iter;
    }

    // Return the new vector and allow its lifetime to be managed by the receiver
    return std::move(*split);
}
