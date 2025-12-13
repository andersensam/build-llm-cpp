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

#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

/* Standard dependencies */
#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <regex>
#include <map>

/* Local dependencies */
#include "Log.hpp"

namespace Tokenizer_NS {

class Tokenizer {
private:
    /* Private data elements */

    /** The uint32_t representation of the newest token (2..n), with n=0 being <|unk|>, n=1 is <|endoftext|> */
    uint32_t m_max_token = 2;

    /** Map of strings to uint32_t tokens */
    std::map<std::string, uint32_t> m_vocab = std::map<std::string, uint32_t>();

    /** Vector of pointers to strings indexed by their uint32_t token values */
    std::vector<const std::string*> m_tokens;

    /* Private functions */

    /**
     * Maybe update the Tokenizer with new vocabulary if it is detected
     * @param input Vector of strings to read from
     * @returns Returns true if the Tokenizer was updated, false if no new tokens were detected
     */
    bool maybe_update_tokenizer(const std::vector<std::string>& input);

    /**
     * Create or update the reverse mapping from the uint32_t tokens to strings
     * @return Returns true if the mapping was updated, false if no changes were made
     */
    bool create_or_update_mapping(void);

    /**
     * Check to see if a particular string corresponds to a valid token id
     * @param str String to check
     * @returns True if a valid token id exists, false otherwise
     */
    bool known(const std::string& str);

    /**
     * Check to see if a particular uint32_t corresponds to a string
     * @param id uint32_t token id to check for
     * @returns True if yes, false otherwise
     */
    bool known(const uint32_t& id);

public:
    /* Public functions */

    /**
     * Constructor for Tokenizer
     * @param path String containing path to the initial vocab to load
     */
    Tokenizer(const std::string& path);

    /**
     * Deconstructor for Tokenizer, ensuring we deallocate m_tokens
     */
    ~Tokenizer();

    /**
     * Read in new text from a file and update the Tokenizer with new vocab
     * @param path String containing the path to read from
     * @returns True if additional vocab is found, false otherwise
     */
    bool add_vocab_from_text_file(const std::string& path);

    /**
     * Tokenize an input and return a vector of uint32_t representing the input
     * @param input Vector of strings to tokenize
     * @returns Returns a vector of uint32_t tokens
     */
    std::vector<uint32_t>&& tokenize(const std::vector<std::string>& input);

    /**
     * Detokenize an input and return a vector of strings
     * @param input Vector of uint32_t tokens
     * @returns Returns a vector of strings
     */
    std::vector<const std::string*>&& detokenize(const std::vector<uint32_t>& input);

};

/**
 * Split an input into smaller string chunks
 * @param target_str The string to split
 * @returns Returns a vector of strings containing the splits
 */
std::vector<std::string>&& split_input(const std::string& target_str);

};

#endif
