/* Large Language Model training framework in C++
 * Copyright © 2026 Samuel Andersen
 *  ________   ___   __    ______   ______   ______    ______   ______   ___   __    ______   ________   ___ __ __     
 * /_______/\ /__/\ /__/\ /_____/\ /_____/\ /_____/\  /_____/\ /_____/\ /__/\ /__/\ /_____/\ /_______/\ /__//_//_/\    
 * \::: _  \ \\::\_\\  \ \\:::_ \ \\::::_\/_\:::_ \ \ \::::_\/_\::::_\/_\::\_\\  \ \\::::_\/_\::: _  \ \\::\| \| \ \   
 *  \::(_)  \ \\:. `-\  \ \\:\ \ \ \\:\/___/\\:(_) ) )_\:\/___/\\:\/___/\\:. `-\  \ \\:\/___/\\::(_)  \ \\:.      \ \  
 *   \:: __  \ \\:. _    \ \\:\ \ \ \\::___\/_\: __ `\ \\_::._\:\\::___\/_\:. _    \ \\_::._\:\\:: __  \ \\:.\-/\  \ \ 
 *    \:.\ \  \ \\. \`-\  \ \\:\/.:| |\:\____/\\ \ `\ \ \ /____\:\\:\____/\\. \`-\  \ \ /____\:\\:.\ \  \ \\. \  \  \ \
 *     \__\/\__\/ \__\/ \__\/ \____/_/ \_____\/ \_\/ \_\/ \_____\/ \_____\/ \__\/ \__\/ \_____\/ \__\/\__\/ \__\/ \__\/    
 *                                                                                                               
 * Project: Large Language Model in C++
 * @author : Samuel Andersen
 * @version: 2026-10-12
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

/* Standard dependencies */
#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>

/* Local dependencies */
#include "AbstractTensor.hpp"

namespace Tokenizer_NS {

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;

// NOLINTBEGIN(cppcoreguidelines-special-member-functions)
/**
 * Abstract base class for Tokenizer, which all supported tokenizers must implement
 */
class Tokenizer {
public:
    /**
     * Virtual destructor since this should be handled by the individual implementations
     */
    virtual ~Tokenizer() = default;

    /**
     * Get the vocab size from the tokenizer
     * @returns Returns the size of the vocabulary
     */
    virtual size_t vocab_size() const = 0;

    /**
     * Get the list of tokens by token id
     * @returns Returns a const vector of tokens, with the index representing the token id
     */
    virtual const std::vector<uint32_t>& token_ids() const = 0;

    /**
     * Tokenize an input, returning a vector of token ids
     * @param s A string to tokenize
     * @returns Returns std::vector<size_t> of token ids
     */
    virtual std::vector<size_t> tokenize(const std::string& s) const = 0;

    /**
     * Detokenize an input, returning a string containing the decoded text
     * @param v A vector of token ids to detokenize
     * @returns Returns a string containing the detokenized output
     */
    virtual std::string detokenize_to_string(const std::vector<size_t>& v) const = 0;

    /**
     * Detokenize an input, returning a string containing the decoded text
     * @param t AbstractTensor reference containing token ids. The Tensor must have a shape of [N, 1] or [1, N]
     * @returns Returns a string containing the detokenized output
     */
    virtual std::string detokenize_to_string(const AbstractTensor<size_t>& t) const = 0;
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace Tokenizer_NS

#endif
