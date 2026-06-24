/*  ________   ___   __    ______   ______   ______    ______   ______   ___   __    ______   ________   ___ __ __     
 * /_______/\ /__/\ /__/\ /_____/\ /_____/\ /_____/\  /_____/\ /_____/\ /__/\ /__/\ /_____/\ /_______/\ /__//_//_/\    
 * \::: _  \ \\::\_\\  \ \\:::_ \ \\::::_\/_\:::_ \ \ \::::_\/_\::::_\/_\::\_\\  \ \\::::_\/_\::: _  \ \\::\| \| \ \   
 *  \::(_)  \ \\:. `-\  \ \\:\ \ \ \\:\/___/\\:(_) ) )_\:\/___/\\:\/___/\\:. `-\  \ \\:\/___/\\::(_)  \ \\:.      \ \  
 *   \:: __  \ \\:. _    \ \\:\ \ \ \\::___\/_\: __ `\ \\_::._\:\\::___\/_\:. _    \ \\_::._\:\\:: __  \ \\:.\-/\  \ \ 
 *    \:.\ \  \ \\. \`-\  \ \\:\/.:| |\:\____/\\ \ `\ \ \ /____\:\\:\____/\\. \`-\  \ \ /____\:\\:.\ \  \ \\. \  \  \ \
 *     \__\/\__\/ \__\/ \__\/ \____/_/ \_____\/ \_\/ \_\/ \_____\/ \_____\/ \__\/ \__\/ \_____\/ \__\/\__\/ \__\/ \__\/    
 *                                                                                                               
 * Project: Lange Language Model in C++
 * @author : Samuel Andersen
 * @version: 2026-06-22
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef BYTEPAIRENCODING_HPP
#define BYTEPAIRENCODING_HPP

/* Standard dependencies */
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <span>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>
#include <cstddef>
#include <functional>

/* Local dependencies */
#include "Log.hpp"
#include "Tokenizer.hpp"

// Have the mask used to extract the first char from a uint16_t stored for use
inline constexpr unsigned FIRST_CHAR_MASK = (1 << 8) - 1;

namespace BytePairEncoding_NS {

/**
 * Object for storing information about a specific byte pair, including its contents
 * and list of positions of its occurrence within a specific string / vector
 */
class BytePositionInfo {
private:
    /* Private data elements */
    /**
     * The target bytep air
     */
    uint16_t m_byte_pair = 0;

    /**
     * A vector containing the positions of the target byte pair throughout a string
     * or vector of characters.
     */
    std::vector<std::pair<size_t, size_t>> m_positions = std::vector<std::pair<size_t, size_t>>();

public:
    /* Public functions */
    /**
     * Default constructor for BytePositionInfo, only used when making clones / deep copies of 
     * an existing BytePositionInfo instance
     * @returns Returns a new instance of BytePositionInfo
     */
    BytePositionInfo();

    /**
     * Initializer for BytePositionInfo, accepting the actual byte pair we want and a
     * std::pair<size_t, size_t> for its occurrence within a vector
     * @param byte_pair The byte pair to store info about
     * @param position std::pair<size_t, size_t> containing the occurrence
     */
    BytePositionInfo(uint16_t byte_pair, std::pair<size_t, size_t> position);

    /**
     * Get the byte pair in question
     * @returns Returns a uint16_t containing the byte pair itself
     */
    uint16_t get_byte_pair() const;

    /**
     * Get the list of occurrences of the byte pair
     * @returns Returns a std::vector reference to the occurrence list
     */
    const std::vector<std::pair<size_t, size_t>>& get_positions() const;

    /**
     * Get the frequency of a byte pair
     * @returns Returns the frequency of the byte pair in the source
     */
    size_t get_frequency() const;

    /**
     * Add a position pair to the list of occurrences
     * @param location std::pair<size_t, size_t> with the occurrence
     */
    void add_position(std::pair<size_t, size_t> location);

    /**
     * Create a deep copy of a BytePositionInfo instance
     * @returns Returns a new instance of BytePositionInfo, containing the data of the caller
     */
    BytePositionInfo&& clone() const;
};

class BytePairEncodingTokenizer {
private:
    /* Private data elements */
    /**
     * The vocabulary known by the BPE tokenizer, represented by the byte pair uint16_t token
     * and a corresponding size_t token id.
     */
    std::map<uint16_t, size_t> m_vocab = std::map<uint16_t, size_t>();
    
    /**
     * The lookup table, using the size_t token id to represent the uint16_t byte pair
     */
    std::vector<uint16_t> m_token_ids = std::vector<uint16_t>();

    /**
     * The size of the vocabulary
     */
    size_t m_vocab_size = 0;

    /**
     * Check to see if a uint16_t token is known
     * @param t Token to check for
     * @returns Returns true if it is known, false otherwise
     */
    bool known(uint16_t t) const;

    /**
     * Check to see if a token id is known
     * @param id Token id to check for
     * @returns Returns true if it is a valid token id, false otherwise
     */
    bool known(size_t id) const;

    /**
     * Add a new token to the vocabulary
     * @param t Token to add
     * @returns Returns the token id for the new token
     */
    size_t add_token(uint16_t t);

public:
    /* Public functions */
    /**
     * Default constructor for the BPE tokenizer
     * @returns Returns a new instance of the BPE tokenizer
     */
    BytePairEncodingTokenizer();

    /**
     * Construct an instance of the BPE tokenizer, basing its vocabulary on a text file
     * @param path Path to a text file to read in and create vocabulary from
     * @returns Returns a new instance of the BPE tokenizer with vocabulary
     */
    BytePairEncodingTokenizer(const std::string& path);

    /**
     * Get the vocab size from the tokenizer
     * @returns Returns the size of the vocabulary
     */
    size_t get_vocab_size() const;

    /**
     * Get the list of tokens by token id
     * @returns Returns a const vector of tokens, with the index representing the token id
     */
    const std::vector<uint16_t>& get_vocab() const;

    /**
     * Add new vocabulary to the BPE tokenizer
     * @param s A string to create new vocabulary from
     * @returns Returns true if additional vocabulary was created, false if all tokens already exist in the tokenizer
     */
    bool update_vocabulary(const std::string& s);

    /**
     * Tokenize an input, returning a vector of token ids
     * @param s A string to tokenize
     * @returns Returns std::vector<size_t> of token ids
     */
    std::vector<size_t>&& tokenize(const std::string& s) const;

    /**
     * Detokenize an input, returning a vector of char containing the decoded text
     * @param v A vector of token ids to detokenize
     * @returns Returns a string containing the detokenized output
     */
    std::string&& detokenize_to_string(const std::vector<size_t>& v) const;
};

/**
 * Read a text file and convert its lines into a single string
 * @param path Path to the text file
 * @returns Returns a string
 */
std::string&& text_file_to_string(const std::string& path);

/**
 * Convert a text string into a vector of uint8_t
 * @param s String to convert
 * @returns Returns std::vector<uint8_t>
 */
std::vector<uint8_t>&& string_to_uint8_t_vector(const std::string& s);

/** 
 * Convert a vector of strings into a vector of uint8_t
 * @param v Vector of strings to convert
 * @returns Returns std::vector<uint8_t>
 */
std::vector<uint8_t>&& string_vector_to_uint8_t_vector(const std::vector<std::string>& v);

/**
 * Convert a vector of token ids to a space-separated string of them
 * @param v Vector containing the token ids
 * @returns Returns a string of space-separated token ids
 */
std::string&& token_vector_to_string(const std::vector<size_t>& v);

/**
 * Pack two uint8_t (char) into a single uint16_t
 * @param c1 The first character
 * @param c2 The second character
 * @returns Returns a single uint16_t with both chars inside
 */
uint16_t chars_to_uint16_t(uint8_t c1, uint8_t c2);

/** 
 * Unpack two chars from a single uint16_t 
 * @param char_pair uint16_t containing two uint8_t (char)
 * @returns Returns std::pair<char, char>
 */
std::pair<char, char> uint16_t_to_char_pair(const uint16_t& char_pair);

/**
 * Scan through a vector of uint8_t and find the byte pair with the highest frequency
 * @param v Vector of uint8_t to analyze
 * @returns Returns a uint16_t containing the two characters making up the byte pair
 */
uint16_t search_top_byte_pair(const std::vector<uint8_t>& v);

/**
 * Scan through a vector of uint8_t, finding the byte pair with the highest frequency
 * and a list of indicies within the vector for where they occur
 * @param v Vector of uint8_t to analyze
 * @returns Returns a BytePositionInfo instance containing the byte pair and its occurrences
 */
BytePositionInfo&& get_top_byte_pair(const std::vector<uint8_t>& v);

/**
 * Remove a pair of indices from a vector
 * @param v Vector to remove the indicies from
 * @param p Pair of indices to remove from the vector
 */
void remove_pair_from_vector(std::vector<uint8_t>& v, const std::pair<size_t, size_t>& p);

};

#endif
