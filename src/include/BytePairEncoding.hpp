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
 * @version: 2026-06-18
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

class BytePairEncoding {
private:
    /* Private data elements */

public:
    /* Public functions */
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

/** Convert a vector of strings into a vector of uint8_t
 * @param v Vector of strings to convert
 * @returns Returns std::vector<uint8_t>
 */
std::vector<uint8_t>&& string_vector_to_uint8_t_vector(const std::vector<std::string>& v);

/** Unpack two chars from a single uint16_t 
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

};

#endif
