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
 * @version: 2026-06-16
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

/* Local dependencies */
#include "Log.hpp"

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
 * Convert a text string into a vector of bytes
 * @param s String to convert
 * @returns Returns std::vector<bytes>
 */
std::vector<uint8_t>&& string_to_uint8_t_vector(const std::string& s);

};

#endif
