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

#include "include/BytePairEncoding.hpp"

std::string&& BytePairEncoding_NS::text_file_to_string(const std::string& path) {

    // Method taken from: https://stackoverflow.com/a/116177
    std::ifstream ifs(path);
    std::string* ret_val = new std::string(std::istreambuf_iterator<char>{ifs}, {});

    return std::move(*ret_val);
}

std::vector<uint8_t>&& BytePairEncoding_NS::string_to_uint8_t_vector(const std::string& s) {

    std::vector<uint8_t>* ret_val = new std::vector<uint8_t>(s.begin(), s.end());

    return std::move(*ret_val);
}

std::vector<uint8_t>&& BytePairEncoding_NS::string_vector_to_uint8_t_vector(const std::vector<std::string>& v) {

    // Allocate a new vector to store the bytes in
    std::vector<uint8_t>* ret_val = new std::vector<uint8_t>();

    // Calculate the total size of the vector that we need
    size_t target_size = 0;
    for (const std::string& s : v) {
        target_size += s.size();
    }

    // Resize the vector to avoid tons of smaller allocations later
    ret_val->reserve(target_size);

    // Loop over the strings and covert each of the individual characters to their uint8_t representations
    for (const std::string& s : v) {
        for (char c : s) {
            ret_val->push_back(static_cast<uint8_t>(c));
        }
    }

    return std::move(*ret_val);
}

std::pair<char, char> BytePairEncoding_NS::uint16_t_to_char_pair(const uint16_t& char_pair) {

    return std::pair<char, char>(static_cast<char>(char_pair >> 8), static_cast<char>(char_pair & FIRST_CHAR_MASK));
}

uint16_t BytePairEncoding_NS::search_top_byte_pair(const std::vector<uint8_t>& v) {

    // Create a map to store all byte pairs we find across the original vector
    std::map<uint16_t, size_t> all_byte_pairs = std::map<uint16_t, size_t>();

    // Use a standard index iterator so we can grab the next character in the vector
    for (size_t i = 0; i < v.size() - 1; ++i) {

        // Pack the bytes into a single uint16_t
        uint16_t current_pair = (static_cast<uint16_t>(v[i]) << 8) + static_cast<uint16_t>(v[i + 1]);

        // If we've already seen this pair, increment its count
        if (all_byte_pairs.count(current_pair) > 0) {

            all_byte_pairs[current_pair] += 1;
        }
        else {
            all_byte_pairs[current_pair] = 1;
        }
    }

    // Iterate over the pairs in the map and extract the byte pair with the highest occurrence
    uint16_t most_frequent = 0;
    size_t highest_frequency = 0;
    for (const auto [byte_pair, frequency] : all_byte_pairs) {

        if (frequency > highest_frequency) {

            most_frequent = byte_pair;
            highest_frequency = frequency;
        }
    }

    return most_frequent;
}
