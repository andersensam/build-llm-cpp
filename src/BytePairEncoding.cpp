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

#include "include/BytePairEncoding.hpp"

std::string&& BytePairEncoding_NS::text_file_to_string(const std::string& path) {

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
