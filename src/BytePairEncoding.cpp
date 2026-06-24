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

#include "include/BytePairEncoding.hpp"
using BytePairEncoding_NS::BytePositionInfo;
using BytePairEncoding_NS::BytePairEncodingTokenizer;

BytePositionInfo::BytePositionInfo() {
    // Blank since we already set defaults in the class header
}

BytePositionInfo::BytePositionInfo(uint16_t byte_pair, std::pair<size_t, size_t> position) {

    m_byte_pair = byte_pair;
    m_positions.push_back(position);
}

uint16_t BytePositionInfo::get_byte_pair() const {

    return m_byte_pair;
}

const std::vector<std::pair<size_t, size_t>>& BytePositionInfo::get_positions() const {

    return m_positions;
}

size_t BytePositionInfo::get_frequency() const {

    return m_positions.size();
}

void BytePositionInfo::add_position(std::pair<size_t, size_t> location) {

    m_positions.push_back(location);
}

BytePositionInfo&& BytePositionInfo::clone() const {

    BytePositionInfo* target = new BytePositionInfo();
    target->m_byte_pair = m_byte_pair;
    target->m_positions.resize(m_positions.size());

    for (size_t i = 0; i < m_positions.size(); ++i) {
        target->m_positions[i] = m_positions[i];
    }

    return std::move(*target);
}

BytePairEncodingTokenizer::BytePairEncodingTokenizer() {
    
    for (uint16_t i = 0; i < 256; ++i) {
        m_vocab[i] = static_cast<size_t>(i);
        m_token_ids.push_back(i);
    }

    m_vocab_size = m_token_ids.size();
}

BytePairEncodingTokenizer::BytePairEncodingTokenizer(const std::string& path) {

    // Initialize with the same basics as the default constructor
    for (uint16_t i = 0; i < 256; ++i) {
        m_vocab[i] = static_cast<size_t>(i);
        m_token_ids.push_back(i);
    }

    m_vocab_size = m_token_ids.size();

    // Dump the contents of the text file into a string
    std::string contents = text_file_to_string(path);

    update_vocabulary(contents);
}

bool BytePairEncodingTokenizer::known(uint16_t t) const {

    return (m_vocab.count(t) != 0);
}

bool BytePairEncodingTokenizer::known(size_t id) const {

    return (id <= m_vocab_size);
}

size_t BytePairEncodingTokenizer::add_token(uint16_t t) {

    if (known(t)) {
        return m_vocab[t];
    }

    // Add the new token to the vocabulary
    m_vocab[t] = m_vocab_size;
    m_token_ids.push_back(t);
    m_vocab_size += 1;

    return m_vocab_size - 1;
}

size_t BytePairEncodingTokenizer::get_vocab_size() const {

    return m_vocab_size;
}

const std::vector<uint16_t>& BytePairEncodingTokenizer::get_vocab() const {

    return m_token_ids;
}

bool BytePairEncodingTokenizer::update_vocabulary(const std::string& s) {

    // Keep track of the original vocab size before we do anything
    size_t original_vocab_size = m_vocab_size;

    // Convert the original string to a vector of uint8_t (char)
    std::vector<uint8_t> string_vector = string_to_uint8_t_vector(s);

    // Loop inifitely, handling termination inside the loop itself
    for ( ; ; ) {
        // Get the most popular byte pair in the string vector
        BytePositionInfo bpi = get_top_byte_pair(string_vector);

        // If the most popular pair has a frequency of one, there's no point in adding any more
        // byte pairs to the vocabulary
        if (bpi.get_frequency() <= 1) {
            break;
        }

        // Use the add_token function since passing an already known token does nothing and we will
        // want to delete occurrences either way
        add_token(bpi.get_byte_pair());

        // Get the vector of positions of the byte pair
        const std::vector<std::pair<size_t, size_t>>& pos = bpi.get_positions();

        // Iterate over the positions and delete them from the source vector
        for (size_t i = 0; i < pos.size(); ++i) {
            // Get the position pair from the vector
            const std::pair<size_t, size_t>& p = pos[pos.size() - 1 - i];
            // Use a helper function to delete the positions from the source vector
            remove_pair_from_vector(string_vector, p);
        }
    }

    return (original_vocab_size != m_vocab_size);
}

std::vector<size_t>&& BytePairEncodingTokenizer::tokenize(const std::string& s) const {

    // Create an output vector that we will return later -- we'll build this token by token
    std::vector<size_t>* output = new std::vector<size_t>();

    // Convert the string to uint8_t vector to start processing
    std::vector<uint8_t> input = string_to_uint8_t_vector(s);
    // A check to determine if the next character should be skipped
    bool skip_next_character = false;
    // Construct byte pairs where possible, looking at the vocab to know which should be tokenized
    // and which should be left as-is
    for (size_t i = 0; i < input.size() - 1; ++i) {
        if (skip_next_character) {
            skip_next_character = false;
            continue;
        }
        // Convert the two uint8_t (char) into a uint16_t
        uint16_t token = chars_to_uint16_t(input[i], input[i + 1]);
        // Check if the token is a byte pair (must be >= 256 since single characters are 0-255)
        if (token >= 256 && known(token)) {
            output->push_back(m_vocab.at(token));
            // Since we are pushing back a byte pair, we skip the next token in the loop
            skip_next_character = true;
        }
        else {
            // Since single characters all have token ids mapping to their uint8_t values, we can
            // just cast to size_t and call it good enough
            output->push_back(static_cast<size_t>(input[i]));
        }
    }
    // If the final character of the input was *not* ingested as a byte pair, add it to
    // our output as a single character
    if (!skip_next_character) {
        output->push_back(static_cast<size_t>(input[input.size() - 1]));
    }
    
    return std::move(*output);
}

std::string&& BytePairEncodingTokenizer::detokenize_to_string(const std::vector<size_t>& v) const {

    std::string* output = new std::string();
    // Reserve at least the size of v to avoid tons of allocations
    output->reserve(v.size());

    for (size_t id : v) {
        if (id < 256) {
            (*output) += static_cast<char>(id);
        }
        else {
            auto [c1, c2] = uint16_t_to_char_pair(m_token_ids[id]);
            (*output) += c1;
            (*output) += c2;
        }
    }

    return std::move(*output);
}

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

std::string&& BytePairEncoding_NS::token_vector_to_string(const std::vector<size_t>& v) {

    std::string* output = new std::string();
    output->reserve(v.size());

    for (size_t i = 0; i < v.size() - 1; ++i) {
        (*output) += std::format("{} ", v[i]);
    }
    (*output) += std::format("{}", v[v.size() - 1]);

    return std::move(*output);
}

uint16_t BytePairEncoding_NS::chars_to_uint16_t(uint8_t c1, uint8_t c2) {

    return ((static_cast<uint16_t>(c1) << 8) + static_cast<uint16_t>(c2));
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
        uint16_t current_pair = chars_to_uint16_t(v[i], v[i + 1]);

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

BytePositionInfo&& BytePairEncoding_NS::get_top_byte_pair(const std::vector<uint8_t>& v) {

    // (TODO): handle the case where the vector is either empty or contains a single byte.

    // Create a map to store all byte pairs we find across the original vector
    std::map<uint16_t, size_t> all_byte_pairs = std::map<uint16_t, size_t>();
    // Create a map to store the positions of the byte pairs we find in the vector
    std::map<uint16_t, BytePositionInfo> byte_pair_positions = std::map<uint16_t, BytePositionInfo>();

    // Use a standard index iterator so we can grab the next character in the vector
    for (size_t i = 0; i < v.size() - 1; ++i) {

        // Pack the bytes into a single uint16_t
        uint16_t current_pair = chars_to_uint16_t(v[i], v[i + 1]);

        // If we've already seen this pair, increment its count
        if (all_byte_pairs.count(current_pair) > 0) {

            all_byte_pairs[current_pair] += 1;
            byte_pair_positions[current_pair].add_position(std::pair<size_t, size_t>(i, i + 1));
        }
        // If we are seeing the pair for the first time
        else {
            all_byte_pairs[current_pair] = 1; 
            byte_pair_positions[current_pair] = BytePositionInfo(current_pair, std::pair<size_t, size_t>(i, i + 1));
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

    return byte_pair_positions[most_frequent].clone();
}

void BytePairEncoding_NS::remove_pair_from_vector(std::vector<uint8_t>& v, const std::pair<size_t, size_t>& p) {

    // Unpack position 1 and position 2 from the std::pair
    auto [p1, p2] = p;

    // Use the vector erase function, starting with p2 since it is later than p1
    v.erase(v.begin() + p2);
    v.erase(v.begin() + p1);
}
