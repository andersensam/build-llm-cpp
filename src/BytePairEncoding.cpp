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
 * @version: 2026-07-21
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#include "include/BytePairEncoding.hpp"

using BytePairEncoding_NS::BytePairEncodingTokenizer;
using BytePairEncoding_NS::BytePositionInfo;

BytePositionInfo::BytePositionInfo() {
    // Blank since we already set defaults in the class header
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
BytePositionInfo::BytePositionInfo(uint32_t byte_sequence, size_t pos_0, size_t pos_1) : 
    m_byte_sequence(byte_sequence) {

    m_positions.emplace_back(pos_0, pos_1);
}

BytePositionInfo::BytePositionInfo(const BytePositionInfo& target) : m_byte_sequence(target.m_byte_sequence),
    m_positions(target.m_positions.size()) {

    for (const auto& [p0, p1] : target.m_positions) {
        m_positions.emplace_back(p0, p1);
    }
}

BytePositionInfo& BytePositionInfo::operator=(const BytePositionInfo& target) {

    if (this == &target) {
        return *this;
    }

    this->m_byte_sequence = target.m_byte_sequence;
    this->m_positions = target.m_positions;

    return *this;
}

BytePositionInfo::BytePositionInfo(BytePositionInfo&& target) noexcept : m_byte_sequence(target.m_byte_sequence),
    m_positions(std::move(target.m_positions)) {

    // Blank since moving the vector is enough to initialize
}

BytePositionInfo& BytePositionInfo::operator=(BytePositionInfo&& target) noexcept {

    if (this == &target) {
        return *this;
    }

    this->m_byte_sequence = target.m_byte_sequence;
    this->m_positions = std::move(target.m_positions);

    return *this;
}

BytePositionInfo::~BytePositionInfo() {
    // Blank since we don't need to do anything except allow objects to go out of scope
}

uint32_t BytePositionInfo::get_byte_sequence() const {

    return m_byte_sequence;
}

const std::vector<std::pair<size_t, size_t>>& BytePositionInfo::get_positions() const {

    return m_positions;
}

size_t BytePositionInfo::get_frequency() const {

    return m_positions.size();
}

void BytePositionInfo::add_position(size_t pos_0, size_t pos_1) {

    m_positions.emplace_back(pos_0, pos_1);
}

BytePairEncodingTokenizer::BytePairEncodingTokenizer() {

    m_token_ids.reserve(MAX_FIRST_BYTE_VAL);
    
    for (uint32_t i = 0; i <= MAX_FIRST_BYTE_VAL; ++i) {
        m_vocab[i] = static_cast<size_t>(i);
        m_token_ids.push_back(i);
    }

    m_vocab_size = m_token_ids.size();
}

BytePairEncodingTokenizer::BytePairEncodingTokenizer(const std::string& path) {

    m_token_ids.reserve(MAX_FIRST_BYTE_VAL);

    // Initialize with the same basics as the default constructor
    for (uint32_t i = 0; i <= MAX_FIRST_BYTE_VAL; ++i) {
        m_vocab[i] = static_cast<size_t>(i);
        m_token_ids.push_back(i);
    }

    m_vocab_size = m_token_ids.size();

    // Dump the contents of the text file into a string
    std::string contents = text_file_to_string(path);

    update_vocabulary(contents);
}

bool BytePairEncodingTokenizer::known(uint32_t t) const {

    return (m_vocab.count(t) != 0);
}

bool BytePairEncodingTokenizer::known(size_t id) const {

    return (id <= m_vocab_size);
}

size_t BytePairEncodingTokenizer::add_token(uint32_t t) {

    if (known(t)) {
        return m_vocab[t];
    }

    // Add the new token to the vocabulary
    m_vocab.at(t) = m_vocab_size;
    m_token_ids.push_back(t);
    m_vocab_size += 1;

    return m_vocab_size - 1;
}

size_t BytePairEncodingTokenizer::vocab_size() const {

    return m_vocab_size;
}

const std::vector<uint32_t>& BytePairEncodingTokenizer::token_ids() const {

    return m_token_ids;
}

bool BytePairEncodingTokenizer::update_vocabulary(const std::string& s) {

    // Keep track of the original vocab size before we do anything
    size_t original_vocab_size = m_vocab_size;

    std::vector<uint32_t> tokens = create_tokens(string_to_byte_vector(s));
    // Iterate over the merged tokens and add unknown tokens to the vocabulary
    // making sure to assign the token id as m_vocab_size, then incrementing it
    for (const auto& t : tokens) {
        // Ensure that we are staying under the max vocab size limit
        if (m_vocab.count(t) == 0 && (m_vocab_size + 1 <= MAX_VOCAB_SIZE)) {
            // Create an entry in the map, assigning the uint32_t token to m_vocab_size token id
            m_vocab[t] = m_vocab_size;
            m_token_ids.push_back(t);
            ++m_vocab_size;
        }
    }

    return original_vocab_size != m_vocab_size;
}

std::vector<size_t> BytePairEncodingTokenizer::tokenize(const std::string& s) const {

    // Create an output vector that we will return later -- we'll build this token by token
    std::vector<size_t> output;
    // We know that the minimum size of output is 1/4th the size of the input, s, assuming
    // there are 4-byte tokens for every single byte
    output.reserve(s.size() / 4);
    // Convert the string into a vector of bytes
    std::vector<std::byte> bytes = string_to_byte_vector(s);
    // Create a span to reference the bytes in the input
    std::span<const std::byte> bytes_span{bytes};
    // Iterate over the bytes and start mapping bytes to tokens
    // Notice that we don't increment automatically since we might skip ahead by 1 - 4 bytes at any time
    for (size_t i = 0; i < bytes.size(); ) {
        // Start with attempting to pack 4 bytes and find the matching token. If the token
        // isn't found, try 3 bytes, then 2, then 1.
        for (size_t j = 4; j >= 1; --j) {
            // Ensure that reading more than one byte doesn't go past the end of bytes
            if ((i + j) > bytes.size()) {
                continue;
            }
            uint32_t packed = pack_bytes(bytes_span.subspan(i, j));
            if (m_vocab.count(packed) != 0) {
                output.push_back(m_vocab.at(packed));
                // Ensure that we advance by the number of bytes packed, when there is actually a match
                i += j;
                break;
            }
        }        
    }

    return output;
}

std::string BytePairEncodingTokenizer::detokenize_to_string(const std::vector<size_t>& v) const {

    std::string output;
    // Reserve at least the size of v to avoid tons of allocations
    output.reserve(v.size());

    for (const size_t& tok_id : v) {
        for (const std::byte& b : unpack_bytes(m_token_ids.at(tok_id))) {
            output.push_back(static_cast<char>(b));
        }
    }

    return output;
}

std::string BytePairEncoding_NS::text_file_to_string(const std::string& path) {

    // Method taken from: https://stackoverflow.com/a/116177
    std::ifstream ifs(path);
    std::string ret_val = std::string(std::istreambuf_iterator<char>{ifs}, {});

    return ret_val;
}

std::vector<std::byte> BytePairEncoding_NS::string_to_byte_vector(const std::string& s) {

    // Create a vector to return and reserve at least as many bytes as are chars in the string
    std::vector<std::byte> ret_val;
    ret_val.reserve(s.size());

    // Create a span to access the raw bytes of the string
    for (const auto& b : std::span<const std::byte>{std::as_bytes(std::span(s))}) {
        ret_val.push_back(b);
    }

    return ret_val;
}

std::string BytePairEncoding_NS::byte_vector_to_string(const std::vector<std::byte>& v) {

    std::string ret_val;
    ret_val.reserve(v.size());

    for (const auto& c : v) {
        ret_val += static_cast<char>(c);
    }

    return ret_val;
}

std::string BytePairEncoding_NS::token_vector_to_string(const std::vector<size_t>& v) {

    std::string output;
    output.reserve(v.size());

    for (size_t i = 0; i < v.size() - 1; ++i) {
        output += std::format("{} ", v.at(i));
    }
    output += std::format("{}", v.at(v.size() - 1));

    return output;
}

size_t BytePairEncoding_NS::get_num_packed_bytes(uint32_t bs) {

    // Handle a blank / empty byte sequence
    if (bs == 0) { return 0; }
    else if (bs <= MAX_FIRST_BYTE_VAL) { return 1; }
    else if (bs <= MAX_SECOND_BYTE_VAL) { return 2; }
    else if (bs <= MAX_THIRD_BYTE_VAL) { return 3; }

    // We can store a maximum of 4 bytes in a single uint32_t, so if none of the
    // other cases are true, it must be 4.
    return 4;
}

uint32_t BytePairEncoding_NS::pack_bytes(const std::span<const std::byte>& sp) {

    uint32_t ret_val = 0;

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    switch (sp.size()) {
        case 1:
            ret_val = static_cast<uint32_t>(sp[0]);
            break;
        case 2:
            ret_val = (static_cast<uint32_t>(sp[1]) << SECOND_BYTE_SHIFT) +
                      static_cast<uint32_t>(sp[0]);
            break;
        case 3:
            ret_val = (static_cast<uint32_t>(sp[2]) << THIRD_BYTE_SHIFT) +
                      (static_cast<uint32_t>(sp[1]) << SECOND_BYTE_SHIFT) +
                      static_cast<uint32_t>(sp[0]);
            break;
        case 4:
            ret_val = (static_cast<uint32_t>(sp[3]) << FOURTH_BYTE_SHIFT) +
                      (static_cast<uint32_t>(sp[2]) << THIRD_BYTE_SHIFT) +
                      (static_cast<uint32_t>(sp[1]) << SECOND_BYTE_SHIFT) +
                      static_cast<uint32_t>(sp[0]);
            break;
        default:
            return ret_val;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

    return ret_val;
}

uint32_t BytePairEncoding_NS::pack_bytes(const std::span<const uint32_t>& sp) {

    // We will never have two bytes that combine to be zero, so this is a decent
    // return value to indicate an error has occured
    uint32_t ret_val = 0;

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    switch (sp.size()) {
        // If we only have one value to pack (shouldn't ever be called), just return
        // the value as-is
        case 1:
            ret_val = sp[0];
            break;
        case 2:
            // Ensure we aren't going over the 4-byte limit by merging tokens
            if ((get_num_packed_bytes(sp[0]) + get_num_packed_bytes(sp[1])) > 4) {
                break;
            }
            // Since the bytes are sequenced already, we should pack into the first byte
            // to preserve the order
            switch (get_num_packed_bytes(sp[0])) {
                case 1:
                    // Prefix sp[1] with sp[0]
                    ret_val = (sp[1] << SECOND_BYTE_SHIFT) + sp[0];
                    break;
                case 2:
                    // Shift sp[1] by 16 bits and then prefix with sp[0]
                    ret_val = (sp[1] << THIRD_BYTE_SHIFT) + sp[0];
                    break;
                case 3:
                    // Shift sp[1] by 24 bits and then prefix with sp[0]
                    ret_val = (sp[1] << FOURTH_BYTE_SHIFT) + sp[0];
                    break;
                default:
                    break;
            }
            break;
        default:
            break; 
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

    return ret_val;
}

std::vector<std::byte> BytePairEncoding_NS::unpack_bytes(uint32_t bs) {

    std::vector<std::byte> ret_val(get_num_packed_bytes(bs));

    switch (ret_val.size()) {
        case 1:
            ret_val.at(0) = static_cast<std::byte>(bs);
            break;
        case 2:
            ret_val.at(0) = static_cast<std::byte>(bs & BYTE_MASK);
            ret_val.at(1) = static_cast<std::byte>((bs >> SECOND_BYTE_SHIFT) & BYTE_MASK);
            break;
        case 3:
            ret_val.at(0) = static_cast<std::byte>(bs & BYTE_MASK);
            ret_val.at(1) = static_cast<std::byte>((bs >> SECOND_BYTE_SHIFT) & BYTE_MASK);
            ret_val.at(2) = static_cast<std::byte>((bs >> THIRD_BYTE_SHIFT) & BYTE_MASK);
            break;
        case 4:
            ret_val.at(0) = static_cast<std::byte>(bs & BYTE_MASK);
            ret_val.at(1) = static_cast<std::byte>((bs >> SECOND_BYTE_SHIFT) & BYTE_MASK);
            ret_val.at(2) = static_cast<std::byte>((bs >> THIRD_BYTE_SHIFT) & BYTE_MASK);
            ret_val.at(3) = static_cast<std::byte>((bs >> FOURTH_BYTE_SHIFT) & BYTE_MASK);
            break;
        default:
            break;
    }

    return ret_val;
}

std::vector<uint32_t> BytePairEncoding_NS::create_tokens(const std::vector<std::byte>& v) {

    // If we receive an empty input, return no tokens
    if (v.size() == 0) { return std::vector<uint32_t>(); }

    // Create a temporary vector of uint32_t to store our raw bytes
    std::vector<uint32_t> uv;
    uv.reserve(v.size());
    for (const std::byte& b : v) {
        uv.push_back(static_cast<uint32_t>(b));
    }

    // Start an infinite loop, iterating through the input and merging bytes until either
    // the entire input is comprised of tokens of 4 bytes, or no bytes occur more than once
    for ( ; ; ) {
        // Create a map, tracking packed uint32_t and frequency, plus locations
        std::unordered_map<uint32_t, BytePositionInfo> token_occurrences;
        // Also track the frequency of the packed uint32_t
        std::unordered_map<uint32_t, size_t> token_frequency;
        // Create a reusuable span for viewing the contents of the vector
        std::span<const uint32_t> uv_span{uv};
        // Start at the beginning of the vector but end one element early as we can't
        // merge something past the end of the vector
        for (size_t i = 0; i < uv.size() - 1; ++i) {
            // Merge together eligible bytes / byte sequences
            // Check that a particular uint32_t isn't already full
            size_t pos_0_bytes = get_num_packed_bytes(uv.at(i));
            size_t pos_1_bytes = get_num_packed_bytes(uv.at(i + 1));
            if ((pos_0_bytes < 4) && (pos_1_bytes < 4) && ((pos_0_bytes + pos_1_bytes) <=4)) {
                // Pack the bytes together
                uint32_t packed = pack_bytes(uv_span.subspan(i, 2));
                // If we get a 0 back from pack_bytes, something went wrong, so continue through the
                // iterations onto other bytes instead
                if (packed == 0) {
                    continue;
                }
                // If we are seeing this token for the first time, add it to the map 
                // and create the BytePositionInfo object
                if (token_occurrences.count(packed) == 0) {
                    token_occurrences.try_emplace(packed, packed, i, i + 1);
                    // Also create a key in token_frequency and set to 1
                    token_frequency[packed] = 1;
                }
                else {
                    auto& bpi = token_occurrences.at(packed);
                    // Add the current position to the list inside of BytePositionInfo
                    bpi.add_position(i, i + 1);
                    // Increment the number of occurences
                    token_frequency[packed] += 1;
                }
            }
        }
        // If token_occurences or token_frequency is empty, it means we were not able to merge
        // any tokens and we can break out of the loop
        if (token_occurrences.empty()) {
            break;
        }
        // Once we've merged all bytes possible and tracked their positions, find the most popular
        // token, merge its occurences and remove its fragments from the uv vector
        auto most_frequent_token = std::max_element(token_frequency.begin(), token_frequency.end(),
                                                    [](const auto& v1, const auto& v2) {
                                                        return v1.second < v2.second;
                                                    });
        // Positions should be std::vector<std::pair<size_t, size_t>>
        const auto& positions = token_occurrences.at(most_frequent_token->first).get_positions();
        // Iterate over the positions in reverse order, changing the first position to be the
        // new merged token, then erasing the second. We do this in reverse so that the positions
        // are still valid as we go forward
        size_t pos_size = positions.size() - 1;
        for (size_t i = 0; i <= pos_size; ++i) {
            // Unpack the two size_t positions inside of uv
            const auto& [p0, p1] = positions.at(pos_size - i);
            // Replace p0 with the merged token
            uv.at(p0) = most_frequent_token->first;
            // Erase p1 from the uv vector
            uv.erase(uv.begin() + static_cast<int64_t>(p1));
        }
    }
    // Since individual bytes (0 - 255) are already mandatory tokens, erase any values
    // that are not >= 256 from the uv vector
    std::erase_if(uv, [](uint32_t x) { return x <= MAX_FIRST_BYTE_VAL; });
    // Also remove duplicates from the uv vector, only returning a list of unique
    // tokens for processing
    std::sort(uv.begin(), uv.end());
    uv.erase(std::unique(uv.begin(), uv.end()), uv.end());

    return uv;
}
