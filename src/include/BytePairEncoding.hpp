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

#ifndef BYTEPAIRENCODING_HPP
#define BYTEPAIRENCODING_HPP

/* Standard dependencies */
#include <algorithm>
#include <bit>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

/* Local dependencies */
#include "Log.hpp"

namespace BytePairEncoding_NS {

// Have the mask used to extract the first char from a uint32_t stored for use
inline constexpr unsigned MAX_FIRST_BYTE_VAL = (1 << 8) - 1;
inline constexpr unsigned MAX_SECOND_BYTE_VAL = (1 << 16) - 1;
inline constexpr unsigned MAX_THIRD_BYTE_VAL = (1 << 24) - 1;
// Bits to shift to access specific characters pac
inline constexpr unsigned SECOND_BYTE_SHIFT = 8;
inline constexpr unsigned THIRD_BYTE_SHIFT = 16;
inline constexpr unsigned FOURTH_BYTE_SHIFT = 24;
// Mask to extract a byte after bit shift
inline constexpr unsigned BYTE_MASK = (1 << 8) - 1;
// The maximum possible vocabulary size
inline constexpr unsigned MAX_VOCAB_SIZE = 65535;

/**
 * Object for storing information about a specific byte pair, including its contents
 * and list of positions of its occurrence within a specific string / vector
 */
class BytePositionInfo {
    /* Private data elements */
private:
    /**
     * The target byte sequence
     */
    uint32_t m_byte_sequence = 0;

    /**
     * A vector containing the positions of the target byte pair throughout a string
     * or vector of characters.
     */
    std::vector<std::pair<size_t, size_t>> m_positions = std::vector<std::pair<size_t, size_t>>();

/* Public functions */
public:
    /**
     * Default constructor for BytePositionInfo
     * @returns Returns a new instance of BytePositionInfo
     */
    BytePositionInfo();

    /**
     * Constructor for BytePositionInfo, accepting the actual byte sequence we want
     * packed inside of ByteSequence and the
     * std::pair<size_t, size_t> for its occurrence within a vector
     * @param byte_pair The byte pair to store info about
     * @param pos_0 Position of the first byte in the byte pair
     * @param pos_1 Position of the second byte in the byte pair
     */
    BytePositionInfo(uint32_t byte_sequence, size_t pos_0, size_t pos_1);

    /**
     * Copy constructor for BytePositionInfo, creating a deep copy
     * @param target Const reference to the BytePositionInfo instance we want to copy
     */
    BytePositionInfo(const BytePositionInfo& target);

    /**
     * Copy assignment operator for BytePositionInfo
     * @param target Const reference for the BytePositionInfo instance we want to copy
     */
    BytePositionInfo& operator=(const BytePositionInfo& target);

    /**
     * Move constructor for BytePositionInfo
     * @param target Reference to the BytePositionInfo instance we want to move
     */
    BytePositionInfo(BytePositionInfo&& target) noexcept;

    /**
     * Move assignment operator for BytePositionInfo
     * @param target Reference to the BytePositionInfo instance we want to move
     */
    BytePositionInfo& operator=(BytePositionInfo&& target) noexcept;

    /**
     * Default destructor for BytePositionInfo
     */
    ~BytePositionInfo();

    /**
     * Get the sequence of bytes
     * @returns Returns a const reference to the underlying std::vector containing
     * the byte sequence
     */
    uint32_t get_byte_sequence() const;

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
     * @param pos_0 Position of the first byte in the byte pair
     * @param pos_1 Position of the second byte in the byte pair
     */
    void add_position(size_t pos_0, size_t pos_1);
};

/**
 * Tokenizer for Byte Pair Encoding (BPE)
 */
class BytePairEncodingTokenizer {
/* Private data elements */
private:
    /**
     * The vocabulary known by the BPE tokenizer, represented by the byte pair uint16_t token
     * and a corresponding size_t token id.
     */
    std::unordered_map<uint32_t, size_t> m_vocab = std::unordered_map<uint32_t, size_t>();
   
    /**
     * The lookup table, using the size_t token id to represent the uint16_t byte pair
     */
    std::vector<uint32_t> m_token_ids = std::vector<uint32_t>();

    /**
     * The size of the vocabulary
     */
    size_t m_vocab_size = 0;

    /**
     * Check to see if a uint16_t token is known
     * @param t Token to check for
     * @returns Returns true if it is known, false otherwise
     */
    bool known(uint32_t t) const;

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
    size_t add_token(uint32_t t);

/* Public functions */
public:
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
    explicit BytePairEncodingTokenizer(const std::string& path);

    /**
     * Get the vocab size from the tokenizer
     * @returns Returns the size of the vocabulary
     */
    size_t vocab_size() const;

    /**
     * Get the list of tokens by token id
     * @returns Returns a const vector of tokens, with the index representing the token id
     */
    const std::vector<uint32_t>& token_ids() const;

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
    std::vector<size_t> tokenize(const std::string& s) const;

    /**
     * Detokenize an input, returning a vector of char containing the decoded text
     * @param v A vector of token ids to detokenize
     * @returns Returns a string containing the detokenized output
     */
    std::string detokenize_to_string(const std::vector<size_t>& v) const;
};

/**
 * Read a text file and convert its lines into a single string
 * @param path Path to the text file
 * @returns Returns a string
 */
std::string text_file_to_string(const std::string& path);

/**
 * Convert a text string into a vector of bytes
 * @param s String to convert
 * @returns Returns std::vector<std::byte>
 */
std::vector<std::byte> string_to_byte_vector(const std::string& s);

/**
 * Convert a vector of bytes into a string
 * @param v Const reference to std::vector<std::byte> to convert
 * @returns Returns a string representing the bytes
 */
std::string byte_vector_to_string(const std::vector<std::byte>& v);

/**
 * Convert a vector of token ids to a space-separated string of them
 * @param v Vector containing the token ids
 * @returns Returns a string of space-separated token ids
 */
std::string token_vector_to_string(const std::vector<size_t>& v);

/**
 * Get the number of bytes packed into a uint32_t
 * @param bs Byte sequence packed into uint32_t
 * @returns Returns the number of bytes packed
 */
size_t get_num_packed_bytes(uint32_t bs);

/**
 * Pack a sequence of bytes into a ByteSequence
 * @param sp Span of std::vector<std::byte> containing the bytes to pack
 * @returns Returns the packed uint32_t
 */
uint32_t pack_bytes(const std::span<const std::byte>& sp);

/**
 * Pack a sequence of uint32_t further into a ByteSequence
 * @param sp Span of std::vector<uint32_t> containing the bytes / ByteSequences to pack
 * the span should not have more than 2 elements, to prevent complex checking
 * for exceeding the 4 byte limit packed into uint32_t
 * @returns Returns a packed uint32_t containing the bytes / ByteSequences
 */
uint32_t pack_bytes(const std::span<const uint32_t>& sp);

/**
 * Unpack a uint32_t into a std::vector<std::byte> instance
 * @param bs Byte sequence to unpack
 * @returns Returns std::vector<std::byte> containing the individual bytes in sequence
 */
std::vector<std::byte> unpack_bytes(uint32_t bs);

/**
 * Create merge rules and generate tokens to be consumed by BytePairEncoding
 * @param v Const reference to std::vector<std::byte> representing the input string
 * @return Returns a vector of tokens, sorted by the number of occurrences
 */
std::vector<uint32_t> create_tokens(const std::vector<std::byte>& v);

}; // namespace BytePairEncoding_NS

#endif
