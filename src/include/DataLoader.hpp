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
 * @version: 2026-07-29
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef DATALOADER_HPP
#define DATALOADER_HPP

/* Standard dependencies */
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

/* Local dependencies */
#include "Log.hpp"
#include "Tensor.hpp"
#include "Tokenizer.hpp"

namespace DataLoader_NS {

using Tokenizer_NS::Tokenizer;
using Tensor_NS::Tensor;
using Tensor_NS::Matrix;

/**
 * Data loader class for reading in text, tokenizing it, and then splitting into defined batches
 */
class DataLoader {
/* Private data elements */
private:
    /**
     * Shared pointer to some implementation of Tokenizer, used for actually tokenizing the input text
     */
    std::shared_ptr<Tokenizer> m_tokenizer = nullptr;

    /**
     * Max length of a sequence returned by the data loader
     */
    size_t m_max_len = 0;

    /**
     * Batch size
     */
    size_t m_batch_size = 0;

    /**
     * Stride used by the data loader
     */
    size_t m_stride = 0;

    /**
     * Vector of tokens ids for the entire input
     */
    std::vector<size_t> m_tokens = std::vector<size_t>();

    /**
     * Index of the next token that should be returned from next_input
     */
    size_t m_next_input_idx = 0;

    /**
     * Index of the next token that should be returned from next_target
     */
    size_t m_next_target_idx = 1;

    /**
     * Tensor representing the input ids
     */
    Matrix<size_t> m_input_ids = Matrix<size_t>({0});

    /**
     * Tensor representing the target ids (i.e input ids + 1)
     */
    Matrix<size_t> m_target_ids = Matrix<size_t>({0});


/* Private functions */
    /**
     * Calculate start and stop indices for populating input_ids and target_ids
     * @param base Either m_next_input_idx or m_next_target_idx
     * @returns Returns a vector of coordinate pairs
     */
    std::vector<std::pair<size_t, size_t>> get_idxs(size_t base) const;

/* Public functions */
public:
    /**
     * Default constructor for DataLoader, accepting a shared_ptr to a Tokenizer
     */
    explicit DataLoader(std::shared_ptr<Tokenizer> tokenizer, size_t max_len, size_t batch_size, size_t stride);

    /**
     * Ingest text from a file, specifying the stride and max length to split the tokenized
     * contents into
     * @param path Path to the text file
     * @param max_len Maximum length to split the text into (e.g. how many tokens per Tensor returned)
     * @param stride How many tokends to advance per iteration
     * @returns True if the text was ingested properly, false otherwise
     */
    bool ingest(const std::string& path);

    /**
     * Get input ids for the next batch
     * @returns Returns a const reference to a Tensor containing the input token ids
     */
    const Matrix<size_t>& next_input();

    /**
     * Get target ids for the next batch
     * @returns Returns a const reference to a Tensor containing the target token ids
     */
    const Matrix<size_t>& next_target();

};

}; // namespace DataLoader_NS

#endif
