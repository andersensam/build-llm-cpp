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

#include "include/DataLoader.hpp"

using DataLoader_NS::DataLoader;
using Tensor_NS::Tensor;
using Tensor_NS::Matrix;
using Log::Log_Priority;
using Log::log_message;

std::vector<std::pair<size_t, size_t>> DataLoader::get_idxs(size_t base) const {
    // Setup the result vector
    std::vector<std::pair<size_t, size_t>> result;
    // Store the total number of tokens in the dataset to avoid calling m_tokens.size() repeatedly
    const size_t dataset_size = m_tokens.size();
    // Before reserving memory, see if we are already at the end of our dataset
    if (base >= dataset_size) {
        // Return an empty vector
        return result;
    }
    // Handle a case where we reach the end of our dataset within the first item
    if (base + m_max_len >= dataset_size) {
        result.emplace_back(base, dataset_size - 1);
        return result;
    }
    // Reserve memory for the batches to avoid repeated allocations
    result.reserve(m_batch_size);
    // Iterate over the batch size and max length
    for (size_t i = 0; i < m_batch_size; ++i) {
        if (base + m_max_len >= dataset_size) {
            // Skip the batch iteration if we hit the end of the dataset
            continue;
        }
        else {
            result.emplace_back(base, base + m_max_len);
            // Increase the base index by m_stride
            base += m_stride;
        }
    }
    return result;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
DataLoader::DataLoader(std::shared_ptr<Tokenizer> tokenizer, size_t max_len, size_t batch_size, size_t stride) : 
                            m_tokenizer(std::move(tokenizer)), m_max_len(max_len), m_batch_size(batch_size), m_stride(stride),
                            m_input_ids({batch_size, max_len}), m_target_ids({batch_size, max_len}) {
    // Left blank
}

bool DataLoader::ingest(const std::string& path) {
    try {
        // Method taken from: https://stackoverflow.com/a/116177
        std::ifstream ifs(path);
        std::string s = std::string(std::istreambuf_iterator<char>{ifs}, {});
        // Tokenize the data source all at once
        m_tokens = m_tokenizer->tokenize(s);
        return true;
    } catch (const std::exception& e) {
        log_message(Log_Priority::ERROR, "DataLoader::ingest", std::format("Exception when ingesting data source: {}", e.what()));
        return false;
    }
}

const Matrix<size_t>& DataLoader::next_input() {
    // Get a list of indices for our next input batch
    std::vector<std::pair<size_t, size_t>> idxs = get_idxs(m_next_input_idx);
    // Zero out the input ids before making any changes
    m_input_ids.fill(0);
    for (size_t i = 0; i < idxs.size(); ++i) {
        const auto& [idx0, idx1] = idxs.at(i);
        for (size_t j = idx0; j < idx1; ++j) {
            m_input_ids.at({i, j - idx0}) = m_tokens.at(j);
        }
        // Increment m_next_input_idx to idx0 + stride
        if (i + 1 == idxs.size()) {
            m_next_input_idx = idx0 + m_stride;
        }
    }
    return m_input_ids;
}

const Matrix<size_t>& DataLoader::next_target() {
    // Get a list of indices for our next input batch, using offset = 1 to fetch
    // the next token too
    std::vector<std::pair<size_t, size_t>> idxs = get_idxs(m_next_target_idx);
    // Zero out the input ids before making any changes
    m_target_ids.fill(0);
    for (size_t i = 0; i < idxs.size(); ++i) {
        const auto& [idx0, idx1] = idxs.at(i);
        for (size_t j = idx0; j < idx1; ++j) {
            m_target_ids.at({i, j - idx0}) = m_tokens.at(j);
        }
        // Increment m_next_target_idx to idx0 + stride
        if (i + 1 == idxs.size()) {
            m_next_target_idx = idx0 + m_stride;
        }
    }
    return m_target_ids;
}
