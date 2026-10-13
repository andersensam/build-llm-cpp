/* Large Language Model training framework in C++
 * Copyright © 2026 Samuel Andersen
 *  ________   ___   __    ______   ______   ______    ______   ______   ___   __    ______   ________   ___ __ __     
 * /_______/\ /__/\ /__/\ /_____/\ /_____/\ /_____/\  /_____/\ /_____/\ /__/\ /__/\ /_____/\ /_______/\ /__//_//_/\    
 * \::: _  \ \\::\_\\  \ \\:::_ \ \\::::_\/_\:::_ \ \ \::::_\/_\::::_\/_\::\_\\  \ \\::::_\/_\::: _  \ \\::\| \| \ \   
 *  \::(_)  \ \\:. `-\  \ \\:\ \ \ \\:\/___/\\:(_) ) )_\:\/___/\\:\/___/\\:. `-\  \ \\:\/___/\\::(_)  \ \\:.      \ \  
 *   \:: __  \ \\:. _    \ \\:\ \ \ \\::___\/_\: __ `\ \\_::._\:\\::___\/_\:. _    \ \\_::._\:\\:: __  \ \\:.\-/\  \ \ 
 *    \:.\ \  \ \\. \`-\  \ \\:\/.:| |\:\____/\\ \ `\ \ \ /____\:\\:\____/\\. \`-\  \ \ /____\:\\:.\ \  \ \\. \  \  \ \
 *     \__\/\__\/ \__\/ \__\/ \____/_/ \_____\/ \_\/ \_\/ \_____\/ \_____\/ \__\/ \__\/ \_____\/ \__\/\__\/ \__\/ \__\/    
 *                                                                                                               
 * Project: Large Language Model in C++
 * @author : Samuel Andersen
 * @version: 2026-10-09
 *
 * General Notes:
 *
 * AbstractEmbedding.hpp: Interface for embeddings
 */

#ifndef ABSTRACT_EMBEDDING_HPP
#define ABSTRACT_EMBEDDING_HPP

/* Standard dependencies */
#include <vector>

/* Local dependencies */
#include "ListTensorSlice.hpp"

namespace AbstractEmbedding_NS {

/* Use ListTensorSlice */
using ListTensorSlice_NS::ListTensorSlice;

// NOLINTBEGIN(cppcoreguidelines-special-member-functions)
/**
 * Define a common interface for embeddings
 */
template <typename T>
requires std::is_arithmetic_v<T>
class AbstractEmbedding {
/* Public functions */
public:
    /**
     * Virtual destructor for AbstractEmbedding
     */
    virtual ~AbstractEmbedding() = default;

    /**
     * Get the embeddings for an input, returning a ListTensorSlice from the underlying weight Tensor
     */
    virtual const ListTensorSlice<T> get_embeddings(const std::vector<size_t>& input) const = 0;
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace AbstractEmbedding_NS

#endif