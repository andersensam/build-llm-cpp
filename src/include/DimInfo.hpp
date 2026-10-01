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
 * @version: 2026-09-29
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef DIMINFO_HPP
#define DIMINFO_HPP

/* Standard dependencies */
#include <cstdint>
#include <cstddef>

namespace DimInfo_NS {

/**
 * Struct containing control information for TensorSlice, required when translating
 * coordinates to access the underlying Tensor's data
 */
struct DimInfo {
    /**
     * Whether or not the dim requires rewrite
     */
    bool requires_rewrite = false;

    /**
     * Dim we are rewriting to, either 0 or 1
     */
    uint8_t rewrite_to = 0;
    
    /**
     * Dim number
     */
    size_t dim = 0;
  
    /**
     * If the dim represents a higher-rank Tensor, store its
     * static value
     */
    size_t base = 0;

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    /**
     * Constructor to use emplace_back
     * @param requires_rewrite Whether or not a rewrite is required for the dim
     * @param rewrite_to Either 0 or 1
     * @param dim The dim in the underlying Tensor
     */
    DimInfo(bool requires_rewrite, uint8_t rewrite_to, size_t dim) : requires_rewrite(requires_rewrite), rewrite_to(rewrite_to), dim(dim) {}

    /**
     * Constructor to use emplace_back, when specifying a base
     */
    DimInfo(bool requires_rewrite, uint8_t rewrite_to, size_t dim, size_t base) : requires_rewrite(requires_rewrite), rewrite_to(rewrite_to), dim(dim),
                                                                                  base(base) {}
    // NOLINTEND(bugprone-easily-swappable-parameters)
};

}; // namespace DimInfo_NS

#endif
