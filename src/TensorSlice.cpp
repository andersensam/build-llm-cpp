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
 * @version: 2026-08-05
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#include "include/TensorSlice.hpp"

using TensorSlice_NS::SliceConfig;
using TensorSlice_NS::VectorSliceConfig;
using TensorSlice_NS::MatrixSliceConfig;
using TensorSlice_NS::SliceOrientation;
using TensorSlice_NS::IndexType;

// NOLINTBEGIN(bugprone-easily-swappable-parameters)

VectorSliceConfig::VectorSliceConfig(size_t idx_dim, size_t idx, size_t dim1, const std::initializer_list<size_t>& dim1_filter,
                                     SliceOrientation orientation, const std::initializer_list<std::pair<size_t,size_t>>& other_dims) : c_dim0(idx_dim),
                                     c_dim1(dim1), c_orientation(orientation) {
    // Store the index
    c_idx.push_back(idx);
    // If we have a filter for the slice dim, persist it
    if (dim1_filter.size() > 0) {
        // Make sure we get exactly two coordinates for the range
        if (dim1_filter.size() != 2) {
            throw std::invalid_argument("VectorSliceConfig.VectorSliceConfig: Applying a filter to dim1 requires exactly two coordinates.\n");
        }
        // Ensure the start index of the filter is less than the end index
        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        size_t start_idx = dim1_filter.begin()[0];
        size_t end_idx = dim1_filter.begin()[1];
        // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        if (start_idx == end_idx) {
            throw std::invalid_argument("VectorSliceConfig.VectorSliceConfig: Start and end indices for dim1 filter cannot be the same.\n");
        }
        if (start_idx > end_idx) {
            throw std::invalid_argument("VectorSliceConfig.VectorSliceConfig: Start index must be less than end index for dim1 filter.\n");
        }
        c_dim1_filter = {start_idx, end_idx};
    }
    // Check to see if we have other dimensions specified
    if (other_dims.size() > 0) {
        // Reserve c_other_dims to avoid multiple reallocations
        c_other_dims.reserve(other_dims.size());
        // Using the std::pair type inside the initializer_list should ensure that we have valid inputs for the other dims
        for (size_t i = 0; i < other_dims.size(); ++i) {
            const auto& [dim_num, dim_val] = other_dims.begin()[i]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            c_other_dims.emplace_back(dim_num, dim_val);
        }
    }
}

// NOLINTEND(bugprone-easily-swappable-parameters)

size_t VectorSliceConfig::get_dim0() const {
    return c_dim0;
}

size_t VectorSliceConfig::get_dim1() const {
    return c_dim1;
}

IndexType VectorSliceConfig::get_idx_type() const {
    return c_idx_type;
}

const std::vector<size_t>& VectorSliceConfig::get_idxs() const {
    return c_idx;
}

bool VectorSliceConfig::has_orientation() const {
    return true;
}

SliceOrientation VectorSliceConfig::get_orientation() const {
    return c_orientation;
}

bool VectorSliceConfig::has_other_dims() const {
    return c_other_dims.size() > 0;
}

const std::vector<std::pair<size_t, size_t>>& VectorSliceConfig::get_other_dims() const {
    return c_other_dims;
}

bool VectorSliceConfig::has_dim1_filter() const {
    // Unpack the filter
    const auto& [start_idx, end_idx] = c_dim1_filter;
    // If either of these is not equal to zero, then we have a filter
    if (start_idx != 0 || end_idx != 0) {
        return true;
    }
    return false;
}

std::pair<size_t, size_t> VectorSliceConfig::get_dim1_filter() const {
    return c_dim1_filter;
}

MatrixSliceConfig::MatrixSliceConfig(size_t idx_dim, IndexType idx_type, const std::vector<size_t>& idxs, size_t dim1, 
                                     const std::initializer_list<size_t>& dim1_filter,
                                     const std::initializer_list<std::pair<size_t,size_t>>& other_dims) :
                                     c_dim0(idx_dim), c_dim1(dim1), c_idx_type(idx_type), c_idxs(idxs) {
    // Ensure we didn't get an empty vector for idxs
    if (idxs.empty()) {
        throw std::invalid_argument("MatrixSliceConfig.MatrixSliceConfig: Index vector cannot be empty.\n");
    }
    // If we have a range index type, ensure we have exactly two values in idxs
    if (idx_type == IndexType::RANGE) {
        if (idxs.size() != 2) {
            throw std::invalid_argument("MatrixSliceConfig.MatrixSliceConfig: Index vector must contain exactly two elements when using RANGE.\n");
        }
        // Ensure the start index is less than the end index
        size_t start_idx = idxs.at(0);
        size_t end_idx = idxs.at(1);
        if (start_idx == end_idx) {
            throw std::invalid_argument("MatrixSliceConfig.MatrixSliceConfig: Start index cannot equal end index.\n");
        }
        if (start_idx > end_idx) {
            throw std::invalid_argument("MatrixSliceConfig.MatrixSliceConfig: Start index must be less than end index.\n");
        }
        if ((end_idx - start_idx) == 1) {
            throw std::invalid_argument("MatrixSliceConfig.MatrixSliceConfig: The difference between the start and end indices must be greater than 1. Use VectorSliceConfig.\n");
        }
    }
    else if (idx_type == IndexType::LIST) {
        // Ensure we have more than one element in idxs, otherwise it is the same as using ELEMENT
        if (idxs.size() < 2) {
            throw std::invalid_argument("MatrixSliceConfig.MatrixSliceConfig: Index list must have at least two elements.\n");
        }
    }
    else {
        throw std::invalid_argument("MatrixSliceConfig.MatrixSliceConfig: ELEMENT type cannot be used with MatrixSliceConfig.\n");
    }
    // If we have a filter for the slice dim, persist it
    if (dim1_filter.size() > 0) {
        // Make sure we get exactly two coordinates for the range
        if (dim1_filter.size() != 2) {
            throw std::invalid_argument("MatrixSliceConfig.MatrixSliceConfig: Applying a filter to dim1 requires exactly two coordinates.\n");
        }
        // Ensure the start index of the filter is less than the end index
        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        size_t start_idx = dim1_filter.begin()[0];
        size_t end_idx = dim1_filter.begin()[1];
        // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        if (start_idx == end_idx) {
            throw std::invalid_argument("MatrixSliceConfig.MatrixSliceConfig: Start and end indices for dim1 filter cannot be the same.\n");
        }
        if (start_idx > end_idx) {
            throw std::invalid_argument("MatrixSliceConfig.MatrixSliceConfig: Start index must be less than end index for dim1 filter.\n");
        }
        c_dim1_filter = {start_idx, end_idx};
    }
    // Check to see if we have other dimensions specified
    if (other_dims.size() > 0) {
        // Reserve c_other_dims to avoid multiple reallocations
        c_other_dims.reserve(other_dims.size());
        // Using the std::pair type inside the initializer_list should ensure that we have valid inputs for the other dims
        for (size_t i = 0; i < other_dims.size(); ++i) {
            const auto& [dim_num, dim_val] = other_dims.begin()[i]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            c_other_dims.emplace_back(dim_num, dim_val);
        }
    }
}

size_t MatrixSliceConfig::get_dim0() const {
    return c_dim0;
}

size_t MatrixSliceConfig::get_dim1() const {
    return c_dim1;
}

IndexType MatrixSliceConfig::get_idx_type() const {
    return c_idx_type;
}

const std::vector<size_t>& MatrixSliceConfig::get_idxs() const {
    return c_idxs;
}

bool MatrixSliceConfig::has_orientation() const {
    return false;
}

bool MatrixSliceConfig::has_other_dims() const {
    return c_other_dims.size() > 0;
}

const std::vector<std::pair<size_t, size_t>>& MatrixSliceConfig::get_other_dims() const {
    return c_other_dims;
}

bool MatrixSliceConfig::has_dim1_filter() const {
    // Unpack the filter
    const auto& [start_idx, end_idx] = c_dim1_filter;
    // If either of these is not equal to zero, then we have a filter
    if (start_idx != 0 || end_idx != 0) {
        return true;
    }
    return false;
}

std::pair<size_t, size_t> MatrixSliceConfig::get_dim1_filter() const {
    return c_dim1_filter;
}
