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

#ifndef SLICECONFIG_HPP
#define SLICECONFIG_HPP

/* Standard dependencies */
#include <array>
#include <format>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace SliceConfig_NS {

/**
 * Enum for whether a 1-D Tensor (vector) should be represented as a row or column
 */
enum class VectorSliceOrientation : uint8_t {
    ROW,
    COLUMN
};

/**
 * Enum for whether the indices provided represent individual elements, or ranges of elements
 */
enum class IndexType : uint8_t {
    ELEMENT,
    LIST,
    RANGE
};

// NOLINTBEGIN(cppcoreguidelines-special-member-functions)
/**
 * Base class for SliceConfig, to be implemented by the 1-D TensorSlice (vector) and
 * 2-D TensorSlice (matrix) classes. The design assumes that we are incredibly unlikely
 * to require a higher rank TensorSlice, simplifying the implementation.
 */
class SliceConfig {
public:
    /**
     * Virtual destructor, that needs to be handled by the individual implementations
     */
    virtual ~SliceConfig() = default;

    /**
     * Get the first dimension for the Slice
     * @returns The first dimension to slice
     */
    virtual size_t get_dim0() const = 0;

    /**
     * Get the second dimension for the Slice
     * @returns The second dimension to slice
     */
    virtual size_t get_dim1() const = 0;

    /**
     * Get the the index type being used
     * @return Returns the enum class IndexType with either ELEMENT or RANGE
     */
    virtual IndexType get_idx_type() const = 0;

    /**
     * Get the index / indices we want to slice
     * @returns Returns a reference to a vector containing the index / indices
     */
    virtual const std::vector<size_t>& get_idxs() const = 0;

    /**
     * Check whether or not we have an orientation (only used by 1-D Slices)
     * @returns True if 1-D Slice, false otherwise
     */
    virtual bool has_orientation() const = 0;

    /**
     * Get the orientation, if present
     * @returns Return the SliceOrentation enum, either ROW or COLUMN
     */
    virtual VectorSliceOrientation get_orientation() const = 0;

    /**
     * Checks whether or not we need to have other_dims defined
     * @returns True if required, false otherwise
     */
    virtual bool has_other_dims() const = 0;

    /**
     * Gets the other axes, if present
     * @returns Returns a const ref to a vector with the other axes
     */
    virtual const std::vector<std::pair<size_t, size_t>>& get_other_dims() const = 0;

    /**
     * Checks whether or not dim1 has a filter
     * @returns True if we filter dim1 (i.e. restrict it to a range), false if we take
     * the entire dim
     */
    virtual bool has_dim1_filter() const = 0;

    /**
     * Gets the dim1 filter, if present
     * @returns Returns a pair<size_t,size_t> with the filter
     */
    virtual std::pair<size_t, size_t> get_dim1_filter() const = 0;
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
/**
 * Implementation of a 1-D SliceConfig
 */
class VectorSliceConfig : public SliceConfig {
/* Private data elements */
private:
    /**
     * The first dimension of the underlying data source
     */
    size_t c_dim0 = 0;

    /**
     * The second dimension of the underlying data source
     */
    size_t c_dim1 = 0;

    /**
     * The index type used to build the Slice, which can only be ELEMENT
     * when constructing a 1-D Slice
     */
    IndexType c_idx_type = IndexType::ELEMENT;

    /**
     * Index to pull from the source
     */
    std::vector<size_t> c_idx = std::vector<size_t>();

    /**
     * The orientation of the Slice, defaulting to a single row with the
     * number of columns coming from the source
     */
    VectorSliceOrientation c_orientation = VectorSliceOrientation::ROW;

    /**
     * Optional other axes, used when building a 1-D Slice from a high rank Tensor
     */
    std::vector<std::pair<size_t, size_t>> c_other_dims = std::vector<std::pair<size_t,size_t>>();

    /**
     * Optional filter on dim1, limiting the number of row / columns to the range
     * specified here
     */
    std::pair<size_t, size_t> c_dim1_filter = std::pair<size_t, size_t>(0, 0);

/* Public functions */
public:
    /**
     * Constructor for the 1-D Slice (vector)
     * @param idx_dim Dimension to index 
     * @param idx Index on the specified dimension to pull
     * @param dim1 Dimension to slice across
     * @param dim1_filter Optional filter for dim1 (limiting number of rows / columns)
     * @param orientation Desired orientation of the vector
     * @param other_dims Other dims coordinates, needed for high-rank Tensors
     */
    VectorSliceConfig(size_t idx_dim, size_t idx, size_t dim1, std::initializer_list<size_t> dim1_filter,
                      VectorSliceOrientation orientation, std::initializer_list<std::pair<size_t,size_t>> other_dims);

    /**
     * Get the first dimension for the Slice
     * @returns The first dimension to slice
     */
    size_t get_dim0() const override;

    /**
     * Get the second dimension for the Slice
     * @returns The second dimension to slice
     */
    size_t get_dim1() const override;

    /**
     * Get the the index type being used
     * @return Returns the enum class IndexType with either ELEMENT or RANGE
     */
    IndexType get_idx_type() const override;

    /**
     * Get the index / indices we want to slice
     * @returns Returns a reference to a vector containing the index / indices
     */
    const std::vector<size_t>& get_idxs() const override;

    /**
     * Check whether or not we have an orientation (only used by 1-D Slices)
     * @returns True if 1-D Slice, false otherwise
     */
    bool has_orientation() const override;

    /**
     * Get the orientation, if present
     * @returns Return the SliceOrentation enum, either ROW or COLUMN
     */
    VectorSliceOrientation get_orientation() const override;

    /**
     * Checks whether or not we need to have other_dims defined
     * @returns True if required, false otherwise
     */
    bool has_other_dims() const override;

    /**
     * Gets the other axes, if present
     * @returns Returns a const ref to a vector with the other axes
     */
    const std::vector<std::pair<size_t, size_t>>& get_other_dims() const override;

    /**
     * Checks whether or not dim1 has a filter
     * @returns True if we filter dim1 (i.e. restrict it to a range), false if we take
     * the entire dim
     */
    bool has_dim1_filter() const override;

    /**
     * Gets the dim1 filter, if present
     * @returns Returns a pair<size_t,size_t> with the filter
     */
    std::pair<size_t, size_t> get_dim1_filter() const override;
};

/**
 * Implementation of a 2-D SliceConfig
 */
class MatrixSliceConfig : public SliceConfig {
/* Private data elements */
private:
    /**
     * The first dimension of the underlying data source
     */
    size_t c_dim0 = 0;

    /**
     * The second dimension of the underlying data source
     */
    size_t c_dim1 = 0;

    /**
     * The index type used to build the Slice, defaulting to RANGE, but can also be
     * a list of elements to aggregate into the Slice
     */
    IndexType c_idx_type = IndexType::RANGE;

    /**
     * Indices or range to pull from the source
     */
    std::vector<size_t> c_idxs = std::vector<size_t>();

    /**
     * Optional other axes, used when building a Slice from a higher-rank Tensor
     */
    std::vector<std::pair<size_t, size_t>> c_other_dims = std::vector<std::pair<size_t,size_t>>();

    /**
     * Optional filter on dim1, limiting the number of row / columns to the range
     * specified here
     */
    std::pair<size_t, size_t> c_dim1_filter = std::pair<size_t, size_t>(0, 0);

/* Public functions */
public:
    /**
     * Constructor for the 2-D Slice (matrix)
     * @param idx_dim Dimension to index 
     * @param idx_type Type of index for building the Slice, either LIST or RANGE
     * @param idxs Either a singular index, a list of indices, or a range
     * @param dim1 Dimension to slice across
     * @param dim1_filter Optional filter for dim1 (limiting number of rows / columns)
     * @param other_dims Other dims coordinates, needed for high-rank Tensors
     */
    MatrixSliceConfig(size_t idx_dim, IndexType idx_type, const std::vector<size_t>& idxs, size_t dim1, std::initializer_list<size_t> dim1_filter,
                      std::initializer_list<std::pair<size_t,size_t>> other_dims);

    /**
     * Get the first dimension for the Slice
     * @returns The first dimension to slice
     */
    size_t get_dim0() const override;

    /**
     * Get the second dimension for the Slice
     * @returns The second dimension to slice
     */
    size_t get_dim1() const override;

    /**
     * Get the the index type being used
     * @return Returns the enum class IndexType with either ELEMENT or RANGE
     */
    IndexType get_idx_type() const override;

    /**
     * Get the index / indices we want to slice
     * @returns Returns a reference to a vector containing the index / indices
     */
    const std::vector<size_t>& get_idxs() const override;

    /**
     * Check whether or not we have an orientation (only used by 1-D Slices)
     * @returns True if 1-D Slice, false otherwise
     */
    bool has_orientation() const override;

    /**
     * Will throw an exception if called on MatrixSliceConfig
     */
    VectorSliceOrientation get_orientation() const override;

    /**
     * Checks whether or not we need to have other_dims defined
     * @returns True if required, false otherwise
     */
    bool has_other_dims() const override;

    /**
     * Gets the other axes, if present
     * @returns Returns a const ref to a vector with the other axes
     */
    const std::vector<std::pair<size_t, size_t>>& get_other_dims() const override;

    /**
     * Checks whether or not dim1 has a filter
     * @returns True if we filter dim1 (i.e. restrict it to a range), false if we take
     * the entire dim
     */
    bool has_dim1_filter() const override;

    /**
     * Gets the dim1 filter, if present
     * @returns Returns a pair<size_t,size_t> with the filter
     */
    std::pair<size_t, size_t> get_dim1_filter() const override;
};
// NOLINTEND(bugprone-easily-swappable-parameters)

}; // namespace SliceConfig_NS

#endif
