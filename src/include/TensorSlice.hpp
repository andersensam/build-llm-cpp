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

#ifndef TENSORSLICE_HPP
#define TENSORSLICE_HPP

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

/* Local dependencies */
#include "Log.hpp"
#include "Tensor.hpp"

namespace TensorSlice_NS {

// Use logging functions
using Log::log_message;
using Log::Log_Priority;

// Use Tensor, Matrix, and helper functions from Tensor_NS
using Tensor_NS::Tensor;
using Tensor_NS::Matrix;
using Tensor_NS::_add_overflow;
using Tensor_NS::_mul_overflow;

/**
 * Enum for whether a 1-D Tensor (vector) should be represented as a row or column
 */
enum class SliceOrientation : uint8_t {
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
    virtual SliceOrientation get_orientation() const = 0;

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
    SliceOrientation c_orientation = SliceOrientation::ROW;

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
    VectorSliceConfig(size_t idx_dim, size_t idx, size_t dim1, const std::initializer_list<size_t>& dim1_filter,
                      SliceOrientation orientation, const std::initializer_list<std::pair<size_t,size_t>>& other_dims);

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
    SliceOrientation get_orientation() const override;

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
    MatrixSliceConfig(size_t idx_dim, IndexType idx_type, const std::vector<size_t>& idxs, size_t dim1, const std::initializer_list<size_t>& dim1_filter,
                      const std::initializer_list<std::pair<size_t,size_t>>& other_dims);

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

/**
 * Struct containing control information for TensorSlice, required when translating
 * coordinates to access the underlying Tensor / Matrix
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

template <typename T>
requires std::is_arithmetic_v<T>
/**
 * Class for storing slices of a Tensor or Matrix, containing mappings of desired dims. The underlying data
 * still belongs to the parents Tensor / Matrix, resulting in a relatively lightweight container
 */
class TensorSlice {
/* Private data elements */
private:
    /**
     * Reference wrapper to the underlying Matrix / Tensor. Note that the reference is marked const
     * so TensorSlice can never modify the contents of the underlying Tensor
     */
    std::reference_wrapper<const Tensor<T>> c_ref;

    /**
     * Map of the slice axis indices to their underlying values. If, for example, we create a 1-D Tensor from a
     * Matrix, we should be able to request an index from the map and get its location in the underlying
     * Matrix instance
     */
    std::map<size_t, size_t> m_dim0_map = std::map<size_t, size_t>();

    /**
     * Map of an optional second slice axis, used when creating a 2-D slice from either a larger Matrix
     * or from a multidimensional Tensor
     */
    std::map<size_t, size_t> m_dim1_map = std::map<size_t, size_t>();

    /**
     * Array containing the dimensions of the TensorSlice, limited to 2 elements since we don't support
     * high-rank TensorSlices
     */
    std::array<size_t, 2> c_slice_dims = {0, 0};

    /**
     * Vector containing the coordinates for element access
     */
    std::map<size_t, DimInfo> c_other_dims = std::map<size_t, DimInfo>();

/* Public methods */
public:
    /**
     * Constructor for TensorSlice, accepting a const reference to a Tensor / Matrix and the SliceConfig object
     * @param tensor Const ref to a Tensor / Matrix
     * @param config SliceConfig object, either VectorSliceConfig or MatrixSliceConfig
     */
    TensorSlice(const Tensor<T>& tensor, const SliceConfig& config) : c_ref(tensor) {
        // Ensure we are dealing with a Tensor with at least rank == 2
        if (tensor.rank() < 2) {
            throw std::invalid_argument("TensorSlice.TensorSlice: Tensor must have at least rank == 2.\n");
        }
        // Ensure that dim0 and dim1 are valid in the Tensor
        const auto& tensor_dims = tensor.dims();
        if (config.get_dim0() >= tensor_dims.size() || config.get_dim1() >= tensor_dims.size()) {
            throw std::invalid_argument("TensorSlice.TensorSlice: Invalid dim0 or dim1 provided.\n");
        }
        // Save which each dim is actually referring do
        c_other_dims.try_emplace(config.get_dim0(), true, 0, config.get_dim0());
        c_other_dims.try_emplace(config.get_dim1(), true, 1, config.get_dim1());
        // Check to see if we need to worry about other dims
        if (config.has_other_dims()) {
            // List the other dimensions and their base indices
            const auto& other_dims = config.get_other_dims();
            for (const auto& [dim_num, dim_idx] : other_dims) {
                // Save the dims we do not want to rewrite
                c_other_dims.try_emplace(dim_num, false, 0, dim_num, dim_idx);
            }
        }
        // The easiest way to determine if we want a 1-D or 2-D Slice is to check has_orientation(), which
        // is only present for 1-D Slices
        if (config.has_orientation()) {
            // Store the target index of dim0, which has to be at [0] for a 1-D Slice
            m_dim0_map[0] = config.get_idxs().at(0);
            // Check to see if we have a filter applied to the second dim, or if we are pulling
            // the entire dim
            size_t dim1_size = 0;
            if (config.has_dim1_filter()) {
                // Store the dimensions of the TensorSlice according to the desired orientation
                const auto& [dim1_begin, dim1_end] = config.get_dim1_filter();
                dim1_size = dim1_end - dim1_begin;
                // Create the rewrite rules for dim1
                for (size_t i = dim1_begin; i < dim1_end; ++i) {
                    m_dim1_map[i - dim1_begin] = i;
                }
            }
            else {
                dim1_size = tensor_dims.at(config.get_dim1());
            }
            // Check the orientation and store it
            if (config.get_orientation() == SliceOrientation::ROW) {
                // Since this is a 1-D Slice, one of the dims will always be == 1
                c_slice_dims = {1, dim1_size};
            }
            else {
                c_slice_dims = {dim1_size, 1};
            }
        }
        // Otherwise, we are building a 2-D TensorSlice (matrix)
        else {
            // Calculate the size of dim0
            size_t dim0_size = 0;
            const auto& idxs = config.get_idxs();
            // Handle the easier case of getting ranges for the dims
            if (config.get_idx_type() == IndexType::RANGE) {
                // Create the rewrite rules for dim0
                size_t start_idx = idxs.at(0);
                size_t end_idx = idxs.at(1);
                for (size_t i = start_idx; i < end_idx; ++i) {
                    m_dim0_map[i - start_idx] = i;
                }
                dim0_size = end_idx - start_idx;
                
            }
            // Handle the case where we want to create a Slice based on a list of values for dim0
            else {
                // Create the rewrite rules based on the index list
                for (size_t i = 0; i < idxs.size(); ++i) {
                    m_dim0_map[i] = idxs.at(i);
                }
                dim0_size = idxs.size();
            }
            // Calculate the size of dim1
            size_t dim1_size = 0;
            // See if we are filtering dim1
            if (config.has_dim1_filter()) {
                const auto& [dim1_begin, dim1_end] = config.get_dim1_filter();
                for (size_t i = dim1_begin; i < dim1_end; ++i) {
                    m_dim1_map[i - dim1_begin] = i;
                }
                // Calculate the size of dim1
                dim1_size = dim1_end - dim1_begin;
            }
            else {
                dim1_size = tensor_dims.at(config.get_dim1());
            }
            // Store the dimensions of the Slice
            c_slice_dims = {dim0_size, dim1_size};
        }
    }

    /**
     * Get the rank of a TensorSlice, which can either be 1 or 2
     * @returns Returns the rank of the TensorSlice
     */
    size_t rank() const {
        if ((c_slice_dims.at(0) > 1) && (c_slice_dims.at(1) > 1)) {
            return 2;
        }
        return 1;
    }

    /**
     * Get the value at a specified index or coordinate, following the syntax of Tensor / Matrix
     * @param c Initializer list with a single index or coordinate to fetch
     * @returns Returns the value at the index / coordinate
     */
    const T& at(const std::initializer_list<size_t>& c) const {
        // Ensure we get at least one coordinate
        if (c.size() == 0) {
            throw std::invalid_argument("TensorSlice.at: No coordinates received.\n");
        }
        // Enable polymorphic behavior
        const auto& target = c_ref.get();
        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        // Handle getting only one index
        if (c.size() == 1) {
            // Ensure we have a 1-D TensorSlice
            if (rank() != 1) {
                throw std::invalid_argument("TensorSlice.at: Only received one index for a 2-D TensorSlice.\n");
            }
            // Determine the rank of the underlying Tensor
            size_t tensor_rank = target.rank();
            // If rank == 2, the operation is simple
            if (tensor_rank == 2) {
                const auto& dim0_rules = c_other_dims.at(0);
                // Check to see if a filter is applied on dim1
                if (!m_dim1_map.empty()) {
                    if (m_dim1_map.count(c.begin()[0]) == 0) {
                        throw std::invalid_argument("TensorSlice.at: Invalid coordinate provided for filtered dim1.\n");
                    }
                    // See if dim0 maps to underlying Tensor dim0
                    if (dim0_rules.rewrite_to == 0) {
                        return target.at({m_dim0_map.at(0), m_dim1_map.at(c.begin()[0])});
                    }
                    // Otherwise
                    return target.at({m_dim1_map.at(c.begin()[0]), m_dim0_map.at(0)});
                }
                else {
                    // If we don't filter dim1, pass through the coordinate directly
                    if (dim0_rules.rewrite_to == 0) {
                        return target.at({m_dim0_map.at(0), c.begin()[0]});
                    }
                    return target.at({c.begin()[0], m_dim0_map.at(0)});
                }
            }
        }
        // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }
};

}; // namespace TensorSlice_NS

#endif
