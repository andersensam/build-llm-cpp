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
 * @version: 2026-09-14
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
#include "Numerics.hpp"
#include "Tensor.hpp"

namespace TensorSlice_NS {

/* Use logging functions */
using Log::log_message;
using Log::Log_Priority;

/* Use Tensor and helper functions from Tensor_NS */
using Tensor_NS::AbstractTensor;
using Tensor_NS::Tensor;
using Tensor_NS::_add_overflow;
using Tensor_NS::_mul_overflow;

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

/**
 * Struct containing control information for TensorSlice, required when translating
 * coordinates to access the underlying Tensor
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

/**
 * ListTensorSlice, a TensorSlice built from a list of values, good for searching through
 * a larger Tensor and creating a "contiguous" representation from it
 */
template <typename T>
requires std::is_arithmetic_v<T>
class ListTensorSlice final : public AbstractTensor<T> {
/* Private data elements */
private:
    /**
     * Shared pointer to a Tensor, ensuring the underlying data source isn't 
     * deleted while the TensorSlice is in scope
     */
    std::shared_ptr<const Tensor<T>> c_ptr;

    /**
     * Rank of the underlying Tensor, stored to avoid calling c_ptr->rank() repeatedly
     */
    size_t c_tensor_rank = 0;

    /**
     * Allow writes, determined by the constructor receiving a const Tensor<T> or a 
     * regular Tensor<T> shared pointer
     */
    bool _is_writable = false;

    /**
     * Map of the slice axis indices to their underlying values. If, for example, we create a 1-D TensorSlice from a
     * Tensor, we should be able to request an index from the map and get its location in the underlying
     * Tensor instance
     */
    std::map<size_t, size_t> m_dim0_map = std::map<size_t, size_t>();

    /**
     * Map of an optional second slice axis, used when creating a 2-D slice from either a larger Tensor
     * or from a multidimensional Tensor
     */
    std::map<size_t, size_t> m_dim1_map = std::map<size_t, size_t>();

    /**
     * Vector containing the dimensions of the TensorSlice, limited to 2 elements since we don't support
     * high-rank TensorSlices
     */
    std::vector<size_t> m_slice_dims = {0, 0};

    /**
     * Vector containing the strides of the dimensions in the underlying Tensor
     */
    std::vector<size_t> m_stride = {0, 0};

    /**
     * Vector containing the coordinates for element access
     */
    std::map<size_t, DimInfo> m_other_dims = std::map<size_t, DimInfo>();

/* Private methods */
    /**
     * Initialize the TensorSlice
     * @param config SliceConfig
     */
    void _initialize(const SliceConfig& config) { 
        // Ensure we actually want to construct a ListTensorSlice
        if (config.get_idx_type() == IndexType::RANGE) {
            throw std::logic_error("ListTensorSlice.ListTensorSlice: IndexType cannot be RANGE.\n");
        }
        // Ensure we are dealing with a Tensor with at least rank == 2
        if (c_tensor_rank < 2) {
            throw std::invalid_argument("ListTensorSlice.ListTensorSlice: Tensor must have at least rank == 2.\n");
        }
        // Ensure that dim0 and dim1 are valid in the Tensor
        const auto& tensor_dims = c_ptr->shape();
        if (config.get_dim0() >= tensor_dims.size() || config.get_dim1() >= tensor_dims.size()) {
            throw std::invalid_argument("ListTensorSlice.ListTensorSlice: Invalid dim0 or dim1 provided.\n");
        }
        // Save which each dim is actually referring do
        m_other_dims.try_emplace(config.get_dim0(), true, 0, config.get_dim0());
        m_other_dims.try_emplace(config.get_dim1(), true, 1, config.get_dim1());
        // Save the strides for the dims
        const auto& tensor_stride = c_ptr->stride();
        m_stride.at(0) = tensor_stride.at(config.get_dim0());
        m_stride.at(1) = tensor_stride.at(config.get_dim1());
        // Check to see if we need to worry about other dims
        if (config.has_other_dims()) {
            // List the other dimensions and their base indices
            const auto& other_dims = config.get_other_dims();
            for (const auto& [dim_num, dim_idx] : other_dims) {
                // Save the dims we do not want to rewrite
                m_other_dims.try_emplace(dim_num, false, 0, dim_num, dim_idx);
            }
        }
        // Ensure that m_other_dims has the same size as c_tensor_rank
        if (m_other_dims.size() != c_tensor_rank) {
            throw std::invalid_argument("ListTensorSlice.ListTensorSlice: Incorrect number of dims in SliceConfig.\n");
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
            if (config.get_orientation() == VectorSliceOrientation::ROW) {
                // Since this is a 1-D Slice, one of the dims will always be == 1
                m_slice_dims.at(0) = 1;
                m_slice_dims.at(1) = dim1_size;
            }
            else {
                m_slice_dims.at(0) = dim1_size;
                m_slice_dims.at(1) = 1;
            }
        }
        // Otherwise, we are building a 2-D TensorSlice (matrix)
        else {
            // Calculate the size of dim0
            size_t dim0_size = 0;
            const auto& idxs = config.get_idxs();
            // Handle the case where we want to create a Slice based on a list of values for dim0
            // Create the rewrite rules based on the index list
            for (size_t i = 0; i < idxs.size(); ++i) {
                m_dim0_map[i] = idxs.at(i);
            }
            dim0_size = idxs.size();
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
            m_slice_dims.at(0) = dim0_size;
            m_slice_dims.at(1) = dim1_size;
        }
    }
    /**
     * Calculate the index in the underlying Tensor using provided coordinates
     * @param c Initializer list containing one or two coordinates
     * @returns Returns the index
     */
    size_t _calculate_idx(std::initializer_list<size_t> c) const {
        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        // Get the stride info from the underlying Tensor
        const auto& tensor_stride = c_ptr->stride();
        // Handle getting only one index
        if (c.size() == 1) {
            // Ensure we have a 1-D TensorSlice
            if (rank() != 1) {
                throw std::invalid_argument("ListTensorSlice._calculate_idx: Only received one index for a 2-D TensorSlice.\n");
            }
            // If rank == 2, the operation is simple
            if (c_tensor_rank == 2) {
                const auto& dim0_rules = m_other_dims.at(0);
                // Check to see if a filter is applied on dim1
                if (!m_dim1_map.empty()) {
                    if (m_dim1_map.count(c.begin()[0]) == 0) {
                        throw std::invalid_argument("ListTensorSlice._calculate_idx: Invalid coordinate provided for filtered dim1.\n");
                    }
                    // See if dim0 maps to underlying Tensor dim0
                    if (dim0_rules.rewrite_to == 0) {
                        // We don't need to check tensor_stride.at(1) since a 2-D Tensor always has stride == 1
                        // for the outer dim
                        return (tensor_stride.at(0) * m_dim0_map.at(0)) + m_dim1_map.at(c.begin()[0]);
                    }
                    // Otherwise
                    return (tensor_stride.at(0) * m_dim1_map.at(c.begin()[0])) + m_dim0_map.at(0);
                }
                else {
                    // If we don't filter dim1, pass through the coordinate directly
                    if (dim0_rules.rewrite_to == 0) {
                        return (tensor_stride.at(0) * m_dim0_map.at(0)) + c.begin()[0];
                    }
                    return (tensor_stride.at(0) * c.begin()[0]) + m_dim0_map.at(0);
                }
            }
            // Deal with a high-rank Tensor
            // Iterate over the dimensions and rewrite into a query for the underlying Tensor.
            // We already validated that m_other_dims matches the Tensor's rank
            size_t target_index = 0;
            for (size_t i = 0; i < c_tensor_rank; ++i) {
                const auto& dim_rule = m_other_dims.at(i);
                if (dim_rule.requires_rewrite) {
                    if (dim_rule.rewrite_to == 0) {
                        target_index += tensor_stride.at(i) * m_dim0_map.at(0);
                    }
                    // If not rewrite_to == 0, then must be == 1
                    else {
                        // Check to see if we are applying a filter on dim1
                        if (!m_dim1_map.empty()) {
                            target_index += tensor_stride.at(i) * m_dim1_map.at(c.begin()[0]);
                        }
                        else {
                            // Otherwise pass the original coordinate
                            target_index += tensor_stride.at(i) * c.begin()[0];
                        }
                    }
                }
                // If we don't require rewrite, pull the static values for the other dims
                else {
                    target_index += tensor_stride.at(i) * dim_rule.base;
                }
            }
            return target_index;
        }
        else if (c.size() == 2) {
            // Ensure we have a TensorSlice of rank == 2
            if (rank() != 2) {
                throw std::invalid_argument("ListTensorSlice._calculate_idx: Two coordinates cannot be passed to a rank 1 TensorSlice.\n");
            }
            // Determine the rank of the Tensor
            if (c_tensor_rank == 2) {
                const auto& dim0_rules = m_other_dims.at(0);
                // Check to see which dim is rewritten to 0
                if (dim0_rules.rewrite_to == 0) {
                    if (!m_dim1_map.empty()) {
                        return (tensor_stride.at(0) * m_dim0_map.at(c.begin()[0])) + m_dim1_map.at(c.begin()[1]);
                    }
                    return (tensor_stride.at(0) * m_dim0_map.at(c.begin()[0])) + c.begin()[1];
                }
                // Otherwise flip the mapping
                if (!m_dim1_map.empty()) {
                    return (tensor_stride.at(0) * m_dim1_map.at(c.begin()[0])) + m_dim0_map.at(c.begin()[1]);
                }
                //return c_ptr->at({c.begin()[0], m_dim0_map.at(c.begin()[1])});
                return (tensor_stride.at(0) * c.begin()[0]) + m_dim0_map.at(c.begin()[1]);
            }
            // Deal with a high-rank Tensor
            size_t target_idx = 0;
            // Iterate over the dimensions and rewrite into a query for the underlying Tensor.
            // We already validated that m_other_dims matches the Tensor's rank
            for (size_t i = 0; i < c_tensor_rank; ++i) {
                const auto& dim_rule = m_other_dims.at(i);
                if (dim_rule.requires_rewrite) {
                    if (dim_rule.rewrite_to == 0) {
                        target_idx += tensor_stride.at(i) * m_dim0_map.at(c.begin()[0]);
                    }
                    // If not rewrite_to == 0, then must be == 1
                    else {
                        if (!m_dim1_map.empty()) {
                            target_idx += tensor_stride.at(i) * m_dim1_map.at(c.begin()[1]);
                        }
                        else {
                            target_idx += tensor_stride.at(i) * c.begin()[1];
                        }
                    }
                }
                // If we don't require rewrite, pull the static values for the other dims
                else {
                    target_idx += tensor_stride.at(i) * dim_rule.base;
                }
            }
            return target_idx;
        }
        // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        throw std::invalid_argument("ListTensorSlice._calculate_idx: Too many coordinates provided.\n");
    }

/* Public methods */
public:
    /**
     * Use _can_overflow and _can_matmul from the AbstractTensor base class
     */
    using AbstractTensor<T>::_can_overflow;
    using AbstractTensor<T>::_can_matmul;

    explicit ListTensorSlice(std::shared_ptr<Tensor<T>> ptr, const SliceConfig& config) : c_ptr(std::move(ptr)), c_tensor_rank(c_ptr->rank()), _is_writable(true) {
        // Set _is_writable to true and then finish initialization
        _initialize(config);
    }

    explicit ListTensorSlice(std::shared_ptr<const Tensor<T>> ptr, const SliceConfig& config) : c_ptr(std::move(ptr)), c_tensor_rank(c_ptr->rank()) {
        _initialize(config);
    }

    /**
     * Get the rank of the TensorSlice
     * @returns Returns the rank
     */
    size_t rank() const override {
        if ((m_slice_dims.at(0) > 1) && (m_slice_dims.at(1) > 1)) {
            return 2;
        }
        return 1;
    }

    /**
     * Get the dimensions of the TensorSlice
     * @returns Returns a const reference to the vector containing the dimensions
     */
    const std::vector<size_t>& shape() const override {
        return m_slice_dims;
    }

    /**
     * Get the stride used to advance inside the TensorSlice
     * @returns Returns a const reference to the vector containing the stride of each dim
     */
    const std::vector<size_t>& stride() const override {
        return m_stride;
    }

    /**
     * Get the extent of a specified dim
     * @param dim Dimension to query
     * @returns Returns the extent of the dim
     */
    size_t extent(size_t dim) const override {
        // Ensure we get a valid dim
        if (dim >= 2) {
            throw std::invalid_argument("ListTensorSlice.extent: Invalid dim provided.\n");
        }
        return m_slice_dims.at(dim);
    }

    /**
     * Get the total number of elements in the TensorSlice
     * @returns Returns the total number of elements
     */
    size_t elements() const override {
        // Store the total number of elements
        size_t result = 1;
        for (const auto& v : m_slice_dims) {
            // Ensure we don't zero out the elements by mistake (rank-1 TensorSlice)
            if (v > 0) {
                result *= v;
            }
        }
        return result;
    }

    /**
     * Get a mutable reference to the value stored at the provided coordinates
     * @param target Initializer list containing the desired coordinates
     * @returns Returns a mutable reference to the desired value
     */
    T& at(std::initializer_list<size_t> target) override {
        // Throw exception if called on a std::shared_ptr<const Tensor<T>>
        if (!_is_writable) {
            throw std::runtime_error("ListTensorSlice.at: Cannot execute at on a const base Tensor.\n");
        }
        // Ensure we get either one or two coordinates
        if (target.size() == 0 || target.size() > 2) {
            throw std::invalid_argument("ListTensorSlice.at: Invalid number of coordinates provided to at.\n");
        }
        // Strip the const qualifier on the std::shared_ptr<const Tensor<T>>
        auto rw_ptr = std::const_pointer_cast<Tensor<T>>(c_ptr);
        // Calculate the target index in the underlying Tensor and return the value
        return rw_ptr->at(_calculate_idx(target));
    }

    /**
     * Get a const reference to the value stored at the provided coordinates
     * @param target Initializer list containing the desired coordinates
     * @returns Returns a const reference to the desired value
     */
    const T& at(std::initializer_list<size_t> target) const override {
        // Ensure we get either one or two coordinates
        if (target.size() == 0 || target.size() > 2) {
            throw std::invalid_argument("ListTensorSlice.at: Invalid number of coordinates provided to at.\n");
        }
        // Calculate the target index in the underlying Tensor and return the value
        return c_ptr->at(_calculate_idx(target));
    }

    /**
     * Get a mutable reference to the value stored at the provided coordinates
     * @param target Const reference to a vector containing the desired coordinates
     * @returns Returns a mutable reference to the desired value
     */
    T& at(const std::vector<size_t>& target) override {
        if (!_is_writable) {
            throw std::runtime_error("ListTensorSlice.at: Cannot execute at on a const base Tensor.\n");
        }
        // Strip the const qualifier on the std::shared_ptr<const Tensor<T>>
        auto rw_ptr = std::const_pointer_cast<Tensor<T>>(c_ptr);
        // Convert the std::vector to std::initializer_list based on size
        switch (target.size()) {
            case 1:
                return rw_ptr->at(_calculate_idx({target.at(0)}));
            case 2:
                return rw_ptr->at(_calculate_idx({target.at(0), target.at(1)}));
            default:
                throw std::invalid_argument("ListTensorSlice.at: Invalid number of coordinates provided to at.\n");
        }
    }

    /**
     * Get a const reference to the value stored at the provided coordinates
     * @param target Const reference to a vector containing the desired coordinates
     * @returns Returns a const reference to the desired value
     */
    const T& at(const std::vector<size_t>& target) const override {
        // Convert the std::vector to std::initializer_list based on size
        switch (target.size()) {
            case 1:
                return c_ptr->at(_calculate_idx({target.at(0)}));
            case 2:
                return c_ptr->at(_calculate_idx({target.at(0), target.at(1)}));
            default:
                throw std::invalid_argument("ListTensorSlice.at: Invalid number of coordinates provided to at.\n");
        }
    }

    // NOLINTBEGIN(clang-diagnostic-unused-parameter)
    /**
     * Get a mutable reference to the value stored at the provided index
     * @param target Index to fetch from
     * @returns Returns a mutable reference to the desired value
     */
    T& at(size_t target) override {
        // Unconditionally throw an exception since we don't have any way to decode
        // the index into a usable translation for the underlying Tensor
        throw std::logic_error("ListTensorSlice.at: at(size_t target) is not supported on TensorSlice.\n");
    }

    /**
     * Get a const reference to the value stored at the provided index
     * @param target Index to fetch from
     * @returns Returns a const reference to the desired value
     */
    const T& at(size_t target) const override {
        throw std::logic_error("ListTensorSlice.at: at(size_t target) is not supported on TensorSlice.\n");
    }
    // NOLINTEND(clang-diagnostic-unused-parameter)

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    /**
     * Transpose a Tensor along two dims
     * @param dim0 First dim to swap
     * @param dim1 Second dim to swap
     */
    ListTensorSlice<T>& transpose(size_t dim0, size_t dim1) override {
        // Ensure we have a 2-D TensorSlice
        if (rank() != 2) {
            throw std::logic_error("ListTensorSlice.transpose: Transpose cannot be called on a 1-D TensorSlice.\n");
        }
        // Ensure dim0 and dim1 are valid
        if (dim0 >= 2 || dim1 >= 2) {
            throw std::invalid_argument("ListTensorSlice.transpose: Invalid dims provided.\n");
        }
        // Swap the maps
        std::swap(m_dim0_map, m_dim1_map);
        // Swap the slice dims
        std::swap(m_slice_dims.at(0), m_slice_dims.at(1));
        // Swap the strides
        std::swap(m_stride.at(0), m_stride.at(1));
        // Iterate over the DimInfo objects and rewrite the mappings
        for (size_t i = 0; i < c_tensor_rank; ++i) {
            DimInfo& d = m_other_dims.at(i);
            if (d.requires_rewrite) {
                if (d.rewrite_to == 0) {
                    d.rewrite_to = 1;
                }
                else {
                    d.rewrite_to = 0;
                }
            }
        }
        return *this;
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    /**
     * Get a string containing information about the underlying TensorSlice
     * @returns Returns a string with the Tensor's info
     */
    std::string info() const override {
        // Prepare the info string
        std::string result = std::format("ListTensorSlice: [{}, {}]. Underlying Tensor: {}", extent(0), extent(1), c_ptr->info());
        return result;
    }

    /**
     * Convert a TensorSlice to a string representation
     * @returns Returns a string representation of the TensorSlice
     */
    std::string to_string() const {
        // Ensure this can only run on 2-D TensorSlices
        if (rank() != 2) {
            throw std::logic_error("ListTensorSlice.to_string: to_string() cannot be called on a non rank-2 TensorSlice.\n");
        }
        // Create a blank string that we will return
        std::string result = "\n";
        // Wrap the Tensor in brackets, with each row properly enclosed too
        result += "[";
        for (size_t i = 0; i < extent(0); ++i) {
            result += "[";
            for (size_t j = 0; j < extent(1); ++j) {
                // Add the value in the Matrix
                result += std::format("{}", at({i, j}));
                if (j + 1 < extent(1)) {
                    // Separate the values by tabs, for readability
                    result += "\t";
                }
            }
            // Close out each row with a corresponding ]
            result += "]";
            // Add a new line after each row if we aren't at the end
            if (i + 1 < extent(0)) {
                result += "\n";
            }
        }
        // Close out the TensorSlice final bracket and print out the info (dims and dtype)
        result += std::format("]. {}.", info());
        return result;
    }

    /**
     * Convert a TensorSlice to a Tensor, copying the data
     * @returns Returns a new Tensor with a copy of the data
     */
    Tensor<T> to_tensor() const {
        // Handle the rank == 1 case first
        if (rank() == 1) {
            // Create a 1-D Tensor with the correct size
            Tensor<T> target({extent(0) != 1 ? extent(0) : extent(1)});
            // Iterate over the elements in the TensorSlice
            for (size_t i = 0; i < elements(); ++i) {
                target.at({i}) = this->at({i});
            }
            return target;
        }
        // Otherwise we must be dealing with a 2-D TensorSlice
        Tensor<T> target({extent(0), extent(1)});
        for (size_t i = 0; i < extent(0); ++i) {
            for (size_t j = 0; j < extent(1); ++j) {
                target.at({i, j}) = this->at({i, j});
            }
        }
        return target;
    }
};

/**
 * RangeTensorSlice, a TensorSlice built from a range on an axis / axes, good for 
 * creating a slice / smaller Tensor
 */
template <typename T>
requires std::is_arithmetic_v<T>
class RangeTensorSlice final : public AbstractTensor<T> {
/* Private data elements */
private:
    /**
     * Shared pointer to a Tensor, ensuring the underlying data source isn't 
     * deleted while the TensorSlice is in scope
     */
    std::shared_ptr<const Tensor<T>> c_ptr;

    /**
     * Rank of the underlying Tensor, stored to avoid calling c_ptr->rank() repeatedly
     */
    size_t c_tensor_rank = 0;

    /**
     * Allow writes, determined by the constructor receiving a const Tensor<T> or a 
     * regular Tensor<T> shared pointer
     */
    bool _is_writable = false;

    /**
     * Start and end values for the first dim of the TensorSlice
     */
    std::array<size_t, 2> m_dim0_range = {0, 0};

    /**
     * Start and end values for the second dim of the TensorSlice
     */
    std::array<size_t, 2> m_dim1_range = {0, 0};

    /**
     * Vector containing the dimensions of the TensorSlice, limited to 2 elements since we don't support
     * high-rank TensorSlices
     */
    std::vector<size_t> m_slice_dims = {0, 0};

    /**
     * Vector containing the strides of the dimensions in the underlying Tensor
     */
    std::vector<size_t> m_stride = {0, 0};

    /**
     * Vector containing the coordinates for element access
     */
    std::map<size_t, DimInfo> m_other_dims = std::map<size_t, DimInfo>();

/* Private methods */
    /**
     * Initialize the TensorSlice
     * @param config SliceConfig
     */
    void _initialize(const SliceConfig& config) { 
        // Ensure we actually want to construct a RangeTensorSlice
        if (config.get_idx_type() != IndexType::RANGE) {
            throw std::logic_error("RangeTensorSlice.RangeTensorSlice: IndexType must be RANGE.\n");
        }
        // Ensure we are dealing with a Tensor with at least rank == 2
        if (c_tensor_rank < 2) {
            throw std::invalid_argument("RangeTensorSlice.RangeTensorSlice: Tensor must have at least rank == 2.\n");
        }
        // Ensure that dim0 and dim1 are valid in the Tensor
        const auto& tensor_dims = c_ptr->shape();
        if (config.get_dim0() >= tensor_dims.size() || config.get_dim1() >= tensor_dims.size()) {
            throw std::invalid_argument("RangeTensorSlice.RangeTensorSlice: Invalid dim0 or dim1 provided.\n");
        }
        // Save which each dim is actually referring do
        m_other_dims.try_emplace(config.get_dim0(), true, 0, config.get_dim0());
        m_other_dims.try_emplace(config.get_dim1(), true, 1, config.get_dim1());
        // Save the strides for the dims
        const auto& tensor_stride = c_ptr->stride();
        m_stride.at(0) = tensor_stride.at(config.get_dim0());
        m_stride.at(1) = tensor_stride.at(config.get_dim1());
        // Check to see if we need to worry about other dims
        if (config.has_other_dims()) {
            // List the other dimensions and their base indices
            const auto& other_dims = config.get_other_dims();
            for (const auto& [dim_num, dim_idx] : other_dims) {
                // Save the dims we do not want to rewrite
                m_other_dims.try_emplace(dim_num, false, 0, dim_num, dim_idx);
            }
        }
        // Ensure that m_other_dims has the same size as c_tensor_rank
        if (m_other_dims.size() != c_tensor_rank) {
            throw std::invalid_argument("RangeTensorSlice.RangeTensorSlice: Incorrect number of dims in SliceConfig.\n");
        }
        // Since IndexType::RANGE implies a 2-D TensorSlice, grab the index values for the filter dim
        const auto& idxs = config.get_idxs();
        // Use the no lint override since we are using arrays with their sizes guaranteed
        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        // The SliceConfig constructor ensures idxs.size() == 2, so grab the start and end
        m_dim0_range[0] = idxs.at(0);
        m_dim0_range[1] = idxs.at(1);
        // Check to see if we are applying a filter to dim1
        if (config.has_dim1_filter()) {
            const auto& [dim1_start, dim1_end] = config.get_dim1_filter();
            m_dim1_range[0] = dim1_start;
            m_dim1_range[1] = dim1_end;
        }
        else {
            m_dim1_range[0] = 0;
            // Get the size of dim1 and use that as the filter
            m_dim1_range[1] = tensor_dims.at(config.get_dim1());
        }
        // Persist the final size of the TensorSlice
        m_slice_dims.at(0) = m_dim0_range[1] - m_dim0_range[0];
        m_slice_dims.at(1) = m_dim1_range[1] - m_dim1_range[0];
        // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    }
    /**
     * Calculate the index in the underlying Tensor using provided coordinates
     * @param c Initializer list containing one or two coordinates
     * @returns Returns the index
     */
    size_t _calculate_idx(std::initializer_list<size_t> c) const {
        // Use the no lint override since we are using arrays with their sizes guaranteed
        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        // Ensure the values are within the range of the TensorSlice
        if ((c.begin()[0] + m_dim0_range[0]) >= m_dim0_range[1] || (c.begin()[1] + m_dim1_range[0]) >= m_dim1_range[1]) {
            throw std::invalid_argument("RangeTensorSlice._calculate_idx: Invalid coordinates provided.\n");
        }
        // Handle the case where the underlying Tensor has rank == 2
        if (c_tensor_rank == 2) {
            // Calculate the index with the offset + stride for dim0 and dim1
            return ((c.begin()[0] + m_dim0_range[0]) * m_stride.at(0)) + ((c.begin()[1] + m_dim1_range[0]) * m_stride.at(1));
        }
        // Get the stride info from the underlying Tensor
        const auto& tensor_stride = c_ptr->stride();
        size_t result = 0;
        for (size_t i = 0; i < c_tensor_rank; ++i) {
            const auto& dim_config = m_other_dims.at(i);
            if (dim_config.requires_rewrite) {
                if (dim_config.rewrite_to == 0) {
                    result += (c.begin()[0] + m_dim0_range[0]) * m_stride.at(0);
                }
                // Since rewrite_to can only be 0 or 1, if it's not 0, it must be 1
                else {
                    result += (c.begin()[1] + m_dim1_range[0]) * m_stride.at(1);
                }
            }
            // If we aren't rewriting the coordinate, get the base per the config
            else {
                result += dim_config.base * tensor_stride.at(dim_config.dim);
            }
        }
        // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        return result;
    }

/* Public methods */
public:
    /**
     * Use _can_overflow and _can_matmul from the AbstractTensor base class
     */
    using AbstractTensor<T>::_can_overflow;
    using AbstractTensor<T>::_can_matmul;

    explicit RangeTensorSlice(std::shared_ptr<Tensor<T>> ptr, const SliceConfig& config) : c_ptr(std::move(ptr)), c_tensor_rank(c_ptr->rank()), _is_writable(true) {
        // Set _is_writable to true and then finish initialization
        _initialize(config);
    }

    explicit RangeTensorSlice(std::shared_ptr<const Tensor<T>> ptr, const SliceConfig& config) : c_ptr(std::move(ptr)), c_tensor_rank(c_ptr->rank()) {
        _initialize(config);
    }

    /**
     * Get the rank of the TensorSlice
     * @returns Returns the rank
     */
    size_t rank() const override {
        // The IndexType::RANGE requires the TensorSlice to be 2-D
        return 2;
    }

    /**
     * Get the dimensions of the TensorSlice
     * @returns Returns a const reference to the vector containing the dimensions
     */
    const std::vector<size_t>& shape() const override {
        return m_slice_dims;
    }

    /**
     * Get the stride used to advance inside the TensorSlice
     * @returns Returns a const reference to the vector containing the stride of each dim
     */
    const std::vector<size_t>& stride() const override {
        return m_stride;
    }

    /**
     * Get the extent of a specified dim
     * @param dim Dimension to query
     * @returns Returns the extent of the dim
     */
    size_t extent(size_t dim) const override {
        // Ensure we get a valid dim
        if (dim >= 2) {
            throw std::invalid_argument("RangeTensorSlice.extent: Invalid dim provided.\n");
        }
        return m_slice_dims.at(dim);
    }

    /**
     * Get the total number of elements in the TensorSlice
     * @returns Returns the total number of elements
     */
    size_t elements() const override {
        // Store the total number of elements
        size_t result = 1;
        for (const auto& v : m_slice_dims) {
            // Ensure we don't zero out the elements by mistake (rank-1 TensorSlice)
            if (v > 0) {
                result *= v;
            }
        }
        return result;
    }

    /**
     * Get a mutable reference to the value stored at the provided coordinates
     * @param target Initializer list containing the desired coordinates
     * @returns Returns a mutable reference to the desired value
     */
    T& at(std::initializer_list<size_t> target) override {
        // Throw exception if called on a std::shared_ptr<const Tensor<T>>
        if (!_is_writable) {
            throw std::runtime_error("RangeTensorSlice.at: Cannot execute at on a const base Tensor.\n");
        }
        // Ensure we get either one or two coordinates
        if (target.size() != 2) {
            throw std::invalid_argument("RangeTensorSlice.at: Invalid number of coordinates provided to at.\n");
        }
        // Strip the const qualifier on the std::shared_ptr<const Tensor<T>>
        auto rw_ptr = std::const_pointer_cast<Tensor<T>>(c_ptr);
        // Calculate the target index in the underlying Tensor and return the value
        return rw_ptr->at(_calculate_idx(target));
    }

    /**
     * Get a const reference to the value stored at the provided coordinates
     * @param target Initializer list containing the desired coordinates
     * @returns Returns a const reference to the desired value
     */
    const T& at(std::initializer_list<size_t> target) const override {
        // Ensure we get either one or two coordinates
        if (target.size() != 2) {
            throw std::invalid_argument("RangeTensorSlice.at: Invalid number of coordinates provided to at.\n");
        }
        // Calculate the target index in the underlying Tensor and return the value
        return c_ptr->at(_calculate_idx(target));
    }

    /**
     * Get a mutable reference to the value stored at the provided coordinates
     * @param target Const reference to a vector containing the desired coordinates
     * @returns Returns a mutable reference to the desired value
     */
    T& at(const std::vector<size_t>& target) override {
        if (!_is_writable) {
            throw std::runtime_error("RangeTensorSlice.at: Cannot execute at on a const base Tensor.\n");
        }
        if (target.size() != 2) {
            throw std::invalid_argument("RangeTensorSlice.at: Invalid number of coordinates provided to at.\n");
        }
        // Strip the const qualifier on the std::shared_ptr<const Tensor<T>>
        auto rw_ptr = std::const_pointer_cast<Tensor<T>>(c_ptr);
        return rw_ptr->at(_calculate_idx({target.at(0), target.at(1)}));
    }

    /**
     * Get a const reference to the value stored at the provided coordinates
     * @param target Const reference to a vector containing the desired coordinates
     * @returns Returns a const reference to the desired value
     */
    const T& at(const std::vector<size_t>& target) const override {
        if (target.size() != 2) {
            throw std::invalid_argument("RangeTensorSlice.at: Invalid number of coordinates provided to at.\n");
        }
        return c_ptr->at(_calculate_idx({target.at(0), target.at(1)}));
    }

    // NOLINTBEGIN(clang-diagnostic-unused-parameter)
    /**
     * Get a mutable reference to the value stored at the provided index
     * @param target Index to fetch from
     * @returns Returns a mutable reference to the desired value
     */
    T& at(size_t target) override {
        // Unconditionally throw an exception since we don't have any way to decode
        // the index into a usable translation for the underlying Tensor
        throw std::logic_error("RangeTensorSlice.at: at(size_t target) is not supported on TensorSlice.\n");
    }

    /**
     * Get a const reference to the value stored at the provided index
     * @param target Index to fetch from
     * @returns Returns a const reference to the desired value
     */
    const T& at(size_t target) const override {
        throw std::logic_error("RangeTensorSlice.at: at(size_t target) is not supported on TensorSlice.\n");
    }
    // NOLINTEND(clang-diagnostic-unused-parameter)

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    /**
     * Transpose a Tensor along two dims
     * @param dim0 First dim to swap
     * @param dim1 Second dim to swap
     */
    RangeTensorSlice<T>& transpose(size_t dim0, size_t dim1) override {
        // Ensure dim0 and dim1 are valid
        if (dim0 >= 2 || dim1 >= 2) {
            throw std::invalid_argument("RangeTensorSlice.transpose: Invalid dims provided.\n");
        }
        // Swap the maps
        std::swap(m_dim0_range, m_dim1_range);
        // Swap the slice dims
        std::swap(m_slice_dims.at(0), m_slice_dims.at(1));
        // Swap the strides
        std::swap(m_stride.at(0), m_stride.at(1));
        // Iterate over the DimInfo objects and rewrite the mappings
        for (size_t i = 0; i < c_tensor_rank; ++i) {
            DimInfo& d = m_other_dims.at(i);
            if (d.requires_rewrite) {
                if (d.rewrite_to == 0) {
                    d.rewrite_to = 1;
                }
                else {
                    d.rewrite_to = 0;
                }
            }
        }
        return *this;
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    /**
     * Get a string containing information about the underlying TensorSlice
     * @returns Returns a string with the Tensor's info
     */
    std::string info() const override {
        // Prepare the info string
        std::string result = std::format("TensorSlice: [{}, {}]. Underlying Tensor: {}", extent(0), extent(1), c_ptr->info());
        return result;
    }

    /**
     * Convert a TensorSlice to a string representation
     * @returns Returns a string representation of the TensorSlice
     */
    std::string to_string() const {
        // Ensure this can only run on 2-D TensorSlices
        if (rank() != 2) {
            throw std::logic_error("RangeTensorSlice.to_string: to_string() cannot be called on a non rank-2 TensorSlice.\n");
        }
        // Create a blank string that we will return
        std::string result = "\n";
        // Wrap the Tensor in brackets, with each row properly enclosed too
        result += "[";
        for (size_t i = 0; i < extent(0); ++i) {
            result += "[";
            for (size_t j = 0; j < extent(1); ++j) {
                // Add the value in the Matrix
                result += std::format("{}", at({i, j}));
                if (j + 1 < extent(1)) {
                    // Separate the values by tabs, for readability
                    result += "\t";
                }
            }
            // Close out each row with a corresponding ]
            result += "]";
            // Add a new line after each row if we aren't at the end
            if (i + 1 < extent(0)) {
                result += "\n";
            }
        }
        // Close out the TensorSlice final bracket and print out the info (dims and dtype)
        result += std::format("]. {}.", info());
        return result;
    }

    /**
     * Convert a TensorSlice to a Tensor, copying the data
     * @returns Returns a new Tensor with a copy of the data
     */
    Tensor<T> to_tensor() const {
        // Construct a 2-D Tensor
        Tensor<T> target({extent(0), extent(1)});
        for (size_t i = 0; i < extent(0); ++i) {
            for (size_t j = 0; j < extent(1); ++j) {
                target.at({i, j}) = this->at({i, j});
            }
        }
        return target;
    }
};

}; // namespace TensorSlice_NS

#endif
