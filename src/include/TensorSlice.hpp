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
 * @version: 2026-10-05
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef TENSORSLICE_HPP
#define TENSORSLICE_HPP

/* Standard dependencies */
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <vector>

/* Local dependencies */
#include "AbstractTensor.hpp"
#include "DimInfo.hpp"
#include "Log.hpp"
#include "Numerics.hpp"
#include "SliceConfig.hpp"
#include "Storage.hpp"
#include "Tensor.hpp"

namespace TensorSlice_NS {

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;

/* Use DimInfo from DimInfo_NS */
using DimInfo_NS::DimInfo;

/* Use logging functions */
using Log::log_message;
using Log::Log_Priority;

/* Use Tensor and helper functions from Tensor_NS */
using Tensor_NS::AbstractTensor;
using Tensor_NS::Tensor;
using Tensor_NS::Storage;

/* Use enum classes from SliceConfig_NS */
using SliceConfig_NS::VectorSliceOrientation;
using SliceConfig_NS::IndexType;
using SliceConfig_NS::SliceConfig;
using SliceConfig_NS::VectorSliceConfig;
using SliceConfig_NS::MatrixSliceConfig;

/* Use the Storage class */
using Storage_NS::Storage;

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
     * Track whether or not the ListTensorSlice has been transposed. Use this to flip
     * coordinates instead of moving around the internal data of the slice
     */
    bool _transposed = false;

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
            // If we have applied a transpose to this ListTensorSlice, flip the coordinates
            size_t c0 = c.begin()[0];
            size_t c1 = c.begin()[1];
            if (_transposed) {
                std::swap(c0, c1);
            }
            // Determine the rank of the Tensor
            if (c_tensor_rank == 2) {
                const auto& dim0_rules = m_other_dims.at(0);
                // Check to see which dim is rewritten to 0
                if (dim0_rules.rewrite_to == 0) {
                    if (!m_dim1_map.empty()) {
                        return (tensor_stride.at(0) * m_dim0_map.at(c0)) + m_dim1_map.at(c1);
                    }
                    return (tensor_stride.at(0) * m_dim0_map.at(c0)) + c1;
                }
                // Otherwise flip the mapping
                if (!m_dim1_map.empty()) {
                    return (tensor_stride.at(0) * m_dim1_map.at(c1)) + m_dim0_map.at(c0);
                }
                return (tensor_stride.at(0) * c1) + m_dim0_map.at(c0);
            }
            // Deal with a high-rank Tensor
            size_t target_idx = 0;
            // Iterate over the dimensions and rewrite into a query for the underlying Tensor.
            // We already validated that m_other_dims matches the Tensor's rank
            for (size_t i = 0; i < c_tensor_rank; ++i) {
                const auto& dim_rule = m_other_dims.at(i);
                if (dim_rule.requires_rewrite) {
                    if (dim_rule.rewrite_to == 0) {
                        target_idx += tensor_stride.at(i) * m_dim0_map.at(c0);
                    }
                    // If not rewrite_to == 0, then must be == 1
                    else {
                        if (!m_dim1_map.empty()) {
                            target_idx += tensor_stride.at(i) * m_dim1_map.at(c1);
                        }
                        else {
                            target_idx += tensor_stride.at(i) * c1;
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
     * Use _can_matmul from the AbstractTensor base class
     */
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

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
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
        if (rank() == 1) {
            return m_slice_dims[0] > 1 ? m_slice_dims[0] : m_slice_dims[1];
        }
        return m_slice_dims.at(dim);
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)

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
        if (dim0 >= 2 || dim1 >= 2 || dim0 == dim1) {
            throw std::invalid_argument("ListTensorSlice.transpose: Invalid dims provided.\n");
        }
        // If we have a 1-D TensorSlice, only swap the dims and then return
        std::swap(m_slice_dims.at(0), m_slice_dims.at(1));
        if (_transposed) {
            _transposed = false;
        }
        else {
            _transposed = true;
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

    /**
     * Check whether a Tensor's memory block is contiguous
     * @returns True if contiguous
     */
    bool contiguous() const override {
        return false;
    }

    /**
     * Get the offset for the start of the Storage memory block
     * @returns Returns the size_t offset
     */
    size_t offset() const override {
        return c_ptr->offset();
    }

    /**
     * Get a pointer to the raw memory block underpinning a Tensor
     * @returns Returns a pointer to the start of the Storage memory block
     */
    T* _data() override {
        if (!_is_writable) {
            throw std::runtime_error("ListTensorSlice._data(): Cannot execute _data() on a const base Tensor.\n");
        }
        // Strip the const qualifier on the std::shared_ptr<const Tensor<T>>
        auto rw_ptr = std::const_pointer_cast<Tensor<T>>(c_ptr);
        return rw_ptr->_data();
    }

    /**
     * Get a const pointer to the raw memory block underpinning a Tensor
     * @returns Returns a pointer to the start of the Storage memory block
     */
    const T* _data() const override {
        return c_ptr->_data();
    }

    /**
     * Get a reference to the Storage object underpinning a Tensor
     * @returns Returns a mutable reference to the Storage object
     */
    Storage<T>& _storage() override {
        if (!_is_writable) {
            throw std::runtime_error("ListTensorSlice._storage(): Cannot execute _storage() on a const base Tensor.\n");
        }
        // Strip the const qualifier on the std::shared_ptr<const Tensor<T>>
        auto rw_ptr = std::const_pointer_cast<Tensor<T>>(c_ptr);
        return rw_ptr->_storage();
    }

    /**
     * Get a const reference to the Storage object underpinning a Tensor
     * @returns Returns a const reference to the Storage object
     */
    const Storage<T>& _storage() const override {
        return c_ptr->_storage();
    }

    /**
     * Rematerialize a Tensor, guaranteeing that it is contiguous
     * with a stide of 1 in the final dim
     * @returns Returns a new contiguous Tensor
     */
    Tensor<T>& remat() override {
        throw std::runtime_error("ListTensorSlice: remat is unimplemented for ListTensorSlice.\n");
    }
};

}; // namespace TensorSlice_NS

#endif
