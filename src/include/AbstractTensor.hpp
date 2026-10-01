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
 * @version: 2026-09-30
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef ABSTRACT_TENSOR_HPP
#define ABSTRACT_TENSOR_HPP

/* Standard dependencies */
#include <cstddef>
#include <expected>
#include <format>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

/* Local dependencies */
#include "Storage.hpp"

namespace AbstractTensor_NS {

/* Use the Storage class */
using Storage_NS::Storage;

// NOLINTBEGIN(cppcoreguidelines-special-member-functions)
/**
 * Abstract Tensor class, to implement both Tensor and the various
 * TensorSlice classes
 */
template <typename T>
requires std::is_arithmetic_v<T>
class AbstractTensor {
public:
    /**
     * Virtual destructor for AbstractTensor
     */
    virtual ~AbstractTensor() = default;

    /**
     * Get the rank of the Tensor
     * @returns Returns the rank
     */
    virtual size_t rank() const = 0;

    /**
     * Get the dimensions of the Tensor
     * @returns Returns a const reference to the vector containing the dimensions
     */
    virtual const std::vector<size_t>& shape() const = 0;

    /**
     * Get the stride used to advance inside the Tensor
     * @returns Returns a const reference to the vector containing the stide of each dim
     */
    virtual const std::vector<size_t>& stride() const = 0;

    /**
     * Get the extent of a specified dim
     * @param dim Dimension to query
     * @returns Returns the extent of the dim
     */
    virtual size_t extent(size_t dim) const = 0;

    /**
     * Get the total number of elements in the Tensor
     * @returns Returns the total number of elements
     */
    virtual size_t elements() const = 0;

    /**
     * Get a mutable reference to the value stored at the provided coordinates
     * @param target Initializer list containing the desired coordinates
     * @returns Returns a mutable reference to the desired value
     */
    virtual T& at(std::initializer_list<size_t> target) = 0;

    /**
     * Get a const reference to the value stored at the provided coordinates
     * @param target Initializer list containing the desired coordinates
     * @returns Returns a const reference to the desired value
     */
    virtual const T& at(std::initializer_list<size_t> target) const = 0;

    /**
     * Get a mutable reference to the value stored at the provided coordinates
     * @param target Const reference to a vector containing the desired coordinates
     * @returns Returns a mutable reference to the desired value
     */
    virtual T& at(const std::vector<size_t>& target) = 0;

    /**
     * Get a const reference to the value stored at the provided coordinates
     * @param target Const reference to a vector containing the desired coordinates
     * @returns Returns a const reference to the desired value
     */
    virtual const T& at(const std::vector<size_t>& target) const = 0;

    /**
     * Get a mutable reference to the value stored at the provided index
     * @param target Index to fetch from
     * @returns Returns a mutable reference to the desired value
     */
    virtual T& at(size_t target) = 0;

    /**
     * Get a const reference to the value stored at the provided index
     * @param target Index to fetch from
     * @returns Returns a const reference to the desired value
     */
    virtual const T& at(size_t target) const = 0;

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    /**
     * Transpose a Tensor along two dims
     * @param dim0 First dim to swap
     * @param dim1 Second dim to swap
     */
    virtual AbstractTensor<T>& transpose(size_t dim0, size_t dim1) = 0;
    // NOLINTEND(bugprone-easily-swappable-parameters)

    /**
     * Determine whether two AbstractTensors (with specified dims) can matmul
     * @param dim0 First dim
     * @param dim1 Second dim
     * @param target The other Tensor to check against
     * @param target_dim0 The target's first dim
     * @param target_dim1 The target's second dim
     */
    [[nodiscard]] std::expected<bool, std::string> _can_matmul(size_t dim0, size_t dim1, const AbstractTensor<T>& target,
                                                               size_t target_dim0, size_t target_dim1) const {
        // Ensure dim0 and dim1 are different
        if (dim0 == dim1 || target_dim0 == target_dim1) {
            return std::unexpected("AbstractTensor::_can_matmul: Caller / target dim0 and dim1 cannot be the same.\n");
        }
        // Ensure dim0 and dim1 are valid for the calling AbstractTensor
        const auto& caller_dims = shape();
        const auto& target_dims = target.shape();
        // We know that shape().size() must be equual to rank()
        size_t caller_rank = rank(), target_rank = target.rank();
        if (dim0 >= caller_rank || dim1 >= caller_rank) {
            return std::unexpected(
                std::format(
                    "AbstractTensor::_can_matmul: Invalid dims specified for caller AbstractTensor. Got {} and {} but have rank {}.\n",
                    dim0, dim1, caller_rank
                )
            );
        }
        if (target_dim0 >= target_rank || target_dim1 >= target_rank) {
            return std::unexpected(
                std::format(
                    "AbstractTensor::_can_matmul: Invalid dims specified for target AbstractTensor. Got {} and {} but have rank {}.\n",
                    dim0, dim1, caller_rank
                )
            );
        }
        // Ensure the inner dims are the same size
        if (caller_dims.at(dim1) != target_dims.at(target_dim0)) {
            return std::unexpected(
                std::format(
                    "AbstractTensor::_can_matmul: Incompatible dims. Got [{}, {}] x [{}, {}], {} != {}.\n",
                    caller_dims.at(dim0), caller_dims.at(dim1), target_dims.at(target_dim0), target_dims.at(target_dim1),
                        caller_dims.at(dim1), target_dims.at(target_dim0)
                )
            );
        }
        return true;
    }

    /**
     * Get a string containing information about the underlying Tensor
     * @returns Returns a string with the Tensor's info
     */
    virtual std::string info() const = 0;

    /**
     * Check whether a Tensor's memory block is contiguous
     * @returns True if contiguous
     */
    virtual bool contiguous() const = 0;

    /**
     * Get the offset for the start of the Storage memory block
     * @returns Returns the size_t offset
     */
    virtual size_t offset() const = 0;

    /**
     * Get a pointer to the raw memory block underpinning a Tensor
     * @returns Returns a pointer to the start of the Storage memory block
     */
    virtual T* _data() = 0;

    /**
     * Get a const pointer to the raw memory block underpinning a Tensor
     * @returns Returns a pointer to the start of the Storage memory block
     */
    virtual const T* _data() const = 0;

    /**
     * Get a reference to the Storage object underpinning a Tensor
     * @returns Returns a mutable reference to the Storage object
     */
    virtual Storage<T>& _storage() = 0;

    /**
     * Get a const reference to the Storage object underpinning a Tensor
     * @returns Returns a const reference to the Storage object
     */
    virtual const Storage<T>& _storage() const = 0;
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

}; // namespace AbstractTensor_NS

#endif
