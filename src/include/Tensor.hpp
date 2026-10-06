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
 * Tensor.hpp: The core Tensor library at the heart of most operations for build-llm-cpp.
 * The Tensor inherits from AbstractTensor (interface for Tensor + TensorSlice).
 * Tensor uses a storage class (Storage<T>) for interacting with the underlying heap-allocated
 * memory block.
 * 
 * Most mathematical operations live inside TensorMath.hpp; however, Tensor.hpp handles
 * cases where vectorization either can't happen or is overly complicated to implement.
 * 
 * Matmul kernels are availabel in TensorMatmul.hpp
 * 
 */

#ifndef TENSOR_HPP
#define TENSOR_HPP

/* Standard dependencies */
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cxxabi.h>
#include <expected>
#include <format>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <vector>

/* Local dependencies */
#include "AbstractTensor.hpp"
#include "Log.hpp"
#include "Numerics.hpp"
#include "Storage.hpp"
#include "TensorMath.hpp"

namespace Tensor_NS {

/* Control verbose copy / move constructor logging, useful for debugging. Does not apply to default constructors */
inline constexpr bool TENSOR_ENABLE_CONSTRUCTOR_LOGGING = true;
/* Store the maximum length for a type name from the demangler API */
inline constexpr size_t TENSOR_MAX_DEMANGLED_NAME_LEN = 32;
/* Control logging for softmax warnings (NAN, inf, divide by 0) */
inline constexpr bool TENSOR_ENABLE_SOFTMAX_WARNINGS = true;
/* Control logging about remat being called on already contiguous Tensors (with dim_stride == 1 in outer dim) */
inline constexpr bool TENSOR_ENABLE_UNNECESSARY_REMAT_LOGGING = true;

/* Use Logging functions */
using Log::Log_Priority;
using Log::log_message;

/* Use the AbstractTensor interface */
using AbstractTensor_NS::AbstractTensor;

/* Use the Storage class */
using Storage_NS::Storage;

/**
 * Enum for controlling negative infinity masking for causal attention, only
 * applicable to square Tensors
 */
enum class CausalMaskType : uint8_t {
    UPPER,
    LOWER
};

/**
 * Enum for operations with squeezed Tensors
 */
enum class SqueezedOpType : uint8_t {
    ADD,
    SUB,
    MUL,
    DIV
};

/**
 * Convert SqueezedOpType to a string view
 * @param op SqueededOpType
 * @returns Returns a std::string_view with the string representation of the type
 */
constexpr std::string_view squeezed_op_to_string(SqueezedOpType op) {
    switch (op) {
        case SqueezedOpType::ADD: return "ADD";
        case SqueezedOpType::SUB: return "SUB";
        case SqueezedOpType::MUL: return "MUL";
        case SqueezedOpType::DIV: return "DIV";
    }
    return "UNKOWN OPERATION";
}

/* Forward declaration for Tensor */
template <typename T> 
requires std::is_arithmetic_v<T>
class Tensor;

/* Forward declarations for uniform_op */
/**
 * Apply a uniform basic math operation over a Tensor, reducing the amount of boilerplate
 * code required for the various operators and friends
 * @param lhs Const reference to a Tensor to serve as the lefthand side
 * @param rhs Const reference to a Tensor to serve as the righthand side
 * @param dest Reference to a Tensor to write the result to
 * @param op Target operation to perform
 * @returns Returns a reference to the destination Tensor
 */
template <typename T> 
requires std::is_arithmetic_v<T>
Tensor<T>& uniform_op(const Tensor<T>& lhs, const Tensor<T>& rhs, Tensor<T>& dest, SqueezedOpType op);

/**
 * Apply a uniform basic math operation over a Tensor, reducing the amount of boilerplate
 * code required for the various operators and friends
 * @param lhs Const reference to a Tensor to serve as the lefthand side
 * @param rhs_val Scalar value to perform the op with lhs
 * @param dest Reference to a Tensor to write the result to
 * @param op Target operation to perform
 * @returns Returns a reference to the destination Tensor
 */
template <typename T> 
requires std::is_arithmetic_v<T>
Tensor<T>& uniform_op(const Tensor<T>& lhs, T rhs_val, Tensor<T>& dest, SqueezedOpType op);

/* Forward declarations for unsafe_uniform_op */
/**
 * Apply a uniform basic math operation over a Tensor, reducing the amount of boilerplate
 * code required for the various operators and friends
 * @param lhs Const reference to a Tensor to serve as the lefthand side
 * @param rhs Const reference to a Tensor to serve as the righthand side
 * @param dest Reference to a Tensor to write the result to
 * @param op Target operation to perform
 * @returns Returns a reference to the destination Tensor
 */
template <typename T> 
requires std::is_floating_point_v<T>
Tensor<T>& unsafe_uniform_op(const Tensor<T>& lhs, const Tensor<T>& rhs, Tensor<T>& dest, SqueezedOpType op);

/**
 * Apply a uniform basic math operation over a Tensor, reducing the amount of boilerplate
 * code required for the various operators and friends
 * @param lhs Const reference to a Tensor to serve as the lefthand side
 * @param rhs_val Scalar value to perform the op with lhs
 * @param dest Reference to a Tensor to write the result to
 * @param op Target operation to perform
 * @returns Returns a reference to the destination Tensor
 */
template <typename T> 
requires std::is_floating_point_v<T>
Tensor<T>& unsafe_uniform_op(const Tensor<T>& lhs, T rhs_val, Tensor<T>& dest, SqueezedOpType op);

// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays, cppcoreguidelines-pro-bounds-pointer-arithmetic, cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
/**
 * Template for a generic Tensor, where T can be any numeric type, like a variety of int,
 * double, float, etc.
 */
template <typename T> 
requires std::is_arithmetic_v<T>
class Tensor final : public AbstractTensor<T> {
/* Private data elements */
private:
    /**
     * Instance of Storage<T>, which owns the memory being used by the Tensor. We wrap Storage in
     * std::shared_ptr to manage its lifetime and deallocate when all Tensor / TensorSlice instances
     * go out of scope
     */
    std::shared_ptr<Storage<T>> m_data = nullptr;

    /**
     * Offset to begin accessing m_data.m_data, useful when creating a TensorSlice
     */
    size_t c_offset = 0;

    /**
     * Total number of elements stored in the Tensor
     */
    size_t c_elements = 1;

    /**
     * Tensor rank, i.e. how many dimensions a Tensor represents, anywhere from 0..N, where rank 0 represents
     * a single scalar value, rank 1 is a vector of values, rank 2 is a matrix, and so on
     */
    size_t c_rank = 0;

    /**
     * Tensor stride, i.e. how many values we need to jump forward between rows, columns, etc. across total
     * tensor rank / dimension
     */
    std::vector<size_t> m_stride = std::vector<size_t>();

    /**
     * Tensor dimensions, sequenced from outer to inner dim
     */
    std::vector<size_t> m_dims = std::vector<size_t>();

    /**
     * Whether or not the Tensor's underlying m_data is contiguous, set to true by default.
     * This will not be true if creating a TensorSlice
     */
    bool c_contiguous = true;
    
/* Public functions */
public:
    /**
     * Use _can_matmul from the AbstractTensor base class
     */
    using AbstractTensor<T>::_can_matmul;

    /**
     * Default constructor for Tensor, taking in an intializer list containing th dimensions
     * for the resulting Tensor
     * @param dims std::initializer_list<size_t> containing the desired dimensions
     */
    Tensor(std::initializer_list<size_t> dims) : c_rank(dims.size()), m_stride(dims.size()), m_dims(dims) {
        // Handle the case where we have a rank-0 tensor (scalar value). Allocate space for the singular
        // element and then return immediately
        if (c_rank == 0) {
            m_data = std::make_shared<Storage<T>>(1);
            return;
        }
        // Store the target 1-D representation of the desired size of our Tensor
        for (const size_t& ds : dims) {
            if (ds == 0) {
                throw std::invalid_argument(std::format("Tensor.Tensor: invalid dim ({}) provided to Tensor. Dims must be >= 1 or should be omitted.", ds));
            }
            c_elements *= ds;
        }
        // Allocate the block of memory for the Tensor
        m_data = std::make_shared<Storage<T>>(c_elements);
        // Calculate the stides needed to get between dims
        if (c_rank == 1) {
            m_stride.at(0) = 1;
        }
        else if (c_rank == 2) {
            m_stride.at(0) = m_dims.at(1);
            m_stride.at(1) = 1;
        }
        else {
            m_stride.at(c_rank - 1) = 1;
            for (size_t i = c_rank - 1; i > 0; --i) {
                m_stride.at(i - 1) = m_stride.at(i) * m_dims.at(i);
            }
        }
    }

    /**
     * Constructor for Tensor, taking in a reference to a vector containing the desired dimensions
     * for the resulting Tensor
     * @param dims std::vector<size_t> containing the desired dimensions
     */
    explicit Tensor(const std::vector<size_t>& dims) : c_rank(dims.size()), m_stride(dims.size()), m_dims(dims) {
        // Handle the case where we have a rank-0 tensor (scalar value). Allocate space for the singular
        // element and then return immediately
        if (c_rank == 0) {
            m_data = std::make_shared<Storage<T>>(1);
            return;
        }
        // Store the target 1-D representation of the desired size of our Tensor
        for (const size_t& ds : dims) {
            if (ds == 0) {
                throw std::invalid_argument(std::format("Tensor.Tensor: invalid dim ({}) provided to Tensor. Dims must be >= 1 or should be omitted.", ds));
            }
            c_elements *= ds;
        }
        // Allocate the block of memory for the Tensor
        m_data = std::make_shared<Storage<T>>(c_elements);
        // Calculate the stides needed to get between dims
        if (c_rank == 1) {
            m_stride.at(0) = 1;
        }
        else if (c_rank == 2) {
            m_stride.at(0) = m_dims.at(1);
            m_stride.at(1) = 1;
        }
        else {
            m_stride.at(c_rank - 1) = 1;
            for (size_t i = c_rank - 1; i > 0; --i) {
                m_stride.at(i - 1) = m_stride.at(i) * m_dims.at(i);
            }
        }
    }

    /**
     * Copy constructor for Tensor, creating a shallow copy, which just increases the reference
     * count in the shared pointer to m_data
     * @param target Tensor to make a copy of
     */
    Tensor(const Tensor<T>& target) : m_data(target.m_data), c_offset(target.c_offset), c_elements(target.c_elements), 
                                      c_rank(target.c_rank), m_stride(target.m_stride), m_dims(target.m_dims), c_contiguous(target.c_contiguous) {
        // Debug logging
        if (TENSOR_ENABLE_CONSTRUCTOR_LOGGING) {
            log_message(Log_Priority::DEBUG, "Tensor.Tensor", "Copy constructor called");
        }
    }

    /**
     * Copy assignment operator, creating a shallow copy (without copying data block)
     * @param target Tensor to make a copy of
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& operator=(const Tensor<T>& target) {
        // Ensure we aren't calling the assignment operator on ourself
        if (this == &target) {
            return *this;
        }
        // Copy data elements into the Tensor
        m_data = target.m_data;
        c_offset = target.c_offset;
        c_elements = target.c_elements;
        c_rank = target.c_rank;
        m_stride = std::vector<size_t>(target.m_stride);
        m_dims = std::vector<size_t>(target.m_dims);
        c_contiguous = target.c_contiguous;
        // Debug logging
        if (TENSOR_ENABLE_CONSTRUCTOR_LOGGING) {
            log_message(Log_Priority::DEBUG, "Tensor.Tensor", "Copy assignment called");
        }

        return *this;
    }

    /**
     * Move constructor for Tensor, taking the data from target
     * @param target Tensor to move data out of
     */
    Tensor(Tensor<T>&& target) noexcept : m_data(std::move(target.m_data)), c_offset(target.c_offset), c_elements(target.c_elements),
                                          c_rank(target.c_rank), m_stride(std::move(target.m_stride)), m_dims(std::move(target.m_dims)),
                                          c_contiguous(target.c_contiguous) {
        // Set target.m_data to nullptr
        target.m_data = nullptr;
        // Debug logging
        if (TENSOR_ENABLE_CONSTRUCTOR_LOGGING) {
            log_message(Log_Priority::DEBUG, "Tensor.Tensor", "Move constructor called");
        }
    }

    /**
     * Move assignment operator, takeing the data from the target and overwriting this Tensor
     * @param target Tensor to move the data out of
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& operator=(Tensor<T>&& target) noexcept {
        // Ensure we aren't calling the assignment operator on ourself
        if (this == &target) {
            return *this;
        }
        // Copy trivial data elements into this Tensor
        c_offset = target.c_offset;
        c_elements = target.c_elements;
        c_rank = target.c_rank;
        c_contiguous = target.c_contiguous;
        // Move eligible vectors
        m_stride = std::move(target.m_stride);
        m_dims = std::move(target.m_dims);
        // Move the memory block
        m_data = std::move(target.m_data);
        // Set target.m_data to nullptr for good measure
        target.m_data = nullptr;
        // Debug logging
        if (TENSOR_ENABLE_CONSTRUCTOR_LOGGING) {
            log_message(Log_Priority::DEBUG, "Tensor.Tensor", "Move assignment operator called");
        }

        return *this;
    }

    /**
     * Default destructor for Tensor
     */
    ~Tensor() override {
        // Do nothing since all of our data elements are either trivial types or will be cleaned up
        // automatically when they go out of scope
    }

    /**
     * Get the number of elements in a Tensor
     * @returns Returns size_t of the number of elements present in the Tensor
     */
    size_t elements() const override {
        return c_elements;
    }

    /**
     * Get the rank of a Tensor
     * @returns Returns size_t of the Tensor's rank
     */
    size_t rank() const override {
        return c_rank;
    }

    /**
     * Get the dims of a Tensor
     * @returns Returns a const ref to m_dims
     */
    const std::vector<size_t>& dims() const {
        return m_dims;
    }

    /**
     * Get the dimensions of the Tensor
     * @returns Returns a const reference to the vector containing the dimensions
     */
    const std::vector<size_t>& shape() const override {
        return m_dims;
    }

    /**
     * Get the stride of a Tensor
     * @returns Returns a const ref to m_stride
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
        // Ensure the dim is valid
        if (dim >= c_rank) {
            throw std::invalid_argument("Tensor.extent: Invalid dim provided.\n");
        }
        return m_dims.at(dim);
    }

    /**
     * Get the stride of a dim in a Tensor
     * @param dim Dimension to check
     * @returns Returns the stride of the specified dim
     */
    size_t dim_stride(size_t dim) const {
        // Ensure the dim is valid
        if (dim >= c_rank) {
            throw std::invalid_argument("Tensor.dim_stride: Invalid dim provided.\n");
        }
        // m_stride always has size() == c_rank
        return m_stride[dim];
    }

    /**
     * Check whether a Tensor's memory block is contiguous
     * @returns True if contiguous
     */
    bool contiguous() const override {
        return c_contiguous;
    }

    /**
     * Get the offset for the start of the Storage memory block
     * @returns Returns the size_t offset
     */
    size_t offset() const override {
        return c_offset;
    }

    /**
     * Get a pointer to the raw memory block underpinning a Tensor
     * @returns Returns a pointer to the start of the Storage memory block
     */
    T* _data() override {
        return m_data->m_data.get();
    }

    /**
     * Get a const pointer to the raw memory block underpinning a Tensor
     * @returns Returns a pointer to the start of the Storage memory block
     */
    const T* _data() const override {
        return m_data->m_data.get();
    }

    /**
     * Get a reference to the Storage object underpinning a Tensor
     * @returns Returns a mutable reference to the Storage object
     */
    Storage<T>& _storage() override {
        return *m_data;
    }

    /**
     * Get a const reference to the Storage object underpinning a Tensor
     * @returns Returns a const reference to the Storage object
     */
    const Storage<T>& _storage() const override {
        return *m_data;
    }

    /**
     * Rematerialize a Tensor, guaranteeing that it is contiguous
     * with a stide of 1 in the final dim
     * @returns Returns a new contiguous Tensor
     */
    Tensor<T>& remat() override {
        // Don't try to remat rank-0 Tensors
        if (c_rank == 0) {
            return *this;
        }
        // If this Tensor already is contiguous and has stride == 1 in its
        // final dim, then return without any op
        if (c_contiguous && (dim_stride(c_rank - 1) == 1)) {
            if constexpr (TENSOR_ENABLE_UNNECESSARY_REMAT_LOGGING) {
                log_message(Log_Priority::WARNING, "Tensor.remat",
                            "Remat called on a Tensor that is already contiguous and with outer dim stride == 1.");
            }
            return *this;
        }
        // Create a new, destination Tensor with the same dims and elements
        // By definition, this will create a contiguous Tensor with a new
        // Storage allocation
        Tensor<T> target(m_dims);
        // Copy the contents of this Tensor into target
        target.copy_from(*this);
        // Steal the Storage shared pointer and let target go out of scope
        m_data = std::move(target.m_data);
        m_stride = std::move(target.m_stride);
        c_offset = target.c_offset;
        c_contiguous = target.c_contiguous;
        return *this;
    }

    /**
     * Calculate the offset from a set of input cooridnates
     * @param c Coordinates to calculate the offset from
     * @returns Returns a size_t representing the offset we want
     */
    [[nodiscard]] size_t _get_offset(std::initializer_list<size_t> c) const {
        // Do some pointer arithmetic to calculate the exact address to retrieve from m_data
        size_t target_offset = c_offset;
        // Ensure the coordinates are valid for our Tensor
        for (size_t i = 0; i < c_rank; ++i) {
            if (c.begin()[i] >= extent(i)) {
                throw std::out_of_range(std::format("Tensor._get_offset: Index {} exceeds end dim ({}).", c.begin()[i], extent(i)));
            }
            target_offset += c.begin()[i] * m_stride.at(i);
        }
        if (target_offset >= m_data->c_elements) {
            throw std::out_of_range(std::format("Tensor._get_offset: Index {} exceeds end of Storage.", target_offset));
        }
        return target_offset;
    }

    /**
     * Calculate the offset from a set of input cooridnates
     * @param c Coordinates (wrapped in std::vector) to calculate the offset from
     * @returns Returns a size_t representing the offset we want
     */
    [[nodiscard]] size_t _get_offset(const std::vector<size_t>& c) const {
        // Do some pointer arithmetic to calculate the exact address to retrieve from m_data
        size_t target_offset = c_offset;
        // Ensure the coordinates are valid for our Tensor
        for (size_t i = 0; i < c_rank; ++i) {
            if (c[i] >= extent(i)) {
                throw std::out_of_range(std::format("Tensor._get_offset: Index {} exceeds end dim ({}).", c[i], extent(i)));
            }
            target_offset += c.at(i) * m_stride.at(i);
        }
        if (target_offset >= m_data->c_elements) {
            throw std::out_of_range(std::format("Tensor._get_offset: Index {} exceeds end of Storage.", target_offset));
        }
        return target_offset;
    }

    /**
     * Calculate the offset for a specific element
     * @param target_idx Target index to get the offset for
     * @returns Returns a size_t representing the offset we want
     */
    [[nodiscard]] size_t _get_offset(size_t target_idx) const {
        // Ensure we have a valid index
        if (target_idx >= c_elements) {
            throw std::out_of_range(std::format("Tensor._get_offset: Index {} exceeds number of elements.", target_idx));
        }
        // If the memory is contiguous, return the target index + base offset
        // The previous check also prevents index > 0 for Tensors with rank == 0
        if (c_contiguous || c_rank == 0) {
            return c_offset + target_idx;
        }
        // Traverse the Tensor shape and strides to get the target offset
        size_t current_idx = target_idx;
        size_t current_offset = c_offset;
        for (size_t i = c_rank; i-- > 0; ) {
            size_t idx = current_idx % m_dims[i];
            current_offset += idx * m_stride[i];
            current_idx /= m_dims[i];
        }
        if (current_offset >= m_data->c_elements) {
            throw std::out_of_range(std::format("Tensor._get_offset: Index {} exceeds end of Storage.", current_offset));
        }
        return current_offset;
    }

    /**
     * Determine if another Tensor has a shape compatible with this one for other operations
     * @param target The other Tensor to check against
     * @returns Returns true if the dims are compatible, false otherwise
     */
    [[nodiscard]] bool _compatible(const AbstractTensor<T>& target) const {
        // If Tensor rank isn't the same, we quickly know the Tensors aren't compatible
        if (c_rank != target.rank()) {
            return false;
        }
        // If the Tensors don't have the same number of elements, we also can easily
        // flag them as incompatible
        if (c_elements != target.elements()) {
            return false;
        }
        // Iterate over the dimensions and validate each matches
        for (size_t i = 0; i < c_rank; ++i) {
            if (m_dims[i] != target.extent(i)) {
                return false;
            }
        }

        return true;
    }

    /**
     * Clone a Tensor, creating a deep copy and a new memory block for the Storage object
     * @returns Returns a new Tensor with the same data as the caller
     */
    Tensor<T> clone() const {
        // Allocate a new Tensor, copying the shape from this one
        Tensor<T> target(m_dims);
        // If the caller is contiguous, use memcpy to copy the underlying data all at once
        if (c_contiguous) {
            std::memcpy(target._data(), _data() + c_offset, c_elements * sizeof(T));
        }
        else {
            T* target_data = target._data();
            const T* self_data = _data();
            for (size_t i = 0; i < c_elements; ++i) {
                target_data[i] = self_data[_get_offset(i)];
            }
        }
        return target;
    }

    /**
     * Copy the contents of another Tensor into this Tensor
     * @param target Tensor to copy from
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& copy_from(const Tensor<T>& target) {
        if (!_compatible(target)) {
            throw std::invalid_argument("Tensor.copy_from: Invalid target Tensor provided to copy_from.\n");
        }
        // Ensure we don't copy if we receive the same Tensor or Storage
        if (this == &target || (_data() == target._data() && c_offset == target.c_offset && is_same_layout(target))) {
            return *this;
        }
        // Ensure the Tensors Storage blocks don't have unsafe aliasing
        if (!is_safe_aliasing(target)) {
            Tensor<T> temp_target = target.clone();
            return copy_from(temp_target);
        }
        // If both are unique Tensors and contiguous, use memcpy
        if ((c_contiguous && target.c_contiguous) && is_unique(target)) {
            std::memcpy(_data() + c_offset, target._data() + target.c_offset, c_elements * sizeof(T));
        }
        // Otherwise copy the elements one by one
        else {
            T* self_data = _data();
            const T* target_data = target._data();
            for (size_t i = 0; i < c_elements; ++i) {
                self_data[_get_offset(i)] = target_data[target._get_offset(i)];
            }
        }
        return *this;
    }

    /**
     * Copy the contents of another AbstractTensor into this Tensor
     * @param target AbstractTensor to copy from
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& copy_from(const AbstractTensor<T>& target) {
        if (!_compatible(target)) {
            throw std::invalid_argument("Tensor.copy_from: Invalid target AbstractTensor provided to copy_from.\n");
        }
        // Ensure the AbstractTensor has rank <= 2
        if (target.rank() > 2) {
            throw std::invalid_argument("Tensor.copy_from: Target AbstractTensor must have rank <= 2.\n");
        }
        // Ensure we don't copy if we receive the same Tensor or Storage
        if (this == &target || (_data() == target._data() && offset() == target.offset() && is_same_layout(target))) {
            return *this;
        }
        // Ensure the Tensors Storage blocks don't have unsafe aliasing
        if (!is_safe_aliasing(target)) {
            // Clone this Tensor and call copy_from on it, ensuring we have safe aliasing
            Tensor<T> temp_target = clone();
            temp_target.copy_from(target);
            // Once the copy is done, then move the result values into this Tensor
            T* self_data = _data();
            const T* target_data = temp_target._data();
            for (size_t i = 0; i < c_elements; ++i) {
                self_data[_get_offset(i)] = target_data[i];
            }
            return *this;
        }
        // If both are unique Tensors and contiguous, use memcpy
        if ((c_contiguous && target.contiguous()) && is_unique(target)) {
            std::memcpy(_data() + c_offset, target._data() + target.offset(), c_elements * sizeof(T));
        }
        // Otherwise copy the elements one by one
        else {
            if (c_rank == 0) {
                at({}) = target.at({});
            }
            else if (c_rank == 1) {
                for (size_t i = 0; i < extent(0); ++i) {
                    at({i}) = target.at({i});
                }
            }
            else {
                for (size_t i = 0; i < extent(0); ++i) {
                    for (size_t j = 0; j < extent(1); ++j) {
                        at({i, j}) = target.at({i, j});
                    }
                }
            }
        }
        return *this;
    }

    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
    /**
     * Check to see if a Tensor is the same as this Tensor, or if the memory blocks overlap
     * with each other
     * @param target Tensor to validate uniqueness with
     * @returns Returns true if the Tensors are unique and do not have overlapping memory
     */
    bool is_unique(const AbstractTensor<T>& target) const {
        // Easiest check to make sure we aren't pointing at the same object
        if (this == &target) {
            return false;
        }
        // If the base pointers are different in m_data->m_data.get(), we
        // must have unique underlying Storage
        if (_data() != target._data()) {
            return true;
        }
        // Determine whether or not the memory blocks overlap based on c_elements
        auto get_memory_range = [](const AbstractTensor<T>& t) {
            uintptr_t start = reinterpret_cast<uintptr_t>(t._data() + t.offset());
            size_t element_span = t.elements();
            // If non-contiguous, compute true span using dimension extents and strides
            if (!t.contiguous() && t.rank() > 0) {
                element_span = 0;
                const auto& strides = t.stride();
                for (size_t i = 0; i < t.rank(); ++i) {
                    if (t.extent(i) > 0) {
                        element_span += (t.extent(i) - 1) * strides.at(i);
                    }
                }
                element_span += 1; // Include final element
            }

            uintptr_t end = start + (element_span * sizeof(T));
            return std::make_pair(start, end);
        };
        // Calculate the ranges for each Tensor
        auto [data_start, data_end] = get_memory_range(*this);
        auto [target_start, target_end] = get_memory_range(target);
        return std::max(data_start, target_start) >= std::min(data_end, target_end);
    }
    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

    /**
     * Check if two Tensors have the same rank, dims, and strides
     * @param target Const ref to an AbstractTensor to check against this one
     * @returns True if dims and strides exactly match
     */
    bool is_same_layout(const AbstractTensor<T>& target) const {
        // c_elements can reveal if any of the fields are not the same
        if (c_elements != target.elements()) { return false; }
        // Ensure the rank is the same
        if (c_rank != target.rank()) { return false; }
        // Get the stides for target
        const auto& strides = target.stride();
        // Check the dims and strides match, m_dims and m_stride always have size() == c_rank
        for (size_t i = 0; i < c_rank; ++i) {
            if (m_dims[i] != target.extent(i)) { return false; }
            if (m_stride[i] != strides.at(i)) { return false; }
        }
        return true;
    }

    /**
     * Check aliasing safety for two Tensors. We either want the Tensors to be unique
     * per the is_unique call, or ensure the Tensors are exactly the same, with the same
     * Storage, offset, shape, and stride
     * @param target Const ref to a Tensor to check for safe aliasing
     * @returns True if the Tensors are unique or have Storage config
     */
    bool is_safe_aliasing(const AbstractTensor<T>& target) const {
        // If we have two completely unique Tensors, return true
        if (is_unique(target)) { return true; }
        // Otherwise check that Storage, offset, and layout are the same
        if ((_data() == target._data()) && (c_offset == target.offset()) && is_same_layout(target)) {
            return true;
        }
        return false;
    }

    /**
     * Get a string representing information about a Tensor, including its dimenstions and underlying type
     * @returns Returns an info string about the Tensor
     */
    std::string info() const override {
        // Setup a base string to add information to
        std::string result = "Tensor: [";
        // Iterate over the dims and concatenate them to the result string
        for (const auto& d : m_dims) {
            result += std::format("{}, ", d);
        }
        // Remove the trailing ", " from the last dim
        if (c_rank > 0) {
            result.erase(result.size() - 2, 2);
        }
        // Allocate an empty buffer to store the name
        std::array<char, TENSOR_MAX_DEMANGLED_NAME_LEN> demangled = {0};
        // Ensure we don't go past the end of the buffer
        size_t buflen = TENSOR_MAX_DEMANGLED_NAME_LEN;
        int status = 0;
        // Use the C++ ABI to demangle the type name
        abi::__cxa_demangle(typeid(T).name(), demangled.data(), &buflen, &status);
        if (status == 0) {
            result += std::format("], dtype={}", demangled.data());
        }
        else {
            result += std::format("], dtype={}", typeid(T).name());
        }
        result += std::format(", contiguous={}", c_contiguous ? "true" : "false");
        return result;
    }

    /**
     * Get or set a value at a specific coordinate inside the Tensor
     * @param target The coordinate (wrapped in std::vector) we want to fetch from the Tensor
     * @returns Returns a reference to the value that can be updated
     */
    T& at(const std::vector<size_t>& target) override {
        // Handle the case where we have a rank-0 tensor
        if (c_rank == 0) {
            return _data()[c_offset];
        }
        // Validate we have the correct number of elements in our coordinates
        if (target.size() != c_rank) {
            throw std::invalid_argument(std::format("Tensor.at: Invalid number of coordinates provided to at, expected {} but got {}.\n", c_rank, target.size()));
        }
        // Calculate the offset and return a reference
        return _data()[_get_offset(target)];
    }

    /**
     * Get or set a value at a specific coordinate inside the Tensor
     * @param target The coordinate we want to fetch from the Tensor
     * @returns Returns a reference to the value that can be updated
     */
    T& at(std::initializer_list<size_t> target) override {
        // Handle the case where we have a rank-0 tensor
        if (c_rank == 0) {
            return _data()[c_offset];
        }
        // Validate we have the correct number of elements in our coordinates
        if (target.size() != c_rank) {
            throw std::invalid_argument(std::format("Tensor.at: Invalid number of coordinates provided to at, expected {} but got {}.\n", c_rank, target.size()));
        }
        // Calculate the offset and return a reference
        return _data()[_get_offset(target)];
    }

    /**
     * Get a value at a specific coordinate inside the Tensor
     * @param target The coordinate (wrapped in std::vector) we want to fetch from the tensor
     * @returns Returns the value at the coordinate
     */
    const T& at(const std::vector<size_t>& target) const override {
        // Handle the case where we have a rank-0 tensor
        if (c_rank == 0) {
            return _data()[c_offset];
        }
        // Validate we have the correct number of elements in our coordinates
        if (target.size() != c_rank) {
            throw std::invalid_argument(std::format("Tensor.at: Invalid number of coordinates provided to at, expected {} but got {}.\n", c_rank, target.size()));
        }
        // Calculate the offset and return a reference
        return _data()[_get_offset(target)];
    }

    /**
     * Get a value at a specific coordinate inside the Tensor
     * @param target The coordinate we want to fetch from the tensor
     * @returns Returns the value at the coordinate
     */
    const T& at(std::initializer_list<size_t> target) const override {
        // Handle the case where we have a rank-0 tensor
        if (c_rank == 0) {
            return _data()[c_offset];
        }
        // Validate we have the correct number of elements in our coordinates
        if (target.size() != c_rank) {
            throw std::invalid_argument(std::format("Tensor.at: Invalid number of coordinates provided to at, expected {} but got {}.\n", c_rank, target.size()));
        }
        // Calculate the offset and return a reference
        return _data()[_get_offset(target)];
    }

    /**
     * Get a mutable reference to the value stored at the provided index
     * @param target Index to fetch from
     * @returns Returns a mutable reference to the desired value
     */
    T& at(size_t target) override {
        // Ensure the index exists
        if (target >= c_elements) {
            throw std::invalid_argument(
                std::format(
                    "Tensor.at: Index {} exceeds number of elements {} in Tensor.\n",
                    target, c_elements
                )
            );
        }
        return _data()[_get_offset(target)];
    }

    /**
     * Get a const reference to the value stored at the provided index
     * @param target Index to fetch from
     * @returns Returns a const reference to the desired value
     */
    const T& at(size_t target) const override {
        // Ensure the index exists
        if (target >= c_elements) {
            throw std::invalid_argument(
                std::format(
                    "Tensor.at: Index {} exceeds number of elements {} in Tensor.\n",
                    target, c_elements
                )
            );
        }
        return _data()[_get_offset(target)];
    }

    /**
     * Addition assignment operator, adding values from target to this Tensor
     * @param target Tensor to add with
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& operator+=(const Tensor<T>& target) {
        return uniform_op(*this, target, *this, SqueezedOpType::ADD);
    }

    /**
     * Scalar addition assignment operator, adding a scalar value to every value in the Tensor
     * @param s Scalar to add to each value in the Tensor
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& operator+=(T s) {
        return uniform_op(*this, s, *this, SqueezedOpType::ADD);
    }

    /**
     * Addition operator, checking dimension compability then creating a new Tensor
     * containing the result
     * @param lhs Const ref to Tensor<T>
     * @param rhs Const ref to Tensor<T>
     * @returns Returns a new Tensor<T> containing the result of the addition
     */
    friend Tensor<T> operator+(const Tensor<T>& lhs, const Tensor<T>& rhs) {
        // Check to see if our Tensor shapes are compatible
        if (!lhs._compatible(rhs)) {
            throw std::invalid_argument("Tensor.+: Incompatible Tensor shapes provided to +.\n");
        }
        // Allocate a new Tensor with the same dimensions as lhs
        Tensor<T> result(lhs.dims());
        uniform_op(lhs, rhs, result, SqueezedOpType::ADD);
        return result;
    }

    /**
     * Scalar addition operator for adding scalar values, storing the result in a new Tensor
     * @param lhs Const ref to the Tensor<T>
     * @param rhs Scalar value we want to use
     * @returns Returns a new Tensor<T> containing the result of the addition
     */
    friend Tensor<T> operator+(const Tensor<T>& lhs, T rhs) {
        // Allocate a new Tensor with the same dimensions as lhs
        Tensor<T> result(lhs.dims());
        uniform_op(lhs, rhs, result, SqueezedOpType::ADD);
        return result;
    }

    /**
     * Subtraction assignment operator, subtracting values from target to this Tensor
     * @param target Tensor to subtract
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& operator-=(const Tensor<T>& target) {
        return uniform_op(*this, target, *this, SqueezedOpType::SUB);
    }

    /**
     * Scalar subtraction assignment operator, subtracting a scalar value from every value in the Tensor
     * @param s Scalar to subtract from each value in the Tensor
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& operator-=(T s) {
        return uniform_op(*this, s, *this, SqueezedOpType::SUB);
    }

    /**
     * Subtraction operator, checking dimension compability then creating a new Tensor
     * containing the result
     * @param lhs Const ref to Tensor<T>
     * @param rhs Const ref to Tensor<T>
     * @returns Returns a new Tensor<T> containing the result of the subtraction
     */
    friend Tensor<T> operator-(const Tensor<T>& lhs, const Tensor<T>& rhs) {
        // Check to see if our Tensor shapes are compatible
        if (!lhs._compatible(rhs)) {
            throw std::invalid_argument("Tensor.-: Incompatible Tensor shapes provided to -.\n");
        }
        // Allocate a new Tensor with the same dimensions as lhs
        Tensor<T> result(lhs.dims());
        uniform_op(lhs, rhs, result, SqueezedOpType::SUB);
        return result;
    }

    /**
     * Scalar subtraction operator for subtracting scalar values, storing the result in a new Tensor
     * @param lhs Const ref to the Tensor<T>
     * @param rhs Scalar value we want to use
     * @returns Returns a new Tensor<T> containing the result of the subtraction
     */
    friend Tensor<T> operator-(const Tensor<T>& lhs, T rhs) {
        // Allocate a new Tensor with the same dimensions as lhs
        Tensor<T> result(lhs.dims());
        uniform_op(lhs, rhs, result, SqueezedOpType::SUB);
        return result;
    }

        /**
     * Multiplication assignment operator, multiplying values from target to this Tensor
     * @param target Tensor to multiply
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& operator*=(const Tensor<T>& target) {
        return uniform_op(*this, target, *this, SqueezedOpType::MUL);
    }

    /**
     * Scalar multiplication assignment operator, multiplying a scalar value from every value in the Tensor
     * @param s Scalar to multiply with each value in the Tensor
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& operator*=(T s) {
        return uniform_op(*this, s, *this, SqueezedOpType::MUL);
    }

    /**
     * Multiplication operator, checking dimension compability then creating a new Tensor
     * containing the result
     * @param lhs Const ref to Tensor<T>
     * @param rhs Const ref to Tensor<T>
     * @returns Returns a new Tensor<T> containing the result of the multiplication
     */
    friend Tensor<T> operator*(const Tensor<T>& lhs, const Tensor<T>& rhs) {
        // Check to see if our Tensor shapes are compatible
        if (!lhs._compatible(rhs)) {
            throw std::invalid_argument("Tensor.*: Incompatible Tensor shapes provided to *.\n");
        }
        // Allocate a new Tensor with the same dimensions as lhs
        Tensor<T> result(lhs.dims());
        uniform_op(lhs, rhs, result, SqueezedOpType::MUL);
        return result;
    }

    /**
     * Scalar multiplication operator for multiplying scalar values, storing the result in a new Tensor
     * @param lhs Const ref to the Tensor<T>
     * @param rhs Scalar value we want to use
     * @returns Returns a new Tensor<T> containing the result of the multiplication
     */
    friend Tensor<T> operator*(const Tensor<T>& lhs, T rhs) {
        // Allocate a new Tensor with the same dimensions as lhs
        Tensor<T> result(lhs.dims());
        uniform_op(lhs, rhs, result, SqueezedOpType::MUL);
        return result;
    }

        /**
     * Division assignment operator, dividing values from target with this Tensor
     * @param target Tensor to divide
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& operator/=(const Tensor<T>& target) {
        return uniform_op(*this, target, *this, SqueezedOpType::DIV);
    }

    /**
     * Scalar division assignment operator, dividing a scalar value from every value in the Tensor
     * @param s Scalar to divide with each value in the Tensor
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& operator/=(T s) {
        return uniform_op(*this, s, *this, SqueezedOpType::DIV);
    }

    /**
     * Division operator, checking dimension compability then creating a new Tensor
     * containing the result
     * @param lhs Const ref to Tensor<T>
     * @param rhs Const ref to Tensor<T>
     * @returns Returns a new Tensor<T> containing the result of the division
     */
    friend Tensor<T> operator/(const Tensor<T>& lhs, const Tensor<T>& rhs) {
        // Check to see if our Tensor shapes are compatible
        if (!lhs._compatible(rhs)) {
            throw std::invalid_argument("Tensor./: Incompatible Tensor shapes provided to /.\n");
        }
        // Allocate a new Tensor with the same dimensions as lhs
        Tensor<T> result(lhs.dims());
        uniform_op(lhs, rhs, result, SqueezedOpType::DIV);
        return result;
    }

    /**
     * Scalar division operator for dividing scalar values, storing the result in a new Tensor
     * @param lhs Const ref to the Tensor<T>
     * @param rhs Scalar value we want to use
     * @returns Returns a new Tensor<T> containing the result of the division
     */
    friend Tensor<T> operator/(const Tensor<T>& lhs, T rhs) {
        // Allocate a new Tensor with the same dimensions as lhs
        Tensor<T> result(lhs.dims());
        uniform_op(lhs, rhs, result, SqueezedOpType::DIV);
        return result;
    }

    /**
     * Fill a Tensor with a value
     * @param v Value to fill the Tensor with
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& fill(const T& v) {
        // Set all elements of the Tensor to v
        T* data = _data();
        for (size_t i = 0; i < c_elements; ++i) {
           data[_get_offset(i)] = v;
        }
        return *this;
    }

    /**
     * Set the contents of a Tensor to a list of values
     * @param v Values to set inside the Tensor
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& set(std::initializer_list<T> v) {
        // Ensure v has enough values to satisfy the number of elements in the Tensor
        if (v.size() != c_elements) {
            throw std::invalid_argument(std::format("Tensor.set: Invalid number of value provided to set. Wanted {} but got {}\n.", c_elements, v.size()));
        }
        T* data = _data();
        // Assuming we have enough values, read them into memory sequentially
        for (size_t i = 0; i < v.size(); ++i) {
            data[_get_offset(i)] = v.begin()[i];
        }
        return *this;
    }

    /**
     * Apply a function to all elements in a Tensor
     * @param f Function to apply
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& apply(T (*f)(T)) {
        // Apply the function pointed to by f to all elements
        try {
            T* data = _data();
            for (size_t i = 0; i < c_elements; ++i) {
                size_t offset = _get_offset(i);
                data[offset] = (*f)(data[offset]);
            }
        } catch (const std::exception& e) {
            throw std::invalid_argument(std::format("Tensor.apply: Exception when applying function to Tensor. Error: {}", e.what()));
        }
        return *this;
    }

    /**
     * Apply a function to all elements in a Tensor
     * @param f Function to apply
     * @param p Parameter to supply to function f
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& apply(T (*f)(T, T), T param) {
        // Apply the function pointed to by f to all elements
        try {
            T* data = _data();
            for (size_t i = 0; i < c_elements; ++i) {
                size_t offset = _get_offset(i);
                data[offset] = (*f)(data[offset], param);
            }
        } catch (const std::exception& e) {
            throw std::invalid_argument(std::format("Tensor.apply: Exception when applying function to Tensor. Error: {}", e.what()));
        }
        return *this;
    }

    /**
     * Fill a Tensor with random values, taken from a specified range
     * @param range_min Inclusive minimum
     * @param range_max Inclusive maximum
     * @returns Returns a pointer to this Tensor
     */
    Tensor<T>& random(T range_min, T range_max) {
        // Prepare to generate random values
        std::random_device rd;
        std::mt19937 gen(rd());

        // std::uniform_real_distribution only applies to float-like types, but we can use the trait types
        // to check for int-like types and use std::uniform_int_distribution instead
        if constexpr (std::integral<T>) {
            std::uniform_int_distribution<T> distrib(range_min, range_max);
            T* data = _data();
            for (size_t i = 0; i < c_elements; ++i) {
                data[_get_offset(i)] = distrib(gen);
            }
        }
        else if constexpr (std::floating_point<T>) {
            std::uniform_real_distribution<T> distrib(range_min, range_max);
            T* data = _data();
            for (size_t i = 0; i < c_elements; ++i) {
                data[_get_offset(i)] = distrib(gen);
            }
        }
        else {
            throw std::domain_error("Tensor.random: Invalid type for use with random");
        }
        return *this;
    }

    /**
     * Calculate the sum of a Tensor
     * @returns Returns the sum
     */
    T sum() const {
        // Store the result
        T result = 0;
        // Route to the proper overflow / underflow function based on type
        if constexpr (std::is_floating_point_v<T>) {
            const T* data = _data();
            // Traverse the Tensor from element 0..N
            for (size_t i = 0; i < c_elements; ++i) {
                result += data[_get_offset(i)];
            }
            if (!std::isfinite(result)) {
                throw std::overflow_error("Tensor.sum: Overflow / underflow detected.\n");
            }
            return result;
        }
        else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
            const T* data = _data();
            bool overflow = false;
            for (size_t i = 0; i < c_elements; ++i) {
                overflow |= TensorMath_NS::_add_overflow_signed(result, data[_get_offset(i)], &result);
            }
            if (overflow) {
                throw std::overflow_error("Tensor.sum: Overflow / underflow detected.\n");
            }
            return result;
        }
        else {
            const T* data = _data();
            bool overflow = false;
            for (size_t i = 0; i < c_elements; ++i) {
                overflow |= TensorMath_NS::_add_overflow_unsigned(result, data[_get_offset(i)], &result);
            }
            if (overflow) {
                throw std::overflow_error("Tensor.sum: Overflow / underflow detected.\n");
            }
            return result;
        }
    }

    /**
     * Update the contiguity of a Tensor, necessary after updates
     * such as transpose()
     */
    void _update_contiguous() {
        // If we have a rank 0 or rank 1 Tensor, they are contiguous
        // by definition. ListTensorSlice handles the corner case where
        // that may not be true
        if (c_rank == 0) {
            c_contiguous = true;
            return;
        }
        size_t target_stride = 1;
        // Iterate over the strides in reverse order, where we expect
        // the final dim to have stride == 1
        for (size_t i = c_rank; i-- > 0; ) {
            if (m_stride[i] != target_stride) {
                c_contiguous = false;
                return;
            }
            target_stride *= m_dims[i];
        }
        c_contiguous = true;
    }

    /**
     * Transpose a Tensor along two specified dims
     * @param dim0 First dim to swap
     * @param dim1 Second dim to swap
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& transpose(size_t dim0, size_t dim1) override {
        // Ensure we have valid dims
        if (dim0 >= c_rank || dim1 >= c_rank) {
            throw std::invalid_argument("Tensor.transpose: Invalid axes provided to transpose.\n");
        }
        // Ensure we aren't trying to swap the same dim
        if (dim0 == dim1) {
            throw std::invalid_argument("Tensor.transpose: Cannot transpose the same axis.\n");
        }
        // Perform the swap and update the strides
        std::swap(m_stride.at(dim0), m_stride.at(dim1));
        std::swap(m_dims.at(dim0), m_dims.at(dim1));
        _update_contiguous();
        return *this;
    }

    /**
     * Transpose a Tensor along two specified dims
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& transpose() {
        // Ensure this can only run on a 2-D Tensor
        if (c_rank != 2) {
            throw std::logic_error("Tensor.transpose: transpose() cannot be run on a non rank-2 Tensor.\n");
        }
        // Perform the swap and update the strides
        std::swap(m_stride.at(0), m_stride.at(1));
        std::swap(m_dims.at(0), m_dims.at(1));
        _update_contiguous();
        return *this;
    }

    /**
     * Apply dropout to a Tensor
     * @param dropout Float from 0 - 1.0 representing the percentage of items in the Tensor
     * should be dropped (set to 0)
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& apply_dropout(float dropout) {
        // Ensure dropout is in the range of > 0, < 1.0
        if (dropout < 0 || dropout >= 1.0f) {
            throw std::invalid_argument("Tensor.apply_dropout: Dropout factor cannot be negative or >= 1.\n");
        }
        // If dropout == 0, return without making any changes
        if (dropout == 0) {
            return *this;
        }
        // Calculate the number of elements that should be zeroed out
        size_t target_elements = static_cast<size_t>(static_cast<float>(c_elements) * dropout);
        // Calculate the scale factor for the remaining elements
        T scale_factor = static_cast<T>(1.0f / (1.0f - dropout));
        // Prepare to generate random values
        std::random_device rd;
        std::mt19937 gen(rd());
        // Set the range to be from 0 to c_elements - 1
        std::uniform_int_distribution<size_t> distrib(0, c_elements - 1);
        // Use an ordered set to force unique values to be stored
        std::unordered_set<size_t> target_idxs;
        target_idxs.reserve(target_elements);
        // Generate enough unique indices to satisfy target_elements
        while (target_idxs.size() < target_elements) {
            target_idxs.insert(distrib(gen));
        }
        // Traverse the indices in the set and zero them out
        T* data = _data();
        for (const auto& idx : target_idxs) {
            data[_get_offset(idx)] = 0;
        }
        // Apply the scaling factor to the rest of the Tensor
        *this *= scale_factor;
        return *this;
    }

    /**
     * Find the maximum value in a Tensor
     * @returns Returns the maximum value
     */
    T max() const {
        if (c_contiguous) {
            // NOLINTNEXTLINE(bugprone-sizeof-expression)
            return *(std::max_element(_data() + c_offset, _data() + c_offset + c_elements));
        }
        T result = std::numeric_limits<T>::lowest();
        const T* self_data = _data();
        for (size_t i = 0; i < c_elements; ++i) {
            T val = self_data[_get_offset(i)];
            result = val > result ? val : result;
        }
        return result;
    }

    /**
     * Find the minimum value in a Tensor
     * @returns Returns the minimum value
     */
    T min() const {
        if (c_contiguous) {
            // NOLINTNEXTLINE(bugprone-sizeof-expression)
            return *(std::min_element(_data() + c_offset, _data() + c_offset + c_elements));
        }
        T result = std::numeric_limits<T>::max();
        const T* self_data = _data();
        for (size_t i = 0; i < c_elements; ++i) {
            T val = self_data[_get_offset(i)];
            result = val < result ? val : result;
        }
        return result;
    }

    /**
     * Returns the number of rows in a 2-D Tensor
     * @returns Returns the number of rows
     */
    size_t rows() const {
        // Ensure this can only run on 2-D Tensors
        if (c_rank != 2) {
            throw std::logic_error("Tensor.rows: rows() cannot be called on a non rank-2 Tensor.\n");
        }
        return m_dims.at(0);
    }

    /**
     * Returns the number of columns in a 2-D Tensor
     * @returns Returns the number of columns
     */
    size_t cols() const {
        // Ensure this can only run on 2-D Tensors
        if (c_rank != 2) {
            throw std::logic_error("Tensor.rows: cols() cannot be called on a non rank-2 Tensor.\n");
        }
        return m_dims.at(1);
    }

    /**
     * Convert a Matrix to a string representation, similar to the << operator above
     * @returns Returns a string representation of the Matrix
     */
    std::string to_string() const {
        // Ensure this can only run on 2-D Tensors
        if (c_rank != 2) {
            throw std::logic_error("Tensor.to_string: to_string() cannot be called on a non rank-2 Tensor.\n");
        }
        // Create a blank string that we will return
        std::string result = "\n";
        // Wrap the Tensor in brackets, with each row properly enclosed too
        result += "[";
        for (size_t i = 0; i < rows(); ++i) {
            result += "[";
            for (size_t j = 0; j < cols(); ++j) {
                // Add the value in the Matrix
                result += std::format("{}", at({i, j}));
                if (j + 1 < cols()) {
                    // Separate the values by tabs, for readability
                    result += "\t";
                }
            }
            // Close out each row with a corresponding ]
            result += "]";
            // Add a new line after each row if we aren't at the end
            if (i + 1 < rows()) {
                result += "\n";
            }
        }
        // Close out the Tensor final bracket and print out the info (dims and dtype)
        result += std::format("]. {}.", info());
        return result;
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    /**
     * Get the sum of a 2-D Tensor's rows or columns
     * @param dim Dimension to sum across
     * @param idx Index on the specified dimension to sum
     * @returns Returns the sum
     */
    T sum(size_t dim, size_t idx) const {
        // Ensure this can only run on 2-D Tensors
        if (c_rank != 2) {
            throw std::logic_error("Tensor.sum: sum(dim, idx) cannot be called on a non rank-2 Tensor.\n");
        }
        if (dim > 1) {
            throw std::invalid_argument("Tensor.sum: Invalid dim specified for sum.\n");
        }
        T result = 0;
        // Sum across a row
        if (dim == 0) {
            if (idx >= rows()) {
                throw std::invalid_argument("Tensor.sum: Invalid index provided to sum.\n");
            }
            if constexpr (std::is_floating_point_v<T>) {
                for (size_t i = 0; i < cols(); ++i) {
                    result += at({idx, i});
                }
                if (!std::isfinite(result)) {
                    throw std::overflow_error("Tensor.sum: Overflow / underflow detected in sum(dim, idx).\n");
                }
            }
            else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
                bool overflow = false;
                for (size_t i = 0; i < cols(); ++i) {
                    overflow |= TensorMath_NS::_add_overflow_signed(result, at({idx, i}), &result);
                }
                if (overflow) {
                    throw std::overflow_error("Tensor.sum: Overflow / underflow detected in sum(dim, idx).\n");
                }
            }
            else {
                bool overflow = false;
                for (size_t i = 0; i < cols(); ++i) {
                    overflow |= TensorMath_NS::_add_overflow_unsigned(result, at({idx, i}), &result);
                }
                if (overflow) {
                    throw std::overflow_error("Tensor.sum: Overflow / underflow detected in sum(dim, idx).\n");
                }
            }
        }
        else {
            if (idx >= cols()) {
                throw std::invalid_argument("Tensor.sum: Invalid index provided to sum.\n");
            }
            if constexpr (std::is_floating_point_v<T>) {
                for (size_t i = 0; i < rows(); ++i) {
                    result += at({i, idx});
                }
                if (!std::isfinite(result)) {
                    throw std::overflow_error("Tensor.sum: Overflow / underflow detected in sum(dim, idx).\n");
                }
            }
            else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
                bool overflow = false;
                for (size_t i = 0; i < rows(); ++i) {
                    overflow |= TensorMath_NS::_add_overflow_signed(result, at({i, idx}), &result);
                }
                if (overflow) {
                    throw std::overflow_error("Tensor.sum: Overflow / underflow detected in sum(dim, idx).\n");
                }
            }
            else {
                bool overflow = false;
                for (size_t i = 0; i < rows(); ++i) {
                    overflow |= TensorMath_NS::_add_overflow_unsigned(result, at({i, idx}), &result);
                }
                if (overflow) {
                    throw std::overflow_error("Tensor.sum: Overflow / underflow detected in sum(dim, idx).\n");
                }
            }
        }
        return result;
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    /**
     * Get the maximum value found along a dim
     * @param dim The dimension to search for the max across
     * @param squeeze True to reduce to a 1-D Tensor with one entry per index on the specified dim,
     * e.g. if finding a max across a [3, 3] Tensor, the squeezed Tensor will have dim [3, 1]
     * @returns Returns a Tensor or with the max values per dim
     */
    Tensor<T> max(size_t dim, bool squeeze) const {
        // Ensure this can only run on 2-D Tensors
        if (c_rank != 2) {
            throw std::logic_error("Tensor.max: sum(dim, squeeze) cannot be called on a non rank-2 Tensor.\n");
        }
        Tensor<T> result = (squeeze ? Tensor<T>({m_dims.at(dim)}) : Tensor<T>({rows(), cols()}));
        // Store the max value
        T dim_max = std::numeric_limits<T>::lowest();
        // Iterate across the rows if dim == 0
        if (dim == 0) {
            for (size_t i = 0; i < rows(); ++i) {
                for (size_t j = 0; j < cols(); ++j) {
                    const auto& test_val = at({i, j});
                    dim_max = dim_max < test_val ? test_val : dim_max;
                }
                if (squeeze) {
                    result.at({i}) = dim_max;
                }
                else {
                    for (size_t j = 0; j < cols(); ++j) {
                        result.at({i, j}) = dim_max;
                    }
                }
                dim_max = std::numeric_limits<T>::lowest();
            }
        }
        else {
            for (size_t i = 0; i < cols(); ++i) {
                for (size_t j = 0; j < rows(); ++j) {
                    const auto& test_val = at({j, i});
                    dim_max = dim_max < test_val ? test_val : dim_max;
                }
                if (squeeze) {
                    result.at({i}) = dim_max;
                }
                else {
                    for (size_t j = 0; j < rows(); ++j) {
                        result.at({j, i}) = dim_max;
                    }
                }
                dim_max = std::numeric_limits<T>::lowest();
            }
        }
        return result;
    }

    /**
     * Apply an operation across each index of a dim in a Tensor
     * @param dim Dimension to apply the operation across
     * @param vals Const ref to a Tensor containing one value per index of the dim
     * @param op_type Operation to apply, either ADD, SUB, MUL, DIV
     */
    Tensor<T>& squeezed_op(size_t dim, const Tensor<T>& vals, SqueezedOpType op_type) {
        // Ensure this can only run on 2-D Tensors
        if (c_rank != 2) {
            throw std::logic_error("Tensor.squeezed_op: squeezed_op(dim, vals, op_type) cannot be called on a non rank-2 Tensor.\n");
        }
        // Check to see that we have a 1-D Tensor
        if (vals.rank() != 1) {
            throw std::invalid_argument("Tensor.squeezed_op: Vals Tensor cannot have rank > 1\n");
        }
        // Ensure that the dim is either 0 or 1
        if (dim > 1) {
            throw std::invalid_argument("Tensor.squeezed_op: Invalid dim specified for squeezed_op.\n");
        }
        // Ensure we have the same number of vals as indices on the specified dim
        if (vals.elements() != m_dims.at(dim)) {
            throw std::invalid_argument("Tensor.squeezed_op: Vals Tensor does not have the correct number of elements.\n");
        }
        size_t target_dim_size = m_dims.at(dim);
        size_t other_dim_size = m_dims.at(dim == 0 ? 1: 0);
        switch (op_type) {
            case SqueezedOpType::ADD:
                for (size_t i = 0; i < target_dim_size; ++i) {
                    const T& squeezed_val = vals.at({i});
                    for (size_t j = 0; j < other_dim_size; ++j) {
                        T& val = (dim == 0 ? at({i, j}) : at({j, i}));
                        if constexpr (std::is_floating_point_v<T>) {
                            val += squeezed_val;
                            if (!std::isfinite(val)) {
                                throw std::overflow_error("Tensor.squeezed_op: Addition causes overflow / underflow.\n");
                            }
                        }
                        else if constexpr (std::is_signed_v<T>) {
                            if (TensorMath_NS::_add_overflow_signed(val, squeezed_val, &val)) {
                                throw std::overflow_error("Tensor.squeezed_op: Addition causes overflow / underflow.\n");
                            }
                        }
                        else {
                            if (TensorMath_NS::_add_overflow_unsigned(val, squeezed_val, &val)) {
                                throw std::overflow_error("Tensor.squeezed_op: Addition causes overflow / underflow.\n");
                            }
                        }
                    }
                }
                break;
            case SqueezedOpType::SUB:
                for (size_t i = 0; i < target_dim_size; ++i) {
                    const T& squeezed_val = vals.at({i});
                    for (size_t j = 0; j < other_dim_size; ++j) {
                        T& val = (dim == 0 ? at({i, j}) : at({j, i}));
                        if constexpr (std::is_floating_point_v<T>) {
                            val -= squeezed_val;
                            if (!std::isfinite(val)) {
                                throw std::overflow_error("Tensor.squeezed_op: Subtraction causes overflow / underflow.\n");
                            }
                        }
                        else if constexpr (std::is_signed_v<T>) {
                            if (TensorMath_NS::_sub_overflow_signed(val, squeezed_val, &val)) {
                                throw std::overflow_error("Tensor.squeezed_op: Subtraction causes overflow / underflow.\n");
                            }
                        }
                        else {
                            if (TensorMath_NS::_sub_overflow_unsigned(val, squeezed_val, &val)) {
                                throw std::overflow_error("Tensor.squeezed_op: Subtraction causes overflow / underflow.\n");
                            }
                        }
                    }
                }
                break;
            case SqueezedOpType::MUL:
                for (size_t i = 0; i < target_dim_size; ++i) {
                    const T& squeezed_val = vals.at({i});
                    for (size_t j = 0; j < other_dim_size; ++j) {
                        T& val = (dim == 0 ? at({i, j}) : at({j, i}));
                        if constexpr (std::is_floating_point_v<T>) {
                            val *= squeezed_val;
                            if (!std::isfinite(val)) {
                                throw std::overflow_error("Tensor.squeezed_op: Multiplication causes overflow / underflow.\n");
                            }
                        }
                        else if constexpr (std::is_signed_v<T>) {
                            if (TensorMath_NS::_mul_overflow_signed(val, squeezed_val, &val)) {
                                throw std::overflow_error("Tensor.squeezed_op: Multiplication causes overflow / underflow.\n");
                            }
                        }
                        else {
                            if (TensorMath_NS::_mul_overflow_unsigned(val, squeezed_val, &val)) {
                                throw std::overflow_error("Tensor.squeezed_op: Multiplication causes overflow / underflow.\n");
                            }
                        }
                    }
                }
                break;
            case SqueezedOpType::DIV:
                for (size_t i = 0; i < target_dim_size; ++i) {
                    const T& squeezed_val = vals.at({i});
                    for (size_t j = 0; j < other_dim_size; ++j) {
                        T& val = (dim == 0 ? at({i, j}) : at({j, i}));
                        if constexpr (std::is_floating_point_v<T>) {
                            val /= squeezed_val;
                            if (!std::isfinite(val)) {
                                throw std::overflow_error("Tensor.squeezed_op: Division causes overflow / underflow.\n");
                            }
                        }
                        else if constexpr (std::is_signed_v<T>) {
                            if (squeezed_val == 0) {
                                throw std::logic_error("Tensor.squeezed_op: Divide by zero detected.\n");
                            }
                            if (TensorMath_NS::_is_signed_div_overflow(val, squeezed_val)) {
                                throw std::overflow_error("Tensor.squeezed_op: Division causes overflow / underflow.\n");
                            }
                            val /= squeezed_val;
                        }
                        else {
                            if (squeezed_val == 0) {
                                throw std::logic_error("Tensor.squeezed_op: Divide by zero detected.\n");
                            }
                            val /= squeezed_val;
                        }
                    }
                }
                break;
        }
        return *this;
    }

    /**
     * Apply an operation across each index of a dim in a Tensor
     * @param dim Dimension to apply the operation across
     * @param vals Const ref to a Tensor containing one value per index of the dim
     * @param op_type Operation to apply, either ADD, SUB, MUL, DIV
     */
    Tensor<T>& unsafe_squeezed_op(size_t dim, const Tensor<T>& vals, SqueezedOpType op_type) {
        // This variant of squeezed op only applies to floating point dtypes
        if (!std::is_floating_point_v<T>) {
            throw std::runtime_error("Tensor.unsafe_squeezed_op: Tensor must be of dtype=float (or variant).\n");
        }
        // Ensure this can only run on 2-D Tensors
        if (c_rank != 2) {
            throw std::logic_error("Tensor.unsafe_squeezed_op: unsafe_squeezed_op(dim, vals, op_type) cannot be called on a non rank-2 Tensor.\n");
        }
        // Check to see that we have a 1-D Tensor
        if (vals.rank() != 1) {
            throw std::invalid_argument("Tensor.unsafe_squeezed_op: Vals Tensor cannot have rank > 1\n");
        }
        // Ensure that the dim is either 0 or 1
        if (dim > 1) {
            throw std::invalid_argument("Tensor.unsafe_squeezed_op: Invalid dim specified for unsafe_squeezed_op.\n");
        }
        // Ensure we have the same number of vals as indices on the specified dim
        if (vals.elements() != m_dims.at(dim)) {
            throw std::invalid_argument("Tensor.unsafe_squeezed_op: Vals Tensor does not have the correct number of elements.\n");
        }
        size_t target_dim_size = m_dims.at(dim);
        size_t other_dim_size = m_dims.at(dim == 0 ? 1: 0);
        switch (op_type) {
            case SqueezedOpType::ADD:
                for (size_t i = 0; i < target_dim_size; ++i) {
                    const T& squeezed_val = vals.at({i});
                    for (size_t j = 0; j < other_dim_size; ++j) {
                        T& val = (dim == 0 ? at({i, j}) : at({j, i}));
                        val += squeezed_val;
                    }
                }
                break;
            case SqueezedOpType::SUB:
                for (size_t i = 0; i < target_dim_size; ++i) {
                    const T& squeezed_val = vals.at({i});
                    for (size_t j = 0; j < other_dim_size; ++j) {
                        T& val = (dim == 0 ? at({i, j}) : at({j, i}));
                        val -= squeezed_val;
                    }
                }
                break;
            case SqueezedOpType::MUL:
                for (size_t i = 0; i < target_dim_size; ++i) {
                    const T& squeezed_val = vals.at({i});
                    for (size_t j = 0; j < other_dim_size; ++j) {
                        T& val = (dim == 0 ? at({i, j}) : at({j, i}));
                        val *= squeezed_val;
                    }
                }
                break;
            case SqueezedOpType::DIV:
                for (size_t i = 0; i < target_dim_size; ++i) {
                    const T& squeezed_val = vals.at({i});
                    for (size_t j = 0; j < other_dim_size; ++j) {
                        T& val = (dim == 0 ? at({i, j}) : at({j, i}));
                        val /= squeezed_val;
                    }
                }
                break;
        }
        return *this;
    }
    
    /**
     * Apply softmax to a Tensor across a specified dim
     * @param dim Dimension to apply softmax across
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& softmax(size_t dim) {
        // Ensure this can only run on 2-D Tensors
        if (c_rank != 2) {
            throw std::logic_error("Tensor.softmax: softmax(dim) cannot be called on a non rank-2 Tensor.\n");
        }
        // Ensure we are using a float point type
        if constexpr (!std::is_floating_point_v<T>) {
            throw std::invalid_argument("Tensor.softmax: Cannot apply softmax to non-floating point Tensor.\n");
        }
        if (dim > 1) {
            throw std::invalid_argument("Tensor.softmax: Invalid dim specified for softmax.\n");
        }
        // Get the max value per specified dim and subtract it from the input to
        // avoid overflow / underflow when using std::exp. Use the squeezed Tensor
        // to avoid unnecessarily using more memory
        Tensor<T> max_vals = max(dim, true);
        // Use the squeezed_op function to subtract the max values from each index on the dimension
        unsafe_squeezed_op(dim, max_vals, SqueezedOpType::SUB);
        // Apply the exp function across all elements in the Matrix
        apply(std::exp);
        // Have a fallback for row / col sums being 0, inf, or NAN
        T fallback = static_cast<T>(1.0f / this->elements());
        // Sum the values across the specified dim
        if (dim == 0) {
            for (size_t i = 0; i < rows(); ++i) {
                T row_sum = sum(dim, i);
                if (row_sum == 0 || !std::isfinite(row_sum)) {
                    if constexpr (TENSOR_ENABLE_SOFTMAX_WARNINGS) {
                        log_message(Log_Priority::WARNING, "Tensor.softmax", std::format("row_sum is either 0, inf, or NAN, replacing with fallback ({}).", fallback));
                    }
                    // Zero out the values if we will divide by zero, infinity, or NAN
                    for (size_t j = 0; j < cols(); ++j) {
                        at({i, j}) = fallback;
                    }
                }
                else {
                    for (size_t j = 0; j < cols(); ++j) {
                        at({i, j}) /= row_sum;
                    }
                }
            }
        }
        else {
            for (size_t i = 0; i < cols(); ++i) {
                T col_sum = sum(dim, i);
                if (col_sum == 0 || !std::isfinite(col_sum)) {
                    if constexpr (TENSOR_ENABLE_SOFTMAX_WARNINGS) {
                        log_message(Log_Priority::WARNING, "Tensor.softmax", std::format("col_sum is either 0, inf, or NAN, replacing with fallback ({}).", fallback));
                    }
                    for (size_t j = 0; j < rows(); ++j) {
                        at({j, i}) = fallback;
                    }
                }
                else {
                    for (size_t j = 0; j < rows(); ++j) {
                        at({j, i}) /= col_sum;
                    }
                }
            }
        }
        return *this;
    }

    /**
     * Fills the Tensor with zeroes and creates a triangular Tensor
     * @param mask_type UPPER or LOWER
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& tri(const CausalMaskType mask_type) {
        // Ensure this can only run on 2-D Tensors
        if (c_rank != 2) {
            throw std::logic_error("Tensor.tri: tri(mask_type) cannot be called on a non rank-2 Tensor.\n");
        }
        // Ensure we are applying this to a square Matrix
        if (rows() != cols()) {
            throw std::invalid_argument("Tensor.tri: Cannot create triangular Tensor on a Tensor that is not square.\n");
        }
        // Iterate across the rows and columns, first setting all values to zero,
        // then putting in ones for anywhere that col >= row
        fill(0);
        if (mask_type == CausalMaskType::UPPER) {
            for (size_t i = 0; i < rows(); ++i) {
                for (size_t j = i + 1; j < cols(); ++j) {
                    at({i, j}) = static_cast<T>(1);
                }
            }
        }
        else {
            for (size_t i = 1; i < rows(); ++i) {
                for (size_t j = 0; j < i; ++j) {
                    at({i, j}) = static_cast<T>(1);
                }
            }
        }
        return *this;
    }

    /**
     * Apply masking to a Tensor along the diagonal, zeroing out values. This operation is
     * the same as creating a mask Tensor with tri or ninf_tri and using * or *=
     * We use the apply_mask op to save the allocation associated with creating a new Tensor
     * @param mask_type UPPER or LOWER (if UPPER, everything above diagonal is 0 and vice-verse)
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& apply_mask(const CausalMaskType mask_type) {
        // Ensure this can only run on 2-D Tensors
        if (c_rank != 2) {
            throw std::logic_error("Tensor.apply_mask: apply_mask(mask_type) cannot be called on a non rank-2 Tensor.\n");
        }
        // Ensure we are applying this to a square Matrix
        if (rows() != cols()) {
            throw std::invalid_argument("Tensor.apply_mask: Cannot apply mask on a Tensor that is not square.\n");
        }
        // Iterate over rows and columns, zeroing out values below the diagonal
        // This is the inverse op of tri and ninf_ti
        if (mask_type == CausalMaskType::UPPER) {
            for (size_t i = 0; i < rows(); ++i) {
                for (size_t j = i + 1; j < cols(); ++j) {
                    at({i, j}) = 0;
                }
            }
        }
        else {
            for (size_t i = 1; i < rows(); ++i) {
                for (size_t j = 0; j < i; ++j) {
                    at({i, j}) = 0;
                }
            }
        }
        return *this;
    }

    /**
     * Replace the values above or below the diagonal with negative infinity.
     * NOTE: When using with softmax, the presence of -inf seems to cause row sums to 
     * equal zero, resulting in a divide by zero, and using the fallback. Use the tri
     * method until this is fixed.
     * @param mask_type UPPER or LOWER
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& ninf_tri(const CausalMaskType mask_type) {
        // Ensure this can only run on 2-D Tensors
        if (c_rank != 2) {
            throw std::logic_error("Tensor.ninf_tri: ninf_tri(mask_type) cannot be called on a non rank-2 Tensor.\n");
        }
        // Ensure we are using a floating point type
        if constexpr (!std::is_floating_point_v<T>) {
            throw std::invalid_argument("Tensor.ninf_tri: Cannot use ninf_tri with a non-floating point Tensor.\n");
        }
        // Ensure we are applying this to a square Matrix
        if (rows() != cols()) {
            throw std::invalid_argument("Tensor.ninf_tri: Cannot apply triangular mask on a Tensor that is not square.\n");
        }
        // Get the negative infinity value for our type
        T ninf = -std::numeric_limits<T>::infinity();
        // Apply the mask
        if (mask_type == CausalMaskType::UPPER) {
            for (size_t i = 0; i < rows(); ++i) {
                for (size_t j = i + 1; j < cols(); ++j) {
                    at({i, j}) = ninf;
                }
            }
        }
        else {
            for (size_t i = 1; i < rows(); ++i) {
                for (size_t j = 0; j < i; ++j) {
                    at({i, j}) = ninf;
                }
            }
        }
        return *this;
    }

    /**
     * Create a slice of a Tensor, creating a lightweight wrapper around the shared Storage
     * class. The resulting slice can perform all normal Tensor ops, though by virtue
     * of selecting a subset of the dims, the slice will not be contiguous.
     * Converting to a contiguous Tensor is possible via the remat() function
     * @param dims List of dims to include in the Tensor slice
     * @param filters Optional std::initializer_list<std::pair<size_t, size_t>> containing start and stops for each dim
     * For a mixed setup where some dims should be filtered and others shouldn't, provide
     * {0, 0} to include the entire extent of the dim
     * @param other_dims Required when creating a slice of a higher rank Tensor. This should
     * include base coordinates for the dims that aren't included in the slice
     */
    Tensor<T> slice(std::initializer_list<size_t> dims,
                    std::initializer_list<std::pair<size_t, size_t>> filters,
                    std::initializer_list<std::pair<size_t, size_t>> other_dims) const {

        // Ensure that we have received at least 1 dim, but not more than this Tensor's rank
        if (dims.size() == 0 || dims.size() > c_rank) {
            throw std::invalid_argument(
                std::format("Tensor.slice: Invalid number of dims provided {}. Tensor's rank: {}.", dims.size(), c_rank));
        }
        // Ensure the specified dims are valid
        for (size_t dim : dims) {
            if (dim >= c_rank) {
                throw std::invalid_argument("Tensor.slice: Invalid dims provided.\n");
            }
        }
        // Ensure dims doesn't have any duplicates
        size_t prev_dim = dims.begin()[0];
        for (size_t i = 1; i < dims.size(); ++i) {
            if (dims.begin()[i] == prev_dim) {
                throw std::invalid_argument("Tensor.slice: Duplicate dim detected.\n");
            }
            prev_dim = dims.begin()[i];
        }
        // Ensure that c_rank - dims.size() = other_dims.size()
        if ((c_rank - dims.size()) != other_dims.size()) {
            throw std::invalid_argument(
                std::format(
                    "Tensor.slice: Invalid other_dims provided. Expected {} but got {}.", c_rank - dims.size(), other_dims.size()
                )
            );
        }
        // Ensure other_dims are valid
        for (auto [dim, idx] : other_dims) {
            if (dim >= c_rank) {
                throw std::invalid_argument("Tensor.slice: Invalid other_dim specified.\n");
            }
        }
        // Create a shallow copy of this Tensor
        Tensor<T> result = *this;
        result.c_rank = dims.size();
        // Resize the m_dims and m_stride vectors to account for the new rank
        result.m_dims.resize(result.c_rank);
        result.m_stride.resize(result.c_rank);
        // Handle the most simple case: no filters applied
        if (filters.size() == 0) {
            // Reset the number of elements back to 1 and recalculate inside the loop
            result.c_elements = 1;
            // Since we have no filters, grab the extents and strides from this Tensor
            for (size_t i = 0; i < result.c_rank; ++i) {
                result.m_dims[i] = extent(dims.begin()[i]);
                result.c_elements *= extent(dims.begin()[i]);
                result.m_stride[i] = dim_stride(dims.begin()[i]);
            }
            // Since we have no filters and a different target rank, we must have other
            // dims to consider. Create a vector and prepare a query to get the new
            // offset for the slice
            std::vector<size_t> offset_query(c_rank, 0);
            for (size_t i = 0; i < other_dims.size(); ++i) {
                auto [dim_num, dim_c] = other_dims.begin()[i];
                if (dim_num >= c_rank) {
                    throw std::invalid_argument(
                        std::format("Tensor.slice: Invalid other dim id provided. Got {} but Tensor has rank {}.", dim_num, c_rank)
                    );
                }
                // Ensure we don't have any duplicates in other_dims
                size_t& offset_query_val = offset_query[dim_num];
                if (offset_query_val != 0) {
                    throw std::invalid_argument(
                        std::format("Tensor.slice: Duplicate other_dim detected: {}.", dim_num)
                    );
                }
                offset_query_val = dim_c;
            }
            // Calculate the offset from this Tensor and persist into the target
            result.c_offset = _get_offset(offset_query);
        }
        else {
            // If we have filters, there must be one per dim specified in dims
            if (dims.size() != filters.size()) {
                throw std::invalid_argument(
                    std::format("Tensor.slice: Invalid filters provided. Expected {} but got {}.", dims.size(), filters.size())
                );
            }
            // Reset the number of elements back to 1 and recalculate inside the loop
            result.c_elements = 1;
            // Create a vector to calculate the offset with filters and other dims
            std::vector<size_t> offset_query(c_rank, 0);
            // Iterate over the filters and construct the extents of each dim
            for (size_t i = 0; i < result.c_rank; ++i) {
                // Grab the stride for the dim
                result.m_stride[i] = dim_stride(dims.begin()[i]);
                // Grab start and end from the std::pair inside filters
                auto [start, end] = filters.begin()[i];
                // If both are zero, pull in the full dim
                if (start == 0 && end == 0) {
                    result.m_dims[i] = extent(dims.begin()[i]);
                }
                else {
                    // Ensure start is less than end and both are valid
                    if (start >= end || start >= extent(dims.begin()[i]) || end > extent(dims.begin()[i])) {
                        throw std::invalid_argument("Tensor.slice: Invalid start and end provided for dim filter.\n");
                    }
                    result.m_dims[i] = end - start;
                }
                // Increment the number of elements with our calculated extent
                result.c_elements *= result.m_dims[i];
                // Add the start coordinate for the dim to the offset query vector
                offset_query[dims.begin()[i]] = start;
            }
            // If we have other dims, grab their coordinates and dim id
            for (size_t i = 0; i < other_dims.size(); ++i) {
                auto [dim_num, dim_c] = other_dims.begin()[i];
                offset_query[dim_num] = dim_c;
            }
            // Calculate the offset from this Tensor and persist into the target
            result.c_offset = _get_offset(offset_query);
        }
        // Always check for contiguity before returning
        result._update_contiguous();
        return result;
    }

// NOLINTEND(cppcoreguidelines-avoid-c-arrays, cppcoreguidelines-pro-bounds-pointer-arithmetic, cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
};

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
/**
 * Apply a uniform basic math operation over a Tensor, reducing the amount of boilerplate
 * code required for the various operators and friends
 * @param lhs Const reference to a Tensor to serve as the lefthand side
 * @param rhs Const reference to a Tensor to serve as the righthand side
 * @param dest Reference to a Tensor to write the result to
 * @param op Target operation to perform
 * @returns Returns a reference to the destination Tensor
 */
template <typename T> 
requires std::is_arithmetic_v<T>
Tensor<T>& uniform_op(const Tensor<T>& lhs, const Tensor<T>& rhs, Tensor<T>& dest, SqueezedOpType op) {
    // Ensure the Tensors are compatible, including that the destination has the right shape
    if (!lhs._compatible(rhs) || !lhs._compatible(dest)) {
        throw std::invalid_argument("Tensor::uniform_op: Incompatible Tensor for uniform_op(lhs, rhs, dest, op).\n");
    }
    // Check for unsafe aliasing early and clone dest if necessary, then calling
    // uniform_op recursively to leverage a vectorized op path
    if (!dest.is_safe_aliasing(lhs) || !dest.is_safe_aliasing(rhs)) {
        // Clone the dest Tensor to ensure we have a fully contiguous,
        // disjoint memory block in Storage, allowing us to properly
        // vectorize the preferred code paths
        Tensor<T> temp_dest = dest.clone();
        uniform_op(lhs, rhs, temp_dest, op);
        // Copy the result back into dest
        dest.copy_from(temp_dest);
        return dest;
    }
    // If lhs, rhs, and dest don't overlap and are contiguous, use the fastest possible, vectorized
    // method to perform the operation. Ensure lhs --> dest <-- rhs are unique to respect
    // __restrict__ rules
    if (lhs.contiguous() && rhs.contiguous() && dest.contiguous() && dest.is_unique(lhs) && dest.is_unique(rhs)) {
        switch (op) {
            case SqueezedOpType::ADD:
                TensorMath_NS::_safe_tensor_add_contiguous_v(lhs._data(), lhs.offset(),
                                                             rhs._data(), rhs.offset(),
                                                             dest._data(), dest.offset(),
                                                             dest.elements());
                break;
            case SqueezedOpType::SUB:
                TensorMath_NS::_safe_tensor_sub_contiguous_v(lhs._data(), lhs.offset(),
                                                             rhs._data(), rhs.offset(),
                                                             dest._data(), dest.offset(),
                                                             dest.elements());
                break;
            case SqueezedOpType::MUL:
                TensorMath_NS::_safe_tensor_mul_contiguous_v(lhs._data(), lhs.offset(),
                                                             rhs._data(), rhs.offset(),
                                                             dest._data(), dest.offset(),
                                                             dest.elements());
                break;
            case SqueezedOpType::DIV:
                TensorMath_NS::_safe_tensor_div_contiguous_v(lhs._data(), lhs.offset(),
                                                             rhs._data(), rhs.offset(),
                                                             dest._data(), dest.offset(),
                                                             dest.elements());
                break;
        }
        return dest;
    }
    // If lhs, rhs, and dest are all contiguous, but the is_unique condition is not met, use
    // the non __restrict__ variant
    else if (lhs.contiguous() && rhs.contiguous() && dest.contiguous()) {
        switch (op) {
            case SqueezedOpType::ADD:
                TensorMath_NS::_safe_tensor_add_contiguous(lhs._data(), lhs.offset(),
                                                           rhs._data(), rhs.offset(),
                                                           dest._data(), dest.offset(),
                                                           dest.elements());
                break;
            case SqueezedOpType::SUB:
                TensorMath_NS::_safe_tensor_sub_contiguous(lhs._data(), lhs.offset(),
                                                           rhs._data(), rhs.offset(),
                                                           dest._data(), dest.offset(),
                                                           dest.elements());
                break;
            case SqueezedOpType::MUL:
                TensorMath_NS::_safe_tensor_mul_contiguous(lhs._data(), lhs.offset(),
                                                           rhs._data(), rhs.offset(),
                                                           dest._data(), dest.offset(),
                                                           dest.elements());
                break;
            case SqueezedOpType::DIV:
                TensorMath_NS::_safe_tensor_div_contiguous(lhs._data(), lhs.offset(),
                                                           rhs._data(), rhs.offset(),
                                                           dest._data(), dest.offset(),
                                                           dest.elements());
                break;
        }
        return dest;
    }
    // If the Tensors are not entirely contiguous, but are rank == 2 and they all have strides
    // of 1 for dim1, we can also speed up this op
    else if (lhs.rank() == 2 && (lhs.dim_stride(1) == 1 && rhs.dim_stride(1) == 1 && dest.dim_stride(1) == 1) && (dest.is_unique(lhs) && dest.is_unique(rhs))) {
        // Check for the case where we have a rank == 2 Tensor with a contiguous, 
        // non-overlapping destination, where we can leverage vectorization
        switch (op) {
            case SqueezedOpType::ADD:
                TensorMath_NS::_safe_2d_tensor_add_contiguous_v(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                rhs._data(), rhs.offset(), rhs.dim_stride(0),
                                                                dest._data(), dest.offset(), dest.dim_stride(0),
                                                                dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::SUB:
                TensorMath_NS::_safe_2d_tensor_sub_contiguous_v(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                rhs._data(), rhs.offset(), rhs.dim_stride(0),
                                                                dest._data(), dest.offset(), dest.dim_stride(0),
                                                                dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::MUL:
                TensorMath_NS::_safe_2d_tensor_mul_contiguous_v(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                rhs._data(), rhs.offset(), rhs.dim_stride(0),
                                                                dest._data(), dest.offset(), dest.dim_stride(0),
                                                                dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::DIV:
                TensorMath_NS::_safe_2d_tensor_div_contiguous_v(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                rhs._data(), rhs.offset(), rhs.dim_stride(0),
                                                                dest._data(), dest.offset(), dest.dim_stride(0),
                                                                dest.extent(0), dest.extent(1));
                break;
        }
        return dest; 
    }
    else if (lhs.rank() == 2 && (lhs.dim_stride(1) == 1 && rhs.dim_stride(1) == 1 && dest.dim_stride(1) == 1)) {
        // Check for the case where we have a rank == 2 Tensor with a contiguous, 
        // non-overlapping destination, where we can leverage vectorization
        switch (op) {
            case SqueezedOpType::ADD:
                TensorMath_NS::_safe_2d_tensor_add_contiguous(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                              rhs._data(), rhs.offset(), rhs.dim_stride(0),
                                                              dest._data(), dest.offset(), dest.dim_stride(0),
                                                              dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::SUB:
                TensorMath_NS::_safe_2d_tensor_sub_contiguous(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                              rhs._data(), rhs.offset(), rhs.dim_stride(0),
                                                              dest._data(), dest.offset(), dest.dim_stride(0),
                                                              dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::MUL:
                TensorMath_NS::_safe_2d_tensor_mul_contiguous(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                              rhs._data(), rhs.offset(), rhs.dim_stride(0),
                                                              dest._data(), dest.offset(), dest.dim_stride(0),
                                                              dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::DIV:
                TensorMath_NS::_safe_2d_tensor_div_contiguous(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                              rhs._data(), rhs.offset(), rhs.dim_stride(0),
                                                              dest._data(), dest.offset(), dest.dim_stride(0),
                                                              dest.extent(0), dest.extent(1));
                break;
        }
        return dest; 
    }
    // If we are dealing with either a non-contiguous Tensor or some other edge case,
    // iterate via at(). Since this cannot be vectorized, throw exceptions immediately
    // if overflow / underflow occur
    else {
        // _compatible ensures that the Tensors have the same shape and number of 
        // elements, so iterate using _get_offset(size_t) to save boilerplate code
        bool overflowed = false;
        // Grab the data pointers to avoid repeat calls
        const T* lhs_data = lhs._data();
        const T* rhs_data = rhs._data();
        T* dest_data = dest._data();
        if constexpr (std::is_floating_point_v<T>) {
            switch (op) {
                case SqueezedOpType::ADD:
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        const T rhs_val = rhs_data[rhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        dest_val = lhs_val + rhs_val;
                        overflowed |= !std::isfinite(dest_val);
                    }
                    break;
                case SqueezedOpType::SUB:
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        const T rhs_val = rhs_data[rhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        dest_val = lhs_val - rhs_val;
                        overflowed |= !std::isfinite(dest_val);
                    }
                    break;
                case SqueezedOpType::MUL:
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        const T rhs_val = rhs_data[rhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        dest_val = lhs_val * rhs_val;
                        overflowed |= !std::isfinite(dest_val);
                    }
                    break;
                case SqueezedOpType::DIV:
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        const T rhs_val = rhs_data[rhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        dest_val = lhs_val / rhs_val;
                        overflowed |= !std::isfinite(dest_val);
                    }
                    break;
            }
            if (overflowed) {
                throw std::overflow_error(std::format("Tensor::uniform_op: Overflow / underflow detected when performing {}", squeezed_op_to_string(op)));
            }
        }
        else if constexpr (std::is_signed_v<T>) {
            switch (op) {
                case SqueezedOpType::ADD:
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        const T rhs_val = rhs_data[rhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        overflowed |= TensorMath_NS::_add_overflow_signed(lhs_val, rhs_val, &dest_val);
                    }
                    break;
                case SqueezedOpType::SUB:
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        const T rhs_val = rhs_data[rhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        overflowed |= TensorMath_NS::_sub_overflow_signed(lhs_val, rhs_val, &dest_val);
                    }
                    break;
                case SqueezedOpType::MUL:
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        const T rhs_val = rhs_data[rhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        overflowed |= TensorMath_NS::_mul_overflow_signed(lhs_val, rhs_val, &dest_val);
                    }
                    break;
                case SqueezedOpType::DIV:
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        const T rhs_val = rhs_data[rhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        if (rhs_val == 0) {
                            throw std::runtime_error("Tensor::uniform_op: Divide by zero detected.\n");
                        }
                        if (TensorMath_NS::_is_signed_div_overflow(lhs_val, rhs_val)) {
                            throw std::overflow_error("Tensor::uniform_op: Division causes overflow / underflow.\n");
                        }
                        dest_val = lhs_val / rhs_val;
                    }
                    break;
            }
            if (overflowed) {
                throw std::overflow_error(std::format("Tensor::uniform_op: Overflow / underflow detected when performing {}", squeezed_op_to_string(op)));
            }
        }
        else {
            switch (op) {
                case SqueezedOpType::ADD:
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        const T rhs_val = rhs_data[rhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        overflowed |= TensorMath_NS::_add_overflow_unsigned(lhs_val, rhs_val, &dest_val);
                    }
                    break;
                case SqueezedOpType::SUB:
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        const T rhs_val = rhs_data[rhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        overflowed |= TensorMath_NS::_sub_overflow_unsigned(lhs_val, rhs_val, &dest_val);
                    }
                    break;
                case SqueezedOpType::MUL:
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        const T rhs_val = rhs_data[rhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        overflowed |= TensorMath_NS::_mul_overflow_unsigned(lhs_val, rhs_val, &dest_val);
                    }
                    break;
                case SqueezedOpType::DIV:
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        const T rhs_val = rhs_data[rhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        if (rhs_val == 0) {
                            throw std::runtime_error("Tensor::uniform_op: Divide by zero detected.\n");
                        }
                        dest_val = lhs_val / rhs_val;
                    }
                    break;
            }
            if (overflowed) {
                throw std::overflow_error(std::format("Tensor::uniform_op: Overflow / underflow detected when performing {}", squeezed_op_to_string(op)));
            }
        }
        return dest;
    }
}

/**
 * Apply a uniform basic math operation over a Tensor, reducing the amount of boilerplate
 * code required for the various operators and friends
 * @param lhs Const reference to a Tensor to serve as the lefthand side
 * @param rhs_val Scalar value to perform the op with lhs
 * @param dest Reference to a Tensor to write the result to
 * @param op Target operation to perform
 * @returns Returns a reference to the destination Tensor
 */
template <typename T> 
requires std::is_arithmetic_v<T>
Tensor<T>& uniform_op(const Tensor<T>& lhs, T rhs_val, Tensor<T>& dest, SqueezedOpType op) {
    // Ensure the Tensors are compatible, including that the destination has the right shape
    if (!lhs._compatible(dest)) {
        throw std::invalid_argument("Tensor::uniform_op: Incompatible Tensor for uniform_op(lhs, rhs_val, dest, op).\n");
    }
    // Check for unsafe aliasing early and clone dest if necessary, then calling
    // uniform_op recursively to leverage a vectorized op path
    if (!dest.is_safe_aliasing(lhs)) {
        // Clone the dest Tensor to ensure we have a fully contiguous,
        // disjoint memory block in Storage, allowing us to properly
        // vectorize the preferred code paths
        Tensor<T> temp_dest = dest.clone();
        uniform_op(lhs, rhs_val, temp_dest, op);
        // Copy the result back into dest
        dest.copy_from(temp_dest);
        return dest;
    }
    // If lhs and dest don't overlap and are contiguous, use the fastest possible, vectorized
    // method to perform the operation. Ensure lhs and dest are unique to respect
    // __restrict__ rules
    if (lhs.contiguous() && dest.contiguous() && dest.is_unique(lhs) ) {
        switch (op) {
            case SqueezedOpType::ADD:
                TensorMath_NS::_safe_tensor_add_contiguous_v(lhs._data(), lhs.offset(),
                                                             rhs_val,
                                                             dest._data(), dest.offset(),
                                                             dest.elements());
                break;
            case SqueezedOpType::SUB:
                TensorMath_NS::_safe_tensor_sub_contiguous_v(lhs._data(), lhs.offset(),
                                                             rhs_val,
                                                             dest._data(), dest.offset(),
                                                             dest.elements());
                break;
            case SqueezedOpType::MUL:
                TensorMath_NS::_safe_tensor_mul_contiguous_v(lhs._data(), lhs.offset(),
                                                             rhs_val,
                                                             dest._data(), dest.offset(),
                                                             dest.elements());
                break;
            case SqueezedOpType::DIV:
                TensorMath_NS::_safe_tensor_div_contiguous_v(lhs._data(), lhs.offset(),
                                                             rhs_val,
                                                             dest._data(), dest.offset(),
                                                             dest.elements());
                break;
        }
        return dest;
    }
    // If lhs and dest are contiguous, but the is_unique condition is not met, use
    // the non __restrict__ variant
    else if (lhs.contiguous() && dest.contiguous()) {
        switch (op) {
            case SqueezedOpType::ADD:
                TensorMath_NS::_safe_tensor_add_contiguous(lhs._data(), lhs.offset(),
                                                           rhs_val,
                                                           dest._data(), dest.offset(),
                                                           dest.elements());
                break;
            case SqueezedOpType::SUB:
                TensorMath_NS::_safe_tensor_sub_contiguous(lhs._data(), lhs.offset(),
                                                           rhs_val,
                                                           dest._data(), dest.offset(),
                                                           dest.elements());
                break;
            case SqueezedOpType::MUL:
                TensorMath_NS::_safe_tensor_mul_contiguous(lhs._data(), lhs.offset(),
                                                           rhs_val,
                                                           dest._data(), dest.offset(),
                                                           dest.elements());
                break;
            case SqueezedOpType::DIV:
                TensorMath_NS::_safe_tensor_div_contiguous(lhs._data(), lhs.offset(),
                                                           rhs_val,
                                                           dest._data(), dest.offset(),
                                                           dest.elements());
                break;
        }
        return dest;
    }
    // If the Tensors are not entirely contiguous, but are rank == 2 and they all have strides
    // of 1 for dim1, we can also speed up this op
    else if (lhs.rank() == 2 && (lhs.dim_stride(1) == 1 && dest.dim_stride(1) == 1) && dest.is_unique(lhs)) {
        // Check for the case where we have a rank == 2 Tensor with a contiguous, 
        // non-overlapping destination, where we can leverage vectorization
        switch (op) {
            case SqueezedOpType::ADD:
                TensorMath_NS::_safe_2d_tensor_add_contiguous_v(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                rhs_val,
                                                                dest._data(), dest.offset(), dest.dim_stride(0),
                                                                dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::SUB:
                TensorMath_NS::_safe_2d_tensor_sub_contiguous_v(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                rhs_val,
                                                                dest._data(), dest.offset(), dest.dim_stride(0),
                                                                dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::MUL:
                TensorMath_NS::_safe_2d_tensor_mul_contiguous_v(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                rhs_val,
                                                                dest._data(), dest.offset(), dest.dim_stride(0),
                                                                dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::DIV:
                TensorMath_NS::_safe_2d_tensor_div_contiguous_v(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                rhs_val,
                                                                dest._data(), dest.offset(), dest.dim_stride(0),
                                                                dest.extent(0), dest.extent(1));
                break;
        }
        return dest; 
    }
    else if (lhs.rank() == 2 && (lhs.dim_stride(1) == 1 && dest.dim_stride(1) == 1)) {
        // Check for the case where we have a rank == 2 Tensor with a contiguous, 
        // non-overlapping destination, where we can leverage vectorization
        switch (op) {
            case SqueezedOpType::ADD:
                TensorMath_NS::_safe_2d_tensor_add_contiguous(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                              rhs_val,
                                                              dest._data(), dest.offset(), dest.dim_stride(0),
                                                              dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::SUB:
                TensorMath_NS::_safe_2d_tensor_sub_contiguous(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                              rhs_val,
                                                              dest._data(), dest.offset(), dest.dim_stride(0),
                                                              dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::MUL:
                TensorMath_NS::_safe_2d_tensor_mul_contiguous(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                              rhs_val,
                                                              dest._data(), dest.offset(), dest.dim_stride(0),
                                                              dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::DIV:
                TensorMath_NS::_safe_2d_tensor_div_contiguous(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                              rhs_val,
                                                              dest._data(), dest.offset(), dest.dim_stride(0),
                                                              dest.extent(0), dest.extent(1));
                break;
        }
        return dest; 
    }
    // If we are dealing with either a non-contiguous Tensor or some other edge case,
    // iterate via at(). Since this cannot be vectorized, throw exceptions immediately
    // if overflow / underflow occur
    else {
        // _compatible ensures that the Tensors have the same shape and number of 
        // elements, so iterate using _get_offset(size_t) to save boilerplate code
        bool overflowed = false;
        // Grab the data pointers to avoid repeat calls
        const T* lhs_data = lhs._data();
        T* dest_data = dest._data();
        if constexpr (std::is_floating_point_v<T>) {
            switch (op) {
                case SqueezedOpType::ADD:
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        dest_val = lhs_val + rhs_val;
                        overflowed |= !std::isfinite(dest_val);
                    }
                    break;
                case SqueezedOpType::SUB:
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        dest_val = lhs_val - rhs_val;
                        overflowed |= !std::isfinite(dest_val);
                    }
                    break;
                case SqueezedOpType::MUL:
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        dest_val = lhs_val * rhs_val;
                        overflowed |= !std::isfinite(dest_val);
                    }
                    break;
                case SqueezedOpType::DIV:
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        dest_val = lhs_val / rhs_val;
                        overflowed |= !std::isfinite(dest_val);
                    }
                    break;
            }
            if (overflowed) {
                throw std::overflow_error(std::format("Tensor::uniform_op: Overflow / underflow detected when performing {}", squeezed_op_to_string(op)));
            }
        }
        else if constexpr (std::is_signed_v<T>) {
            switch (op) {
                case SqueezedOpType::ADD:
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        overflowed |= TensorMath_NS::_add_overflow_signed(lhs_val, rhs_val, &dest_val);
                    }
                    break;
                case SqueezedOpType::SUB:
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        overflowed |= TensorMath_NS::_sub_overflow_signed(lhs_val, rhs_val, &dest_val);
                    }
                    break;
                case SqueezedOpType::MUL:
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        overflowed |= TensorMath_NS::_mul_overflow_signed(lhs_val, rhs_val, &dest_val);
                    }
                    break;
                case SqueezedOpType::DIV:
                    if (rhs_val == 0) {
                        throw std::runtime_error("Tensor::uniform_op: Divide by zero detected.\n");
                    }
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        if (TensorMath_NS::_is_signed_div_overflow(lhs_val, rhs_val)) {
                            throw std::overflow_error("Tensor::uniform_op: Division causes overflow / underflow.\n");
                        }
                        dest_val = lhs_val / rhs_val;
                    }
                    break;
            }
            if (overflowed) {
                throw std::overflow_error(std::format("Tensor::uniform_op: Overflow / underflow detected when performing {}", squeezed_op_to_string(op)));
            }
        }
        else {
            switch (op) {
                case SqueezedOpType::ADD:
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        overflowed |= TensorMath_NS::_add_overflow_unsigned(lhs_val, rhs_val, &dest_val);
                    }
                    break;
                case SqueezedOpType::SUB:
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        overflowed |= TensorMath_NS::_sub_overflow_unsigned(lhs_val, rhs_val, &dest_val);
                    }
                    break;
                case SqueezedOpType::MUL:
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        overflowed |= TensorMath_NS::_mul_overflow_unsigned(lhs_val, rhs_val, &dest_val);
                    }
                    break;
                case SqueezedOpType::DIV:
                    if (rhs_val == 0) {
                        throw std::runtime_error("Tensor::uniform_op: Divide by zero detected.\n");
                    }
                    for (size_t i = 0; i < dest.elements(); ++i) {
                        const T lhs_val = lhs_data[lhs._get_offset(i)];
                        T& dest_val = dest_data[dest._get_offset(i)];
                        dest_val = lhs_val / rhs_val;
                    }
                    break;
            }
            if (overflowed) {
                throw std::overflow_error(std::format("Tensor::uniform_op: Overflow / underflow detected when performing {}", squeezed_op_to_string(op)));
            }
        }
        return dest;
    }
}

/**
 * Apply a uniform basic math operation over a Tensor, reducing the amount of boilerplate
 * code required for the various operators and friends
 * @param lhs Const reference to a Tensor to serve as the lefthand side
 * @param rhs Const reference to a Tensor to serve as the righthand side
 * @param dest Reference to a Tensor to write the result to
 * @param op Target operation to perform
 * @returns Returns a reference to the destination Tensor
 */
template <typename T> 
requires std::is_floating_point_v<T>
Tensor<T>& unsafe_uniform_op(const Tensor<T>& lhs, const Tensor<T>& rhs, Tensor<T>& dest, SqueezedOpType op) {
    // Ensure the Tensors are compatible, including that the destination has the right shape
    if (!lhs._compatible(rhs) || !lhs._compatible(dest)) {
        throw std::invalid_argument("Tensor::unsafe_uniform_op: Incompatible Tensor for unsafe_uniform_op(lhs, rhs, dest, op).\n");
    }
    // Check for unsafe aliasing early and clone dest if necessary, then calling
    // unsafe_uniform_op recursively to leverage a vectorized op path
    if (!dest.is_safe_aliasing(lhs) || !dest.is_safe_aliasing(rhs)) {
        // Clone the dest Tensor to ensure we have a fully contiguous,
        // disjoint memory block in Storage, allowing us to properly
        // vectorize the preferred code paths
        Tensor<T> temp_dest = dest.clone();
        unsafe_uniform_op(lhs, rhs, temp_dest, op);
        // Copy the result back into dest
        dest.copy_from(temp_dest);
        return dest;
    }
    // If lhs, rhs, and dest don't overlap and are contiguous, use the fastest possible, vectorized
    // method to perform the operation. Ensure lhs --> dest <-- rhs are unique to respect
    // __restrict__ rules
    if (lhs.contiguous() && rhs.contiguous() && dest.contiguous() && dest.is_unique(lhs) && dest.is_unique(rhs)) {
        switch (op) {
            case SqueezedOpType::ADD:
                TensorMath_NS::_unsafe_float_tensor_add_contiguous_v(lhs._data(), lhs.offset(),
                                                                     rhs._data(), rhs.offset(),
                                                                     dest._data(), dest.offset(),
                                                                     dest.elements());
                break;
            case SqueezedOpType::SUB:
                TensorMath_NS::_unsafe_float_tensor_sub_contiguous_v(lhs._data(), lhs.offset(),
                                                                     rhs._data(), rhs.offset(),
                                                                     dest._data(), dest.offset(),
                                                                     dest.elements());
                break;
            case SqueezedOpType::MUL:
                TensorMath_NS::_unsafe_float_tensor_mul_contiguous_v(lhs._data(), lhs.offset(),
                                                                     rhs._data(), rhs.offset(),
                                                                     dest._data(), dest.offset(),
                                                                     dest.elements());
                break;
            case SqueezedOpType::DIV:
                TensorMath_NS::_unsafe_float_tensor_div_contiguous_v(lhs._data(), lhs.offset(),
                                                                     rhs._data(), rhs.offset(),
                                                                     dest._data(), dest.offset(),
                                                                     dest.elements());
                break;
        }
        return dest;
    }
    // If lhs, rhs, and dest are all contiguous, but the is_unique condition is not met, use
    // the non __restrict__ variant
    else if (lhs.contiguous() && rhs.contiguous() && dest.contiguous()) {
        switch (op) {
            case SqueezedOpType::ADD:
                TensorMath_NS::_unsafe_float_tensor_add_contiguous(lhs._data(), lhs.offset(),
                                                                   rhs._data(), rhs.offset(),
                                                                   dest._data(), dest.offset(),
                                                                   dest.elements());
                break;
            case SqueezedOpType::SUB:
                TensorMath_NS::_unsafe_float_tensor_sub_contiguous(lhs._data(), lhs.offset(),
                                                                   rhs._data(), rhs.offset(),
                                                                   dest._data(), dest.offset(),
                                                                   dest.elements());
                break;
            case SqueezedOpType::MUL:
                TensorMath_NS::_unsafe_float_tensor_mul_contiguous(lhs._data(), lhs.offset(),
                                                                   rhs._data(), rhs.offset(),
                                                                   dest._data(), dest.offset(),
                                                                   dest.elements());
                break;
            case SqueezedOpType::DIV:
                TensorMath_NS::_unsafe_float_tensor_div_contiguous(lhs._data(), lhs.offset(),
                                                                   rhs._data(), rhs.offset(),
                                                                   dest._data(), dest.offset(),
                                                                   dest.elements());
                break;
        }
        return dest;
    }
    // If the Tensors are not entirely contiguous, but are rank == 2 and they all have strides
    // of 1 for dim1, we can also speed up this op
    else if (lhs.rank() == 2 && (lhs.dim_stride(1) == 1 && rhs.dim_stride(1) == 1 && dest.dim_stride(1) == 1) && (dest.is_unique(lhs) && dest.is_unique(rhs))) {
        // Check for the case where we have a rank == 2 Tensor with a contiguous, 
        // non-overlapping destination, where we can leverage vectorization
        switch (op) {
            case SqueezedOpType::ADD:
                TensorMath_NS::_unsafe_float_2d_tensor_add_contiguous_v(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                        rhs._data(), rhs.offset(), rhs.dim_stride(0),
                                                                        dest._data(), dest.offset(), dest.dim_stride(0),
                                                                        dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::SUB:
                TensorMath_NS::_unsafe_float_2d_tensor_sub_contiguous_v(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                        rhs._data(), rhs.offset(), rhs.dim_stride(0),
                                                                        dest._data(), dest.offset(), dest.dim_stride(0),
                                                                        dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::MUL:
                TensorMath_NS::_unsafe_float_2d_tensor_mul_contiguous_v(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                        rhs._data(), rhs.offset(), rhs.dim_stride(0),
                                                                        dest._data(), dest.offset(), dest.dim_stride(0),
                                                                        dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::DIV:
                TensorMath_NS::_unsafe_float_2d_tensor_div_contiguous_v(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                        rhs._data(), rhs.offset(), rhs.dim_stride(0),
                                                                        dest._data(), dest.offset(), dest.dim_stride(0),
                                                                        dest.extent(0), dest.extent(1));
                break;
        }
        return dest; 
    }
    else if (lhs.rank() == 2 && (lhs.dim_stride(1) == 1 && rhs.dim_stride(1) == 1 && dest.dim_stride(1) == 1)) {
        // Check for the case where we have a rank == 2 Tensor with a contiguous, 
        // non-overlapping destination, where we can leverage vectorization
        switch (op) {
            case SqueezedOpType::ADD:
                TensorMath_NS::_unsafe_float_2d_tensor_add_contiguous(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                      rhs._data(), rhs.offset(), rhs.dim_stride(0),
                                                                      dest._data(), dest.offset(), dest.dim_stride(0),
                                                                      dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::SUB:
                TensorMath_NS::_unsafe_float_2d_tensor_sub_contiguous(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                      rhs._data(), rhs.offset(), rhs.dim_stride(0),
                                                                      dest._data(), dest.offset(), dest.dim_stride(0),
                                                                      dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::MUL:
                TensorMath_NS::_unsafe_float_2d_tensor_mul_contiguous(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                      rhs._data(), rhs.offset(), rhs.dim_stride(0),
                                                                      dest._data(), dest.offset(), dest.dim_stride(0),
                                                                      dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::DIV:
                TensorMath_NS::_unsafe_float_2d_tensor_div_contiguous(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                      rhs._data(), rhs.offset(), rhs.dim_stride(0),
                                                                      dest._data(), dest.offset(), dest.dim_stride(0),
                                                                      dest.extent(0), dest.extent(1));
                break;
        }
        return dest; 
    }
    // If we are dealing with either a non-contiguous Tensor or some other edge case,
    // iterate via at(). Since this cannot be vectorized, throw exceptions immediately
    // if overflow / underflow occur
    else {
        // _compatible ensures that the Tensors have the same shape and number of 
        // elements, so iterate using _get_offset(size_t) to save boilerplate code
        // Grab the data pointers to avoid repeat calls
        const T* lhs_data = lhs._data();
        const T* rhs_data = rhs._data();
        T* dest_data = dest._data();
        switch (op) {
            case SqueezedOpType::ADD:
                for (size_t i = 0; i < dest.elements(); ++i) {
                    const T lhs_val = lhs_data[lhs._get_offset(i)];
                    const T rhs_val = rhs_data[rhs._get_offset(i)];
                    T& dest_val = dest_data[dest._get_offset(i)];
                    dest_val = lhs_val + rhs_val;
                }
                break;
            case SqueezedOpType::SUB:
                for (size_t i = 0; i < dest.elements(); ++i) {
                    const T lhs_val = lhs_data[lhs._get_offset(i)];
                    const T rhs_val = rhs_data[rhs._get_offset(i)];
                    T& dest_val = dest_data[dest._get_offset(i)];
                    dest_val = lhs_val - rhs_val;
                }
                break;
            case SqueezedOpType::MUL:
                for (size_t i = 0; i < dest.elements(); ++i) {
                    const T lhs_val = lhs_data[lhs._get_offset(i)];
                    const T rhs_val = rhs_data[rhs._get_offset(i)];
                    T& dest_val = dest_data[dest._get_offset(i)];
                    dest_val = lhs_val * rhs_val;
                }
                break;
            case SqueezedOpType::DIV:
                for (size_t i = 0; i < dest.elements(); ++i) {
                    const T lhs_val = lhs_data[lhs._get_offset(i)];
                    const T rhs_val = rhs_data[rhs._get_offset(i)];
                    T& dest_val = dest_data[dest._get_offset(i)];
                    dest_val = lhs_val / rhs_val;
                }
                break;
            }
        return dest;
    }
}

/**
 * Apply a uniform basic math operation over a Tensor, reducing the amount of boilerplate
 * code required for the various operators and friends
 * @param lhs Const reference to a Tensor to serve as the lefthand side
 * @param rhs_val Scalar value to perform the op with lhs
 * @param dest Reference to a Tensor to write the result to
 * @param op Target operation to perform
 * @returns Returns a reference to the destination Tensor
 */
template <typename T> 
requires std::is_floating_point_v<T>
Tensor<T>& unsafe_uniform_op(const Tensor<T>& lhs, T rhs_val, Tensor<T>& dest, SqueezedOpType op) {
    // Ensure the Tensors are compatible, including that the destination has the right shape
    if (!lhs._compatible(dest)) {
        throw std::invalid_argument("Tensor::unsafe_uniform_op: Incompatible Tensor for unsafe_uniform_op(lhs, rhs_val, dest, op).\n");
    }
    // Check for unsafe aliasing early and clone dest if necessary, then calling
    // unsafe_uniform_op recursively to leverage a vectorized op path
    if (!dest.is_safe_aliasing(lhs)) {
        // Clone the dest Tensor to ensure we have a fully contiguous,
        // disjoint memory block in Storage, allowing us to properly
        // vectorize the preferred code paths
        Tensor<T> temp_dest = dest.clone();
        unsafe_uniform_op(lhs, rhs_val, temp_dest, op);
        // Copy the result back into dest
        dest.copy_from(temp_dest);
        return dest;
    }
    // If lhs and dest don't overlap and are contiguous, use the fastest possible, vectorized
    // method to perform the operation. Ensure lhs and dest are unique to respect
    // __restrict__ rules
    if (lhs.contiguous() && dest.contiguous() && dest.is_unique(lhs) ) {
        switch (op) {
            case SqueezedOpType::ADD:
                TensorMath_NS::_unsafe_float_tensor_add_contiguous_v(lhs._data(), lhs.offset(),
                                                                     rhs_val,
                                                                     dest._data(), dest.offset(),
                                                                     dest.elements());
                break;
            case SqueezedOpType::SUB:
                TensorMath_NS::_unsafe_float_tensor_sub_contiguous_v(lhs._data(), lhs.offset(),
                                                                     rhs_val,
                                                                     dest._data(), dest.offset(),
                                                                     dest.elements());
                break;
            case SqueezedOpType::MUL:
                TensorMath_NS::_unsafe_float_tensor_mul_contiguous_v(lhs._data(), lhs.offset(),
                                                                     rhs_val,
                                                                     dest._data(), dest.offset(),
                                                                     dest.elements());
                break;
            case SqueezedOpType::DIV:
                TensorMath_NS::_unsafe_float_tensor_div_contiguous_v(lhs._data(), lhs.offset(),
                                                                     rhs_val,
                                                                     dest._data(), dest.offset(),
                                                                     dest.elements());
                break;
        }
        return dest;
    }
    // If lhs and dest are contiguous, but the is_unique condition is not met, use
    // the non __restrict__ variant
    else if (lhs.contiguous() && dest.contiguous()) {
        switch (op) {
            case SqueezedOpType::ADD:
                TensorMath_NS::_unsafe_float_tensor_add_contiguous(lhs._data(), lhs.offset(),
                                                                   rhs_val,
                                                                   dest._data(), dest.offset(),
                                                                   dest.elements());
                break;
            case SqueezedOpType::SUB:
                TensorMath_NS::_unsafe_float_tensor_sub_contiguous(lhs._data(), lhs.offset(),
                                                                   rhs_val,
                                                                   dest._data(), dest.offset(),
                                                                   dest.elements());
                break;
            case SqueezedOpType::MUL:
                TensorMath_NS::_unsafe_float_tensor_mul_contiguous(lhs._data(), lhs.offset(),
                                                                   rhs_val,
                                                                   dest._data(), dest.offset(),
                                                                   dest.elements());
                break;
            case SqueezedOpType::DIV:
                TensorMath_NS::_unsafe_float_tensor_div_contiguous(lhs._data(), lhs.offset(),
                                                                   rhs_val,
                                                                   dest._data(), dest.offset(),
                                                                   dest.elements());
                break;
        }
        return dest;
    }
    // If the Tensors are not entirely contiguous, but are rank == 2 and they all have strides
    // of 1 for dim1, we can also speed up this op
    else if (lhs.rank() == 2 && (lhs.dim_stride(1) == 1 && dest.dim_stride(1) == 1) && dest.is_unique(lhs)) {
        // Check for the case where we have a rank == 2 Tensor with a contiguous, 
        // non-overlapping destination, where we can leverage vectorization
        switch (op) {
            case SqueezedOpType::ADD:
                TensorMath_NS::_unsafe_float_2d_tensor_add_contiguous_v(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                        rhs_val,
                                                                        dest._data(), dest.offset(), dest.dim_stride(0),
                                                                        dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::SUB:
                TensorMath_NS::_unsafe_float_2d_tensor_sub_contiguous_v(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                        rhs_val,
                                                                        dest._data(), dest.offset(), dest.dim_stride(0),
                                                                        dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::MUL:
                TensorMath_NS::_unsafe_float_2d_tensor_mul_contiguous_v(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                        rhs_val,
                                                                        dest._data(), dest.offset(), dest.dim_stride(0),
                                                                        dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::DIV:
                TensorMath_NS::_unsafe_float_2d_tensor_div_contiguous_v(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                        rhs_val,
                                                                        dest._data(), dest.offset(), dest.dim_stride(0),
                                                                        dest.extent(0), dest.extent(1));
                break;
        }
        return dest; 
    }
    else if (lhs.rank() == 2 && (lhs.dim_stride(1) == 1 && dest.dim_stride(1) == 1)) {
        // Check for the case where we have a rank == 2 Tensor with a contiguous, 
        // non-overlapping destination, where we can leverage vectorization
        switch (op) {
            case SqueezedOpType::ADD:
                TensorMath_NS::_unsafe_float_2d_tensor_add_contiguous(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                      rhs_val,
                                                                      dest._data(), dest.offset(), dest.dim_stride(0),
                                                                      dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::SUB:
                TensorMath_NS::_unsafe_float_2d_tensor_sub_contiguous(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                      rhs_val,
                                                                      dest._data(), dest.offset(), dest.dim_stride(0),
                                                                      dest.extent(0), dest.extent(1));
                break;
            case SqueezedOpType::MUL:
                TensorMath_NS::_unsafe_float_2d_tensor_mul_contiguous(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                      rhs_val,
                                                                      dest._data(), dest.offset(), dest.dim_stride(0),
                                                                      dest.extent(0), dest.extent(1));
            case SqueezedOpType::DIV:
                TensorMath_NS::_unsafe_float_2d_tensor_div_contiguous(lhs._data(), lhs.offset(), lhs.dim_stride(0),
                                                                      rhs_val,
                                                                      dest._data(), dest.offset(), dest.dim_stride(0),
                                                                      dest.extent(0), dest.extent(1));
                break;
        }
        return dest; 
    }
    // If we are dealing with either a non-contiguous Tensor or some other edge case,
    // iterate via at(). Since this cannot be vectorized, throw exceptions immediately
    // if overflow / underflow occur
    else {
        // _compatible ensures that the Tensors have the same shape and number of 
        // elements, so iterate using _get_offset(size_t) to save boilerplate code
        // Grab the data pointers to avoid repeat calls
        const T* lhs_data = lhs._data();
        T* dest_data = dest._data();
        switch (op) {
            case SqueezedOpType::ADD:
                for (size_t i = 0; i < dest.elements(); ++i) {
                    const T lhs_val = lhs_data[lhs._get_offset(i)];
                    T& dest_val = dest_data[dest._get_offset(i)];
                    dest_val = lhs_val + rhs_val;
                }
                break;
            case SqueezedOpType::SUB:
                for (size_t i = 0; i < dest.elements(); ++i) {
                    const T lhs_val = lhs_data[lhs._get_offset(i)];
                    T& dest_val = dest_data[dest._get_offset(i)];
                    dest_val = lhs_val - rhs_val;
                }
                break;
            case SqueezedOpType::MUL:
                for (size_t i = 0; i < dest.elements(); ++i) {
                    const T lhs_val = lhs_data[lhs._get_offset(i)];
                    T& dest_val = dest_data[dest._get_offset(i)];
                    dest_val = lhs_val * rhs_val;
                }
                break;
            case SqueezedOpType::DIV:
                for (size_t i = 0; i < dest.elements(); ++i) {
                    const T lhs_val = lhs_data[lhs._get_offset(i)];
                    T& dest_val = dest_data[dest._get_offset(i)];
                    dest_val = lhs_val / rhs_val;
                }
                break;
        }
        return dest;
    }
}
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

}; // namespace Tensor_NS

#endif
