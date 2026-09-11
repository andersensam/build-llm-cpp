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
 * @version: 2026-09-10
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef TENSOR_HPP
#define TENSOR_HPP

/* Standard dependencies */
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
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
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <vector>

/* Local dependencies */
#include "Log.hpp"

namespace Tensor_NS {

/* Control verbose copy / move constructor logging, useful for debugging. Does not apply to default constructors */
inline constexpr bool TENSOR_ENABLE_CONSTRUCTOR_LOGGING = true;
/* Store the maximum length for a type name from the demangler API */
inline constexpr size_t TENSOR_MAX_DEMANGLED_NAME_LEN = 32;
/* Control logging for softmax warnings (NAN, inf, divide by 0) */
inline constexpr bool TENSOR_ENABLE_SOFTMAX_WARNINGS = true;

/* Use Logging functions */
using Log::Log_Priority;
using Log::log_message;

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
 * Compiler-independent check for addition overflow / underflow
 * @param a First value
 * @param b Second value
 * @param result Pointer to store the result in
 * @returns Returns true if the operation would cause overflow / underflow
 */
template <typename T> 
requires std::is_arithmetic_v<T>
[[nodiscard]] inline bool _add_overflow(T a, T b, T* result) {
    // We always check to see if the type is eligible for overflow before calling
    // this function, so no need to check again
    if constexpr (std::numeric_limits<T>::is_signed) {
        if (((b > 0) && (a > (std::numeric_limits<T>::max() - b))) || 
            ((b < 0) && (a < (std::numeric_limits<T>::min() - b)))) {
            return true;
        }
    }
    else {
        if (a > (std::numeric_limits<T>::max() - b)) {
            return true;
        }
    }
    *result = a + b;
    return false;
}

/**
 * Compiler-independent check for subtraction overflow / underflow
 * @param a First value
 * @param b Second value
 * @param result Pointer to store the result in
 * @returns Returns true if the operation would cause overflow / underflow
 */
template <typename T> 
requires std::is_arithmetic_v<T>
[[nodiscard]] inline bool _sub_overflow(T a, T b, T* result) {
    // We always check to see if the type is eligible for overflow before calling
    // this function, so no need to check again
    if constexpr (std::numeric_limits<T>::is_signed) {
        if (((b > 0) && (a < (std::numeric_limits<T>::min() + b))) || 
            ((b < 0) && (a > (std::numeric_limits<T>::max() + b)))) {
            return true;
        }
    }
    else {
        if (b > a) {
            return true;
        }
    }
    *result = a - b;
    return false;
}

/**
 * Compiler-independent check for multiplication overflow / underflow
 * @param a First value
 * @param b Second value
 * @param result Pointer to store the result in
 * @returns Returns true if the operation would cause overflow / underflow
 */
template <typename T> 
requires std::is_arithmetic_v<T>
[[nodiscard]] inline bool _mul_overflow(T a, T b, T* result) {
    // We always check to see if the type is eligible for overflow before calling
    // this function, so no need to check again
    if (a == 0 || b == 0) {
        *result = 0;
        return false;
    }
    // Get the min and max values for the type T
    T min_val = std::numeric_limits<T>::min();
    T max_val = std::numeric_limits<T>::max();
    // Validate the multiplication
    if (a > 0) {
        if (b > 0) {
            if (a > (max_val / b)) {
                return true;
            }
            else {
                *result = a * b;
                return false;
            }
        }
        else {
            if (b < (min_val / a)) {
                return true;
            }
            else {
                *result = a * b;
                return false;
            }
        }
    }
    else {
        if (b > 0) {
            if (a < (min_val / b)) {
                return true;
            }
            else {
                *result = a * b;
                return false;
            }
        }
        else {
            if (a < (max_val) / b) {
                return true;
            }
            else {
                *result = a * b;
                return false;
            }
        }
    }
}

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
     * Whether or not the underlying data type of the Tensor has a risk of overflow / underflow
     */
    static constexpr bool _can_overflow = std::is_same_v<T, char> || std::is_same_v<T, signed char> || std::is_same_v<T, int> || std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> || std::is_same_v<T, unsigned char> || std::is_same_v<T, unsigned int> || std::is_same_v<T, size_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>;

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
};
// NOLINTEND(cppcoreguidelines-special-member-functions)

/* Forward delcataion for Tensor and the matmul functions */
template <typename T> 
requires std::is_arithmetic_v<T>
class Tensor;

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
/**
 * Perform a matmul across two Tensors, with their target dims specified
 * @param lhs Lefthand Tensor to matmul
 * @param lhs_dim0 The first dim of lhs for the matmul
 * @param lhs_dim1 The second dim of lhs for the matmul
 * @param lhs_coordinates Base coordinates if we don't want to use 0
 * @param rhs Righthand Tensor to matmul
 * @param rhs_dim0 The first dim of the rhs for the matmul
 * @param rhs_dim1 The second dim of the rhs for the matmul
 * @param rhs_coordinates Base coordinates if we don't want to use 0
 * @param destination Tensor reference to write the result to
 */
template <typename T> 
requires std::is_arithmetic_v<T>
inline Tensor<T>& matmul(const AbstractTensor<T>& lhs, size_t lhs_dim0, size_t lhs_dim1, std::vector<size_t>& lhs_coordinates,
                         const AbstractTensor<T>& rhs, size_t rhs_dim0, size_t rhs_dim1, std::vector<size_t>& rhs_coordinates,
                         Tensor<T>& destination);

/**
 * Perform a matmul across two Tensors, with their target dims specified
 * @param lhs Lefthand Tensor to matmul
 * @param lhs_dim0 The first dim of lhs for the matmul
 * @param lhs_dim1 The second dim of lhs for the matmul
 * @param lhs_coordinates Base coordinates if we don't want to use 0
 * @param rhs Righthand Tensor to matmul
 * @param rhs_dim0 The first dim of the rhs for the matmul
 * @param rhs_dim1 The second dim of the rhs for the matmul
 * @param rhs_coordinates Base coordinates if we don't want to use 0
 */
template <typename T> 
requires std::is_arithmetic_v<T>
inline Tensor<T> matmul(const AbstractTensor<T>& lhs, size_t lhs_dim0, size_t lhs_dim1, std::vector<size_t>& lhs_coordinates,
                        const AbstractTensor<T>& rhs, size_t rhs_dim0, size_t rhs_dim1, std::vector<size_t>& rhs_coordinates);


/**
 * Perform a matmul across two Tensors, with their target dims specified
 * @param lhs Lefthand Tensor to matmul
 * @param lhs_dim0 The first dim of lhs for the matmul
 * @param lhs_dim1 The second dim of lhs for the matmul
 * @param rhs Righthand Tensor to matmul
 * @param rhs_dim0 The first dim of the rhs for the matmul
 * @param rhs_dim1 The second dim of the rhs for the matmul
 * @param destination Tensor reference to write the result to
 */
template <typename T> 
requires std::is_arithmetic_v<T>
inline Tensor<T>& matmul(const AbstractTensor<T>& lhs, size_t lhs_dim0, size_t lhs_dim1,
                         const AbstractTensor<T>& rhs, size_t rhs_dim0, size_t rhs_dim1,
                         Tensor<T>& destination);

/**
 * Perform a matmul across two Tensors, with their target dims specified
 * @param lhs Lefthand Tensor to matmul
 * @param lhs_dim0 The first dim of lhs for the matmul
 * @param lhs_dim1 The second dim of lhs for the matmul
 * @param rhs Righthand Tensor to matmul
 * @param rhs_dim0 The first dim of the rhs for the matmul
 * @param rhs_dim1 The second dim of the rhs for the matmul
 */
template <typename T> 
requires std::is_arithmetic_v<T>
inline Tensor<T> matmul(const AbstractTensor<T>& lhs, size_t lhs_dim0, size_t lhs_dim1,
                        const AbstractTensor<T>& rhs, size_t rhs_dim0, size_t rhs_dim1);


/**
 * Perform a matmul across two Tensors, with their target dims specified
 * @param lhs Lefthand Tensor to matmul
 * @param rhs Righthand Tensor to matmul
 * @param destination Tensor reference to write the result to
 */
template <typename T> 
requires std::is_arithmetic_v<T>
inline Tensor<T>& matmul(const AbstractTensor<T>& lhs, const AbstractTensor<T>& rhs, Tensor<T>& destination);

/**
 * Perform a matmul across two Tensors, with their target dims specified
 * @param lhs Lefthand Tensor to matmul
 * @param rhs Righthand Tensor to matmul
 */
template <typename T> 
requires std::is_arithmetic_v<T>
inline Tensor<T> matmul(const AbstractTensor<T>& lhs, const AbstractTensor<T>& rhs);

/**
 * Naive matmul implementation
 * @param lhs Lefthand Tensor to matmul
 * @param lhs_dim0 The first dim of lhs for the matmul
 * @param lhs_dim1 The second dim of lhs for the matmul
 * @param lhs_coordinates Reference to a coordinate vector to handle dims of high-rank Tensors
 * @param rhs Righthand Tensor to matmul
 * @param rhs_dim0 The first dim of the rhs for the matmul
 * @param rhs_dim1 The second dim of the rhs for the matmul
 * @param rhs_coordinates Reference to a coordinate vector to handle dims of high-rank Tensors
 * @param destination Reference to the Tensor to put the output into
 * NOTE: We assume this will never be called directly so we skip the additional
 * safety checks you would otherwise need to do (handled in Tensor::matmul, etc.)
 */
template <typename T> 
requires std::is_arithmetic_v<T>
inline void _naive_matmul_impl(const AbstractTensor<T>& lhs, size_t lhs_dim0, size_t lhs_dim1, std::vector<size_t>& lhs_coordinates,
                               const AbstractTensor<T>& rhs, size_t rhs_dim0, size_t rhs_dim1, std::vector<size_t>& rhs_coordinates,
                               Tensor<T>& destination);
/**
 * Naive matmul implementation for Tensors of rank == 2 only
 * @param lhs Lefthand Tensor to matmul
 * @param rhs Righthand Tensor to matmul
 * @param destination Reference to the Tensor to put the output into
 * NOTE: We assume this will never be called directly so we skip the additional
 * safety checks you would otherwise need to do (handled in Tensor::matmul, etc.)
 */
template <typename T> 
requires std::is_arithmetic_v<T>
inline void _naive_matmul_impl(const AbstractTensor<T>& lhs, const AbstractTensor<T>& rhs, Tensor<T>& destination);
// NOLINTEND(bugprone-easily-swappable-parameters)

/**
 * Template for a generic Tensor, where T can be any numeric type, like a variety of int,
 * double, float, etc.
 */
template <typename T> 
requires std::is_arithmetic_v<T>
class Tensor : public AbstractTensor<T> {
// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays, cppcoreguidelines-pro-bounds-pointer-arithmetic)
/* Private data elements */
private:
    /**
     * Smart pointer representing the memory block allocated to the Tensor. We use a smart pointer here
     * to aid in lifecycle management
     */
    std::unique_ptr<T[]> m_data = nullptr;

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

/* Private functions */
    /**
     * Calculate the offset from a set of input cooridnates
     * @param c Coordinates to calculate the offset from
     * @returns Returns a size_t representing the offset we want
     */
    [[nodiscard]] std::expected<size_t, std::string> _get_offset(std::initializer_list<size_t> c) const {
        // Do some pointer arithmetic to calculate the exact address to retrieve from m_data
        size_t target_offset = 0;
        // Ensure the coordinates are valid for our Tensor
        for (size_t i = 0; i < c_rank; ++i) {
            if (c.begin()[i] >= m_dims.at(i)) {
                return std::unexpected(std::format("Tensor._get_offset: Index {} exceeds dim ({}). Max possible index is {}. {}.", 
                                                    c.begin()[i], m_dims.at(i), m_dims.at(i) - 1, info()));
            }
            target_offset += c.begin()[i] * m_stride.at(i);
        }
        return target_offset;
    }

    /**
     * Calculate the offset from a set of input cooridnates
     * @param c Coordinates (wrapped in std::vector) to calculate the offset from
     * @returns Returns a size_t representing the offset we want
     */
    [[nodiscard]] std::expected<size_t, std::string> _get_offset(const std::vector<size_t>& c) const {
        // Do some pointer arithmetic to calculate the exact address to retrieve from m_data
        size_t target_offset = 0;
        // Ensure the coordinates are valid for our Tensor
        for (size_t i = 0; i < c_rank; ++i) {
            if (c.at(i) >= m_dims.at(i)) {
                return std::unexpected(std::format("Tensor._get_offset: Index {} exceeds dim ({}). Max possible index is {}. {}.", 
                                                    c.at(i), m_dims.at(i), m_dims.at(i) - 1, info()));
            }
            target_offset += c.at(i) * m_stride.at(i);
        }
        return target_offset;
    }

    /**
     * Determine if another Tensor has a shape compatible with this one for other operations
     * @param target The other Tensor to check against
     * @returns Returns true if the dims are compatible, false otherwise
     */
    [[nodiscard]] bool _compatible(const Tensor<T>& target) const {
        // If Tensor rank isn't the same, we quickly know the Tensors aren't compatible
        if (c_rank != target.c_rank) {
            return false;
        }
        // If the Tensors don't have the same number of elements, we also can easily
        // flag them as incompatible
        if (c_elements != target.c_elements) {
            return false;
        }
        // Iterate over the dimensions and validate each matches
        for (size_t i = 0; i < c_rank; ++i) {
            if (m_dims.at(i) != target.m_dims.at(i)) {
                return false;
            }
        }

        return true;
    }

/* Public functions */
public:
    /**
     * Use _can_overflow and _can_matmul from the AbstractTensor base class
     */
    using AbstractTensor<T>::_can_overflow;
    using AbstractTensor<T>::_can_matmul;

    /**
     * Default constructor for Tensor, taking in a reference to a vector containing the desired dimensions
     * for the resulting Tensor
     * @param dims std::initializer_list<size_t> containing the desired dimensions
     */
    Tensor(std::initializer_list<size_t> dims) : c_rank(dims.size()), m_stride(dims.size()), m_dims(dims) {
        // Handle the case where we have a rank-0 tensor (scalar value). Allocate space for the singular
        // element and then return immediately
        if (c_rank == 0) {
            m_data = std::make_unique<T[]>(1);
            std::memset(m_data.get(), 0, sizeof(T));
            return;
        }
        // Store the target 1-D representation of the desired size of our Tensor
        for (const size_t& ds : dims) {
            if (ds <= 1) {
                throw std::invalid_argument(std::format("Tensor.Tensor: invalid dim ({}) provided to Tensor. Dims must be >= 1 or should be omitted.", ds));
            }
            c_elements *= ds;
        }
        // Allocate the block of memory for the Tensor
        m_data = std::make_unique<T[]>(c_elements);
        std::memset(m_data.get(), 0, sizeof(T) * c_elements);
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
            for (size_t i = c_rank - 2; i >= 0; --i) {
                m_stride.at(i) = m_stride.at(i + 1) * m_dims.at(i);
            }
        }
    }

    /**
     * Copy constructor for Tensor, creating a deep copy
     * @param target Tensor to make a copy of
     */
    Tensor(const Tensor<T>& target) : c_elements(target.c_elements), c_rank(target.c_rank), m_stride(target.m_stride), m_dims(target.m_dims) {
        // Create a new unique_ptr memory block
        m_data = std::make_unique<T[]>(c_elements);
        // Copy the entire m_data block
        std::memcpy(m_data.get(), target.m_data.get(), sizeof(T) * c_elements);
        // Debug logging
        if (TENSOR_ENABLE_CONSTRUCTOR_LOGGING) {
            log_message(Log_Priority::DEBUG, "Tensor.Tensor", "Copy constructor called");
        }
    }

    /**
     * Copy assignment operator, creating a deep copy and overwriting this one
     * @param target Tensor to make a copy of
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& operator=(const Tensor<T>& target) {
        // Ensure we aren't calling the assignment operator on ourself
        if (this == &target) {
            return *this;
        }
        // Copy data elements into the Tensor
        c_elements = target.c_elements;
        c_rank = target.c_rank;
        m_stride = std::vector<size_t>(target.m_stride);
        m_dims = std::vector<size_t>(target.m_dims);
        // Allocate a new block of memory for the data. Since we are using std::unique_ptr, this should cause any
        // existing pointer to go out of scope and be cleaned up automatically
        m_data = std::make_unique<T[]>(c_elements);
        std::memcpy(m_data.get(), target.m_data.get(), sizeof(T) * c_elements);
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
    Tensor(Tensor<T>&& target) noexcept : m_data(std::move(target.m_data)), c_elements(target.c_elements), c_rank(target.c_rank), 
                                          m_stride(std::move(target.m_stride)), m_dims(std::move(target.m_dims)) {
        // Left blank since using the move constructors for std::vector and std::unique_ptr handle the setup
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
        c_elements = target.c_elements;
        c_rank = target.c_rank;
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
     * Addition assignment operator, adding values from target to this Tensor
     * @param target Tensor to add with
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& operator+=(const Tensor<T>& target) {
        // Check to see if our Tensor shapes are compatible
        if (!_compatible(target)) {
            throw std::invalid_argument("Tensor.+=: Incompatible Tensor shapes provided to +=.\n");
        }
        // Create variables for using _add_overflow and avoid calling get() multiple times
        T lhs = 0, rhs = 0, result = 0;
        for (size_t i = 0; i < c_elements; ++i) {
            lhs = m_data.get()[i];
            rhs = target.m_data.get()[i];
            if constexpr (_can_overflow) {
                if (_add_overflow(lhs, rhs, &result)) {
                    throw std::overflow_error(std::format("Tensor.+=: adding {} and {} results in overflow / underflow.\n", lhs, rhs));
                }
                else {
                    m_data.get()[i] = result;
                }
            }
            else {
                m_data.get()[i] += rhs;
            }
        }
        return *this;
    }

    /**
     * Scalar addition assignment operator, adding a scalar value to every value in the Tensor
     * @param s Scalar to add to each value in the Tensor
     * @returns Reterns a reference to this Tensor
     */
    Tensor<T>& operator+=(const T& s) {
        // Create variables for using _add_overflow and avoid calling get() multiple times
        T lhs = 0, result = 0;
        for (size_t i = 0; i < c_elements; ++i) {
            lhs = m_data.get()[i];
            if constexpr (_can_overflow) {
                if (_add_overflow(lhs, s, &result)) {
                    throw std::overflow_error(std::format("Tensor.+=: adding {} and {} results in overflow / underflow.\n", lhs, s));
                }
                else {
                    m_data.get()[i] = result;
                }
            }
            else {
                m_data.get()[i] += s;
            }
        }
        return *this;
    }

    /**
     * Addition operator, checking dimension compability then creating a new Tensor
     * containing the result
     * @param lhs Const ref to Tensor<T>
     * @param rhs Const ref to Tensor<T>
     * @returns Returns a new Tensor<T> containing the result of the addition
     */
    friend Tensor<T> operator+(const Tensor<T>& lhs, const Tensor<T>& rhs) {
        if (!lhs._compatible(rhs)) {
            throw std::invalid_argument("Tensor.+: Incompatible Tensor shapes provided to +.\n");
        }
        Tensor<T> result = Tensor<T>(lhs);
        result += rhs;
        return result;
    }

    /**
     * Scalar addition operator for adding scalar values, storing the result in a new Tensor
     * @param lhs Const ref to the Tensor<T>
     * @param rhs Const ref to the scalar value we want to use
     * @returns Returns a new Tensor<T> containing the result of the addition
     */
    friend Tensor<T> operator+(const Tensor<T>& lhs, const T& rhs) {
        Tensor<T> result = Tensor<T>(lhs);
        result += rhs;
        return result;
    }

    /**
     * Subtraction assignment operator, subtracting values from target to this Tensor
     * @param target Tensor to subtract with
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& operator-=(const Tensor<T>& target) {
        // Check to see if our Tensor shapes are compatible
        if (!_compatible(target)) {
            throw std::invalid_argument("Tensor.-=: Incompatible Tensor shapes provided to -=.\n");
        }
        T lhs = 0, rhs = 0, result = 0;
        for (size_t i = 0; i < c_elements; ++i) {
            lhs = m_data.get()[i];
            rhs = target.m_data.get()[i];
            if constexpr (_can_overflow) {
                if (_sub_overflow(lhs, rhs, &result)) {
                    throw std::overflow_error(std::format("Tensor.-=: Subtracting {} and {} results in overflow / underflow.\n", lhs, rhs));
                }
                else {
                    m_data.get()[i] = result;
                }
            }
            else {
                m_data.get()[i] -= rhs;
            }
        }
        return *this;
    }

    /**
     * Scalar subtraction assignment operator, subtracting a scalar value to every value in the Tensor
     * @param s Scalar to subtract to each value in the Tensor
     * @returns Reterns a reference to this Tensor
     */
    Tensor<T>& operator-=(const T& s) {
        // Create variables for using _sub_overflow and avoid calling get() multiple times
        T lhs = 0, result = 0;
        for (size_t i = 0; i < c_elements; ++i) {
            lhs = m_data.get()[i];
            if constexpr (_can_overflow) {
                if (_sub_overflow(lhs, s, &result)) {
                    throw std::overflow_error(std::format("Tensor.-=: Subtracting {} and {} results in overflow / underflow.\n", lhs, s));
                }
                else {
                    m_data.get()[i] = result;
                }
            }
            else {
                m_data.get()[i] -= s;
            }
        }
        return *this;
    }

    /**
     * Subtraction operator, checking dimension compability then creating a new Tensor
     * containing the result
     * @param lhs Const ref to Tensor<T>
     * @param rhs Const ref to Tensor<T>
     * @returns Returns a new Tensor<T> containing the result of the subtraction
     */
    friend Tensor<T> operator-(const Tensor<T>& lhs, const Tensor<T>& rhs) {
        if (!lhs._compatible(rhs)) {
            throw std::invalid_argument("Tensor.-: Incompatible Tensor shapes provided to -.\n");
        }
        Tensor<T> result = Tensor<T>(lhs);
        result -= rhs;
        return result;
    }

    /**
     * Scalar subtraction operator for subtracting scalar values, storing the result in a new Tensor
     * @param lhs Const ref to the Tensor<T>
     * @param rhs Const ref to the scalar value we want to use
     * @returns Returns a new Tensor<T> containing the result of the subtraction
     */
    friend Tensor<T> operator-(const Tensor<T>& lhs, const T& rhs) {
        Tensor<T> result = Tensor<T>(lhs);
        result -= rhs;
        return result;
    }

    /**
     * Multiplication assignment operator, multiplying values from target with this Tensor
     * @param target Tensor to multiply with
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& operator*=(const Tensor<T>& target) {
        // Check to see if our Tensor shapes are compatible
        if (!_compatible(target)) {
            throw std::invalid_argument("Tensor.*=: Incompatible Tensor shapes provided to *=.\n");
        }
        T lhs = 0, rhs = 0, result = 0;
        for (size_t i = 0; i < c_elements; ++i) {
            lhs = m_data.get()[i];
            rhs = target.m_data.get()[i];
            if constexpr (_can_overflow) {
                if (_mul_overflow(lhs, rhs, &result)) {
                    throw std::overflow_error(std::format("Tensor.*=: Multiplying {} and {} results in overflow / underflow.\n", lhs, rhs));
                }
                else {
                    m_data.get()[i] = result;
                }
            }
            else {
                m_data.get()[i] *= rhs;
            }
        }
        return *this;
    }

    /**
     * Scalar multiplication assignment operator, multipling a scalar value with every value in the Tensor
     * @param s Scalar to multiply with each value in the Tensor
     * @returns Reterns a reference to this Tensor
     */
    Tensor<T>& operator*=(const T& s) {
        // Create variables for using _mul_overflow and avoid calling get() multiple times
        T lhs = 0, result = 0;
        for (size_t i = 0; i < c_elements; ++i) {
            lhs = m_data.get()[i];
            if constexpr (_can_overflow) {
                if (_mul_overflow(lhs, s, &result)) {
                    throw std::overflow_error(std::format("Tensor.*=: Multiplying {} and {} results in overflow / underflow.\n", lhs, s));
                }
                else {
                    m_data.get()[i] = result;
                }
            }
            else {
                m_data.get()[i] *= s;
            }
        }
        return *this;
    }

    /**
     * Multiplication operator, checking dimension compability then creating a new Tensor
     * containing the result
     * @param lhs Const ref to Tensor<T>
     * @param rhs Const ref to Tensor<T>
     * @returns Returns a new Tensor<T> containing the result of the multiplication
     */
    friend Tensor<T> operator*(const Tensor<T>& lhs, const Tensor<T>& rhs) {
        if (!lhs._compatible(rhs)) {
            throw std::invalid_argument("Tensor.*: Incompatible Tensor shapes provided to *.\n");
        }
        Tensor<T> result = Tensor<T>(lhs);
        result *= rhs;
        return result;
    }

    /**
     * Scalar multiplication operator for multiplying scalar values, storing the result in a new Tensor
     * @param lhs Const ref to the Tensor<T>
     * @param rhs Const ref to the scalar value we want to use
     * @returns Returns a new Tensor<T> containing the result of the multiplication
     */
    friend Tensor<T> operator*(const Tensor<T>& lhs, const T& rhs) {
        Tensor<T> result = Tensor<T>(lhs);
        result *= rhs;
        return result;
    }

    /**
     * Division assignment operator, dividing values from target with this Tensor
     * @param target Tensor to divide with
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& operator/=(const Tensor<T>& target) {
        // Check to see if our Tensor shapes are compatible
        if (!_compatible(target)) {
            throw std::invalid_argument("Tensor./=: Incompatible Tensor shapes provided to /=.\n");
        }
        // Do a special check to avoid corner case of dividing INT_MIN / -1
        if constexpr (std::is_same_v<T, int>) {
            T lhs = 0, rhs = 0;
            for (size_t i = 0; i < c_elements; ++i) {
                lhs = m_data.get()[i];
                rhs = target.m_data.get()[i];
                if (lhs == INT_MIN && rhs == -1) {
                    throw std::overflow_error(std::format("Tensor./=: Dividing {} by {} results in overflow / underflow.\n", lhs, rhs));
                }
                else if (rhs == 0) {
                    throw std::invalid_argument("Tensor./=: Dividing by 0 is not supported\n");
                }
                m_data.get()[i] /= rhs;
            }
        }
        else {
            for (size_t i = 0; i < c_elements; ++i) {
                T rhs = target.m_data.get()[i];
                // Ensure we aren't dividing by zero
                if (rhs == 0) {
                    throw std::invalid_argument("Tensor./=: Dividing by 0 is not supported\n");
                }
                m_data.get()[i] /= rhs;
            }
        }
        return *this;
    }

    /**
     * Scalar division assignment operator, dividing every value in the Tensor by the scalar
     * @param s Scalar to divide each value in the Tensor by
     * @returns Reterns a reference to this Tensor
     */
    Tensor<T>& operator/=(const T& s) {
        // Only check that s != 0
        if (s == 0) {
            throw std::invalid_argument("Tensor./=: Dividing by 0 is not supported\n");
        }
        for (size_t i = 0; i < c_elements; ++i) {
            m_data.get()[i] /= s;
        }
        return *this;
    }

    /**
     * Division operator, checking dimension compability then creating a new Tensor
     * containing the result
     * @param lhs Const ref to Tensor<T>
     * @param rhs Const ref to Tensor<T>
     * @returns Returns a new Tensor<T> containing the result of the division
     */
    friend Tensor<T> operator/(const Tensor<T>& lhs, const Tensor<T>& rhs) {
        if (!lhs._compatible(rhs)) {
            throw std::invalid_argument("Tensor./: Incompatible Tensor shapes provided to /.\n");
        }
        Tensor<T> result = Tensor<T>(lhs);
        result /= rhs;
        return result;
    }

    /**
     * Scalar division operator for dividing by scalar values, storing the result in a new Tensor
     * @param lhs Const ref to the Tensor<T>
     * @param rhs Const ref to the scalar value we want to use
     * @returns Returns a new Tensor<T> containing the result of the division
     */
    friend Tensor<T> operator/(const Tensor<T>& lhs, const T& rhs) {
        Tensor<T> result = Tensor<T>(lhs);
        result /= rhs;
        return result;
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
        return m_stride.at(dim);
    }

    /**
     * Get or set a value at a specific coordinate inside the Tensor
     * @param target The coordinate (wrapped in std::vector) we want to fetch from the Tensor
     * @returns Returns a reference to the value that can be updated
     */
    T& at(const std::vector<size_t>& target) override {
        // Handle the case where we have a rank-0 tensor
        if (c_rank == 0) {
            return m_data.get()[0];
        }
        // Validate we have the correct number of elements in our coordinates
        if (target.size() != c_rank) {
            throw std::invalid_argument(std::format("Tensor.at: Invalid number of coordinates provided to at, expected {} but got {}.\n", c_rank, target.size()));
        }
        // Calculate the offset we want, or get the error back
        auto target_offset = _get_offset(target);
        if (target_offset) {
            return m_data.get()[target_offset.value()];
        }
        throw std::out_of_range(std::format("Tensor.at: Out of range: {}", target_offset.error()));
    }

    /**
     * Get or set a value at a specific coordinate inside the Tensor
     * @param target The coordinate we want to fetch from the Tensor
     * @returns Returns a reference to the value that can be updated
     */
    T& at(std::initializer_list<size_t> target) override {
        // Handle the case where we have a rank-0 tensor
        if (c_rank == 0) {
            return m_data.get()[0];
        }
        // Validate we have the correct number of elements in our coordinates
        if (target.size() != c_rank) {
            throw std::invalid_argument(std::format("Tensor.at: Invalid number of coordinates provided to at, expected {} but got {}.\n", c_rank, target.size()));
        }
        // Calculate the offset we want, or get the error back
        auto target_offset = _get_offset(target);
        if (target_offset) {
            return m_data.get()[target_offset.value()];
        }
        throw std::out_of_range(std::format("Tensor.at: Out of range: {}", target_offset.error()));
    }

    /**
     * Get a value at a specific coordinate inside the Tensor
     * @param target The coordinate (wrapped in std::vector) we want to fetch from the tensor
     * @returns Returns the value at the coordinate
     */
    const T& at(const std::vector<size_t>& target) const override {
        // Handle the case where we have a rank-0 tensor
        if (c_rank == 0) {
            return m_data.get()[0];
        }
        // Validate we have the correct number of elements in our coordinates
        if (target.size() != c_rank) {
            throw std::invalid_argument(std::format("Tensor.at: Invalid number of coordinates provided to at, expected {} but got {}.\n", c_rank, target.size()));
        }
        // Calculate the offset we want, or get the error back
        auto target_offset = _get_offset(target);
        if (target_offset) {
            return m_data.get()[target_offset.value()];
        }
        throw std::out_of_range(std::format("Tensor.at: Out of range: {}", target_offset.error()));
    }

    /**
     * Get a value at a specific coordinate inside the Tensor
     * @param target The coordinate we want to fetch from the tensor
     * @returns Returns the value at the coordinate
     */
    const T& at(std::initializer_list<size_t> target) const override {
        // Handle the case where we have a rank-0 tensor
        if (c_rank == 0) {
            return m_data.get()[0];
        }
        // Validate we have the correct number of elements in our coordinates
        if (target.size() != c_rank) {
            throw std::invalid_argument(std::format("Tensor.at: Invalid number of coordinates provided to at, expected {} but got {}.\n", c_rank, target.size()));
        }
        // Calculate the offset we want, or get the error back
        auto target_offset = _get_offset(target);
        if (target_offset) {
            return m_data.get()[target_offset.value()];
        }
        throw std::out_of_range(std::format("Tensor.at: Out of range: {}", target_offset.error()));
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
        return m_data.get()[target];
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
        return m_data.get()[target];
    }

    /**
     * Fill a Tensor with a value
     * @param v Value to fill the Tensor with
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& fill(const T& v) {
        // Set all elements of the Tensor to v
        for (size_t i = 0; i < c_elements; ++i) {
            m_data.get()[i] = v;
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
            std::invalid_argument(std::format("Tensor.set: Invalid number of value provided to set. Wanted {} but got {}\n.", c_elements, v.size()));
        }
        // Assuming we have enough values, read them into memory sequentially
        for (size_t i = 0; i < v.size(); ++i) {
            m_data.get()[i] = v.begin()[i];
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
            for (size_t i = 0; i < c_elements; ++i) {
                m_data.get()[i] = (*f)(m_data.get()[i]);
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
            for (size_t i = 0; i < c_elements; ++i) {
                m_data.get()[i] = (*f)(m_data.get()[i], param);
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

        // std::uniform_real_distribution only applies to float-like types, but we can use the _can_overflow
        // to check for int-like types and use std::uniform_int_distribution instead
        if constexpr (std::integral<T>) {
            std::uniform_int_distribution<T> distrib(range_min, range_max);
            for (size_t i = 0; i < c_elements; ++i) {
                m_data.get()[i] = distrib(gen);
            }
        }
        else if constexpr (std::floating_point<T>) {
            std::uniform_real_distribution<T> distrib(range_min, range_max);
            for (size_t i = 0; i < c_elements; ++i) {
                m_data.get()[i] = distrib(gen);
            }
        }
        else {
            throw std::domain_error("Tensor.random: Invalid type for use with random");
        }
        return *this;
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
        result.erase(result.size() - 2, 2);
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
        return result;
    }

    /**
     * Calculate the dot product of two Tensors
     * @param lhs Tensor to calculate dot product with
     * @returns Returns the dot product, of type T
     */
    T dot(const Tensor<T>& lhs) const {
        // Ensure that both of our Tensors have rank == 1, otherwise dot cannot execute
        if (c_rank != 1 || lhs.c_rank != 1) {
            throw std::invalid_argument("Tensor.dot: Tensors must be rank == 1 for dot\n");
        }
        if (c_elements != lhs.c_elements) {
            throw std::invalid_argument("Tensor.dot: Tensors must have the same number of elements for dot\n");
        }
        // Check if we need to worry about overflow
        if constexpr (_can_overflow) {
            // Multiply each element and add to the sum
            T result = 0, mul_result = 0;
            for (size_t i = 0; i < c_elements; ++i) {
                if (_mul_overflow(m_data.get()[i], lhs.m_data.get()[i], &mul_result)) {
                    throw std::overflow_error("Tensor.dot: Multiplication in dot will cause overflow / underflow\n");
                }
                if (_add_overflow(result, mul_result, &result)) {
                    throw std::overflow_error("Tensor.dot: Addition of multiplication result will cause overflow / underflow\n");
                }
            }
            return result;
        }
        // If we won't need to worry about overflow, perform the op directly
        T result = 0;
        for (size_t i = 0; i < c_elements; ++i) {
            result += m_data.get()[i] * lhs.m_data.get()[i];
        }
        return result;
    }

    /**
     * Calculate the sum of a Tensor
     * @returns Returns the sum
     */
    T sum() const {
        // Check to see if we need to worry about overflow / underflow
        if constexpr (_can_overflow) {
            T result = 0;
            for (size_t i = 0; i < c_elements; ++i) {
                if (_add_overflow(result, m_data.get()[i], &result)) {
                    throw std::overflow_error("Tensor.sum: Addition will cause overflow / underflow.\n");
                }
            }
            return result;
        }
        T result = 0;
        for (size_t i = 0; i < c_elements; ++i) {
            result += m_data.get()[i];
        }
        return result;
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
        T scale_factor = static_cast<T>(1.0f / dropout);
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
        for (const auto& idx : target_idxs) {
            m_data.get()[idx] = 0;
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
        // NOLINTNEXTLINE(bugprone-sizeof-expression)
        return *(std::max_element(m_data.get(), m_data.get() + c_elements));
    }

    /** 
     * Perform a matmul on a Tensor instance with itself
     * @param transpose Whether to transpose self when performing the matmul, resulting in
     * self @ self.transpose()
     * @returns Returns a new Tensor instance containing the matmul result 
     */
    Tensor<T> matmul_self(bool transpose) const {
        // Ensure this can only run on 2-D Tensors
        if (c_rank != 2) {
            throw std::logic_error("Tensor.matmul_self: matmul_self(transpose) cannot be called on a non rank-2 Tensor.\n");
        }
        if (!transpose) {
            if (m_dims.at(0) != m_dims.at(1)) {
                throw std::invalid_argument("Tensor.matmul_self: Cannot perform matmul_self on a Tensor that is not square.\n");
            }
        }
        // Create a new Tensor and initialze its values to zero (done in the Tensor constructor)
        Tensor<T> result({rows(), rows()});
        if (transpose) {
            matmul(*this, 0, 1, *this, 1, 0);
        }
        else {
            matmul(*this, *this);
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
            if constexpr (_can_overflow) {
                for (size_t i = 0; i < cols(); ++i) {
                    if (_add_overflow(result, at({idx, i}), &result)) {
                        throw std::overflow_error("Tensor.sum: Addition will cause overflow / underflow.");
                    }
                }
            }
            else {
                for (size_t i = 0; i < cols(); ++i) {
                    result += at({idx, i});
                }
            }
        }
        else {
            if (idx >= cols()) {
                throw std::invalid_argument("Tensor.sum: Invalid index provided to sum.\n");
            }
            if constexpr (_can_overflow) {
                for (size_t i = 0; i < rows(); ++i) {
                    if (_add_overflow(result, at({i, idx}), &result)) {
                        throw std::overflow_error("Tensor.sum: Addition will cause overflow / underflow.");
                    }
                }
            }
            else {
                for (size_t i = 0; i < rows(); ++i) {
                    result += at({i, idx});
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
        T dim_max = 0;
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
                dim_max = 0;
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
                dim_max = 0;
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
                    for (size_t j = 0; j < other_dim_size; ++j) {
                        if constexpr (_can_overflow) {
                            T& val = at({i, j});
                            if (_add_overflow(val, vals.at({i}), &val)) {
                                throw std::overflow_error("Tensor.squeezed_op: Addition causes overflow / underflow.\n");
                            }
                        }
                        else {
                            at({i, j}) += vals.at({i});
                        }
                    }
                }
                break;
            case SqueezedOpType::SUB:
                for (size_t i = 0; i < target_dim_size; ++i) {
                    for (size_t j = 0; j < other_dim_size; ++j) {
                        if constexpr (_can_overflow) {
                            T& val = at({i, j});
                            if (_sub_overflow(val, vals.at({i}), &val)) {
                                throw std::overflow_error("Tensor.squeezed_op: Subtraction causes overflow / underflow.\n");
                            }
                        }
                        else {
                            at({i, j}) -= vals.at({i});
                        }
                    }
                }
                break;
            case SqueezedOpType::MUL:
                for (size_t i = 0; i < target_dim_size; ++i) {
                    for (size_t j = 0; j < other_dim_size; ++j) {
                        if constexpr (_can_overflow) {
                            T& val = at({i, j});
                            if (_mul_overflow(val, vals.at({i}), &val)) {
                                throw std::overflow_error("Tensor.squeezed_op: Multiplication causes overflow / underflow.\n");
                            }
                        }
                        else {
                            at({i, j}) *= vals.at({i});
                        }
                    }
                }
                break;
            case SqueezedOpType::DIV:
                for (size_t i = 0; i < target_dim_size; ++i) {
                    if (vals.at({i}) == 0) {
                        throw std::invalid_argument("Tensor.squeezed_op: Divide by zero detected.\n");
                    }
                    for (size_t j = 0; j < other_dim_size; ++j) {
                        at({i, j}) /= vals.at({i});
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
        squeezed_op(dim, max_vals, SqueezedOpType::SUB);
        // Apply the exp function across all elements in the Matrix
        apply(std::exp);
        // Have a fallback for row / col sums being 0, inf, or NAN
        T fallback = static_cast<T>(1.0f / this->elements());
        // Sum the values across the specified dim
        if (dim == 0) {
            for (size_t i = 0; i < rows(); ++i) {
                T row_sum = sum(dim, i);
                if (row_sum == 0 || std::isinf(row_sum) || std::isnan(row_sum)) {
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
                if constexpr (TENSOR_ENABLE_SOFTMAX_WARNINGS) {
                    log_message(Log_Priority::WARNING, "Tensor.softmax", std::format("col_sum is either 0, inf, or NAN, replacing with fallback ({}).", fallback));
                }
                if (col_sum == 0 || std::isinf(col_sum) || std::isnan(col_sum)) {
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
                for (size_t j = i; j < cols(); ++j) {
                    at({i, j}) = static_cast<T>(1);
                }
            }
        }
        else {
            for (size_t i = 0; i < rows(); ++i) {
                for (size_t j = 0; j <= i; ++j) {
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
     * @param mask_type UPPER or LOWER (if UPPER, everything below diagonal is 0 and vice-verse)
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
                for (size_t j = 0; j <= i; ++j) {
                    at({i, j}) = 0;
                }
            }
        }
        else {
            for (size_t i = 0; i < rows(); ++i) {
                for (size_t j = i; j < cols(); ++j) {
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
                for (size_t j = i; j < cols(); ++j) {
                    at({i, j}) = ninf;
                }
            }
        }
        else {
            for (size_t i = 0; i < rows(); ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    at({i, j}) = ninf;
                }
            }
        }
        return *this;
    }

// NOLINTEND(cppcoreguidelines-avoid-c-arrays, cppcoreguidelines-pro-bounds-pointer-arithmetic)
};

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
/**
 * Perform a matmul across two Tensors, with their target dims specified
 * @param lhs Lefthand Tensor to matmul
 * @param lhs_dim0 The first dim of lhs for the matmul
 * @param lhs_dim1 The second dim of lhs for the matmul
 * @param lhs_coordinates Base coordinates if we don't want to use 0
 * @param rhs Righthand Tensor to matmul
 * @param rhs_dim0 The first dim of the rhs for the matmul
 * @param rhs_dim1 The second dim of the rhs for the matmul
 * @param rhs_coordinates Base coordinates if we don't want to use 0
 * @param destination Tensor reference to write the result to
 */
template <typename T> 
requires std::is_arithmetic_v<T>
inline Tensor<T>& matmul(const AbstractTensor<T>& lhs, size_t lhs_dim0, size_t lhs_dim1, std::vector<size_t>& lhs_coordinates,
                         const AbstractTensor<T>& rhs, size_t rhs_dim0, size_t rhs_dim1, std::vector<size_t>& rhs_coordinates,
                         Tensor<T>& destination) {
    // Use the checker function instead of writing the check manually several times
    auto compat = lhs._can_matmul(lhs_dim0, lhs_dim1, rhs, rhs_dim0, rhs_dim1);
    if (!compat.has_value()) {
        throw std::invalid_argument(std::format("Tensor::matmul: Unable to matmul. Error: {}.", compat.error()));
    }
    // Ensure lhs_coordinates and rhs_coordinates have the correct size
    if (lhs_coordinates.size() != lhs.rank() || rhs_coordinates.size() != rhs.rank()) {
        throw std::invalid_argument("Tensor::matmul: Invalid coordinate vector(s) provided.\n");
    }
    // Ensure the destination Tensor has the right dimensions
    if (destination.extent(0) != lhs.extent(lhs_dim0) || destination.extent(1) != rhs.extent(rhs_dim1)) {
        throw std::invalid_argument(
            std::format(
                "Tensor::matmul: Destination Tensor has shape [{}, {}] but should be [{}, {}].\n",
                destination.extent(0), destination.extent(1), lhs.extent(lhs_dim0), rhs.extent(rhs_dim1)
            )
        );
    }
    // Zero out the content of the destination
    destination.fill(0);
    // Use the desired matmul impl to execute the operation
    _naive_matmul_impl(lhs, lhs_dim0, lhs_dim1, lhs_coordinates, rhs, rhs_dim0, rhs_dim1, rhs_coordinates, destination);
    return destination;
}

/**
 * Perform a matmul across two Tensors, with their target dims specified
 * @param lhs Lefthand Tensor to matmul
 * @param lhs_dim0 The first dim of lhs for the matmul
 * @param lhs_dim1 The second dim of lhs for the matmul
 * @param lhs_coordinates Base coordinates if we don't want to use 0
 * @param rhs Righthand Tensor to matmul
 * @param rhs_dim0 The first dim of the rhs for the matmul
 * @param rhs_dim1 The second dim of the rhs for the matmul
 * @param rhs_coordinates Base coordinates if we don't want to use 0
 */
template <typename T> 
requires std::is_arithmetic_v<T>
inline Tensor<T> matmul(const AbstractTensor<T>& lhs, size_t lhs_dim0, size_t lhs_dim1, std::vector<size_t>& lhs_coordinates,
                        const AbstractTensor<T>& rhs, size_t rhs_dim0, size_t rhs_dim1, std::vector<size_t>& rhs_coordinates) {
    // Use the checker function instead of writing the check manually several times
    auto compat = lhs._can_matmul(lhs_dim0, lhs_dim1, rhs, rhs_dim0, rhs_dim1);
    if (!compat.has_value()) {
        throw std::invalid_argument(std::format("Tensor::matmul: Unable to matmul. Error: {}.", compat.error()));
    }
    // Ensure lhs_coordinates and rhs_coordinates have the correct size
    if (lhs_coordinates.size() != lhs.rank() || rhs_coordinates.size() != rhs.rank()) {
        throw std::invalid_argument("Tensor::matmul: Invalid coordinate vector(s) provided.\n");
    }
    // Create a Tensor to store the result
    Tensor<T> result({lhs.extent(lhs_dim0), rhs.extent(rhs_dim1)});
    // Use the desired matmul impl to execute the operation
    _naive_matmul_impl(lhs, lhs_dim0, lhs_dim1, lhs_coordinates, rhs, rhs_dim0, rhs_dim1, rhs_coordinates, result);
    // Use RVO to return the result without copying
    return result;
}

/**
 * Perform a matmul across two Tensors, with their target dims specified
 * @param lhs Lefthand Tensor to matmul
 * @param lhs_dim0 The first dim of lhs for the matmul
 * @param lhs_dim1 The second dim of lhs for the matmul
 * @param rhs Righthand Tensor to matmul
 * @param rhs_dim0 The first dim of the rhs for the matmul
 * @param rhs_dim1 The second dim of the rhs for the matmul
 * @param destination Tensor reference to write the result to
 */
template <typename T> 
requires std::is_arithmetic_v<T>
inline Tensor<T>& matmul(const AbstractTensor<T>& lhs, size_t lhs_dim0, size_t lhs_dim1,
                         const AbstractTensor<T>& rhs, size_t rhs_dim0, size_t rhs_dim1,
                         Tensor<T>& destination) {
    // Use the checker function instead of writing the check manually several times
    auto compat = lhs._can_matmul(lhs_dim0, lhs_dim1, rhs, rhs_dim0, rhs_dim1);
    if (!compat.has_value()) {
        throw std::invalid_argument(std::format("Tensor::matmul: Unable to matmul. Error: {}.", compat.error()));
    }
    // Create reusable std::vector to use with Tensor.at
    std::vector<size_t> lhs_coordinates(lhs.rank(), 0);
    std::vector<size_t> rhs_coordinates(rhs.rank(), 0);
    // Ensure the destination Tensor has the right dimensions
    if (destination.extent(0) != lhs.extent(lhs_dim0) || destination.extent(1) != rhs.extent(rhs_dim1)) {
        throw std::invalid_argument(
            std::format(
                "Tensor::matmul: Destination Tensor has shape [{}, {}] but should be [{}, {}].\n",
                destination.extent(0), destination.extent(1), lhs.extent(lhs_dim0), rhs.extent(rhs_dim1)
            )
        );
    }
    // Zero out the content of the destination
    destination.fill(0);
    // Use the desired matmul impl to execute the operation
    _naive_matmul_impl(lhs, lhs_dim0, lhs_dim1, lhs_coordinates, rhs, rhs_dim0, rhs_dim1, rhs_coordinates, destination);
    return destination;
}

/**
 * Perform a matmul across two Tensors, with their target dims specified
 * @param lhs Lefthand Tensor to matmul
 * @param lhs_dim0 The first dim of lhs for the matmul
 * @param lhs_dim1 The second dim of lhs for the matmul
 * @param rhs Righthand Tensor to matmul
 * @param rhs_dim0 The first dim of the rhs for the matmul
 * @param rhs_dim1 The second dim of the rhs for the matmul
 */
template <typename T> 
requires std::is_arithmetic_v<T>
inline Tensor<T> matmul(const AbstractTensor<T>& lhs, size_t lhs_dim0, size_t lhs_dim1,
                        const AbstractTensor<T>& rhs, size_t rhs_dim0, size_t rhs_dim1) {
    // Use the checker function instead of writing the check manually several times
    auto compat = lhs._can_matmul(lhs_dim0, lhs_dim1, rhs, rhs_dim0, rhs_dim1);
    if (!compat.has_value()) {
        throw std::invalid_argument(std::format("Tensor::matmul: Unable to matmul. Error: {}.", compat.error()));
    }
    // Create reusable std::vector to use with Tensor.at
    std::vector<size_t> lhs_coordinates(lhs.rank(), 0);
    std::vector<size_t> rhs_coordinates(rhs.rank(), 0);
    // Create a Tensor to store the result
    Tensor<T> result({lhs.extent(lhs_dim0), rhs.extent(rhs_dim1)});
    // Use the desired matmul impl to execute the operation
    _naive_matmul_impl(lhs, lhs_dim0, lhs_dim1, lhs_coordinates, rhs, rhs_dim0, rhs_dim1, rhs_coordinates, result);
    // Use RVO to return the result without copying
    return result;
}

/**
 * Perform a matmul across two Tensors, with their target dims specified
 * @param lhs Lefthand Tensor to matmul
 * @param rhs Righthand Tensor to matmul
 * @param destination Tensor reference to write the result to
 */
template <typename T> 
requires std::is_arithmetic_v<T>
inline Tensor<T> matmul(const AbstractTensor<T>& lhs, const AbstractTensor<T>& rhs, Tensor<T>& destination) {
    // Assume we want dims 0 and 1 from lhs and rhs and that their rank must == 2
    if (lhs.rank() != 2 || rhs.rank() != 2) {
        throw std::invalid_argument(
            std::format(
                "Tensor::matmul: Invalid Tensor rank for matmul(lhs, rhs). lhs.rank == {}. rhs.rank == {}.",
                    lhs.rank(), rhs.rank()));
    }
    auto compat = lhs._can_matmul(0, 1, rhs, 0, 1);
    if (!compat.has_value()) {
        throw std::invalid_argument(std::format("Tensor::matmul: Unable to matmul. Error: {}.", compat.error()));
    }
    // Ensure the destination Tensor has the right dimensions
    if (destination.extent(0) != lhs.extent(0) || destination.extent(1) != rhs.extent(1)) {
        throw std::invalid_argument(
            std::format(
                "Tensor::matmul: Destination Tensor has shape [{}, {}] but should be [{}, {}].\n",
                destination.extent(0), destination.extent(1), lhs.extent(0), rhs.extent(1)
            )
        );
    }
    // Zero out the content of the destination
    destination.fill(0);
    // Use the desired matmul impl to execute the operation
    _naive_matmul_impl(lhs, rhs, destination);
    return destination;
}

/**
 * Perform a matmul across two Tensors, with their target dims specified
 * @param lhs Lefthand Tensor to matmul
 * @param rhs Righthand Tensor to matmul
 */
template <typename T> 
requires std::is_arithmetic_v<T>
inline Tensor<T> matmul(const AbstractTensor<T>& lhs, const AbstractTensor<T>& rhs) {
    // Assume we want dims 0 and 1 from lhs and rhs and that their rank must == 2
    if (lhs.rank() != 2 || rhs.rank() != 2) {
        throw std::invalid_argument(
            std::format(
                "Tensor::matmul: Invalid Tensor rank for matmul(lhs, rhs). lhs.rank == {}. rhs.rank == {}.",
                    lhs.rank(), rhs.rank()));
    }
    auto compat = lhs._can_matmul(0, 1, rhs, 0, 1);
    if (!compat.has_value()) {
        throw std::invalid_argument(std::format("Tensor::matmul: Unable to matmul. Error: {}.", compat.error()));
    }
    // Create a Tensor to store the result
    Tensor<T> result({lhs.extent(0), rhs.extent(1)});
    // Use the desired matmul impl to execute the operation
    _naive_matmul_impl(lhs, rhs, result);
    // Use RVO to return the result without copying
    return result;
}

/**
 * Naive matmul implementation
 * @param lhs Lefthand Tensor to matmul
 * @param lhs_dim0 The first dim of lhs for the matmul
 * @param lhs_dim1 The second dim of lhs for the matmul
 * @param lhs_coordinates Reference to a coordinate vector to handle dims of high-rank Tensors
 * @param rhs Righthand Tensor to matmul
 * @param rhs_dim0 The first dim of the rhs for the matmul
 * @param rhs_dim1 The second dim of the rhs for the matmul
 * @param rhs_coordinates Reference to a coordinate vector to handle dims of high-rank Tensors
 * @param destination Reference to the Tensor to put the output into
 * NOTE: We assume this will never be called directly so we skip the additional
 * safety checks you would otherwise need to do (handled in Tensor::matmul, etc.)
 */
template <typename T> 
requires std::is_arithmetic_v<T>
inline void _naive_matmul_impl(const AbstractTensor<T>& lhs, size_t lhs_dim0, size_t lhs_dim1, std::vector<size_t>& lhs_coordinates,
                               const AbstractTensor<T>& rhs, size_t rhs_dim0, size_t rhs_dim1, std::vector<size_t>& rhs_coordinates,
                               Tensor<T>& destination) {
    // Per the NOTE above, skip the regular safety checks and perform the operation
    if constexpr (destination._can_overflow) {
        // Create buffer for overflow / underflow checking
        T mul_result = 0;
        for (size_t i = 0; i < destination.extent(0); ++i) {
            // Update the coordinate for i in lhs
            lhs_coordinates.at(lhs_dim0) = i;
            for (size_t j = 0; j < destination.extent(1); ++j) {
                // Update the coordinate for j in rhs
                rhs_coordinates.at(rhs_dim1) = j;
                // Get a pointer to the result's [i, j]
                T* result_i_j = &(destination.at({i, j}));
                for (size_t k = 0; k < lhs.extent(lhs_dim1); ++k) {
                    // Update the coordinate for k in lhs and rhs
                    lhs_coordinates.at(lhs_dim1) = k;
                    rhs_coordinates.at(rhs_dim0) = k;
                    // First multiply [i, k] * [k, j]
                    if (_mul_overflow(lhs.at(lhs_coordinates), rhs.at(rhs_coordinates), &mul_result)) {
                        throw std::overflow_error("Tensor::_naive_matmul_impl: Multiplication results in overflow / underflow.\n");
                    }
                    if (_add_overflow(*result_i_j, rhs.at(rhs_coordinates), result_i_j)) {
                        throw std::overflow_error("Tensor::_naive_matmul_impl: Addition results in overflow / underflow.\n");
                    }
                }
            }
        }
    }
    else {
        // Use a naive loop to perform matmul
        for (size_t i = 0; i < destination.extent(0); ++i) {
            // Update the coordinate for i in lhs
            lhs_coordinates.at(lhs_dim0) = i;
            for (size_t j = 0; j < destination.extent(1); ++j) {
                // Update the coordinate for j in rhs
                rhs_coordinates.at(rhs_dim1) = j;
                for (size_t k = 0; k < lhs.extent(lhs_dim1); ++k) {
                    // Update the coordinate for k in lhs and rhs
                    lhs_coordinates.at(lhs_dim1) = k;
                    rhs_coordinates.at(rhs_dim0) = k;
                    destination.at({i, j}) += lhs.at(lhs_coordinates) * rhs.at(rhs_coordinates);
                }
            }
        }
    }
}

/**
 * Naive matmul implementation for Tensors of rank == 2 only
 * @param lhs Lefthand Tensor to matmul
 * @param rhs Righthand Tensor to matmul
 * @param destination Reference to the Tensor to put the output into
 * NOTE: We assume this will never be called directly so we skip the additional
 * safety checks you would otherwise need to do (handled in Tensor::matmul, etc.)
 */
template <typename T> 
requires std::is_arithmetic_v<T>
inline void _naive_matmul_impl(const AbstractTensor<T>& lhs, const AbstractTensor<T>& rhs, Tensor<T>& destination) {
    // Per the NOTE above, skip the regular safety checks and perform the operation
    if constexpr (destination._can_overflow) {
        // Create buffer for overflow / underflow checking
        T mul_result = 0;
        for (size_t i = 0; i < destination.extent(0); ++i) {
            for (size_t j = 0; j < destination.extent(1); ++j) {
                // Get a pointer to the result's [i, j]
                T* result_i_j = &(destination.at({i, j}));
                for (size_t k = 0; k < lhs.extent(1); ++k) {
                    // First multiply [i, k] * [k, j]
                    if (_mul_overflow(lhs.at({i, k}), rhs.at({k, j}), &mul_result)) {
                        throw std::overflow_error("Tensor::_naive_matmul_impl: Multiplication results in overflow / underflow.\n");
                    }
                    if (_add_overflow(*result_i_j, rhs.at({k, j}), result_i_j)) {
                        throw std::overflow_error("Tensor::_naive_matmul_impl: Addition results in overflow / underflow.\n");
                    }
                }
            }
        }
    }
    else {
        // Use a naive loop to perform matmul
        for (size_t i = 0; i < destination.extent(0); ++i) {
            for (size_t j = 0; j < destination.extent(1); ++j) {
                for (size_t k = 0; k < lhs.extent(1); ++k) {
                    destination.at({i, j}) += lhs.at({i, k}) * rhs.at({k, j});
                }
            }
        }
    }
}
// NOLINTEND(bugprone-easily-swappable-parameters)

}; // namespace Tensor_NS

#endif
