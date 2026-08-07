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
 * @version: 2026-08-06
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
#include <mdspan>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <vector>

/* Local dependencies */
#include "Log.hpp"

namespace Tensor_NS {

// Control verbose copy / move constructor logging, useful for debugging. Does not apply to default constructors
inline constexpr bool TENSOR_ENABLE_CONSTRUCTOR_LOGGING = true;
inline constexpr bool MATRIX_ENABLE_CONSTRUCTOR_LOGGING = true;
// Store the maximum length for a type name from the demangler API
inline constexpr size_t TENSOR_MAX_DEMANGLED_NAME_LEN = 32;

// Use Logging functions
using Log::Log_Priority;
using Log::log_message;

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

/**
 * Template for a generic Tensor, where T can be any numeric type, like a variety of int,
 * double, float, etc.
 */
template <typename T> 
requires std::is_arithmetic_v<T>
class Tensor {
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

    /**
     * Determine if a numeric type can overflow and should be checked by the compiler's built
     * in checking function
     */
    static constexpr bool _can_overflow = std::is_same_v<T, char> || std::is_same_v<T, signed char> || std::is_same_v<T, int> || std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> || std::is_same_v<T, unsigned char> || std::is_same_v<T, unsigned int> || std::is_same_v<T, size_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>;

/* Private functions */
    /**
     * Calculate the offset from a set of input cooridnates
     * @param c Coordinates to calculate the offset from
     * @returns Returns a size_t representing the offset we want
     */
    [[nodiscard]] std::expected<size_t, std::string> _get_offset(const std::initializer_list<size_t>& c) const {

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

/* Protected functions */
protected:
    /**
     * Get the raw pointer to m_data for use by derived classes (like Matrix)
     * @returns Returns a pointer to m_data
     */
    [[nodiscard]] T* data() {
        return m_data.get();
    }

    /**
     * Update stides and dims to reflect a transpose operation on a Matrix instance
     */
    bool transpose_tensor() {
        // Do a sanity check to ensure Tensor rank == 2
        if (c_rank != 2) {
            return false;
        }
        std::swap(m_stride.at(0), m_stride.at(1));
        std::swap(m_dims.at(0), m_dims.at(1));
        return true;
    }

/* Public functions */
public:
    /**
     * Default constructor for Tensor, taking in a reference to a vector containing the desired dimensions
     * for the resulting Tensor
     * @param dims Const reference to std::initializer_list<size_t> containing the desired dimensions
     */
    Tensor(const std::initializer_list<size_t>& dims) : c_rank(dims.size()), m_stride(dims.size()), m_dims(dims) {
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
        m_stride.at(0) = 1;
        for (size_t i = 1; i < c_rank; ++i) {
            m_stride.at(i) = m_stride.at(i - 1) * m_dims.at(i);
        }
    }

    /**
     * Constuct a Tensor from std::mdspan, creating copy of its data and inheriting its traits
     * @param span Const reference to std::mdspan
     */
    explicit Tensor(const std::mdspan<const T, std::dextents<size_t, 1>, std::layout_stride>& span) : c_elements(span.extent(0)), c_rank(1), m_stride({1}), m_dims({1}) {
        // Store the dim information from the span
        m_dims = std::vector<size_t>{span.extent(0)};
        // Allocate the block of memory for the Tensor
        m_data = std::make_unique<T[]>(c_elements);
        std::memset(m_data.get(), 0, sizeof(T) * c_elements);
        // Copy the values out of the span into the Tensor
        for (size_t i = 0; i < c_elements; ++i) {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
            m_data.get()[i] = span[i];
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
    virtual ~Tensor() {
        // Do nothing since all of our data elements are either trivial types or will be cleaned up
        // automatically when they go out of scope
    }

    /**
     * Get the number of elements in a Tensor
     * @returns Returns size_t of the number of elements present in the Tensor
     */
    size_t elements() const {
        return c_elements;
    }

    /**
     * Get the rank of a Tensor
     * @returns Returns size_t of the Tensor's rank
     */
    virtual size_t rank() const {
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
     * Get the stride of a Tensor
     * @returns Returns a const ref to m_stride
     */
    const std::vector<size_t>& stride() const {
        return m_stride;
    }

    /**
     * Get or set a value at a specific coordinate inside the Tensor
     * @param target The coordinate (wrapped in std::vector) we want to fetch from the Tensor
     * @returns Returns a reference to the value that can be updated
     */
    T& at(const std::vector<size_t>& target) {
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
    virtual T& at(const std::initializer_list<size_t>& target) {
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
    const T& at(const std::vector<size_t>& target) const {
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
    virtual const T& at(const std::initializer_list<size_t>& target) const {
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
     * Fill a Tensor with a value
     * @param v Value to fill the Tensor with
     */
    void fill(const T& v) {
        // Set all elements of the Tensor to v
        for (size_t i = 0; i < c_elements; ++i) {
            m_data.get()[i] = v;
        }
    }

    /**
     * Set the contents of a Tensor to a list of values
     * @param v Values to set inside the Tensor
     */
    void set(const std::initializer_list<T>& v) {
        // Ensure v has enough values to satisfy the number of elements in the Tensor
        if (v.size() != c_elements) {
            std::invalid_argument(std::format("Tensor.set: Invalid number of value provided to set. Wanted {} but got {}\n.", c_elements, v.size()));
        }
        // Assuming we have enough values, read them into memory sequentially
        for (size_t i = 0; i < v.size(); ++i) {
            m_data.get()[i] = v.begin()[i];
        }
    }

    /**
     * Apply a function to all elements in a Tensor
     * @param f Function to apply
     */
    void apply(T (*f)(T)) {
        // Apply the function pointed to by f to all elements
        try {
            for (size_t i = 0; i < c_elements; ++i) {
                m_data.get()[i] = (*f)(m_data.get()[i]);
            }
        } catch (const std::exception& e) {
            throw std::invalid_argument(std::format("Tensor.apply: Exception when applying function to Tensor. Error: {}", e.what()));
        }
    }

    /**
     * Apply a function to all elements in a Tensor
     * @param f Function to apply
     * @param p Parameter to supply to function f
     */
    void apply(T (*f)(T, T), T param) {
        // Apply the function pointed to by f to all elements
        try {
            for (size_t i = 0; i < c_elements; ++i) {
                m_data.get()[i] = (*f)(m_data.get()[i], param);
            }
        } catch (const std::exception& e) {
            throw std::invalid_argument(std::format("Tensor.apply: Exception when applying function to Tensor. Error: {}", e.what()));
        }
    }

    /**
     * Slice a Tensor, getting a std::mdspan representing the desired dim
     * @param target_dim Dimension to create the std::mdspan for
     * @returns Returns a read-write std::mdspan
     */
    std::mdspan<T, std::dextents<size_t, 1>, std::layout_stride> slice(size_t target_dim) {
        // Ensure we have a valid dim
        if (target_dim >= c_rank) {
            throw std::invalid_argument(std::format("Tensor.slice: Invalid dim {} provided to slice. Max dim: {}\n", target_dim, c_rank - 1));
        }
        // Setup the strides and shapes
        std::dextents<size_t, 1> shape{m_dims.at(target_dim)};
        std::array<size_t, 1> strides{m_stride.at(target_dim)};
        // Setup the span
        return std::mdspan<T, std::dextents<size_t, 1>, std::layout_stride>{m_data.get(), std::layout_stride::mapping{shape, strides}};
    }

    /**
     * Slice a Tensor, getting a read-only std::mdspan representing the desired dim
     * @param target_dim Dimension to create the std::mdspan for
     * @returns Returns a read-only std::mdspan
     */
    std::mdspan<const T, std::dextents<size_t, 1>, std::layout_stride> slice_const(size_t target_dim) const {
        // Ensure we have a valid dim
        if (target_dim >= c_rank) {
            throw std::invalid_argument(std::format("Tensor.slice: Invalid dim {} provided to slice. Max dim: {}\n", target_dim, c_rank - 1));
        }
        // Setup the strides and shapes
        std::dextents<size_t, 1> shape{m_dims.at(target_dim)};
        std::array<size_t, 1> strides{m_stride.at(target_dim)};
        // Setup the span
        return std::mdspan<const T, std::dextents<size_t, 1>, std::layout_stride>{m_data.get(), std::layout_stride::mapping{shape, strides}};
    }

    /**
     * Fill a Tensor with random values, taken from a specified range
     * @param range_min Inclusive minimum
     * @param range_max Inclusive maximum
     */
    void random(T range_min, T range_max) {
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
    }

    /**
     * Get a string representing information about a Tensor, including its dimenstions and underlying type
     * @returns Returns an info string about the Tensor
     */
    std::string info() const {
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
     * Expose whether a type can overflow, preventing extra checks if not necessary
     */
    [[nodiscard]] constexpr bool can_overflow() const {
        return _can_overflow;
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
     * Transpose a Tensor along two specified axes
     * @param axis0 First axis to swap
     * @param axis1 Second axis to swap
     */
    void transpose(size_t axis0, size_t axis1) {
        // Ensure we have valid axes
        if (axis0 >= m_dims.size() || axis1 >= m_dims.size()) {
            throw std::invalid_argument("Tensor.transpose: Invalid axes provided to transpose.\n");
        }
        // Ensure we aren't trying to swap the same axis
        if (axis0 == axis1) {
            throw std::invalid_argument("Tensor.transpose: Cannot transpose the same axis.\n");
        }
        // Perform the swap and update the strides
        std::swap(m_stride.at(axis0), m_stride.at(axis1));
        std::swap(m_dims.at(axis0), m_dims.at(axis1));
    }

// NOLINTEND(cppcoreguidelines-avoid-c-arrays, cppcoreguidelines-pro-bounds-pointer-arithmetic)
};

/**
 * Template for a special case of Tensor with c_rank = 2, aka a Matrix. We use a derived Matrix
 * class to add Matrix-specific operations like matmul and optimize where possible
 */
template <typename T>
requires std::is_arithmetic_v<T>
class Matrix : public Tensor<T> {
// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays, cppcoreguidelines-pro-bounds-pointer-arithmetic)
/* Private data elements */
private:
    /**
     * A read-write std::mdspan for accessing the underlying Matrix data
     */
    std::mdspan<T, std::dextents<size_t, 2>, std::layout_stride> m_rw_span = std::mdspan<T, std::dextents<size_t, 2>, std::layout_stride>();

    /**
     * A read-only std::mdspan for accessing the underlying Matrix data
     */
    std::mdspan<const T, std::dextents<size_t, 2>, std::layout_stride> m_ro_span = std::mdspan<const T, std::dextents<size_t, 2>, std::layout_stride>();

    /**
     * Validate the coordinates provided
     * @param c Coordinates to validate
     * @returns Returns true if coordinates are valid, false otherwise
     */
    [[nodiscard]] bool _valid(const std::initializer_list<size_t>& c) const {

        if (c.begin()[0] >= m_ro_span.extent(0) || c.begin()[1] >= m_ro_span.extent(1)) {
            return false;
        }
        return true;
    }

    /**
     * Validates if a Matrix is an eligible target for matmul
     * @param target Other Matrix to check matmul compatibility with
     * @returns True if compatible, false otherwise
     */
    [[nodiscard]] bool _matrix_can_matmul(const Matrix<T>& target) const {

        if (m_ro_span.extent(1) != target.m_ro_span.extent(0)) {
            return false;
        }
        return true;
    }

    /**
     * Validates if a Tensor is an eligible target for matmul
     * @param target Tensor to check compatibility with
     * @returns True if compatible, false otherwise
     */
    [[nodiscard]] bool _tensor_can_matmul(const Tensor<T>& target) const {

        if (target.rank() != 2) {
            return false;
        }
        // Get the dimensions of the Tensor
        const std::vector<size_t>& tensor_dims = target.dims();
        if (m_ro_span.extent(1) != tensor_dims.at(0)) {
            return false;
        }
        return true;
    }

/* Public functions */
public:
    /**
     * Default constructor for Matrix, accepting the dimensions
     * @param dims Const reference to std::initializer_list<size_t> containing the two Matrix dims
     */
    Matrix(const std::initializer_list<size_t>& dims) : Tensor<T>(dims) {
        // Validate that we are getting exactly 2 dims
        if (dims.size() != 2) {
            throw std::invalid_argument(std::format("Matrix.Matrix: Matrix requires 2 dims, but got {}.\n", dims.size()));
        }
        // Setup the strides and shapes
        std::dextents<size_t, 2> shape{dims.begin()[0], dims.begin()[1]};
        std::array<size_t, 2> strides{shape.extent(1), 1};
        // Setup the spans
        m_rw_span = std::mdspan<T, std::dextents<size_t, 2>, std::layout_stride>{this->data(), std::layout_stride::mapping{shape, strides}};
        m_ro_span = std::mdspan<const T, std::dextents<size_t, 2>, std::layout_stride>{this->data(), std::layout_stride::mapping{shape, strides}};
    }

    /**
     * Constuctor for Matrix that converts an eligible Tensor into a Matrix
     * @param t Tensor to convert
     */
    explicit Matrix(Tensor<T>& t) {
        // Validate that the Tensor has rank 2
        if (t.rank() != 2) {
            throw std::invalid_argument(std::format("Matrix.Matrix: Matrix requires rank 2, but got a Tensor of rank {}.\n", t.rank()));
        }
        const std::vector<size_t>& dims = t.dims();
        // Setup the strides and shapes
        std::dextents<size_t, 2> shape{dims.at(0), dims.at(1)};
        std::array<size_t, 2> strides{shape.extent(1), 1};
        // Setup the std::mdspan objects to reference the Tensor's m_data
        m_rw_span = std::mdspan<T, std::dextents<size_t, 2>, std::layout_stride>{t.data(), std::layout_stride::mapping{shape, strides}};
        m_ro_span = std::mdspan<const T, std::dextents<size_t, 2>, std::layout_stride>{t.data(), std::layout_stride::mapping{shape, strides}};
    }

    /**
     * Copy constructor for Matrix, creating a deep copy of it and the underlying Tensor
     * @param target Matrix instance to copy
     */
    Matrix(const Matrix<T>& target) : Tensor<T>(target) {
        // Grab the dimensions after using the Tensor copy constructor
        const std::vector<size_t>& dims = target.dims();
        // Setup the strides and shapes
        std::dextents<size_t, 2> shape{dims.at(0), dims.at(1)};
        std::array<size_t, 2> strides{shape.extent(1), 1};
        // Setup the spans
        m_rw_span = std::mdspan<T, std::dextents<size_t, 2>, std::layout_stride>{this->data(), std::layout_stride::mapping{shape, strides}};
        m_ro_span = std::mdspan<const T, std::dextents<size_t, 2>, std::layout_stride>{this->data(), std::layout_stride::mapping{shape, strides}};
        // Debug logging
        if (MATRIX_ENABLE_CONSTRUCTOR_LOGGING) {
            log_message(Log_Priority::DEBUG, "Matrix.Matrix", "Copy constructor called");
        }
    }

    /**
     * Copy assignment operator, creating a deep copy of the Matrix, overwriting this one
     * @param target Matrix to copy from
     * @returns Returns a reference to the overwritted Matrix
     */
    Matrix<T>& operator=(const Matrix<T>& target) {
        // Ensure we aren't trying to copy ourself
        if (this == &target) {
            return *this;
        }
        // Use the Tensor copy assignment operator to setup the new instance
        Tensor<T>::operator=(target);
        // Grab the dimensions after using the Tensor copy assignment
        const std::vector<size_t>& dims = target.dims();
        // Setup the strides and shapes
        std::dextents<size_t, 2> shape{dims.at(0), dims.at(1)};
        std::array<size_t, 2> strides{shape.extent(1), 1};
        // Setup the spans
        m_rw_span = std::mdspan<T, std::dextents<size_t, 2>, std::layout_stride>{this->data(), std::layout_stride::mapping{shape, strides}};
        m_ro_span = std::mdspan<const T, std::dextents<size_t, 2>, std::layout_stride>{this->data(), std::layout_stride::mapping{shape, strides}};
        // Debug logging
        if (MATRIX_ENABLE_CONSTRUCTOR_LOGGING) {
            log_message(Log_Priority::DEBUG, "Matrix.Matrix", "Copy assignment operator called");
        }

        return *this;
    }

    /**
     * Move constructor for Matrix, moving the data from one Matrix instance to this one
     * @param target Matrix to grab the data from
     */
    Matrix(Matrix<T>&& target) noexcept : Tensor<T>(std::move(target)) {
        // While we can initialize the spans via initializers, the readability is extremely poor
        // Grab the dimensions after using the Tensor move assignment
        const std::vector<size_t>& dims = this->dims();
        // We need to disable linting for the two lines since move assignment operator is noexcept
        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        // Setup the strides and shapes
        std::dextents<size_t, 2> shape{dims[0], dims[1]};
        // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        std::array<size_t, 2> strides{shape.extent(1), 1};
        // Setup the spans
        m_rw_span = std::mdspan<T, std::dextents<size_t, 2>, std::layout_stride>{this->data(), std::layout_stride::mapping{shape, strides}};
        m_ro_span = std::mdspan<const T, std::dextents<size_t, 2>, std::layout_stride>{this->data(), std::layout_stride::mapping{shape, strides}};
        // Debug logging
        if (MATRIX_ENABLE_CONSTRUCTOR_LOGGING) {
            log_message(Log_Priority::DEBUG, "Matrix.Matrix", "Move constructor called");
        }
    }

    /**
     * Move assignment operator, taking the data from the target Matrix and overwriting this one
     * @param target Matrix to move the data from
     * @returns Returns a reference to this Matrix with the updated data
     */
    Matrix<T>& operator=(Matrix<T>&& target) noexcept {
        // Ensure we aren't performing this op on ourself
        if (this == &target) {
            return *this;
        }
        // Use Tensor's move assignment operator to get started
        Tensor<T>::operator=(std::move(target));
        // Grab the dimensions after using the Tensor move assignment
        const std::vector<size_t>& dims = target.dims();
        // We need to disable linting for the two lines since move assignment operator is noexcept
        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        // Setup the strides and shapes
        std::dextents<size_t, 2> shape{dims[0], dims[1]};
        // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        std::array<size_t, 2> strides{shape.extent(1), 1};
        // Setup the spans
        m_rw_span = std::mdspan<T, std::dextents<size_t, 2>, std::layout_stride>{this->data(), std::layout_stride::mapping{shape, strides}};
        m_ro_span = std::mdspan<const T, std::dextents<size_t, 2>, std::layout_stride>{this->data(), std::layout_stride::mapping{shape, strides}};
        // Debug logging
        if (MATRIX_ENABLE_CONSTRUCTOR_LOGGING) {
            log_message(Log_Priority::DEBUG, "Matrix.Matrix", "Move assignment operator called");
        }

        return *this;
    }

    /**
     * Addition assignment operator
     * @param target Matrix to add
     * @returns Returns a pointer to this Matrix
     */
    Matrix<T>& operator+=(const Matrix<T>& target) {
        Tensor<T>::operator+=(target);
        return *this;
    }

    /**
     * Scalar addition assignment operator
     * @param s Scalar to add
     * @returns Returns a pointer to this Matrix
     */
    Matrix<T>& operator+=(const T& s) {
        Tensor<T>::operator+=(s);
        return *this;
    }

    /**
     * Addition operator
     * @param lhs Base Matrix
     * @param rhs Matrix to add
     * @returns Returns a new Matrix containing the result
     */
    friend Matrix<T> operator+(const Matrix<T>& lhs, const Matrix<T>& rhs) {
        Matrix<T> result = lhs;
        result += rhs;
        return result;
    }

    /**
     * Scalar addition operator
     * @param lhs Base Matrix
     * @param s Scalar value to add
     * @returns Returns a new Matrix containing the result
     */
    friend Matrix<T> operator+(const Matrix<T>& lhs, const T& s) {
        Matrix<T> result = lhs;
        result += s;
        return result;
    }
    
    /**
     * Subtraction assignment operator
     * @param target Matrix to subtract
     * @returns Returns a pointer to this Matrix
     */
    Matrix<T>& operator-=(const Matrix<T>& target) {
        Tensor<T>::operator-=(target);
        return *this;
    }

    /**
     * Scalar subtraction assignment operator
     * @param s Scalar to subtract
     * @returns Returns a pointer to this Matrix
     */
    Matrix<T>& operator-=(const T& s) {
        Tensor<T>::operator-=(s);
        return *this;
    }

    /**
     * Subtraction operator
     * @param lhs Base Matrix
     * @param rhs Matrix to subtract
     * @returns Returns a pointer to a new Matrix containing the result
     */
    friend Matrix<T> operator-(const Matrix<T>& lhs, const Matrix<T>& rhs) {
        Matrix<T> result = lhs;
        result -= rhs;
        return result;
    }

    /**
     * Scalar subtraction operator
     * @param lhs Base Matrix
     * @param s Scalar value to add
     * @returns Returns a new Matrix containing the result
     */
    friend Matrix<T> operator-(const Matrix<T>& lhs, const T& s) {
        Matrix<T> result = lhs;
        result -= s;
        return result;
    }

    /**
     * Multiplication assignment operator
     * @param target Matrix to multiply by
     * @returns Returns a pointer to this Matrix
     */
    Matrix<T>& operator*=(const Matrix<T>& target) {
        Tensor<T>::operator*=(target);
        return *this;
    }

    /**
     * Scalar multiplication assignment operator
     * @param s Scalar to multiply by
     * @returns Returns a pointer to this Matrix
     */
    Matrix<T>& operator*=(const T& s) {
        Tensor<T>::operator*=(s);
        return *this;
    }

    /**
     * Multiplication operator
     * @param lhs Base Matrix
     * @param rhs Matrix to multiply by
     * @returns Returns a pointer to a new Matrix containing the result
     */
    friend Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs) {
        Matrix<T> result = lhs;
        result *= rhs;
        return result;
    }

    /**
     * Scalar multiplication operator
     * @param lhs Base Matrix
     * @param s Scalar value to multiply by
     * @returns Returns a new Matrix containing the result
     */
    friend Matrix<T> operator*(const Matrix<T>& lhs, const T& s) {
        Matrix<T> result = lhs;
        result *= s;
        return result;
    }

    /**
     * Division assignment operator
     * @param target Matrix to divide by
     * @returns Returns a pointer to this Matrix
     */
    Matrix<T>& operator/=(const Matrix<T>& target) {
        Tensor<T>::operator/=(target);
        return *this;
    }

    /**
     * Scalar division assignment operator
     * @param s Scalar to divide by
     * @returns Returns a pointer to this Matrix
     */
    Matrix<T>& operator/=(const T& s) {
        Tensor<T>::operator/=(s);
        return *this;
    }

    /**
     * Division operator
     * @param lhs Base Matrix
     * @param rhs Matrix to divide by
     * @returns Returns a pointer to a new Matrix containing the result
     */
    friend Matrix<T> operator/(const Matrix<T>& lhs, const Matrix<T>& rhs) {
        Matrix<T> result = lhs;
        result /= rhs;
        return result;
    }

    /**
     * Scalar division operator
     * @param lhs Base Matrix
     * @param s Scalar value to divide by
     * @returns Returns a new Matrix containing the result
     */
    friend Matrix<T> operator/(const Matrix<T>& lhs, const T& s) {
        Matrix<T> result = lhs;
        result /= s;
        return result;
    }

    /**
     * Destructor for Matrix
     */
    ~Matrix() override {
        // Do nothing since Matrix doesn't have any heap allocated data -- it all lives in Tensor
    }

    /**
     * Any Matrix must have rank == 2, so use constexpr
     * @returns Returns Matrix rank, 2 by definition
     */
    constexpr size_t rank() const override {
        return 2;
    }

    /**
     * Get the number of rows in a Matrix
     * @returns Returns row count
     */
    size_t rows() const {
        return m_ro_span.extent(0);
    }

    /**
     * Get the number of columns in a Matrix
     * @returns Return the column count
     */
    size_t cols() const {
        return m_ro_span.extent(1);
    }

    /**
     * Get or set a value at a specific coordinate inside the Matrix
     * @param target The coordinate we want to fetch from the Matrix
     * @returns Returns a reference to the value that can be updated
     */
    T& at(const std::initializer_list<size_t>& target) override {

        if (target.size() != 2) {
            throw std::invalid_argument(std::format("Matrix.at: Require 2 coordinates for at, got {}.\n", target.size()));
        }
        if (!_valid(target)) {
            throw std::out_of_range(std::format("Matrix.at: Invalid coordinates provided ({}, {}). {}.\n",
                                                target.begin()[0], target.begin()[1], this->info()));
        }
        // Return the value by accessing it via the read-write mdspan
        return m_rw_span[target.begin()[0], target.begin()[1]];
    }

    /**
     * Get a value at a specific coordinate inside the Matrix
     * @param target The coordinate we want to fetch from the Matrix
     * @returns Returns the value at the coordinate
     */
    const T& at(const std::initializer_list<size_t>& target) const override {
        // We use the same logic as the mutable case, just using the read-only span
        if (target.size() != 2) {
            throw std::invalid_argument(std::format("Matrix.at: Require 2 coordinates for at, got {}.\n", target.size()));
        }
        if (!_valid(target)) {
            throw std::out_of_range(std::format("Matrix.at: Invalid coordinates provided ({}, {}). {}.\n",
                                                target.begin()[0], target.begin()[1], this->info()));
        }
        // Return the value by accessing it via the read-only mdspan
        return m_ro_span[target.begin()[0], target.begin()[1]];
    }

    /** 
     * Perform a matmul on a Matrix instance, storing the results in a new Matrix
     * @param rhs Const reference to a Matrix
     * @returns Returns a new Matrix instance containing the matmul result 
     */
    Matrix<T> matmul(const Matrix<T>& rhs) const {
        if (!_matrix_can_matmul(rhs)) {
            throw std::invalid_argument(std::format("Matrix.matmul: Incompatible Matrix provided to matmul, cannot matmul [{}, {}] with [{}, {}]\n",
                                                    m_ro_span.extent(0), m_ro_span.extent(1), rhs.m_ro_span.extent(0), rhs.m_ro_span.extent(1)));
        }
        // Create a new Matrix and initialze its values to zero (done in the Tensor constructor)
        Matrix<T> result({rows(), rhs.cols()});
        // Check if we have to be concerned about overflow / underflow
        if (this->can_overflow()) {
            // Create buffer for overflow / underflow checking
            T mul_result = 0;
            for (size_t i = 0; i < result.rows(); ++i) {
                for (size_t j = 0; j < result.cols(); ++j) {
                    // Get a pointer to the result's [i, j]
                    T* result_i_j = &(result.m_rw_span[i, j]);
                    for (size_t k = 0; k < cols(); ++k) {
                        // First multiply [i, k] * [k, j]
                        if (_mul_overflow(m_ro_span[i, k], rhs.m_ro_span[k, j], &mul_result)) {
                            throw std::overflow_error("Matrix.matmul: Multiplication results in overflow / underflow.\n");
                        }
                        if (_add_overflow(*result_i_j, mul_result, result_i_j)) {
                            throw std::overflow_error("Matrix.matmul: Addition results in overflow / underflow.\n");
                        }
                    }
                }
            }
        }
        else {
            // Use a naive loop to perform matmul
            for (size_t i = 0; i < result.rows(); ++i) {
                for (size_t j = 0; j < result.cols(); ++j) {
                    for (size_t k = 0; k < cols(); ++k) {
                        // Benefit of using spans to access the underlying data
                        result.m_rw_span[i, j] += m_ro_span[i, k] * rhs.m_ro_span[k, j];
                    }
                }
            }
        }
        return result;
    }

    /**
     * Perform a matmul on a Tensor without casting it to a Matrix, storing the result in a new Matrix
     * @param rhs Const reference to a Tensor
     * @returns Returns a new Matrix instance containing the matmul result
     */
    Matrix<T> matmul(const Tensor<T>& rhs) const {
        // Grab the Tensor's dimensions for reference later
        const std::vector<size_t>& tensor_dims = rhs.dims();
        if (!_tensor_can_matmul(rhs)) {
            throw std::invalid_argument(std::format("Matrix.matmul: Incompatible Tensor provided to matmul, cannot matmul [{}, {}] with [{}, {}]\n",
                                                    rows(), cols(), tensor_dims.at(0), tensor_dims.at(1)));
        }
        // Create a new Matrix and initialze its values to zero (done in the Tensor constructor)
        Matrix<T> result({rows(), tensor_dims.at(1)});
        // Check if we have to be concerned about overflow / underflow
        if (this->can_overflow()) {
            // Create buffer for overflow / underflow checking
            T mul_result = 0;
            for (size_t i = 0; i < result.rows(); ++i) {
                for (size_t j = 0; j < result.cols(); ++j) {
                    // Get a pointer to the result's [i, j]
                    T* result_i_j = &(result.m_rw_span[i, j]);
                    for (size_t k = 0; k < cols(); ++k) {
                        // First multiply [i, k] * [k, j]
                        if (_mul_overflow(m_ro_span[i, k], rhs.at({k, j}), &mul_result)) {
                            throw std::overflow_error("Matrix.matmul: Multiplication results in overflow / underflow.\n");
                        }
                        if (_add_overflow(*result_i_j, rhs.at({k, j}), result_i_j)) {
                            throw std::overflow_error("Matrix.matmul: Addition results in overflow / underflow.\n");
                        }
                    }
                }
            }
        }
        else {
            // Use a naive loop to perform matmul
            for (size_t i = 0; i < result.rows(); ++i) {
                for (size_t j = 0; j < result.cols(); ++j) {
                    for (size_t k = 0; k < cols(); ++k) {
                        // Benefit of using spans to access the underlying data
                        result.m_rw_span[i, j] += m_ro_span[i, k] * rhs.at({k, j});
                    }
                }
            }
        }
        return result;
    }

    /** 
     * Perform a matmul on a Matrix instance with itself
     * @param transpose Whether to transpose self when performing the matmul, resulting in
     * self @ self.transpose()
     * @returns Returns a new Matrix instance containing the matmul result 
     */
    Matrix<T> matmul_self(bool transpose) const {
        if (!transpose) {
            if (m_ro_span.extent(0) != m_ro_span.extent(1)) {
                throw std::invalid_argument("Matrix.matmul_self: Cannot perform matmul_self on a Matrix that is not square\n");
            }
        }
        // Create a new Matrix and initialze its values to zero (done in the Tensor constructor)
        Matrix<T> result({rows(), rows()});
        // Check if we have to be concerned about overflow / underflow
        if (this->can_overflow()) {
            // Create buffer for overflow / underflow checking
            T mul_result = 0;
            for (size_t i = 0; i < result.rows(); ++i) {
                for (size_t j = 0; j < result.cols(); ++j) {
                    // Get a pointer to the result's [i, j]
                    T lhs_val = 0, rhs_val = 0;
                    T* result_i_j = &(result.m_rw_span[i, j]);
                    for (size_t k = 0; k < cols(); ++k) {
                        // First multiply [i, k] * [k, j]
                        lhs_val = m_ro_span[i, k];
                        if (transpose) {
                            rhs_val = m_ro_span[j, k];
                        }
                        else {
                            rhs_val = lhs_val;
                        }
                        if (_mul_overflow(lhs_val, rhs_val, &mul_result)) {
                            throw std::overflow_error("Matrix.matmul: Multiplication results in overflow / underflow.\n");
                        }
                        if (_add_overflow(*result_i_j, mul_result, result_i_j)) {
                            throw std::overflow_error("Matrix.matmul: Addition results in overflow / underflow.\n");
                        }
                    }
                }
            }
        }
        else {
            T lhs_val = 0, rhs_val = 0;
            for (size_t i = 0; i < result.rows(); ++i) {
                for (size_t j = 0; j < result.cols(); ++j) {
                    for (size_t k = 0; k < rows(); ++k) {
                        lhs_val = m_ro_span[i, k];
                        if (transpose) {
                            rhs_val = m_ro_span[j, k];
                        }
                        else {
                            rhs_val = lhs_val;
                        }
                        result.m_rw_span[i, j] = lhs_val * rhs_val;
                    }
                }
            }
        }
        return result;
    }

    /**
     * Transpose this Matrix
     * Inspired by: https://towardsdev.com/using-std-layout-stride-with-std-mdspan-cpp23-d025aa0c9ac9
     * @returns Returns a reference to this Matrix with the transpose op completed
     */
    Matrix<T>& transpose() {
        // Store the current strides to reference later
        size_t cols = m_rw_span.extent(1);
        size_t rows = m_rw_span.extent(0);
        // Setup the strides and shapes
        std::dextents<size_t, 2> shape{cols, rows};
        std::array<size_t, 2> strides{1, shape.extent(1)};
        // Setup the spans
        m_rw_span = std::mdspan<T, std::dextents<size_t, 2>, std::layout_stride>{this->data(), std::layout_stride::mapping{shape, strides}};
        m_ro_span = std::mdspan<const T, std::dextents<size_t, 2>, std::layout_stride>{this->data(), std::layout_stride::mapping{shape, strides}};
        // Update the underlying Tensor with the swapped dims / strides
        if (!(this->transpose_tensor())) {
            throw std::runtime_error("Matrix.transpose: Failed to transpose underlying Tensor\n");
        }

        return *this;
    }

    /**
     * Allow a Matrix instance to be printed out to the console using the regular << operator
     * @param os std::ostream for output
     * @param target Const reference to a Matrix we want to print
     * @returns Returns a reference to the std::ostream
     */
    friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& target) {
        // Wrap the Matrix in brackets, with each row properly enclosed too
        os << "[";
        for (size_t i = 0; i < target.rows(); ++i) {
            os << "[";
            for (size_t j = 0; j < target.cols(); ++j) {
                // Add the value in the Matrix
                os << target.m_ro_span[i, j];
                if (j + 1 < target.cols()) {
                    // Separate the values by tabs, for readability
                    os << "\t";
                }
            }
            // Close out each row with a corresponding ]
            os << "]";
            // Add a new line after each row if we aren't at the end
            if (i + 1 < target.rows()) {
                os << "\n";
            }
        }
        // Close out the Matrix final bracket and print out the info (dims and dtype)
        os << "]. " << target.info() << ".";
        return os;
    }

    /**
     * Convert a Matrix to a string representation, similar to the << operator above
     * @returns Returns a string representation of the Matrix
     */
    std::string to_string() const {
        // Create a blank string that we will return
        std::string result = "\n";
        // Wrap the Matrix in brackets, with each row properly enclosed too
        result += "[";
        for (size_t i = 0; i < rows(); ++i) {
            result += "[";
            for (size_t j = 0; j < cols(); ++j) {
                // Add the value in the Matrix
                result += std::format("{}", m_ro_span[i, j]);
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
        // Close out the Matrix final bracket and print out the info (dims and dtype)
        result += std::format("]. {}.", this->info());
        return result;
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    /**
     * Get the sum of a Matrix's rows or columns
     * @param dim Dimension to sum across
     * @param idx Index on the specified dimension to sum
     * @returns Returns the sum
     */
    T sum(size_t dim, size_t idx) const {
        if (dim > 1) {
            throw std::invalid_argument("Matrix.sum: Invalid dim specified for sum.\n");
        }
        T result = 0;
        // Sum across a row
        if (dim == 0) {
            if (idx >= rows()) {
                throw std::invalid_argument("Matrix.sum: Invalid index provided to sum.\n");
            }
            if (this->can_overflow()) {
                for (size_t i = 0; i < cols(); ++i) {
                    if (_add_overflow(result, at({idx, i}), &result)) {
                        throw std::overflow_error("Matrix.sum: Addition will cause overflow / underflow.");
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
                throw std::invalid_argument("Matrix.sum: Invalid index provided to sum.\n");
            }
            if (this->can_overflow()) {
                for (size_t i = 0; i < rows(); ++i) {
                    if (_add_overflow(result, at({i, idx}), &result)) {
                        throw std::overflow_error("Matrix.sum: Addition will cause overflow / underflow.");
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
     * Apply softmax to a Matrix across a specified dim
     * @param dim Dimension to apply softmax across
     * @returns Returns a reference to this Matrix
     */
    Matrix<T>& softmax(size_t dim) {
        if constexpr (!std::is_floating_point_v<T>) {
            throw std::invalid_argument("Matrix.softmax: Cannot apply softmax to non-floating point Matrix.\n");
        }
        if (dim > 1) {
            throw std::invalid_argument("Matrix.softmax: Invalid dim specified for softmax.\n");
        }
        // Apply the exp function across all elements in the Matrix
        this->apply(std::exp);
        // Sum the values across the specified dim
        if (dim == 0) {
            for (size_t i = 0; i < rows(); ++i) {
                T row_sum = sum(dim, i);
                for (size_t j = 0; j < cols(); ++j) {
                    at({i, j}) /= row_sum;
                }
            }
        }
        else {
            for (size_t i = 0; i < cols(); ++i) {
                T col_sum = sum(dim, i);
                for (size_t j = 0; j < rows(); ++j) {
                    at({j, i}) /= col_sum;
                }
            }
        }
        return *this;
    }


// NOLINTEND(cppcoreguidelines-avoid-c-arrays, cppcoreguidelines-pro-bounds-pointer-arithmetic)
};

}; // namespace Tensor_NS

#endif
