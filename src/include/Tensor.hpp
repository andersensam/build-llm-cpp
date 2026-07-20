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
 * @version: 2026-07-19
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef TENSOR_HPP
#define TENSOR_HPP

/* Standard dependencies */
#include <algorithm>
#include <cstring>
#include <expected>
#include <format>
#include <functional>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <stdexcept>
#include <vector>

/* Local dependencies */
#include "Log.hpp"

namespace Tensor_NS {

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
     * to aid in lifecycle management of Matrix
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

/* Private methods */
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
                return std::unexpected(std::format("Tensor._get_offset: Index {} exceeds dim ({}). Max possible index is {}.", 
                                                    c.begin()[i], m_dims.at(i), m_dims.at(i) - 1));
            }
            target_offset += c.begin()[i] * m_stride.at(i);
            //std::cout << "Current offset: " << target_offset << "\n";
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

/* Public methods */
public:
    /**
     * Default constructor for Tensor, taking in a reference to a vector containing the desired dimensions
     * for the resulting Tensor
     * @param dims Const reference to std::vector<size_t> containing the desired dimensions
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
        // Allocate the block of memory for the tensor
        m_data = std::make_unique<T[]>(c_elements);
        std::memset(m_data.get(), 0, sizeof(T) * c_elements);
        // Calculate the stides needed to get between dims
        m_stride.at(0) = 1;
        for (size_t i = 1; i < c_rank; ++i) {
            m_stride.at(i) = m_stride.at(i - 1) * m_dims.at(i);
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
        std::cerr << "Making copy\n";
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
        std::cerr << "Making copy assignment\n";

        return *this;
    }

    /**
     * Move constructor for Tensor, taking the data from target
     * @param target Tensor to move data out of
     */
    Tensor(Tensor<T>&& target) noexcept : m_data(std::move(target.m_data)), c_elements(target.c_elements), c_rank(target.c_rank), 
                                          m_stride(std::move(target.m_stride)), m_dims(std::move(target.m_dims)) {
        // Left blank since using the move constructors for std::vector and std::unique_ptr handle the setup
        std::cerr << "Moving\n";
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
        std::cerr << "Moving via assignment\n";

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
        for (size_t i = 0; i < c_elements; ++i) {
            m_data.get()[i] += target.m_data.get()[i];
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
     * Subtraction assignment operator, subtracting values from target to this Tensor
     * @param target Tensor to subtract with
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& operator-=(const Tensor<T>& target) {
        // Check to see if our Tensor shapes are compatible
        if (!_compatible(target)) {
            throw std::invalid_argument("Tensor.-=: Incompatible Tensor shapes provided to -=.\n");
        }
        for (size_t i = 0; i < c_elements; ++i) {
            m_data.get()[i] -= target.m_data.get()[i];
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
     * Multiplication assignment operator, multiplying values from target with this Tensor
     * @param target Tensor to multiply with
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& operator*=(const Tensor<T>& target) {
        // Check to see if our Tensor shapes are compatible
        if (!_compatible(target)) {
            throw std::invalid_argument("Tensor.*=: Incompatible Tensor shapes provided to *=.\n");
        }
        for (size_t i = 0; i < c_elements; ++i) {
            m_data.get()[i] *= target.m_data.get()[i];
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
     * Division assignment operator, dividing values from target with this Tensor
     * @param target Tensor to divide with
     * @returns Returns a reference to this Tensor
     */
    Tensor<T>& operator/=(const Tensor<T>& target) {
        // Check to see if our Tensor shapes are compatible
        if (!_compatible(target)) {
            throw std::invalid_argument("Tensor./=: Incompatible Tensor shapes provided to /=.\n");
        }
        for (size_t i = 0; i < c_elements; ++i) {
            m_data.get()[i] /= target.m_data.get()[i];
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
     * Default destructor for Tensor
     */
    ~Tensor() {
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
    size_t rank() const {
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
     * @param target The coordinate we want to fetch from the Tensor
     * @returns Returns a reference to the value that can be updated
     */
    T& at(const std::initializer_list<size_t>& target) {

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
            //std::cout << "Final offset.. attempting to get value @ " << target_offset.value() << "\n";
            return m_data.get()[target_offset.value()];
        }
        throw std::out_of_range(std::format("Tensor.at: Out of range: {}", target_offset.error()));
    }

    /**
     * Get a value at a specific coordinate inside the Tensor
     * @param target The coordinate we want to fetch from the tensor
     * @returns Returns the value at the coordinate
     */
    const T& at(const std::initializer_list<size_t>& target) const {
        // We use the same logic as the mutable case, just casting to a const reference
        return at(target);
    }

    /**
     * Fill a Tensor with a valid
     * @param v Value to fill the Tensor with
     */
    void fill(const T& v) {
        // Set all elements of the Tensor to v
        for (size_t i = 0; i < c_elements; ++i) {
            m_data.get()[i] = v;
        }
    }


// NOLINTEND(cppcoreguidelines-avoid-c-arrays, cppcoreguidelines-pro-bounds-pointer-arithmetic)
};

}; // namespace Tensor_NS

#endif
