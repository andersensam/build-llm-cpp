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
     * Reference wrapper to Tensor / Matrix
     */
    std::reference_wrapper<const Tensor<T>> c_ref;

    /**
     * The dimension(s) we want to slice across
     */
    std::array<size_t, 2> c_slice_dims = {0};

    /**
     * Number of dimensions we are slicing across, either 1 or 2
     */
    uint8_t c_num_slice_dims = 0;

    /**
     * Rows in the resulting slice
     */
    size_t c_rows = 0;

    /**
     * Columns in the resulting slice
     */
    size_t c_cols = 0;

    /**
     * Map of new dimensions for the first axis
     */
    std::map<size_t, size_t> m_dim0_map = std::map<size_t, size_t>();

    /**
     * Map of new dmensions for the second (optional) axis
     */
    std::map<size_t, size_t> m_dim1_map = std::map<size_t, size_t>();

    /**
     * Base coordinates for other axes in the base Tensor, only used when the
     * Tensor rank > Slice rank. This vector will get passed to the .at() function
     * for the underlying Tensor, avoiding copies along the way
     */
    std::vector<size_t> m_other_axes = std::vector<size_t>();

/* Private functions*/

    /**
     * Rewrite function for retrieving data from the underlying Tensor / Matrix
     * @param c Initializer list containing the coordinates
     * @returns Returns a const ref to a vector with the rewritten coordinates
     */
    const std::vector<size_t>& rewrite(const std::initializer_list<size_t>& c) {
        // Use the m_other_axes vector we already have and simply update the remapped
        // coordinates as needed
        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        // If we only get one coordinate, we know it must be along the second axis
        if (c.size() == 1) {
            std::cerr << "Vector size: " << m_other_axes.size() << "\n";
            m_other_axes.at(0) = m_dim0_map.at(0);
            m_other_axes.at(1) = c.begin()[0];
            return m_other_axes;
        }
        // If we are getting a vector from a high-rank Tensor, only map the one axis
        m_other_axes.at(c_slice_dims.at(0)) = m_dim0_map.at(c.begin()[0]);
        if (c_num_slice_dims == 2 && !m_dim1_map.empty()) {
            // If we are getting a Matrix from a high-rank Tensor, ensure we map both
            m_other_axes.at(c_slice_dims.at(1)) = m_dim1_map.at(c.begin()[1]);
        }
        // Ensure c has two coordinates, otherwise pass m_other_axes with only
        // one mapped coordinate back
        else if (c.size() > 1) {
            // If m_dim1_map is empty, pass through the second axis directly
            m_other_axes.at(c_slice_dims.at(1)) = c.begin()[1];
        }
        // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        return m_other_axes;
    }
/* Public functions */
public:
    /**
     * Constructor for TensorSlice, accepting either a Tensor or Matrix, an axis to fetch from,
     * and a target list of values to pull from that source
     * @param tensor Tensor or Matrix reference
     * @param axes Axes to slice across
     * @param idxs Tensor containing the indices we want to pull from the Tensor, for a 1-D Tensor Slice, this 
     * should simply be a list of values [0, 1, ... n], for a 2-D Tensor Slice, this should be formatted as 
     * the following:  [[axis 0 start, axis 0 end]
     *                  [axis 1 start, axis 1 end]]
     * @param other_axes A set of coordinates covering other axes in the Tensor, used when we are taking a slice
     * of a higher-rank Tensor, i.e. a 1-D slice from a 2-D Tensor, or a 2-D slice from a 3-D Tensor
     */
    TensorSlice(const Tensor<T>& tensor, const std::initializer_list<size_t>& axes, const Tensor<size_t>& idxs,
                const std::initializer_list<size_t>& other_axes) : c_ref(tensor), c_num_slice_dims(axes.size()) {
        // Ensure the Tensor has rank == 2, otherwise we cannot create a Tensor Slice
        if (tensor.rank() < 2) {
            throw std::invalid_argument("TensorSlice.TensorSlice: Tensor rank < 2.\n");
        }
        // Ensure we get a valid index Tensor that isn't empty
        if (idxs.rank() > 2 || idxs.elements() == 0) {
            throw std::invalid_argument("TensorSlice.TensorSlice: Invalid index Tensor provided.\n");
        }
        // Ensure we get the proper axis / set of axes
        if (axes.size() == 0 || axes.size() > 2) {
            throw std::invalid_argument("TensorSlice.TensorSlice: Invalid axis / axes provided\n");
        }
        // If we are forming a lower rank slice than the base Tensor, validate we have the right coordinates
        if (axes.size() < tensor.rank()) {
            if (tensor.rank() != other_axes.size()) {
                throw std::invalid_argument("TensorSlice.TensorSlice: Invalid base coordinates provided. tensor.rank() != other_axes.size()\n");
            }
            // Store the base coordinates
            m_other_axes.resize(other_axes.size());
            for (size_t bc : other_axes) {
                m_other_axes.push_back(bc);
            }
        }
        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        // Handle a case where we want to get a subset of rows or a subset of columns, but keep the other,
        // for example, getting an embedding projection where you might want 5 rows with all columns in the
        // embedding dimension
        if (idxs.rank() <= 1) {
            // Store the two axes properly, though we won't ever use m_dim1_map for this case
            c_slice_dims = {axes.begin()[0], axes.begin()[1]};
            // If we have a rank-1 Tensor for idxs, we know that we are getting a list of indices to pull
            // from the larger Tensor. We take index 0 (axes.begin()[0]) and only grab those values
            // while keeping the extent for axes.begin()[1]
            c_rows = idxs.elements();
            // Fetch the dims from the Tensor
            c_cols = tensor.dims().at(axes.begin()[1]);
            // Map each specified value to idxs
            for (size_t i = 0; i < c_rows; ++i) {
                m_dim0_map[i] = idxs.at({i});
            }
        }
        else {
            // If constructing a 2-D Tensor Slice, ensure our idxs Tensor is exactly a 2 x 2
            const auto& target_ranges = idxs.dims();
            for (size_t r : target_ranges) {
                if (r != 2) {
                    throw std::invalid_argument("TensorSlice.TensorSlice: Invalid ranges provided for 2-D Tensor Slice\n");
                }
            }
            // Store both axes
            c_slice_dims = {axes.begin()[0], axes.begin()[1]};
            // For the first axis, extract the number of elements to determine our extent
            c_rows = idxs.at({0, 1}) - idxs.at({0, 0});
            // Iterate over the target rows and map them
            size_t row_start = idxs.at({0, 0});
            for (size_t i = 0; i < c_rows; ++i) {
                m_dim0_map[i] = row_start + i;
            }
            // For the second axis, map to the number of columns
            c_cols = idxs.at({1, 1}) - idxs.at({1, 0});
            size_t col_start = idxs.at({1, 0});
            for (size_t i = 0; i < c_cols; ++i) {
                m_dim1_map[i] = col_start + i;
            }
        }
        // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }

    /**
     * Constructor for TensorSlice, accepting either a Tensor or Matrix, an axis to fetch from,
     * and a target list of values to pull from that source
     * @param tensor Tensor or Matrix reference
     * @param axes Axes to slice across
     * @param idxs TensorSlice containing the indices we want to pull from the Tensor, for a 1-D Tensor Slice, this 
     * should simply be a list of values [0, 1, ... n], for a 2-D Tensor Slice, this should be formatted as 
     * the following:  [[axis 0 start, axis 0 end]
     *                  [axis 1 start, axis 1 end]]
     * We don't need the TensorSlice to be const given its a read-only view of the underlying data
     * @param other_axes A set of coordinates covering other axes in the Tensor, used when we are taking a slice
     * of a higher-rank Tensor, i.e. a 1-D slice from a 2-D Tensor, or a 2-D slice from a 3-D Tensor
     */
    TensorSlice(const Tensor<T>& tensor, const std::initializer_list<size_t>& axes, TensorSlice<size_t>& idxs,
                const std::initializer_list<size_t>& other_axes) : c_ref(tensor), c_num_slice_dims(axes.size()) {
        // Ensure the Tensor has rank == 2, otherwise we cannot create a Tensor Slice
        if (tensor.rank() < 2) {
            throw std::invalid_argument("TensorSlice.TensorSlice: Tensor rank < 2.\n");
        }
        // Ensure we get a valid index Tensor that isn't empty
        if (idxs.rank() > 2 || idxs.elements() == 0) {
            throw std::invalid_argument("TensorSlice.TensorSlice: Invalid index Tensor provided.\n");
        }
        // Ensure we get the proper axis / set of axes
        if (axes.size() == 0 || axes.size() > 2) {
            throw std::invalid_argument("TensorSlice.TensorSlice: Invalid axis / axes provided\n");
        }
        // If we are forming a lower rank slice than the base Tensor, validate we have the right coordinates
        if (axes.size() < tensor.rank()) {
            if (tensor.rank() != other_axes.size()) {
                throw std::invalid_argument("TensorSlice.TensorSlice: Invalid base coordinates provided. tensor.rank() != other_axes.size()\n");
            }
            // Store the base coordinates
            m_other_axes.resize(other_axes.size());
            for (size_t bc : other_axes) {
                m_other_axes.push_back(bc);
            }
        }
        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        // Handle a case where we want to get a subset of rows or a subset of columns, but keep the other,
        // for example, getting an embedding projection where you might want 5 rows with all columns in the
        // embedding dimension
        if (idxs.rank() <= 1) {
            // Store the two axes properly, though we won't ever use m_dim1_map for this case
            c_slice_dims = {axes.begin()[0], axes.begin()[1]};
            // If we have a rank-1 Tensor for idxs, we know that we are getting a list of indices to pull
            // from the larger Tensor. We take index 0 (axes.begin()[0]) and only grab those values
            // while keeping the extent for axes.begin()[1]
            c_rows = idxs.elements();
            // Fetch the dims from the Tensor
            c_cols = tensor.dims().at(axes.begin()[1]);
            // Map each specified value to idxs
            bool row_aligned = (idxs.rows() == 1);
            for (size_t i = 0; i < c_rows; ++i) {
                if (row_aligned) {
                    m_dim0_map[i] = idxs.at({0, i});
                }
                else {
                    m_dim0_map[i] = idxs.at({i, 0});
                }
            }
        }
        else {
            // If constructing a 2-D Tensor Slice, ensure our idxs Tensor is exactly a 2 x 2
            const auto& [dim0, dim1] = idxs.dims();
            if (dim0 != 2 || dim1 != 2) {
                throw std::invalid_argument("TensorSlice.TensorSlice: Invalid ranges provided for 2-D Tensor Slice\n");
            }
            // Store both axes
            c_slice_dims = {axes.begin()[0], axes.begin()[1]};
            // For the first axis, extract the number of elements to determine our extent
            c_rows = idxs.at({0, 1}) - idxs.at({0, 0});
            // Iterate over the target rows and map them
            size_t row_start = idxs.at({0, 0});
            for (size_t i = 0; i < c_rows; ++i) {
                m_dim0_map[i] = row_start + i;
            }
            // For the second axis, map to the number of columns
            c_cols = idxs.at({1, 1}) - idxs.at({1, 0});
            size_t col_start = idxs.at({1, 0});
            for (size_t i = 0; i < c_cols; ++i) {
                m_dim1_map[i] = col_start + i;
            }
        }
        // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }

    /**
     * Access the element from an underlying Tensor / Matrix, read-only
     * @param c Coordinates we want to pull from
     * @returns Returns a read-only reference to the underlying value
     */
    const T& at(const std::initializer_list<size_t>& c) {
        // Enable polymorphic behavior
        const auto& tm = c_ref.get();
        // Check to see the Tensor rank and handle appropriately
        if (tm.rank() == 2) {
            const auto& rw = rewrite(c);
            std::cerr << "trying to rewrite: ";
            for (size_t idx : rw) {
                std::cerr << idx << " ";
            }
            std::cerr << "\n";
            return tm.at({rw.at(0), rw.at(1)});
        }
        return tm.at(rewrite(c));
    }

    /**
     * Get the rank of a TensorSlice
     * @returns Returns the rank
     */
    size_t rank() const {
        if (c_rows <= 1 || c_cols <= 1) {
            return 1;
        }
        return 2;
    }

    /**
     * Get the number of rows
     * @returns Returns the number of rows
     */
    size_t rows() const {
        return c_rows;
    }

    /**
     * Get the number of columns
     * @returns Returns the number of columns
     */
    size_t cols() const {
        return c_cols;
    }

    /**
     * Get the number of elements in a TensorSlice
     * @returns Returns the number of elements
     */
    size_t elements() const {
        if (c_num_slice_dims == 1) {
            return c_rows;
        }
        return c_rows * c_cols;
    }

    /**
     * Get the dimensions / extents of the TensorSlice
     * @returns Returns a std::pair<size_t, size_t> of the dims
     */
    std::pair<size_t, size_t> dims() const {
        return {c_rows, c_cols};
    }

    /**
     * Get a string containing the information about the TensorSlice and the
     * underlying Tensor
     * @returns Returns a string containing information about the slice
     */
    std::string info() const {
        // Prepare the info string
        std::string result = "TensorSlice: [";
        if (c_num_slice_dims == 1) {
            result += std::format("{}]. Underlying Tensor: {}", c_rows, c_ref.get().info());
            return result;
        }
        result += std::format("{}, {}]. Underlying Tensor: {}", c_rows, c_cols, c_ref.get().info());
        return result;
    }

    /**
     * Convert a TensorSlice to a Tensor
     * @returns Returns a new Tensor instance, containing only the values present in the TensorSlice
     */
    Tensor<T> to_tensor() {
        // Create a new blank Tensor with the dimensions of the slice
        if (c_num_slice_dims == 1) {
            // If we are creating a vector, use the rows
            Tensor<T> result({c_rows});
            // Iterate over the data, copying into the new result Tensor
            for (size_t i = 0; i < c_rows; ++i) {
                result.at({i}) = this->at({i});
            }
            return result;
        }
        else {
            // Create a 2-D Tensor
            Tensor<T> result({c_rows, c_cols});
            for (size_t i = 0; i < c_rows; ++i) {
                for (size_t j = 0; j < c_cols; ++j) {
                    result.at({i, j}) = this->at({i, j});
                }
            }
            return result;
        }
    }

    /**
     * Convert a TensorSlice to a Matrix
     * @returns Returns a new Matrix instance, containing only the values present in the TensorSlice
     */
    Matrix<T> to_matrix() {
        // Ensure we actually have two slice dimensions
        if (c_num_slice_dims != 2) {
            throw std::invalid_argument("TensorSlice.to_matrix: Cannot convert a 1-D TensorSlice to a Matrix\n");
        }
        Matrix<T> result({c_rows, c_cols});
        for (size_t i = 0; i < c_rows; ++i) {
            for (size_t j = 0; j < c_cols; ++j) {
                result.at({i, j}) = this->at({i, j});
            }
        }
        return result;
    }

    /**
     * Calculate the dot product of two TensorSlices
     * @param lhs TensorSlice to calculate dot product with
     * @returns Returns the sum of the slices
     */
    T dot(TensorSlice<T>& lhs) {
        // Ensure each TensorSlice has rank == 1
        if (rank() != 1 || lhs.rank() != 1) {
            throw std::invalid_argument("TensorSlice.dot: TensorSlices must have rank == 1 for dot.\n");
        }
        // Ensure each TensorSlice has the same number of elements
        if (elements() != lhs.elements()) {
            throw std::invalid_argument("TensorSlice.dot: TensorSlices must have the same number of elements\n");
        }
        // Check to see if we need to be concerned about overflow / underflow
        if (c_ref.get().can_overflow()) {
            // Use the overflow / underflow detection for safety
            T result = 0, mul_result = 0;
            // Iterate over the elements and multiply them element-wise, then sum
            for (size_t i = 0; i < elements(); ++i) {
                if (_mul_overflow(this->at({i}), lhs.at({i}), &mul_result)) {
                    throw std::overflow_error("TensorSlice.dot: Multiplication in dot will cause overflow / underflow\n");
                }
                if (_add_overflow(result, mul_result, &result)) {
                    throw std::overflow_error("TensorSlice.dot: Addition of multiplication result will cause overflow / underflow\n");
                }
            }
            return result;
        }
        // Otherwise, perform the op directly
        T result = 0;
        for (size_t i = 0; i < elements(); ++i) {
            result += this->at({i}) * lhs.at({i});
        }
        return result;
    }
};