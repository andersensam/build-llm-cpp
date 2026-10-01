#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "SliceConfig.hpp"
#include "Tensor.hpp"
#include "TensorSlice.hpp"

namespace {

using SliceConfig_NS::IndexType;
using SliceConfig_NS::MatrixSliceConfig;
using SliceConfig_NS::VectorSliceConfig;
using SliceConfig_NS::VectorSliceOrientation;
using Tensor_NS::Tensor;
using TensorSlice_NS::ListTensorSlice;
using TensorSlice_NS::RangeTensorSlice;

// ============================================================================
// SliceConfig Validation
// ============================================================================

TEST(SliceTest, SliceConfigValidation) {
  // Valid 1D row slice config
  VectorSliceConfig vsc(0, 2, 1, {}, VectorSliceOrientation::ROW, {});
  EXPECT_EQ(vsc.get_dim0(), 0U);
  EXPECT_EQ(vsc.get_dim1(), 1U);
  EXPECT_TRUE(vsc.has_orientation());
  EXPECT_EQ(vsc.get_orientation(), VectorSliceOrientation::ROW);
  EXPECT_FALSE(vsc.has_dim1_filter());

  // Invalid dim1_filter on VectorSliceConfig (start >= end or wrong size)
  EXPECT_THROW(
    VectorSliceConfig(0, 0, 1, {3, 1}, VectorSliceOrientation::ROW, {}),
    std::invalid_argument
  );
  EXPECT_THROW(
    VectorSliceConfig(0, 0, 1, {2}, VectorSliceOrientation::ROW, {}),
    std::invalid_argument
  );

  // Duplicate dim0 == dim1 is rejected when constructing ListTensorSlice
  auto base = std::make_shared<Tensor<int32_t>>(std::initializer_list<size_t>{3, 3});
  VectorSliceConfig dup_dim_vsc(0, 0, 0, {}, VectorSliceOrientation::ROW, {});
  EXPECT_THROW((ListTensorSlice<int32_t>(base, dup_dim_vsc)), std::invalid_argument);

  // Invalid range where start >= end or span == 1 on MatrixSliceConfig
  EXPECT_THROW(
    MatrixSliceConfig(0, IndexType::RANGE, {3, 1}, 1, {}, {}),
    std::invalid_argument
  );
  EXPECT_THROW(
    MatrixSliceConfig(0, IndexType::RANGE, {1, 2}, 1, {}, {}),
    std::invalid_argument
  );
}

// ============================================================================
// RangeTensorSlice Tests
// ============================================================================

TEST(SliceTest, RangeTensorSliceReadWriteAndMaterialize) {
  auto base = std::make_shared<Tensor<int32_t>>(std::initializer_list<size_t>{4, 4});
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      base->at({i, j}) = static_cast<int32_t>(i * 10 + j);
    }
  }

  // Slice rows [1, 3) and cols [1, 4) -> 2x3 submatrix
  MatrixSliceConfig msc(0, IndexType::RANGE, {1, 3}, 1, {1, 4}, {});
  RangeTensorSlice<int32_t> slice(base, msc);

  EXPECT_EQ(slice.rank(), 2U);
  EXPECT_EQ(slice.shape(), std::vector<size_t>({2, 3}));
  EXPECT_EQ(slice.elements(), 6U);
  // RangeTensorSlice is non-contiguous (validates commit bd3dd51 fix)
  EXPECT_FALSE(slice.contiguous());

  EXPECT_EQ(slice.at({0, 0}), 11);
  EXPECT_EQ(slice.at({0, 2}), 13);
  EXPECT_EQ(slice.at({1, 0}), 21);
  EXPECT_EQ(slice.at({1, 2}), 23);

  // Mutate through slice and verify underlying Tensor is updated
  slice.at({0, 1}) = 999;
  EXPECT_EQ(base->at({1, 2}), 999);

  // Materialize to standalone contiguous Tensor
  Tensor<int32_t> mat = slice.to_tensor();
  EXPECT_EQ(mat.shape(), std::vector<size_t>({2, 3}));
  EXPECT_TRUE(mat.contiguous());
  EXPECT_EQ(mat.at({0, 1}), 999);
  EXPECT_EQ(mat.at({1, 2}), 23);
}

TEST(SliceTest, ConstRangeTensorSliceDisallowsWrite) {
  auto base = std::make_shared<Tensor<int32_t>>(std::initializer_list<size_t>{3, 3});
  base->fill(7);
  std::shared_ptr<const Tensor<int32_t>> const_base = base;

  MatrixSliceConfig msc(0, IndexType::RANGE, {0, 2}, 1, {}, {});
  RangeTensorSlice<int32_t> const_slice(const_base, msc);

  EXPECT_EQ(std::as_const(const_slice).at({0, 0}), 7);
  EXPECT_THROW(const_slice.at({0, 0}) = 42, std::runtime_error);
}

// ============================================================================
// ListTensorSlice Tests
// ============================================================================

TEST(SliceTest, ListTensorSlice1DVectorAnd2DGather) {
  auto emb = std::make_shared<Tensor<float>>(std::initializer_list<size_t>{5, 3});
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      emb->at({i, j}) = static_cast<float>(i * 100 + j);
    }
  }

  // 1D row slice for token 3
  VectorSliceConfig vsc(0, 3, 1, {}, VectorSliceOrientation::ROW, {});
  ListTensorSlice<float> row_slice(emb, vsc);
  EXPECT_EQ(row_slice.rank(), 1U);
  EXPECT_EQ(row_slice.elements(), 3U);
  EXPECT_FALSE(row_slice.contiguous());
  EXPECT_FLOAT_EQ(row_slice.at({0}), 300.0f);
  EXPECT_FLOAT_EQ(row_slice.at({1}), 301.0f);
  EXPECT_FLOAT_EQ(row_slice.at({2}), 302.0f);

  // 2D gather slice for tokens {4, 1, 3}
  MatrixSliceConfig msc(0, IndexType::LIST, {4, 1, 3}, 1, {}, {});
  ListTensorSlice<float> gather_slice(emb, msc);
  EXPECT_EQ(gather_slice.rank(), 2U);
  EXPECT_EQ(gather_slice.shape(), std::vector<size_t>({3, 3}));
  EXPECT_FLOAT_EQ(gather_slice.at({0, 0}), 400.0f);
  EXPECT_FLOAT_EQ(gather_slice.at({1, 2}), 102.0f);
  EXPECT_FLOAT_EQ(gather_slice.at({2, 1}), 301.0f);

  // Materialize gathered rows into a contiguous Tensor
  Tensor<float> gathered = gather_slice.to_tensor();
  EXPECT_EQ(gathered.shape(), std::vector<size_t>({3, 3}));
  EXPECT_FLOAT_EQ(gathered.at({0, 2}), 402.0f);
  EXPECT_FLOAT_EQ(gathered.at({1, 0}), 100.0f);
  EXPECT_FLOAT_EQ(gathered.at({2, 2}), 302.0f);
}

} // namespace
