#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "ListTensorSlice.hpp"
#include "SliceConfig.hpp"
#include "Tensor.hpp"

namespace {

using ListTensorSlice_NS::ListTensorSlice;
using SliceConfig_NS::IndexType;
using SliceConfig_NS::MatrixSliceConfig;
using SliceConfig_NS::VectorSliceConfig;
using SliceConfig_NS::VectorSliceOrientation;
using Tensor_NS::Tensor;

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
  EXPECT_FALSE(row_slice.is_contiguous());
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

// ============================================================================
// Corner Cases: Transpose, 3D Slices, AbstractTensor copy_from, Axis Ordering
// ============================================================================

TEST(SliceTest, ListTensorSlice3DWithOtherDims) {
  auto base3d = std::make_shared<Tensor<int32_t>>(std::initializer_list<size_t>{2, 5, 3});
  for (size_t b = 0; b < 2; ++b) {
    for (size_t r = 0; r < 5; ++r) {
      for (size_t c = 0; c < 3; ++c) {
        base3d->at({b, r, c}) = static_cast<int32_t>(b * 100 + r * 10 + c);
      }
    }
  }

  MatrixSliceConfig msc(1, IndexType::LIST, {4, 1}, 2, {}, {{0, 1}});
  ListTensorSlice<int32_t> l_slice(base3d, msc);
  EXPECT_EQ(l_slice.shape(), std::vector<size_t>({2, 3}));
  EXPECT_EQ(l_slice.at({0, 0}), 140);
  EXPECT_EQ(l_slice.at({0, 2}), 142);
  EXPECT_EQ(l_slice.at({1, 1}), 111);
}

TEST(SliceTest, TensorCopyFrom2DRangeAndListTensorSlices) {
  auto base = std::make_shared<Tensor<int32_t>>(std::initializer_list<size_t>{4, 3});
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      base->at({i, j}) = static_cast<int32_t>((i + 1) * 10 + j);
    }
  }

  MatrixSliceConfig l_cfg(0, IndexType::LIST, {3, 0}, 1, {}, {});
  ListTensorSlice<int32_t> l_slice(base, l_cfg);
  Tensor<int32_t> dest_l({2, 3});
  dest_l.copy_from(static_cast<const Tensor_NS::AbstractTensor<int32_t>&>(l_slice));
  EXPECT_EQ(dest_l.at({0, 0}), 40);
  EXPECT_EQ(dest_l.at({1, 2}), 12);
}

TEST(SliceTest, TensorCopyFrom1DRowListTensorSlice) {
  auto base = std::make_shared<Tensor<int32_t>>(std::initializer_list<size_t>{4, 3});
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      base->at({i, j}) = static_cast<int32_t>((i + 1) * 10 + j);
    }
  }

  // 1D ROW slice of row 2 -> 3 elements {30, 31, 32}
  VectorSliceConfig row_cfg(0, 2, 1, {}, VectorSliceOrientation::ROW, {});
  ListTensorSlice<int32_t> row_slice(base, row_cfg);
  ASSERT_EQ(row_slice.rank(), 1U);
  ASSERT_EQ(row_slice.elements(), 3U);

  Tensor<int32_t> dest_1d({3});
  dest_1d.copy_from(static_cast<const Tensor_NS::AbstractTensor<int32_t>&>(row_slice));
  EXPECT_EQ(dest_1d.at({0}), 30);
  EXPECT_EQ(dest_1d.at({1}), 31);
  EXPECT_EQ(dest_1d.at({2}), 32);
}

TEST(SliceTest, ListTensorSliceTransposeWithoutDim1FilterReadsTransposedCoordinates) {
  auto emb = std::make_shared<Tensor<int32_t>>(std::initializer_list<size_t>{5, 3});
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      emb->at({i, j}) = static_cast<int32_t>(i * 100 + j);
    }
  }

  // Gather rows {4, 1} without a dim1_filter -> shape [2, 3]
  MatrixSliceConfig msc(0, IndexType::LIST, {4, 1}, 1, {}, {});
  ListTensorSlice<int32_t> gather_slice(emb, msc);
  ASSERT_EQ(gather_slice.shape(), std::vector<size_t>({2, 3}));

  // Transpose to [3, 2] and read (col, gathered_row)
  gather_slice.transpose(0, 1);
  EXPECT_EQ(gather_slice.shape(), std::vector<size_t>({3, 2}));
  EXPECT_EQ(gather_slice.at({0, 0}), 400);
  EXPECT_EQ(gather_slice.at({2, 0}), 402);
  EXPECT_EQ(gather_slice.at({0, 1}), 100);
  EXPECT_EQ(gather_slice.at({2, 1}), 102);
}

TEST(SliceTest, ListTensorSliceWithColumnsAsDim0AndRowsAsDim1) {
  auto mat = std::make_shared<Tensor<int32_t>>(std::initializer_list<size_t>{5, 3});
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      mat->at({i, j}) = static_cast<int32_t>(i * 100 + j);
    }
  }

  // Gather columns {2, 0} along dim0=1, with all 5 rows along dim1=0 -> shape [2, 5]
  MatrixSliceConfig msc(1, IndexType::LIST, {2, 0}, 0, {}, {});
  ListTensorSlice<int32_t> col_gather(mat, msc);
  ASSERT_EQ(col_gather.shape(), std::vector<size_t>({2, 5}));
  EXPECT_EQ(col_gather.at({0, 0}), 2);    // row 0, col 2
  EXPECT_EQ(col_gather.at({0, 4}), 402);  // row 4, col 2
  EXPECT_EQ(col_gather.at({1, 0}), 0);    // row 0, col 0
  EXPECT_EQ(col_gather.at({1, 4}), 400);  // row 4, col 0
}

} // namespace
