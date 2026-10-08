#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "SliceConfig.hpp"
#include "Tensor.hpp"
#include "TensorMatmul.hpp"
#include "TensorSlice.hpp"

namespace {

using SliceConfig_NS::IndexType;
using SliceConfig_NS::MatrixSliceConfig;
using TensorMatmul_NS::matmul;
using Tensor_NS::Tensor;
using TensorSlice_NS::ListTensorSlice;

// ============================================================================
// Basic 2D Matrix Multiplication Across Numeric Types
// ============================================================================

TEST(MatmulTest, Basic2DFloatMatmul) {
  // [2, 3] x [3, 2] -> [2, 2]
  Tensor<float> lhs({2, 3});
  lhs.set({1.0f, 2.0f, 3.0f,
       4.0f, 5.0f, 6.0f});

  Tensor<float> rhs({3, 2});
  rhs.set({7.0f, 8.0f,
       9.0f, 10.0f,
       11.0f, 12.0f});

  Tensor<float> res = matmul(lhs, rhs);
  EXPECT_EQ(res.shape(), std::vector<size_t>({2, 2}));
  // Row 0: [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] = [58, 64]
  // Row 1: [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] = [139, 154]
  EXPECT_FLOAT_EQ(res.at({0, 0}), 58.0f);
  EXPECT_FLOAT_EQ(res.at({0, 1}), 64.0f);
  EXPECT_FLOAT_EQ(res.at({1, 0}), 139.0f);
  EXPECT_FLOAT_EQ(res.at({1, 1}), 154.0f);
}

TEST(MatmulTest, SignedIntegerMatmulAccumulatesMulResultCorrectly) {
  // Specifically validates commit c945eb2: _naive_matmul_impl_v1 signed integer path
  // must accumulate lhs[i, k] * rhs[k, j] (mul_result), not rhs[k, j]!
  Tensor<int32_t> lhs({2, 2});
  lhs.set({3, 4,
       -2, 5});

  Tensor<int32_t> rhs({2, 2});
  rhs.set({10, -3,
       7, 6});

  Tensor<int32_t> res = matmul(lhs, rhs);
  // [3*10 + 4*7, 3*(-3) + 4*6]  = [58, 15]
  // [-2*10 + 5*7, -2*(-3) + 5*6] = [15, 36]
  EXPECT_EQ(res.at({0, 0}), 58);
  EXPECT_EQ(res.at({0, 1}), 15);
  EXPECT_EQ(res.at({1, 0}), 15);
  EXPECT_EQ(res.at({1, 1}), 36);
}

TEST(MatmulTest, UnsignedIntegerMatmulAccumulatesMulResultCorrectly) {
  // Specifically validates commit c945eb2: _naive_matmul_impl_v1 unsigned integer path
  Tensor<uint32_t> lhs({2, 2});
  lhs.set({3U, 4U,
       5U, 6U});

  Tensor<uint32_t> rhs({2, 2});
  rhs.set({10U, 20U,
       30U, 40U});

  Tensor<uint32_t> res = matmul(lhs, rhs);
  // [3*10 + 4*30, 3*20 + 4*40] = [150, 220]
  // [5*10 + 6*30, 5*20 + 6*40] = [230, 340]
  EXPECT_EQ(res.at({0, 0}), 150U);
  EXPECT_EQ(res.at({0, 1}), 220U);
  EXPECT_EQ(res.at({1, 0}), 230U);
  EXPECT_EQ(res.at({1, 1}), 340U);
}

TEST(MatmulTest, SizeTMatmulAndAccumulatorType) {
  static_assert(std::is_same_v<typename Numerics_NS::Accumulator<size_t>::type, size_t>);
  Tensor<size_t> lhs({2, 2});
  lhs.set({1UL, 2UL, 3UL, 4UL});
  Tensor<size_t> rhs({2, 2});
  rhs.set({5UL, 6UL, 7UL, 8UL});

  Tensor<size_t> res = matmul(lhs, rhs);
  EXPECT_EQ(res.at({0, 0}), 19UL);
  EXPECT_EQ(res.at({0, 1}), 22UL);
  EXPECT_EQ(res.at({1, 0}), 43UL);
  EXPECT_EQ(res.at({1, 1}), 50UL);
}

TEST(MatmulTest, DestinationTensorOverloadZeroesPreviousContents) {
  Tensor<int32_t> lhs({2, 2});
  lhs.set({1, 2, 3, 4});
  Tensor<int32_t> rhs({2, 2});
  rhs.set({5, 6, 7, 8});

  Tensor<int32_t> dest({2, 2});
  dest.fill(999);

  matmul(lhs, rhs, dest);
  EXPECT_EQ(dest.at({0, 0}), 19);
  EXPECT_EQ(dest.at({0, 1}), 22);
  EXPECT_EQ(dest.at({1, 0}), 43);
  EXPECT_EQ(dest.at({1, 1}), 50);
}

// ============================================================================
// Transposed and Dimension-Parameterized Matmul
// ============================================================================

TEST(MatmulTest, MatmulWithImplicitTransposeDims) {
  // Compute Q @ Q^T via matmul(Q, 0, 1, Q, 1, 0) where Q is [2, 3]
  Tensor<float> q({2, 3});
  q.set({1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f});

  Tensor<float> scores = matmul(q, 0, 1, q, 1, 0);
  EXPECT_EQ(scores.shape(), std::vector<size_t>({2, 2}));
  // [0, 0] = 1^2 + 2^2 + 3^2 = 14
  // [0, 1] = 1*4 + 2*5 + 3*6 = 32
  // [1, 0] = 32
  // [1, 1] = 4^2 + 5^2 + 6^2 = 77
  EXPECT_FLOAT_EQ(scores.at({0, 0}), 14.0f);
  EXPECT_FLOAT_EQ(scores.at({0, 1}), 32.0f);
  EXPECT_FLOAT_EQ(scores.at({1, 0}), 32.0f);
  EXPECT_FLOAT_EQ(scores.at({1, 1}), 77.0f);
}

TEST(MatmulTest, MatmulWith3DBatchCoordinateSlice) {
  // 3D tensors [batch=2, rows=2, cols=2], multiply batch 1 of lhs with batch 1 of rhs
  Tensor<int32_t> lhs({2, 2, 2});
  Tensor<int32_t> rhs({2, 2, 2});
  // Populate batch 1
  lhs.at({1, 0, 0}) = 2; lhs.at({1, 0, 1}) = 3;
  lhs.at({1, 1, 0}) = 4; lhs.at({1, 1, 1}) = 5;

  rhs.at({1, 0, 0}) = 6; rhs.at({1, 0, 1}) = 7;
  rhs.at({1, 1, 0}) = 8; rhs.at({1, 1, 1}) = 9;

  std::vector<size_t> lhs_coords = {1, 0, 0};
  std::vector<size_t> rhs_coords = {1, 0, 0};
  Tensor<int32_t> res = matmul(lhs, 1, 2, lhs_coords, rhs, 1, 2, rhs_coords);
  EXPECT_EQ(res.shape(), std::vector<size_t>({2, 2}));
  // [2*6 + 3*8, 2*7 + 3*9] = [36, 41]
  // [4*6 + 5*8, 4*7 + 5*9] = [64, 73]
  EXPECT_EQ(res.at({0, 0}), 36);
  EXPECT_EQ(res.at({0, 1}), 41);
  EXPECT_EQ(res.at({1, 0}), 64);
  EXPECT_EQ(res.at({1, 1}), 73);
}

// ============================================================================
// Matmul with a Tensor slice and ListTensorSlice
// ============================================================================

TEST(MatmulTest, MatmulWithRangeAndListTensorSlices) {
  auto base = std::make_shared<Tensor<float>>(std::initializer_list<size_t>{4, 3});
  base->set({1.0f, 0.0f, 2.0f,
        0.0f, 3.0f, 1.0f,
        2.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 1.0f});

  // Range slice: rows [1, 3) -> 2x3 matrix:
  // [0, 3, 1]
  // [2, 1, 0]
  Tensor<float> r_slice = base->slice({0, 1}, {{1, 3}, {0, 0}}, {});

  // List slice: rows {0, 3} -> 2x3 matrix:
  // [1, 0, 2]
  // [1, 1, 1]
  MatrixSliceConfig l_cfg(0, IndexType::LIST, {0, 3}, 1, {}, {});
  ListTensorSlice<float> l_slice(base, l_cfg);

  // Multiply r_slice (2x3) @ l_slice^T (3x2) -> 2x2
  Tensor<float> out = matmul(r_slice, 0, 1, l_slice, 1, 0);
  EXPECT_EQ(out.shape(), std::vector<size_t>({2, 2}));
  // Row 0 of r_slice [0, 3, 1] dot Row 0 of l_slice [1, 0, 2] = 2
  // Row 0 of r_slice [0, 3, 1] dot Row 1 of l_slice [1, 1, 1] = 4
  // Row 1 of r_slice [2, 1, 0] dot Row 0 of l_slice [1, 0, 2] = 2
  // Row 1 of r_slice [2, 1, 0] dot Row 1 of l_slice [1, 1, 1] = 3
  EXPECT_FLOAT_EQ(out.at({0, 0}), 2.0f);
  EXPECT_FLOAT_EQ(out.at({0, 1}), 4.0f);
  EXPECT_FLOAT_EQ(out.at({1, 0}), 2.0f);
  EXPECT_FLOAT_EQ(out.at({1, 1}), 3.0f);
}

// ============================================================================
// Error Handling & Overflow Detection
// ============================================================================

TEST(MatmulTest, IncompatibleShapesAndRanksThrow) {
  Tensor<float> a({2, 3});
  Tensor<float> b({4, 2});
  EXPECT_THROW((void)matmul(a, b), std::invalid_argument);

  // Same dim0 and dim1 must fail _can_matmul
  EXPECT_THROW((void)matmul(a, 0, 0, b, 0, 1), std::invalid_argument);

  // Wrong destination shape must throw
  Tensor<float> c({3, 2});
  Tensor<float> bad_dest({3, 3});
  EXPECT_THROW(matmul(a, c, bad_dest), std::invalid_argument);

  // Rank != 2 for 2D matmul(lhs, rhs) must throw
  Tensor<float> rank3({2, 2, 2});
  EXPECT_THROW((void)matmul(rank3, a), std::invalid_argument);
}

TEST(MatmulTest, IntegerAndFloatOverflowThrows) {
  // Signed int8_t overflow during accumulation
  Tensor<int8_t> s_lhs({2, 2});
  s_lhs.fill(10);
  Tensor<int8_t> s_rhs({2, 2});
  s_rhs.fill(10);
  // 10*10 + 10*10 = 200 > 127 (INT8_MAX)
  EXPECT_THROW((void)matmul(s_lhs, s_rhs), std::overflow_error);

  // Unsigned uint8_t overflow during multiplication
  Tensor<uint8_t> u_lhs({2, 2});
  u_lhs.fill(20);
  Tensor<uint8_t> u_rhs({2, 2});
  u_rhs.fill(20);
  // 20*20 = 400 > 255 (UINT8_MAX)
  EXPECT_THROW((void)matmul(u_lhs, u_rhs), std::overflow_error);

  // Floating-point overflow
  Tensor<float> f_lhs({2, 2});
  f_lhs.fill(std::numeric_limits<float>::max());
  Tensor<float> f_rhs({2, 2});
  f_rhs.fill(2.0f);
  EXPECT_THROW((void)matmul(f_lhs, f_rhs), std::overflow_error);
}

// ============================================================================
// Corner Cases: Tensor::slice Views, Signed Min Overflow, Coordinate Validation
// ============================================================================

TEST(MatmulTest, MatmulWithNonContiguousTensorSliceViewsMatchesContiguous) {
  Tensor<float> q_full({4, 6});
  Tensor<float> k_full({4, 6});
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 6; ++j) {
      q_full.at({i, j}) = static_cast<float>(i + j + 1);
      k_full.at({i, j}) = static_cast<float>((i + 1) * (j + 1));
    }
  }

  // Non-contiguous column slices [4, 3] representing a single attention head
  Tensor<float> q_slice = q_full.slice({0, 1}, {{0, 0}, {2, 5}}, {});
  Tensor<float> k_slice = k_full.slice({0, 1}, {{0, 0}, {2, 5}}, {});
  ASSERT_FALSE(q_slice.is_contiguous());
  ASSERT_FALSE(k_slice.is_contiguous());

  Tensor<float> scores_from_slices = matmul(q_slice, 0, 1, k_slice, 1, 0);

  Tensor<float> q_contig = q_slice.clone();
  Tensor<float> k_contig = k_slice.clone();
  ASSERT_TRUE(q_contig.is_contiguous());
  ASSERT_TRUE(k_contig.is_contiguous());

  Tensor<float> scores_from_contig = matmul(q_contig, 0, 1, k_contig, 1, 0);
  ASSERT_EQ(scores_from_slices.shape(), scores_from_contig.shape());
  for (size_t i = 0; i < scores_from_slices.elements(); ++i) {
    EXPECT_FLOAT_EQ(scores_from_slices.at(i), scores_from_contig.at(i));
  }
}

TEST(MatmulTest, SignedIntMinTimesMinusOneThrowsOverflowWithoutUB) {
  Tensor<int8_t> lhs({1, 1});
  lhs.at({0, 0}) = std::numeric_limits<int8_t>::min(); // -128
  Tensor<int8_t> rhs({1, 1});
  rhs.at({0, 0}) = -1;
  // -128 * -1 = +128 > INT8_MAX (127) -> must throw overflow_error
  EXPECT_THROW((void)matmul(lhs, rhs), std::overflow_error);

  Tensor<int32_t> lhs32({1, 1});
  lhs32.at({0, 0}) = std::numeric_limits<int32_t>::min();
  Tensor<int32_t> rhs32({1, 1});
  rhs32.at({0, 0}) = -1;
  EXPECT_THROW((void)matmul(lhs32, rhs32), std::overflow_error);
}

TEST(MatmulTest, Matmul3DInvalidCoordsThrows) {
  Tensor<float> lhs({2, 3, 4});
  Tensor<float> rhs({2, 4, 3});
  lhs.fill(1.0f);
  rhs.fill(1.0f);

  // Wrong coordinate vector length
  std::vector<size_t> short_coords = {0, 0};
  std::vector<size_t> valid_coords = {0, 0, 0};
  EXPECT_THROW((void)matmul(lhs, 1, 2, short_coords, rhs, 1, 2, valid_coords), std::invalid_argument);
}

} // namespace
