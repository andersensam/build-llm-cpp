#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "Tensor.hpp"

namespace {

using Tensor_NS::CausalMaskType;
using Tensor_NS::SqueezedOpType;
using Tensor_NS::Tensor;
using Tensor_NS::uniform_op;

// ============================================================================
// Construction, Shape, Strides, and Element Access
// ============================================================================

TEST(TensorTest, ConstructRank0Scalar) {
  Tensor<float> scalar({});
  EXPECT_EQ(scalar.rank(), 0U);
  EXPECT_EQ(scalar.elements(), 1U);
  EXPECT_TRUE(scalar.contiguous());
  scalar.at({}) = 42.5f;
  EXPECT_FLOAT_EQ(scalar.at({}), 42.5f);
  EXPECT_FLOAT_EQ(scalar.at(0), 42.5f);
}

TEST(TensorTest, Construct1D2D3D4DStrides) {
  // 1D Tensor
  Tensor<int32_t> t1({5});
  EXPECT_EQ(t1.rank(), 1U);
  EXPECT_EQ(t1.elements(), 5U);
  EXPECT_EQ(t1.stride(), std::vector<size_t>({1}));

  // 2D Tensor
  Tensor<int32_t> t2({3, 4});
  EXPECT_EQ(t2.rank(), 2U);
  EXPECT_EQ(t2.elements(), 12U);
  EXPECT_EQ(t2.rows(), 3U);
  EXPECT_EQ(t2.cols(), 4U);
  EXPECT_EQ(t2.stride(), std::vector<size_t>({4, 1}));

  // 3D Tensor via initializer_list (validates rank >= 3 stride fix)
  Tensor<int32_t> t3({2, 3, 4});
  EXPECT_EQ(t3.rank(), 3U);
  EXPECT_EQ(t3.elements(), 24U);
  EXPECT_EQ(t3.stride(), std::vector<size_t>({12, 4, 1}));

  // 4D Tensor via std::vector constructor (validates rank >= 3 stride fix)
  std::vector<size_t> dims4 = {2, 3, 4, 5};
  Tensor<int32_t> t4(dims4);
  EXPECT_EQ(t4.rank(), 4U);
  EXPECT_EQ(t4.elements(), 120U);
  EXPECT_EQ(t4.stride(), std::vector<size_t>({60, 20, 5, 1}));
}

TEST(TensorTest, ZeroDimensionThrows) {
  EXPECT_THROW((Tensor<float>({0, 4})), std::invalid_argument);
  EXPECT_THROW((Tensor<float>(std::vector<size_t>{3, 0, 2})), std::invalid_argument);
}

TEST(TensorTest, ElementAccess3DCoordinatesMatchLinearOrder) {
  Tensor<int32_t> t({2, 3, 4});
  int32_t val = 0;
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 4; ++k) {
        t.at({i, j, k}) = val;
        EXPECT_EQ(t.at(std::vector<size_t>{i, j, k}), val);
        EXPECT_EQ(t.at(static_cast<size_t>(val)), val);
        ++val;
      }
    }
  }
  EXPECT_THROW((void)t.at({2, 0, 0}), std::out_of_range);
  EXPECT_THROW((void)t.at({0, 0}), std::invalid_argument);
  EXPECT_THROW((void)t.at(24), std::invalid_argument);
}

// ============================================================================
// Copy, Clone, Move, and Aliasing Semantics
// ============================================================================

TEST(TensorTest, ShallowCopySharesStorageWhileCloneIsUnique) {
  Tensor<float> a({2, 2});
  a.set({1.0f, 2.0f, 3.0f, 4.0f});

  // Copy constructor creates shallow alias sharing Storage
  Tensor<float> b(a);
  EXPECT_FALSE(a.is_unique(b));
  EXPECT_TRUE(a.is_safe_aliasing(b));
  b.at({0, 0}) = 99.0f;
  EXPECT_FLOAT_EQ(a.at({0, 0}), 99.0f);

  // Clone creates deep independent copy
  Tensor<float> c = a.clone();
  EXPECT_TRUE(a.is_unique(c));
  EXPECT_TRUE(a.is_safe_aliasing(c));
  c.at({0, 0}) = -5.0f;
  EXPECT_FLOAT_EQ(a.at({0, 0}), 99.0f);
  EXPECT_FLOAT_EQ(c.at({0, 0}), -5.0f);
}

TEST(TensorTest, CopyFromCopiesValuesWithoutSharingStorage) {
  Tensor<int32_t> a({2, 3});
  a.set({1, 2, 3, 4, 5, 6});
  Tensor<int32_t> b({2, 3});
  b.copy_from(a);
  EXPECT_TRUE(a.is_unique(b));
  for (size_t i = 0; i < a.elements(); ++i) {
    EXPECT_EQ(b.at(i), a.at(i));
  }

  Tensor<int32_t> incompatible({3, 2});
  EXPECT_THROW(incompatible.copy_from(a), std::invalid_argument);
}

// ============================================================================
// Elementwise Arithmetic & Self-Aliasing (__restrict__ safety)
// ============================================================================

TEST(TensorTest, ElementwiseTensorArithmeticFloat) {
  Tensor<float> a({2, 3});
  Tensor<float> b({2, 3});
  a.set({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  b.set({2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f});

  Tensor<float> sum = a + b;
  Tensor<float> diff = b - a;
  Tensor<float> prod = a * b;
  Tensor<float> quot = b / a;

  for (size_t i = 0; i < a.elements(); ++i) {
    EXPECT_FLOAT_EQ(sum.at(i), a.at(i) + b.at(i));
    EXPECT_FLOAT_EQ(diff.at(i), b.at(i) - a.at(i));
    EXPECT_FLOAT_EQ(prod.at(i), a.at(i) * b.at(i));
    EXPECT_FLOAT_EQ(quot.at(i), b.at(i) / a.at(i));
  }
}

TEST(TensorTest, ElementwiseScalarArithmetic) {
  Tensor<int32_t> a({2, 2});
  a.set({10, 20, 30, 40});

  Tensor<int32_t> added = a + 5;
  EXPECT_EQ(added.at({0, 0}), 15);
  EXPECT_EQ(added.at({1, 1}), 45);

  Tensor<int32_t> subbed = a - 5;
  EXPECT_EQ(subbed.at({0, 0}), 5);
  EXPECT_EQ(subbed.at({1, 1}), 35);

  Tensor<int32_t> mult = a * 3;
  EXPECT_EQ(mult.at({0, 0}), 30);
  EXPECT_EQ(mult.at({1, 1}), 120);

  Tensor<int32_t> div = a / 10;
  EXPECT_EQ(div.at({0, 0}), 1);
  EXPECT_EQ(div.at({1, 1}), 4);
}

TEST(TensorTest, SelfAliasingInPlaceOperationsAreSafe) {
  Tensor<int32_t> a({2, 3});
  a.set({1, 2, 3, 4, 5, 6});

  // Self-addition: a += a (exercises non-__restrict__ path in uniform_op)
  a += a;
  EXPECT_EQ(a.at({0, 0}), 2);
  EXPECT_EQ(a.at({0, 1}), 4);
  EXPECT_EQ(a.at({1, 2}), 12);

  // Self-multiplication: a *= a
  a *= a;
  EXPECT_EQ(a.at({0, 0}), 4);
  EXPECT_EQ(a.at({0, 1}), 16);
  EXPECT_EQ(a.at({1, 2}), 144);

  // Self-division: a /= a
  a /= a;
  for (size_t i = 0; i < a.elements(); ++i) {
    EXPECT_EQ(a.at(i), 1);
  }

  // Self-subtraction: a -= a
  a -= a;
  for (size_t i = 0; i < a.elements(); ++i) {
    EXPECT_EQ(a.at(i), 0);
  }
}

TEST(TensorTest, TransposedSelfAliasingClonesSafely) {
  // Create a 2x2 tensor and a shallow copy that is transposed
  Tensor<int32_t> a({2, 2});
  a.set({1, 2, 3, 4});
  Tensor<int32_t> a_transposed(a);
  a_transposed.transpose();

  // a and a_transposed share Storage with different strides -> !is_safe_aliasing
  EXPECT_FALSE(a.is_safe_aliasing(a_transposed));

  // uniform_op clones destination safely before computing a + a^T
  a += a_transposed;
  EXPECT_EQ(a.at({0, 0}), 2); // 1 + 1
  EXPECT_EQ(a.at({0, 1}), 5); // 2 + 3
  EXPECT_EQ(a.at({1, 0}), 5); // 3 + 2
  EXPECT_EQ(a.at({1, 1}), 8); // 4 + 4
}

TEST(TensorTest, SignedAndUnsignedOverflowAndDivideByZeroThrows) {
  Tensor<int8_t> s1({2, 2});
  s1.fill(100);
  Tensor<int8_t> s2({2, 2});
  s2.fill(50);
  EXPECT_THROW((void)(s1 + s2), std::overflow_error);
  EXPECT_THROW(s1 *= static_cast<int8_t>(2), std::overflow_error);

  Tensor<uint8_t> u1({2, 2});
  u1.fill(10);
  Tensor<uint8_t> u2({2, 2});
  u2.fill(20);
  EXPECT_THROW((void)(u1 - u2), std::overflow_error);

  Tensor<int32_t> z({2, 2});
  z.fill(0);
  Tensor<int32_t> numer({2, 2});
  numer.fill(5);
  EXPECT_THROW((void)(numer / z), std::runtime_error);
  EXPECT_THROW(numer /= 0, std::runtime_error);

  // INT_MIN / -1 signed division overflow
  Tensor<int32_t> min_tensor({2, 2});
  min_tensor.fill(std::numeric_limits<int32_t>::min());
  EXPECT_THROW(min_tensor /= -1, std::overflow_error);
}

// ============================================================================
// Reductions: sum, min, max (including all-negative values)
// ============================================================================

TEST(TensorTest, SetWrongElementCountThrows) {
  Tensor<int32_t> t({2, 2});
  EXPECT_THROW(t.set({1, 2}), std::invalid_argument);
}

TEST(TensorTest, SumTotalAndPerDimension) {
  Tensor<int32_t> t({2, 2});
  t.set({1, 2, 3, 4});
  EXPECT_EQ(t.sum(), 10);
  // Row sums (dim = 0)
  EXPECT_EQ(t.sum(0, 0), 3);
  EXPECT_EQ(t.sum(0, 1), 7);
  // Column sums (dim = 1)
  EXPECT_EQ(t.sum(1, 0), 4);
  EXPECT_EQ(t.sum(1, 1), 6);
}

TEST(TensorTest, SumNonSquareMatrixRowsAndColumns) {
  Tensor<int32_t> t({2, 3});
  t.set({1, 2, 3,
      4, 5, 6});
  // Row sums (dim = 0)
  EXPECT_EQ(t.sum(0, 0), 6);
  EXPECT_EQ(t.sum(0, 1), 15);
  // Column sums (dim = 1) on non-square [2, 3] matrix
  EXPECT_EQ(t.sum(1, 0), 5);
  EXPECT_EQ(t.sum(1, 1), 7);
  EXPECT_EQ(t.sum(1, 2), 9);
}

TEST(TensorTest, MaxAndMinWithAllNegativeValues) {
  Tensor<float> t({2, 3});
  t.set({-10.0f, -3.5f, -25.0f, -1.0f, -50.0f, -2.0f});

  EXPECT_FLOAT_EQ(t.max(), -1.0f);
  EXPECT_FLOAT_EQ(t.min(), -50.0f);

  // Row-wise max (dim = 0, squeezed = true) -> validates all-negative row reset fix
  Tensor<float> row_max_sq = t.max(0, true);
  EXPECT_EQ(row_max_sq.shape(), std::vector<size_t>({2}));
  EXPECT_FLOAT_EQ(row_max_sq.at({0}), -3.5f);
  EXPECT_FLOAT_EQ(row_max_sq.at({1}), -1.0f);

  // Row-wise max (dim = 0, squeezed = false)
  Tensor<float> row_max_unsq = t.max(0, false);
  EXPECT_EQ(row_max_unsq.shape(), std::vector<size_t>({2, 3}));
  for (size_t j = 0; j < 3; ++j) {
    EXPECT_FLOAT_EQ(row_max_unsq.at({0, j}), -3.5f);
    EXPECT_FLOAT_EQ(row_max_unsq.at({1, j}), -1.0f);
  }

  // Col-wise max (dim = 1, squeezed = true)
  Tensor<float> col_max_sq = t.max(1, true);
  EXPECT_EQ(col_max_sq.shape(), std::vector<size_t>({3}));
  EXPECT_FLOAT_EQ(col_max_sq.at({0}), -1.0f);
  EXPECT_FLOAT_EQ(col_max_sq.at({1}), -3.5f);
  EXPECT_FLOAT_EQ(col_max_sq.at({2}), -2.0f);
}

// ============================================================================
// Softmax, Masks, and Dropout
// ============================================================================

TEST(TensorTest, SoftmaxRowWiseSumsToOneAndIsShiftInvariant) {
  Tensor<float> logits({2, 4});
  logits.set({1.0f, 2.0f, 3.0f, 4.0f, 1001.0f, 1002.0f, 1003.0f, 1004.0f});
  logits.softmax(0);

  // Each row should sum to 1.0f
  EXPECT_NEAR(logits.sum(0, 0), 1.0f, 1e-5f);
  EXPECT_NEAR(logits.sum(0, 1), 1.0f, 1e-5f);

  // Row 1 was shifted by +1000 relative to Row 0, so probabilities must match
  for (size_t j = 0; j < 4; ++j) {
    EXPECT_NEAR(logits.at({0, j}), logits.at({1, j}), 1e-5f);
    // Probabilities must be strictly increasing along the row
    if (j > 0) {
      EXPECT_GT(logits.at({0, j}), logits.at({0, j - 1}));
    }
  }
}

TEST(TensorTest, SoftmaxColumnWiseSumsToOne) {
  Tensor<float> logits({2, 3});
  logits.set({1.0f, 10.0f, -5.0f,
        2.0f, 10.0f, -3.0f});
  logits.softmax(1);

  for (size_t col = 0; col < 3; ++col) {
    EXPECT_NEAR(logits.sum(1, col), 1.0f, 1e-5f);
  }
  // Column 1 had equal logits (10.0f, 10.0f), so each entry must be 0.5f
  EXPECT_NEAR(logits.at({0, 1}), 0.5f, 1e-5f);
  EXPECT_NEAR(logits.at({1, 1}), 0.5f, 1e-5f);
}

TEST(TensorTest, TriangularAndCausalMasks) {
  Tensor<float> upper({3, 3});
  upper.tri(CausalMaskType::UPPER);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (j >= i) {
        EXPECT_FLOAT_EQ(upper.at({i, j}), 1.0f);
      } else {
        EXPECT_FLOAT_EQ(upper.at({i, j}), 0.0f);
      }
    }
  }

  Tensor<float> lower({3, 3});
  lower.tri(CausalMaskType::LOWER);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (j <= i) {
        EXPECT_FLOAT_EQ(lower.at({i, j}), 1.0f);
      } else {
        EXPECT_FLOAT_EQ(lower.at({i, j}), 0.0f);
      }
    }
  }

  Tensor<float> ninf({3, 3});
  ninf.fill(1.0f);
  ninf.ninf_tri(CausalMaskType::UPPER);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (j >= i) {
        EXPECT_TRUE(std::isinf(ninf.at({i, j})) && ninf.at({i, j}) < 0);
      } else {
        EXPECT_FLOAT_EQ(ninf.at({i, j}), 1.0f);
      }
    }
  }

  Tensor<float> non_square({2, 3});
  EXPECT_THROW(non_square.tri(CausalMaskType::LOWER), std::invalid_argument);
  EXPECT_THROW(non_square.apply_mask(CausalMaskType::UPPER), std::invalid_argument);
  EXPECT_THROW(non_square.ninf_tri(CausalMaskType::UPPER), std::invalid_argument);
}

TEST(TensorTest, ApplyDropoutValidatesBoundsAndScalesSurvivingElements) {
  Tensor<float> t({10, 10});
  t.fill(2.0f);

  // Invalid dropout bounds must throw
  EXPECT_THROW(t.apply_dropout(-0.1f), std::invalid_argument);
  EXPECT_THROW(t.apply_dropout(1.0f), std::invalid_argument);
  EXPECT_THROW(t.apply_dropout(1.5f), std::invalid_argument);

  // Zero dropout is a no-op
  t.apply_dropout(0.0f);
  EXPECT_FLOAT_EQ(t.sum(), 200.0f);

  // 25% dropout on 100 elements drops exactly 25 elements to 0.0f
  // and scales the remaining 75 elements by 1 / (1 - 0.25) = 4/3
  const float p = 0.25f;
  const float expected_surviving_val = 2.0f * (1.0f / (1.0f - p));
  t.apply_dropout(p);

  size_t zero_count = 0;
  size_t scaled_count = 0;
  for (size_t i = 0; i < t.elements(); ++i) {
    if (t.at(i) == 0.0f) {
      ++zero_count;
    } else {
      EXPECT_NEAR(t.at(i), expected_surviving_val, 1e-5f);
      ++scaled_count;
    }
  }
  EXPECT_EQ(zero_count, 25U);
  EXPECT_EQ(scaled_count, 75U);
  // Because exact fraction p is dropped and rest scaled by 1/(1-p), total sum is preserved!
  EXPECT_NEAR(t.sum(), 200.0f, 1e-3f);
}

// ============================================================================
// Tensor::slice, Tensor::remat, 2D Strided Ops, and Corner Cases
// ============================================================================

TEST(TensorTest, TensorSlice2DRowAndColSlicesShareStorage) {
  Tensor<int32_t> base({4, 5});
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      base.at({i, j}) = static_cast<int32_t>(i * 10 + j);
    }
  }

  // Slice rows [1, 3) and cols [1, 4) -> shape [2, 3], stride {5, 1}, offset 6
  Tensor<int32_t> sub = base.slice({0, 1}, {{1, 3}, {1, 4}}, {});
  EXPECT_EQ(sub.rank(), 2U);
  EXPECT_EQ(sub.shape(), std::vector<size_t>({2, 3}));
  EXPECT_EQ(sub.stride(), std::vector<size_t>({5, 1}));
  EXPECT_EQ(sub.offset(), 6U);
  EXPECT_FALSE(sub.contiguous());

  EXPECT_EQ(sub.at({0, 0}), 11);
  EXPECT_EQ(sub.at({0, 2}), 13);
  EXPECT_EQ(sub.at({1, 0}), 21);
  EXPECT_EQ(sub.at({1, 2}), 23);
  // Linear indexing on non-contiguous slice
  EXPECT_EQ(sub.at(0), 11);
  EXPECT_EQ(sub.at(2), 13);
  EXPECT_EQ(sub.at(3), 21);
  EXPECT_EQ(sub.at(5), 23);

  // Mutating the slice mutates the underlying base Tensor
  sub.at({1, 1}) = 777;
  EXPECT_EQ(base.at({2, 2}), 777);
}

TEST(TensorTest, TensorSlice3DReduceTo2DAnd1D) {
  Tensor<int32_t> t3({2, 3, 4});
  for (size_t b = 0; b < 2; ++b) {
    for (size_t r = 0; r < 3; ++r) {
      for (size_t c = 0; c < 4; ++c) {
        t3.at({b, r, c}) = static_cast<int32_t>(b * 100 + r * 10 + c);
      }
    }
  }

  // Fix dim 0 = 1, slice dim 1 in [1, 3) and dim 2 in [1, 4) -> 2D [2, 3]
  Tensor<int32_t> s2d = t3.slice({1, 2}, {{1, 3}, {1, 4}}, {{0, 1}});
  EXPECT_EQ(s2d.rank(), 2U);
  EXPECT_EQ(s2d.shape(), std::vector<size_t>({2, 3}));
  EXPECT_FALSE(s2d.contiguous());
  EXPECT_EQ(s2d.at({0, 0}), 111);
  EXPECT_EQ(s2d.at({0, 2}), 113);
  EXPECT_EQ(s2d.at({1, 0}), 121);
  EXPECT_EQ(s2d.at({1, 2}), 123);

  // Fix dim 0 = 1, dim 1 = 2, slice dim 2 in [1, 3) -> 1D [2]
  Tensor<int32_t> s1d = t3.slice({2}, {{1, 3}}, {{0, 1}, {1, 2}});
  EXPECT_EQ(s1d.rank(), 1U);
  EXPECT_EQ(s1d.shape(), std::vector<size_t>({2}));
  EXPECT_EQ(s1d.at({0}), 121);
  EXPECT_EQ(s1d.at({1}), 122);
}

TEST(TensorTest, TensorSliceNestedSubSliceAccumulatesOffset) {
  Tensor<int32_t> base({6, 6});
  for (size_t i = 0; i < 6; ++i) {
    for (size_t j = 0; j < 6; ++j) {
      base.at({i, j}) = static_cast<int32_t>(i * 10 + j);
    }
  }

  // First slice: rows [1, 5), cols [1, 5) -> [4, 4] starting at (1, 1), offset = 7
  Tensor<int32_t> s1 = base.slice({0, 1}, {{1, 5}, {1, 5}}, {});
  EXPECT_EQ(s1.offset(), 7U);

  // Nested slice of s1: rows [1, 3), cols [2, 4) -> [2, 2] starting at (2, 3) in base, offset = 15
  Tensor<int32_t> s2 = s1.slice({0, 1}, {{1, 3}, {2, 4}}, {});
  EXPECT_EQ(s2.shape(), std::vector<size_t>({2, 2}));
  EXPECT_EQ(s2.offset(), 15U);
  EXPECT_EQ(s2.at({0, 0}), 23);
  EXPECT_EQ(s2.at({0, 1}), 24);
  EXPECT_EQ(s2.at({1, 0}), 33);
  EXPECT_EQ(s2.at({1, 1}), 34);
}

TEST(TensorTest, UniformOpOn2DRowContiguousSlicesExercises2DStridedKernels) {
  // Create two [3, 6] tensors and take column slices [3, 2] with stride {6, 1}
  // This exercises _safe_2d_tensor_{add,sub,mul,div}_contiguous_v and _contiguous
  Tensor<float> base_a({3, 6});
  Tensor<float> base_b({3, 6});
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 6; ++j) {
      base_a.at({i, j}) = static_cast<float>((i + 1) * 10 + j);
      base_b.at({i, j}) = static_cast<float>((i + 1) * 2);
    }
  }

  Tensor<float> slice_a = base_a.slice({0, 1}, {{0, 0}, {1, 3}}, {});
  Tensor<float> slice_b = base_b.slice({0, 1}, {{0, 0}, {2, 4}}, {});
  ASSERT_FALSE(slice_a.contiguous());
  ASSERT_EQ(slice_a.dim_stride(1), 1U);

  Tensor<float> sum = slice_a + slice_b;
  Tensor<float> diff = slice_a - slice_b;
  Tensor<float> prod = slice_a * slice_b;
  Tensor<float> quot = slice_a / slice_b;

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      EXPECT_FLOAT_EQ(sum.at({i, j}), slice_a.at({i, j}) + slice_b.at({i, j}));
      EXPECT_FLOAT_EQ(diff.at({i, j}), slice_a.at({i, j}) - slice_b.at({i, j}));
      EXPECT_FLOAT_EQ(prod.at({i, j}), slice_a.at({i, j}) * slice_b.at({i, j}));
      EXPECT_FLOAT_EQ(quot.at({i, j}), slice_a.at({i, j}) / slice_b.at({i, j}));
    }
  }

  // In-place self-aliasing on 2D strided slice (exercises non-__restrict__ _safe_2d_tensor_*_contiguous)
  Tensor<int32_t> int_base({3, 4});
  for (size_t i = 0; i < 12; ++i) {
    int_base.at(i) = static_cast<int32_t>(i + 1);
  }
  Tensor<int32_t> int_slice = int_base.slice({0, 1}, {{0, 0}, {1, 3}}, {});
  int_slice += int_slice;
  EXPECT_EQ(int_slice.at({0, 0}), 4);   // (1+1) * 2
  EXPECT_EQ(int_slice.at({2, 1}), 22);  // (10+1) * 2
  EXPECT_EQ(int_base.at({0, 1}), 4);
  EXPECT_EQ(int_base.at({2, 2}), 22);

  // Scalar ops on 2D strided slice
  int_slice *= 3;
  EXPECT_EQ(int_slice.at({0, 0}), 12);
  int_slice /= 2;
  EXPECT_EQ(int_slice.at({0, 0}), 6);
}

TEST(TensorTest, RematMakesNonContiguousSliceAndTransposeContiguousAndUnique) {
  Tensor<int32_t> base({3, 4});
  for (size_t i = 0; i < 12; ++i) {
    base.at(i) = static_cast<int32_t>(i + 1);
  }

  // Column slice: shape [3, 2], non-contiguous
  Tensor<int32_t> col_slice = base.slice({0, 1}, {{0, 0}, {1, 3}}, {});
  EXPECT_FALSE(col_slice.contiguous());
  EXPECT_FALSE(col_slice.is_unique(base));

  col_slice.remat();
  EXPECT_TRUE(col_slice.contiguous());
  EXPECT_EQ(col_slice.offset(), 0U);
  EXPECT_EQ(col_slice.shape(), std::vector<size_t>({3, 2}));
  EXPECT_EQ(col_slice.stride(), std::vector<size_t>({2, 1}));
  EXPECT_TRUE(col_slice.is_unique(base));
  EXPECT_EQ(col_slice.at({0, 0}), 2);
  EXPECT_EQ(col_slice.at({0, 1}), 3);
  EXPECT_EQ(col_slice.at({2, 0}), 10);
  EXPECT_EQ(col_slice.at({2, 1}), 11);

  // Mutating rematerialized tensor no longer affects base
  col_slice.at({0, 0}) = 999;
  EXPECT_EQ(base.at({0, 1}), 2);

  // Transpose then remat
  Tensor<int32_t> t({2, 3});
  t.set({1, 2, 3, 4, 5, 6});
  t.transpose();
  EXPECT_FALSE(t.contiguous());
  t.remat();
  EXPECT_TRUE(t.contiguous());
  EXPECT_EQ(t.shape(), std::vector<size_t>({3, 2}));
  EXPECT_EQ(t.stride(), std::vector<size_t>({2, 1}));
  EXPECT_EQ(t.at({0, 0}), 1);
  EXPECT_EQ(t.at({0, 1}), 4);
  EXPECT_EQ(t.at({2, 0}), 3);
  EXPECT_EQ(t.at({2, 1}), 6);
}

TEST(TensorTest, ReductionsAndSoftmaxOnNonContiguousSlice) {
  Tensor<float> base({3, 5});
  base.set({
    -100.0f, -2.0f, -4.0f, -6.0f, -100.0f,
    -100.0f, -1.0f, -9.0f, -3.0f, -100.0f,
    -100.0f, -8.0f, -5.0f, -7.0f, -100.0f
  });

  // Non-contiguous [3, 3] slice of middle 3 columns
  Tensor<float> sub = base.slice({0, 1}, {{0, 0}, {1, 4}}, {});
  ASSERT_FALSE(sub.contiguous());

  EXPECT_FLOAT_EQ(sub.max(), -1.0f);
  EXPECT_FLOAT_EQ(sub.min(), -9.0f);
  EXPECT_FLOAT_EQ(sub.sum(), -45.0f);

  Tensor<float> row_max = sub.max(0, true);
  EXPECT_FLOAT_EQ(row_max.at({0}), -2.0f);
  EXPECT_FLOAT_EQ(row_max.at({1}), -1.0f);
  EXPECT_FLOAT_EQ(row_max.at({2}), -5.0f);

  Tensor<float> col_max = sub.max(1, true);
  EXPECT_FLOAT_EQ(col_max.at({0}), -1.0f);
  EXPECT_FLOAT_EQ(col_max.at({1}), -4.0f);
  EXPECT_FLOAT_EQ(col_max.at({2}), -3.0f);

  sub.softmax(0);
  for (size_t r = 0; r < 3; ++r) {
    EXPECT_NEAR(sub.sum(0, r), 1.0f, 1e-5f);
  }
}

TEST(TensorTest, InfoOnRank0ScalarDoesNotCorruptBracket) {
  Tensor<float> scalar({});
  scalar.at({}) = 1.0f;
  std::string info_str = scalar.info();
  // Should contain "Tensor: []" rather than erasing " [" into "Tensor:]"
  EXPECT_NE(info_str.find("Tensor: []"), std::string::npos) << "Actual info(): " << info_str;
}

TEST(TensorTest, ApplyMaskUpperZeroesFutureTokensAboveDiagonalAndPreservesSelfAndPast) {
  // In causal self-attention, token i can attend to j <= i (lower triangle + diagonal),
  // and future tokens j > i (strictly upper triangle) are masked out.
  Tensor<float> scores({3, 3});
  scores.fill(1.0f);
  scores.apply_mask(CausalMaskType::UPPER);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (j <= i) {
        EXPECT_FLOAT_EQ(scores.at({i, j}), 1.0f)
            << "Past/current token (" << i << ", " << j << ") should not be zeroed";
      } else {
        EXPECT_FLOAT_EQ(scores.at({i, j}), 0.0f)
            << "Future token (" << i << ", " << j << ") should be masked to 0";
      }
    }
  }
}

TEST(TensorTest, SliceReorderingDimsWithoutFiltersTransposesOrThrows) {
  Tensor<int32_t> t({2, 3});
  t.set({1, 2, 3, 4, 5, 6});
  // Passing dims = {1, 0} with empty filters requests axis 1 then axis 0;
  // it must not silently return an un-transposed [2, 3] copy.
  Tensor<int32_t> s = t.slice({1, 0}, {}, {});
  EXPECT_EQ(s.shape(), std::vector<size_t>({3, 2}));
}

TEST(TensorTest, SliceWithOutOfBoundsOtherDimThrowsInsteadOfHeapOverflow) {
  Tensor<float> t({3, 4});
  t.fill(1.0f);
  // Passing dim_num = 5 (>= rank 2) in other_dims must throw, not write out-of-bounds
  EXPECT_THROW((void)t.slice({0}, {{0, 2}}, {{5, 0}}), std::invalid_argument);
}

} // namespace
