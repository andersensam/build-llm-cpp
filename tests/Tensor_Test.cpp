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

} // namespace
