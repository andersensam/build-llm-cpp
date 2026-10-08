#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "FeedForward.hpp"
#include "GELULayer.hpp"
#include "Layer.hpp"
#include "LinearLayer.hpp"
#include "NormalizationLayer.hpp"
#include "SliceConfig.hpp"
#include "Tensor.hpp"
#include "TensorSlice.hpp"

namespace {

using FeedForward_NS::FeedForward;
using GELULayer_NS::GELULayer;
using Layer_NS::Layer;
using LinearLayer_NS::LinearLayer;
using NormalizationLayer_NS::NormalizationLayer;
using SliceConfig_NS::IndexType;
using SliceConfig_NS::MatrixSliceConfig;
using Tensor_NS::Tensor;
using TensorSlice_NS::ListTensorSlice;

// ============================================================================
// LinearLayer Tests
// ============================================================================

TEST(LayerTest, LinearLayerValidationAndShapes) {
  LinearLayer<float> linear(8, 12);

  // Rank != 2 throws
  Tensor<float> rank1({8});
  Tensor<float> rank3({2, 8, 4});
  EXPECT_THROW(linear.forward(rank1), std::invalid_argument);
  EXPECT_THROW(linear.forward(rank3), std::invalid_argument);

  // Incompatible inner dimension throws
  Tensor<float> bad_in({4, 6});
  EXPECT_THROW(linear.forward(bad_in), std::invalid_argument);

  // Valid input produces expected output shape and matches destination overload
  Tensor<float> input({5, 8});
  input.random(-1.0f, 1.0f);

  Tensor<float> out = linear.forward(input);
  EXPECT_EQ(out.shape(), std::vector<size_t>({5, 12}));

  Tensor<float> dest({5, 12});
  linear.forward(input, dest);
  for (size_t i = 0; i < out.elements(); ++i) {
    EXPECT_FLOAT_EQ(out.at(i), dest.at(i));
  }

  // Incompatible dest shape throws
  Tensor<float> bad_dest({5, 8});
  EXPECT_THROW(linear.forward(input, bad_dest), std::invalid_argument);
}

TEST(LayerTest, LinearLayerSupportsSignedUnsignedAndInPlaceSquareForward) {
  LinearLayer<int32_t> signed_linear(4, 4);
  Tensor<int32_t> x_i32({3, 4});
  for (size_t i = 0; i < x_i32.elements(); ++i) {
    x_i32.at(i) = static_cast<int32_t>((i % 3) + 1);
  }
  Tensor<int32_t> expected_i32 = signed_linear.forward(x_i32);
  signed_linear.forward(x_i32, x_i32);
  for (size_t i = 0; i < x_i32.elements(); ++i) {
    EXPECT_EQ(x_i32.at(i), expected_i32.at(i));
  }

  LinearLayer<uint32_t> unsigned_linear(4, 3);
  Tensor<uint32_t> x_u32({2, 4});
  x_u32.fill(1);
  Tensor<uint32_t> out_u32 = unsigned_linear.forward(x_u32);
  EXPECT_EQ(out_u32.shape(), std::vector<size_t>({2, 3}));
}

// ============================================================================
// NormalizationLayer Tests
// ============================================================================

TEST(LayerTest, NormalizationLayerMatchesAnalyticalRowWiseLayerNorm) {
  NormalizationLayer<float> norm;
  Tensor<float> input({2, 4});
  // Row 0: [1, 2, 3, 4] -> mean = 2.5, var = 1.25
  input.at({0, 0}) = 1.0f;
  input.at({0, 1}) = 2.0f;
  input.at({0, 2}) = 3.0f;
  input.at({0, 3}) = 4.0f;
  // Row 1: [-2, 0, 2, 8] -> mean = 2.0, var = (16 + 4 + 0 + 36) / 4 = 14.0
  input.at({1, 0}) = -2.0f;
  input.at({1, 1}) = 0.0f;
  input.at({1, 2}) = 2.0f;
  input.at({1, 3}) = 8.0f;

  Tensor<float> out = norm.forward(input);
  EXPECT_EQ(out.shape(), std::vector<size_t>({2, 4}));

  constexpr float eps = 1e-5f;
  const float std_row0 = std::sqrt(1.25f + eps);
  for (size_t j = 0; j < 4; ++j) {
    const float expected = (input.at({0, j}) - 2.5f) / std_row0;
    EXPECT_NEAR(out.at({0, j}), expected, 1e-5f);
  }

  const float std_row1 = std::sqrt(14.0f + eps);
  for (size_t j = 0; j < 4; ++j) {
    const float expected = (input.at({1, j}) - 2.0f) / std_row1;
    EXPECT_NEAR(out.at({1, j}), expected, 1e-5f);
  }

  // Verify zero mean per row
  for (size_t i = 0; i < 2; ++i) {
    const float row_mean = out.sum(0, i) / 4.0f;
    EXPECT_NEAR(row_mean, 0.0f, 1e-5f);
  }
}

TEST(LayerTest, NormalizationLayerSupportsInPlaceAliasingAndListTensorSlice) {
  NormalizationLayer<float> norm;
  auto backing = std::make_shared<Tensor<float>>(std::initializer_list<size_t>{3, 4});
  for (size_t i = 0; i < backing->elements(); ++i) {
    backing->at(i) = static_cast<float>(i * 0.5f - 2.0f);
  }

  Tensor<float> expected = norm.forward(*backing);

  // In-place forward(x, x) must match out-of-place forward(x)
  Tensor<float> inplace = backing->clone();
  norm.forward(inplace, inplace);
  for (size_t i = 0; i < expected.elements(); ++i) {
    EXPECT_NEAR(inplace.at(i), expected.at(i), 1e-5f);
  }

  // Polymorphic ListTensorSlice input via AbstractTensor overload
  MatrixSliceConfig cfg(0, IndexType::LIST, {0, 2}, 1, {}, {});
  ListTensorSlice<float> slice_view(backing, cfg);

  Tensor<float> slice_dest({2, 4});
  norm.forward(slice_view, slice_dest);
  for (size_t j = 0; j < 4; ++j) {
    EXPECT_NEAR(slice_dest.at({0, j}), expected.at({0, j}), 1e-5f);
    EXPECT_NEAR(slice_dest.at({1, j}), expected.at({2, j}), 1e-5f);
  }

  // Validation checks
  Tensor<float> rank1({4});
  Tensor<float> bad_dest({3, 2});
  EXPECT_THROW(norm.forward(rank1), std::invalid_argument);
  EXPECT_THROW(norm.forward(*backing, bad_dest), std::invalid_argument);
}

// ============================================================================
// GELULayer Tests
// ============================================================================

TEST(LayerTest, GELULayerMatchesTanhApproximationFormulaAndInPlaceAliasing) {
  GELULayer<float> gelu;
  Tensor<float> input({2, 5});
  const std::vector<float> test_vals = {
      -3.0f, -1.0f, -0.5f, 0.0f, 0.5f,
      1.0f,  2.0f,  3.0f, -2.0f, 1.5f
  };
  for (size_t i = 0; i < test_vals.size(); ++i) {
    input.at(i) = test_vals[i];
  }

  Tensor<float> out = gelu.forward(input);
  EXPECT_EQ(out.shape(), std::vector<size_t>({2, 5}));

  const float kScale = std::sqrt(2.0f / std::numbers::pi_v<float>);
  for (size_t i = 0; i < test_vals.size(); ++i) {
    const float x = test_vals[i];
    const float expected = 0.5f * x * (1.0f + std::tanh(kScale * (x + 0.044715f * std::pow(x, 3.0f))));
    EXPECT_NEAR(out.at(i), expected, 1e-5f);
  }

  // Verify GELU(0.0f) == 0.0f strictly
  EXPECT_FLOAT_EQ(out.at({0, 3}), 0.0f);

  // Verify in-place forward(x, x) matches out-of-place forward(x)
  Tensor<float> inplace = input.clone();
  gelu.forward(inplace, inplace);
  for (size_t i = 0; i < out.elements(); ++i) {
    EXPECT_NEAR(inplace.at(i), out.at(i), 1e-5f);
  }

  // Rank validation
  Tensor<float> rank1({5});
  EXPECT_THROW(gelu.forward(rank1), std::invalid_argument);
  EXPECT_THROW(gelu.forward(rank1, inplace), std::invalid_argument);
}

// ============================================================================
// FeedForward Tests
// ============================================================================

TEST(LayerTest, FeedForwardValidationInPlaceAndRowIndependence) {
  const size_t emb_dim = 8;
  const size_t scale_dim = 4;
  FeedForward<float> ffn(emb_dim, scale_dim);

  // Validation checks
  Tensor<float> rank1({emb_dim});
  Tensor<float> bad_emb({3, emb_dim + 2});
  Tensor<float> valid_in({3, emb_dim});
  valid_in.random(-1.0f, 1.0f);
  Tensor<float> bad_dest({4, emb_dim});

  EXPECT_THROW(ffn.forward(rank1), std::invalid_argument);
  EXPECT_THROW(ffn.forward(bad_emb), std::invalid_argument);
  EXPECT_THROW(ffn.forward(valid_in, bad_dest), std::invalid_argument);

  // Returning vs destination vs in-place forward equivalence
  Tensor<float> out = ffn.forward(valid_in);
  EXPECT_EQ(out.shape(), std::vector<size_t>({3, emb_dim}));
  for (size_t i = 0; i < out.elements(); ++i) {
    EXPECT_TRUE(std::isfinite(out.at(i)));
  }

  Tensor<float> dest({3, emb_dim});
  ffn.forward(valid_in, dest);
  for (size_t i = 0; i < out.elements(); ++i) {
    EXPECT_FLOAT_EQ(out.at(i), dest.at(i));
  }

  Tensor<float> inplace = valid_in.clone();
  ffn.forward(inplace, inplace);
  for (size_t i = 0; i < out.elements(); ++i) {
    EXPECT_FLOAT_EQ(out.at(i), inplace.at(i));
  }

  // Row-wise position independence: perturbing row 2 leaves rows 0 and 1 identical
  Tensor<float> perturbed = valid_in.clone();
  for (size_t j = 0; j < emb_dim; ++j) {
    perturbed.at({2, j}) += 10.0f;
  }
  Tensor<float> perturbed_out = ffn.forward(perturbed);
  for (size_t row = 0; row < 2; ++row) {
    for (size_t col = 0; col < emb_dim; ++col) {
      EXPECT_NEAR(out.at({row, col}), perturbed_out.at({row, col}), 1e-5f);
    }
  }
}

TEST(LayerTest, PolymorphicLayerInterfacePipeline) {
  std::vector<std::unique_ptr<Layer<float>>> pipeline;
  pipeline.push_back(std::make_unique<NormalizationLayer<float>>());
  pipeline.push_back(std::make_unique<FeedForward<float>>(8, 4));
  pipeline.push_back(std::make_unique<NormalizationLayer<float>>());

  Tensor<float> x({4, 8});
  x.random(-1.0f, 1.0f);

  for (const auto& layer : pipeline) {
    x = layer->forward(x);
  }

  EXPECT_EQ(x.shape(), std::vector<size_t>({4, 8}));
  for (size_t i = 0; i < x.elements(); ++i) {
    EXPECT_TRUE(std::isfinite(x.at(i)));
  }
}

} // namespace
