#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "Attention.hpp"
#include "Tensor.hpp"

namespace {

using Attention_NS::CausalAttention;
using Attention_NS::MultiHeadAttention;
using Tensor_NS::Tensor;

// ============================================================================
// CausalAttention Tests
// ============================================================================

TEST(AttentionTest, CausalAttentionConstructorValidation) {
  EXPECT_THROW((CausalAttention<float>(16, 16, -0.1f)), std::invalid_argument);
  EXPECT_THROW((CausalAttention<float>(16, 16, 1.0f)), std::invalid_argument);
  EXPECT_NO_THROW((CausalAttention<float>(16, 16, 0.0f)));
  EXPECT_NO_THROW((CausalAttention<float>(16, 8, 0.2f)));
}

TEST(AttentionTest, CausalAttentionForwardProducesExpectedShapeAndFiniteValues) {
  const size_t seq_len = 6;
  const size_t emb_dim = 16;
  const size_t out_dim = 8;

  CausalAttention<float> ca(emb_dim, out_dim, 0.0f);
  Tensor<float> input({seq_len, emb_dim});
  input.random(-1.0f, 1.0f);

  Tensor<float> out = ca.forward(input);
  EXPECT_EQ(out.shape(), std::vector<size_t>({seq_len, out_dim}));
  for (size_t i = 0; i < out.elements(); ++i) {
    EXPECT_TRUE(std::isfinite(out.at(i)));
  }

  // With dropout = 0.0f, repeated forward passes on the same input are deterministic
  Tensor<float> out2 = ca.forward(input);
  for (size_t i = 0; i < out.elements(); ++i) {
    EXPECT_FLOAT_EQ(out.at(i), out2.at(i));
  }
}

// ============================================================================
// AttentionHead and MultiHeadAttention Tests
// ============================================================================

TEST(AttentionTest, AttentionHeadShardsWeightMatricesCorrectly) {
  Tensor<float> wq({12, 16});

  Tensor<float> head = wq.slice({0, 1}, {{4, 8}, {0, 0}}, {});
  EXPECT_EQ(head.shape(), std::vector<size_t>({4, 16}));
}

TEST(AttentionTest, MultiHeadAttentionConstructorValidation) {
  // Invalid dropout
  EXPECT_THROW((MultiHeadAttention<float>(16, 16, 32, -0.1f, 4)), std::invalid_argument);
  EXPECT_THROW((MultiHeadAttention<float>(16, 16, 32, 1.0f, 4)), std::invalid_argument);
  // Output dim not divisible by num_heads
  EXPECT_THROW((MultiHeadAttention<float>(16, 15, 32, 0.0f, 4)), std::invalid_argument);
  // Valid configuration
  EXPECT_NO_THROW((MultiHeadAttention<float>(16, 16, 32, 0.0f, 4)));
}

TEST(AttentionTest, MultiHeadAttentionForwardProducesExpectedShapeAndFiniteValues) {
  const size_t seq_len = 4;
  const size_t emb_dim = 32;
  const size_t out_dim = 32;
  const size_t num_heads = 4;

  MultiHeadAttention<float> mha(emb_dim, out_dim, 64, 0.0f, num_heads);
  Tensor<float> input({seq_len, emb_dim});
  input.random(-1.0f, 1.0f);

  Tensor<float> out = mha.forward(input);
  EXPECT_EQ(out.shape(), std::vector<size_t>({seq_len, out_dim}));
  for (size_t i = 0; i < out.elements(); ++i) {
    EXPECT_TRUE(std::isfinite(out.at(i)));
  }

  // Deterministic when dropout == 0.0f
  Tensor<float> out2 = mha.forward(input);
  for (size_t i = 0; i < out.elements(); ++i) {
    EXPECT_FLOAT_EQ(out.at(i), out2.at(i));
  }
}

} // namespace
