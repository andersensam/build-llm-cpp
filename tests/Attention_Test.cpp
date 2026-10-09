#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "CausalAttention.hpp"
#include "MultiHeadAttention.hpp"
#include "Tensor.hpp"

namespace {

using CausalAttention_NS::CausalAttention;
using MultiHeadAttention_NS::MultiHeadAttention;
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

TEST(AttentionTest, MultiHeadAttentionNonSquareEmbAndOutDimAndSingleToken) {
  // emb_dim (24) != out_dim (16), seq_len = 1
  MultiHeadAttention<float> mha(24, 16, 32, 0.0f, 4);
  Tensor<float> single_token({1, 24});
  single_token.random(-1.0f, 1.0f);

  Tensor<float> out = mha.forward(single_token);
  EXPECT_EQ(out.shape(), std::vector<size_t>({1, 16}));
  for (size_t i = 0; i < out.elements(); ++i) {
    EXPECT_TRUE(std::isfinite(out.at(i)));
  }
}

TEST(AttentionTest, CausalAttentionDoesNotLeakFutureTokensIntoPastOutputs) {
  // Fundamental autoregressive property: modifying token at position t_future (row 3)
  // must NOT change the attention output at earlier token positions t < 3 (rows 0, 1, 2).
  const size_t seq_len = 4;
  const size_t emb_dim = 8;
  const size_t out_dim = 8;

  CausalAttention<float> ca(emb_dim, out_dim, 0.0f);

  Tensor<float> input_a({seq_len, emb_dim});
  for (size_t i = 0; i < seq_len; ++i) {
    for (size_t j = 0; j < emb_dim; ++j) {
      input_a.at({i, j}) = static_cast<float>((i + 1) * 0.1f + (j + 1) * 0.05f);
    }
  }

  Tensor<float> input_b = input_a.clone();
  // Perturb ONLY the last token (row 3)
  for (size_t j = 0; j < emb_dim; ++j) {
    input_b.at({3, j}) = 50.0f + static_cast<float>(j);
  }

  Tensor<float> out_a = ca.forward(input_a);
  Tensor<float> out_b = ca.forward(input_b);

  // Outputs for tokens 0, 1, 2 must be invariant to changes in token 3
  for (size_t t = 0; t < 3; ++t) {
    for (size_t d = 0; d < out_dim; ++d) {
      EXPECT_NEAR(out_a.at({t, d}), out_b.at({t, d}), 1e-5f)
          << "Future token 3 leaked into causal attention output at token " << t << ", dim " << d;
    }
  }
}

TEST(AttentionTest, MultiHeadAttentionDoesNotLeakFutureTokensIntoPastOutputs) {
  const size_t seq_len = 4;
  const size_t emb_dim = 16;
  const size_t out_dim = 16;
  const size_t num_heads = 4;

  MultiHeadAttention<float> mha(emb_dim, out_dim, 32, 0.0f, num_heads);

  Tensor<float> input_a({seq_len, emb_dim});
  for (size_t i = 0; i < seq_len; ++i) {
    for (size_t j = 0; j < emb_dim; ++j) {
      input_a.at({i, j}) = static_cast<float>((i + 1) * 0.1f + (j + 1) * 0.05f);
    }
  }

  Tensor<float> input_b = input_a.clone();
  // Perturb ONLY the last token (row 3)
  for (size_t j = 0; j < emb_dim; ++j) {
    input_b.at({3, j}) = 50.0f + static_cast<float>(j);
  }

  Tensor<float> out_a = mha.forward(input_a);
  Tensor<float> out_b = mha.forward(input_b);

  // Outputs for tokens 0, 1, 2 must be invariant to changes in token 3
  for (size_t t = 0; t < 3; ++t) {
    for (size_t d = 0; d < out_dim; ++d) {
      EXPECT_NEAR(out_a.at({t, d}), out_b.at({t, d}), 1e-5f)
          << "Future token 3 leaked into MHA output at token " << t << ", dim " << d;
    }
  }
}

} // namespace
