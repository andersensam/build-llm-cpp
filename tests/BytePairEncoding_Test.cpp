#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "BytePairEncoding.hpp"

namespace {

using BytePairEncoding_NS::byte_vector_to_string;
using BytePairEncoding_NS::BytePairEncodingTokenizer;
using BytePairEncoding_NS::BytePositionInfo;
using BytePairEncoding_NS::create_tokens;
using BytePairEncoding_NS::get_num_packed_bytes;
using BytePairEncoding_NS::pack_bytes;
using BytePairEncoding_NS::string_to_byte_vector;
using BytePairEncoding_NS::token_vector_to_string;
using BytePairEncoding_NS::unpack_bytes;

TEST(BPETest, PackAndUnpackBytesRoundTrip1To4Bytes) {
  EXPECT_EQ(get_num_packed_bytes(0U), 0U);

  const std::string sample = "ABCD";
  std::vector<std::byte> bytes = string_to_byte_vector(sample);
  std::span<const std::byte> sp(bytes);

  for (size_t len = 1; len <= 4; ++len) {
    uint32_t packed = pack_bytes(sp.subspan(0, len));
    EXPECT_EQ(get_num_packed_bytes(packed), len);

    std::vector<std::byte> unpacked = unpack_bytes(packed);
    EXPECT_EQ(unpacked.size(), len);
    EXPECT_EQ(byte_vector_to_string(unpacked), sample.substr(0, len));
  }

  // Combining two 2-byte packed uint32_t tokens into one 4-byte token
  uint32_t ab = pack_bytes(sp.subspan(0, 2));
  uint32_t cd = pack_bytes(sp.subspan(2, 2));
  std::vector<uint32_t> pair_vec = {ab, cd};
  uint32_t abcd = pack_bytes(std::span<const uint32_t>(pair_vec));
  EXPECT_EQ(get_num_packed_bytes(abcd), 4U);
  EXPECT_EQ(byte_vector_to_string(unpack_bytes(abcd)), "ABCD");

  // Combining a 3-byte token and a 2-byte token (> 4 bytes) returns 0
  uint32_t abc = pack_bytes(sp.subspan(0, 3));
  std::vector<uint32_t> too_large = {abc, cd};
  EXPECT_EQ(pack_bytes(std::span<const uint32_t>(too_large)), 0U);
}

TEST(BPETest, StringAndTokenVectorConversionHelpers) {
  const std::string text = "Hello, BPE world!";
  std::vector<std::byte> bytes = string_to_byte_vector(text);
  EXPECT_EQ(bytes.size(), text.size());
  EXPECT_EQ(byte_vector_to_string(bytes), text);

  std::vector<size_t> toks = {10, 20, 30};
  EXPECT_EQ(token_vector_to_string(toks), "10 20 30");
}

TEST(BPETest, TokenizerUpdateVocabAndRoundTripTokenizeDetokenize) {
  BytePairEncodingTokenizer tokenizer;
  EXPECT_EQ(tokenizer.vocab_size(), 256U);

  const std::string corpus = "the cat in the hat sat on the mat";
  EXPECT_TRUE(tokenizer.update_vocabulary(corpus));
  EXPECT_GT(tokenizer.vocab_size(), 256U);

  // Updating with the exact same corpus a second time adds no new tokens
  EXPECT_FALSE(tokenizer.update_vocabulary(corpus));

  std::vector<size_t> token_ids = tokenizer.tokenize(corpus);
  // Merged multi-byte tokens must compress the sequence length below corpus.size()
  EXPECT_LT(token_ids.size(), corpus.size());

  std::string round_trip = tokenizer.detokenize_to_string(token_ids);
  EXPECT_EQ(round_trip, corpus);
}

TEST(BPETest, TokenizerExportAndReloadModelFilePreservesVocab) {
  BytePairEncodingTokenizer tokenizer;
  const std::string corpus = "learning byte pair encoding in modern c++";
  tokenizer.update_vocabulary(corpus);

  const std::filesystem::path temp_model =
      std::filesystem::temp_directory_path() / "bpe_unit_test_model.bin";

  ASSERT_TRUE(tokenizer.export_to_file(temp_model.string()));

  BytePairEncodingTokenizer loaded(temp_model.string());
  std::filesystem::remove(temp_model);

  EXPECT_EQ(loaded.vocab_size(), tokenizer.vocab_size());
  EXPECT_EQ(loaded.token_ids(), tokenizer.token_ids());

  std::vector<size_t> orig_tokens = tokenizer.tokenize(corpus);
  std::vector<size_t> loaded_tokens = loaded.tokenize(corpus);
  EXPECT_EQ(loaded_tokens, orig_tokens);
  EXPECT_EQ(loaded.detokenize_to_string(loaded_tokens), corpus);
}

TEST(BPETest, BytePositionInfoCopyConstructorPreservesFrequencyAndPositions) {
  BytePositionInfo original(0x6162U, 0, 1);
  original.add_position(4, 5);
  ASSERT_EQ(original.get_frequency(), 2U);

  BytePositionInfo copy(original);
  EXPECT_EQ(copy.get_byte_sequence(), 0x6162U);
  EXPECT_EQ(copy.get_frequency(), 2U);
  ASSERT_EQ(copy.get_positions().size(), 2U);
  EXPECT_EQ(copy.get_positions().at(0), std::make_pair(size_t{0}, size_t{1}));
  EXPECT_EQ(copy.get_positions().at(1), std::make_pair(size_t{4}, size_t{5}));
}

TEST(BPETest, CreateTokensOnOverlappingRepeatedCharactersForms4ByteToken) {
  // In "aaaaaaaa" (8 'a's), the first BPE pass should merge non-overlapping pairs into
  // four "aa" tokens, and the second pass should merge those into "aaaa" (4 bytes).
  // Currently, recording overlapping positions (0,1), (1,2), ..., (6,7) and erasing all
  // of them collapses all 8 'a's into a single 2-byte "aa" token!
  std::vector<std::byte> bytes = string_to_byte_vector("aaaaaaaa");
  std::vector<uint32_t> tokens = create_tokens(bytes);
  ASSERT_FALSE(tokens.empty());
  bool found_4byte_aaaa = false;
  for (uint32_t tok : tokens) {
    if (get_num_packed_bytes(tok) == 4U && byte_vector_to_string(unpack_bytes(tok)) == "aaaa") {
      found_4byte_aaaa = true;
    }
  }
  EXPECT_TRUE(found_4byte_aaaa);
}

TEST(BPETest, TokenVectorToStringOnEmptyVectorReturnsEmptyString) {
  std::vector<size_t> empty_tokens;
  EXPECT_NO_THROW({
    EXPECT_EQ(token_vector_to_string(empty_tokens), "");
  });
}

} // namespace
