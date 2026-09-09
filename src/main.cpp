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
 * @version: 2026-09-08
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

/* Standard dependencies */
#include <exception>
#include <format>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

/* Local dependencies */
#include "include/Attention.hpp"
#include "include/BytePairEncoding.hpp"
#include "include/DataLoader.hpp"
#include "include/Log.hpp"
#include "include/Tensor.hpp"
#include "include/TensorSlice.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
int main() {

    using Log::Log_Priority;
    using Log::log_message;

    using Tensor_NS::Tensor;
    using Tensor_NS::CausalMaskType;
    using Tensor_NS::matmul;

    using TensorSlice_NS::TensorSlice;
    using TensorSlice_NS::VectorSliceConfig;
    using TensorSlice_NS::MatrixSliceConfig;
    using TensorSlice_NS::VectorSliceOrientation;
    using TensorSlice_NS::IndexType;

    using DataLoader_NS::DataLoader;

    using Attention_NS::CausalAttention;
    using Attention_NS::MultiHeadAttention;

    try {
        
        log_message(Log_Priority::INFO, "main", "Initializing tokenizer");
        auto tokenizer_ptr = std::make_shared<BytePairEncoding_NS::BytePairEncodingTokenizer>("./models/BPE.model");
        BytePairEncoding_NS::BytePairEncodingTokenizer& BPET = *tokenizer_ptr;
        log_message(Log_Priority::INFO, "main", std::format("Tokenizer initialized with vocab size {}", BPET.vocab_size()));

        // Define the embedding dimension and vocab size
        size_t vocab_size = BPET.vocab_size();
        size_t emb_dim = 256;

        // Create the embedding Matrix
        //Tensor<float> emb({vocab_size, emb_dim});
        std::shared_ptr<Tensor<float>> emb_ptr = std::make_shared<Tensor<float>>(std::initializer_list<size_t>{vocab_size, emb_dim});
        Tensor<float>& emb = *(emb_ptr);
        // Fill the Matrix with random values
        emb.random(-2.f, 2.f);

        // Create a data loader using a shared ptr to the tokenizer
        DataLoader d(tokenizer_ptr, 4, 8, 4);
        d.ingest("./data/the-verdict.txt");
        // Get the first input batch and then target batch (shifted by one token)
        const Tensor<size_t>& input_batch = d.next_input();
        const Tensor<size_t>& target_batch = d.next_target();
        log_message(Log_Priority::INFO, "main", std::format("Input batch info: {}\nTarget batch info: {}", input_batch.to_string(), target_batch.to_string()));

        // Work on basic self-attention without trainable weights
        // Define our query, which is the 2nd input token
        size_t query_token = input_batch.at({0, 1}); // First batch, second token
        // Lookup the embedding for the query_token
        VectorSliceConfig query_vsc(0, query_token, 1, {}, VectorSliceOrientation::ROW, {});
        TensorSlice<float> query_emb(emb_ptr, query_vsc);
        log_message(Log_Priority::INFO, "main", std::format("Querying for token id {}. Slice info: {}", query_token, query_emb.info()));

        // Instead of creating a single vector with one token's embeddings, get the embeddings of every token in the batch
        // Since we have a batch size of 4, size the vector properly
        std::vector<size_t> lookup_tokens(4, 0);
        for (size_t i = 0; i < lookup_tokens.size(); ++i) {
            // Get all token ids from batch 0
            lookup_tokens.at(i) = input_batch.at({0, i});
        }
        // Create the MatrixSliceConfig to get all tokens listed
        MatrixSliceConfig query_msc(0, IndexType::LIST, lookup_tokens, 1, {}, {});
        TensorSlice<float> query_mat(emb_ptr, query_msc);
        log_message(Log_Priority::INFO, "main", std::format("Lookup TensorSlice: {}", query_mat.info()));

        // Convert the TensorSlice to a Matrix
        Tensor<float> query_result = query_mat.to_tensor();
        Tensor<float> attn_scores = query_result.matmul_self(true);
        log_message(Log_Priority::INFO, "main", std::format("Attention scores for batch 0: {}", attn_scores.to_string()));

        // Apply softmax to get the attention weights
        attn_scores.softmax(0);
        log_message(Log_Priority::INFO, "main", std::format("Attention weights: {}\nRow 0 sum: {}", attn_scores.to_string(), attn_scores.sum(0, 0)));

        // Create the context Matrix
        //Tensor<float> context = attn_scores.matmul(query_result);
        Tensor<float> context = matmul(attn_scores, query_result);
        log_message(Log_Priority::INFO, "main", std::format("Context Matrix: {}", context.info()));

        // Create the QKV matrices
        Tensor<float> W_q({emb_dim, emb_dim});
        Tensor<float> W_k({emb_dim, emb_dim});
        Tensor<float> W_v({emb_dim, emb_dim});
        // Fill the matrices with random weights
        W_q.random(-2.f, 2.f);
        W_k.random(-2.f, 2.f);
        W_v.random(-2.f, 2.f);

        // Calculate all keys and values
        Tensor<float> queries = matmul(query_result, W_q);
        Tensor<float> keys = matmul(query_result, W_k);
        Tensor<float> values = matmul(query_result, W_v);
        log_message(Log_Priority::INFO, "main", std::format("Query Matrix: {}. Key Matrix: {}. Value Matrix: {}.", queries.info(), keys.info(), values.info()));

        // Transpose keys for using the matmul
        Tensor<float> keys_t = keys.transpose();
        Tensor<float> attn_scores_full = matmul(queries, keys_t);
        // Apply masking
        //attn_scores_full.ninf_tri(CausalMaskType::LOWER);
        // Create a triangular Matrix for masking as using ninf appears to cause strange behavior
        Tensor<float> lower_tri({attn_scores_full.rows(), attn_scores_full.cols()});
        lower_tri.tri(CausalMaskType::LOWER);
        attn_scores_full *= lower_tri;
        // Scale the attention scores by the squareroot of the embedding dimension
        attn_scores_full /= std::sqrtf(static_cast<float>(emb_dim));
        // Apply softmax on the attention scores
        attn_scores_full.softmax(0);
        // Apply dropout
        attn_scores_full.apply_dropout(0.1);
        // Calculate the context vector
        Tensor<float> context_full = matmul(attn_scores_full, values);
        log_message(Log_Priority::INFO, "main", std::format("Context Matrix: {}", context_full.info()));

        // Create a CausalAttention head
        CausalAttention<float> head(emb_dim, emb_dim, 0.1);
        Tensor<float> ca_result = head.forward(query_result);

        // Test matmul with TensorSlice
        MatrixSliceConfig emb_msc(0, IndexType::RANGE, std::vector<size_t>{0, 10}, 1, {0, 10}, {});
        TensorSlice<float> emb_tsc(emb_ptr, emb_msc);
        Tensor<float> mm_res = emb_tsc.matmul_self(true);
        log_message(Log_Priority::INFO, "main", std::format("Slice Matrix: {}", mm_res.to_string()));

        // Ensure Matrix can initiate matmul with TensorSlice
        Tensor<float> mm_res_b = TensorSlice_NS::matmul(mm_res, emb_tsc);
        log_message(Log_Priority::INFO, "main", std::format("Slice Matrix 2: {}", mm_res_b.to_string()));

        // Create MultiHeadAttention instance
        MultiHeadAttention<float> mha(emb_dim, emb_dim, 1024, 0.1, 8);

    } catch (const std::exception& e) {

        std::cout << "Exception: " << e.what() << "\n";
        return -1;
    }

    return 0;
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
