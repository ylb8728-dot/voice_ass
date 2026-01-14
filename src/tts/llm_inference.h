/**
 * @file llm_inference.h
 * @brief CosyVoice2 LLM自回归推理模块
 *
 * 实现基于ONNX的LLM推理，支持：
 * 1. 简化模式：单次前向传播（适用于短序列）
 * 2. 组件模式：使用导出的各个组件拼接推理
 * 3. Top-K采样
 */

#ifndef VOICE_ASSISTANT_LLM_INFERENCE_H
#define VOICE_ASSISTANT_LLM_INFERENCE_H

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <memory>
#include <random>

namespace voice_assistant {
namespace tts {

/**
 * @brief LLM推理配置
 */
struct LLMConfig {
    std::string llm_model_path;         // 简化版LLM模型路径
    std::string text_embedding_path;    // Text embedding模型
    std::string speech_embedding_path;  // Speech embedding模型
    std::string spk_affine_path;        // Speaker affine模型
    std::string decoder_path;           // Output decoder模型
    std::string embedding_weights_path; // LLM embedding weights (.npy)

    int llm_input_size = 896;
    int llm_output_size = 896;
    int speech_token_size = 6561;
    int spk_embed_dim = 192;

    // 采样参数
    float temperature = 0.8f;
    int top_k = 25;
    float top_p = 0.8f;

    // 生成长度比例
    float min_token_text_ratio = 2.0f;
    float max_token_text_ratio = 20.0f;

    bool use_gpu = true;
    int num_threads = 4;
};

/**
 * @brief LLM推理引擎
 */
class LLMInference {
public:
    explicit LLMInference(const LLMConfig& config);
    ~LLMInference();

    /**
     * @brief 初始化LLM推理引擎
     */
    bool initialize();

    /**
     * @brief 自回归生成speech tokens
     *
     * @param text_token 文本token序列
     * @param prompt_text_token Prompt文本token序列
     * @param prompt_speech_token Prompt语音token序列
     * @param embedding Speaker embedding
     * @return 生成的speech token序列
     */
    std::vector<int64_t> generate(
        const std::vector<int64_t>& text_token,
        const std::vector<int64_t>& prompt_text_token,
        const std::vector<int64_t>& prompt_speech_token,
        const std::vector<float>& embedding);

private:
    /**
     * @brief Top-K采样
     */
    int64_t sampleTopK(const std::vector<float>& logits, bool ignore_eos = false);

    /**
     * @brief Softmax函数
     */
    std::vector<float> softmax(const std::vector<float>& logits);

    /**
     * @brief 简化模式推理（一次性生成）
     */
    std::vector<int64_t> inferenceSimplified(
        const std::vector<float>& lm_input,
        int seq_len);

    /**
     * @brief 组件模式推理（使用分离的embedding和decoder）
     */
    std::vector<int64_t> inferenceComponents(
        const std::vector<int64_t>& text_token,
        const std::vector<int64_t>& prompt_text_token,
        const std::vector<int64_t>& prompt_speech_token,
        const std::vector<float>& embedding);

    /**
     * @brief 运行text embedding
     */
    std::vector<float> runTextEmbedding(const std::vector<int64_t>& tokens);

    /**
     * @brief 运行speech embedding
     */
    std::vector<float> runSpeechEmbedding(const std::vector<int64_t>& tokens);

    /**
     * @brief 运行speaker affine layer
     */
    std::vector<float> runSpkAffine(const std::vector<float>& embedding);

    /**
     * @brief 运行decoder (logits输出)
     */
    std::vector<float> runDecoder(const std::vector<float>& hidden_states);

    /**
     * @brief 构建LLM输入序列
     * 格式：[SOS] [embedding] [text] [TaskID] [prompt_speech]
     */
    std::vector<float> buildLLMInput(
        const std::vector<float>& text_embedding,
        const std::vector<float>& prompt_speech_embedding,
        const std::vector<float>& spk_embedding);

    LLMConfig config_;
    bool initialized_;

    // ONNX Runtime
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;

    // 模型sessions
    std::unique_ptr<Ort::Session> llm_session_;          // 简化版LLM
    std::unique_ptr<Ort::Session> text_emb_session_;     // Text embedding
    std::unique_ptr<Ort::Session> speech_emb_session_;   // Speech embedding
    std::unique_ptr<Ort::Session> spk_affine_session_;   // Speaker affine
    std::unique_ptr<Ort::Session> decoder_session_;      // Output decoder

    Ort::MemoryInfo memory_info_;

    // LLM特殊tokens和embeddings
    std::vector<float> sos_eos_emb_;   // SOS/EOS embedding
    std::vector<float> task_id_emb_;   // Task ID embedding
    std::vector<float> speech_embedding_weights_;  // Speech token embedding table

    // 随机数生成器（用于采样）
    std::mt19937 rng_;
};

} // namespace tts
} // namespace voice_assistant

#endif // VOICE_ASSISTANT_LLM_INFERENCE_H
