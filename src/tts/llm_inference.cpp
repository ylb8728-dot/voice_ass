#include "llm_inference.h"
#include "../utils/logger.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>

namespace voice_assistant {
namespace tts {

LLMInference::LLMInference(const LLMConfig& config)
    : config_(config)
    , initialized_(false)
    , env_(nullptr)
    , session_options_(nullptr)
    , memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    , rng_(std::random_device{}())
{
}

LLMInference::~LLMInference() {
}

bool LLMInference::initialize() {
    LOG_INFO("Initializing LLM Inference Engine...");

    try {
        // 初始化ONNX Runtime环境
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "LLMInference");

        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(config_.num_threads);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (config_.use_gpu) {
#ifdef USE_CUDA
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            session_options_->AppendExecutionProvider_CUDA(cuda_options);
            LOG_INFO("Using CUDA for LLM inference");
#else
            LOG_WARN("CUDA not available, using CPU");
            config_.use_gpu = false;
#endif
        }

        // 加载模型（根据可用的模型文件）
        // 优先尝试加载组件模式的模型
        bool use_components = false;

        if (!config_.text_embedding_path.empty() &&
            !config_.speech_embedding_path.empty() &&
            !config_.decoder_path.empty()) {

            LOG_INFO("Loading LLM component models...");

            // Text embedding
            text_emb_session_ = std::make_unique<Ort::Session>(
                *env_, config_.text_embedding_path.c_str(), *session_options_);

            // Speech embedding
            speech_emb_session_ = std::make_unique<Ort::Session>(
                *env_, config_.speech_embedding_path.c_str(), *session_options_);

            // Speaker affine
            if (!config_.spk_affine_path.empty()) {
                spk_affine_session_ = std::make_unique<Ort::Session>(
                    *env_, config_.spk_affine_path.c_str(), *session_options_);
            }

            // Decoder
            decoder_session_ = std::make_unique<Ort::Session>(
                *env_, config_.decoder_path.c_str(), *session_options_);

            use_components = true;
            LOG_INFO("Component models loaded");

        } else if (!config_.llm_model_path.empty()) {

            LOG_INFO("Loading simplified LLM model...");
            llm_session_ = std::make_unique<Ort::Session>(
                *env_, config_.llm_model_path.c_str(), *session_options_);
            LOG_INFO("Simplified LLM model loaded");

        } else {
            LOG_ERROR("No valid LLM model path provided");
            return false;
        }

        // 加载LLM embedding weights (用于SOS/EOS和TaskID)
        if (!config_.embedding_weights_path.empty()) {
            // TODO: 实现.npy文件加载
            LOG_WARN("LLM embedding weights loading not implemented");

            // 占位：使用随机初始化
            sos_eos_emb_.resize(config_.llm_input_size, 0.0f);
            task_id_emb_.resize(config_.llm_input_size, 0.01f);
        }

        initialized_ = true;
        LOG_INFO("LLM Inference Engine initialized");
        return true;

    } catch (const Ort::Exception& e) {
        LOG_ERROR("ONNX Runtime error: ", e.what());
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR("Initialization error: ", e.what());
        return false;
    }
}

std::vector<int64_t> LLMInference::generate(
    const std::vector<int64_t>& text_token,
    const std::vector<int64_t>& prompt_text_token,
    const std::vector<int64_t>& prompt_speech_token,
    const std::vector<float>& embedding) {

    if (!initialized_) {
        LOG_ERROR("LLM not initialized");
        return std::vector<int64_t>();
    }

    LOG_INFO("Generating speech tokens...");
    LOG_INFO("  Text tokens: ", text_token.size());
    LOG_INFO("  Prompt text tokens: ", prompt_text_token.size());
    LOG_INFO("  Prompt speech tokens: ", prompt_speech_token.size());

    // 根据可用的模型选择推理方式
    if (text_emb_session_ && speech_emb_session_ && decoder_session_) {
        return inferenceComponents(text_token, prompt_text_token,
                                  prompt_speech_token, embedding);
    } else if (llm_session_) {
        // 简化模式：需要先构建完整的输入embedding
        // TODO: 实现完整的输入构建
        LOG_WARN("Simplified LLM inference not fully implemented");
        return std::vector<int64_t>();
    }

    LOG_ERROR("No valid LLM model loaded");
    return std::vector<int64_t>();
}

std::vector<int64_t> LLMInference::inferenceComponents(
    const std::vector<int64_t>& text_token,
    const std::vector<int64_t>& prompt_text_token,
    const std::vector<int64_t>& prompt_speech_token,
    const std::vector<float>& embedding) {

    LOG_INFO("Running component-based LLM inference...");

    // 1. 合并text tokens: [prompt_text, text]
    std::vector<int64_t> full_text_token = prompt_text_token;
    full_text_token.insert(full_text_token.end(),
                          text_token.begin(), text_token.end());

    // 2. 运行text embedding
    std::vector<float> text_emb = runTextEmbedding(full_text_token);

    // 3. 运行speech embedding (prompt)
    std::vector<float> prompt_speech_emb = runSpeechEmbedding(prompt_speech_token);

    // 4. 运行speaker affine
    std::vector<float> spk_emb = runSpkAffine(embedding);

    // 5. 构建LLM输入
    std::vector<float> lm_input = buildLLMInput(text_emb, prompt_speech_emb, spk_emb);

    // 6. 计算生成长度
    int text_len = text_token.size();
    int min_len = static_cast<int>(text_len * config_.min_token_text_ratio);
    int max_len = static_cast<int>(text_len * config_.max_token_text_ratio);

    LOG_INFO("  Generation length: min=", min_len, ", max=", max_len);

    // 7. 自回归生成循环
    // 注意：这里需要一个完整的Qwen2模型来执行自回归生成
    // 当前简化实现：直接返回一些占位tokens

    LOG_WARN("Autoregressive generation not fully implemented");
    LOG_WARN("Returning placeholder speech tokens");

    std::vector<int64_t> speech_tokens;
    for (int i = 0; i < max_len / 2; ++i) {
        speech_tokens.push_back(1000 + (i % config_.speech_token_size));
    }

    return speech_tokens;
}

std::vector<float> LLMInference::runTextEmbedding(
    const std::vector<int64_t>& tokens) {

    try {
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(tokens.size())};

        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info_,
            const_cast<int64_t*>(tokens.data()),
            tokens.size(),
            input_shape.data(),
            input_shape.size());

        const char* input_names[] = {"text_token"};
        const char* output_names[] = {"text_embedding"};

        auto output_tensors = text_emb_session_->Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1);

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        size_t output_size = 1;
        for (auto dim : output_shape) {
            output_size *= dim;
        }

        return std::vector<float>(output_data, output_data + output_size);

    } catch (const Ort::Exception& e) {
        LOG_ERROR("Text embedding error: ", e.what());
        return std::vector<float>(tokens.size() * config_.llm_input_size, 0.0f);
    }
}

std::vector<float> LLMInference::runSpeechEmbedding(
    const std::vector<int64_t>& tokens) {

    if (tokens.empty()) {
        return std::vector<float>();
    }

    try {
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(tokens.size())};

        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info_,
            const_cast<int64_t*>(tokens.data()),
            tokens.size(),
            input_shape.data(),
            input_shape.size());

        const char* input_names[] = {"speech_token"};
        const char* output_names[] = {"speech_embedding"};

        auto output_tensors = speech_emb_session_->Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1);

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        size_t output_size = 1;
        for (auto dim : output_shape) {
            output_size *= dim;
        }

        return std::vector<float>(output_data, output_data + output_size);

    } catch (const Ort::Exception& e) {
        LOG_ERROR("Speech embedding error: ", e.what());
        return std::vector<float>(tokens.size() * config_.llm_input_size, 0.0f);
    }
}

std::vector<float> LLMInference::runSpkAffine(const std::vector<float>& embedding) {
    if (!spk_affine_session_) {
        // 如果没有affine层，直接返回（或者做简单的扩展）
        return embedding;
    }

    try {
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(embedding.size())};

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(embedding.data()),
            embedding.size(),
            input_shape.data(),
            input_shape.size());

        const char* input_names[] = {"spk_embedding"};
        const char* output_names[] = {"spk_embedding_projected"};

        auto output_tensors = spk_affine_session_->Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1);

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        size_t output_size = 1;
        for (auto dim : output_shape) {
            output_size *= dim;
        }

        return std::vector<float>(output_data, output_data + output_size);

    } catch (const Ort::Exception& e) {
        LOG_ERROR("Speaker affine error: ", e.what());
        return embedding;
    }
}

std::vector<float> LLMInference::runDecoder(const std::vector<float>& hidden_states) {
    try {
        // hidden_states shape: [batch, seq_len, hidden_dim]
        // 这里需要知道shape信息
        size_t hidden_dim = config_.llm_output_size;
        size_t seq_len = hidden_states.size() / hidden_dim;

        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(seq_len),
                                           static_cast<int64_t>(hidden_dim)};

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(hidden_states.data()),
            hidden_states.size(),
            input_shape.data(),
            input_shape.size());

        const char* input_names[] = {"hidden_states"};
        const char* output_names[] = {"logits"};

        auto output_tensors = decoder_session_->Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1);

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        size_t output_size = 1;
        for (auto dim : output_shape) {
            output_size *= dim;
        }

        return std::vector<float>(output_data, output_data + output_size);

    } catch (const Ort::Exception& e) {
        LOG_ERROR("Decoder error: ", e.what());
        return std::vector<float>();
    }
}

std::vector<float> LLMInference::buildLLMInput(
    const std::vector<float>& text_embedding,
    const std::vector<float>& prompt_speech_embedding,
    const std::vector<float>& spk_embedding) {

    std::vector<float> lm_input;

    // 构建输入序列：[SOS] [spk_emb] [text] [TaskID] [prompt_speech]
    // 注意：这里需要根据实际的embedding shape来拼接

    // 1. SOS/EOS embedding
    lm_input.insert(lm_input.end(), sos_eos_emb_.begin(), sos_eos_emb_.end());

    // 2. Speaker embedding (unsqueeze to [1, hidden_dim])
    lm_input.insert(lm_input.end(), spk_embedding.begin(), spk_embedding.end());

    // 3. Text embedding
    lm_input.insert(lm_input.end(), text_embedding.begin(), text_embedding.end());

    // 4. Task ID embedding
    lm_input.insert(lm_input.end(), task_id_emb_.begin(), task_id_emb_.end());

    // 5. Prompt speech embedding
    lm_input.insert(lm_input.end(), prompt_speech_embedding.begin(),
                   prompt_speech_embedding.end());

    return lm_input;
}

std::vector<float> LLMInference::softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());

    // 找到最大值（数值稳定性）
    float max_logit = *std::max_element(logits.begin(), logits.end());

    // Exp
    float sum_exp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum_exp += probs[i];
    }

    // Normalize
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sum_exp;
    }

    return probs;
}

int64_t LLMInference::sampleTopK(const std::vector<float>& logits, bool ignore_eos) {
    // 应用温度
    std::vector<float> scaled_logits(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        scaled_logits[i] = logits[i] / config_.temperature;
    }

    // Softmax
    std::vector<float> probs = softmax(scaled_logits);

    // Top-K过滤
    std::vector<std::pair<float, int64_t>> prob_idx;
    for (size_t i = 0; i < probs.size(); ++i) {
        // 如果ignore_eos，跳过EOS token
        if (ignore_eos && i == config_.speech_token_size) {
            continue;
        }
        prob_idx.push_back({probs[i], static_cast<int64_t>(i)});
    }

    // 按概率降序排序
    std::partial_sort(prob_idx.begin(),
                     prob_idx.begin() + std::min(config_.top_k, static_cast<int>(prob_idx.size())),
                     prob_idx.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

    // 只保留Top-K
    prob_idx.resize(std::min(config_.top_k, static_cast<int>(prob_idx.size())));

    // 重新归一化
    float sum_prob = 0.0f;
    for (const auto& p : prob_idx) {
        sum_prob += p.first;
    }

    // 采样
    std::uniform_real_distribution<float> dist(0.0f, sum_prob);
    float rand_val = dist(rng_);

    float cumsum = 0.0f;
    for (const auto& p : prob_idx) {
        cumsum += p.first;
        if (cumsum >= rand_val) {
            return p.second;
        }
    }

    // Fallback
    return prob_idx[0].second;
}

} // namespace tts
} // namespace voice_assistant
