#include "llm_client.h"
#include "../utils/logger.h"
#include <sstream>
#include <cstring>
#include <vector>

#include "llama.h"
#include "ggml-backend.h"

namespace voice_assistant {
namespace query {

// ==================== LocalLLMClient ====================

class LocalLLMClient::Impl {
public:
    llama_context* ctx = nullptr;
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    llama_sampler* sampler = nullptr;

    bool loadModel(const std::string& model_path, const Config& config) {
        LOG_INFO("Loading local LLM model: ", model_path);

        // 加载所有后端
        ggml_backend_load_all();

        // 设置模型参数
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = config.n_gpu_layers;

        // 加载模型
        model = llama_model_load_from_file(model_path.c_str(), model_params);
        if (!model) {
            LOG_ERROR("Failed to load model from: ", model_path);
            return false;
        }

        // 获取词汇表
        vocab = llama_model_get_vocab(model);

        // 设置上下文参数
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = config.context_length;
        ctx_params.n_batch = 512;
        ctx_params.n_threads = 4;
        ctx_params.no_perf = true;

        // 创建上下文
        ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) {
            LOG_ERROR("Failed to create llama context");
            llama_model_free(model);
            model = nullptr;
            return false;
        }

        // 初始化采样器
        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = true;
        sampler = llama_sampler_chain_init(sparams);

        // 添加采样策略
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(config.top_k));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(config.top_p, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(config.temperature));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        LOG_INFO("Local LLM model loaded successfully");
        return true;
    }

    std::string generate(const std::string& prompt, int max_tokens, const Config& config) {
        (void)config;  // config已在loadModel中使用

        if (!ctx || !model || !vocab || !sampler) {
            LOG_ERROR("LLM not initialized");
            return "";
        }

        LOG_DEBUG("Generating response for prompt");

        // 清空KV缓存，确保每次生成都是独立的
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) {
            llama_memory_clear(mem, true);
        }

        // 重置采样器状态
        llama_sampler_reset(sampler);

        // 分词（预分配足够大的缓冲区，一次调用完成）
        std::vector<llama_token> prompt_tokens(prompt.size() * 2 + 32);
        int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                      prompt_tokens.data(), prompt_tokens.size(), true, true);
        if (n_tokens < 0) {
            LOG_ERROR("Failed to tokenize prompt");
            return "";
        }
        prompt_tokens.resize(n_tokens);

        // 准备批处理
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

        // 评估提示
        if (llama_decode(ctx, batch)) {
            LOG_ERROR("Failed to decode prompt");
            return "";
        }

        // 生成token
        std::string result;
        int n_decode = 0;
        std::string last_segment;  // 用于检测重复

        for (int n_pos = prompt_tokens.size(); n_decode < max_tokens; n_decode++) {
            // 采样下一个token
            llama_token new_token_id = llama_sampler_sample(sampler, ctx, -1);

            // 检查是否是结束token
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            // 将token转换为文本
            char buf[256];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                LOG_ERROR("Failed to convert token to piece");
                break;
            }

            std::string piece(buf, n);
            result.append(piece);

            // 检测应停止的模式
            if (result.size() > 30) {
                bool should_stop = false;
                size_t truncate_pos = std::string::npos;

                // 检测重复的prompt模式
                const char* stop_patterns[] = {
                    "用户问题:",
                    "用户问题：",
                    "请用中文简洁地回答:",
                    "请用中文简洁地回答：",
                    "\n\n\n",
                    "看起来用户",
                    "根据用户要求",
                    "我需要",
                    "需要检查",
                    "<think>",
                    "</think>",
                };

                for (const char* pattern : stop_patterns) {
                    size_t pos = result.find(pattern);
                    if (pos != std::string::npos && pos > 10) {
                        should_stop = true;
                        if (truncate_pos == std::string::npos || pos < truncate_pos) {
                            truncate_pos = pos;
                        }
                    }
                }

                if (should_stop && truncate_pos != std::string::npos) {
                    result = result.substr(0, truncate_pos);
                    break;
                }
            }

            // 准备下一个batch
            batch = llama_batch_get_one(&new_token_id, 1);

            // 评估
            if (llama_decode(ctx, batch)) {
                LOG_ERROR("Failed to decode");
                break;
            }

            n_pos++;
        }

        // 清理结果：移除末尾多余的空白
        while (!result.empty() && (result.back() == '\n' || result.back() == ' ')) {
            result.pop_back();
        }

        LOG_DEBUG("Generated ", n_decode, " tokens");
        return result;
    }

    void cleanup() {
        if (sampler) {
            llama_sampler_free(sampler);
            sampler = nullptr;
        }
        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }
        if (model) {
            llama_model_free(model);
            model = nullptr;
        }
        vocab = nullptr;
    }
};

LocalLLMClient::LocalLLMClient(const Config& config)
    : config_(config)
    , initialized_(false)
    , impl_(std::make_unique<Impl>()) {
}

LocalLLMClient::~LocalLLMClient() {
    if (initialized_) {
        impl_->cleanup();
    }
}

bool LocalLLMClient::initialize() {
    LOG_INFO("Initializing local LLM client...");
    LOG_INFO("Model: ", config_.model_path);

    if (!impl_->loadModel(config_.model_path, config_)) {
        LOG_ERROR("Failed to load local LLM model");
        return false;
    }

    initialized_ = true;
    LOG_INFO("Local LLM client initialized");
    return true;
}

std::string LocalLLMClient::query(const std::string& prompt, int max_tokens) {
    if (!initialized_) {
        LOG_ERROR("Local LLM client not initialized");
        return "";
    }

    LOG_DEBUG("Local LLM query: ", prompt.substr(0, 50), "...");

    std::string response = impl_->generate(prompt, max_tokens, config_);

    LOG_DEBUG("Local LLM response: ", response.substr(0, 50), "...");
    return response;
}

// ==================== RemoteLLMClient ====================

RemoteLLMClient::RemoteLLMClient(const Config& config)
    : config_(config)
    , initialized_(false) {
}

RemoteLLMClient::~RemoteLLMClient() = default;

bool RemoteLLMClient::initialize() {
    LOG_INFO("Initializing remote LLM client...");
    LOG_INFO("API URL: ", config_.api_url);

    // 检查配置
    if (config_.api_url.empty()) {
        LOG_ERROR("API URL is empty");
        return false;
    }

    initialized_ = true;
    LOG_INFO("Remote LLM client initialized");
    return true;
}

std::string RemoteLLMClient::query(const std::string& prompt, int max_tokens) {
    if (!initialized_) {
        LOG_ERROR("Remote LLM client not initialized");
        return "";
    }

    LOG_DEBUG("Remote LLM query: ", prompt.substr(0, 50), "...");

    // 构建 JSON 请求
    std::ostringstream json;
    json << "{"
         << "\"model\": \"" << config_.model_name << "\","
         << "\"messages\": [{\"role\": \"user\", \"content\": \"" << prompt << "\"}],"
         << "\"max_tokens\": " << max_tokens << ","
         << "\"temperature\": " << config_.temperature
         << "}";

    // 发送 HTTP 请求
    std::string response = sendHttpRequest(json.str());

    // TODO: 解析 JSON 响应
    // 这里简化处理，实际需要使用 JSON 库（如 nlohmann/json）
    LOG_DEBUG("Remote LLM response received");

    return "这是远程LLM的占位响应。实际使用时需要集成 HTTP 客户端和 JSON 解析库。";
}

std::string RemoteLLMClient::sendHttpRequest(const std::string& json_payload) {
    // TODO: 实际实现需要使用 HTTP 客户端库（如 libcurl, cpp-httplib 等）
    // 这里提供一个占位实现

    LOG_DEBUG("Sending HTTP POST request to ", config_.api_url);
    LOG_DEBUG("Payload: ", json_payload.substr(0, 100), "...");

    // 占位返回
    return "{\"choices\":[{\"message\":{\"content\":\"Response from API\"}}]}";
}

} // namespace query
} // namespace voice_assistant
