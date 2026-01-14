#ifndef VOICE_ASSISTANT_LLM_CLIENT_H
#define VOICE_ASSISTANT_LLM_CLIENT_H

#include <string>
#include <memory>

namespace voice_assistant {
namespace query {

/**
 * @brief LLM 客户端接口
 */
class ILLMClient {
public:
    virtual ~ILLMClient() = default;

    /**
     * @brief 查询 LLM
     * @param prompt 提示文本
     * @param max_tokens 最大生成 token 数
     * @return 生成的文本
     */
    virtual std::string query(const std::string& prompt, int max_tokens = 512) = 0;

    /**
     * @brief 检查是否已初始化
     */
    virtual bool isInitialized() const = 0;
};

/**
 * @brief 本地 LLM 客户端 (llama.cpp)
 */
class LocalLLMClient : public ILLMClient {
public:
    struct Config {
        std::string model_path;
        int context_length = 4096;
        int n_gpu_layers = -1;  // -1 表示全部放GPU
        float temperature = 0.7f;
        float top_p = 0.9f;
        int top_k = 40;
    };

    explicit LocalLLMClient(const Config& config);
    ~LocalLLMClient() override;

    /**
     * @brief 初始化本地 LLM
     */
    bool initialize();

    std::string query(const std::string& prompt, int max_tokens = 512) override;
    bool isInitialized() const override { return initialized_; }

private:
    Config config_;
    bool initialized_;

    // llama.cpp 上下文（使用 pimpl 模式隐藏实现）
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief 远程 LLM 客户端 (API)
 */
class RemoteLLMClient : public ILLMClient {
public:
    struct Config {
        std::string api_url;
        std::string api_key;
        float temperature = 0.7f;
        std::string model_name = "gpt-3.5-turbo";
    };

    explicit RemoteLLMClient(const Config& config);
    ~RemoteLLMClient() override;

    /**
     * @brief 初始化远程 LLM 客户端
     */
    bool initialize();

    std::string query(const std::string& prompt, int max_tokens = 512) override;
    bool isInitialized() const override { return initialized_; }

private:
    /**
     * @brief 发送 HTTP POST 请求
     */
    std::string sendHttpRequest(const std::string& json_payload);

    Config config_;
    bool initialized_;
};

} // namespace query
} // namespace voice_assistant

#endif // VOICE_ASSISTANT_LLM_CLIENT_H
