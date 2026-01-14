#ifndef VOICE_ASSISTANT_QUERY_ENGINE_H
#define VOICE_ASSISTANT_QUERY_ENGINE_H

#include "sqlite_db.h"
#include "llm_client.h"
#include <memory>
#include <string>

namespace voice_assistant {
namespace query {

/**
 * @brief 查询引擎
 *
 * 整合本地数据库和 LLM，提供统一的查询接口
 */
class QueryEngine {
public:
    enum class Mode {
        LOCAL_ONLY,   // 仅使用本地数据库
        LLM_ONLY,     // 仅使用 LLM
        HYBRID        // 混合模式：优先本地，必要时使用 LLM
    };

    struct Config {
        Mode mode = Mode::HYBRID;

        // 本地数据库配置
        SQLiteDB::Config db_config;

        // LLM 配置
        std::string llm_type = "local";  // local, remote
        LocalLLMClient::Config local_llm_config;
        RemoteLLMClient::Config remote_llm_config;

        // 混合模式配置
        bool use_local_first = true;
        float local_confidence_threshold = 0.7f;
    };

    explicit QueryEngine(const Config& config);
    ~QueryEngine();

    // 禁止拷贝
    QueryEngine(const QueryEngine&) = delete;
    QueryEngine& operator=(const QueryEngine&) = delete;

    /**
     * @brief 初始化查询引擎
     */
    bool initialize();

    /**
     * @brief 查询
     * @param text 查询文本
     * @return 答案
     */
    std::string query(const std::string& text);

    /**
     * @brief 添加知识到本地数据库
     */
    bool addKnowledge(const std::string& question,
                     const std::string& answer,
                     const std::string& category = std::string());

    /**
     * @brief 检查是否已初始化
     */
    bool isInitialized() const { return initialized_; }

private:
    /**
     * @brief 本地数据库查询
     */
    std::string queryLocal(const std::string& text);

    /**
     * @brief LLM 查询
     */
    std::string queryLLM(const std::string& text);

    /**
     * @brief 混合查询
     */
    std::string queryHybrid(const std::string& text);

    Config config_;
    bool initialized_;

    std::unique_ptr<SQLiteDB> db_;
    std::unique_ptr<ILLMClient> llm_;
};

} // namespace query
} // namespace voice_assistant

#endif // VOICE_ASSISTANT_QUERY_ENGINE_H
