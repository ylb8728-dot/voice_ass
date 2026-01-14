#include "query_engine.h"
#include "../utils/logger.h"

namespace voice_assistant {
namespace query {

QueryEngine::QueryEngine(const Config& config)
    : config_(config)
    , initialized_(false) {
}

QueryEngine::~QueryEngine() = default;

bool QueryEngine::initialize() {
    LOG_INFO("Initializing query engine (mode: ",
             config_.mode == Mode::LOCAL_ONLY ? "LOCAL_ONLY" :
             config_.mode == Mode::LLM_ONLY ? "LLM_ONLY" : "HYBRID", ")");

    // 初始化本地数据库
    if (config_.mode == Mode::LOCAL_ONLY || config_.mode == Mode::HYBRID) {
        db_ = std::make_unique<SQLiteDB>(config_.db_config);
        if (!db_->initialize()) {
            LOG_ERROR("Failed to initialize local database");
            return false;
        }
    }

    // 初始化 LLM
    if (config_.mode == Mode::LLM_ONLY || config_.mode == Mode::HYBRID) {
        if (config_.llm_type == "local") {
            auto local_llm = std::make_unique<LocalLLMClient>(config_.local_llm_config);
            if (!local_llm->initialize()) {
                LOG_ERROR("Failed to initialize local LLM");
                return false;
            }
            llm_ = std::move(local_llm);
        } else if (config_.llm_type == "remote") {
            auto remote_llm = std::make_unique<RemoteLLMClient>(config_.remote_llm_config);
            if (!remote_llm->initialize()) {
                LOG_ERROR("Failed to initialize remote LLM");
                return false;
            }
            llm_ = std::move(remote_llm);
        } else {
            LOG_ERROR("Unknown LLM type: ", config_.llm_type);
            return false;
        }
    }

    initialized_ = true;
    LOG_INFO("Query engine initialized successfully");
    return true;
}

std::string QueryEngine::query(const std::string& text) {
    if (!initialized_) {
        LOG_ERROR("Query engine not initialized");
        return "抱歉，系统未初始化。";
    }

    LOG_INFO("Query: ", text);

    std::string answer;

    switch (config_.mode) {
        case Mode::LOCAL_ONLY:
            answer = queryLocal(text);
            break;

        case Mode::LLM_ONLY:
            answer = queryLLM(text);
            break;

        case Mode::HYBRID:
            answer = queryHybrid(text);
            break;
    }

    if (answer.empty()) {
        answer = "抱歉，我没有找到相关信息。";
    }

    LOG_INFO("Answer: ", answer);
    return answer;
}

bool QueryEngine::addKnowledge(const std::string& question,
                              const std::string& answer,
                              const std::string& category) {
    if (!db_) {
        LOG_ERROR("Local database not available");
        return false;
    }

    std::string cat = category.empty() ? "general" : category;
    return db_->insert(question, answer, cat);
}

std::string QueryEngine::queryLocal(const std::string& text) {
    if (!db_) {
        return "";
    }

    auto results = db_->search(text, 1);
    if (!results.empty()) {
        LOG_DEBUG("Found local result with score: ", results[0].score);
        return results[0].answer;
    }

    return "";
}

std::string QueryEngine::queryLLM(const std::string& text) {
    if (!llm_) {
        return "";
    }

    // 构建提示
    std::string prompt = "用户问题: " + text + "\n请用中文简洁地回答:";

    return llm_->query(prompt);
}

std::string QueryEngine::queryHybrid(const std::string& text) {
    // 优先查询本地数据库
    if (config_.use_local_first && db_) {
        auto results = db_->search(text, 1);
        if (!results.empty() && results[0].score >= config_.local_confidence_threshold) {
            LOG_DEBUG("Using local result (score: ", results[0].score, ")");
            return results[0].answer;
        }
    }

    // 回退到 LLM
    LOG_DEBUG("Local result insufficient, using LLM");
    return queryLLM(text);
}

} // namespace query
} // namespace voice_assistant
