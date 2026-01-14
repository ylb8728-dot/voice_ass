#ifndef VOICE_ASSISTANT_SQLITE_DB_H
#define VOICE_ASSISTANT_SQLITE_DB_H

#include <sqlite3.h>
#include <string>
#include <vector>
#include <memory>

namespace voice_assistant {
namespace query {

struct QueryResult {
    std::string question;
    std::string answer;
    std::string category;
    float score;
};

/**
 * @brief SQLite 数据库封装类
 *
 * 提供全文检索功能
 */
class SQLiteDB {
public:
    struct Config {
        std::string db_path;
        bool enable_fts = true;
        int cache_size = 10000;
    };

    explicit SQLiteDB(const Config& config);
    ~SQLiteDB();

    // 禁止拷贝
    SQLiteDB(const SQLiteDB&) = delete;
    SQLiteDB& operator=(const SQLiteDB&) = delete;

    /**
     * @brief 初始化数据库
     * @return 成功返回 true
     */
    bool initialize();

    /**
     * @brief 全文检索
     * @param query 查询文本
     * @param limit 返回结果数量
     * @return 查询结果列表
     */
    std::vector<QueryResult> search(const std::string& query, int limit = 5);

    /**
     * @brief 插入知识条目
     * @param question 问题
     * @param answer 答案
     * @param category 类别
     * @return 成功返回 true
     */
    bool insert(const std::string& question,
                const std::string& answer,
                const std::string& category = std::string());

    /**
     * @brief 批量插入
     */
    bool insertBatch(const std::vector<std::tuple<std::string, std::string, std::string>>& entries);

    /**
     * @brief 检查数据库是否已初始化
     */
    bool isInitialized() const { return db_ != nullptr; }

private:
    /**
     * @brief 创建 FTS5 表
     */
    bool createFTS5Table();

    /**
     * @brief 执行 SQL 语句
     */
    bool executeSQL(const std::string& sql);

    Config config_;
    sqlite3* db_;
};

} // namespace query
} // namespace voice_assistant

#endif // VOICE_ASSISTANT_SQLITE_DB_H
