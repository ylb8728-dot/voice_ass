#include "sqlite_db.h"
#include "../utils/logger.h"
#include <sstream>

namespace voice_assistant {
namespace query {

SQLiteDB::SQLiteDB(const Config& config)
    : config_(config)
    , db_(nullptr) {
}

SQLiteDB::~SQLiteDB() {
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

bool SQLiteDB::initialize() {
    LOG_INFO("Initializing SQLite database: ", config_.db_path);

    int rc = sqlite3_open(config_.db_path.c_str(), &db_);
    if (rc != SQLITE_OK) {
        LOG_ERROR("Failed to open database: ", sqlite3_errmsg(db_));
        return false;
    }

    // 设置缓存大小
    std::ostringstream cache_sql;
    cache_sql << "PRAGMA cache_size = " << config_.cache_size;
    executeSQL(cache_sql.str());

    // 创建表
    if (config_.enable_fts) {
        if (!createFTS5Table()) {
            return false;
        }
    }

    LOG_INFO("SQLite database initialized");
    return true;
}

std::vector<QueryResult> SQLiteDB::search(const std::string& query, int limit) {
    std::vector<QueryResult> results;

    if (!db_) {
        LOG_ERROR("Database not initialized");
        return results;
    }

    // 构建 FTS5 查询
    std::ostringstream sql;
    sql << "SELECT question, answer, category, rank "
        << "FROM knowledge_fts "
        << "WHERE knowledge_fts MATCH ? "
        << "ORDER BY rank "
        << "LIMIT ?";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql.str().c_str(), -1, &stmt, nullptr);

    if (rc != SQLITE_OK) {
        LOG_ERROR("Failed to prepare statement: ", sqlite3_errmsg(db_));
        return results;
    }

    // 绑定参数
    sqlite3_bind_text(stmt, 1, query.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 2, limit);

    // 执行查询
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        QueryResult result;
        result.question = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        result.answer = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        result.category = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        result.score = static_cast<float>(sqlite3_column_double(stmt, 3));
        results.push_back(result);
    }

    sqlite3_finalize(stmt);

    LOG_DEBUG("Search query '", query, "' returned ", results.size(), " results");
    return results;
}

bool SQLiteDB::insert(const std::string& question,
                     const std::string& answer,
                     const std::string& category) {
    if (!db_) {
        LOG_ERROR("Database not initialized");
        return false;
    }

    const char* sql = "INSERT INTO knowledge_fts (question, answer, category) VALUES (?, ?, ?)";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);

    if (rc != SQLITE_OK) {
        LOG_ERROR("Failed to prepare statement: ", sqlite3_errmsg(db_));
        return false;
    }

    sqlite3_bind_text(stmt, 1, question.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, answer.c_str(), -1, SQLITE_TRANSIENT);
    std::string cat = category.empty() ? "general" : category;
    sqlite3_bind_text(stmt, 3, cat.c_str(), -1, SQLITE_TRANSIENT);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        LOG_ERROR("Failed to insert data: ", sqlite3_errmsg(db_));
        return false;
    }

    return true;
}

bool SQLiteDB::insertBatch(
    const std::vector<std::tuple<std::string, std::string, std::string>>& entries) {

    if (!db_) {
        LOG_ERROR("Database not initialized");
        return false;
    }

    // 开始事务
    executeSQL("BEGIN TRANSACTION");

    for (const auto& entry : entries) {
        if (!insert(std::get<0>(entry), std::get<1>(entry), std::get<2>(entry))) {
            executeSQL("ROLLBACK");
            return false;
        }
    }

    // 提交事务
    executeSQL("COMMIT");

    LOG_INFO("Inserted ", entries.size(), " entries into database");
    return true;
}

bool SQLiteDB::createFTS5Table() {
    const char* sql =
        "CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5("
        "    question, "
        "    answer, "
        "    category"
        ")";

    if (!executeSQL(sql)) {
        LOG_ERROR("Failed to create FTS5 table");
        return false;
    }

    LOG_INFO("FTS5 table created");
    return true;
}

bool SQLiteDB::executeSQL(const std::string& sql) {
    if (!db_) {
        return false;
    }

    char* err_msg = nullptr;
    int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err_msg);

    if (rc != SQLITE_OK) {
        LOG_ERROR("SQL error: ", err_msg);
        sqlite3_free(err_msg);
        return false;
    }

    return true;
}

} // namespace query
} // namespace voice_assistant
