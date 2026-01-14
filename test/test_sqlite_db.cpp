/**
 * @file test_sqlite_db.cpp
 * @brief SQLiteDB 数据库测试
 *
 * 测试内容:
 * 1. 数据库初始化
 * 2. 数据插入
 * 3. 全文检索 (FTS5)
 * 4. 批量操作
 */

#include "query/sqlite_db.h"
#include "utils/logger.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdio>

using namespace voice_assistant::query;
using namespace voice_assistant::utils;

// 测试数据
const std::vector<std::tuple<std::string, std::string, std::string>> TEST_DATA = {
    {"今天天气怎么样", "今天天气晴朗，温度适宜。", "天气"},
    {"明天会下雨吗", "根据预报，明天可能有小雨。", "天气"},
    {"现在几点了", "请查看您的时钟。", "时间"},
    {"帮我设置一个闹钟", "好的，请告诉我需要设置几点的闹钟。", "工具"},
    {"播放音乐", "正在为您播放音乐。", "娱乐"},
    {"讲一个笑话", "为什么程序员总是分不清万圣节和圣诞节？因为 Oct 31 = Dec 25。", "娱乐"},
    {"什么是人工智能", "人工智能是计算机科学的一个分支，致力于创造能够模拟人类智能的系统。", "科技"},
    {"Python是什么", "Python是一种广泛使用的高级编程语言，以简洁和可读性著称。", "科技"},
    {"你好", "你好！有什么可以帮助您的吗？", "问候"},
    {"再见", "再见，期待下次与您交流！", "问候"},
};

// 删除测试数据库文件
void cleanupTestDB(const std::string& db_path) {
    std::remove(db_path.c_str());
}

// 测试1: 初始化测试
bool testInitialize() {
    std::cout << "\n========== 测试1: 数据库初始化 ==========" << std::endl;

    const std::string test_db = "test_sqlite_init.db";
    cleanupTestDB(test_db);

    SQLiteDB::Config config;
    config.db_path = test_db;
    config.enable_fts = true;
    config.cache_size = 1000;

    SQLiteDB db(config);

    bool result = db.initialize();
    if (result) {
        std::cout << "[PASS] 数据库初始化成功" << std::endl;
    } else {
        std::cout << "[FAIL] 数据库初始化失败" << std::endl;
    }

    cleanupTestDB(test_db);
    return result;
}

// 测试2: 插入测试
bool testInsert() {
    std::cout << "\n========== 测试2: 数据插入 ==========" << std::endl;

    const std::string test_db = "test_sqlite_insert.db";
    cleanupTestDB(test_db);

    SQLiteDB::Config config;
    config.db_path = test_db;
    config.enable_fts = true;

    SQLiteDB db(config);
    if (!db.initialize()) {
        std::cout << "[FAIL] 初始化失败" << std::endl;
        return false;
    }

    // 插入单条数据
    bool result = db.insert("测试问题", "测试答案", "测试类别");
    if (result) {
        std::cout << "[PASS] 单条插入成功" << std::endl;
    } else {
        std::cout << "[FAIL] 单条插入失败" << std::endl;
        cleanupTestDB(test_db);
        return false;
    }

    // 插入不带类别的数据
    result = db.insert("另一个问题", "另一个答案");
    if (result) {
        std::cout << "[PASS] 无类别插入成功" << std::endl;
    } else {
        std::cout << "[FAIL] 无类别插入失败" << std::endl;
    }

    cleanupTestDB(test_db);
    return result;
}

// 测试3: 批量插入测试
bool testBatchInsert() {
    std::cout << "\n========== 测试3: 批量插入 ==========" << std::endl;

    const std::string test_db = "test_sqlite_batch.db";
    cleanupTestDB(test_db);

    SQLiteDB::Config config;
    config.db_path = test_db;
    config.enable_fts = true;

    SQLiteDB db(config);
    if (!db.initialize()) {
        std::cout << "[FAIL] 初始化失败" << std::endl;
        return false;
    }

    auto start = std::chrono::high_resolution_clock::now();
    bool result = db.insertBatch(TEST_DATA);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (result) {
        std::cout << "[PASS] 批量插入 " << TEST_DATA.size() << " 条数据成功" << std::endl;
        std::cout << "      耗时: " << duration.count() << " ms" << std::endl;
    } else {
        std::cout << "[FAIL] 批量插入失败" << std::endl;
    }

    cleanupTestDB(test_db);
    return result;
}

// 测试4: 全文检索测试
bool testSearch() {
    std::cout << "\n========== 测试4: 全文检索 ==========" << std::endl;

    const std::string test_db = "test_sqlite_search.db";
    cleanupTestDB(test_db);

    SQLiteDB::Config config;
    config.db_path = test_db;
    config.enable_fts = true;

    SQLiteDB db(config);
    if (!db.initialize()) {
        std::cout << "[FAIL] 初始化失败" << std::endl;
        return false;
    }

    // 插入测试数据
    if (!db.insertBatch(TEST_DATA)) {
        std::cout << "[FAIL] 插入测试数据失败" << std::endl;
        cleanupTestDB(test_db);
        return false;
    }

    bool all_passed = true;

    // 测试查询
    struct TestCase {
        std::string query;
        std::string expected_keyword;
    };

    std::vector<TestCase> test_cases = {
        {"天气", "天气"},
        {"音乐", "播放"},
        {"Python", "Python"},
        {"你好", "你好"},
    };

    for (const auto& tc : test_cases) {
        std::cout << "\n查询: \"" << tc.query << "\"" << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        auto results = db.search(tc.query, 3);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "  结果数: " << results.size() << " (耗时: " << duration.count() << " us)" << std::endl;

        if (!results.empty()) {
            std::cout << "  最佳匹配:" << std::endl;
            std::cout << "    问题: " << results[0].question << std::endl;
            std::cout << "    答案: " << results[0].answer << std::endl;
            std::cout << "    类别: " << results[0].category << std::endl;
            std::cout << "    分数: " << results[0].score << std::endl;

            // 检查结果是否包含期望关键词
            if (results[0].question.find(tc.expected_keyword) != std::string::npos ||
                results[0].answer.find(tc.expected_keyword) != std::string::npos) {
                std::cout << "  [PASS]" << std::endl;
            } else {
                std::cout << "  [WARN] 未找到期望关键词" << std::endl;
            }
        } else {
            std::cout << "  [FAIL] 无结果" << std::endl;
            all_passed = false;
        }
    }

    cleanupTestDB(test_db);
    return all_passed;
}

// 测试5: 空查询和边界测试
bool testEdgeCases() {
    std::cout << "\n========== 测试5: 边界情况测试 ==========" << std::endl;

    const std::string test_db = "test_sqlite_edge.db";
    cleanupTestDB(test_db);

    SQLiteDB::Config config;
    config.db_path = test_db;
    config.enable_fts = true;

    SQLiteDB db(config);
    if (!db.initialize()) {
        std::cout << "[FAIL] 初始化失败" << std::endl;
        return false;
    }

    db.insertBatch(TEST_DATA);

    bool all_passed = true;

    // 测试不存在的内容
    std::cout << "\n查询不存在的内容: \"xyz123不存在\"" << std::endl;
    auto results = db.search("xyz123不存在", 5);
    std::cout << "  结果数: " << results.size() << std::endl;
    if (results.empty()) {
        std::cout << "  [PASS] 正确返回空结果" << std::endl;
    }

    // 测试特殊字符
    std::cout << "\n测试特殊字符插入" << std::endl;
    bool insert_result = db.insert("包含'引号\"的问题", "包含<>&特殊字符的答案", "special");
    std::cout << "  特殊字符插入: " << (insert_result ? "[PASS]" : "[FAIL]") << std::endl;
    if (!insert_result) all_passed = false;

    // 测试空字符串
    std::cout << "\n测试空字符串查询" << std::endl;
    results = db.search("", 5);
    std::cout << "  空查询结果数: " << results.size() << std::endl;

    cleanupTestDB(test_db);
    return all_passed;
}

// 测试6: 性能测试
bool testPerformance() {
    std::cout << "\n========== 测试6: 性能测试 ==========" << std::endl;

    const std::string test_db = "test_sqlite_perf.db";
    cleanupTestDB(test_db);

    SQLiteDB::Config config;
    config.db_path = test_db;
    config.enable_fts = true;
    config.cache_size = 10000;

    SQLiteDB db(config);
    if (!db.initialize()) {
        std::cout << "[FAIL] 初始化失败" << std::endl;
        return false;
    }

    // 生成大量测试数据
    const int NUM_ENTRIES = 1000;
    std::vector<std::tuple<std::string, std::string, std::string>> large_data;
    large_data.reserve(NUM_ENTRIES);

    for (int i = 0; i < NUM_ENTRIES; ++i) {
        std::string q = "测试问题编号" + std::to_string(i) + "关键词ABC";
        std::string a = "这是问题" + std::to_string(i) + "的详细答案内容";
        std::string c = "类别" + std::to_string(i % 10);
        large_data.push_back({q, a, c});
    }

    // 批量插入性能
    std::cout << "\n批量插入 " << NUM_ENTRIES << " 条数据..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    bool result = db.insertBatch(large_data);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "  插入耗时: " << duration.count() << " ms" << std::endl;
    std::cout << "  每秒插入: " << (NUM_ENTRIES * 1000 / (duration.count() + 1)) << " 条" << std::endl;

    if (!result) {
        std::cout << "[FAIL] 批量插入失败" << std::endl;
        cleanupTestDB(test_db);
        return false;
    }

    // 查询性能
    const int NUM_QUERIES = 100;
    std::cout << "\n执行 " << NUM_QUERIES << " 次查询..." << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_QUERIES; ++i) {
        db.search("关键词ABC", 5);
    }
    end = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "  查询耗时: " << duration.count() << " ms" << std::endl;
    std::cout << "  每秒查询: " << (NUM_QUERIES * 1000 / (duration.count() + 1)) << " 次" << std::endl;
    std::cout << "  平均延迟: " << (duration.count() * 1.0 / NUM_QUERIES) << " ms" << std::endl;

    cleanupTestDB(test_db);
    return true;
}

void printUsage(const char* prog) {
    std::cout << "用法: " << prog << " [选项]" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  --init         初始化测试" << std::endl;
    std::cout << "  --insert       插入测试" << std::endl;
    std::cout << "  --batch        批量插入测试" << std::endl;
    std::cout << "  --search       全文检索测试" << std::endl;
    std::cout << "  --edge         边界情况测试" << std::endl;
    std::cout << "  --perf         性能测试" << std::endl;
    std::cout << "  --all          运行所有测试" << std::endl;
    std::cout << "  --help         显示帮助" << std::endl;
}

int main(int argc, char* argv[]) {
    Logger::getInstance().setLevel(LogLevel::INFO);

    std::cout << "===== SQLiteDB 数据库测试 =====" << std::endl;

    if (argc < 2) {
        printUsage(argv[0]);
        return 0;
    }

    std::string cmd = argv[1];

    if (cmd == "--help" || cmd == "-h") {
        printUsage(argv[0]);
        return 0;
    }

    bool success = true;

    if (cmd == "--init") {
        success = testInitialize();
    } else if (cmd == "--insert") {
        success = testInsert();
    } else if (cmd == "--batch") {
        success = testBatchInsert();
    } else if (cmd == "--search") {
        success = testSearch();
    } else if (cmd == "--edge") {
        success = testEdgeCases();
    } else if (cmd == "--perf") {
        success = testPerformance();
    } else if (cmd == "--all") {
        success = testInitialize() &&
                  testInsert() &&
                  testBatchInsert() &&
                  testSearch() &&
                  testEdgeCases() &&
                  testPerformance();
    } else {
        std::cerr << "未知选项: " << cmd << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "测试结果: " << (success ? "全部通过" : "存在失败") << std::endl;

    return success ? 0 : 1;
}
