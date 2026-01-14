/**
 * @file test_query_engine.cpp
 * @brief QueryEngine 查询引擎综合测试
 *
 * 测试内容:
 * 1. LOCAL_ONLY 模式测试
 * 2. LLM_ONLY 模式测试
 * 3. HYBRID 混合模式测试
 * 4. 交互式查询测试
 */

#include "query/query_engine.h"
#include "utils/logger.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <cstdio>

using namespace voice_assistant::query;
using namespace voice_assistant::utils;

// 测试知识库数据
const std::vector<std::tuple<std::string, std::string, std::string>> KNOWLEDGE_BASE = {
    {"今天天气怎么样", "今天天气晴朗，温度约25度，适合户外活动。", "天气"},
    {"明天会下雨吗", "根据天气预报，明天多云转小雨。", "天气"},
    {"现在几点", "请查看您设备上的时钟。", "时间"},
    {"你是谁", "我是一个语音助手，可以回答您的问题。", "介绍"},
    {"你能做什么", "我可以回答问题、播放音乐、设置闹钟等。", "介绍"},
    {"播放音乐", "正在为您播放轻音乐。", "娱乐"},
    {"讲个笑话", "为什么程序员喜欢黑暗？因为Light是个Bug！", "娱乐"},
    {"什么是人工智能", "人工智能是让计算机模拟人类智能的技术。", "科技"},
    {"Python怎么学", "建议从基础语法开始，然后做项目实践。", "科技"},
    {"你好", "你好！有什么可以帮您的？", "问候"},
    {"再见", "再见，祝您生活愉快！", "问候"},
    {"谢谢", "不客气，这是我应该做的！", "问候"},
};

// 清理测试数据库
void cleanupTestDB(const std::string& db_path) {
    std::remove(db_path.c_str());
}

// 测试1: LOCAL_ONLY 模式
void testLocalOnlyMode() {
    std::cout << "\n========== 测试1: LOCAL_ONLY 模式 ==========" << std::endl;

    const std::string test_db = "test_query_local.db";
    cleanupTestDB(test_db);

    QueryEngine::Config config;
    config.mode = QueryEngine::Mode::LOCAL_ONLY;
    config.db_config.db_path = test_db;
    config.db_config.enable_fts = true;

    QueryEngine engine(config);

    std::cout << "初始化..." << std::endl;
    if (!engine.initialize()) {
        std::cout << "[FAIL] 初始化失败" << std::endl;
        return;
    }
    std::cout << "[PASS] 初始化成功" << std::endl;

    // 添加知识
    std::cout << "\n添加知识库数据..." << std::endl;
    for (const auto& [q, a, c] : KNOWLEDGE_BASE) {
        engine.addKnowledge(q, a, c);
    }
    std::cout << "已添加 " << KNOWLEDGE_BASE.size() << " 条知识" << std::endl;

    // 测试查询
    std::vector<std::string> test_queries = {
        "天气怎么样",
        "你是谁",
        "讲笑话",
        "Python",
    };

    std::cout << "\n查询测试:" << std::endl;
    for (const auto& q : test_queries) {
        auto start = std::chrono::high_resolution_clock::now();
        std::string answer = engine.query(q);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "\n问: " << q << std::endl;
        std::cout << "答: " << answer << std::endl;
        std::cout << "耗时: " << duration.count() << " us" << std::endl;
    }

    cleanupTestDB(test_db);
}

// 测试2: LLM_ONLY 模式
void testLLMOnlyMode(const std::string& model_path, int n_gpu_layers) {
    std::cout << "\n========== 测试2: LLM_ONLY 模式 ==========" << std::endl;

    QueryEngine::Config config;
    config.mode = QueryEngine::Mode::LLM_ONLY;
    config.llm_type = "local";
    config.local_llm_config.model_path = model_path;
    config.local_llm_config.context_length = 4096;
    config.local_llm_config.n_gpu_layers = n_gpu_layers;
    config.local_llm_config.temperature = 0.7f;

    QueryEngine engine(config);

    std::cout << "初始化 (加载LLM模型)..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    if (!engine.initialize()) {
        std::cout << "[FAIL] 初始化失败" << std::endl;
        return;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "[PASS] 初始化成功! 耗时: " << duration.count() << " ms" << std::endl;

    // 测试查询
    std::vector<std::string> test_queries = {
        "什么是机器学习？",
        "用一句话介绍你自己",
        "1加1等于几？",
    };

    std::cout << "\n查询测试:" << std::endl;
    for (const auto& q : test_queries) {
        std::cout << "\n问: " << q << std::endl;

        start = std::chrono::high_resolution_clock::now();
        std::string answer = engine.query(q);
        end = std::chrono::high_resolution_clock::now();

        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "答: " << answer << std::endl;
        std::cout << "耗时: " << duration.count() << " ms" << std::endl;
    }
}

// 测试3: HYBRID 混合模式
void testHybridMode(const std::string& model_path, int n_gpu_layers) {
    std::cout << "\n========== 测试3: HYBRID 混合模式 ==========" << std::endl;

    const std::string test_db = "test_query_hybrid.db";
    cleanupTestDB(test_db);

    QueryEngine::Config config;
    config.mode = QueryEngine::Mode::HYBRID;

    // 本地数据库配置
    config.db_config.db_path = test_db;
    config.db_config.enable_fts = true;

    // LLM 配置
    config.llm_type = "local";
    config.local_llm_config.model_path = model_path;
    config.local_llm_config.context_length = 4096;
    config.local_llm_config.n_gpu_layers = n_gpu_layers;
    config.local_llm_config.temperature = 0.7f;

    // 混合模式配置
    config.use_local_first = true;
    config.local_confidence_threshold = 0.5f;

    QueryEngine engine(config);

    std::cout << "初始化..." << std::endl;
    if (!engine.initialize()) {
        std::cout << "[FAIL] 初始化失败" << std::endl;
        return;
    }
    std::cout << "[PASS] 初始化成功" << std::endl;

    // 添加部分知识
    std::cout << "\n添加知识库数据..." << std::endl;
    for (const auto& [q, a, c] : KNOWLEDGE_BASE) {
        engine.addKnowledge(q, a, c);
    }

    // 测试查询 - 应该命中本地
    std::vector<std::string> local_queries = {
        "天气怎么样",
        "你好",
    };

    std::cout << "\n本地可命中的查询:" << std::endl;
    for (const auto& q : local_queries) {
        std::cout << "\n问: " << q << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        std::string answer = engine.query(q);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "答: " << answer << std::endl;
        std::cout << "耗时: " << duration.count() << " ms (应该很快)" << std::endl;
    }

    // 测试查询 - 应该回退到LLM
    std::vector<std::string> llm_queries = {
        "地球到月球有多远",
        "什么是量子力学",
    };

    std::cout << "\n需要LLM回答的查询:" << std::endl;
    for (const auto& q : llm_queries) {
        std::cout << "\n问: " << q << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        std::string answer = engine.query(q);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "答: " << answer << std::endl;
        std::cout << "耗时: " << duration.count() << " ms (LLM生成)" << std::endl;
    }

    cleanupTestDB(test_db);
}

// 测试4: 交互式测试
void testInteractive(const std::string& model_path, int n_gpu_layers) {
    std::cout << "\n========== 测试4: 交互式查询 ==========" << std::endl;

    const std::string test_db = "test_query_interactive.db";
    cleanupTestDB(test_db);

    QueryEngine::Config config;
    config.mode = QueryEngine::Mode::HYBRID;
    config.db_config.db_path = test_db;
    config.db_config.enable_fts = true;
    config.llm_type = "local";
    config.local_llm_config.model_path = model_path;
    config.local_llm_config.context_length = 4096;
    config.local_llm_config.n_gpu_layers = n_gpu_layers;
    config.local_llm_config.temperature = 0.7f;
    config.use_local_first = true;
    config.local_confidence_threshold = 0.5f;

    QueryEngine engine(config);

    std::cout << "初始化..." << std::endl;
    if (!engine.initialize()) {
        std::cout << "[FAIL] 初始化失败" << std::endl;
        return;
    }

    // 添加知识
    for (const auto& [q, a, c] : KNOWLEDGE_BASE) {
        engine.addKnowledge(q, a, c);
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "交互式查询模式" << std::endl;
    std::cout << "输入问题进行查询，输入 'quit' 退出" << std::endl;
    std::cout << "输入 'add:问题|答案' 添加知识" << std::endl;
    std::cout << "========================================\n" << std::endl;

    std::string input;
    while (true) {
        std::cout << "问: ";
        std::getline(std::cin, input);

        if (input == "quit" || input == "exit") {
            break;
        }

        if (input.empty()) {
            continue;
        }

        // 添加知识命令
        if (input.substr(0, 4) == "add:") {
            std::string content = input.substr(4);
            size_t pos = content.find('|');
            if (pos != std::string::npos) {
                std::string q = content.substr(0, pos);
                std::string a = content.substr(pos + 1);
                if (engine.addKnowledge(q, a)) {
                    std::cout << "已添加知识: " << q << std::endl;
                } else {
                    std::cout << "添加失败" << std::endl;
                }
            } else {
                std::cout << "格式错误，使用: add:问题|答案" << std::endl;
            }
            continue;
        }

        // 查询
        auto start = std::chrono::high_resolution_clock::now();
        std::string answer = engine.query(input);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "答: " << answer << std::endl;
        std::cout << "[" << duration.count() << " ms]\n" << std::endl;
    }

    std::cout << "再见!" << std::endl;
    cleanupTestDB(test_db);
}

void printUsage(const char* prog) {
    std::cout << "用法: " << prog << " [选项]" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  --model <路径>     LLM 模型路径 (GGUF格式，LLM和HYBRID模式需要)" << std::endl;
    std::cout << "  --gpu-layers <n>   GPU层数 (-1=全部, 默认35)" << std::endl;
    std::cout << "  --local            LOCAL_ONLY 模式测试 (不需要LLM模型)" << std::endl;
    std::cout << "  --llm              LLM_ONLY 模式测试" << std::endl;
    std::cout << "  --hybrid           HYBRID 混合模式测试" << std::endl;
    std::cout << "  --interactive      交互式测试" << std::endl;
    std::cout << "  --all              运行所有测试" << std::endl;
    std::cout << "  --help             显示帮助" << std::endl;
    std::cout << "\n示例:" << std::endl;
    std::cout << "  " << prog << " --local" << std::endl;
    std::cout << "  " << prog << " --model models/qwen2-7b-instruct-q4_0.gguf --hybrid" << std::endl;
    std::cout << "  " << prog << " --model models/qwen2-7b-instruct-q4_0.gguf --interactive" << std::endl;
}

int main(int argc, char* argv[]) {
    Logger::getInstance().setLevel(LogLevel::INFO);

    std::cout << "===== QueryEngine 查询引擎测试 =====" << std::endl;

    if (argc < 2) {
        printUsage(argv[0]);
        return 0;
    }

    std::string model_path;
    int n_gpu_layers = 35;
    bool run_local = false;
    bool run_llm = false;
    bool run_hybrid = false;
    bool run_interactive = false;
    bool run_all = false;

    // 解析参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--gpu-layers" && i + 1 < argc) {
            n_gpu_layers = std::atoi(argv[++i]);
        } else if (arg == "--local") {
            run_local = true;
        } else if (arg == "--llm") {
            run_llm = true;
        } else if (arg == "--hybrid") {
            run_hybrid = true;
        } else if (arg == "--interactive") {
            run_interactive = true;
        } else if (arg == "--all") {
            run_all = true;
        }
    }

    // LOCAL模式不需要LLM模型
    if (run_local && !run_llm && !run_hybrid && !run_interactive && !run_all) {
        testLocalOnlyMode();
        std::cout << "\n测试完成!" << std::endl;
        return 0;
    }

    // 其他模式需要LLM模型
    if ((run_llm || run_hybrid || run_interactive || run_all) && model_path.empty()) {
        std::cerr << "错误: LLM/HYBRID/交互模式需要指定模型路径 (--model)" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    if (!model_path.empty()) {
        std::cout << "模型: " << model_path << std::endl;
        std::cout << "GPU层数: " << n_gpu_layers << std::endl;
    }

    // 运行测试
    if (run_all || run_local) {
        testLocalOnlyMode();
    }

    if (run_all || run_llm) {
        testLLMOnlyMode(model_path, n_gpu_layers);
    }

    if (run_all || run_hybrid) {
        testHybridMode(model_path, n_gpu_layers);
    }

    if (run_interactive) {
        testInteractive(model_path, n_gpu_layers);
    }

    std::cout << "\n测试完成!" << std::endl;
    return 0;
}
