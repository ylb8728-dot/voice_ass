/**
 * @file test_llm_client.cpp
 * @brief LLM 客户端测试
 *
 * 测试内容:
 * 1. 本地 LLM 初始化
 * 2. 本地 LLM 查询
 * 3. 流式生成测试
 * 4. 性能基准测试
 */

#include "query/llm_client.h"
#include "utils/logger.h"
#include <iostream>
#include <chrono>
#include <vector>

using namespace voice_assistant::query;
using namespace voice_assistant::utils;

// 测试1: 本地 LLM 初始化
bool testLocalLLMInit(const std::string& model_path, int n_gpu_layers) {
    std::cout << "\n========== 测试1: 本地 LLM 初始化 ==========" << std::endl;

    LocalLLMClient::Config config;
    config.model_path = model_path;
    config.context_length = 4096;
    config.n_gpu_layers = n_gpu_layers;
    config.temperature = 0.7f;
    config.top_p = 0.9f;
    config.top_k = 40;

    std::cout << "配置:" << std::endl;
    std::cout << "  模型: " << config.model_path << std::endl;
    std::cout << "  上下文长度: " << config.context_length << std::endl;
    std::cout << "  GPU层数: " << (config.n_gpu_layers == -1 ? "全部" : std::to_string(config.n_gpu_layers)) << std::endl;
    std::cout << "  温度: " << config.temperature << std::endl;

    LocalLLMClient client(config);

    auto start = std::chrono::high_resolution_clock::now();
    bool result = client.initialize();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (result) {
        std::cout << "[PASS] 初始化成功! 耗时: " << duration.count() << " ms" << std::endl;
    } else {
        std::cout << "[FAIL] 初始化失败!" << std::endl;
    }

    return result;
}

// 测试2: 基本查询测试
void testBasicQuery(const std::string& model_path, int n_gpu_layers) {
    std::cout << "\n========== 测试2: 基本查询测试 ==========" << std::endl;

    LocalLLMClient::Config config;
    config.model_path = model_path;
    config.context_length = 4096;
    config.n_gpu_layers = n_gpu_layers;
    config.temperature = 0.7f;

    LocalLLMClient client(config);
    if (!client.initialize()) {
        std::cerr << "初始化失败!" << std::endl;
        return;
    }

    // 测试问题
    std::vector<std::pair<std::string, int>> test_queries = {
        {"你好，请简单介绍一下自己。", 128},
        {"1+1等于多少？", 32},
        {"什么是人工智能？请用一句话解释。", 64},
    };

    for (const auto& [query, max_tokens] : test_queries) {
        std::cout << "\n问题: " << query << std::endl;
        std::cout << "最大token: " << max_tokens << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        std::string response = client.query(query, max_tokens);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "回答: " << response << std::endl;
        std::cout << "耗时: " << duration.count() << " ms" << std::endl;
    }
}

// 测试3: 不同温度测试
void testTemperature(const std::string& model_path, int n_gpu_layers) {
    std::cout << "\n========== 测试3: 温度参数测试 ==========" << std::endl;

    std::string query = "用一句话描述天空。";
    float temperatures[] = {0.0f, 0.5f, 1.0f};

    for (float temp : temperatures) {
        std::cout << "\n温度: " << temp << std::endl;

        LocalLLMClient::Config config;
        config.model_path = model_path;
        config.context_length = 4096;
        config.n_gpu_layers = n_gpu_layers;
        config.temperature = temp;

        LocalLLMClient client(config);
        if (!client.initialize()) {
            std::cerr << "初始化失败!" << std::endl;
            continue;
        }

        // 同一问题多次查询，观察变化
        for (int i = 0; i < 2; ++i) {
            std::string response = client.query(query, 64);
            std::cout << "  响应" << (i+1) << ": " << response << std::endl;
        }
    }
}

// 测试4: 性能基准测试
void testBenchmark(const std::string& model_path, int n_gpu_layers) {
    std::cout << "\n========== 测试4: 性能基准测试 ==========" << std::endl;

    LocalLLMClient::Config config;
    config.model_path = model_path;
    config.context_length = 4096;
    config.n_gpu_layers = n_gpu_layers;
    config.temperature = 0.0f;  // 贪婪解码

    LocalLLMClient client(config);
    if (!client.initialize()) {
        std::cerr << "初始化失败!" << std::endl;
        return;
    }

    std::string query = "请简单介绍一下中国。";
    int max_tokens_list[] = {32, 64, 128, 256};

    std::cout << "\n不同输出长度的性能:" << std::endl;
    std::cout << "Max Tokens\t耗时(ms)\t约Token/s" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    for (int max_tokens : max_tokens_list) {
        auto start = std::chrono::high_resolution_clock::now();
        std::string response = client.query(query, max_tokens);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // 粗略估计生成的token数（假设每个汉字约1.5个token）
        int approx_tokens = response.size() / 3;
        float tokens_per_sec = approx_tokens * 1000.0f / (duration.count() + 1);

        std::cout << max_tokens << "\t\t" << duration.count() << "\t\t"
                  << tokens_per_sec << std::endl;
    }
}

// 测试5: 长上下文测试
void testLongContext(const std::string& model_path, int n_gpu_layers) {
    std::cout << "\n========== 测试5: 长上下文测试 ==========" << std::endl;

    LocalLLMClient::Config config;
    config.model_path = model_path;
    config.context_length = 4096;
    config.n_gpu_layers = n_gpu_layers;
    config.temperature = 0.7f;

    LocalLLMClient client(config);
    if (!client.initialize()) {
        std::cerr << "初始化失败!" << std::endl;
        return;
    }

    // 构建长上下文
    std::string long_context = "以下是一段对话历史:\n";
    for (int i = 1; i <= 5; ++i) {
        long_context += "用户: 问题" + std::to_string(i) + "\n";
        long_context += "助手: 这是对问题" + std::to_string(i) + "的回答。\n";
    }
    long_context += "用户: 请总结一下我们的对话。\n助手:";

    std::cout << "输入长度: " << long_context.size() << " 字符" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    std::string response = client.query(long_context, 128);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "回答: " << response << std::endl;
    std::cout << "耗时: " << duration.count() << " ms" << std::endl;
}

// 测试6: 远程 LLM 客户端测试
void testRemoteLLM() {
    std::cout << "\n========== 测试6: 远程 LLM 客户端测试 ==========" << std::endl;

    RemoteLLMClient::Config config;
    config.api_url = "http://localhost:8000/v1/chat/completions";
    config.api_key = "";
    config.model_name = "gpt-3.5-turbo";
    config.temperature = 0.7f;

    std::cout << "配置:" << std::endl;
    std::cout << "  API URL: " << config.api_url << std::endl;
    std::cout << "  模型: " << config.model_name << std::endl;

    RemoteLLMClient client(config);

    bool init_result = client.initialize();
    std::cout << "初始化: " << (init_result ? "[PASS]" : "[FAIL]") << std::endl;

    if (init_result) {
        std::string response = client.query("你好", 64);
        std::cout << "查询响应: " << response << std::endl;
        std::cout << "[INFO] 注意: 远程LLM当前是占位实现" << std::endl;
    }
}

void printUsage(const char* prog) {
    std::cout << "用法: " << prog << " --model <模型路径> [选项]" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  --model <路径>     LLM 模型路径 (GGUF格式)" << std::endl;
    std::cout << "  --gpu-layers <n>   GPU层数 (-1=全部, 默认35)" << std::endl;
    std::cout << "  --init             只测试初始化" << std::endl;
    std::cout << "  --basic            基本查询测试" << std::endl;
    std::cout << "  --temp             温度参数测试" << std::endl;
    std::cout << "  --bench            性能基准测试" << std::endl;
    std::cout << "  --context          长上下文测试" << std::endl;
    std::cout << "  --remote           远程LLM测试" << std::endl;
    std::cout << "  --all              运行所有测试" << std::endl;
    std::cout << "  --help             显示帮助" << std::endl;
    std::cout << "\n示例:" << std::endl;
    std::cout << "  " << prog << " --model models/qwen2-7b-instruct-q4_0.gguf --all" << std::endl;
    std::cout << "  " << prog << " --model models/qwen2-7b-instruct-q4_0.gguf --basic" << std::endl;
}

int main(int argc, char* argv[]) {
    Logger::getInstance().setLevel(LogLevel::INFO);

    std::cout << "===== LLM 客户端测试 =====" << std::endl;

    if (argc < 2) {
        printUsage(argv[0]);
        return 0;
    }

    std::string model_path;
    int n_gpu_layers = 35;
    bool run_init = false;
    bool run_basic = false;
    bool run_temp = false;
    bool run_bench = false;
    bool run_context = false;
    bool run_remote = false;
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
        } else if (arg == "--init") {
            run_init = true;
        } else if (arg == "--basic") {
            run_basic = true;
        } else if (arg == "--temp") {
            run_temp = true;
        } else if (arg == "--bench") {
            run_bench = true;
        } else if (arg == "--context") {
            run_context = true;
        } else if (arg == "--remote") {
            run_remote = true;
        } else if (arg == "--all") {
            run_all = true;
        }
    }

    // 远程LLM测试不需要模型
    if (run_remote && !run_all && !run_init && !run_basic && !run_temp && !run_bench && !run_context) {
        testRemoteLLM();
        return 0;
    }

    if (model_path.empty()) {
        std::cerr << "错误: 必须指定模型路径 (--model)" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    std::cout << "模型: " << model_path << std::endl;
    std::cout << "GPU层数: " << n_gpu_layers << std::endl;

    // 运行测试
    if (run_all || run_init) {
        if (!testLocalLLMInit(model_path, n_gpu_layers)) {
            std::cerr << "初始化失败，跳过后续测试" << std::endl;
            return 1;
        }
    }

    if (run_all || run_basic) {
        testBasicQuery(model_path, n_gpu_layers);
    }

    if (run_all || run_temp) {
        testTemperature(model_path, n_gpu_layers);
    }

    if (run_all || run_bench) {
        testBenchmark(model_path, n_gpu_layers);
    }

    if (run_all || run_context) {
        testLongContext(model_path, n_gpu_layers);
    }

    if (run_all || run_remote) {
        testRemoteLLM();
    }

    std::cout << "\n测试完成!" << std::endl;
    return 0;
}
