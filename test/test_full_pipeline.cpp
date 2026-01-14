/**
 * @file test_full_pipeline.cpp
 * @brief 完整流水线测试: 语音输入 + ASR + Query
 *
 * 测试流程:
 * 麦克风采集 -> VAD检测 -> ASR识别 -> QueryEngine查询 -> 输出答案
 */

#include "audio/audio_input.h"
#include "audio/vad_detector.h"
#include "asr/asr_engine.h"
#include "query/query_engine.h"
#include "utils/logger.h"
#include <iostream>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <queue>
#include <csignal>
#include <chrono>
#include <cmath>

using namespace voice_assistant::audio;
using namespace voice_assistant::asr;
using namespace voice_assistant::query;
using namespace voice_assistant::utils;

// 全局控制
std::atomic<bool> g_running{true};
std::mutex g_mutex;
std::condition_variable g_cv;

// 语音段队列
struct SpeechSegment {
    std::vector<float> audio;
    std::chrono::steady_clock::time_point timestamp;
};

std::queue<SpeechSegment> g_speech_queue;

void signalHandler(int signum) {
    (void)signum;
    std::cout << "\n收到退出信号..." << std::endl;
    g_running = false;
    g_cv.notify_all();
}

// 显示音量条
void displayVolumeMeter(float rms, bool is_speech) {
    const int bar_width = 30;
    float db = 20.0f * std::log10(rms + 1e-10f);
    float normalized = (db + 60.0f) / 60.0f;
    normalized = std::max(0.0f, std::min(1.0f, normalized));

    int filled = static_cast<int>(normalized * bar_width);

    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < filled) {
            std::cout << (is_speech ? "#" : "=");
        } else {
            std::cout << " ";
        }
    }
    std::cout << "] " << (is_speech ? "SPEECH" : "      ");
    std::cout << std::flush;
}

// 处理线程: ASR + Query
void processingThread(ASREngine* asr_engine, QueryEngine* query_engine) {
    int segment_count = 0;

    while (g_running) {
        SpeechSegment segment;

        // 等待语音段
        {
            std::unique_lock<std::mutex> lock(g_mutex);
            g_cv.wait(lock, [] { return !g_speech_queue.empty() || !g_running; });

            if (!g_running && g_speech_queue.empty()) {
                break;
            }

            if (!g_speech_queue.empty()) {
                segment = std::move(g_speech_queue.front());
                g_speech_queue.pop();
            } else {
                continue;
            }
        }

        segment_count++;
        float duration = segment.audio.size() / 16000.0f;

        std::cout << "\n" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "语音段 #" << segment_count << " (" << duration << " 秒)" << std::endl;

        // 1. ASR 识别
        std::cout << "\n[ASR] 识别中..." << std::endl;
        auto asr_start = std::chrono::high_resolution_clock::now();
        std::string text = asr_engine->recognize(segment.audio);
        auto asr_end = std::chrono::high_resolution_clock::now();
        auto asr_time = std::chrono::duration_cast<std::chrono::milliseconds>(asr_end - asr_start);

        std::cout << "[ASR] 识别结果: \"" << text << "\"" << std::endl;
        std::cout << "[ASR] 耗时: " << asr_time.count() << " ms" << std::endl;

        // 2. Query 查询
        if (!text.empty()) {
            std::cout << "\n[Query] 查询中..." << std::endl;
            auto query_start = std::chrono::high_resolution_clock::now();
            std::string answer = query_engine->query(text);
            auto query_end = std::chrono::high_resolution_clock::now();
            auto query_time = std::chrono::duration_cast<std::chrono::milliseconds>(query_end - query_start);

            std::cout << "[Query] 答案: \"" << answer << "\"" << std::endl;
            std::cout << "[Query] 耗时: " << query_time.count() << " ms" << std::endl;

            std::cout << "\n总耗时: " << (asr_time.count() + query_time.count()) << " ms" << std::endl;
        } else {
            std::cout << "[Skip] 未识别到有效文本" << std::endl;
        }

        std::cout << "========================================" << std::endl;
        std::cout << std::flush;
    }

    std::cout << "处理线程退出" << std::endl;
}

// 预置知识库
const std::vector<std::tuple<std::string, std::string, std::string>> KNOWLEDGE_BASE = {
    {"今天天气怎么样", "今天天气晴朗，温度约25度，适合户外活动。", "天气"},
    {"明天会下雨吗", "根据天气预报，明天多云转小雨。", "天气"},
    {"现在几点", "请查看您设备上的时钟。", "时间"},
    {"你是谁", "我是一个语音助手，可以回答您的问题。", "介绍"},
    {"你能做什么", "我可以回答问题、提供信息等。", "介绍"},
    {"播放音乐", "正在为您播放轻音乐。", "娱乐"},
    {"讲个笑话", "为什么程序员喜欢黑暗？因为Light是个Bug！", "娱乐"},
    {"什么是人工智能", "人工智能是让计算机模拟人类智能的技术。", "科技"},
    {"你好", "你好！有什么可以帮您的？", "问候"},
    {"再见", "再见，祝您生活愉快！", "问候"},
    {"谢谢", "不客气，这是我应该做的！", "问候"},
};

// 主测试函数
void runFullPipeline(const std::string& asr_model_path,
                     const std::string& llm_model_path,
                     bool use_gpu,
                     QueryEngine::Mode query_mode) {

    std::cout << "\n===== 完整流水线测试: 语音 + ASR + Query =====" << std::endl;

    // 1. 初始化 ASR 引擎
    std::cout << "\n[1/4] 初始化 ASR 引擎..." << std::endl;
    ASREngine::Config asr_config;
    asr_config.model_path = asr_model_path;
    asr_config.language = "zh";
    asr_config.num_threads = 4;
    asr_config.use_gpu = use_gpu;
    asr_config.no_timestamps = true;
    asr_config.temperature = 0.0f;

    ASREngine asr_engine(asr_config);
    if (!asr_engine.initialize()) {
        std::cerr << "ASR 引擎初始化失败!" << std::endl;
        return;
    }
    std::cout << "ASR 引擎初始化成功" << std::endl;

    // 2. 初始化 Query 引擎
    std::cout << "\n[2/4] 初始化 Query 引擎 (模式: ";
    switch (query_mode) {
        case QueryEngine::Mode::LOCAL_ONLY: std::cout << "LOCAL_ONLY"; break;
        case QueryEngine::Mode::LLM_ONLY: std::cout << "LLM_ONLY"; break;
        case QueryEngine::Mode::HYBRID: std::cout << "HYBRID"; break;
    }
    std::cout << ")..." << std::endl;

    QueryEngine::Config query_config;
    query_config.mode = query_mode;
    query_config.db_config.db_path = "test_full_pipeline.db";
    query_config.db_config.enable_fts = true;

    if (query_mode == QueryEngine::Mode::LLM_ONLY ||
        query_mode == QueryEngine::Mode::HYBRID) {
        query_config.llm_type = "local";
        query_config.local_llm_config.model_path = llm_model_path;
        query_config.local_llm_config.context_length = 4096;
        query_config.local_llm_config.n_gpu_layers = use_gpu ? 35 : 0;
        query_config.local_llm_config.temperature = 0.7f;
    }

    query_config.use_local_first = true;
    query_config.local_confidence_threshold = 0.5f;

    QueryEngine query_engine(query_config);
    if (!query_engine.initialize()) {
        std::cerr << "Query 引擎初始化失败!" << std::endl;
        return;
    }
    std::cout << "Query 引擎初始化成功" << std::endl;

    // 添加预置知识
    if (query_mode == QueryEngine::Mode::LOCAL_ONLY ||
        query_mode == QueryEngine::Mode::HYBRID) {
        std::cout << "添加知识库数据..." << std::endl;
        for (const auto& [q, a, c] : KNOWLEDGE_BASE) {
            query_engine.addKnowledge(q, a, c);
        }
        std::cout << "已添加 " << KNOWLEDGE_BASE.size() << " 条知识" << std::endl;
    }

    // 3. 初始化 VAD
    std::cout << "\n[3/4] 初始化 VAD 检测器..." << std::endl;
    VADDetector::Config vad_config;
    vad_config.energy_threshold = 0.008f;   // 稍微提高能量阈值，减少误触发
    vad_config.zero_crossing_threshold = 0.5f;
    vad_config.min_speech_frames = 8;       // 需要连续8帧语音才确认开始
    vad_config.min_silence_frames = 40;     // 需要连续40帧静音才确认结束
    vad_config.frame_size = 256;

    VADDetector vad(vad_config);
    std::cout << "VAD 检测器初始化完成" << std::endl;
    std::cout << "  能量阈值: " << vad_config.energy_threshold << std::endl;
    std::cout << "  最小语音帧: " << vad_config.min_speech_frames << std::endl;
    std::cout << "  最小静音帧: " << vad_config.min_silence_frames << std::endl;

    // 4. 初始化音频输入
    std::cout << "\n[4/4] 初始化音频输入..." << std::endl;
    AudioInput audio_input(16000, 1, 256);
    if (!audio_input.initialize()) {
        std::cerr << "音频输入初始化失败!" << std::endl;
        return;
    }
    std::cout << "音频输入初始化成功" << std::endl;

    // 启动处理线程
    std::thread process_thread(processingThread, &asr_engine, &query_engine);

    // 音频缓冲区和状态
    std::vector<float> speech_buffer;
    speech_buffer.reserve(16000 * 30);
    bool was_in_speech = false;
    bool segment_submitted = false;  // 防止重复提交
    const size_t min_speech_samples = 16000;  // 最小1秒
    int silence_count_after_speech = 0;  // 语音结束后的静音计数
    const int required_silence_frames = 20;  // 需要连续20帧静音才确认语音结束

    // 音频回调
    auto callback = [&](const float* data, size_t frames) {
        bool is_speech = vad.isSpeech(data, frames);

        // 计算RMS显示
        float rms = 0.0f;
        for (size_t i = 0; i < frames; ++i) {
            rms += data[i] * data[i];
        }
        rms = std::sqrt(rms / frames);
        displayVolumeMeter(rms, is_speech);

        if (is_speech) {
            // 正在说话
            speech_buffer.insert(speech_buffer.end(), data, data + frames);
            was_in_speech = true;
            segment_submitted = false;  // 重置提交标志
            silence_count_after_speech = 0;  // 重置静音计数
        } else if (was_in_speech && !segment_submitted) {
            // 语音刚结束，但还需要确认
            silence_count_after_speech++;

            // 继续收集一小段尾音（防止截断）
            if (silence_count_after_speech <= 5) {
                speech_buffer.insert(speech_buffer.end(), data, data + frames);
            }

            // 确认语音结束
            if (silence_count_after_speech >= required_silence_frames) {
                if (speech_buffer.size() >= min_speech_samples) {
                    SpeechSegment segment;
                    segment.audio = std::move(speech_buffer);
                    segment.timestamp = std::chrono::steady_clock::now();

                    {
                        std::lock_guard<std::mutex> lock(g_mutex);
                        g_speech_queue.push(std::move(segment));
                    }
                    g_cv.notify_one();
                    segment_submitted = true;  // 标记已提交
                }

                speech_buffer.clear();
                speech_buffer.reserve(16000 * 30);
                was_in_speech = false;
                silence_count_after_speech = 0;
            }
        }
    };

    // 开始录音
    std::cout << "\n========================================" << std::endl;
    std::cout << "系统就绪! 请开始说话..." << std::endl;
    std::cout << "按 Ctrl+C 退出" << std::endl;
    std::cout << "========================================\n" << std::endl;

    if (!audio_input.start(callback)) {
        std::cerr << "启动录音失败!" << std::endl;
        g_running = false;
        g_cv.notify_all();
        process_thread.join();
        return;
    }

    // 主循环
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 清理
    std::cout << "\n正在停止..." << std::endl;
    audio_input.stop();

    // 处理剩余语音
    if (speech_buffer.size() >= min_speech_samples) {
        SpeechSegment segment;
        segment.audio = std::move(speech_buffer);
        segment.timestamp = std::chrono::steady_clock::now();

        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_speech_queue.push(std::move(segment));
        }
        g_cv.notify_one();
    }

    g_running = false;
    g_cv.notify_all();
    process_thread.join();

    // 清理临时数据库
    std::remove("test_full_pipeline.db");

    std::cout << "测试结束" << std::endl;
}

void printUsage(const char* prog) {
    std::cout << "用法: " << prog << " --asr-model <ASR模型> [选项]" << std::endl;
    std::cout << "\n必需参数:" << std::endl;
    std::cout << "  --asr-model <路径>   Whisper ASR 模型路径" << std::endl;
    std::cout << "\n可选参数:" << std::endl;
    std::cout << "  --llm-model <路径>   LLM 模型路径 (LLM/HYBRID模式需要)" << std::endl;
    std::cout << "  --mode <模式>        查询模式: local, llm, hybrid (默认: local)" << std::endl;
    std::cout << "  --no-gpu             禁用 GPU" << std::endl;
    std::cout << "  --help               显示帮助" << std::endl;
    std::cout << "\n示例:" << std::endl;
    std::cout << "  # 仅使用本地知识库 (无需LLM模型，响应快)" << std::endl;
    std::cout << "  " << prog << " --asr-model whisper.cpp/models/ggml-large-v3-turbo.bin --mode local" << std::endl;
    std::cout << "\n  # 使用LLM回答" << std::endl;
    std::cout << "  " << prog << " --asr-model whisper.cpp/models/ggml-large-v3-turbo.bin \\" << std::endl;
    std::cout << "               --llm-model models/qwen2-7b-instruct-q4_0.gguf --mode llm" << std::endl;
    std::cout << "\n  # 混合模式 (优先本地，不确定时用LLM)" << std::endl;
    std::cout << "  " << prog << " --asr-model whisper.cpp/models/ggml-large-v3-turbo.bin \\" << std::endl;
    std::cout << "               --llm-model models/qwen2-7b-instruct-q4_0.gguf --mode hybrid" << std::endl;
}

int main(int argc, char* argv[]) {
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    Logger::getInstance().setLevel(LogLevel::INFO);

    std::cout << "===== 完整流水线测试: 语音输入 + ASR + Query =====" << std::endl;

    if (argc < 2) {
        printUsage(argv[0]);
        return 0;
    }

    std::string asr_model_path;
    std::string llm_model_path;
    bool use_gpu = true;
    QueryEngine::Mode query_mode = QueryEngine::Mode::LOCAL_ONLY;

    // 解析参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--asr-model" && i + 1 < argc) {
            asr_model_path = argv[++i];
        } else if (arg == "--llm-model" && i + 1 < argc) {
            llm_model_path = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "local") {
                query_mode = QueryEngine::Mode::LOCAL_ONLY;
            } else if (mode == "llm") {
                query_mode = QueryEngine::Mode::LLM_ONLY;
            } else if (mode == "hybrid") {
                query_mode = QueryEngine::Mode::HYBRID;
            } else {
                std::cerr << "未知模式: " << mode << std::endl;
                return 1;
            }
        } else if (arg == "--no-gpu") {
            use_gpu = false;
        }
    }

    if (asr_model_path.empty()) {
        std::cerr << "错误: 必须指定 ASR 模型路径 (--asr-model)" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    if ((query_mode == QueryEngine::Mode::LLM_ONLY ||
         query_mode == QueryEngine::Mode::HYBRID) && llm_model_path.empty()) {
        std::cerr << "错误: LLM/HYBRID 模式需要指定 LLM 模型路径 (--llm-model)" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    std::cout << "配置:" << std::endl;
    std::cout << "  ASR模型: " << asr_model_path << std::endl;
    if (!llm_model_path.empty()) {
        std::cout << "  LLM模型: " << llm_model_path << std::endl;
    }
    std::cout << "  查询模式: " << (query_mode == QueryEngine::Mode::LOCAL_ONLY ? "LOCAL" :
                                   query_mode == QueryEngine::Mode::LLM_ONLY ? "LLM" : "HYBRID") << std::endl;
    std::cout << "  GPU: " << (use_gpu ? "启用" : "禁用") << std::endl;

    runFullPipeline(asr_model_path, llm_model_path, use_gpu, query_mode);

    return 0;
}
