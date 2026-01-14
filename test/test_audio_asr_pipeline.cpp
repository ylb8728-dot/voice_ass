/**
 * @file test_audio_asr_pipeline.cpp
 * @brief 音频输入 + VAD + ASR 完整流水线测试
 *
 * 测试完整流程:
 * 麦克风采集 -> VAD检测 -> 语音段累积 -> ASR识别 -> 输出结果
 *
 * 这是一个交互式测试程序，实时识别麦克风输入的语音
 */

#include "audio/audio_input.h"
#include "audio/vad_detector.h"
#include "asr/asr_engine.h"
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
#include <fstream>

using namespace voice_assistant::audio;
using namespace voice_assistant::asr;
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

// WAV 文件保存
struct WavHeader {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t file_size = 0;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_size = 16;
    uint16_t audio_format = 3;
    uint16_t num_channels = 1;
    uint32_t sample_rate = 16000;
    uint32_t byte_rate = 64000;
    uint16_t block_align = 4;
    uint16_t bits_per_sample = 32;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size = 0;
};

bool saveWav(const std::string& filename, const std::vector<float>& audio) {
    WavHeader header;
    header.data_size = audio.size() * sizeof(float);
    header.file_size = 36 + header.data_size;

    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    file.write(reinterpret_cast<const char*>(audio.data()), audio.size() * sizeof(float));
    return true;
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

// ASR 工作线程
void asrWorkerThread(ASREngine* engine, bool save_audio) {
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

        std::cout << "\n========================================" << std::endl;
        std::cout << "语音段 #" << segment_count << " (" << duration << " 秒, "
                  << segment.audio.size() << " 样本)" << std::endl;

        // 保存音频（可选）
        if (save_audio) {
            std::string filename = "segment_" + std::to_string(segment_count) + ".wav";
            if (saveWav(filename, segment.audio)) {
                std::cout << "已保存: " << filename << std::endl;
            }
        }

        // 执行 ASR 识别
        auto start = std::chrono::high_resolution_clock::now();
        std::string result = engine->recognize(segment.audio);
        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        float rtf = elapsed.count() / 1000.0f / duration;

        std::cout << "识别结果: \"" << result << "\"" << std::endl;
        std::cout << "耗时: " << elapsed.count() << " ms, RTF: " << rtf << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::flush;
    }

    std::cout << "ASR 工作线程退出" << std::endl;
}

// 主测试函数
void runPipelineTest(const std::string& model_path, bool use_gpu, bool save_audio) {
    std::cout << "\n===== 音频-VAD-ASR 流水线测试 =====" << std::endl;

    // 初始化 ASR 引擎
    std::cout << "\n[1/3] 初始化 ASR 引擎..." << std::endl;

    ASREngine::Config asr_config;
    asr_config.model_path = model_path;
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

    // 初始化 VAD
    std::cout << "\n[2/3] 初始化 VAD 检测器..." << std::endl;

    VADDetector::Config vad_config;
    vad_config.energy_threshold = 0.005f;     // 能量阈值
    vad_config.zero_crossing_threshold = 0.5f;
    vad_config.min_speech_frames = 5;         // 最小语音帧数 (5 * 256 / 16000 = 80ms)
    vad_config.min_silence_frames = 30;       // 最小静音帧数 (30 * 256 / 16000 = 480ms)
    vad_config.frame_size = 256;

    VADDetector vad(vad_config);
    std::cout << "VAD 检测器初始化完成" << std::endl;

    // 初始化音频输入
    std::cout << "\n[3/3] 初始化音频输入..." << std::endl;

    AudioInput audio_input(16000, 1, 256);
    if (!audio_input.initialize()) {
        std::cerr << "音频输入初始化失败!" << std::endl;
        return;
    }
    std::cout << "音频输入初始化成功" << std::endl;

    // 启动 ASR 工作线程
    std::thread asr_thread(asrWorkerThread, &asr_engine, save_audio);

    // 音频缓冲区和状态
    std::vector<float> speech_buffer;
    speech_buffer.reserve(16000 * 30);  // 预留 30 秒
    bool was_in_speech = false;
    const size_t min_speech_samples = 16000;  // 最小 1 秒

    // 音频回调
    auto callback = [&](const float* data, size_t frames) {
        // VAD 检测
        bool is_speech = vad.isSpeech(data, frames);

        // 计算 RMS 用于显示
        float rms = 0.0f;
        for (size_t i = 0; i < frames; ++i) {
            rms += data[i] * data[i];
        }
        rms = std::sqrt(rms / frames);
        displayVolumeMeter(rms, is_speech);

        // 语音段状态机
        if (is_speech) {
            // 在语音段中，累积音频
            speech_buffer.insert(speech_buffer.end(), data, data + frames);
            was_in_speech = true;
        } else if (was_in_speech) {
            // 语音结束，检查是否有足够的音频
            if (speech_buffer.size() >= min_speech_samples) {
                // 提交到队列
                SpeechSegment segment;
                segment.audio = std::move(speech_buffer);
                segment.timestamp = std::chrono::steady_clock::now();

                {
                    std::lock_guard<std::mutex> lock(g_mutex);
                    g_speech_queue.push(std::move(segment));
                }
                g_cv.notify_one();
            }

            speech_buffer.clear();
            speech_buffer.reserve(16000 * 30);
            was_in_speech = false;
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
        asr_thread.join();
        return;
    }

    // 主循环
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 清理
    std::cout << "\n正在停止..." << std::endl;
    audio_input.stop();

    // 处理剩余的语音
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

    // 等待 ASR 线程处理完
    g_running = false;
    g_cv.notify_all();
    asr_thread.join();

    std::cout << "测试结束" << std::endl;
}

// 简单的录音-识别测试（无 VAD）
void runSimpleTest(const std::string& model_path, int duration_seconds, bool use_gpu) {
    std::cout << "\n===== 简单录音识别测试 (" << duration_seconds << " 秒) =====" << std::endl;

    // 初始化 ASR 引擎
    std::cout << "初始化 ASR 引擎..." << std::endl;

    ASREngine::Config asr_config;
    asr_config.model_path = model_path;
    asr_config.language = "zh";
    asr_config.num_threads = 4;
    asr_config.use_gpu = use_gpu;

    ASREngine asr_engine(asr_config);
    if (!asr_engine.initialize()) {
        std::cerr << "ASR 引擎初始化失败!" << std::endl;
        return;
    }

    // 初始化音频输入
    std::cout << "初始化音频输入..." << std::endl;

    AudioInput audio_input(16000, 1, 256);
    if (!audio_input.initialize()) {
        std::cerr << "音频输入初始化失败!" << std::endl;
        return;
    }

    // 录音
    std::vector<float> recorded_audio;
    recorded_audio.reserve(16000 * duration_seconds);

    auto callback = [&recorded_audio](const float* data, size_t frames) {
        recorded_audio.insert(recorded_audio.end(), data, data + frames);
    };

    std::cout << "\n开始录音，请说话..." << std::endl;

    if (!audio_input.start(callback)) {
        std::cerr << "启动录音失败!" << std::endl;
        return;
    }

    auto start_time = std::chrono::steady_clock::now();
    while (g_running) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        int seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
        if (seconds >= duration_seconds) {
            break;
        }
        std::cout << "\r录音中... " << seconds << "/" << duration_seconds << " 秒" << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    audio_input.stop();
    std::cout << std::endl;

    if (recorded_audio.empty()) {
        std::cout << "未录到音频" << std::endl;
        return;
    }

    std::cout << "录音完成: " << recorded_audio.size() << " 样本" << std::endl;

    // 保存录音
    saveWav("simple_test_recording.wav", recorded_audio);
    std::cout << "已保存: simple_test_recording.wav" << std::endl;

    // 执行识别
    std::cout << "\n开始识别..." << std::endl;

    auto asr_start = std::chrono::high_resolution_clock::now();
    std::string result = asr_engine.recognize(recorded_audio);
    auto asr_end = std::chrono::high_resolution_clock::now();

    auto asr_duration = std::chrono::duration_cast<std::chrono::milliseconds>(asr_end - asr_start);

    std::cout << "识别结果: \"" << result << "\"" << std::endl;
    std::cout << "识别耗时: " << asr_duration.count() << " ms" << std::endl;
}

void printUsage(const char* prog) {
    std::cout << "用法: " << prog << " --model <模型路径> [选项]" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  --model <路径>     Whisper 模型路径 (必需)" << std::endl;
    std::cout << "  --no-gpu           禁用 GPU" << std::endl;
    std::cout << "  --save             保存每个语音段为 WAV 文件" << std::endl;
    std::cout << "  --simple [秒数]    简单录音测试 (默认5秒)" << std::endl;
    std::cout << "  --pipeline         运行完整流水线测试 (默认)" << std::endl;
    std::cout << "  --help             显示帮助" << std::endl;
    std::cout << "\n示例:" << std::endl;
    std::cout << "  " << prog << " --model whisper.cpp/models/ggml-large-v3-turbo.bin" << std::endl;
    std::cout << "  " << prog << " --model whisper.cpp/models/ggml-large-v3-turbo.bin --simple 10" << std::endl;
}

int main(int argc, char* argv[]) {
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    Logger::getInstance().setLevel(LogLevel::INFO);

    std::cout << "===== 音频输入 + VAD + ASR 综合测试 =====" << std::endl;

    if (argc < 2) {
        printUsage(argv[0]);
        return 0;
    }

    std::string model_path;
    bool use_gpu = true;
    bool save_audio = false;
    bool run_simple = false;
    int simple_duration = 5;
    bool run_pipeline = true;

    // 解析参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--no-gpu") {
            use_gpu = false;
        } else if (arg == "--save") {
            save_audio = true;
        } else if (arg == "--simple") {
            run_simple = true;
            run_pipeline = false;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                simple_duration = std::atoi(argv[++i]);
            }
        } else if (arg == "--pipeline") {
            run_pipeline = true;
            run_simple = false;
        }
    }

    if (model_path.empty()) {
        std::cerr << "错误: 必须指定模型路径 (--model)" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    std::cout << "配置:" << std::endl;
    std::cout << "  模型: " << model_path << std::endl;
    std::cout << "  GPU: " << (use_gpu ? "启用" : "禁用") << std::endl;
    std::cout << "  保存音频: " << (save_audio ? "是" : "否") << std::endl;

    if (run_simple) {
        runSimpleTest(model_path, simple_duration, use_gpu);
    } else if (run_pipeline) {
        runPipelineTest(model_path, use_gpu, save_audio);
    }

    std::cout << "\n程序结束" << std::endl;
    return 0;
}
