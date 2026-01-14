/**
 * @file test_asr_engine.cpp
 * @brief ASREngine 语音识别引擎测试
 *
 * 测试内容:
 * 1. 引擎初始化
 * 2. WAV 文件识别
 * 3. 流式识别测试
 * 4. 性能基准测试
 */

#include "asr/asr_engine.h"
#include "utils/logger.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstring>
#include <cmath>

using namespace voice_assistant::asr;
using namespace voice_assistant::utils;

// WAV 文件头结构
#pragma pack(push, 1)
struct WavHeader {
    char riff[4];
    uint32_t file_size;
    char wave[4];
    char fmt[4];
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data[4];
    uint32_t data_size;
};
#pragma pack(pop)

// 读取 WAV 文件
bool readWavFile(const std::string& filename, std::vector<float>& audio, int& sample_rate) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return false;
    }

    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    // 验证 RIFF 标识
    if (std::strncmp(header.riff, "RIFF", 4) != 0 ||
        std::strncmp(header.wave, "WAVE", 4) != 0) {
        std::cerr << "无效的 WAV 文件格式" << std::endl;
        return false;
    }

    sample_rate = header.sample_rate;

    std::cout << "WAV 文件信息:" << std::endl;
    std::cout << "  采样率: " << header.sample_rate << " Hz" << std::endl;
    std::cout << "  声道数: " << header.num_channels << std::endl;
    std::cout << "  位深度: " << header.bits_per_sample << " bits" << std::endl;
    std::cout << "  格式: " << (header.audio_format == 1 ? "PCM" :
                               (header.audio_format == 3 ? "IEEE Float" : "未知")) << std::endl;

    size_t num_samples = header.data_size / (header.bits_per_sample / 8) / header.num_channels;
    audio.resize(num_samples);

    if (header.audio_format == 3 && header.bits_per_sample == 32) {
        // IEEE Float32
        file.read(reinterpret_cast<char*>(audio.data()), header.data_size);
    } else if (header.audio_format == 1 && header.bits_per_sample == 16) {
        // PCM 16-bit
        std::vector<int16_t> pcm_data(num_samples);
        file.read(reinterpret_cast<char*>(pcm_data.data()), header.data_size);
        for (size_t i = 0; i < num_samples; ++i) {
            audio[i] = pcm_data[i] / 32768.0f;
        }
    } else {
        std::cerr << "不支持的音频格式" << std::endl;
        return false;
    }

    std::cout << "  样本数: " << num_samples << std::endl;
    std::cout << "  时长: " << (num_samples / (float)sample_rate) << " 秒" << std::endl;

    return true;
}

// 生成测试音频 (静音 + 正弦波)
std::vector<float> generateTestAudio(float duration_seconds, int sample_rate = 16000) {
    size_t num_samples = static_cast<size_t>(duration_seconds * sample_rate);
    std::vector<float> audio(num_samples);

    for (size_t i = 0; i < num_samples; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        // 生成 440Hz 正弦波
        audio[i] = 0.3f * std::sin(2.0f * M_PI * 440.0f * t);
    }

    return audio;
}

// 重采样 (简单线性插值)
std::vector<float> resample(const std::vector<float>& input, int from_rate, int to_rate) {
    if (from_rate == to_rate) {
        return input;
    }

    double ratio = static_cast<double>(to_rate) / from_rate;
    size_t output_size = static_cast<size_t>(input.size() * ratio);
    std::vector<float> output(output_size);

    for (size_t i = 0; i < output_size; ++i) {
        double src_idx = i / ratio;
        size_t idx0 = static_cast<size_t>(src_idx);
        size_t idx1 = std::min(idx0 + 1, input.size() - 1);
        float frac = static_cast<float>(src_idx - idx0);
        output[i] = input[idx0] * (1.0f - frac) + input[idx1] * frac;
    }

    return output;
}

// 测试1: 引擎初始化
bool testInitialize(const std::string& model_path, bool use_gpu) {
    std::cout << "\n========== 测试1: 引擎初始化 ==========" << std::endl;

    ASREngine::Config config;
    config.model_path = model_path;
    config.language = "zh";
    config.num_threads = 4;
    config.use_gpu = use_gpu;
    config.gpu_device_id = 0;
    config.no_timestamps = true;
    config.temperature = 0.0f;

    std::cout << "配置:" << std::endl;
    std::cout << "  模型: " << config.model_path << std::endl;
    std::cout << "  语言: " << config.language << std::endl;
    std::cout << "  线程数: " << config.num_threads << std::endl;
    std::cout << "  GPU: " << (config.use_gpu ? "启用" : "禁用") << std::endl;

    ASREngine engine(config);

    auto start = std::chrono::high_resolution_clock::now();
    bool result = engine.initialize();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (result) {
        std::cout << "[PASS] 初始化成功! 耗时: " << duration.count() << " ms" << std::endl;
        return true;
    } else {
        std::cout << "[FAIL] 初始化失败!" << std::endl;
        return false;
    }
}

// 测试2: 识别 WAV 文件
void testRecognizeFile(const std::string& model_path, const std::string& wav_path, bool use_gpu) {
    std::cout << "\n========== 测试2: 识别 WAV 文件 ==========" << std::endl;

    // 读取 WAV 文件
    std::vector<float> audio;
    int sample_rate;
    if (!readWavFile(wav_path, audio, sample_rate)) {
        return;
    }

    // 如果采样率不是 16kHz，进行重采样
    if (sample_rate != 16000) {
        std::cout << "重采样从 " << sample_rate << " Hz 到 16000 Hz..." << std::endl;
        audio = resample(audio, sample_rate, 16000);
    }

    // 初始化引擎
    ASREngine::Config config;
    config.model_path = model_path;
    config.language = "zh";
    config.num_threads = 4;
    config.use_gpu = use_gpu;

    ASREngine engine(config);
    if (!engine.initialize()) {
        std::cerr << "引擎初始化失败!" << std::endl;
        return;
    }

    // 执行识别
    std::cout << "\n开始识别..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    std::string result = engine.recognize(audio);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    float rtf = duration.count() / 1000.0f / (audio.size() / 16000.0f);

    std::cout << "识别结果: \"" << result << "\"" << std::endl;
    std::cout << "识别耗时: " << duration.count() << " ms" << std::endl;
    std::cout << "实时因子 (RTF): " << rtf << std::endl;

    if (rtf < 1.0f) {
        std::cout << "[INFO] 识别速度快于实时 (RTF < 1.0)" << std::endl;
    } else {
        std::cout << "[WARN] 识别速度慢于实时 (RTF >= 1.0)" << std::endl;
    }
}

// 测试3: 流式识别测试
void testStreamingRecognition(const std::string& model_path, bool use_gpu) {
    std::cout << "\n========== 测试3: 流式识别测试 ==========" << std::endl;

    ASREngine::Config config;
    config.model_path = model_path;
    config.language = "zh";
    config.num_threads = 4;
    config.use_gpu = use_gpu;

    ASREngine engine(config);
    if (!engine.initialize()) {
        std::cerr << "引擎初始化失败!" << std::endl;
        return;
    }

    // 生成 3 秒测试音频
    auto test_audio = generateTestAudio(3.0f, 16000);
    std::cout << "生成测试音频: " << test_audio.size() << " 样本 (3秒)" << std::endl;

    // 流式处理
    engine.startStream();

    size_t chunk_size = 256;  // 每次送入 256 个样本
    size_t offset = 0;

    std::cout << "开始流式输入..." << std::endl;

    while (offset < test_audio.size()) {
        size_t remaining = test_audio.size() - offset;
        size_t current_chunk = std::min(chunk_size, remaining);

        engine.feedAudio(test_audio.data() + offset, current_chunk);
        offset += current_chunk;

        // 每输入 1 秒音频，获取一次部分结果
        if (offset % 16000 == 0) {
            std::string partial = engine.getPartialResult();
            std::cout << "  [" << (offset / 16000) << "秒] 部分结果: \"" << partial << "\"" << std::endl;
        }
    }

    // 结束流式识别
    std::string final_result = engine.endStream();
    std::cout << "最终结果: \"" << final_result << "\"" << std::endl;
    std::cout << "[INFO] 流式识别测试完成" << std::endl;
}

// 测试4: 性能基准测试
void testBenchmark(const std::string& model_path, bool use_gpu) {
    std::cout << "\n========== 测试4: 性能基准测试 ==========" << std::endl;

    ASREngine::Config config;
    config.model_path = model_path;
    config.language = "zh";
    config.num_threads = 4;
    config.use_gpu = use_gpu;

    ASREngine engine(config);
    if (!engine.initialize()) {
        std::cerr << "引擎初始化失败!" << std::endl;
        return;
    }

    // 测试不同时长的音频
    float durations[] = {1.0f, 2.0f, 5.0f, 10.0f};

    std::cout << "\n音频时长 vs 识别时间:" << std::endl;
    std::cout << "时长(秒)\t识别时间(ms)\tRTF" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    for (float duration : durations) {
        auto audio = generateTestAudio(duration, 16000);

        auto start = std::chrono::high_resolution_clock::now();
        engine.recognize(audio);
        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        float rtf = elapsed.count() / 1000.0f / duration;

        std::cout << duration << "\t\t" << elapsed.count() << "\t\t" << rtf << std::endl;
    }
}

// 测试5: 空音频和边界测试
void testEdgeCases(const std::string& model_path, bool use_gpu) {
    std::cout << "\n========== 测试5: 边界情况测试 ==========" << std::endl;

    ASREngine::Config config;
    config.model_path = model_path;
    config.language = "zh";
    config.num_threads = 4;
    config.use_gpu = use_gpu;

    ASREngine engine(config);
    if (!engine.initialize()) {
        std::cerr << "引擎初始化失败!" << std::endl;
        return;
    }

    // 测试空音频
    std::cout << "空音频测试: ";
    std::string result = engine.recognize(nullptr, 0);
    std::cout << "结果=\"" << result << "\" [期望: 空字符串]" << std::endl;

    // 测试极短音频 (0.1秒)
    std::cout << "极短音频测试 (0.1秒): ";
    auto short_audio = generateTestAudio(0.1f);
    result = engine.recognize(short_audio);
    std::cout << "结果=\"" << result << "\"" << std::endl;

    // 测试静音
    std::cout << "静音测试 (1秒): ";
    std::vector<float> silence(16000, 0.0f);
    result = engine.recognize(silence);
    std::cout << "结果=\"" << result << "\"" << std::endl;
}

void printUsage(const char* prog) {
    std::cout << "用法: " << prog << " --model <模型路径> [选项]" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  --model <路径>   Whisper 模型路径 (必需)" << std::endl;
    std::cout << "  --wav <路径>     WAV 文件路径 (用于识别测试)" << std::endl;
    std::cout << "  --no-gpu         禁用 GPU" << std::endl;
    std::cout << "  --init           只测试初始化" << std::endl;
    std::cout << "  --stream         流式识别测试" << std::endl;
    std::cout << "  --benchmark      性能基准测试" << std::endl;
    std::cout << "  --edge           边界情况测试" << std::endl;
    std::cout << "  --all            运行所有测试" << std::endl;
    std::cout << "  --help           显示帮助" << std::endl;
    std::cout << "\n示例:" << std::endl;
    std::cout << "  " << prog << " --model whisper.cpp/models/ggml-large-v3-turbo.bin --all" << std::endl;
    std::cout << "  " << prog << " --model whisper.cpp/models/ggml-large-v3-turbo.bin --wav test.wav" << std::endl;
}

int main(int argc, char* argv[]) {
    Logger::getInstance().setLevel(LogLevel::INFO);

    std::cout << "===== ASREngine 语音识别引擎测试 =====" << std::endl;

    if (argc < 2) {
        printUsage(argv[0]);
        return 0;
    }

    std::string model_path;
    std::string wav_path;
    bool use_gpu = true;
    bool run_init = false;
    bool run_stream = false;
    bool run_benchmark = false;
    bool run_edge = false;
    bool run_all = false;

    // 解析参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--wav" && i + 1 < argc) {
            wav_path = argv[++i];
        } else if (arg == "--no-gpu") {
            use_gpu = false;
        } else if (arg == "--init") {
            run_init = true;
        } else if (arg == "--stream") {
            run_stream = true;
        } else if (arg == "--benchmark") {
            run_benchmark = true;
        } else if (arg == "--edge") {
            run_edge = true;
        } else if (arg == "--all") {
            run_all = true;
        }
    }

    if (model_path.empty()) {
        std::cerr << "错误: 必须指定模型路径 (--model)" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    std::cout << "模型路径: " << model_path << std::endl;
    std::cout << "GPU: " << (use_gpu ? "启用" : "禁用") << std::endl;

    // 运行测试
    if (run_all || run_init) {
        if (!testInitialize(model_path, use_gpu)) {
            std::cerr << "初始化失败，跳过后续测试" << std::endl;
            return 1;
        }
    }

    if (!wav_path.empty()) {
        testRecognizeFile(model_path, wav_path, use_gpu);
    }

    if (run_all || run_stream) {
        testStreamingRecognition(model_path, use_gpu);
    }

    if (run_all || run_benchmark) {
        testBenchmark(model_path, use_gpu);
    }

    if (run_all || run_edge) {
        testEdgeCases(model_path, use_gpu);
    }

    std::cout << "\n测试完成!" << std::endl;
    return 0;
}
