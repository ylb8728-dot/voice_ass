/**
 * @file test_vad.cpp
 * @brief VADDetector 语音活动检测测试
 *
 * 测试内容:
 * 1. 静音检测
 * 2. 语音信号检测
 * 3. 合成信号测试
 * 4. 状态机转换测试
 */

#include "audio/vad_detector.h"
#include "utils/logger.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace voice_assistant::audio;
using namespace voice_assistant::utils;

// 生成静音信号
std::vector<float> generateSilence(size_t num_samples, float noise_level = 0.001f) {
    std::vector<float> audio(num_samples);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, noise_level);

    for (size_t i = 0; i < num_samples; ++i) {
        audio[i] = dist(gen);
    }
    return audio;
}

// 生成正弦波信号 (模拟语音)
std::vector<float> generateSineWave(size_t num_samples, float frequency, float amplitude, int sample_rate = 16000) {
    std::vector<float> audio(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        audio[i] = amplitude * std::sin(2.0f * M_PI * frequency * i / sample_rate);
    }
    return audio;
}

// 生成复合语音信号 (多频率叠加)
std::vector<float> generateVoice(size_t num_samples, float amplitude, int sample_rate = 16000) {
    std::vector<float> audio(num_samples, 0.0f);

    // 语音基频 (F0) 约 100-300 Hz
    float f0 = 150.0f;

    // 添加基频和谐波
    for (int harmonic = 1; harmonic <= 5; ++harmonic) {
        float freq = f0 * harmonic;
        float harm_amp = amplitude / harmonic;  // 谐波幅度递减
        for (size_t i = 0; i < num_samples; ++i) {
            audio[i] += harm_amp * std::sin(2.0f * M_PI * freq * i / sample_rate);
        }
    }

    // 添加一些随机噪声使信号更真实
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, amplitude * 0.1f);
    for (size_t i = 0; i < num_samples; ++i) {
        audio[i] += dist(gen);
    }

    return audio;
}

// 计算信号能量
float calculateEnergy(const std::vector<float>& audio) {
    float sum = 0.0f;
    for (float sample : audio) {
        sum += sample * sample;
    }
    return sum / audio.size();
}

// 测试1: 静音检测
void testSilenceDetection() {
    std::cout << "\n========== 测试1: 静音检测 ==========" << std::endl;

    VADDetector::Config config;
    config.energy_threshold = 0.01f;
    config.min_speech_frames = 3;
    config.min_silence_frames = 5;
    config.frame_size = 256;

    VADDetector vad(config);

    // 生成静音
    auto silence = generateSilence(256, 0.001f);
    float energy = calculateEnergy(silence);
    std::cout << "静音信号能量: " << energy << std::endl;

    // 连续检测多帧
    int speech_frames = 0;
    int silence_frames = 0;
    for (int i = 0; i < 20; ++i) {
        bool is_speech = vad.isSpeech(silence);
        if (is_speech) {
            speech_frames++;
        } else {
            silence_frames++;
        }
    }

    std::cout << "检测结果: 语音帧=" << speech_frames << ", 静音帧=" << silence_frames << std::endl;

    if (silence_frames > speech_frames) {
        std::cout << "[PASS] 正确识别为静音" << std::endl;
    } else {
        std::cout << "[FAIL] 误判为语音" << std::endl;
    }
}

// 测试2: 语音信号检测
void testSpeechDetection() {
    std::cout << "\n========== 测试2: 语音信号检测 ==========" << std::endl;

    VADDetector::Config config;
    config.energy_threshold = 0.01f;
    config.min_speech_frames = 3;
    config.min_silence_frames = 5;
    config.frame_size = 256;

    VADDetector vad(config);

    // 生成语音信号
    auto voice = generateVoice(256, 0.3f);
    float energy = calculateEnergy(voice);
    std::cout << "语音信号能量: " << energy << std::endl;

    // 连续检测多帧
    int speech_frames = 0;
    int silence_frames = 0;
    for (int i = 0; i < 20; ++i) {
        bool is_speech = vad.isSpeech(voice);
        if (is_speech) {
            speech_frames++;
        } else {
            silence_frames++;
        }
    }

    std::cout << "检测结果: 语音帧=" << speech_frames << ", 静音帧=" << silence_frames << std::endl;

    if (speech_frames > silence_frames) {
        std::cout << "[PASS] 正确识别为语音" << std::endl;
    } else {
        std::cout << "[FAIL] 误判为静音" << std::endl;
    }
}

// 测试3: 状态转换测试 (静音 -> 语音 -> 静音)
void testStateTransition() {
    std::cout << "\n========== 测试3: 状态转换测试 ==========" << std::endl;

    VADDetector::Config config;
    config.energy_threshold = 0.01f;
    config.min_speech_frames = 3;
    config.min_silence_frames = 5;
    config.frame_size = 256;

    VADDetector vad(config);

    std::cout << "序列: 10帧静音 -> 10帧语音 -> 10帧静音" << std::endl;

    auto silence = generateSilence(256, 0.001f);
    auto voice = generateVoice(256, 0.3f);

    std::cout << "帧序列状态: ";

    // 前10帧静音
    for (int i = 0; i < 10; ++i) {
        bool in_speech = vad.isSpeech(silence);
        std::cout << (in_speech ? "S" : "_");
    }

    // 中间10帧语音
    for (int i = 0; i < 10; ++i) {
        bool in_speech = vad.isSpeech(voice);
        std::cout << (in_speech ? "S" : "_");
    }

    // 后10帧静音
    for (int i = 0; i < 10; ++i) {
        bool in_speech = vad.isSpeech(silence);
        std::cout << (in_speech ? "S" : "_");
    }

    std::cout << std::endl;
    std::cout << "说明: S=语音段, _=非语音段" << std::endl;
    std::cout << "[INFO] 观察状态是否正确转换 (需要min_speech_frames和min_silence_frames达到阈值)" << std::endl;
}

// 测试4: 不同阈值测试
void testThresholds() {
    std::cout << "\n========== 测试4: 不同阈值测试 ==========" << std::endl;

    // 生成中等能量信号
    auto medium_signal = generateVoice(256, 0.1f);
    float energy = calculateEnergy(medium_signal);
    std::cout << "测试信号能量: " << energy << std::endl;

    float thresholds[] = {0.001f, 0.005f, 0.01f, 0.05f, 0.1f};

    std::cout << "\n阈值对比:" << std::endl;
    for (float threshold : thresholds) {
        VADDetector::Config config;
        config.energy_threshold = threshold;
        config.min_speech_frames = 1;  // 立即响应
        config.min_silence_frames = 1;
        config.frame_size = 256;

        VADDetector vad(config);

        // 预热几帧
        for (int i = 0; i < 5; ++i) {
            vad.isSpeech(medium_signal);
        }

        bool is_speech = vad.isSpeech(medium_signal);
        std::cout << "  阈值=" << threshold << " -> " << (is_speech ? "语音" : "静音") << std::endl;
    }
}

// 测试5: Reset 功能测试
void testReset() {
    std::cout << "\n========== 测试5: Reset 功能测试 ==========" << std::endl;

    VADDetector::Config config;
    config.energy_threshold = 0.01f;
    config.min_speech_frames = 3;
    config.min_silence_frames = 5;
    config.frame_size = 256;

    VADDetector vad(config);

    auto voice = generateVoice(256, 0.3f);

    // 进入语音状态
    for (int i = 0; i < 10; ++i) {
        vad.isSpeech(voice);
    }

    std::cout << "语音段后状态: " << (vad.isInSpeech() ? "在语音段" : "不在语音段") << std::endl;

    // 重置
    vad.reset();
    std::cout << "Reset后状态: " << (vad.isInSpeech() ? "在语音段" : "不在语音段") << std::endl;

    if (!vad.isInSpeech()) {
        std::cout << "[PASS] Reset功能正常" << std::endl;
    } else {
        std::cout << "[FAIL] Reset功能异常" << std::endl;
    }
}

// 测试6: 边界情况测试
void testEdgeCases() {
    std::cout << "\n========== 测试6: 边界情况测试 ==========" << std::endl;

    VADDetector vad;

    // 空指针测试
    std::cout << "空指针测试: ";
    bool result = vad.isSpeech(nullptr, 0);
    std::cout << (result ? "语音" : "非语音") << " [期望: 非语音]" << std::endl;

    // 空数组测试
    std::cout << "空数组测试: ";
    std::vector<float> empty;
    result = vad.isSpeech(empty);
    std::cout << (result ? "语音" : "非语音") << " [期望: 非语音]" << std::endl;

    // 单样本测试
    std::cout << "单样本测试: ";
    float single = 0.5f;
    result = vad.isSpeech(&single, 1);
    std::cout << (result ? "语音" : "非语音") << std::endl;
}

void printUsage(const char* prog) {
    std::cout << "用法: " << prog << " [选项]" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  --silence    静音检测测试" << std::endl;
    std::cout << "  --speech     语音检测测试" << std::endl;
    std::cout << "  --state      状态转换测试" << std::endl;
    std::cout << "  --threshold  阈值测试" << std::endl;
    std::cout << "  --reset      Reset测试" << std::endl;
    std::cout << "  --edge       边界情况测试" << std::endl;
    std::cout << "  --all        运行所有测试" << std::endl;
    std::cout << "  --help       显示帮助" << std::endl;
}

int main(int argc, char* argv[]) {
    Logger::getInstance().setLevel(LogLevel::DEBUG);

    std::cout << "===== VADDetector 语音活动检测测试 =====" << std::endl;

    if (argc < 2) {
        printUsage(argv[0]);
        return 0;
    }

    std::string cmd = argv[1];

    if (cmd == "--help" || cmd == "-h") {
        printUsage(argv[0]);
        return 0;
    }

    if (cmd == "--silence") {
        testSilenceDetection();
    } else if (cmd == "--speech") {
        testSpeechDetection();
    } else if (cmd == "--state") {
        testStateTransition();
    } else if (cmd == "--threshold") {
        testThresholds();
    } else if (cmd == "--reset") {
        testReset();
    } else if (cmd == "--edge") {
        testEdgeCases();
    } else if (cmd == "--all") {
        testSilenceDetection();
        testSpeechDetection();
        testStateTransition();
        testThresholds();
        testReset();
        testEdgeCases();
    } else {
        std::cerr << "未知选项: " << cmd << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    std::cout << "\n测试完成!" << std::endl;
    return 0;
}
