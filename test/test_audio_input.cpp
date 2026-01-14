/**
 * @file test_audio_input.cpp
 * @brief AudioInput 音频输入测试
 *
 * 测试内容:
 * 1. 列出可用音频设备
 * 2. 初始化音频输入
 * 3. 录音并保存为 WAV 文件
 * 4. 实时音量监测
 */

#include "audio/audio_input.h"
#include "utils/logger.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <cmath>
#include <fstream>
#include <atomic>
#include <csignal>

using namespace voice_assistant::audio;
using namespace voice_assistant::utils;

// 全局退出标志
std::atomic<bool> g_running{true};

void signalHandler(int signum) {
    (void)signum;
    std::cout << "\n收到中断信号，正在停止..." << std::endl;
    g_running = false;
}

// WAV 文件头结构
struct WavHeader {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t file_size = 0;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_size = 16;
    uint16_t audio_format = 3;  // IEEE float
    uint16_t num_channels = 1;
    uint32_t sample_rate = 16000;
    uint32_t byte_rate = 64000;  // sample_rate * num_channels * bytes_per_sample
    uint16_t block_align = 4;   // num_channels * bytes_per_sample
    uint16_t bits_per_sample = 32;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size = 0;
};

// 保存 WAV 文件
bool saveWav(const std::string& filename, const std::vector<float>& audio, int sample_rate) {
    WavHeader header;
    header.sample_rate = sample_rate;
    header.byte_rate = sample_rate * sizeof(float);
    header.data_size = audio.size() * sizeof(float);
    header.file_size = 36 + header.data_size;

    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "无法创建文件: " << filename << std::endl;
        return false;
    }

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    file.write(reinterpret_cast<const char*>(audio.data()), audio.size() * sizeof(float));

    std::cout << "保存 WAV 文件: " << filename << " (" << audio.size() << " 样本)" << std::endl;
    return true;
}

// 计算 RMS 音量
float calculateRMS(const float* data, size_t frames) {
    float sum = 0.0f;
    for (size_t i = 0; i < frames; ++i) {
        sum += data[i] * data[i];
    }
    return std::sqrt(sum / frames);
}

// 显示音量条
void displayVolumeBar(float rms) {
    const int bar_width = 50;
    float db = 20.0f * std::log10(rms + 1e-10f);
    float normalized = (db + 60.0f) / 60.0f;  // 归一化到 0-1
    normalized = std::max(0.0f, std::min(1.0f, normalized));

    int filled = static_cast<int>(normalized * bar_width);
    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < filled) {
            std::cout << "=";
        } else {
            std::cout << " ";
        }
    }
    std::cout << "] " << std::fixed;
    std::cout.precision(1);
    std::cout << db << " dB  " << std::flush;
}

// 测试1: 列出设备
void testListDevices() {
    std::cout << "\n========== 测试1: 列出音频输入设备 ==========" << std::endl;

    auto devices = AudioInput::listDevices();

    if (devices.empty()) {
        std::cout << "未找到音频输入设备!" << std::endl;
        return;
    }

    std::cout << "找到 " << devices.size() << " 个输入设备:" << std::endl;
    for (size_t i = 0; i < devices.size(); ++i) {
        std::cout << "  [" << i << "] " << devices[i] << std::endl;
    }

    std::cout << "\n默认设备: " << AudioInput::getDefaultDevice() << std::endl;
}

// 测试2: 初始化测试
void testInitialize() {
    std::cout << "\n========== 测试2: 初始化音频输入 ==========" << std::endl;

    AudioInput audio_input(16000, 1, 256);

    bool result = audio_input.initialize();
    if (result) {
        std::cout << "初始化成功!" << std::endl;
    } else {
        std::cout << "初始化失败!" << std::endl;
    }
}

// 测试3: 录音测试
void testRecording(int duration_seconds) {
    std::cout << "\n========== 测试3: 录音测试 (" << duration_seconds << " 秒) ==========" << std::endl;

    AudioInput audio_input(16000, 1, 256);

    if (!audio_input.initialize()) {
        std::cerr << "初始化失败!" << std::endl;
        return;
    }

    std::vector<float> recorded_audio;
    recorded_audio.reserve(16000 * duration_seconds);

    auto callback = [&recorded_audio](const float* data, size_t frames) {
        recorded_audio.insert(recorded_audio.end(), data, data + frames);
    };

    std::cout << "开始录音..." << std::endl;

    if (!audio_input.start(callback)) {
        std::cerr << "启动录音失败!" << std::endl;
        return;
    }

    // 等待指定时间
    auto start_time = std::chrono::steady_clock::now();
    while (g_running) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= duration_seconds) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // 显示进度
        int seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
        std::cout << "\r录音中... " << seconds << "/" << duration_seconds << " 秒, "
                  << recorded_audio.size() << " 样本" << std::flush;
    }

    audio_input.stop();
    std::cout << std::endl;

    std::cout << "录音结束! 总共 " << recorded_audio.size() << " 样本 ("
              << (recorded_audio.size() / 16000.0) << " 秒)" << std::endl;

    // 保存 WAV 文件
    if (!recorded_audio.empty()) {
        saveWav("test_recording.wav", recorded_audio, 16000);
    }
}

// 测试4: 实时音量监测
void testVolumeMonitor(int duration_seconds) {
    std::cout << "\n========== 测试4: 实时音量监测 (" << duration_seconds << " 秒) ==========" << std::endl;

    AudioInput audio_input(16000, 1, 256);

    if (!audio_input.initialize()) {
        std::cerr << "初始化失败!" << std::endl;
        return;
    }

    std::atomic<float> current_rms{0.0f};

    auto callback = [&current_rms](const float* data, size_t frames) {
        current_rms = calculateRMS(data, frames);
    };

    std::cout << "开始监测音量，请说话..." << std::endl;

    if (!audio_input.start(callback)) {
        std::cerr << "启动录音失败!" << std::endl;
        return;
    }

    auto start_time = std::chrono::steady_clock::now();
    while (g_running) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= duration_seconds) {
            break;
        }

        displayVolumeBar(current_rms);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    audio_input.stop();
    std::cout << std::endl;
    std::cout << "音量监测结束!" << std::endl;
}

void printUsage(const char* prog) {
    std::cout << "用法: " << prog << " [选项]" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  --list           列出音频设备" << std::endl;
    std::cout << "  --init           测试初始化" << std::endl;
    std::cout << "  --record [秒数]  录音测试 (默认5秒)" << std::endl;
    std::cout << "  --monitor [秒数] 音量监测 (默认10秒)" << std::endl;
    std::cout << "  --all            运行所有测试" << std::endl;
    std::cout << "  --help           显示帮助" << std::endl;
}

int main(int argc, char* argv[]) {
    // 设置信号处理
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // 设置日志级别
    Logger::getInstance().setLevel(LogLevel::INFO);

    std::cout << "===== AudioInput 音频输入测试 =====" << std::endl;

    if (argc < 2) {
        printUsage(argv[0]);
        return 0;
    }

    std::string cmd = argv[1];

    if (cmd == "--help" || cmd == "-h") {
        printUsage(argv[0]);
        return 0;
    }

    if (cmd == "--list") {
        testListDevices();
    } else if (cmd == "--init") {
        testInitialize();
    } else if (cmd == "--record") {
        int duration = 5;
        if (argc > 2) {
            duration = std::atoi(argv[2]);
        }
        testRecording(duration);
    } else if (cmd == "--monitor") {
        int duration = 10;
        if (argc > 2) {
            duration = std::atoi(argv[2]);
        }
        testVolumeMonitor(duration);
    } else if (cmd == "--all") {
        testListDevices();
        testInitialize();
        testRecording(5);
        testVolumeMonitor(10);
    } else {
        std::cerr << "未知选项: " << cmd << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    std::cout << "\n测试完成!" << std::endl;
    return 0;
}
