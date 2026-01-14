#include "tts/cosyvoice2_engine.h"
#include "utils/logger.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

using namespace voice_assistant;

// WAV文件头结构
struct WAVHeader {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t chunk_size;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_size = 16;
    uint16_t audio_format = 1; // PCM
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size;
};

/**
 * @brief 读取WAV文件（16-bit PCM）
 */
bool readWAV(const std::string& filename, std::vector<float>& audio, int& sample_rate) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open file: ", filename);
        return false;
    }

    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));

    if (std::strncmp(header.riff, "RIFF", 4) != 0 ||
        std::strncmp(header.wave, "WAVE", 4) != 0) {
        LOG_ERROR("Invalid WAV file: ", filename);
        return false;
    }

    sample_rate = header.sample_rate;
    size_t num_samples = header.data_size / (header.bits_per_sample / 8) / header.num_channels;

    if (header.bits_per_sample == 16) {
        std::vector<int16_t> samples(num_samples * header.num_channels);
        file.read(reinterpret_cast<char*>(samples.data()), header.data_size);

        // 转换为float并只取第一个声道
        audio.resize(num_samples);
        for (size_t i = 0; i < num_samples; ++i) {
            audio[i] = samples[i * header.num_channels] / 32768.0f;
        }
    } else {
        LOG_ERROR("Unsupported bit depth: ", header.bits_per_sample);
        return false;
    }

    file.close();
    LOG_INFO("Loaded ", audio.size(), " samples from ", filename, " (", sample_rate, " Hz)");
    return true;
}

/**
 * @brief 写入WAV文件（16-bit PCM）
 */
bool writeWAV(const std::string& filename, const std::vector<float>& audio, int sample_rate) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Failed to create file: ", filename);
        return false;
    }

    WAVHeader header;
    header.num_channels = 1;
    header.sample_rate = sample_rate;
    header.bits_per_sample = 16;
    header.byte_rate = sample_rate * header.num_channels * header.bits_per_sample / 8;
    header.block_align = header.num_channels * header.bits_per_sample / 8;
    header.data_size = audio.size() * header.bits_per_sample / 8;
    header.chunk_size = 36 + header.data_size;

    file.write(reinterpret_cast<const char*>(&header), sizeof(WAVHeader));

    // 转换float到int16并写入
    std::vector<int16_t> samples(audio.size());
    for (size_t i = 0; i < audio.size(); ++i) {
        float sample = std::max(-1.0f, std::min(1.0f, audio[i]));
        samples[i] = static_cast<int16_t>(sample * 32767.0f);
    }

    file.write(reinterpret_cast<const char*>(samples.data()), header.data_size);
    file.close();

    LOG_INFO("Saved ", audio.size(), " samples to ", filename, " (", sample_rate, " Hz)");
    return true;
}

/**
 * @brief 打印使用说明
 */
void printUsage(const char* program_name) {
    std::cout << "CosyVoice2 TTS Engine Test Program\n";
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --model-dir DIR          Model directory (required)\n";
    std::cout << "  --prompt-wav FILE        Prompt audio file (16kHz WAV, required)\n";
    std::cout << "  --prompt-text TEXT       Prompt text (required)\n";
    std::cout << "  --tts-text TEXT          Text to synthesize (required)\n";
    std::cout << "  --output FILE            Output WAV file (default: output.wav)\n";
    std::cout << "  --stream                 Enable streaming mode\n";
    std::cout << "  --gpu                    Use GPU (default: true)\n";
    std::cout << "  --cpu                    Use CPU only\n";
    std::cout << "  --help                   Show this help\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << program_name << " --model-dir models/cosyvoice2 \\\n";
    std::cout << "    --prompt-wav zero_shot_prompt.wav \\\n";
    std::cout << "    --prompt-text \"你好，这是一段测试语音\" \\\n";
    std::cout << "    --tts-text \"欢迎使用语音助手\" \\\n";
    std::cout << "    --output result.wav --stream\n";
}

int main(int argc, char* argv[]) {
    // 初始化日志
    utils::Logger::getInstance().setLevel(utils::LogLevel::INFO);

    // 解析命令行参数
    std::string model_dir;
    std::string prompt_wav;
    std::string prompt_text;
    std::string tts_text;
    std::string output_file = "output.wav";
    bool use_stream = false;
    bool use_gpu = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--model-dir" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (arg == "--prompt-wav" && i + 1 < argc) {
            prompt_wav = argv[++i];
        } else if (arg == "--prompt-text" && i + 1 < argc) {
            prompt_text = argv[++i];
        } else if (arg == "--tts-text" && i + 1 < argc) {
            tts_text = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--stream") {
            use_stream = true;
        } else if (arg == "--gpu") {
            use_gpu = true;
        } else if (arg == "--cpu") {
            use_gpu = false;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    // 检查必需参数
    if (model_dir.empty() || prompt_wav.empty() || prompt_text.empty() || tts_text.empty()) {
        std::cerr << "Error: Missing required arguments\n\n";
        printUsage(argv[0]);
        return 1;
    }

    LOG_INFO("=== CosyVoice2 TTS Test ===");
    LOG_INFO("Model directory: ", model_dir);
    LOG_INFO("Prompt WAV: ", prompt_wav);
    LOG_INFO("Prompt text: ", prompt_text);
    LOG_INFO("TTS text: ", tts_text);
    LOG_INFO("Output file: ", output_file);
    LOG_INFO("Streaming: ", use_stream ? "enabled" : "disabled");
    LOG_INFO("Device: ", use_gpu ? "GPU" : "CPU");

    // 读取prompt音频
    std::vector<float> prompt_audio;
    int prompt_sample_rate;
    if (!readWAV(prompt_wav, prompt_audio, prompt_sample_rate)) {
        LOG_ERROR("Failed to load prompt audio");
        return 1;
    }

    if (prompt_sample_rate != 16000) {
        LOG_ERROR("Prompt audio must be 16kHz, got ", prompt_sample_rate, " Hz");
        return 1;
    }

    // 检查音频时长（最长30秒）
    float duration = static_cast<float>(prompt_audio.size()) / 16000.0f;
    if (duration > 30.0f) {
        LOG_ERROR("Prompt audio too long (", duration, "s), max 30s supported");
        return 1;
    }
    LOG_INFO("Prompt audio duration: ", duration, " seconds");

    // 初始化CosyVoice2引擎
    tts::CosyVoice2Engine::Config config;
    config.model_dir = model_dir;
    config.use_gpu = use_gpu;
    config.enable_streaming = use_stream;
    config.num_threads = 4;

    tts::CosyVoice2Engine engine(config);
    if (!engine.initialize()) {
        LOG_ERROR("Failed to initialize CosyVoice2 engine");
        return 1;
    }

    // 执行TTS合成
    std::vector<float> output_audio;

    if (use_stream) {
        LOG_INFO("Starting streaming synthesis...");

        // Streaming模式：收集所有音频块
        std::vector<std::vector<float>> audio_chunks;

        auto callback = [&audio_chunks](const float* audio, size_t num_samples) {
            LOG_INFO("Received audio chunk: ", num_samples, " samples");
            std::vector<float> chunk(audio, audio + num_samples);
            audio_chunks.push_back(chunk);
        };

        engine.synthesizeZeroShotStream(
            tts_text, prompt_text,
            prompt_audio.data(), prompt_audio.size(),
            callback);

        // 合并所有音频块
        for (const auto& chunk : audio_chunks) {
            output_audio.insert(output_audio.end(), chunk.begin(), chunk.end());
        }

    } else {
        LOG_INFO("Starting offline synthesis...");

        output_audio = engine.synthesizeZeroShot(
            tts_text, prompt_text,
            prompt_audio.data(), prompt_audio.size());
    }

    if (output_audio.empty()) {
        LOG_ERROR("Synthesis failed - no audio generated");
        return 1;
    }

    // 保存输出
    if (!writeWAV(output_file, output_audio, 24000)) {
        LOG_ERROR("Failed to save output audio");
        return 1;
    }

    float output_duration = static_cast<float>(output_audio.size()) / 24000.0f;
    LOG_INFO("=== Synthesis Complete ===");
    LOG_INFO("Output duration: ", output_duration, " seconds");
    LOG_INFO("Output saved to: ", output_file);

    return 0;
}
